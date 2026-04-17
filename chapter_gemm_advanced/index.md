# GEMM Steps 7--9: Warp Specialization to cuBLAS
:label:`chap_gemm_advanced`

This chapter covers the three advanced GEMM optimizations that bridge the gap from a basic tiled kernel to cuBLAS-level performance. We start with warp specialization and software pipelining (Step 7), introduce CTA clusters for cooperative MMA (Step 8), and finish with multi-consumer warp specialization (Step 9).

> **Layout recap.** The per-step SMEM / TMEM / register layouts repeat the pattern established in :numref:`chap_gemm_basics` and :numref:`chap_gemm_async`. Step 8 additionally uses cluster-scope buffers; for the `cbx` / `cby` axes and how a single layout can span two CTAs, refer to :numref:`chap_layouts`, section *Cluster and Remote Layouts*.


## Step 7: Warp Specialization + Pipeline
:label:`chap_warp_specialization`

This is the biggest architectural change: instead of all threads doing load-then-compute sequentially, we dedicate specific warps to specific tasks and overlap them with software pipelining. We benchmark with M=N=K=4096.

### What You Will Learn

- Warp specialization: dedicating different warps/warpgroups to different tasks

- High-level barrier abstractions: `TMABar`, `TCGen05Bar`, `MBarrier`

- `PipelineState` for automatic stage/phase management

- Software pipelining with multi-buffered SMEM

- Persistent kernel with `ClusterPersistentScheduler2D`

### From Sequential to Concurrent

![Warp Specialization Timeline](../img/warp_specialization_timeline.png)

The TMA producer prefetches data while the MMA consumer computes, and writeback runs independently. Two barriers synchronize the handoff:

- **`tma2mma`** (TMA → MMA): signals that the loaded SMEM data is ready for MMA to consume.
- **`mma2tma`** (MMA → TMA): signals that MMA has finished reading a buffer, so TMA can reuse it for the next load.

With `PIPE_DEPTH=2`, there are two SMEM buffers (stage 0 and stage 1). TMA Load k=0 fills buffer 0, TMA Load k=1 fills buffer 1. When MMA Compute k=0 finishes reading buffer 0, it signals `mma2tma` — but the next load that needs buffer 0 is TMA Load k=2 (not k=1, which uses buffer 1). So the `mma2tma` arrow from MMA Compute k=0 points to TMA Load k=2.

### Warp Roles

With `WG_NUMBER=2`, the kernel uses two warpgroups (abbreviated WG below):

| Actor | Location | Job |
|-------|----------|-----|
| **TMA Producer** | Warpgroup 1, warp 3 | Continuously loads A and B tiles via TMA |
| **MMA Consumer** | Warpgroup 1, warp 0 | Runs MMA as soon as data is ready |
| **Writeback** | Warpgroup 0 (all warps) | Reads TMEM results, writes to GMEM |

### 4 Barrier Types

The three actors are synchronized by four barrier types. The forward path (TMA → MMA → Writeback) signals data readiness; the backward path (Writeback → MMA → TMA) signals buffer release. We name each barrier as `source2destination` — for example, `tma2mma` is the barrier where TMA signals MMA:

| Barrier | Type | Direction | Meaning |
|---------|------|-----------|---------|
| **tma2mma** | `TMABar` | TMA -> MMA | "SMEM data is ready" |
| **mma2tma** | `TCGen05Bar` | MMA -> TMA | "SMEM buffer can be reused" |
| **mma2ld** | `TCGen05Bar` | MMA -> Writeback | "TMEM results are ready" |
| **ld2mma** | `MBarrier` | Writeback -> MMA | "TMEM is free for next tile" |

Note the different barrier types: **TMA Loads** use `TMABar` (mbarrier with byte counting) — the TMA hardware automatically arrives on the barrier when the transfer completes, notifying consumers. **TMA Stores** use a completely different mechanism: `cp_async.bulk.commit_group()` + `wait_group(0)`. There is no mbarrier involved because stores don't need to notify other actors — the issuing thread just waits for completion. **MMA operations** use `TCGen05Bar` — the `tcgen05.commit()` instruction signals the barrier when MMA finishes.

### PipelineState

`PipelineState` manages stage indices and phase counters:

```python
tma_ps = PipelineState("tma", PIPE_DEPTH)
tma_ps.init(is_producer=True)   # Producer starts ready (phase=1)
# tma_ps.stage = current stage index
# tma_ps.phase = current phase (0 or 1)
tma_ps.move_to_next_stage()     # Advance to next stage
```

**`is_producer` controls the initial phase:**
- `is_producer=True` -> phase starts at 1 -> first `wait(phase=1)` sees barrier phase 0 != 1, so it **passes immediately** (buffers start empty, producer can write)

- `is_producer=False` -> phase starts at 0 -> first `wait(phase=0)` sees barrier phase 0 == 0, so it **blocks** (consumer waits for data)

Getting this wrong causes deadlock or data corruption.

### `warpgroup_sync` Barrier IDs

The GPU hardware has 16 named barriers (ID 0-15). `cta_sync()` uses barrier #0 by default. In warp-specialized kernels, different warpgroups run different code paths, so we cannot use `cta_sync()` (which requires ALL threads). Instead, we use `warpgroup_sync(10)` — this syncs only threads within the warpgroup, using barrier #10 to avoid conflicting with barrier #0. In multi-consumer kernels (Step 9), each warpgroup uses a different ID: `warpgroup_sync(wg_id + 10)`.

### Implementation

The complete Step 7 kernel. `PIPE_DEPTH=2` for clarity (production kernels use `PIPE_DEPTH=4` or higher — see "Pipeline depth tuning" below for the SMEM trade-offs involved).

```{.python .input}
import tvm
from tvm.script import tirx as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.operator.scope_op_dispatch.cuda.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.pipeline import TMABar, TCGen05Bar, MBarrier, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D

SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
F16_SIZE = 2

def hgemm_v7(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2
    WG_NUMBER = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            # --- Allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")
            mma2ld  = TCGen05Bar(pool, 1, "mma2ld")
            ld2mma  = MBarrier(pool, 1, "ld2mma")
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

            # --- Barrier init ---
            tma2mma.init(1)
            mma2tma.init(1)
            mma2ld.init(1)
            ld2mma.init(128)   # all 128 Warpgroup 0 threads arrive
            pool.commit()

            # --- TMEM alloc + fence ---
            if wg_id == 0:
                if warp_id == 0:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)
            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer(
                (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
                layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            # --- Tile scheduler ---
            tile_scheduler = ClusterPersistentScheduler2D(
                "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
                l2_group_size=8, num_clusters=SM_COUNT)
            tile_scheduler.init(bx)
            m_st = Tx.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = Tx.meta_var(tile_scheduler.n_idx * BLK_N)

            # =============================================
            # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
            # =============================================
            if wg_id == 1:
                if warp_id == 3:
                    # === TMA Producer ===
                    tma_ps = PipelineState("tma", PIPE_DEPTH)
                    tma_ps.init(is_producer=True)

                    @Tx.inline
                    def tma_load(k_offset):
                        Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                      A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                      dispatch="tma", cta_group=1,
                                      mbar=tma2mma.ptr_to([tma_ps.stage]))
                        Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                      B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                      dispatch="tma", cta_group=1,
                                      mbar=tma2mma.ptr_to([tma_ps.stage]))

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            for k in range(K_TILES):
                                mma2tma.wait(tma_ps.stage, tma_ps.phase)
                                tma_load(k * BLK_K)
                                tma2mma.arrive(tma_ps.stage,
                                               (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                                tma_ps.move_to_next_stage()
                            tile_scheduler.next_tile()

                elif warp_id == 0:
                    # === MMA Consumer ===
                    mma_ps = PipelineState("mma", PIPE_DEPTH)
                    mma_ps.init(is_producer=False)
                    ld_ps = PipelineState("ld", 1)
                    ld_ps.init(is_producer=True)

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            # Wait for TMEM to be free from previous tile's writeback
                            ld2mma.wait(ld_ps.stage, ld_ps.phase)
                            ld_ps.move_to_next_stage()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.ptx.tcgen05.fence.after_thread_sync()
                                Tx.gemm_async(
                                    tmem[:, :BLK_N],
                                    Asmem[mma_ps.stage, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=1)
                                mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
                                mma_ps.move_to_next_stage()

                            # Signal results ready for writeback
                            mma2ld.arrive(0, cta_group=1, cta_mask=0)
                            tile_scheduler.next_tile()

            # =============================================
            # Warpgroup 0: Writeback
            # =============================================
            elif wg_id == 0:
                wb_ps = PipelineState("wb", 1)
                wb_ps.init(is_producer=False)
                reg_f16 = Tx.alloc_local((BLK_N,), d_type)

                while tile_scheduler.valid():
                    # Wait for MMA results
                    mma2ld.wait(wb_ps.stage, wb_ps.phase)
                    wb_ps.move_to_next_stage()

                    # Read TMEM -> registers (warpgroup scope)
                    reg = Tx.alloc_local((BLK_N,), acc_type)
                    reg_wg = reg.view(128, BLK_N,
                        layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
                    with Tx.warpgroup():
                        Tx.copy(reg_wg[:], tmem[:, :BLK_N])

                    # Signal TMEM free (all 128 threads arrive)
                    ld2mma.arrive(0, cta_id=0, pred=True)

                    # Cast fp32 -> fp16
                    with Tx.thread():
                        Tx.cast(reg_f16[:], reg[:])

                    # Write to Dsmem + TMA store
                    with Tx.thread():
                        Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                        Tx.ptx.fence.proxy_async("shared::cta")
                        Tx.cuda.warpgroup_sync(10)
                    with Tx.thread(parent="warpgroup")[Tx.ptx.elect_sync()]:
                        Tx.copy_async(D[m_st:m_st+BLK_M, n_st:n_st+BLK_N],
                                      Dsmem[:, :], dispatch="tma")
                        Tx.ptx.cp_async.bulk.commit_group()
                        Tx.ptx.cp_async.bulk.wait_group(0)
                    Tx.cuda.warpgroup_sync(10)

                    tile_scheduler.next_tile()

            # --- Cleanup ---
            Tx.cuda.cta_sync()
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

Compile and run:

```{.python .input}
import torch

M, N, K = 4096, 4096, 4096
kernel = hgemm_v7(M, N, K)
target = tvm.target.Target("cuda -arch=sm_100a")
with target:
    mod = tvm.IRModule({"main": kernel})
    lib = tvm.compile(mod, target=target, tir_pipeline="tirx")

device = torch.device('cuda')  # gpu(0)
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

lib["main"](tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor),
            tvm.runtime.from_dlpack(D_tensor))
D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Step 7 Warp Specialization: M={M}, N={N}, K={K}")
print(f"Max error vs torch reference: {max_err:.6f}")
assert max_err < 0.5, f"FAIL: max_err={max_err}"  # smoke test; fp16 4096x4096 typically gives ~0.25
print("PASS")

args = [tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor), tvm.runtime.from_dlpack(D_tensor)]
for _ in range(10):
    lib["main"](*args)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    lib["main"](*args)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

### Epilogue (Writeback) Details

The writeback warpgroup processes the output in `EPI_N`-column chunks rather than writing the full `BLK_N=128` columns at once. This is because reading all 128 columns of TMEM into registers at once would require too many registers per thread (128 x fp32 = 512 bytes), causing register spills to local memory and killing performance. By processing in smaller chunks (e.g., `EPI_N=64`), each iteration only needs a fraction of the registers, and the Dsmem buffer is also smaller:

1. Wait for MMA: `mma2ld.wait(phase)` + `fence.after_thread_sync()` — ensure TMEM writes are visible
2. Loop over `BLK_N // EPI_N` chunks:
   - Read `EPI_N` columns of TMEM -> registers
   - Cast fp32 -> fp16
   - Write to Dsmem (`EPI_N` columns, not the full `BLK_N`)
   - TMA store: `Tx.copy_async(D[..., col_st:col_st+EPI_N], Dsmem, dispatch="tma")`
3. Signal MMA: `ld2mma.arrive()` (all 128 threads arrive) — TMEM is now free for the next tile

### Key Points

- **Persistent kernel**: `bx = Tx.cta_id([SM_COUNT], parent="kernel")` --- one CTA per SM, loops over tiles

- **L2-friendly scheduling**: `ClusterPersistentScheduler2D` orders tiles for cache locality

- This pattern --- warp specialization + software pipeline --- is used in **ALL production GPU kernels** (CUTLASS, cuBLAS, etc.)

### Debugging Guide

This is the most bug-prone step. Warp specialization introduces concurrency between TMA, MMA, and writeback — if any barrier is misconfigured, the kernel deadlocks, crashes, or silently produces wrong results.

#### Inspecting Generated Code

The first debugging tool is reading the generated CUDA source. After compiling any kernel:

```python
cuda_source = lib.mod.imports[0].inspect_source()
print(cuda_source)
```

The generated code maps TIRX constructs to CUDA like this:

| TIRX | Generated CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` internal guard | `((int)threadIdx.x) < 1` (CTA thread 0 only) |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

#### Reference: Correct Step 7 Generated CUDA

A correctly compiled Step 7 kernel produces CUDA code with this structure. Comparing your generated code against this reference is the fastest way to find bugs:

```c
// ---- Barrier inits: threadIdx.x < 1 guard (CTA thread 0 only) ----
if (((int)threadIdx.x) < 1) {
  tvm_builtin_ptx_mbarrier_init(&s_buf[4], 1);   // tma2mma slot 0
  tvm_builtin_ptx_mbarrier_init(&s_buf[5], 1);   // tma2mma slot 1
}
if (((int)threadIdx.x) < 1) {
  tvm_builtin_ptx_mbarrier_init(&s_buf[6], 1);   // mma2tma slot 0
  tvm_builtin_ptx_mbarrier_init(&s_buf[7], 1);   // mma2tma slot 1
}
if (((int)threadIdx.x) < 1) {
  tvm_builtin_ptx_mbarrier_init(&s_buf[8], 1);   // mma2ld
}
if (((int)threadIdx.x) < 1) {
  tvm_builtin_ptx_mbarrier_init(&s_buf[9], 128); // ld2mma
}

// ---- TMEM alloc: WG0 warp0 (all 32 lanes, no lane guard) ----
if ((warp_id_in_cta >> 2) == 0) {       // wg_id == 0
  if ((warp_id_in_cta % 4) == 0) {      // warp_id == 0
    tvm_builtin_ptx_tcgen05_alloc_cta_group_1(&s_buf[0], 512);
  }
}

// ---- Fences + sync ----
tvm_builtin_ptx_fence_proxy_async_shared_cta();
tvm_builtin_ptx_fence_mbarrier_init();
tvm_builtin_cuda_cta_sync();

// ---- Pipeline phase init ----
tma_phase_ptr[0] = 1;     // producer: starts at phase 1 (first wait passes)
mma_phase_ptr[0] = 0;     // consumer: starts at phase 0 (first wait blocks)
ld_phase_ptr[0] = 1;      // producer
wb_phase_ptr[0] = 0;      // consumer

// ---- WG1: TMA warp (warp 3) ----
if ((warp_id_in_cta >> 2) == 1) {       // wg_id == 1
  if ((warp_id_in_cta & 3) == 3) {      // warp_id == 3
    if (tvm_builtin_elect_one_sync_op()) {
      while (valid) {
        for (int k = 0; k < 16; ++k) {  // K_TILES iterations
          mbarrier_wait(mma2tma[stage], phase);   // wait for SMEM free
          cp_async_bulk_tensor(Asmem[stage], ...); // TMA load A
          cp_async_bulk_tensor(Bsmem[stage], ...); // TMA load B
          mbarrier_arrive_expect_tx(tma2mma[stage], 32768); // signal MMA
          stage = (stage + 1) % 2;
          if (stage == 0) phase ^= 1;
        }
        next_tile();
      }
    }
  } else {
    // ---- WG1: MMA warp (warp 0) ----
    if ((warp_id_in_cta % 4) == 0) {    // warp_id == 0
      if (tvm_builtin_elect_one_sync_op()) {
        while (valid) {
          mbarrier_wait(ld2mma, ld_phase);         // wait for TMEM free
          for (int k = 0; k < 16; ++k) {           // K_TILES iterations
            mbarrier_wait(tma2mma[stage], phase);   // wait for data
            tcgen05_fence_after_thread_sync();
            tcgen05_mma(...);                       // MMA
            tcgen05_commit(mma2tma[stage]);          // signal TMA
            stage = (stage + 1) % 2;
            if (stage == 0) phase ^= 1;
          }
          tcgen05_commit(mma2ld);                   // signal writeback
          next_tile();
        }
      }
    }
  }
} else {
  // ---- WG0: Writeback ----
  if ((warp_id_in_cta >> 2) == 0) {     // wg_id == 0
    while (valid) {
      mbarrier_wait(mma2ld, wb_phase);             // wait for MMA done
      tcgen05_ld(...);                              // read TMEM -> registers
      mbarrier_arrive_remote(ld2mma, 0, true);     // signal MMA: TMEM free
      // ... cast, write to Dsmem, TMA store ...
      next_tile();
    }
  }
}

// ---- Cleanup ----
cta_sync();
if ((warp_id_in_cta % 4) == 0) {       // warp_id == 0 (all 32 lanes)
  tcgen05_relinquish_alloc_permit();
  tcgen05_dealloc(s_buf[0], 512);
}
```

Key things to verify in your generated code:
- Barrier inits are at **top level** (not inside a `wg_id` guard)
- `tcgen05_alloc` has a warp guard but **no lane guard** (all 32 lanes participate)
- TMA and MMA both loop `K_TILES` times (same `k < 16`)
- Phase init: producer=1, consumer=0

#### Debugging Deadlocks

**How to tell it's a deadlock**: The kernel hangs for ~30 seconds, then reports "unspecified launch failure." A crash (XID 43) fails instantly.

**Step 1: Verify barrier arrival counts match init counts.**

| Barrier | init(count) | Who arrives | Expected arrivals |
|---------|-------------|-------------|-------------------|
| tma2mma (TMABar) | init(1) | TMA warp via arrive(stage, bytes) | 1 |
| mma2tma (TCGen05Bar) | init(1) | MMA warp via tcgen05.commit | 1 |
| mma2ld (TCGen05Bar) | init(1) | MMA warp via tcgen05.commit | 1 |
| ld2mma (MBarrier) | init(128) | All Warpgroup 0 threads via arrive | 128 |

Common mistake: `ld2mma.init(128)` but `arrive` is guarded by `if warp_id == 0: if lane_id == 0:` — only 1 thread arrives, barrier never completes.

**Step 2: Check barrier inits actually execute.** The `.init()` wrapper uses `threadIdx.x < 1` internally. If you nest it inside `if wg_id == 1:`, no thread satisfies both conditions — thread 0 is in Warpgroup 0. Search the generated code:

```python
# This should show mbarrier_init at TOP level, not inside a wg_id guard
for i, line in enumerate(cuda_source.split("\n")):
    if "mbarrier_init" in line:
        # Print surrounding context to check nesting
        for j in range(max(0,i-5), i+1):
            print(f"  {cuda_source.split(chr(10))[j].rstrip()}")
        print()
```

**Step 3: Check cta_sync is not inside a warpgroup branch.** `cta_sync()` = `__syncthreads()`, requires ALL CTA threads. If called inside `if wg_id == 0:`, WG1 threads never reach it — instant deadlock. Fix: use `Tx.cuda.warpgroup_sync(10)` instead.

**Step 4: Check tile_scheduler.next_tile() is called by all threads.** `tile_scheduler` tracks per-thread state. If only one thread calls `next_tile()`, other threads loop forever.

**Step 5: Check both TMA and MMA iterate the same number of K tiles.** If MMA does `K_TILES - 1` iterations instead of `K_TILES`, barrier phases drift and the second tile deadlocks.

**Step 6: Check PipelineState is_producer settings.** Producer starts at phase 1 (first wait passes immediately). Consumer starts at phase 0 (first wait blocks until data arrives). If both start at the same phase, they deadlock on the first iteration.

#### Debugging Crashes (XID 43 / Illegal Memory Access)

These corrupt the CUDA context — even `torch.randn` fails afterward.

**Common causes:**

1. **pool.alloc after pool.commit()**: Barrier wrappers (`TMABar`, `TCGen05Bar`, `MBarrier`) call `pool.alloc()` internally. If created after `pool.commit()`, their SMEM addresses are invalid. Correct order:
```
tmem_addr → barrier wrappers → move_base_to(1024) → Asmem/Bsmem/Dsmem → commit()
```

2. **tcgen05.alloc/dealloc with lane guard**: These need all 32 lanes in the warp to participate. A `if lane_id == 0:` guard means only 1 thread executes — hardware requires all 32.

3. **Missing cta_sync() before tcgen05.dealloc**: TMEM freed while writeback still reading.

#### Debugging Wrong Results

If you see specific mismatch counts like 128, 253, or 381 (multiples of 128 = corrupted rows), it's a synchronization bug, not an arithmetic error.

**Common causes:**

1. **Missing fence.after_thread_sync() after mbarrier.wait**: Threads read TMEM before MMA hardware finishes writing it.

2. **Missing fence.proxy_async("shared::cta") before TMA store**: TMA engine doesn't see the SMEM writes from threads.

3. **Missing cp_async.bulk.commit_group() / wait_group(0) after TMA store**: The store doesn't complete before SMEM is reused for the next tile.

4. **tcgen05.commit outside elect_sync guard**: All 32 threads create commit groups, but only the elected thread has a non-empty one. The 31 empty commits signal the mbarrier immediately — TMA overwrites SMEM before MMA finishes reading it. Output is zeros or garbage.

5. **Persistent kernel: intermittent failures at small sizes (e.g., 1024x1024)**: If mismatch counts are multiples of 128 (128, 253, 254, 381 = corrupted rows), this is NOT GPU flakiness — it's a race condition. Barrier phases or TMA store synchronization are broken in the persistent tile loop. Larger sizes pass because longer K-loops change timing and mask the race. Verify that barrier phases are correctly reset between tiles, and that TMA stores complete (`commit_group` + `wait_group`) before the next tile starts.




**Pipeline depth tuning**: The demo above uses `PIPE_DEPTH=2` (the minimum). In production, increasing `PIPE_DEPTH` to 4 or 6 lets the TMA producer get further ahead of the MMA consumer, better hiding memory latency. B200 has 228 KB SMEM per SM (see :numref:`chap_background`, *Numbers to Remember*). With `BLK_M=BLK_N=128, BLK_K=64, fp16`, each pipeline stage uses `(128*64 + 128*64) * 2 = 32 KB` for A+B, and the `Dsmem` writeback staging buffer consumes `128*128*2 = 32 KB`. `PIPE_DEPTH=4` uses 128 KB for A+B, leaving ~68 KB for Dsmem and metadata. `PIPE_DEPTH=6` pushes total SMEM usage to ~224 KB (192 KB A+B + 32 KB Dsmem), fitting just inside the 228 KB budget. Going deeper (e.g., `PIPE_DEPTH=7` needing 256 KB) requires shrinking or eliminating the Dsmem buffer — for example, writing back to GMEM directly from registers instead of staging through SMEM+TMA store.

---

With warp specialization in place, the next optimization is to have multiple CTAs cooperate on a larger tile.


## Step 8: 2-CTA Cluster
:label:`chap_cta_cluster`

Two CTAs cooperate via cross-CTA shared memory, doubling the output width. M=N=K=4096.

### What You Will Learn

- CTA clusters: multiple CTAs cooperating on a larger tile

- Cross-CTA SMEM access via `map_shared_rank`

- `cta_group=2` for cooperative MMA (256x256 vs 128x128)

- Cross-CTA barrier signaling with `cta_mask`


### The Key Insight

With `cta_group=2`, the MMA hardware reads B from **both** CTAs' shared memory. Each CTA loads only 128 columns of B, but the MMA sees all 256 columns:

![2-CTA Cluster](../img/cta_cluster.png)

**Why A is split but B is shared**: Think of `D = A × B`. With 2 CTAs in a cluster:

- **A is split vertically**: CTA-0 holds A0 (rows 0-127), CTA-1 holds A1 (rows 128-255). Stacked: `[A0; A1]` (256 rows).
- **B is split horizontally**: CTA-0 loads B0 (columns 0-127), CTA-1 loads B1 (columns 128-255). Side by side: `[B0 | B1]` (256 columns).
- With `cta_group=2`, the MMA hardware reads B from **both** CTAs' SMEM via cross-CTA shared memory access, so it sees all 256 columns of `[B0 | B1]`.
- Result: `[A0; A1] × [B0 | B1]` = 256×256 output tile.

Each CTA only loads 128×K of A and 128×K of B, but together they compute a 256×256 output — 2x arithmetic intensity with the same memory traffic.

### Tile Address Calculation

With 2-CTA clusters, the tile scheduler works in units of 256x256 cluster tiles. Each `(m_idx, n_idx)` from the scheduler identifies a 256x256 region, and the two CTAs within the cluster split the work. The address calculation for each CTA is:

```python
m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
```

Here `cbx` is the CTA's position within the cluster (0 or 1). Both `m_st` and `n_st` depend on `cbx` because each CTA owns a different 128x128 sub-tile of the 256x256 cluster tile. The scheduler's `num_m_tiles = M // 256` and `num_n_tiles = N // 256` count cluster tiles, not individual CTA tiles.

### Code Changes from Step 7

The changes are surprisingly small --- TIRX abstracts away most of the complexity:

```python
# 1. Cluster launch
cbx, cby = Tx.cta_id([CTA_GROUP, 1], parent="cluster")   # cbx = CTA index within cluster (0 or 1)

# 2. Cooperative MMA (was cta_group=1)
Tx.gemm_async(..., cta_group=2)

# 3. Cross-CTA shared memory access
B_remote = Tx.ptx.map_shared_rank(Bsmem, cta_id=1)

# 4. Cross-CTA barrier
tma2mma_cta0 = Tx.decl_buffer(
    [CTA_GROUP], "uint64",
    data=Tx.ptx.map_shared_rank(tma2mma.ptr_to([0]), 0),
    scope="shared"
)

# 5. Cluster sync replaces cta_sync at the end
Tx.cuda.cluster_sync()
```


### New Concepts

- **Cluster CTA ID**: `cbx` tells each CTA its position in the cluster (0 or 1). CTA-0 handles A rows 0-127, CTA-1 handles rows 128-255.

- **Remote barrier view**: In a cluster, each CTA has its own SMEM and barriers. To synchronize across CTAs, CTA-1 needs to access CTA-0's barrier. `map_shared_rank(tma2mma.ptr_to([0]), 0)` returns a pointer to CTA-0's barrier that is accessible from any CTA in the cluster. The TIRx wrapper is `tma2mma.remote_view(0)`. All barrier operations (arrive, wait) target CTA-0's barriers — this is the coordination point.

- **MMA dispatch from CTA-0 only**: With `cta_group=2`, CTA-0 issues one `tcgen05.mma` instruction, and the hardware automatically dispatches the computation to **both** CTAs' tensor cores simultaneously. CTA-1 does not issue any MMA instruction — the hardware handles this coordination internally. This is why the code guards MMA with `if cbx == 0:`.

- **Multicast arrive**: Similarly, `tcgen05.commit(..., cta_group=2, cta_mask=3)` is issued only by CTA-0 but signals both CTAs' barriers. `cta_mask=3` (binary `11`) means both CTA-0 and CTA-1 are targeted.

- **ld2mma init count**: `init(128 * CTA_GROUP)` --- both CTAs' writeback warpgroups (128 threads each) arrive.


### Implementation

```{.python .input}
def hgemm_v8(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_M, MMA_N = 256, 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    WG_NUMBER = 2
    DTYPE_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, 128))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            cbx, cby = Tx.cta_id([CTA_GROUP, 1], parent="cluster")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            # --- Allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")
            mma2ld  = TCGen05Bar(pool, 1, "mma2ld")
            ld2mma  = MBarrier(pool, 1, "ld2mma")
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((BLK_M, 128), d_type, layout=D_layout)

            # --- Barrier init ---
            tma2mma.init(1)
            mma2tma.init(1)
            mma2ld.init(1)
            ld2mma.init(128 * CTA_GROUP)  # both CTAs' writeback threads
            pool.commit()

            # --- TMEM alloc (cooperative) ---
            if wg_id == 0:
                if warp_id == 0:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer(
                (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
                layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            # --- Tile scheduler (cluster tiles) ---
            tile_scheduler = ClusterPersistentScheduler2D(
                "ts", num_m_tiles=M // 256, num_n_tiles=N // 256,
                l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
            tile_scheduler.init(bx // CTA_GROUP)
            m_idx = Tx.meta_var(tile_scheduler.m_idx)
            n_idx = Tx.meta_var(tile_scheduler.n_idx)
            m_st = Tx.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
            n_st = Tx.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

            # --- Cross-CTA barrier view ---
            tma2mma_cta0 = tma2mma.remote_view(0)

            # =============================================
            # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
            # =============================================
            if wg_id == 1:
                if warp_id == 3:
                    tma_ps = PipelineState("tma", PIPE_DEPTH)
                    tma_ps.init(is_producer=True)

                    @Tx.inline
                    def tma_load(k_offset):
                        Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                      A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                      dispatch="tma", cta_group=CTA_GROUP,
                                      mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                        Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                      B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                      dispatch="tma", cta_group=CTA_GROUP,
                                      mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            for k in range(K_TILES):
                                mma2tma.wait(tma_ps.stage, tma_ps.phase)
                                tma_load(k * BLK_K)
                                if cbx == 0:
                                    tma2mma_cta0.arrive(tma_ps.stage,
                                        CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * DTYPE_SIZE)
                                tma_ps.move_to_next_stage()
                            tile_scheduler.next_tile()

                elif warp_id == 0:
                    mma_ps = PipelineState("mma", PIPE_DEPTH)
                    mma_ps.init(is_producer=False)
                    ld_ps = PipelineState("ld", 1)
                    ld_ps.init(is_producer=True)

                    if cbx == 0:
                        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                            while tile_scheduler.valid():
                                ld2mma.wait(ld_ps.stage, ld_ps.phase)
                                ld_ps.move_to_next_stage()

                                for k in range(K_TILES):
                                    tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()
                                    Tx.gemm_async(
                                        tmem[:, :MMA_N],
                                        Asmem[mma_ps.stage, :, :],
                                        Bsmem[mma_ps.stage, :, :],
                                        accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                    mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                    mma_ps.move_to_next_stage()

                                mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)
                                tile_scheduler.next_tile()

            # =============================================
            # Warpgroup 0: Writeback (256 columns in 2 x 128-column chunks)
            # =============================================
            elif wg_id == 0:
                wb_ps = PipelineState("wb", 1)
                wb_ps.init(is_producer=False)
                reg_f16 = Tx.alloc_local((128,), d_type)

                while tile_scheduler.valid():
                    mma2ld.wait(wb_ps.stage, wb_ps.phase)
                    wb_ps.move_to_next_stage()
                    Tx.ptx.tcgen05.fence.after_thread_sync()

                    for no in Tx.unroll(2):  # 2 chunks of 128 columns = 256 total
                        reg = Tx.alloc_local((128,), acc_type)
                        reg_wg = reg.view(128, 128,
                            layout=TileLayout(S[(128, 128) : (1@tid_in_wg, 1)]))
                        with Tx.warpgroup():
                            Tx.copy(reg_wg[:], tmem[:, no * 128:(no + 1) * 128])
                        with Tx.thread():
                            Tx.cast(reg_f16[:], reg[:])
                        with Tx.thread():
                            Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(10)
                        with Tx.thread(parent="warpgroup")[Tx.ptx.elect_sync()]:
                            n_st_epi = Tx.meta_var(n_idx * 256 + no * 128)
                            Tx.copy_async(D[m_st:m_st+BLK_M, n_st_epi:n_st_epi+128],
                                          Dsmem[:, :], dispatch="tma")
                            Tx.ptx.cp_async.bulk.commit_group()
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(10)

                    ld2mma.arrive(0, cta_id=0, pred=True)
                    tile_scheduler.next_tile()

            # --- Cleanup ---
            Tx.cuda.cluster_sync()
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

Compile and run:

```{.python .input}
M, N, K = 4096, 4096, 4096
kernel = hgemm_v8(M, N, K)
target = tvm.target.Target("cuda -arch=sm_100a")
with target:
    mod = tvm.IRModule({"main": kernel})
    lib = tvm.compile(mod, target=target, tir_pipeline="tirx")

A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

lib["main"](tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor),
            tvm.runtime.from_dlpack(D_tensor))
D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Step 8 2-CTA Cluster: M={M}, N={N}, K={K}")
print(f"Max error vs torch reference: {max_err:.6f}")
assert max_err < 0.5, f"FAIL: max_err={max_err}"  # smoke test; fp16 4096x4096 typically gives ~0.25
print("PASS")

args = [tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor), tvm.runtime.from_dlpack(D_tensor)]
for _ in range(10):
    lib["main"](*args)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    lib["main"](*args)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

### Key Points

- `CTA_GROUP = 2`, `MMA_N = BLK_N * CTA_GROUP = 256`

- `ld2mma.init(128 * CTA_GROUP)` --- both CTAs' writeback WGs arrive

- TMA arrive byte count includes both CTAs: `CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`

- `tcgen05.alloc` and `tcgen05.dealloc` must use `cta_group=2`

- Writeback uses `EPI_N=64` --- reading all 256 TMEM columns at once exceeds register capacity

- `cluster_sync()` replaces `cta_sync()` at the end (ensures all CTAs are done before TMEM dealloc)

With the final multi-consumer optimization (Step 9), we reach **0.12 ms** --- within 10% of cuBLAS.

If Step 8 is slower than Step 7, check: (1) TMA arrive byte count should be `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`, (2) tile scheduler dimensions should be `num_m_tiles=M//256, num_n_tiles=N//256` for the 256x256 cluster tile, (3) writeback has 4 TMA stores (256 cols / EPI_N=64) — make sure each one completes before reusing Dsmem.

---

The cluster optimization maximizes arithmetic intensity per memory access. The final step pushes compute density further by adding a second MMA consumer within each CTA.


## Step 9: Multi-Consumer Warp Specialization
:label:`chap_multi_consumer`

The final optimization adds a second MMA consumer, doubling the compute density per CTA. The cluster output grows from 256x256 to 512x256. M=N=K=4096.


### What You Will Learn

- Multiple MMA warps (consumers) for higher throughput

- Multiple writeback warpgroups with independent barrier slots

- The structure of production-grade Blackwell GEMM kernels


### Background

With `NUM_CONSUMER=2` and `WG_NUMBER=3`, there are three warpgroups (abbreviated WG below):

| Warpgroup | Warp | Role |
|-----------|------|------|
| **WG 2** | warp 0 | MMA consumer 0: `A[0:128,:] x B` -> TMEM cols `[0:256]` |
| **WG 2** | warp 1 | MMA consumer 1: `A[128:256,:] x B` -> TMEM cols `[256:512]` |
| **WG 2** | warp 3 | TMA producer: loads 2x A blocks + 1x B block per stage |
| **WG 0** | all | Writeback for consumer 0: reads TMEM `[0:256]` |
| **WG 1** | all | Writeback for consumer 1: reads TMEM `[256:512]` |


### Changes from Step 8

- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` --- 2 A blocks per stage, one per consumer

- TMA loads both `Asmem[stage, 0]` and `Asmem[stage, 1]`, with TMA arrive bytes now `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE` (extra A block)

- MMA warp `warp_id` selects which A block and TMEM range

- `mma2tma.init(NUM_CONSUMER)` --- both consumers signal TMA per stage

- `mma2ld` and `ld2mma` have `depth=NUM_CONSUMER` --- each consumer uses its own barrier slot (`warp_id` for MMA side, `wg_id` for writeback side)

- Tile address: `m_st = (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M` --- M direction has the extra `NUM_CONSUMER` factor because each cluster tile now spans `NUM_CONSUMER` consumers in M. Tile scheduler uses `num_m_tiles = M // 256 // NUM_CONSUMER` (cluster tile is 512x256)

- Writeback **must** use chunked `EPI_N` --- reading 256 TMEM columns at once exceeds register capacity


### Implementation

```{.python .input}
def hgemm_v9(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    CTA_GROUP = 2
    NUM_CONSUMER = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    MMA_N = BLK_N * CTA_GROUP   # 256
    K_TILES = K // BLK_K
    PIPE_DEPTH = 4
    EPI_N = 64
    WG_NUMBER = 3
    DTYPE_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (NUM_CONSUMER, BLK_M, EPI_N))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT], parent="kernel")
            cbx, cby = Tx.cta_id([CTA_GROUP, 1], parent="cluster")
            wg_id = Tx.warpgroup_id([WG_NUMBER], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            # --- Allocation ---
            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH, "tma2mma")
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH, "mma2tma")
            mma2ld  = TCGen05Bar(pool, NUM_CONSUMER, "mma2ld")   # depth=2, one slot per consumer
            ld2mma  = MBarrier(pool, NUM_CONSUMER, "ld2mma")     # depth=2, one slot per consumer
            pool.move_base_to(1024)
            Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
            Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)

            # --- Barrier init ---
            tma2mma.init(1)
            mma2tma.init(NUM_CONSUMER)  # each stage expects 2 arrivals
            mma2ld.init(1)              # each slot gets 1 arrival
            ld2mma.init(128 * CTA_GROUP)  # both CTAs' writeback threads
            pool.commit()

            # --- TMEM alloc (cooperative) ---
            if wg_id == 0:
                if warp_id == 0:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer(
                (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
                layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            # --- Tile scheduler (512x256 cluster tiles) ---
            tile_scheduler = ClusterPersistentScheduler2D(
                "ts", num_m_tiles=M // 256 // NUM_CONSUMER, num_n_tiles=N // 256,
                l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
            tile_scheduler.init(bx // CTA_GROUP)
            m_idx = Tx.meta_var(tile_scheduler.m_idx)
            n_idx = Tx.meta_var(tile_scheduler.n_idx)
            m_st = Tx.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
            n_st = Tx.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

            tma2mma_cta0 = tma2mma.remote_view(0)

            # =============================================
            # Warpgroup 2: TMA Producer (warp 3) + 2 MMA Consumers (warp 0, 1)
            # =============================================
            if wg_id == 2:
                if warp_id == 3:
                    # === TMA Producer: loads 2 A blocks + 1 B block per stage ===
                    tma_ps = PipelineState("tma", PIPE_DEPTH)
                    tma_ps.init(is_producer=True)

                    @Tx.inline
                    def tma_load(k_offset):
                        m_st_c1 = Tx.meta_var(m_st + CTA_GROUP * BLK_M)
                        Tx.copy_async(Asmem[tma_ps.stage, 0, :, :],
                                      A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                      dispatch="tma", cta_group=CTA_GROUP,
                                      mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                        Tx.copy_async(Asmem[tma_ps.stage, 1, :, :],
                                      A[m_st_c1:m_st_c1+BLK_M, k_offset:k_offset+BLK_K],
                                      dispatch="tma", cta_group=CTA_GROUP,
                                      mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                        Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                      B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                      dispatch="tma", cta_group=CTA_GROUP,
                                      mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        while tile_scheduler.valid():
                            for k in range(K_TILES):
                                mma2tma.wait(tma_ps.stage, tma_ps.phase)
                                tma_load(k * BLK_K)
                                if cbx == 0:
                                    tma2mma_cta0.arrive(tma_ps.stage,
                                        CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * DTYPE_SIZE)
                                tma_ps.move_to_next_stage()
                            tile_scheduler.next_tile()

                elif warp_id < NUM_CONSUMER:
                    # === MMA Consumer: warp_id selects A block and TMEM range ===
                    mma_ps = PipelineState("mma", PIPE_DEPTH)
                    mma_ps.init(is_producer=False)
                    ld_ps = PipelineState("ld", 1)
                    ld_ps.init(is_producer=True)

                    if cbx == 0:
                        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                            while tile_scheduler.valid():
                                ld2mma.wait(warp_id, ld_ps.phase)
                                ld_ps.move_to_next_stage()

                                for k in range(K_TILES):
                                    tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                    Tx.ptx.tcgen05.fence.after_thread_sync()
                                    Tx.gemm_async(
                                        tmem[:, warp_id * MMA_N:warp_id * MMA_N + MMA_N],
                                        Asmem[mma_ps.stage, warp_id, :, :],
                                        Bsmem[mma_ps.stage, :, :],
                                        accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                    mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                    mma_ps.move_to_next_stage()

                                mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                                tile_scheduler.next_tile()

            # =============================================
            # Warpgroup 0/1: Writeback (each reads its consumer's TMEM range)
            # =============================================
            elif wg_id < NUM_CONSUMER:
                wb_ps = PipelineState("wb", 1)
                wb_ps.init(is_producer=False)
                reg_f16 = Tx.alloc_local((EPI_N,), d_type)

                while tile_scheduler.valid():
                    mma2ld.wait(wg_id, wb_ps.phase)  # wait for THIS consumer
                    wb_ps.move_to_next_stage()
                    Tx.ptx.tcgen05.fence.after_thread_sync()

                    # Read TMEM in EPI_N=64 column chunks (4 iterations for 256 cols)
                    for i in Tx.unroll(MMA_N // EPI_N):
                        reg = Tx.alloc_local((EPI_N,), acc_type)
                        reg_wg = reg.view(128, EPI_N,
                            layout=TileLayout(S[(128, EPI_N) : (1@tid_in_wg, 1)]))
                        with Tx.warpgroup():
                            col_st = Tx.meta_var(wg_id * MMA_N + i * EPI_N)
                            col_end = Tx.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
                            Tx.copy(reg_wg[:], tmem[:, col_st:col_end])
                        with Tx.thread():
                            Tx.cast(reg_f16[:], reg[:])
                        with Tx.thread():
                            Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], reg_f16[:])
                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(wg_id + 10)
                        with Tx.thread(parent="warpgroup")[Tx.ptx.elect_sync()]:
                            m_st_epi = Tx.meta_var(
                                (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                            n_st_epi = Tx.meta_var(n_idx * MMA_N + i * EPI_N)
                            Tx.copy_async(
                                D[m_st_epi:m_st_epi+BLK_M, n_st_epi:n_st_epi+EPI_N],
                                Dsmem[wg_id, :, :], dispatch="tma")
                            Tx.ptx.cp_async.bulk.commit_group()
                            Tx.ptx.cp_async.bulk.wait_group(0)
                        Tx.cuda.warpgroup_sync(wg_id + 10)

                    ld2mma.arrive(wg_id, cta_id=0, pred=True)
                    tile_scheduler.next_tile()

            # --- Cleanup ---
            Tx.cuda.cluster_sync()
            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

Compile and run:

```{.python .input}
M, N, K = 4096, 4096, 4096
kernel = hgemm_v9(M, N, K)
target = tvm.target.Target("cuda -arch=sm_100a")
with target:
    mod = tvm.IRModule({"main": kernel})
    lib = tvm.compile(mod, target=target, tir_pipeline="tirx")

A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

lib["main"](tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor),
            tvm.runtime.from_dlpack(D_tensor))
D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Step 9 Multi-Consumer: M={M}, N={N}, K={K}")
print(f"Max error vs torch reference: {max_err:.6f}")
assert max_err < 0.5, f"FAIL: max_err={max_err}"  # smoke test; fp16 4096x4096 typically gives ~0.25
print("PASS")

args = [tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor), tvm.runtime.from_dlpack(D_tensor)]
for _ in range(10):
    lib["main"](*args)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    lib["main"](*args)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

### Key Points

- `mma2ld` and `ld2mma` must be **one shared object** each with `depth=NUM_CONSUMER`, NOT separate per-consumer objects. Slot 0 connects MMA warp 0 to Warpgroup 0; Slot 1 connects MMA warp 1 to Warpgroup 1. MMA side uses `warp_id` as index, writeback side uses `wg_id`.

- This is the structure of production-grade Blackwell GEMM kernels


## Exercises

1. What happens if you set `is_producer=False` for both the TMA and MMA pipeline states in Step 7? Draw the deadlock scenario.
2. With `cta_group=2` in Step 8, the TMA arrive byte count is `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`. Why multiply by `CTA_GROUP` when each CTA loads its own data?
3. In Step 9, each consumer handles different M rows but the same B tile. Why is sharing B (not A) the right choice?


**Benchmarking:**

```python
M, N, K = 4096, 4096, 4096
kernel = hgemm_v9(M, N, K)
target = tvm.target.Target("cuda -arch=sm_100a")
with target:
    lib = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

# Time the kernel
device = torch.device('cuda')
A = torch.randn(M, K, dtype=torch.float16, device=device)
B = torch.randn(N, K, dtype=torch.float16, device=device)
D = torch.zeros(M, N, dtype=torch.float16, device=device)
f = lib["main"]
args = [tvm.runtime.from_dlpack(A), tvm.runtime.from_dlpack(B), tvm.runtime.from_dlpack(D)]
for _ in range(10):
    f(*args)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    f(*args)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
tflops = 2 * M * N * K / ms / 1e9
print(f"{ms:.3f} ms, {tflops:.1f} TFLOPS")
```


### The Complete Journey

Benchmarked on NVIDIA B200, M=N=K=4096, fp16:

| Step | Technique | Time | Speedup |
|------|-----------|------|---------|
| 1 | Sync load + MMA | 70 ms | 1x |
| 2 | K-loop accumulation | --- | Handle arbitrary K |
| 3 | Spatial tiling | --- | Handle arbitrary M, N |
| 4 | TMA async load | 0.50 ms | 141x |
| 5 | Software pipeline | --- | Overlap load + compute |
| 6 | Persistent kernel | --- | L2 cache locality |
| 7 | Warp specialization | 0.23 ms | 300x |
| 8 | 2-CTA cluster | 0.13 ms | 537x |
| 9 | Multi-consumer | 0.12 ms | 582x |
| --- | cuBLAS (reference) | 0.11 ms | 614x |

The 4 key optimization techniques that deliver the biggest gains:
1. **TMA Async Data Movement** --- hardware copy engine replaces software copy (141x speedup)
2. **Software Pipelining + Warp Specialization** --- overlap load and compute with dedicated roles (2x over TMA alone)
3. **CTA Clusters** --- 2-SM cooperative MMA doubles arithmetic intensity (2x speedup)
4. **Multi-Consumer** --- 2 MMA warps for higher compute density (additional ~10% over Step 8)

![GEMM Optimization Journey](../img/gemm_perf.png)

582x speedup: from 70 ms to 0.12 ms --- within 10% of cuBLAS!
