# GEMM Steps 7--9: Warp Specialization to cuBLAS
:label:`chap_gemm_advanced`

This chapter covers the three advanced GEMM optimizations that bridge the gap from a basic tiled kernel to cuBLAS-level performance. We start with warp specialization and software pipelining (Step 7), introduce CTA clusters for cooperative MMA (Step 8), and finish with multi-consumer warp specialization (Step 9).

> **Layout recap.** The per-step SMEM / TMEM / register layouts repeat the pattern established in :numref:`chap_gemm_basics` and :numref:`chap_gemm_async`. Step 8 additionally uses cluster-scope buffers; for the `cbx` / `cby` axes and how a single layout can span two CTAs, see the *Where Replicas Show Up* paragraph in :numref:`chap_layouts`.


## Step 7: Warp Specialization + Pipeline
:label:`chap_warp_specialization`

This is the biggest architectural change: instead of all threads doing load-then-compute sequentially, we dedicate specific warps to specific tasks and overlap them with software pipelining. We benchmark with M=N=K=4096.

### What You Will Learn

- Warp specialization: dedicating different warps/warpgroups to different tasks

- High-level barrier abstractions: `TMABar`, `TCGen05Bar`, `MBarrier`

- `RingState` for automatic stage/phase management

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

### RingState

`RingState` manages stage indices and phase counters for a software-pipelined ring buffer:

```python
tma_ps = RingState(PIPE_DEPTH, phase=1)   # Producer starts ready (phase=1)
# tma_ps.stage = current stage index
# tma_ps.phase = current phase (0 or 1)
tma_ps.advance()                          # Advance to next stage
```

**The initial `phase` argument controls when the first `wait` passes:**
- `phase=1` (producer) -> first `wait(phase=1)` sees barrier phase 0 != 1, so it **passes immediately** (buffers start empty, producer can write)

- `phase=0` (consumer) -> first `wait(phase=0)` sees barrier phase 0 == 0, so it **blocks** (consumer waits for data)

Getting this wrong causes deadlock or data corruption.

### `warpgroup_sync` Barrier IDs

The GPU hardware has 16 named barriers (ID 0-15). `cta_sync()` uses barrier #0 by default. In warp-specialized kernels, different warpgroups run different code paths, so we cannot use `cta_sync()` (which requires ALL threads). Instead, we use `warpgroup_sync(10)` — this syncs only threads within the warpgroup, using barrier #10 to avoid conflicting with barrier #0. In multi-consumer kernels (Step 9), each warpgroup uses a different ID: `warpgroup_sync(wg_id + 10)`.

### Implementation

The complete Step 7 kernel. `PIPE_DEPTH=2` for clarity (production kernels use `PIPE_DEPTH=4` or higher — see "Pipeline depth tuning" below for the SMEM trade-offs involved).

```{.python .input}
import tvm
from tvm.script import tirx as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.operator.tile_primitive.cuda.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.pipeline import TMABar, TCGen05Bar, MBarrier, RingState
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

    @Tx.prim_func
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT])
            wg_id = Tx.warpgroup_id([WG_NUMBER])
            warp_id = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])

            # --- Allocation ---
            pool = Tx.SMEMPool()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH)
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
            mma2ld  = TCGen05Bar(pool, 1)
            ld2mma  = MBarrier(pool, 1)
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
                    tma_ps = RingState(PIPE_DEPTH, phase=1)

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

                    with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
                        while tile_scheduler.valid():
                            for k in range(K_TILES):
                                mma2tma.wait(tma_ps.stage, tma_ps.phase)
                                tma_load(k * BLK_K)
                                tma2mma.arrive(tma_ps.stage,
                                               (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                                tma_ps.advance()
                            tile_scheduler.next_tile()

                elif warp_id == 0:
                    # === MMA Consumer ===
                    mma_ps = RingState(PIPE_DEPTH, phase=0)
                    ld_ps = RingState(1, phase=1)

                    with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
                        while tile_scheduler.valid():
                            # Wait for TMEM to be free from previous tile's writeback
                            ld2mma.wait(ld_ps.stage, ld_ps.phase)
                            ld_ps.advance()

                            for k in range(K_TILES):
                                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                Tx.gemm_async(
                                    tmem[:, :BLK_N],
                                    Asmem[mma_ps.stage, :, :],
                                    Bsmem[mma_ps.stage, :, :],
                                    accum=(k != 0), dispatch="tcgen05", cta_group=1)
                                mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
                                mma_ps.advance()

                            # Signal results ready for writeback
                            mma2ld.arrive(0, cta_group=1, cta_mask=0)
                            tile_scheduler.next_tile()

            # =============================================
            # Warpgroup 0: Writeback
            # =============================================
            elif wg_id == 0:
                wb_ps = RingState(1, phase=0)
                reg_f16 = Tx.alloc_local((BLK_N,), d_type)

                while tile_scheduler.valid():
                    # Wait for MMA results
                    mma2ld.wait(wb_ps.stage, wb_ps.phase)
                    wb_ps.advance()

                    # Read TMEM -> registers (warpgroup scope)
                    reg = Tx.alloc_local((BLK_N,), acc_type)
                    reg_wg = reg.view(128, BLK_N,
                        layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
                    with Tx.warpgroup():
                        Tx.copy_async(reg_wg[:], tmem[:, :BLK_N])
                        Tx.ptx.tcgen05.wait.ld()

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
                    with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
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
target = tvm.target.Target("cuda")
with target:
    mod = tvm.IRModule({"main": kernel})
    lib = tvm.compile(mod, target=target, tir_pipeline="tirx")

device = torch.device('cuda')  # gpu(0)
torch.cuda.empty_cache()
torch.cuda.synchronize()
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
ITERS = 10
for _ in range(3):
    lib["main"](*args)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    lib["main"](*args)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

### Epilogue (Writeback) Details

Step 7 keeps the epilogue simple: the writeback warpgroup reads the full `BLK_N=128` columns of TMEM into registers in one pass, then issues a single TMA store per tile. The sequence is:

1. Wait for MMA: `mma2ld.wait(phase)`. The `tirx-kernels` reference inserts `fence.after_thread_sync()` here as a conservative extra — the MMA-completion mbarrier already covers the ordering, and most kernels (including CUTLASS) omit it.
2. Read TMEM -> registers (128 fp32 per thread, warpgroup scope via `Tx.copy_async(reg_wg, tmem[:, :BLK_N])` followed by `Tx.ptx.tcgen05.wait.ld()`).
3. Signal MMA: `ld2mma.arrive()` (all 128 threads arrive) — TMEM is now free for the next tile.
4. Cast fp32 -> fp16 in registers.
5. Write registers -> Dsmem, then `fence.proxy_async("shared::cta") + warpgroup_sync(10)` to flush.
6. TMA store Dsmem -> GMEM via `cp_async.bulk.commit_group() + wait_group(0)`.

Step 8 (with `BLK_N=256`) and Step 9 (with `MMA_N=256` per consumer) revisit this epilogue and split the writeback into `EPI_N`-column chunks (`EPI_N=64`). The reason is register pressure: reading 256 fp32 per thread (256 * 4 = 1024 bytes, distributed across 128 threads and their registers) risks spills to local memory, and it also forces a larger Dsmem buffer. By chunking into 64-column groups, each iteration keeps only `EPI_N` fp32 registers live and issues a smaller TMA store.

### Key Points

- **Persistent kernel**: `bx = Tx.cta_id([SM_COUNT])` --- one CTA per SM, loops over tiles

- **L2-friendly scheduling**: `ClusterPersistentScheduler2D` orders tiles for cache locality

- This pattern --- warp specialization + software pipeline --- is used in **ALL production GPU kernels** (CUTLASS, cuBLAS, etc.)

### When Step 7 misbehaves

Warp specialization is the most bug-prone step in the tutorial: TMA, MMA, and writeback now run concurrently, and any barrier mistake either deadlocks, crashes the CUDA context, or silently corrupts the output. The exact same failure modes resurface in Steps 8 and 9, so we collect the debugging playbook in one place — see :numref:`chap_warp_spec_debug` at the end of this chapter when something goes wrong.

**Pipeline depth tuning.** The kernel above uses `PIPE_DEPTH=2` (the minimum). In production, increasing it to 4 or 6 lets the TMA producer get further ahead of the MMA consumer, hiding more memory latency. B200 has 228 KB SMEM per SM (see :numref:`chap_background`, *Numbers to Remember*). With `BLK_M=BLK_N=128, BLK_K=64, fp16`, each pipeline stage uses `(128*64 + 128*64) * 2 = 32 KB` for A+B, and the `Dsmem` writeback staging buffer consumes another 32 KB. `PIPE_DEPTH=4` ⇒ 128 KB A+B + 32 KB Dsmem ≈ 160 KB used. `PIPE_DEPTH=6` ⇒ 192 KB + 32 KB = 224 KB, fitting just inside the SMEM budget. Going to `PIPE_DEPTH=7` (256 KB) requires dropping the Dsmem staging buffer — e.g. writing back to GMEM directly from registers.

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

Each CTA still loads 128×K of A and 128×K of B (its own halves), but the cooperative MMA uses both CTAs' SMEM tiles to compute a 256×256 output — so the *flops per byte fetched by each CTA* roughly doubles.

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
cbx, cby = Tx.cta_id_in_cluster([CTA_GROUP, 1])   # cbx = CTA index within cluster (0 or 1)

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

- **Remote barrier view**: In a cluster, each CTA has its own SMEM and barriers. To synchronize across CTAs, CTA-1 needs to access CTA-0's barrier. `map_shared_rank(tma2mma.ptr_to([0]), 0)` returns a pointer to CTA-0's barrier that is accessible from any CTA in the cluster. The TIRX wrapper is `tma2mma.remote_view(0)`. All barrier operations (arrive, wait) target CTA-0's barriers — this is the coordination point.

- **MMA dispatch from CTA-0 only**: With `cta_group=2`, CTA-0 issues one `tcgen05.mma`. The hardware drives a *single cooperative* MMA that spans both CTAs in the cluster — it reads operands from both SMs' SMEM and writes the accumulator across both SMs' TMEM. CTA-1 does not issue any MMA instruction. (Each SM still has only one `tcgen05` engine; "cta_group=2" is one cross-SM MMA, not two engines firing in parallel.) This is why the code guards MMA with `if cbx == 0:`.

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

    @Tx.prim_func
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT])
            cbx, cby = Tx.cta_id_in_cluster([CTA_GROUP, 1])
            wg_id = Tx.warpgroup_id([WG_NUMBER])
            warp_id = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])

            # --- Allocation ---
            pool = Tx.SMEMPool()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH)
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
            mma2ld  = TCGen05Bar(pool, 1)
            ld2mma  = MBarrier(pool, 1)
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
                    tma_ps = RingState(PIPE_DEPTH, phase=1)

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

                    with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
                        while tile_scheduler.valid():
                            for k in range(K_TILES):
                                mma2tma.wait(tma_ps.stage, tma_ps.phase)
                                tma_load(k * BLK_K)
                                if cbx == 0:
                                    tma2mma_cta0.arrive(tma_ps.stage,
                                        CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * DTYPE_SIZE)
                                tma_ps.advance()
                            tile_scheduler.next_tile()

                elif warp_id == 0:
                    mma_ps = RingState(PIPE_DEPTH, phase=0)
                    ld_ps = RingState(1, phase=1)

                    if cbx == 0:
                        with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
                            while tile_scheduler.valid():
                                ld2mma.wait(ld_ps.stage, ld_ps.phase)
                                ld_ps.advance()

                                for k in range(K_TILES):
                                    tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                    Tx.gemm_async(
                                        tmem[:, :MMA_N],
                                        Asmem[mma_ps.stage, :, :],
                                        Bsmem[mma_ps.stage, :, :],
                                        accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                    mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                    mma_ps.advance()

                                mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)
                                tile_scheduler.next_tile()

            # =============================================
            # Warpgroup 0: Writeback (256 columns in 2 x 128-column chunks)
            # =============================================
            elif wg_id == 0:
                wb_ps = RingState(1, phase=0)
                reg_f16 = Tx.alloc_local((128,), d_type)

                while tile_scheduler.valid():
                    mma2ld.wait(wb_ps.stage, wb_ps.phase)
                    wb_ps.advance()
                    Tx.ptx.tcgen05.fence.after_thread_sync()

                    for no in Tx.unroll(2):  # 2 chunks of 128 columns = 256 total
                        reg = Tx.alloc_local((128,), acc_type)
                        reg_wg = reg.view(128, 128,
                            layout=TileLayout(S[(128, 128) : (1@tid_in_wg, 1)]))
                        with Tx.warpgroup():
                            Tx.copy_async(reg_wg[:], tmem[:, no * 128:(no + 1) * 128])
                            Tx.ptx.tcgen05.wait.ld()
                        with Tx.thread():
                            Tx.cast(reg_f16[:], reg[:])
                        with Tx.thread():
                            Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(10)
                        with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
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
target = tvm.target.Target("cuda")
with target:
    mod = tvm.IRModule({"main": kernel})
    lib = tvm.compile(mod, target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
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
ITERS = 10
for _ in range(3):
    lib["main"](*args)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    lib["main"](*args)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

### Key Points

- `CTA_GROUP = 2`, `MMA_N = BLK_N * CTA_GROUP = 256`

- `ld2mma.init(128 * CTA_GROUP)` --- both CTAs' writeback WGs arrive

- TMA arrive byte count includes both CTAs: `CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`

- `tcgen05.alloc` and `tcgen05.dealloc` must use `cta_group=2`

- Writeback splits the 256 output columns into two 128-column chunks --- reading all 256 TMEM columns at once exceeds register capacity. Step 9 shrinks the chunk further to `EPI_N=64`

- `cluster_sync()` replaces `cta_sync()` at the end (ensures all CTAs are done before TMEM dealloc)

Step 8 brings us to **0.13 ms** at 4096³ (~538× over Step 1). Step 9 below adds a second MMA consumer to close the remaining gap to cuBLAS.

If Step 8 is slower than Step 7, check: (1) TMA arrive byte count should be `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`, (2) tile scheduler dimensions should be `num_m_tiles=M//256, num_n_tiles=N//256` for the 256x256 cluster tile, (3) writeback issues 2 TMA stores (one per 128-column chunk) — make sure each one completes before reusing Dsmem.

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

    @Tx.prim_func
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx = Tx.cta_id([SM_COUNT])
            cbx, cby = Tx.cta_id_in_cluster([CTA_GROUP, 1])
            wg_id = Tx.warpgroup_id([WG_NUMBER])
            warp_id = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])

            # --- Allocation ---
            pool = Tx.SMEMPool()
            tmem_addr = pool.alloc((1,), "uint32")
            tma2mma = TMABar(pool, PIPE_DEPTH)
            mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
            mma2ld  = TCGen05Bar(pool, NUM_CONSUMER)   # depth=2, one slot per consumer
            ld2mma  = MBarrier(pool, NUM_CONSUMER)     # depth=2, one slot per consumer
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
                    tma_ps = RingState(PIPE_DEPTH, phase=1)

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

                    with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
                        while tile_scheduler.valid():
                            for k in range(K_TILES):
                                mma2tma.wait(tma_ps.stage, tma_ps.phase)
                                tma_load(k * BLK_K)
                                if cbx == 0:
                                    tma2mma_cta0.arrive(tma_ps.stage,
                                        CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * DTYPE_SIZE)
                                tma_ps.advance()
                            tile_scheduler.next_tile()

                elif warp_id < NUM_CONSUMER:
                    # === MMA Consumer: warp_id selects A block and TMEM range ===
                    mma_ps = RingState(PIPE_DEPTH, phase=0)
                    ld_ps = RingState(1, phase=1)

                    if cbx == 0:
                        with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
                            while tile_scheduler.valid():
                                ld2mma.wait(warp_id, ld_ps.phase)
                                ld_ps.advance()

                                for k in range(K_TILES):
                                    tma2mma.wait(mma_ps.stage, mma_ps.phase)
                                    Tx.gemm_async(
                                        tmem[:, warp_id * MMA_N:warp_id * MMA_N + MMA_N],
                                        Asmem[mma_ps.stage, warp_id, :, :],
                                        Bsmem[mma_ps.stage, :, :],
                                        accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                                    mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                                    mma_ps.advance()

                                mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
                                tile_scheduler.next_tile()

            # =============================================
            # Warpgroup 0/1: Writeback (each reads its consumer's TMEM range)
            # =============================================
            elif wg_id < NUM_CONSUMER:
                wb_ps = RingState(1, phase=0)
                reg_f16 = Tx.alloc_local((EPI_N,), d_type)

                while tile_scheduler.valid():
                    mma2ld.wait(wg_id, wb_ps.phase)  # wait for THIS consumer
                    wb_ps.advance()
                    Tx.ptx.tcgen05.fence.after_thread_sync()

                    # Read TMEM in EPI_N=64 column chunks (4 iterations for 256 cols)
                    for i in Tx.unroll(MMA_N // EPI_N):
                        reg = Tx.alloc_local((EPI_N,), acc_type)
                        reg_wg = reg.view(128, EPI_N,
                            layout=TileLayout(S[(128, EPI_N) : (1@tid_in_wg, 1)]))
                        with Tx.warpgroup():
                            col_st = Tx.meta_var(wg_id * MMA_N + i * EPI_N)
                            col_end = Tx.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
                            Tx.copy_async(reg_wg[:], tmem[:, col_st:col_end])
                            Tx.ptx.tcgen05.wait.ld()
                        with Tx.thread():
                            Tx.cast(reg_f16[:], reg[:])
                        with Tx.thread():
                            Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], reg_f16[:])
                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.cuda.warpgroup_sync(wg_id + 10)
                        with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
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
target = tvm.target.Target("cuda")
with target:
    mod = tvm.IRModule({"main": kernel})
    lib = tvm.compile(mod, target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
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
ITERS = 10
for _ in range(3):
    lib["main"](*args)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    lib["main"](*args)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

### Key Points

- `mma2ld` and `ld2mma` must be **one shared object** each with `depth=NUM_CONSUMER`, NOT separate per-consumer objects. Slot 0 connects MMA warp 0 to Warpgroup 0; Slot 1 connects MMA warp 1 to Warpgroup 1. MMA side uses `warp_id` as index, writeback side uses `wg_id`.

- This is the structure of production-grade Blackwell GEMM kernels


## Debugging Warp-Specialized Kernels
:label:`chap_warp_spec_debug`

Steps 7, 8 and 9 all share the same set of failure modes: TMA, MMA, and writeback run concurrently and any barrier mistake either deadlocks, crashes the CUDA context, or silently corrupts the output. This section is the consolidated playbook — bookmark it; you will come back here.

### Inspecting Generated Code

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

### Reference: Generated CUDA Skeleton

A correctly compiled Step 7 kernel has the following top-level skeleton. When debugging, run `inspect_source()` and check that your output matches this structure:

```c
// (1) Barrier inits  — top level, CTA thread 0 only
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // arrived by all 128 WG0 threads
}

// (2) TMEM alloc  — WG0 warp 0, all 32 lanes (no lane guard)
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) Fences + cta_sync, then phase init: producer=1, consumer=0

// (4) Warp-specialized loop
if (wg_id == 1 && warp_id == 3 && elect_sync) { /* TMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 1 && warp_id == 0 && elect_sync) { /* MMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 0)                                { /* WB   */ while(valid){ ... next_tile(); } }

// (5) Cleanup  — warp 0, no lane guard
cta_sync();
if (warp_id == 0) { tcgen05_relinquish_alloc_permit(); tcgen05_dealloc(..., 512); }
```

Things to verify in your generated code:

- Barrier inits sit at **top level** (not inside a `wg_id` guard) — see *Debugging Deadlocks* below.
- `tcgen05_alloc`/`dealloc` have a **warp guard but no lane guard** — all 32 lanes participate.
- TMA and MMA loops both iterate `K_TILES` times.
- Phase init: producer=`1`, consumer=`0`.

### Symptom → Likely Cause

First diagnose by the *shape* of the failure:

| Symptom | Failure class | Section below |
|---|---|---|
| Kernel hangs ~30 s, then "unspecified launch failure" | Deadlock | *Deadlocks* |
| Crash within ms; subsequent `torch.randn` also fails | XID 43 / illegal memory access | *Crashes* |
| Output max-err in multiples of 128 (128 / 253 / 381 rows) | Sync race | *Wrong results* |
| Output `NaN` everywhere | MMA descriptor / TMA descriptor mismatch | *Wrong results* |

### Deadlocks

Most deadlocks are *one* of these — work through the list:

- **Arrival count ≠ init count.** Common case: `MBarrier.init(128)` but `arrive` is guarded by `if warp_id == 0: if lane_id == 0:` — only 1 thread arrives, the wait never returns. Reference:

  | Barrier | init(count) | Who arrives | Arrivals |
  |---|---|---|---|
  | `TMABar` (tma→mma) | 1 | TMA engine via `arrive(stage, bytes)` | 1 |
  | `TCGen05Bar` (mma→tma, mma→ld) | 1 | MMA warp via `tcgen05.commit` | 1 |
  | `MBarrier` (ld→mma) | 128 | All WG0 threads via `arrive` | 128 |

- **Barrier init nested inside a `wg_id` guard.** `.init()` lowers to `if threadIdx.x < 1:` (i.e., thread 0 of the CTA, which lives in WG0). Nest it inside `if wg_id == 1:` and **no** thread satisfies both — barrier stays uninitialized. Inits must be at top level; `grep mbarrier_init` in `inspect_source()` to verify.

- **`cta_sync()` inside a warpgroup branch.** `cta_sync` is `__syncthreads()` — requires *all* CTA threads. Inside `if wg_id == 0:`, WG1 never reaches it. Use `Tx.cuda.warpgroup_sync(10)` instead.

- **`tile_scheduler.next_tile()` not called by all threads in the consumer warpgroup.** The scheduler tracks per-thread state; threads that skip it loop forever.

- **TMA and MMA disagree on K-tile count.** If MMA does `K_TILES - 1` instead of `K_TILES`, barrier phases drift and the second outer tile deadlocks.

- **`RingState` initial phase wrong.** Producer must start at `phase=1` (first wait passes); consumer at `phase=0` (first wait blocks). Same starting phase ⇒ instant deadlock.

### Crashes (XID 43 / illegal memory access)

These corrupt the CUDA context — subsequent `torch.randn` will also fail.

- **`pool.alloc` after `pool.commit()`.** Barrier wrappers call `alloc` internally. Correct order: `tmem_addr → barrier wrappers → move_base_to(1024) → Asmem / Bsmem / Dsmem → commit()`.
- **`tcgen05.alloc` / `dealloc` with a lane guard.** They require all 32 lanes of the warp to participate; `if lane_id == 0:` runs one thread and the hardware traps.
- **Missing `cta_sync()` before `tcgen05.dealloc`** — TMEM is freed while writeback is still reading.

### Wrong results

If you see mismatch counts that are exact multiples of 128 (128 / 253 / 381 rows), it's a sync race, not an arithmetic error.

- **`tcgen05.commit` outside `elect_sync`** — all 32 threads create commit groups; the 31 empty ones signal the mbarrier immediately. TMA overwrites SMEM before MMA reads it. Output is zeros or garbage.
- **Missing `fence.proxy_async("shared::cta")` before TMA store** — TMA engine doesn't see SMEM writes from threads.
- **Missing `cp_async.bulk.commit_group()` + `wait_group(0)` after TMA store** — next tile reuses Dsmem before the store drains.
- **Persistent kernel, intermittent fails at small sizes (e.g., 1024×1024)** — not GPU flakiness. Larger sizes mask the race with longer K-loops. Re-check phase reset between tiles and the TMA-store commit/wait.
- **`fence.after_thread_sync()` is *almost never* the cause.** The MMA-completion mbarrier already carries release→acquire semantics, so most production kernels (including CUTLASS) omit it. The `tirx-kernels` reference kernels add it as a conservative measure on the writeback edge (after `mma2ld.wait`, before the first `tcgen05.ld`). Do **not** add it on the TMA→MMA edge. If your output is wrong, the cause is almost certainly one of the bullets above.


## The Complete Journey

Reference numbers on NVIDIA B200, M=N=K=4096, fp16, locked clocks, 1000-iteration timed benchmark:

| Step | Technique | Time | Speedup |
|------|-----------|------|---------|
| 1 | Sync load + MMA | 70 ms | 1× |
| 2 | K-loop accumulation | --- | Handle arbitrary K |
| 3 | Spatial tiling | --- | Handle arbitrary M, N |
| 4 | TMA async load | 0.50 ms | ~140× |
| 5 | Software pipeline | --- | Overlap load + compute |
| 6 | Persistent kernel | --- | L2 cache locality |
| 7 | Warp specialization | 0.23 ms | ~304× |
| 8 | 2-CTA cluster | 0.13 ms | ~538× |
| 9 | Multi-consumer | 0.12 ms | ~583× |
| --- | cuBLAS (reference) | 0.11 ms | ~636× |

The `{.python .input}` benchmark cells embedded in each step above ran a short 10-iteration timed loop without clock locking, so the numbers you saw printed are typically 2-5× slower than the table on a casual GPU — same ordering, same trends, just not peak.

The 4 key optimization techniques that deliver the biggest gains:

1. **TMA Async Data Movement** — hardware copy engine replaces software copy (~140× from Step 1 → Step 4).
2. **Software Pipelining + Warp Specialization** — overlap load and compute with dedicated roles (~2.2× from Step 4 → Step 7).
3. **CTA Clusters** — 2-SM cooperative MMA doubles per-CTA FLOPs/byte (~1.8× from Step 7 → Step 8).
4. **Multi-Consumer** — two MMA warps for higher compute density (~8% from Step 8 → Step 9).

![GEMM Optimization Journey](../img/gemm_perf.png)

~583× total speedup: 70 ms → 0.12 ms — within ~10% of cuBLAS (0.11 ms).


## Exercises

1. What happens if you set the initial `phase` to `0` for both the TMA and MMA `RingState` in Step 7? Draw the deadlock scenario.
2. With `cta_group=2` in Step 8, the TMA arrive byte count is `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`. Why multiply by `CTA_GROUP` when each CTA loads its own data?
3. In Step 9, each consumer handles different M rows but the same B tile. Why is sharing B (not A) the right choice?
