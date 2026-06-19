(chap_gemm_async)=
# Pipelining GEMM with TMA

:::{admonition} Overview
:class: overview

- The basic GEMM wastes time taking turns — copy a tile, compute, copy the next — when the two could run at once.
- Step 4 switches to TMA async loads, Step 5 double-buffers SMEM and prefetches (PIPE_DEPTH=2); full load/compute overlap arrives with warp specialization in Step 7, Step 6 makes the kernel persistent with a tile scheduler.
- The goal is to load the next tile while the Tensor Cores chew through the current one.
:::

The previous chapter got us a correct tiled GEMM, but it leaves the Tensor Cores idle for much of the run — threads copy a tile, then sit idle while the Tensor Cores work, then copy the next tile. The compute and the data movement take turns when they could be running at once. This chapter keeps the tile data path exactly as it was and goes after that idleness, by changing *when* and *by whom* the work is scheduled.

We do it in three incremental steps. Step 4 hands the bulk GMEM <-> SMEM transfers to TMA, so dedicated hardware moves tiles instead of threads. Step 5 adds a two-stage software pipeline so the load for the next K tile can overlap the MMA on the current one. Step 6 reshapes the launch into a persistent kernel driven by a tile scheduler, amortizing setup and improving locality. Throughout, the SMEM, TMEM, and register layouts stay on the contracts from before; what is new is the asynchronous handoff between hardware units.

(chap_tma_async)=
## Step 4: TMA Async Load

Step 4 moves the copy itself off the critical path. In Steps 1-3 every thread in the CTA computes addresses and issues load instructions just to shuttle tiles into SMEM — instruction bandwidth spent on plumbing rather than math. Step 4 replaces synchronous `Tx.copy` with TMA: one thread issues a single command and the TMA engine carries out the whole tile transfer. From here on the examples run at the full M=N=K=4096 size rather than the small sizes of Steps 1-3, and their end-to-end timings appear in the *End-to-End Result* table at the end of {ref}`chap_gemm_advanced`.

> **What this step changes — Dispatch**
> - Scope: unchanged — one warpgroup.
> - Layout: unchanged — same SMEM/TMEM/register tiles.
> - Dispatch: GMEM → SMEM loads move from sync `Tx.copy` to the TMA engine.

### TMA Issue Pattern

The edit to the source is only a few lines, but the execution model behind those lines is different in kind. A synchronous `Tx.copy` is work the CTA threads do themselves; a TMA copy is a command one thread issues, after which the TMA hardware does the moving. The two side by side:

**Before (Step 3)** — all 128 threads participate in the copy, then `cta_sync` makes the shared-memory writes visible:
```python
Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])   # all 128 threads
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()
```

**After (Step 4)** — one thread issues the TMA load, and the mbarrier tracks when the hardware transfer is complete:
```python
tid = warp_id * 32 + lane_id                 # 0..127 within the warpgroup
if tid == 0:  # exactly one thread starts TMA
    Tx.copy_async(Asmem, A[...], dispatch="tma")
    Tx.copy_async(Bsmem, B[...], dispatch="tma")
    T.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)  # bytes expected from TMA
T.ptx.mbarrier.try_wait(tma_bar, phase)                  # wait before MMA reads SMEM
```

Notice that the load is gated on `tid == 0`, not on `elect_sync()`. The distinction matters here. `elect.sync` elects one active lane *per warp*, and a warpgroup has four warps, so `elect_sync()` would let four threads enter the load protocol. That protocol announces the expected byte count to the mbarrier, and it must announce it exactly once; four announcements would corrupt the count. A single warpgroup-wide thread id is the clean way to pick precisely one thread.

Step 4 still waits after every TMA load, so the speedup is not from load/compute overlap — that arrives in Step 5. It comes only from the change of data-movement path:

- `Tx.copy` uses CTA threads to compute addresses and issue load/store instructions.
- TMA uses one issued command to start a hardware tile transfer. Address generation, coalescing, and swizzling are described by the TMA descriptor and carried out by the TMA engine.

Even though Step 4 still blocks on each load, it is faster: TMA absorbs the bulk transfer, so the CTA threads spend far less instruction bandwidth shuffling tiles.

### TMA Load and Store Synchronization

Switching to TMA changes two things at once — who starts a copy and how the code knows it finished — and getting the second one wrong is a silent correctness bug. With `Tx.cta.copy`, the CTA threads do the copy together and a following `cta_sync()` is enough. With TMA, one selected thread issues `Tx.copy_async(..., dispatch="tma")`, the engine performs the transfer on its own schedule, and it signals completion through an mbarrier.

That difference is why `cta_sync()` is no longer sufficient. `cta_sync()` waits only for the CTA's own threads and orders only their shared-memory writes; it has no knowledge of an in-flight TMA transfer and returns while the tile is still arriving. So for a TMA load the selected thread first tells the mbarrier how many bytes to expect, and the CTA waits on *that* mbarrier before any MMA touches the SMEM tile.

![TMA Async Load: Synchronization Flow](../img/tma_sync_flow.png)

The figure traces the load side only. Where it says "Elected Thread" it means the selected thread that starts TMA — in our code that is the `tid == 0` thread, not an `elect_sync()` lane.

Putting it together on the load path: the selected thread issues both `copy_async` calls and then `arrive.expect_tx(total_bytes)`, where the byte count is precisely how much data the mbarrier should hold out for. Once the engine has moved that many bytes, `mbarrier.try_wait(phase)` releases and the SMEM tile is safe to feed to MMA.

The store side travels the same hardware but waits differently, so keep the two protocols apart: loads track completion with mbarriers and byte counts, stores track it with commit groups and wait groups. After the threads write their fp16 results into `Dsmem` and synchronize, one selected thread starts `Tx.copy_async(D[...], Dsmem, dispatch="tma")`, and then `cp_async.bulk.commit_group()` followed by `cp_async.bulk.wait_group(0)` block until the store drains — necessary before `Dsmem` can be reused for the next tile.

**Try with your agent**: Trace the Step 4 load and store synchronization for one K tile. Identify which thread starts each TMA command, which mbarrier or commit group tracks completion, which wait protects MMA reads of `Asmem` and `Bsmem`, and which wait protects reuse of `Dsmem`. Why would `elect_sync()` be the wrong thread selection for the TMA load protocol here?

### Complete Kernel

The complete kernel folds the TMA load and store into the Step 3 structure. Same imports:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

It is wrapped in `hgemm_v4(M, N, K)`, which keeps the shape-dependent constants and layouts right next to the kernel that uses them:

```python
def hgemm_v4(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    F16_SIZE = 2

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM allocation (now includes Dsmem for TMA store) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((1,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()
    
        # --- Barrier + TMEM init ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
    
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()
    
        tmem = T.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )
    
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0
    
        # --- Inline helpers ---
        @T.inline
        def tma_load(k_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([0])
            })
            Tx.copy_async(Asmem[:, :],
                          A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[:, :],
                          B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([0]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
            )
    
        @T.inline
        def mma(accum):
            Tx.gemm_async(
                tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                accum=accum, dispatch="tcgen05", cta_group=1
            )
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        # --- K-loop with TMA async ---
        tid = T.meta_var(warp_id * 32 + lane_id)
        for k in range(K_TILES):
            k_st = T.meta_var(k * BLK_K)
    
            # Single thread issues TMA load
            if tid == 0:
                tma_load(k_st)
    
            # Wait for TMA to finish; the mbarrier release carries SMEM
            # visibility to the subsequent MMA, so no extra fence is needed.
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)
    
            # Single thread issues MMA
            if tid == 0:
                mma(accum=k != 0)
    
            # Wait for MMA to finish
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_tma ^= 1
            phase_mma ^= 1
    
        # --- TMA Store Writeback ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
    
        # Read TMEM -> registers (async; wait.ld then cta_sync to ensure read completes)
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        # Cast fp32 -> fp16
        Tx.cast(Dreg_f16[:], Dreg[:])
        # Write registers -> Dsmem, flush, then sync
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        # TMA store: Dsmem -> GMEM. One selected thread starts the store and drains the
        # store group before Dsmem is reused.
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)
    
        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

### TMA Configuration in the Kernel

Five configuration points carry the TMA semantics; the rest is unchanged from Step 3:

- **TMA config**: `{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}` tells `Tx.copy_async` to use TMA and to report load completion through `tma_bar`.

- **Byte count**: `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` is the number of bytes loaded by the two fp16 operand tiles. `arrive.expect_tx(...)` gives this count to the mbarrier.

- **mbarrier initialization**: `init(tma_bar.ptr_to([0]), 1)` creates the completion barrier used by the TMA load.

- **`@T.inline`**: `tma_load(...)` and `mma(...)` are helper functions. They are expanded into the kernel body at compile time and can use variables from the surrounding kernel.

- **TMA store synchronization**: The epilogue first writes fp16 rows into `Dsmem`. `fence.proxy_async` and `warpgroup_sync` make those thread-written SMEM values ready for the TMA store path. The store then uses `commit_group()` and `wait_group(0)` to wait for the SMEM-to-GMEM transfer to finish.

Step 4 has the right machinery but the wrong rhythm: it still finishes each load before starting the matching MMA, so the load and the multiply never run at the same time. The next step leaves the TMA load/store path alone and rearranges the schedule, so that loading one K tile can proceed while compute runs on another.

(chap_software_pipeline)=
## Step 5: Software Pipeline (PIPE_DEPTH=2)

The obstacle to overlap in Step 4 is storage: with one SMEM tile pair, the next load cannot begin until the current MMA has finished reading that pair, or it would overwrite data still in use. Step 5 removes the conflict by double-buffering shared memory — two stages, so the kernel can fill one while computing on the other. Same full M=N=K=4096 size.

> **What this step changes — Layout**
> - Scope: unchanged — one warpgroup.
> - Layout: the single SMEM tile pair becomes a `PIPE_DEPTH`-stage ring buffer.
> - Dispatch: unchanged — TMA load and `tcgen05` MMA, now overlapped across stages.

### Pipeline Walkthrough

With `PIPE_DEPTH=2`, the kernel allocates two SMEM stages, and the two hardware engines can then work a tile apart from each other. The figure shows the target schedule across the K tiles:

![*Pipeline PIPE_DEPTH=2 — the target schedule; this single-warpgroup step only prefetches, full overlap arrives with warp specialization in Step 7*](../img/pipe_depth2.png)

The schedule reads as a steady alternation once it is primed. Two TMA loads fill both stages up front; from then on, while MMA computes on `k0`, TMA loads `k2` into the stage that will be reused next, and while MMA computes on `k1`, TMA loads `k3`, and so on. This works because the two engines are genuinely independent: TMA moves GMEM -> SMEM tiles, while `tcgen05.mma` consumes an already-loaded SMEM stage and writes the accumulator to TMEM, so neither blocks the other once each has its own buffer.

One caveat about this step. The single-warpgroup pipeline here overlaps only the *prefetched* stages — the main loop still waits for the current MMA (`try_wait(mma_bar, phase_mma)`) before issuing the next TMA load, so it stops short of the fully concurrent picture the figure paints. Reaching that picture needs a dedicated load warp issuing TMA while a separate MMA warp computes, which is warp specialization, introduced as Step 7 in {ref}`chap_gemm_advanced`. Step 5 lays the foundation: it builds the multi-stage SMEM ring buffer and the per-stage barriers that Step 7 turns into real overlap, and the prefetch already pulls the first loads off the critical path.

The code differs from Step 4 in four places:

1. `Asmem` and `Bsmem` gain a leading `PIPE_DEPTH` dimension, so each stage has its own SMEM storage.
2. `tma_bar` becomes an array with one mbarrier per stage.
3. Before the main K loop, the kernel prefetches the first two stages.
4. The K loop uses `stage = k % PIPE_DEPTH`: wait for the current stage, run MMA on it, then reuse that stage for `k + PIPE_DEPTH`.

### Pipeline Mechanics

Those four changes split into three concerns — priming the buffers, steady-state iteration, and tracking the barrier phases — taken here in that order.

**1. Prefetch**: before the main loop, load the first `PIPE_DEPTH` stages so the loop always has data ready when it starts:
```python
for s in range(min(PIPE_DEPTH, K_TILES)):
    tma_load(s, s * BLK_K)
```

**2. Main loop**: for each K tile, wait for its stage to be ready, run MMA on it, then immediately reuse that now-free stage by launching the load for the tile `PIPE_DEPTH` ahead:
```python
stage = k % PIPE_DEPTH
wait(tma_bar[stage], phase_tma)
mma(stage, accum)
wait(mma_bar[0], phase_mma)
phase_mma ^= 1
tma_load(stage, next_k * BLK_K)
```

**3. Phase management**: the phase flip rule for each barrier follows directly from how many slots that barrier has, so the two barriers flip on different cadences. The MMA accumulator is one TMEM slot, so `mma_bar` is a single barrier (`mma_bar.ptr_to([0])`) that every iteration revisits — hence `phase_mma` flips every iteration. The TMA barriers, by contrast, form a `PIPE_DEPTH`-element array with one barrier per stage; any given stage's barrier comes back around only once per trip through the ring, so `phase_tma` flips only when the stage index wraps to 0:
```python
if stage == PIPE_DEPTH - 1:
    phase_tma ^= 1
```

**Try with your agent**: With `PIPE_DEPTH=2` and `K_TILES=5`, ask it to trace the main loop. For each `k`, list `stage`, the `phase_tma` and `phase_mma` values passed to the waits, and whether a new prefetch is issued. Where exactly does `phase_tma` flip, and why is there no prefetch for the last two iterations?

### Complete Kernel

The complete kernel keeps the Step 4 TMA load/store path verbatim and wraps it in the staged buffers and phase logic above. Same imports:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
```

It is wrapped in `hgemm_v5(M, N, K)`. The `PIPE_DEPTH=2` constant sets the number of pipeline stages — two here, for double buffering — and is the single knob that turns this from a one-stage kernel into a pipelined one:

```python
PIPE_DEPTH = 2

def hgemm_v5(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    # Double-buffered layouts: first dimension is pipeline stage
    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        # Double-buffered TMA barriers (one per stage), single MMA barrier
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # Initialize barriers: PIPE_DEPTH for TMA, 1 for MMA
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                for s in range(PIPE_DEPTH):
                    T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)
        phase_tma: T.int32 = 0
        phase_mma: T.int32 = 0

        @T.inline
        def tma_load(stage, k_offset):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        tid = T.meta_var(warp_id * 32 + lane_id)

        # === Prefetch: load first PIPE_DEPTH stages ===
        if tid == 0:
            for s in range(min(PIPE_DEPTH, K_TILES)):
                tma_load(s, s * BLK_K)

        # === Main loop ===
        for k in range(K_TILES):
            stage = k % PIPE_DEPTH

            # Wait for TMA to finish loading this stage
            T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

            # MMA on this stage's data
            if tid == 0:
                mma(stage, accum=(k != 0))

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

            # Issue next prefetch load (k + PIPE_DEPTH)
            next_k = k + PIPE_DEPTH
            if next_k < K_TILES:
                if tid == 0:
                    tma_load(stage, next_k * BLK_K)

            # TMA phase flips when stage wraps around
            if stage == PIPE_DEPTH - 1:
                phase_tma ^= 1

        # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.cast(Dreg_f16[:], Dreg[:])
        Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        if tid == 0:
            Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                          Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

(chap_persistent_kernel)=
## Step 6: Persistent Kernel + Tile Scheduler

Up to now we have optimized inside a tile; Step 6 optimizes across tiles. Step 5 launches one CTA per 128 x 128 output tile — 1024 CTAs for a 4096 x 4096 output — and each one pays its own startup cost and then disappears. Step 6 instead launches a fixed pool of CTAs and has each one process many tiles in turn. That buys two things: per-tile setup is amortized over the many tiles a CTA handles, and because tile assignment now lives in the kernel, the scheduler can choose a tile *order* that reuses operands. Same full M=N=K=4096 size.

> **What this step changes — Scope**
> - Scope: a fixed pool of persistent CTAs, each looping over many output tiles via the scheduler.
> - Layout: unchanged — the same per-tile SMEM/TMEM/register path.
> - Dispatch: unchanged.

### Persistent Scheduling

A persistent kernel sizes its grid to the hardware, not to the problem: it launches `SM_COUNT` CTAs — about one per SM — regardless of how many output tiles there are, aiming for persistent occupancy. Exact 1:1 residency is not guaranteed; it depends on occupancy and how the hardware schedules CTAs. On the B200 here, `SM_COUNT=148`, and each of those 148 CTAs loops over the tiles handed to it by `ClusterPersistentScheduler2D`. Amortization is the first effect — TMEM allocation, barrier initialization, and scheduling now happen once per CTA and are reused across the roughly 7 tiles it handles, instead of 1024 times across throwaway CTAs. The second effect comes from the order the scheduler picks: `l2_group_size=8` groups nearby tiles so that tiles sharing a row band reuse the same A row-tiles (and tiles sharing a column band, the same B tiles), and running them back-to-back keeps those operands hot in L2 rather than re-fetching them from HBM. This is the reuse Step 3 left on the table.

```python
bx = T.cta_id([SM_COUNT])  # 1D grid, one CTA per SM

tile_scheduler = ClusterPersistentScheduler2D(
    "ts",
    num_m_tiles=M // BLK_M,
    num_n_tiles=N // BLK_N,
    l2_group_size=8,       # Group 8 nearby tiles together
    num_clusters=SM_COUNT
)
tile_scheduler.init(bx)
```

Looping over tiles has one correctness consequence. Each tile runs its own fresh K-loop, so its barrier phases must start from a known state. In Step 5 a CTA handled exactly one tile, so initializing `phase_tma` and `phase_mma` once was fine. In Step 6 those initializers move *inside* the `while tile_scheduler.valid()` loop, so each tile begins with phase state that matches its own TMA and MMA work rather than inheriting whatever the previous tile left behind:

```python
while tile_scheduler.valid():
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0
    ...
```

### Complete Kernel

Structurally the kernel is Step 5's pipeline wrapped in a tile-level outer loop. It needs one extra import, the scheduler:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D
```

The grid dimension is now `SM_COUNT` instead of `(M//BLK_M, N//BLK_N)`, and a `ClusterPersistentScheduler2D` hands each CTA its tiles:

```python
SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
PIPE_DEPTH = 2

def hgemm_v6(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")
    F16_SIZE = 2
    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                  (BLK_M, BLK_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 1D grid: one CTA per SM (not a 2D grid anymore!)
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- SMEM allocation (same as Step 5) ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
        Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
        pool.commit()

        # --- Barrier + TMEM init (same as Step 5) ---
        if warp_id == 0 and lane_id == 0:
            T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        # Tile scheduler: assigns tiles to CTAs in L2-friendly order
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT
        )
        tile_scheduler.init(bx)

        tid = T.meta_var(warp_id * 32 + lane_id)

        @T.inline
        def tma_load(stage, k_offset, m_st, n_st):
            tma_config = T.meta_var({
                "dispatch": "tma", "cta_group": 1,
                "mbar": tma_bar.ptr_to([stage])
            })
            Tx.copy_async(Asmem[stage, :, :],
                          A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                          **tma_config)
            Tx.copy_async(Bsmem[stage, :, :],
                          B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                          **tma_config)
            T.ptx.mbarrier.arrive.expect_tx(
                tma_bar.ptr_to([stage]),
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

        @T.inline
        def mma(stage, accum):
            Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
                          accum=accum, dispatch="tcgen05", cta_group=1)
            T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        # === Outer loop: iterate over tiles ===
        while tile_scheduler.valid():
            # Get current tile position from scheduler
            m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

            # === Inner loop: same pipeline as Step 5 ===
            phase_tma: T.int32 = 0
            phase_mma: T.int32 = 0

            # Prefetch first PIPE_DEPTH stages
            if tid == 0:
                for s in range(min(PIPE_DEPTH, K_TILES)):
                    tma_load(s, s * BLK_K, m_st, n_st)

            # Main K-loop
            for k in range(K_TILES):
                stage = k % PIPE_DEPTH
                T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)
                if tid == 0:
                    mma(stage, accum=(k != 0))
                T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma ^= 1
                next_k = k + PIPE_DEPTH
                if next_k < K_TILES:
                    if tid == 0:
                        tma_load(stage, next_k * BLK_K, m_st, n_st)
                if stage == PIPE_DEPTH - 1:
                    phase_tma ^= 1

            # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
            Dreg = T.alloc_local((BLK_N,), acc_type)
            Dreg_f16 = T.alloc_local((BLK_N,), d_type)
            Dreg_wg = Dreg.view(128, BLK_N,
                                layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
            Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
            T.ptx.tcgen05.wait.ld()
            T.cuda.cta_sync()
            Tx.cast(Dreg_f16[:], Dreg[:])
            Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.warpgroup_sync(10)
            if tid == 0:
                Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
                              Dsmem[:, :], dispatch="tma")
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group(0)
            T.cuda.warpgroup_sync(10)

            T.cuda.cta_sync()
            tile_scheduler.next_tile()  # Move to next tile

        # Deallocate TMEM
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## Exercises

1. In Step 4, `arrive.expect_tx` uses `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` bytes. What does the mbarrier wait for if this byte count is too small or too large?
2. In Step 5, why does each SMEM stage need its own TMA barrier instead of sharing one `tma_bar` for both stages?
3. In Step 6, a 4096 x 4096 output with `BLK_M=BLK_N=128` has how many output tiles? With `SM_COUNT=148`, how many tiles does each persistent CTA process on average?
