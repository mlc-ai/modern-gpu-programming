(chap_gemm_advanced)=
# Scaling GEMM with Warp Specialization and Clusters

:::{admonition} Overview
:class: overview

- The pipelined GEMM still has one warpgroup doing load, MMA, and writeback in sequence — the bottleneck this chapter removes.
- Step 7 specializes warps into roles, Step 8 adds a 2-CTA cluster, Step 9 adds multiple consumers.
- Each step removes a serial bottleneck, ending near state-of-the-art throughput.
:::

The previous chapter ({ref}`chap_gemm_async`) ended with a persistent, software-pipelined GEMM, but it still left one warpgroup doing load, MMA, and writeback in sequence. That single warpgroup is the bottleneck this chapter attacks. We scale the kernel in three steps. Each step stops one team of threads from taking turns at jobs that could run at once, and makes more hardware cooperate on each tile.

The three steps widen cooperation one level at a time. Step 7 splits the warpgroup into specialized roles — a warp issuing TMA loads (*producer*), a warp running the MMA (*consumer*), and a writeback warpgroup — so loading and computing overlap instead of alternating. Step 8 lets two CTAs cooperate as a cluster, so one `tcgen05` MMA produces a 256×256 tile spanning both CTAs and a single B load feeds twice the MMA work. Step 9 adds a second MMA consumer, growing the cluster output to 512×256 so each staged B tile is reused by both consumers — the densest variant in the tutorial.

Throughout, the SMEM, TMEM, and register layouts still follow the contracts from the previous two chapters; what changes is *who cooperates*. Step 8 is the first time the cooperating scope widens past a single CTA: its operand tiles are split across the two CTAs' shared memory, and one layout spans both CTAs along the `cbx` cluster axis introduced there.


(chap_warp_specialization)=
## Step 7: Warp Specialization + Pipeline

The single-warpgroup kernel leaves performance on the table because every thread walks the same path — load, then compute, then write — so while it loads, the Tensor Cores are idle, and while it computes, the TMA engine is idle. The fix is *warp specialization*: hand each job to a dedicated warp and let those warps run concurrently, connected by a software pipeline. This is the biggest architectural change in the GEMM path, and the rest of the chapter builds on it. Benchmarks use M=N=K=4096.

> **What this step changes — Scope**
> - Scope: one warpgroup walking load → MMA → writeback in order becomes three concurrent roles — TMA producer, MMA consumer, writeback — connected by full/empty barriers.
> - Layout: unchanged — same SMEM stages and TMEM accumulator as Step 6.
> - Dispatch: unchanged — TMA loads, `tcgen05` MMA.

**Topics.**

- Warp specialization: dedicating different warps/warpgroups to different tasks

- High-level barrier abstractions: `TMABar`, `TCGen05Bar`, `MBarrier`

- `PipelineState` for automatic stage/phase management

- `warpgroup_sync` barrier IDs for per-warpgroup synchronization

(The multi-stage SMEM pipeline and the persistent `ClusterPersistentScheduler2D` are reused unchanged from Steps 5–6; only the scope split is new here.)

### From Sequential to Concurrent

![Warp Specialization Timeline](../img/warp_specialization_timeline.png)

The figure is a before-and-after contrast. On top, single-warpgroup Step 6: the one warp must finish the MMA before it can issue the next TMA load, so the TMA engine sits idle through the whole MMA and the Tensor Cores sit idle through the whole load. On the bottom, specialization breaks that turn-taking: the TMA producer prefetches data while the MMA consumer computes, and writeback runs independently. Producer warp 3 issues the next load while consumer warp 0 is still running the current MMA, so neither engine waits on the other. The price of overlap is coordination — the warps now need to tell each other when data is ready and when a buffer is free. Two barriers carry those messages:

- **`tma2mma`** (TMA → MMA): signals that the loaded SMEM data is ready for MMA to consume.
- **`mma2tma`** (MMA → TMA): signals that MMA has finished reading a buffer, so TMA can reuse it for the next load.

The `mma2tma` arrows in the figure skip ahead by a stage. With `PIPE_DEPTH=2` there are two SMEM buffers (stage 0 and stage 1). TMA Load k=0 fills buffer 0, TMA Load k=1 fills buffer 1. When MMA Compute k=0 finishes reading buffer 0 it signals `mma2tma` — but the load that actually wants buffer 0 back is TMA Load k=2, not k=1 (which uses buffer 1). So the `mma2tma` arrow from MMA Compute k=0 points all the way to TMA Load k=2. The buffer release skips a stage because the ring buffer has two slots.

### Warp Roles

With `WG_NUMBER=2`, the kernel uses two warpgroups (abbreviated WG in the role table):

| Actor | Location | Job |
|-------|----------|-----|
| **TMA Producer** | Warpgroup 1, warp 3 | Continuously loads A and B tiles via TMA |
| **MMA Consumer** | Warpgroup 1, warp 0 | Runs MMA as soon as data is ready |
| **Writeback** | Warpgroup 0 (all warps) | Reads TMEM results, writes to GMEM |

### 4 Barriers

Three concurrent actors need four barriers, and the four fall into two opposite directions. The forward path (TMA → MMA → Writeback) signals data *readiness* — "the tile you need is here." The backward path (Writeback → MMA → TMA) signals buffer *release* — "the slot you wanted is free again." Reading the names is easy once you know the convention: each is `source2destination`, so `tma2mma` is the barrier on which TMA signals MMA.

| Barrier | Type | Direction | Meaning |
|---------|------|-----------|---------|
| **tma2mma** | `TMABar` | TMA -> MMA | "SMEM data is ready" |
| **mma2tma** | `TCGen05Bar` | MMA -> TMA | "SMEM buffer can be reused" |
| **mma2ld** | `TCGen05Bar` | MMA -> Writeback | "TMEM results are ready" |
| **ld2mma** | `MBarrier` | Writeback -> MMA | "TMEM is free for next tile" |

The barrier *type* follows from how the producer announces completion. **TMA Loads** use `TMABar` (an mbarrier with byte counting): the TMA hardware arrives on the barrier itself once the transfer's bytes land, so consumers learn the data is ready without any thread polling. **TMA Stores** use a different mechanism, `cp_async.bulk.commit_group()` + `wait_group(0)`, because a store has nobody to notify — the issuing thread waits for its own write to drain. **MMA operations** use `TCGen05Bar`, where the `tcgen05.commit()` instruction signals the barrier when the MMA finishes.

One detail here pays off in Step 8: the `arrive` calls pass `cta_mask=0`, since in a single-CTA kernel there is no other CTA to signal. When Step 8 forms a cluster, that same argument turns nonzero to wake the cooperating CTAs.

### PipelineState

A ring buffer needs two pieces of bookkeeping: which slot we are on, and which "phase" of that slot's barrier we are waiting for. Tracking both by hand across a pipelined loop invites off-by-one errors that deadlock a kernel, so `PipelineState` keeps them together:

```python
tma_ps = PipelineState(PIPE_DEPTH, phase=1)   # Producer starts ready (phase=1)
# tma_ps.stage = current stage index
# tma_ps.phase = current phase (0 or 1)
tma_ps.advance()                          # Advance to next stage
```

The initial `phase` decides whether the first `wait` lets a role run or makes it block, and the right answer is opposite for the two ends of the pipe:
- `phase=1` (producer) -> the first `wait(phase=1)` sees barrier phase 0 != 1, so it **passes immediately**. That is what we want: the buffers start empty, so the producer should be free to write at once.

- `phase=0` (consumer) -> the first `wait(phase=0)` sees barrier phase 0 == 0, so it **blocks**. Also what we want: there is no data yet, so the consumer should wait.

Give both ends the same starting phase and you get a deadlock or silent corruption.

### `warpgroup_sync` Barrier IDs

Specialization creates a synchronization trap. Once each warpgroup runs a different code path, `cta_sync()` deadlocks: it uses hardware barrier #0 and demands that *every* CTA thread arrive, but a warpgroup branch only ever holds some of them. We need a barrier scoped to a single warpgroup instead. The GPU has 16 named barriers (ID 0–15), so the kernels use `warpgroup_sync(10)`, which synchronizes only the threads inside one warpgroup. When several warpgroups must each sync independently — as in the multi-consumer Step 9 — they take distinct IDs with `warpgroup_sync(wg_id + 10)` so they do not collide on the same hardware barrier.

**Implementation.**

We set `PIPE_DEPTH=2` here — the smallest depth that still overlaps load and compute. Deeper variants hide more memory latency, bounded by the SMEM budget; the *When Step 7 misbehaves* discussion below works through that trade-off. With the pieces in hand — roles, barriers, `PipelineState`, and warpgroup-scoped sync — here is the full kernel:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
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

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
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
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
            l2_group_size=8, num_clusters=SM_COUNT)
        tile_scheduler.init(bx)
        m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
        n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

        # =============================================
        # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                # === TMA Producer ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=1,
                                  mbar=tma2mma.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
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
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if T.filter(lane_id, T.ptx.elect_sync()):
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
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((BLK_N,), d_type)

            while tile_scheduler.valid():
                # Wait for MMA results
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()

                # Read TMEM -> registers (warpgroup scope)
                reg = T.alloc_local((BLK_N,), acc_type)
                reg_wg = reg.view(128, BLK_N,
                    layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
                Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
                T.ptx.tcgen05.wait.ld()

                # Signal TMEM free (all 128 threads arrive)
                ld2mma.arrive(0, cta_id=0, pred=True)

                # Cast fp32 -> fp16
                Tx.cast(reg_f16[:], reg[:])

                # Write to Dsmem + TMA store
                Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.warpgroup_sync(10)
                if warp_id == 0:
                    if lane_id == 0:
                        Tx.copy_async(D[m_st:m_st+BLK_M, n_st:n_st+BLK_N],
                                      Dsmem[:, :], dispatch="tma")
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0)
                T.cuda.warpgroup_sync(10)

                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

To run any of these kernels, reuse the compile / run / check harness shown once in Step 1 ({ref}`chap_gemm_basics`): replace `hgemm_v1` with `hgemm_v7`, `hgemm_v8`, or `hgemm_v9` and pick a problem size (e.g. `M=N=K=4096`). The clustered steps need `M`, `N` to be multiples of their cluster tile — `256×256` for Step 8, `512×256` for Step 9 — so a tiny `128×128` size produces no tiles. Compile one step per fresh Python session (restart the kernel before switching steps), since the kernels reuse inner names and the compiler keeps per-session state. Per-step timings are summarized in *End-to-End Result* below.

### Epilogue (Writeback) Details

Step 7 can afford a simple epilogue: with only `BLK_N=128` columns, the writeback warpgroup reads the whole TMEM tile into registers in one pass and issues a single TMA store. (Steps 8 and 9 cannot, which is the point of the chunking we add later.) The sequence is:

1. Wait for MMA: `mma2ld.wait(phase)`. Steps 8 and 9 in this tutorial add a `fence.after_thread_sync()` here as a conservative extra — the MMA-completion mbarrier already covers the ordering, and most kernels (including CUTLASS) omit it, so Step 7 does too.
2. Read TMEM -> registers (128 fp32 per thread, warpgroup scope via `Tx.copy_async(reg_wg, tmem[:, :BLK_N])` followed by `T.ptx.tcgen05.wait.ld()`).
3. Signal MMA: `ld2mma.arrive(0, cta_id=0, pred=True)` (all 128 threads arrive) — TMEM is now free for the next tile. The two `arrive` kwargs recur in the clustered steps: `cta_id` names *which CTA's* copy of the barrier to signal (`0` = this CTA, the local barrier; in Step 8 the cooperative arrives target CTA-0 via `cta_mask` instead), and `pred` is a per-thread predicate gating whether this thread actually arrives (`True` here, so every writeback thread counts toward the arrival total).
4. Cast fp32 -> fp16 in registers.
5. Write registers -> Dsmem, then `fence.proxy_async("shared::cta") + warpgroup_sync(10)` to flush.
6. TMA store Dsmem -> GMEM via `cp_async.bulk.commit_group() + wait_group(0)`.

Step 8 (with `BLK_N=256`) and Step 9 (with `MMA_N=256` per consumer) cannot keep this one-pass form because of register pressure. Reading 256 fp32 per thread means 256 × 4 = 1024 bytes live in each thread's registers at once, which risks spilling to local memory, and it also forces a larger Dsmem buffer. So those steps split the writeback into `EPI_N`-column chunks (`EPI_N=64`): each iteration keeps only `EPI_N` fp32 registers live and issues a correspondingly smaller TMA store.

**Implementation notes.**

- **Persistent kernel**: `bx = T.cta_id([SM_COUNT])` --- one CTA per SM, loops over tiles

- **L2-friendly scheduling**: `ClusterPersistentScheduler2D` orders tiles for cache locality

- This pattern --- warp specialization plus software pipelining --- is common in high-performance GEMM kernels, including CUTLASS-style designs.

### When Step 7 misbehaves

Concurrency makes Step 7 fast, and also makes it the first GEMM kernel that is easy to break. With TMA, MMA, and writeback all running at once, a single misplaced barrier can deadlock the kernel, crash the CUDA context, or corrupt the output. The same failure modes return in Steps 8 and 9, so rather than repeat them three times we collect the debugging playbook in *Debugging Warp-Specialized Kernels* at the end of this chapter; reach for it when something goes wrong.

**Pipeline depth tuning.** The Step 7 kernel uses `PIPE_DEPTH=2` (the minimum). Increasing it to 4 or 6 can let the TMA producer get further ahead of the MMA consumer, hiding more memory latency, but it consumes more SMEM. B200 has 228 KB SMEM per SM (see *Numbers to Keep in Mind* in {ref}`chap_background`). With `BLK_M=BLK_N=128, BLK_K=64, fp16`, each pipeline stage uses `(128*64 + 128*64) * 2 = 32 KB` for A+B, and the `Dsmem` writeback staging buffer consumes another 32 KB. `PIPE_DEPTH=4` uses about 160 KB; `PIPE_DEPTH=6` uses about 224 KB, close to the SMEM budget. Going deeper requires changing the writeback staging strategy.

---

Warp specialization made the threads of one CTA cooperate. The next step makes two CTAs cooperate on a single larger tile.


(chap_cta_cluster)=
## Step 8: 2-CTA Cluster

Step 7 overlapped the engines but each CTA still computed its own 128×128 tile in isolation, reloading operands no neighbor could share. Step 8 changes that: two CTAs join into a cluster and reach into each other's shared memory, so a single cooperative `tcgen05` MMA produces one 256×256 tile across both. Each B load now feeds twice the MMA work. M=N=K=4096.

> **What this step changes — Scope + Layout + Dispatch**
> - Scope: the cooperating scope now spans two CTAs in a cluster, not one.
> - Layout: operand tiles are split across the two CTAs' SMEM; CTA 0 owns the shared completion barrier (`remote_view`).
> - Dispatch: the MMA gains `cta_group` / `cta_mask` so `tcgen05` runs as a 2-CTA cooperative op.

**Topics.**

- CTA clusters: multiple CTAs cooperating on a larger tile

- Cross-CTA SMEM access via `map_shared_rank`

- `cta_group=2` for cooperative MMA over a 256x256 cluster tile

- Cross-CTA barrier signaling with `cta_mask`


### Cluster Tile Shape

This optimization rests on one hardware capability: with `cta_group=2`, the MMA can read operand tiles staged by *both* CTAs, not just its own. Each CTA loads one 128-row slice of stored B — which becomes 128 logical output columns after the transpose — and the cooperative MMA stitches the two slices into one operand:

![2-CTA Cluster](../img/cta_cluster.png)

**Why A and B are split across the cluster**: To see how the 256×256 tile is partitioned, recall that the tutorial stores GEMM as `D = A @ B.T`, where stored B has shape `N x K`. With 2 CTAs in a cluster:

- **A is split vertically**: CTA-0 holds A0 (rows 0-127), CTA-1 holds A1 (rows 128-255). Stacked: `[A0; A1]` (256 rows).
- **Stored B is split by rows**: CTA-0 loads B rows 0-127, CTA-1 loads B rows 128-255. Because the math uses `B.T`, those two stored row slices become two 128-column slices of the logical right-hand operand.
- With `cta_group=2`, the MMA hardware reads B from **both** CTAs' SMEM via cross-CTA shared memory access, so it sees the full logical output-column span.
- Result: the two CTAs cooperate on one 256x256 output tile. Each CTA writes a 128x256 row stripe of that tile.

This is a win, not just a reshuffle. Each CTA still loads 128×K of A and 128×K of B, so the cluster stages only about 2× a single CTA's operands — yet it produces a 256×256 tile, about 4× the output FLOPs of a 128×128 tile. The MMA therefore does roughly twice the work per staged-operand byte, because each CTA's B slice gets reused by the other CTA's A slice through the cooperative MMA. Arithmetic intensity roughly doubles, which is what helps a still memory-leaning kernel: the ~1.8× speedup in the End-to-End table comes from feeding the same bytes to more math.

### Tile Address Calculation

Because the cluster is now the unit of work, the tile scheduler also has to count in cluster tiles. Each `(m_idx, n_idx)` it hands back names a 256×256 region, and the two CTAs inside the cluster divide that region between them. Translating a cluster coordinate into the per-CTA slice each one loads looks like this:

```python
m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
```

The two CTAs work on the *same* 256×256 cluster tile, and the single coordinate `cbx` (the CTA's position in the cluster, 0 or 1) picks this CTA's contribution along both axes. `m_st` selects the output row stripe this CTA owns; `n_st` selects the stored-B slice it feeds into the cooperative MMA; the writeback later emits both 128-column halves of the 256-column output span. Note that `num_m_tiles = M // 256` and `num_n_tiles = N // 256` count cluster tiles, not individual CTA tiles.

`cbx` shows up in both `m_st` and `n_st`, as if a row offset had leaked into the column, but both uses are correct. On the writeback path `cbx` belongs to the M axis alone: each CTA owns a distinct 128-row stripe (`m_st = (m_idx * CTA_GROUP + cbx) * BLK_M`, so CTA-0 writes rows `m_idx*256 .. +128` and CTA-1 the next 128), yet both CTAs write the *full* 256 output columns of the cluster tile. That is precisely why the store derives its column from the cluster's `n_idx` — `n_st_epi = n_idx * 256 + no * 128`, independent of `cbx` — rather than from the per-CTA `n_st`. `n_st` carries `cbx` only because each CTA loads a different stored-B row slice into the MMA: it is a *load* offset, not the CTA's output-column offset.

### Code Changes from Step 7

Despite the conceptual jump, the diff against Step 7 is six edits, each encoding one piece of the cluster contract described above:

```python
# 1. Cluster launch
cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])   # cbx = CTA index within cluster (0 or 1)

# 2. Cooperative MMA (was cta_group=1)
Tx.gemm_async(..., cta_group=2)

# 3. Cross-CTA shared memory access
B_remote = T.ptx.map_shared_rank(Bsmem, cta_id=1)

# 4. Cross-CTA barrier
tma2mma_cta0 = T.decl_buffer(
    [CTA_GROUP], "uint64",
    data=T.ptx.map_shared_rank(tma2mma.ptr_to([0]), 0),
    scope="shared"
)

# 5. mma2tma / mma2ld arrives go from cta_mask=0 (single CTA, Step 7)
#    to cta_mask=3 (signal both CTAs in the cluster)
mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)

# 6. Cluster sync replaces cta_sync at the end
T.cuda.cluster_sync()
```


### Cluster-Scope Changes

- **Cluster CTA ID**: `cbx` tells each CTA its position in the cluster (0 or 1). CTA-0 handles A rows 0-127, CTA-1 handles rows 128-255.

- **Remote barrier view**: In a cluster each CTA has its own SMEM and its own barriers. If CTA-1 must wait on something CTA-0 produces, whose barrier does it touch? Designate CTA-0's barriers as the single coordination point and let any CTA reach them. `map_shared_rank(tma2mma.ptr_to([0]), 0)` returns a cluster-wide pointer to CTA-0's barrier; the TIRx wrapper is `tma2mma.remote_view(0)`. Every arrive and wait targets CTA-0's copy.

- **MMA dispatch from CTA-0 only**: `cta_group=2` does not fire two engines in parallel. CTA-0 issues exactly one `tcgen05.mma`, and the hardware drives a *single cooperative* MMA spanning both CTAs — reading operands from both SMs' SMEM and writing the accumulator across both SMs' TMEM. CTA-1 issues no MMA at all. (Each SM has only one `tcgen05` engine; `cta_group=2` is one cross-SM MMA, not two engines in parallel.) That is why the code guards the MMA with `if cbx == 0:`.

- **Multicast arrive**: `tcgen05.commit(..., cta_group=2, cta_mask=3)` is issued only by CTA-0 but signals both CTAs' barriers. `cta_mask=3` (binary `11`) means both CTA-0 and CTA-1 are targeted.

- **ld2mma init count**: `init(128 * CTA_GROUP)` --- both CTAs' writeback warpgroups (128 threads each) arrive.


**Implementation.**

```python
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
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, 128))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
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
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler (cluster tiles) ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        # --- Cross-CTA barrier view ---
        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
        # =============================================
        if wg_id == 1:
            if warp_id == 3:
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    Tx.copy_async(Asmem[tma_ps.stage, :, :],
                                  A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
                    Tx.copy_async(Bsmem[tma_ps.stage, :, :],
                                  B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
                                  dispatch="tma", cta_group=CTA_GROUP,
                                  mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id == 0:
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
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
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((128,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wb_ps.stage, wb_ps.phase)
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                for no in T.unroll(2):  # 2 chunks of 128 columns = 256 total
                    reg = T.alloc_local((128,), acc_type)
                    reg_wg = reg.view(128, 128,
                        layout=TileLayout(S[(128, 128) : (1@tid_in_wg, 1)]))
                    Tx.wg.copy_async(reg_wg[:], tmem[:, no * 128:(no + 1) * 128])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(10)
                    if warp_id == 0:
                        if lane_id == 0:
                            n_st_epi = T.meta_var(n_idx * 256 + no * 128)
                            Tx.copy_async(D[m_st:m_st+BLK_M, n_st_epi:n_st_epi+128],
                                          Dsmem[:, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(10)

                ld2mma.arrive(0, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**What changes for 2 CTAs.**

- `CTA_GROUP = 2`, `MMA_N = BLK_N * CTA_GROUP = 256`

- `ld2mma.init(128 * CTA_GROUP)` --- both CTAs' writeback WGs arrive

- TMA arrive byte count includes both CTAs: `CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`

- `tcgen05.alloc` and `tcgen05.dealloc` must use `cta_group=2`

- Writeback splits the 256 output columns into two 128-column chunks --- reading all 256 TMEM columns at once exceeds register capacity. Step 9 shrinks the chunk further to `EPI_N=64`

- `cluster_sync()` replaces `cta_sync()` at the end (ensures all CTAs are done before TMEM dealloc)

The extra arithmetic intensity shows up in the wall clock: Step 8 reaches **0.13 ms** at 4096³, about 538× over the 70 ms Step-1 algorithm at the same size (see the End-to-End table). The kernel is now leaning toward compute-bound, which sets up Step 9 — adding a second MMA consumer to keep more Tensor Core work in flight.

If Step 8 turns out *slower* than Step 7, it is almost always one of the new cluster contracts entered wrong. Check three things: (1) the TMA arrive byte count is `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`; (2) the scheduler dimensions are `num_m_tiles=M//256, num_n_tiles=N//256` for the 256×256 cluster tile; (3) writeback issues two TMA stores, one per 128-column chunk, and each drains before Dsmem is reused.

---

Clusters raised reuse *across* CTAs. The last step raises compute density *within* each CTA by giving the producer a second MMA consumer to feed.


(chap_multi_consumer)=
## Step 9: Multi-Consumer Warp Specialization

By Step 8 the MMA is busy, but one consumer warp can only work through a staged B tile so fast, and that B tile sits in SMEM the whole time, available to anyone. The final optimization adds a second MMA consumer that multiplies a *different* A block against the *same* B tile, doubling the compute density per CTA and growing the cluster output from 256×256 to 512×256. M=N=K=4096.

> **What this step changes — Scope + Layout**
> - Scope: one MMA consumer becomes two, selected by `warp_id`.
> - Layout: one staged B tile is reused by both consumers; A gains a consumer axis.
> - Dispatch: unchanged.

**Topics.**

- Multiple MMA warps (consumers) for higher throughput

- Multiple writeback warpgroups with independent barrier slots

- The structure used by the most optimized GEMM variant in this tutorial


### Multi-Consumer Structure

With `NUM_CONSUMER=2` and `WG_NUMBER=3`, there are three warpgroups (abbreviated WG in the role table):

| Warpgroup | Warp | Role |
|-----------|------|------|
| **WG 2** | warp 0 | MMA consumer 0: `Asmem[..., 0] x B` -> TMEM cols `[0:256]` |
| **WG 2** | warp 1 | MMA consumer 1: `Asmem[..., 1] x B` -> TMEM cols `[256:512]` |
| **WG 2** | warp 3 | TMA producer: loads 2x A blocks + 1x B block per stage |
| **WG 0** | all | Writeback for consumer 0: reads TMEM `[0:256]` |
| **WG 1** | all | Writeback for consumer 1: reads TMEM `[256:512]` |

The arrangement turns on one asymmetry. Each consumer multiplies its own A block against the *same* staged B tile, so one B load feeds 2× the MMA work and B's load cost per useful FLOP is halved. We share B and not A because the two consumers cover different M-row stripes — their A blocks are different data, while B is identical for both. Exercise 3 asks you to confirm this is the only sharing that works.

### Changes from Step 8

- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` --- 2 A blocks per stage, one per consumer

- TMA loads both `Asmem[stage, 0]` and `Asmem[stage, 1]`, with TMA arrive bytes now `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE` (extra A block)

- MMA warp `warp_id` selects which A block and TMEM range

- `mma2tma.init(NUM_CONSUMER)` --- both consumers signal TMA per stage

- `mma2ld` and `ld2mma` have `depth=NUM_CONSUMER` --- each consumer uses its own barrier slot (`warp_id` for MMA side, `wg_id` for writeback side)

- Tile address: `m_st = (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M` --- M direction has the extra `NUM_CONSUMER` factor because each cluster tile now spans `NUM_CONSUMER` consumers in M. Tile scheduler uses `num_m_tiles = M // 256 // NUM_CONSUMER` (cluster tile is 512x256)

- Writeback uses chunked `EPI_N` so each iteration keeps fewer TMEM-readback values live in registers


**Implementation.**

```python
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
    F16_SIZE = 2  # fp16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (PIPE_DEPTH, BLK_N, BLK_K))
    D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
                                 (NUM_CONSUMER, BLK_M, EPI_N))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx = T.cta_id([SM_COUNT])
        cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
        wg_id = T.warpgroup_id([WG_NUMBER])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        # --- Allocation ---
        pool = T.SMEMPool()
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
                T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=CTA_GROUP)
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
            (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        # --- Tile scheduler (512x256 cluster tiles) ---
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts", num_m_tiles=M // 256 // NUM_CONSUMER, num_n_tiles=N // 256,
            l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
        tile_scheduler.init(bx // CTA_GROUP)
        m_idx = T.meta_var(tile_scheduler.m_idx)
        n_idx = T.meta_var(tile_scheduler.n_idx)
        m_st = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
        n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

        tma2mma_cta0 = tma2mma.remote_view(0)

        # =============================================
        # Warpgroup 2: TMA Producer (warp 3) + 2 MMA Consumers (warp 0, 1)
        # =============================================
        if wg_id == 2:
            if warp_id == 3:
                # === TMA Producer: loads 2 A blocks + 1 B block per stage ===
                tma_ps = PipelineState(PIPE_DEPTH, phase=1)

                @T.inline
                def tma_load(k_offset):
                    m_st_c1 = T.meta_var(m_st + CTA_GROUP * BLK_M)
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

                if T.filter(lane_id, T.ptx.elect_sync()):
                    while tile_scheduler.valid():
                        for k in range(K_TILES):
                            mma2tma.wait(tma_ps.stage, tma_ps.phase)
                            tma_load(k * BLK_K)
                            if cbx == 0:
                                tma2mma_cta0.arrive(tma_ps.stage,
                                    CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
                            tma_ps.advance()
                        tile_scheduler.next_tile()

            elif warp_id < NUM_CONSUMER:
                # === MMA Consumer: warp_id selects A block and TMEM range ===
                mma_ps = PipelineState(PIPE_DEPTH, phase=0)
                ld_ps = PipelineState(1, phase=1)

                if cbx == 0:
                    if T.filter(lane_id, T.ptx.elect_sync()):
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
            wb_ps = PipelineState(1, phase=0)
            reg_f16 = T.alloc_local((EPI_N,), d_type)

            while tile_scheduler.valid():
                mma2ld.wait(wg_id, wb_ps.phase)  # wait for THIS consumer
                wb_ps.advance()
                T.ptx.tcgen05.fence.after_thread_sync()

                # Read TMEM in EPI_N=64 column chunks (4 iterations for 256 cols)
                for i in T.unroll(MMA_N // EPI_N):
                    reg = T.alloc_local((EPI_N,), acc_type)
                    reg_wg = reg.view(128, EPI_N,
                        layout=TileLayout(S[(128, EPI_N) : (1@tid_in_wg, 1)]))
                    col_st = T.meta_var(wg_id * MMA_N + i * EPI_N)
                    col_end = T.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
                    Tx.wg.copy_async(reg_wg[:], tmem[:, col_st:col_end])
                    T.ptx.tcgen05.wait.ld()
                    Tx.cast(reg_f16[:], reg[:])
                    Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], reg_f16[:])
                    T.ptx.fence.proxy_async("shared::cta")
                    T.cuda.warpgroup_sync(wg_id + 10)
                    if warp_id == 0:
                        if lane_id == 0:
                            m_st_epi = T.meta_var(
                                (m_idx * NUM_CONSUMER * CTA_GROUP + wg_id * CTA_GROUP + cbx) * BLK_M)
                            n_st_epi = T.meta_var(n_idx * MMA_N + i * EPI_N)
                            Tx.copy_async(
                                D[m_st_epi:m_st_epi+BLK_M, n_st_epi:n_st_epi+EPI_N],
                                Dsmem[wg_id, :, :], dispatch="tma")
                            T.ptx.cp_async.bulk.commit_group()
                            T.ptx.cp_async.bulk.wait_group(0)
                    T.cuda.warpgroup_sync(wg_id + 10)

                ld2mma.arrive(wg_id, cta_id=0, pred=True)
                tile_scheduler.next_tile()

        # --- Cleanup ---
        T.cuda.cluster_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

    return kernel
```

**Implementation notes.**

- In this Step 9 design, `mma2ld` and `ld2mma` are one shared object each with `depth=NUM_CONSUMER`, not separate per-consumer objects. Slot 0 connects MMA warp 0 to Warpgroup 0; Slot 1 connects MMA warp 1 to Warpgroup 1. MMA side uses `warp_id` as index, writeback side uses `wg_id`.

- This is the most optimized GEMM structure shown in the tutorial


(chap_warp_spec_debug)=
## Debugging Warp-Specialized Kernels

Steps 7, 8, and 9 fail in the same handful of ways, for the same reason: TMA, MMA, and writeback run concurrently, so one wrong barrier can deadlock the kernel, crash the CUDA context, or corrupt the output. The failures are stereotyped — once you can recognize the *shape* of a bug, the fix is usually a one-liner. Keep this checklist close; these bugs repeat.

### Inspecting Generated Code

When a warp-specialized kernel misbehaves, look first at what the compiler emitted — whether the role guards and barrier inits landed where you expect. For any compiled kernel:

```python
cuda_source = ex.mod.imports[0].inspect_source("cuda")
print(cuda_source)
```

The generated code maps TIRx constructs to CUDA like this:

| TIRx | Generated CUDA |
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

- Barrier inits sit at **top level** (not inside a `wg_id` guard) — see *Deadlocks*.
- `tcgen05_alloc`/`dealloc` have a **warp guard but no lane guard** — all 32 lanes participate.
- TMA and MMA loops both iterate `K_TILES` times.
- Phase init: producer=`1`, consumer=`0`.

### Symptom Map

The symptom narrows the search to one section below before you reach for any tool:

| Symptom | Failure class | Where to look |
|---|---|---|
| Kernel hangs ~30 s, then "unspecified launch failure" | Deadlock | *Deadlocks* |
| Crash within ms; subsequent `torch.randn` also fails | XID 43 / illegal memory access | *Crashes* |
| Output max-err in multiples of 128 (128 / 256 / 384 rows) | Sync race | *Wrong results* |
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

- **`cta_sync()` inside a warpgroup branch.** `cta_sync` is `__syncthreads()` — requires *all* CTA threads. Inside `if wg_id == 0:`, WG1 never reaches it. Use `T.cuda.warpgroup_sync(10)` instead.

- **`tile_scheduler.next_tile()` not called by all threads in the consumer warpgroup.** The scheduler tracks per-thread state; threads that skip it loop forever.

- **TMA and MMA disagree on K-tile count.** If MMA does `K_TILES - 1` instead of `K_TILES`, barrier phases drift and the second outer tile deadlocks.

- **`PipelineState` initial phase wrong.** Producer must start at `phase=1` (first wait passes); consumer at `phase=0` (first wait blocks). Same starting phase ⇒ instant deadlock.

### Crashes (XID 43 / illegal memory access)

These corrupt the CUDA context, so the giveaway is that a *later* innocent `torch.randn` fails too. All three causes are allocation- or warp-shape mistakes:

- **`pool.alloc` after `pool.commit()`.** Barrier wrappers call `alloc` internally. Correct order: `tmem_addr → barrier wrappers → move_base_to(1024) → Asmem / Bsmem / Dsmem → commit()`.
- **`tcgen05.alloc` / `dealloc` with a lane guard.** They require all 32 lanes of the warp to participate; `if lane_id == 0:` runs one thread, which is undefined behavior — often observed as an illegal-instruction or context error, a hang, or (worst) silently wrong results.
- **Missing `cta_sync()` before `tcgen05.dealloc`** — TMEM is freed while writeback is still reading.

### Wrong results

The tell here is in the error pattern: mismatch counts that are exact multiples of 128 (128 / 256 / 384 rows) mean a sync race, not arithmetic — whole warpgroup-sized stripes are wrong because a handoff slipped, not because a number was computed incorrectly.

- **`tcgen05.commit` outside `elect_sync`** — all 32 threads create commit groups; the 31 empty ones signal the mbarrier immediately. TMA overwrites SMEM before MMA reads it. Output is zeros or garbage.
- **Missing `fence.proxy_async("shared::cta")` before TMA store** — TMA engine doesn't see SMEM writes from threads.
- **Missing `cp_async.bulk.commit_group()` + `wait_group(0)` after TMA store** — next tile reuses Dsmem before the store drains.
- **Persistent kernel, intermittent fails at small sizes (e.g., 1024×1024)** — not GPU flakiness. Larger sizes mask the race with longer K-loops. Re-check phase reset between tiles and the TMA-store commit/wait.
- **`fence.after_thread_sync()` is usually not the cause.** The MMA-completion mbarrier already carries release→acquire semantics. Steps 8 and 9 here add it conservatively on the writeback edge (after `mma2ld.wait`, before the first `tcgen05.ld`), but it is not a general MMA-boundary fix. Do not add it routinely on the TMA→MMA edge; if your output is wrong, check the barrier and store-drain bullets first.


## End-to-End Result

Reference numbers on NVIDIA B200, M=N=K=4096, fp16, locked clocks, 1000-iteration timed benchmark:

| Step | Technique | Time | Speedup |
|------|-----------|------|---------|
| 1 | Sync load + MMA | 70 ms | 1× |
| 2 | K-loop accumulation | --- | Handle K larger than one tile |
| 3 | Spatial tiling | --- | Handle multiple M and N tiles |
| 4 | TMA async load | 0.50 ms | ~140× |
| 5 | Software pipeline | --- | Overlap load + compute |
| 6 | Persistent kernel | --- | L2 cache locality |
| 7 | Warp specialization | 0.23 ms | ~304× |
| 8 | 2-CTA cluster | 0.13 ms | ~538× |
| 9 | Multi-consumer | 0.12 ms | ~583× |
| --- | cuBLAS (reference) | 0.11 ms | ~636× |

All times in this table — including the 70 ms Step 1 baseline — are measured at the same M=N=K=4096 size, so the speedup chain is comparable. To be precise about what the 70 ms is: it is *not* the single-tile Step-1 kernel shown in {ref}`chap_gemm_basics` run at 4096³. That shown kernel computes one 128×128 tile and runs at small sizes only. The 70 ms is a naive full-size baseline that implements the same sequential, single-tile approach scaled to the full 4096³ problem. Steps 1–3 are introduced in {ref}`chap_gemm_basics` at small sizes (128×128 and 256³) to keep the first walkthroughs simple; the dashes (Steps 2, 3, 5, 6) mark steps shown for structure but not separately timed.

Treat these as one B200 reference run under controlled conditions, not a leaderboard entry. The `{.python .input}` benchmark cells embedded in each step are smoke benchmarks meant for checking trends, not for claiming peak performance.

Four techniques account for almost all of the gain:

1. **TMA Async Data Movement** — hardware copy engine replaces software copy (~140× from Step 1 → Step 4). This 140× reflects going from a single 128×128-tile kernel (grid 1×1) to a full tiled-and-parallel kernel with a K-loop, spatial tiling, and many CTAs *together with* TMA, not TMA's isolated contribution; isolating TMA would require comparing two full-size kernels that differ only in the copy mechanism.
2. **Software Pipelining + Warp Specialization** — overlap load and compute with dedicated roles (~2.2× from Step 4 → Step 7).
3. **CTA Clusters** — 2-SM cooperative MMA improves B-tile reuse across CTAs (~1.8× from Step 7 → Step 8 in this benchmark).
4. **Multi-Consumer** — two MMA warps for higher compute density (~8% from Step 8 → Step 9).

![GEMM Optimization Journey](../img/gemm_perf.png)

In this benchmark the path moves from 70 ms to 0.12 ms, close to the cuBLAS reference above.

The gains shrink down the list for a structural reason. The early steps attack *memory* bottlenecks — TMA replaces software copies, clusters raise arithmetic intensity — and that is where most of the 70 ms lived, so they pay off the most. By Step 8 the kernel is already within ~18% of cuBLAS (0.13 vs 0.11 ms), close to *compute-bound*, so there is little memory stall left to hide; Step 9's multi-consumer overlap recovers most of what remains. A single-digit final gain is what to expect near the compute ceiling, the diminishing returns of a problem nearly solved rather than a weak optimization.

Everything we built here — TMA loads, `tcgen05` MMA, TMEM readback, and warp-specialized barriers — carries straight into the next chapter. Flash Attention reuses all of it, but raises the difficulty by wedging an online-softmax step between two MMA phases instead of repeating a single one.


## Exercises

1. What happens if you set the initial `phase` to `0` for both the TMA and MMA `PipelineState` in Step 7? Draw the deadlock scenario.
2. With `cta_group=2` in Step 8, the TMA arrive byte count is `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`. Why multiply by `CTA_GROUP` when each CTA loads its own data?
3. In Step 9, each consumer handles different M rows but the same B tile. Why is sharing B (not A) the right choice?

**Try with your agent**: Paste the Step 7 kernel and ask it to trace one K-tile through the four barriers (`tma2mma`, `mma2tma`, `mma2ld`, `ld2mma`). For each, ask who waits, who arrives, what tile becomes safe to read, and which buffer becomes reusable afterward.
