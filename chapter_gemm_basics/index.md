(chap_gemm_basics)=
# Building a Tiled GEMM

:::{admonition} Overview
:class: overview

- Builds a correct tiled GEMM from the TIRx tile primitives, starting from a single output tile.
- Step 1 is a single-tile GEMM, Step 2 adds the K-loop accumulation, Step 3 tiles spatially across CTAs for full matrices.
- Correctness comes first; performance is the job of the next two chapters.
:::

So far we have the TIRx tile primitives in the abstract — the scope / layout / dispatch model from {ref}`chap_tirx_primer` and {ref}`chap_data_layouts`. Now we put them to work on a real kernel. We start small, with a single 128 x 128 output tile, and grow it into a kernel that handles full-size matrices. Getting there takes two additions on top of the first tile: accumulating partial products along the K dimension, and tiling the output spatially across many CTAs.

This is the first of three chapters that walk one GEMM optimization path end to end. Here we build a correct tiled kernel. The next chapter ({ref}`chap_gemm_async`) replaces thread copies with TMA and overlaps movement with compute through pipelining, and {ref}`chap_gemm_advanced` adds warp specialization and CTA clusters. Each chapter assumes the previous one, so the kernels accumulate features rather than restart.

A useful way to read each step is as an edit to a single contract: which **scope** runs the operation, which **layout** the operand tiles use, which **dispatch** path executes it. Almost every optimization touches exactly one of these three pillars and leaves the other two alone, which is why a small card at the start of each step names the pillar that changes. Step 1 sets the baseline that the rest of the path edits.

## GEMM

GEMM is the dense matrix multiply that sits underneath linear layers, attention projections, and many convolution implementations, so a fast GEMM kernel pays off almost everywhere. The examples in this tutorial use $D = A B^{\top}$:

- $A$ has shape $M \times K$.
- $B$ has shape $N \times K$.
- $D$ has shape $M \times N$.
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$.

The transpose appears not as an extra operation but as a consequence of storage: the examples keep $B$ as $N$ rows of length $K$, which is how linear-layer weights are usually laid out, so contracting along $K$ naturally reads $B^{\top}$.

Throughout, we measure a kernel by its throughput in TFLOPS, which counts the two floating-point operations per multiply-add against the wall-clock time:

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM Data Path

Before writing any code, it helps to see where the data lives and how it moves, because every optimization in this tutorial is ultimately about that movement. A Blackwell GEMM kernel is organized around two activities — moving tiles between memories and computing on them:

![*Memory Data Flow*](../img/memory_dataflow.png)

Reading the figure left to right: operand tiles move from GMEM to SMEM, `tcgen05.mma` consumes the SMEM operands and writes accumulators to TMEM, and the epilogue reads TMEM back into registers before storing the final result to GMEM. Keep this path in mind — every step below changes *how* one of these hops happens, never the hops themselves.

## Optimization Path

The plain data path above is enough to get a correct answer, but it leaves most of the hardware idle. The rest of the tutorial closes that gap by reaching for Blackwell features one at a time, each expressed through a TIRx tile primitive:

- **TMA async movement:** move GMEM <-> SMEM tiles through Blackwell's hardware copy path, with barriers tracking completion.
- **Software pipelining:** use multiple SMEM stages so data movement for the next K tile can overlap Tensor Core compute on the current tile.
- **Persistent scheduling:** a fixed pool of CTAs each processes many output tiles via a tile scheduler, instead of one CTA per tile.
- **Warp specialization:** split producer, MMA consumer, and writeback roles across warpgroups.
- **CTA clusters:** let two CTAs cooperate on a larger Blackwell MMA tile.
- **Multi-consumer execution:** use multiple consumer warpgroups to compute different parts of the tile, increasing compute density.

---

(chap_single_tile)=
## Step 1: Sequential Single-Tile GEMM

The simplest GEMM that exercises the full hardware path is a single output tile, so that is where we begin. Step 1 computes one 128 x 128 output tile with K = 64 — small enough that nothing loops and every piece of the data path appears exactly once.

> **What this step establishes — the baseline**
> - Scope: one warpgroup (128 threads) walks the whole path in order.
> - Layout: A/B tiles in SMEM, accumulator in TMEM, result staged through registers.
> - Dispatch: sync `Tx.copy` for the loads, `tcgen05` for the MMA.

### Single-Tile Dataflow

Concretely, the first kernel walks the core GEMM data path exactly once, allocating its working memory, loading the operands, computing, writing back, and cleaning up:

1. **Allocate**: SMEM (pool allocator), TMEM (`tcgen05.alloc`), mbarrier
2. **Load**: All 128 threads cooperatively copy A and B tiles from GMEM to SMEM (sync `Tx.copy`)
3. **Compute**: Single elected thread issues `Tx.gemm_async` + `tcgen05.commit`; all threads wait on mbarrier
4. **Writeback**: Warpgroup reads TMEM → registers; each thread casts fp32→fp16 and writes to GMEM
5. **Deallocate**: TMEM deallocation

### Four Pieces of the First Kernel

The full kernel is a few dozen lines, which is a lot to absorb at once, so we read it in four pieces first — memory allocation, the synchronous load, the MMA dispatch, and the writeback — and only then assemble them. The API names here are the TIRx tile-primitive vocabulary from Part II ({ref}`chap_tirx_primer`, {ref}`chap_data_layouts`).

**Memory allocation.** Everything downstream needs a place to live, so the kernel first carves out shared memory for the operands, a slot for the TMEM address, and an mbarrier:

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 bytes)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 bytes)
pool.move_base_to(1024)                           # Skip to offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

Two details are worth pausing on. The `pool.move_base_to(1024)` pushes Asmem/Bsmem to offset 1024, reserving the low addresses for the small metadata above them so the bulky operand tiles sit on a clean boundary. And `layout=A_layout` is not cosmetic: it asks `tma_shared_layout` for a swizzled SMEM placement that both TMA and `tcgen05.mma` can read directly, which is exactly the kind of layout-as-contract obligation Part II described.

**Synchronous load.** With the buffers in place, the operands have to reach SMEM. In this first version we let the CTA's own threads do the copying:

```python
Tx.cta.copy(Asmem[:, :], A[:, :])
Tx.cta.copy(Bsmem[:, :], B[:, :])
T.cuda.cta_sync()
```

Because there is only one tile (M=N=128, K=64), copying the entire A and B is the whole load. `Tx.cta.copy(...)` makes the CTA cooperate on the copy, with each thread responsible for a slice of the data. The `T.cuda.cta_sync()` that follows is not optional bookkeeping: it both waits for every thread to finish and publishes their shared-memory writes, so that when the MMA later reads `Asmem` and `Bsmem` it sees complete tiles rather than a half-filled buffer. This thread-driven copy is the obvious way to move data and also the first thing we will replace — the next chapter ({ref}`chap_gemm_async`) swaps it for TMA.

**MMA dispatch.** The operands are now in SMEM, so we can multiply them. The MMA is issued from a single elected thread:

```python
if warp_id == 0:
    if T.ptx.elect_sync():
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

The two nested guards narrow the issuer down in two steps. The outer `if warp_id == 0` keeps only warp 0 of the warpgroup; the inner `if T.ptx.elect_sync():` then elects a single active lane within that warp. Together they leave exactly one thread to run `Tx.gemm_async` and `tcgen05.commit`.

That single thread does *not* mean a single-threaded multiply. The computation is still a full tile-level MMA — the hardware performs the cooperative multiply for the tile described by the SMEM operand layouts and the TMEM accumulator layout. The reason only one thread issues it is that `tcgen05.mma` is a *single-instruction* cooperative op: one launch drives the entire tile MMA, so if all 128 threads issued it, the same tile would be launched 128 times. Finally, `accum=False` tells the MMA to overwrite the TMEM destination rather than add into it, which is correct here because there is no prior partial sum to extend.

**Writeback.** The product now sits in TMEM, but the caller wants it in GMEM as fp16, so the epilogue has to bring it down through registers and cast it on the way:

```python
Dreg = T.alloc_local((BLK_N,), acc_type)        # per-thread fp32 register row
Dreg_f16 = T.alloc_local((BLK_N,), d_type)      # same row, cast to fp16
Dreg_wg = Dreg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

Why this dance? The MMA leaves a 128 x 128 fp32 accumulator tile in TMEM. It is fp32 on purpose — GEMM sums many products along K, and a higher-precision running sum holds down rounding error. But `D` is fp16, so the values cannot go straight out; they first land in registers, get narrowed to fp16, and only then reach GMEM.

The two register buffers play distinct roles. `Dreg` is a per-thread buffer of `BLK_N` elements, and `Dreg_wg` is a warpgroup-wide *view* of those same registers under a chosen layout:

```python
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

This layout maps the tile's first dimension onto warpgroup threads: thread 0 owns row 0, thread 1 owns row 1, and so on through row 127. The second dimension stays inside each thread's own register buffer, so a thread holds all the columns of its single row. With 128 threads in a warpgroup and 128 rows in the tile, the 128 x 128 output is split one row per thread — a clean, contention-free partition.

Reading the accumulator out under that view is what `Tx.wg.copy_async(Dreg_wg, tmem)` does; it lowers to the Blackwell TMEM load path (`tcgen05.ld`). Because that load is asynchronous, `T.ptx.tcgen05.wait.ld()` must complete before any thread touches `Dreg`, otherwise it would read registers the load has not filled yet.

Once the wait returns, each thread's private `Dreg[:]` holds the fp32 values for its one logical output row. The thread narrows them to fp16 in `Dreg_f16`, works out which global row it is responsible for,

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
```

and writes `D[m_thr, n_st:n_st + BLK_N]`. Warp 0 writes rows 0-31, warp 1 writes rows 32-63, warp 2 writes rows 64-95, and warp 3 writes rows 96-127.

### Complete Kernel

With the four pieces understood, here they are stitched together into one runnable kernel (M=N=128, K=64). The imports come first:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

The kernel is wrapped in the same `hgemm_vX(M, N, K)` style used by the later steps. Step 1 still runs with `M=N=128, K=64`, so the launch contains exactly one output tile:

```python
def hgemm_v1(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    # MMA_M/MMA_N/MMA_K document the underlying hardware MMA tile; they are not
    # passed to gemm_async (which derives the MMA shape from the operand and
    # accumulator tiles), so the later steps omit them.
    MMA_M, MMA_N, MMA_K = 128, 128, 16

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # Step 1 is a single-tile kernel: M = BLK_M and N = BLK_N, so the grid
        # is 1x1. Starting with a 1x1 grid keeps the per-CTA tile offsets
        # (m_st, n_st) trivially zero; Steps 3+ generalise this to larger M / N.
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])      # single warpgroup, so wg_id is always 0 (unused below)
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
    
        # --- SMEM allocation ---
        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()
    
        # --- Barrier + TMEM init (warp 0 only) ---
        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
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
        phase_mma: T.int32 = 0
    
        # --- Load: all threads copy global -> shared (synchronous).
        # With M=BLK_M and N=BLK_N the slices below cover the full matrices;
        # the slice form is kept so the diff to Step 3 (multi-tile) is minimal.
        Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
        Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        T.cuda.cta_sync()
    
        # --- Compute: single elected thread issues MMA ---
        if warp_id == 0:
            if T.ptx.elect_sync():
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
    
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
    
        # --- Writeback: TMEM -> RF -> GMEM ---
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()
        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
    
        # --- Deallocate TMEM ---
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

Every GEMM step that follows compiles, runs, and checks itself the same way, so we spell that scaffolding out once here and from then on show only the kernel. To run a later step, drop in its `hgemm_vX` and the matching problem size in place of the ones below. One caveat: compile a single step per fresh Python session and restart before trying another, because the examples reuse inner names and the compiler holds per-session state.

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device('cuda')  # gpu(0)

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

# ex.mod(...) takes torch tensors directly — the same call form used in every chapter.
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
# Relative tolerance, like the warp-specialization and Flash Attention cells:
# output magnitude grows with K, so a fixed absolute bound would fail at larger K.
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")

# Optional timing for larger kernels.
ITERS = 10
for _ in range(3):
    ex.mod(A_tensor, B_tensor, D_tensor)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
    ex.mod(A_tensor, B_tensor, D_tensor)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

Steps 1–3 run at small sizes (128×128 here, 256³ in Step 3) to keep the first walkthroughs simple. The cross-step *End-to-End Result* table at the end of {ref}`chap_gemm_advanced` instead measures every step — including this Step 1 algorithm — at a single M=N=K=4096 size, so its speedup ratios are comparable.

### Limits of the Single-Tile Kernel

This kernel is correct, and that is the point of Step 1 — but it is correct only in a very narrow world. Four limitations are baked in on purpose, and the rest of the path exists to lift them one by one:

- It handles only one K tile.
- It handles only one output tile.
- It uses synchronous GMEM -> SMEM copies instead of TMA.
- It does not overlap data movement and compute.

---

(chap_k_loop)=
## Step 2: K-Loop Accumulation

The first limit to remove is the smallest: Step 1 handles only a single 64-wide K tile, but real matrices contract over far more. Step 2 keeps the one output tile and lets K span many 64-wide chunks. The idea is simple — repeat the load -> MMA -> wait sequence once per chunk and let each MMA accumulate into the same TMEM slot. The catch is in the synchronization. Reusing one mbarrier across iterations introduces this chapter's first real correctness hazard: if the code tracks the wrong phase, a wait can return *before* its MMA has finished and silently corrupt the result. The mechanics below show exactly how that goes wrong, and how to avoid it.

> **What this step changes — Layout**
> - Scope: unchanged — one warpgroup.
> - Layout: the same SMEM tile pair and TMEM accumulator slot are reused across the K-loop — no new storage, the operand tiles just stream through one fixed pair of buffers.
> - Dispatch: unchanged.

### K-Loop Mechanics

To cover K > 64, we walk K in chunks of `BLK_K=64`. Each iteration loads the next A and B K-slice into SMEM and issues `Tx.gemm_async`. The `accum` flag is what stitches the chunks into one dot product: on the first chunk `accum=False` initializes the TMEM accumulator, and on every later chunk `accum=True` adds that chunk's product into the running sum already in TMEM.

Now the synchronization. We reuse a single mbarrier for every MMA completion, and reusing it safely means tracking which barrier phase we are waiting on. An mbarrier carries a 1-bit phase, 0 or 1, and it flips to the other value each time the expected arrival lands. The subtlety is in the wait condition: `try_wait(bar, phase)` blocks until the barrier's internal phase *differs* from the `phase` argument, so the argument has to name the phase we expect to leave behind, not the one we are waiting to reach:

| K iteration | Local `phase_mma` before wait | What `try_wait` waits for | Local update after wait |
|---|---:|---|---:|
| 0 | 0 | barrier flips to 1 | `phase_mma = 1` |
| 1 | 1 | barrier flips to 0 | `phase_mma = 0` |
| 2 | 0 | barrier flips to 1 | `phase_mma = 1` |

The single line `phase_mma ^= 1` is what keeps the table honest. Drop it, and the second iteration still calls `try_wait(bar, 0)` — but the barrier already flipped to phase 1 after the first MMA, so the wait sees a mismatch and returns immediately, before the second MMA has even been issued. The kernel reads a half-computed accumulator and reports a wrong answer with no error. This is the kind of bug that compiles, runs, and lies, which is why the phase flip is worth this much attention.

### Complete Kernel

The full kernel below is Step 1 with the K-loop and phase flip folded in. Same imports:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

It is wrapped in `hgemm_v2(M, N, K)`. The grid is still `[1, 1]`, since we are still computing a single output tile — only its K extent has grown:

```python
def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])  # still one output tile (M=N=128)
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # === K-loop: iterate over K in chunks of BLK_K ===
        for i in T.serial(K_TILES):   # serial device loop (keeps the full-K A/B parameters correctly shaped)
            # Load the i-th K chunk
            Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            # MMA: accum=False for first tile, True for rest
            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            # Wait for MMA, then flip phase
            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # === Writeback (same as Step 1) ===
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

---

(chap_spatial_tiling)=
## Step 3: Spatial Tiling (Multi-CTA)

The K-loop fixed the contraction dimension, but M and N are still pinned to a single 128 x 128 tile. A real output is far larger than one tile, so the last piece of the basic kernel is to cover M and N with many tiles at once. Step 3 launches a 2D grid of CTAs, one per output tile, so the GPU computes all the tiles in parallel. The example uses M=N=K=256, which is a 2x2 grid of tiles — just enough to make the indexing non-trivial.

> **What this step changes — Scope**
> - Scope: a 2D grid of CTAs, each CTA owns one 128 x 128 output tile.
> - Layout: unchanged — per CTA, the same SMEM/TMEM/register path as Step 2.
> - Dispatch: unchanged.

### Grid Mapping

The grid shape falls out of the tiling directly: with one CTA per 128 x 128 output tile, we need `[M // BLK_M, N // BLK_N]` CTAs. The only new work compared to Step 2 is teaching each CTA which slice of the matrices is *its* slice.

CTA `(bx, by)` owns this output region:

```text
D[bx * BLK_M : (bx + 1) * BLK_M,
  by * BLK_N : (by + 1) * BLK_N]
```

and to produce it, inside the K-loop that CTA repeatedly loads the matching K-slices of its own row band of A and column band of B:

```text
A[bx * BLK_M : (bx + 1) * BLK_M, k : k + BLK_K]
B[by * BLK_N : (by + 1) * BLK_N, k : k + BLK_K]
```

The indexing follows straight from the `D = A @ B.T` convention: `bx` selects rows of A and D, while `by` selects rows of B, which become the columns of D after the transpose.

One tile per CTA is the simplest possible mapping, and it works, but it is wasteful: every CTA in a row reloads the same A tiles from GMEM, and every CTA in a column reloads the same B tiles. Nothing exploits that the neighbors next door already pulled the data in. We leave the waste here for now — persistent scheduling (Step 6 in {ref}`chap_gemm_async`) comes back to it and keeps those shared operands hot in L2.

**Try with your agent**: With `M=N=K=256`, `BLK_M=BLK_N=128`, and `BLK_K=64`, ask it to trace CTA `(1, 0)` and CTA `(0, 1)`. For each CTA, list `m_st`, `n_st`, the A and B slices loaded for each K iteration, and the D region written. Which B rows become D columns because the kernel computes `D = A @ B.T`?

### Complete Kernel

The kernel is again Step 2 with two changes, the grid shape and the per-CTA offsets; the inner K-loop and writeback are untouched. Same imports:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

The grid becomes `[M // BLK_M, N // BLK_N]` rather than `[1, 1]`, and the loads and stores are now offset by the CTA's `m_st` and `n_st`:

```python
def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),
        B: T.Buffer((N, K), b_type),
        D: T.Buffer((M, N), d_type),
    ):
        T.device_entry()
        # 2D grid: one CTA per 128x128 output tile
        bx, by = T.cta_id([M // BLK_M, N // BLK_N])
        wg_id = T.warpgroup_id([1])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])

        pool = T.SMEMPool()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        if warp_id == 0:
            if lane_id == 0:
                T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()
        T.cuda.cta_sync()

        tmem = T.decl_buffer(
        (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
        layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

        phase_mma: T.int32 = 0

        # Per-CTA tile offsets
        m_st = T.meta_var(bx * BLK_M)
        n_st = T.meta_var(by * BLK_N)

        # K-loop with offset A and B slices
        for i in T.serial(K_TILES):   # serial device loop (keeps the full-K A/B parameters correctly shaped)
            Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])
            Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])

            T.cuda.cta_sync()

            if warp_id == 0:
                if T.ptx.elect_sync():
                    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                                  accum=(i != 0), dispatch="tcgen05", cta_group=1)
                    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

            T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
            phase_mma ^= 1

        # Writeback to the correct output tile
        Dreg = T.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = T.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

        Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        Tx.cast(Dreg_f16[:], Dreg[:])
        m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
        Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])

        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

## Exercises

1. In Steps 1-3, `Tx.copy` moves A and B tiles into SMEM before MMA. Why does the kernel need `T.cuda.cta_sync()` before `Tx.gemm_async` reads those SMEM tiles?
2. In Step 2, what happens if `phase_mma ^= 1` is removed from the K-loop? Does the kernel wait for every MMA, or can a later wait pass too early?
3. For M=N=4096 with BLK_M=BLK_N=128, how many CTAs are launched in Step 3? Which operand tiles are logically reused across neighboring CTAs, and does Step 3 exploit that reuse?
