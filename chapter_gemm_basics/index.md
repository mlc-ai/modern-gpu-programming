# GEMM Steps 1--3: From Single Tile to Spatial Tiling
:label:`chap_gemm_basics`

*This is the start of Part III: GEMM Deep Dive. Unlike Part II's standalone operators, the next 3 chapters (9 steps) tell a single continuous story --- one GEMM kernel, progressively optimized from 714ms to 0.9ms. Each step builds on the previous one.*

## What is GEMM

GEMM (General Matrix Multiplication) is the fundamental operation: $D = A \times B^T$ where:

- $A$ is $M \times K$ (M rows of K-dimensional vectors)

- $B$ is $N \times K$ (N rows of K-dimensional vectors)

- $D$ is $M \times N$ (output)

- Operation: $D[m,n] = \sum_k A[m,k] \times B[n,k]$ (i.e., $D = A \times B^T$ since B is stored as N rows of K columns)

GEMM is the workhorse of deep learning --- every linear layer, attention computation, and convolution reduces to GEMM. Optimizing GEMM is critical for both training and inference performance.

GEMM performance is measured in TFLOPS (Tera Floating-Point Operations Per Second):

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

## The Optimization Journey

Over the next 3 chapters (9 steps), we will progressively optimize a GEMM kernel on Blackwell GPUs until it matches cuBLAS. The 4 key techniques are:
- **Async Data Movement** --- let hardware (TMA) do the copying, freeing 127 threads
- **Software Pipelining** --- overlap load and compute with multi-buffering
- **Warp Specialization** --- dedicated warp roles for maximum parallelism
- **CTA Clusters** --- 2SM cooperative MMA for higher arithmetic intensity

If you're short on time, Steps 1, 4, 7, and 9 are the most important.

---

## Step 1: Synchronous GEMM
:label:`chap_single_tile`

In this section, we build the simplest possible working GEMM kernel for a single tile (M=N=128, K=64). All operations are synchronous.

### What You Will Learn

- Synchronous data loading: all 128 threads copy from GMEM → SMEM

- tcgen05 MMA invocation: single thread issues MMA, all threads wait

- TMEM allocation/deallocation and writeback (TMEM → RF → GMEM)

### Overview

The kernel follows the standard Blackwell data flow (see :numref:`chap_background`). Each step happens **sequentially** --- the next step waits for the previous to complete.

1. **Allocate**: SMEM (pool allocator), TMEM (`tcgen05.alloc`), mbarrier
2. **Load**: All 128 threads cooperatively copy A and B tiles from GMEM to SMEM (sync `Tx.copy`)
3. **Compute**: Single elected thread issues `Tx.gemm_async` + `tcgen05.commit`; all threads wait on mbarrier
4. **Writeback**: Warpgroup reads TMEM → registers; each thread casts fp32→fp16 and writes to GMEM
5. **Deallocate**: TMEM deallocation

### Key Concepts

Before looking at the full kernel, let's understand each part.

#### Memory Allocation

```python
pool = Tx.PoolAllocator()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 bytes)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 bytes)
pool.move_base_to(1024)                           # Skip to offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

The `pool.move_base_to(1024)` ensures Asmem/Bsmem start at offset 1024, leaving room for metadata. The `layout=A_layout` uses `tma_shared_layout` to create a swizzled layout for bank-conflict-free access.

#### Synchronous Load

```python
with Tx.cta():  # All 128 threads cooperate
    Tx.copy(Asmem[:, :], A[:, :])
    Tx.copy(Bsmem[:, :], B[:, :])
Tx.cuda.cta_sync()                           # Wait for all threads
Tx.ptx.tcgen05.fence.after_thread_sync()     # Make SMEM visible to MMA HW
```

Step 1 only has one tile (M=N=128, K=64), so we copy the entire A and B. All 128 threads cooperatively copy data. In Step 4, we'll replace this with TMA where a single thread issues the copy.

#### MMA Dispatch

```python
if warp_id == 0:
    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

`elect_sync()` selects one thread per warp to issue the MMA — this is the hardware-correct pattern. `accum=False` means overwrite TMEM (not accumulate).

#### Writeback

```python
Dreg_wg = Dreg.view(128, BLK_N, layout=TileLayout(([128, BLK_N], [1@tid_in_wg, 1])))
with Tx.warpgroup():          # All 128 threads cooperate on TMEM read
    Tx.copy(Dreg_wg[:, :], tmem[:, :BLK_N])
with Tx.thread():             # Each thread writes its row
    Tx.cast(Dreg_f16[:], Dreg[:])     # fp32 → fp16
    m_thr = Tx.meta_var(m_st + warp_id * 32 + lane_id)
    Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

Thread row mapping: warp 0 handles rows 0-31, warp 1 handles rows 32-63, etc. Each thread's row is `m_st + warp_id * 32 + lane_id`.

### Complete Implementation

With the above walkthrough in mind, here is the complete runnable kernel (M=N=128, K=64):

```{.python .input}

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.op_dispatch.cuda.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

Constants and swizzled layouts for the A and B tiles:

```{.python .input}
a_type = tvm.DataType("float16")
b_type = tvm.DataType("float16")
d_type = tvm.DataType("float16")
acc_type = tvm.DataType("float32")
M, N, K = 128, 128, 64

BLK_M, BLK_N, BLK_K = 128, 128, 64
MMA_M, MMA_N, MMA_K = 128, 128, 16

A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
```

### Thread hierarchy in GEMM kernels

GEMM kernels use the full thread hierarchy from :numref:`chap_background` (CTA → Warpgroup → Warp → Thread). Unlike the earlier elementwise kernels that only needed `cta_id` + `thread_id`, GEMM requires all three levels because tcgen05 MMA operates at the **warpgroup** level — all 128 threads must cooperate to read TMEM results:

- **`wg_id = Tx.warpgroup_id([1])`** — Which warpgroup (Step 1 uses only 1). Later steps use 2 warpgroups for warp specialization.
- **`warp_id = Tx.warp_id([4])`** — Which warp within the warpgroup (0-3).
- **`lane_id = Tx.thread_id([32])`** — Which thread within the warp (0-31).

The `warp_id` and `lane_id` are used for thread-level operations like data loading and writeback (each thread handles a different row: `row = warp_id * 32 + lane_id`).

The kernel itself:

```{.python .input}
@Tx.prim_func(tirx=True)
def hgemm_ver1(
    A: Tx.Buffer((M, K), a_type),
    B: Tx.Buffer((N, K), b_type),
    D: Tx.Buffer((M, N), d_type),
):
    with Tx.kernel():
        bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
        wg_id = Tx.warpgroup_id([1], parent="cta")
        warp_id = Tx.warp_id([4], parent="warpgroup")
        lane_id = Tx.thread_id([32], parent="warp")

        # --- SMEM allocation ---
        pool = Tx.PoolAllocator()
        tmem_addr = pool.alloc((1,), "uint32")
        mma_bar = pool.alloc((1,), "uint64", align=8)
        pool.move_base_to(1024)
        Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
        Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
        pool.commit()

        # --- Barrier + TMEM init (warp 0 only) ---
        if warp_id == 0:
            if lane_id == 0:
                Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
            Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

        Tx.ptx.fence.proxy_async("shared::cta")
        Tx.ptx.fence.mbarrier_init()
        Tx.cuda.cta_sync()

        tmem = Tx.decl_buffer(
            (128, 512), "float32", scope="tmem", allocated_addr=0,
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = Tx.meta_var(bx * BLK_M)
        n_st = Tx.meta_var(by * BLK_N)
        phase_mma: Tx.int32
        phase_mma = 0

        # --- Load: all threads copy global -> shared (synchronous) ---
        with Tx.cta():
            Tx.copy(Asmem[:, :], A[:, :])
            Tx.copy(Bsmem[:, :], B[:, :])
        Tx.cuda.cta_sync()
        Tx.ptx.tcgen05.fence.after_thread_sync()

        # --- Compute: single elected thread issues MMA ---
        if warp_id == 0:
            with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                Tx.gemm_async(
                    tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                    accum=False, dispatch="tcgen05", cta_group=1
                )
                Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

        Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

        # --- Writeback: TMEM -> RF -> GMEM ---
        Dreg = Tx.alloc_local((BLK_N,), acc_type)
        Dreg_f16 = Tx.alloc_local((BLK_N,), d_type)
        Dreg_wg = Dreg.view(128, BLK_N,
                            layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        with Tx.warpgroup():
            Tx.copy(Dreg_wg[:, :], tmem[:, :BLK_N])
        with Tx.thread():
            Tx.cast(Dreg_f16[:], Dreg[:])
            m_thr = Tx.meta_var(m_st + warp_id * 32 + lane_id)
            Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

        # --- Deallocate TMEM ---
        Tx.cuda.cta_sync()
        if warp_id == 0:
            Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)
```

Compile and run (using M=N=128, K=64 -- a single tile -- to verify correctness):

```{.python .input}
import torch

target = tvm.target.Target("cuda -arch=sm_100a")
with target:
    mod = tvm.IRModule({"main": hgemm_ver1})
    lib = tvm.compile(mod, target=target, tir_pipeline="tirx")

device = torch.device('cuda')  # gpu(0)
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

f = lib["main"]
f(tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor),
  tvm.runtime.from_dlpack(D_tensor))

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Step 1 Synchronous GEMM: M={M}, N={N}, K={K}")
print(f"Max error vs torch reference: {max_err:.6f}")
assert max_err < 1.0, f"FAIL: max_err={max_err}"
print("PASS")

# Benchmark
args = [tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor), tvm.runtime.from_dlpack(D_tensor)]
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
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

### What to Optimize Next

This kernel is correct but has two major bottlenecks:

1. **Synchronous loads**: All 128 threads are occupied with data copying. The TMA hardware engine sits idle.
2. **No pipelining**: Load → Compute → Load → Compute --- the MMA unit is idle during loads, and the memory system is idle during compute.

The next steps address these one by one.



---

With a single-tile kernel working, the next step is to handle matrices where K is larger than one tile.

## Step 2: K-Loop Accumulation
:label:`chap_k_loop`

Step 1 only handles K=64 (one tile). Real matrices have K >> 64. In this section, we extend the kernel to loop over the K dimension with accumulation (M=N=128, K=256).

### What You Will Learn

- Iterating over the K dimension with multiple MMA invocations

- The `accum` flag: `False` for the first K tile, `True` for subsequent tiles

- mbarrier phase flipping for repeated synchronization

### Background

To handle matrices where K > 64, we loop over K in chunks of `BLK_K=64`. Each iteration loads a new (128x64) A tile and (128x64) B tile, then runs an MMA. The `accum` parameter tells the Tensor Core whether to overwrite TMEM (`False`) or add to it (`True`).

The mbarrier is reused across iterations, but we need to prevent a subtle race: after `try_wait` returns, the barrier is in the "arrived" state. If we call `try_wait` again without changing anything, it would return immediately — before the next MMA has even started.

The solution is **phase tracking**. Each mbarrier has a 1-bit phase (0 or 1) that flips automatically after all expected arrivals. `try_wait(bar, phase)` blocks until the barrier's internal phase **differs** from the `phase` argument:

```
Iteration 0: phase=0, call try_wait(bar, 0) → blocks until barrier flips to phase 1
             MMA completes → barrier arrives → internal phase becomes 1 → wait returns
             We flip: phase = 0 ^ 1 = 1

Iteration 1: phase=1, call try_wait(bar, 1) → blocks until barrier flips to phase 0
             MMA completes → barrier arrives → internal phase becomes 0 → wait returns
             We flip: phase = 1 ^ 1 = 0

...and so on
```

Without `phase_mma ^= 1`, the second `try_wait(bar, 0)` would see the barrier already at phase 1 (from iteration 0's arrival) and return immediately — before the second MMA finishes.

### Complete Implementation

```{.python .input}

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.op_dispatch.cuda.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg as axis_tid_in_wg
```

The kernel is wrapped in a function `hgemm_v2(M, N, K)` that returns a TIRX kernel for the given dimensions. The grid is `[1, 1]` (single CTA) since this step only handles M=N=128:

```{.python .input}
def hgemm_v2(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx, by = Tx.cta_id([1, 1], parent="kernel")  # Single CTA
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer(
                (128, 512), "float32", scope="tmem", allocated_addr=0,
                layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            phase_mma: Tx.int32
            phase_mma = 0
            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            # === K-loop: iterate over K in chunks of BLK_K ===
            for i in range(K_TILES):
                # Load the i-th K chunk
                with Tx.cta():
                    Tx.copy(Asmem[:, :], A[:, i*64:(i+1)*64])
                    Tx.copy(Bsmem[:, :], B[:, i*64:(i+1)*64])

                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

                # MMA: accum=False for first tile, True for rest
                if warp_id == 0:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        Tx.gemm_async(tmem[:, :128], Asmem[:], Bsmem[:],
                                      accum=(i != 0), dispatch="tcgen05", cta_group=1)
                        Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

                # Wait for MMA, then flip phase
                Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma ^= 1

            # === Writeback (same as Step 1) ===
            reg = Tx.alloc_local((128,), acc_type)
            reg_f16 = Tx.alloc_local((128,), d_type)
            reg_wg = reg.view(128, 128,
                              layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))

            with Tx.warpgroup():
                Tx.copy(reg_wg[:], tmem[:, :128])
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

            with Tx.thread():
                Tx.cast(reg_f16[:], reg[:])
                m_thr = Tx.meta_var(m_st + 32 * warp_id + lane_id)
                Tx.copy(D[m_thr, :], reg_f16[:])

            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

Compile and run with M=N=128, K=256 (4 K-tiles) to verify the K-loop accumulation:

```{.python .input}
import torch

M, N, K = 128, 128, 256
target = tvm.target.Target("cuda -arch=sm_100a")

kernel = hgemm_v2(M, N, K)
with target:
    mod = tvm.IRModule({"main": kernel})
    lib = tvm.compile(mod, target=target, tir_pipeline="tirx")

device = torch.device('cuda')  # gpu(0)
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

f = lib["main"]
f(tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor),
  tvm.runtime.from_dlpack(D_tensor))

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Step 2 K-loop GEMM: M={M}, N={N}, K={K} ({K // 64} K-tiles)")
print(f"Max error vs torch reference: {max_err:.6f}")
assert max_err < 1.0, f"FAIL: max_err={max_err}"
print("PASS")

# Benchmark
args = [tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor), tvm.runtime.from_dlpack(D_tensor)]
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
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

### What Changed from Step 1

| Change | Step 1 | Step 2 |
|--------|--------|--------|
| K dimension | K=64 (one tile) | K=any multiple of 64 |
| Loop | None | `for i in range(K_TILES)` |
| Accumulation | `accum=False` always | `accum=(i != 0)` |
| Phase tracking | Single wait | `phase_mma ^= 1` after each wait |

**Why phase flipping matters**: The mbarrier auto-toggles its phase after all arrivals. If we don't track the phase, the second `try_wait` would see the old (already-passed) phase and return immediately --- before the second MMA finishes.



---

Now that we can handle arbitrary K, the final piece is tiling over M and N to support full-sized matrices.

## Step 3: Spatial Tiling (Multi-CTA)
:label:`chap_spatial_tiling`

Steps 1-2 only handle a single output tile (M=N=128). In this section, we launch a 2D grid of CTAs to cover arbitrary matrix dimensions (M=N=K=256).

### What You Will Learn

- Launching a 2D grid of CTAs to cover arbitrary M and N dimensions

- Per-CTA tile offset calculation with `Tx.meta_var`

### Background

To support larger matrices, we launch a 2D grid of CTAs: `[M // BLK_M, N // BLK_N]`. Each CTA computes one 128x128 output tile.

CTA `(bx, by)` computes `D[bx*128 : (bx+1)*128, by*128 : (by+1)*128]` by loading `A[bx*128 : (bx+1)*128, :]` and `B[by*128 : (by+1)*128, :]`.

### Complete Implementation

```{.python .input}

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.op_dispatch.cuda.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg as axis_tid_in_wg
```

The key change from Step 2: the grid is now `[M // BLK_M, N // BLK_N]` instead of `[1, 1]`, and loads/stores use per-CTA offsets `m_st` and `n_st`:

```{.python .input}
def hgemm_v3(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float16")
    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K

    A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
    B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))

    @Tx.prim_func(tirx=True)
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            # 2D grid: one CTA per 128x128 output tile
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N], parent="kernel")
            wg_id = Tx.warpgroup_id([1], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")

            pool = Tx.PoolAllocator()
            tmem_addr = pool.alloc((1,), "uint32")
            mma_bar = pool.alloc((1,), "uint64", align=8)
            pool.move_base_to(1024)
            Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
            Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
            pool.commit()

            if warp_id == 0:
                if lane_id == 0:
                    Tx.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
                Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)

            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            Tx.cuda.cta_sync()

            tmem = Tx.decl_buffer(
                (128, 512), "float32", scope="tmem", allocated_addr=0,
                layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            phase_mma: Tx.int32
            phase_mma = 0

            # Per-CTA tile offsets
            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            # K-loop with offset A and B slices
            for i in range(K_TILES):
                with Tx.cta():
                    Tx.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*64:(i+1)*64])
                    Tx.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*64:(i+1)*64])

                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

                if warp_id == 0:
                    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
                        Tx.gemm_async(tmem[:, :128], Asmem[:], Bsmem[:],
                                      accum=(i != 0), dispatch="tcgen05", cta_group=1)
                        Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

                Tx.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
                phase_mma ^= 1

            # Writeback to the correct output tile
            reg = Tx.alloc_local((128,), acc_type)
            reg_f16 = Tx.alloc_local((128,), d_type)
            reg_wg = reg.view(128, 128,
                              layout=TileLayout(S[(128, BLK_N) : (1@axis_tid_in_wg, 1)]))

            with Tx.warpgroup():
                Tx.copy(reg_wg[:], tmem[:, :128])
                Tx.cuda.cta_sync()
                Tx.ptx.tcgen05.fence.after_thread_sync()

            with Tx.thread():
                Tx.cast(reg_f16[:], reg[:])
                m_thr = Tx.meta_var(m_st + 32 * warp_id + lane_id)
                Tx.copy(D[m_thr, n_st:n_st+BLK_N], reg_f16[:])

            if warp_id == 0:
                Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
                Tx.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

    return kernel
```

Compile and run with M=N=K=256 (a 2x2 grid of CTAs) to verify multi-CTA correctness:

```{.python .input}
import torch

M, N, K = 256, 256, 256
target = tvm.target.Target("cuda -arch=sm_100a")

kernel = hgemm_v3(M, N, K)
with target:
    mod = tvm.IRModule({"main": kernel})
    lib = tvm.compile(mod, target=target, tir_pipeline="tirx")

device = torch.device('cuda')  # gpu(0)
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

f = lib["main"]
f(tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor),
  tvm.runtime.from_dlpack(D_tensor))

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Step 3 Spatial Tiling GEMM: M={M}, N={N}, K={K} ({M//128}x{N//128} CTA grid)")
print(f"Max error vs torch reference: {max_err:.6f}")
assert max_err < 1.0, f"FAIL: max_err={max_err}"
print("PASS")

# Benchmark
args = [tvm.runtime.from_dlpack(A_tensor), tvm.runtime.from_dlpack(B_tensor), tvm.runtime.from_dlpack(D_tensor)]
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
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

### What Changed from Step 2

| Change | Step 2 | Step 3 |
|--------|--------|--------|
| Grid | `[1, 1]` (single CTA) | `[M//BLK_M, N//BLK_N]` (2D grid) |
| Tile offset | None | `m_st = bx * BLK_M`, `n_st = by * BLK_N` |
| Load slice | `A[:, k:k+64]` | `A[m_st:m_st+BLK_M, k:k+64]` |
| Writeback | `D[m_thr, :]` | `D[m_thr, n_st:n_st+BLK_N]` |

The K-loop body is identical to Step 2 --- only the grid dimensions and offset calculations change.

## Exercises

1. Why is `cta_sync() + fence.after_thread_sync()` needed between the sync copy and the MMA? What could go wrong without it?
2. What happens if you remove `phase_mma ^= 1` in Step 2's K-loop? Will the kernel deadlock, produce wrong results, or both?
3. For M=N=4096 with BLK_M=BLK_N=128, how many CTAs are launched in Step 3? Do adjacent CTAs share any data from GMEM?


## Debugging: Inspecting Generated CUDA Source

When your kernel produces wrong results, deadlocks, or crashes, inspecting the generated CUDA code is the most effective debugging tool — it shows you exactly which threads execute which instructions.

```python
cuda_source = lib.mod.imports[0].inspect_source()
print(cuda_source)
```

### Key Mappings from TIRX to Generated CUDA

| TIRX | Generated CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` internal guard | `((int)threadIdx.x) < 1` (CTA thread 0 only) |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

### What to Look For

- **Barrier init count**: Search for `mbarrier_init` — check that the arrival count matches your code.
- **Thread guards**: MMA and commit should be inside an `elect_sync` guard. If they are inside `threadIdx.x < 1`, only CTA thread 0 executes them.
- **TMEM alloc**: Search for `tcgen05_alloc` — verify it runs from the correct warpgroup and warp.
- **MMA unrolling**: For K=64 with MMA_K=16, you should see 4 `ptx_tcgen05_mma` calls with increasing descriptor offsets.
- **Swizzle arithmetic**: SMEM address calculations contain XOR expressions like `((v ^ ((v & 56) >> 3)) << 3)` — this is the swizzle pattern for bank-conflict-free access.

