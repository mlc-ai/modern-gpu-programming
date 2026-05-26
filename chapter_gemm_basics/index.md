# GEMM with Tile Primitives
:label:`chap_gemm_basics`

The GEMM path starts from a single-tile baseline, then adds the pieces needed for larger tiled matrices. This chapter covers single-tile GEMM, K-loop accumulation, and spatial tiling across CTAs.

GEMM kernels use three recurring layout patterns: a swizzled `tma_shared_layout` for SMEM, `S[(128, N) : (1@TLane, 1@TCol)]` for TMEM, and `S[(128, N) : (1@tid_in_wg, 1)]` for the warpgroup register view. :numref:`chap_layouts` covers named axes, swizzle modes, and TMEM-RF matching.

## GEMM

GEMM (General Matrix Multiplication) is the dense matrix multiply used throughout this tutorial. The examples use $D = A B^{\top}$ where:

- $A$ is $M \times K$ (M rows of K-dimensional vectors)

- $B$ is $N \times K$ (N rows of K-dimensional vectors)

- $D$ is $M \times N$ (output)

- Operation: $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$ (i.e., $D = A B^{\top}$ since B is stored as N rows of K columns)

GEMM is central to deep learning workloads: linear layers, attention, and many convolutions reduce to matrix multiplication. GEMM throughput matters for both training and inference.

GEMM performance is measured in TFLOPS (Tera Floating-Point Operations Per Second):

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM Data Path

A Blackwell GEMM kernel is organized around a tile data path:

![*Memory Data Flow*](../img/memory_dataflow.png)

Operand tiles move from GMEM to SMEM, `tcgen05.mma` consumes the SMEM operands and writes accumulators to TMEM, the epilogue reads TMEM back into registers, and the final result is stored to GMEM. Optimized kernels use TMA for the large GMEM <-> SMEM transfers and overlap those transfers with MMA.

Step 1 below is not the fully optimized version of this picture. It is the smallest kernel that exposes the same main objects: SMEM operand tiles, a TMEM accumulator, a warpgroup TMEM readback, and a final GMEM store. Later steps replace the simple parts with TMA, pipelining, warp specialization, and CTA clusters.

## Optimization Path

The optimization path adds hardware features one at a time:
- **Async Data Movement** --- use TMA so a thread issues a tile copy and the hardware transfer engine performs the movement
- **Software Pipelining** --- overlap load and compute with multi-buffering
- **Warp Specialization** --- split producer, consumer, and writeback work across warp roles
- **CTA Clusters** --- 2SM cooperative MMA for higher arithmetic intensity

Later chapters add persistent scheduling and multi-consumer execution on top of these pieces.

---

## Step 1: Sequential Single-Tile GEMM
:label:`chap_single_tile`

The first working GEMM kernel computes one output tile (M=N=128, K=64). It is sequential rather than pipelined: each async operation is explicitly waited on before the next dependent step starts.

**Topics.**

- Synchronous data loading: all 128 threads copy from GMEM → SMEM

- tcgen05 MMA invocation: single thread issues MMA, all threads wait

- TMEM allocation/deallocation and writeback (TMEM → RF → GMEM)

### Step 1 Dataflow

In Step 1, each step happens **sequentially** --- the next step waits for the previous to complete. Two parts are deliberately simpler than the optimized path above: GMEM -> SMEM uses synchronous `Tx.copy` instead of TMA, and writeback stores from registers directly to GMEM instead of staging through SMEM for a TMA store.

1. **Allocate**: SMEM (pool allocator), TMEM (`tcgen05.alloc`), mbarrier
2. **Load**: All 128 threads cooperatively copy A and B tiles from GMEM to SMEM (sync `Tx.copy`)
3. **Compute**: Single elected thread issues `Tx.gemm_async` + `tcgen05.commit`; all threads wait on mbarrier
4. **Writeback**: Warpgroup reads TMEM → registers; each thread casts fp32→fp16 and writes to GMEM
5. **Deallocate**: TMEM deallocation

### Kernel Pieces

Before reading the full source, keep this code map in mind. It is not meant to compile by itself; it shows where the important objects and tile primitives appear in the real kernel below.

```python
@Tx.prim_func
def hgemm_v1(A, B, D):
    with Tx.kernel():
        # 1. Identify this CTA and the threads inside it.
        bx, by = Tx.cta_id([M // BLK_M, N // BLK_N])
        warp_id = Tx.warp_id_in_wg([4])
        lane_id = Tx.lane_id([32])

        # 2. Allocate SMEM metadata, operand tiles, a TMEM accumulator,
        #    and per-thread registers for readback.
        pool = Tx.SMEMPool()
        Asmem = pool.alloc(...)
        Bsmem = pool.alloc(...)
        tmem = Tx.decl_buffer(..., scope="tmem", ...)
        Dreg = Tx.alloc_local(...)
        Dreg_f16 = Tx.alloc_local(...)
        Dreg_wg = Dreg.view(...)

        # 3. Move operand tiles into SMEM.
        with Tx.cta():
            Tx.copy(Asmem[:, :], A[:, :])
            Tx.copy(Bsmem[:, :], B[:, :])

        # 4. One selected issuer starts tcgen05 MMA into TMEM,
        #    then the CTA waits for MMA completion.
        with Tx.thread(...):  # selected issuer only
            Tx.gemm_async(tmem[:, :], Asmem[:, :], Bsmem[:, :],
                          dispatch="tcgen05")
            Tx.ptx.tcgen05.commit(...)
        Tx.ptx.mbarrier.try_wait(...)

        # 5. Read TMEM into registers, then store the result.
        with Tx.warpgroup():
            Tx.copy_async(Dreg_wg[:, :], tmem[:, :])
            Tx.ptx.tcgen05.wait.ld()
        with Tx.thread():
            row = ...
            Tx.cast(Dreg_f16[:], Dreg[:])
            Tx.copy(D[row, :], Dreg_f16[:])
```

The rest of this section expands the code map one piece at a time.

The walkthrough below groups the real kernel into four pieces.

#### Memory Allocation

```python
pool = Tx.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 bytes)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 bytes)
pool.move_base_to(1024)                           # Skip to offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

The `pool.move_base_to(1024)` ensures Asmem/Bsmem start at offset 1024, leaving room for metadata. The `layout=A_layout` uses `tma_shared_layout` for a swizzled SMEM placement compatible with TMA and `tcgen05.mma`; :numref:`chap_layouts` explains why SMEM layout is part of the tile primitive contract.

#### Synchronous Load

```python
with Tx.cta():  # All 128 threads cooperate
    Tx.copy(Asmem[:, :], A[:, :])
    Tx.copy(Bsmem[:, :], B[:, :])
Tx.cuda.cta_sync()                           # Wait for all threads; bar.sync
                                             # also releases the SMEM writes
                                             # to the subsequent MMA.
```

Step 1 only has one tile (M=N=128, K=64), so we copy the entire A and B. `with Tx.cta()` is a **scope marker** that means all 128 threads in the CTA participate — every thread executes the `Tx.copy` inside, each handling a portion of the data. This is the simplest data-movement path: 128 threads compute addresses and issue load/store instructions. Step 4 replaces this with TMA, where one selected thread issues the tile copy and dedicated hardware performs the transfer.

#### MMA Dispatch

```python
if warp_id == 0:
    with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
        Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
                      accum=False, dispatch="tcgen05", cta_group=1)
        Tx.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

`elect_sync()` selects one thread per active warp. Because this block is also guarded by `if warp_id == 0`, exactly one thread issues the MMA and commit. `accum=False` means overwrite TMEM (not accumulate).

#### Writeback

```python
Dreg_wg = Dreg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
with Tx.warpgroup():          # All 128 threads cooperate on TMEM read
    Tx.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])  # → tcgen05.ld (async)
    Tx.ptx.tcgen05.wait.ld()                       # synchronize before reading Dreg
with Tx.thread():             # Each thread writes its row
    Tx.cast(Dreg_f16[:], Dreg[:])     # fp32 → fp16
    m_thr = Tx.meta_var(m_st + warp_id * 32 + lane_id)
    Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

Thread row mapping: warp 0 handles rows 0-31, warp 1 handles rows 32-63, etc. Each thread's row is `m_st + warp_id * 32 + lane_id`.

**Implementation.**

With the walkthrough in mind, here is the complete runnable kernel (M=N=128, K=64):

```{.python .input}

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.operator.tile_primitive.cuda.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

Constants and swizzled SMEM layouts for the A and B tiles (the `tma_shared_layout` helper is introduced in :numref:`chap_layouts`):

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

GEMM kernels use the full thread hierarchy from :numref:`chap_background` (CTA → Warpgroup → Warp → Thread). Unlike the earlier elementwise kernels that only needed `cta_id` + `thread_id`, GEMM touches every level: a single elected thread *issues* each `tcgen05.mma`, but the resulting TMEM tile spans 128 rows — a full warpgroup cooperates to read it back via `tcgen05.ld`, and per-warp / per-lane IDs partition the writeback rows across threads:

- **`Tx.warpgroup_id([1])`** — Number of warpgroups per CTA. Step 1 only uses a single warpgroup so this is `1`; the result is bound to `_` because the kernel never references the warpgroup index inside its body. The declaration still matters because the compiler needs the full thread hierarchy (cta -> warpgroup -> warp -> thread).
- **`warp_id = Tx.warp_id_in_wg([4])`** — Which warp within the warpgroup (0-3).
- **`lane_id = Tx.lane_id([32])`** — Which thread within the warp (0-31).

Later steps that need warp specialization (:numref:`chap_gemm_advanced` Step 7 onward) bind `Tx.warpgroup_id([2])` to a name and dispatch on it.

The parent level is encoded in the function name itself (`*_in_wg` for warpgroup-scoped, `lane_id` for warp-scoped, `cta_id_in_cluster` for cluster-scoped). For example, `Tx.warp_id_in_wg([4])` enumerates the 4 warps inside each warpgroup, while `Tx.warp_id([8])` would enumerate all 8 warps of a 2-warpgroup CTA. :numref:`chap_layouts` introduces these symbolic coordinates as part of execution scope.

The `warp_id` and `lane_id` are used for thread-level operations like data loading and writeback (each thread handles a different row: `row = warp_id * 32 + lane_id`).

The kernel itself:

```{.python .input}
@Tx.prim_func
def hgemm_v1(
    A: Tx.Buffer((M, K), a_type),
    B: Tx.Buffer((N, K), b_type),
    D: Tx.Buffer((M, N), d_type),
):
    with Tx.kernel():
        # Step 1 is a single-tile kernel: M = BLK_M and N = BLK_N, so the grid
        # is 1x1. Starting with a 1x1 grid keeps the per-CTA tile offsets
        # (m_st, n_st) trivially zero; Steps 3+ generalise this to larger M / N.
        bx, by = Tx.cta_id([M // BLK_M, N // BLK_N])
        _ = Tx.warpgroup_id([1])          # single warpgroup, index never used inside the kernel
        warp_id = Tx.warp_id_in_wg([4])
        lane_id = Tx.lane_id([32])

        # --- SMEM allocation ---
        pool = Tx.SMEMPool()
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
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
        )

        m_st = Tx.meta_var(bx * BLK_M)
        n_st = Tx.meta_var(by * BLK_N)
        phase_mma: Tx.int32 = 0

        # --- Load: all threads copy global -> shared (synchronous).
        # With M=BLK_M and N=BLK_N the slices below cover the full matrices;
        # the slice form is kept so the diff to Step 3 (multi-tile) is minimal.
        with Tx.cta():
            Tx.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
            Tx.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
        Tx.cuda.cta_sync()

        # --- Compute: single elected thread issues MMA ---
        if warp_id == 0:
            with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
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
            Tx.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
            Tx.ptx.tcgen05.wait.ld()
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

target = tvm.target.Target("cuda")
with target:
    mod = tvm.IRModule({"main": hgemm_v1})
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
assert max_err < 0.1, f"FAIL: max_err={max_err}"
print("PASS")

```

### Next Bottlenecks

The kernel is correct but has two major bottlenecks:

1. **Synchronous loads**: All 128 threads are occupied with data copying. The TMA hardware engine sits idle.
2. **No pipelining**: Load → Compute → Load → Compute --- the MMA unit is idle during loads, and the memory system is idle during compute.

The following steps address these bottlenecks one by one.



---

With a single-tile kernel working, K can grow beyond one tile.

## Step 2: K-Loop Accumulation
:label:`chap_k_loop`

Step 1 only handles K=64 (one tile). Many useful GEMMs have a larger K dimension. The Step 2 kernel loops over K with accumulation (M=N=128, K=256).

**Changes from Step 1** (everything else stays the same):

| What | Step 1 | Step 2 |
|---|---|---|
| K dimension | K = 64 (one tile) | K = multiple of 64 in this example |
| K-loop | none | `for i in range(K_TILES):` |
| Accumulation | `accum=False` always | `accum=(i != 0)` (False on first tile, True after) |
| Phase tracking | single `try_wait` | `phase_mma ^= 1` after each wait |

**Topics.**

- Iterating over the K dimension with multiple MMA invocations

- The `accum` flag: `False` for the first K tile, `True` for subsequent tiles

- mbarrier phase flipping for repeated synchronization

### K-Loop Mechanics

To handle matrices where K > 64, we loop over K in chunks of `BLK_K=64`. Each iteration loads a new (128x64) A tile and (128x64) B tile, then runs an MMA. The `accum` parameter tells the Tensor Core whether to overwrite TMEM (`False`) or add to it (`True`).

The mbarrier is reused across iterations, but phase tracking prevents a race: after `try_wait` returns, the barrier is in the "arrived" state. If we call `try_wait` again without changing anything, it would return immediately — before the next MMA has even started.

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

**Implementation.**

```{.python .input}

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.operator.tile_primitive.cuda.tma_utils import tma_shared_layout, SwizzleMode
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

    @Tx.prim_func
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            bx, by = Tx.cta_id([1, 1])  # Single CTA
            _ = Tx.warpgroup_id([1])
            warp_id = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])

            pool = Tx.SMEMPool()
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
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            phase_mma: Tx.int32 = 0
            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            # === K-loop: iterate over K in chunks of BLK_K ===
            for i in range(K_TILES):
                # Load the i-th K chunk
                with Tx.cta():
                    Tx.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
                    Tx.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

                Tx.cuda.cta_sync()

                # MMA: accum=False for first tile, True for rest
                if warp_id == 0:
                    with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
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
                Tx.copy_async(reg_wg[:], tmem[:, :128])
                Tx.ptx.tcgen05.wait.ld()
                Tx.cuda.cta_sync()

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
target = tvm.target.Target("cuda")

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
assert max_err < 0.1, f"FAIL: max_err={max_err}"
print("PASS")

```

**Why phase flipping matters**: The mbarrier auto-toggles its phase after all arrivals. If we don't track the phase, the second `try_wait` would see the old (already-passed) phase and return immediately — before the second MMA finishes. The diff table spells out exactly what changed from Step 1.



---

K-loop accumulation still leaves one missing piece: tiling over M and N to support larger matrices.

## Step 3: Spatial Tiling (Multi-CTA)
:label:`chap_spatial_tiling`

Steps 1-2 only handle a single output tile (M=N=128). A 2D grid of CTAs covers larger M and N dimensions. The example uses M=N=K=256.

**Changes from Step 2** (the K-loop body is unchanged):

| What | Step 2 | Step 3 |
|---|---|---|
| Grid | `[1, 1]` (single CTA) | `[M // BLK_M, N // BLK_N]` (2D grid of CTAs) |
| Tile offset | none | `m_st = bx * BLK_M`, `n_st = by * BLK_N` |
| Load slice | `A[:, k:k+64]` | `A[m_st:m_st+BLK_M, k:k+64]` |
| Writeback | `D[m_thr, :]` | `D[m_thr, n_st:n_st+BLK_N]` |

**Topics.**

- Launching a 2D grid of CTAs to cover multiple M and N tiles

- Per-CTA tile offset calculation with `Tx.meta_var`

### Grid Mapping

To support larger matrices, we launch a 2D grid of CTAs: `[M // BLK_M, N // BLK_N]`. Each CTA computes one 128x128 output tile.

CTA `(bx, by)` computes `D[bx*128 : (bx+1)*128, by*128 : (by+1)*128]` by loading `A[bx*128 : (bx+1)*128, :]` and `B[by*128 : (by+1)*128, :]`.

**Implementation.**

```{.python .input}

import tvm
from tvm.script import tirx as Tx
from tvm.tirx.operator.tile_primitive.cuda.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg as axis_tid_in_wg
```

Step 3 changes the grid to `[M // BLK_M, N // BLK_N]` instead of `[1, 1]`, and loads/stores use per-CTA offsets `m_st` and `n_st`:

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

    @Tx.prim_func
    def kernel(
        A: Tx.Buffer((M, K), a_type),
        B: Tx.Buffer((N, K), b_type),
        D: Tx.Buffer((M, N), d_type),
    ):
        with Tx.kernel():
            # 2D grid: one CTA per 128x128 output tile
            bx, by = Tx.cta_id([M // BLK_M, N // BLK_N])
            _ = Tx.warpgroup_id([1])
            warp_id = Tx.warp_id_in_wg([4])
            lane_id = Tx.lane_id([32])

            pool = Tx.SMEMPool()
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
            (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

            phase_mma: Tx.int32 = 0

            # Per-CTA tile offsets
            m_st = Tx.meta_var(bx * BLK_M)
            n_st = Tx.meta_var(by * BLK_N)

            # K-loop with offset A and B slices
            for i in range(K_TILES):
                with Tx.cta():
                    Tx.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])
                    Tx.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])

                Tx.cuda.cta_sync()

                if warp_id == 0:
                    with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
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
                Tx.copy_async(reg_wg[:], tmem[:, :128])
                Tx.ptx.tcgen05.wait.ld()
                Tx.cuda.cta_sync()

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
target = tvm.target.Target("cuda")

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
assert max_err < 0.1, f"FAIL: max_err={max_err}"
print("PASS")

```

As the diff table shows, the K-loop body is identical to Step 2 — only the grid dimensions and offset calculations change.

## Exercises

1. Why is `cta_sync()` needed between the synchronous SMEM copy and the MMA? What could go wrong without it?
2. What happens if you remove `phase_mma ^= 1` in Step 2's K-loop? Will the kernel deadlock, produce wrong results, or both?
3. For M=N=4096 with BLK_M=BLK_N=128, how many CTAs are launched in Step 3? Do adjacent CTAs share any data from GMEM?
