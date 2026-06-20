(chap_tirx_primer)=
# Introduction to TIRx

:::{admonition} Overview
:class: overview

- TIRx is a Python DSL for writing GPU kernels at the IR level: you name hardware directly, but through structured IR.
- Every tile operation is controlled by three design elements — *scope* (which threads), *layout* (where tiles live), and *dispatch* (which hardware path).
- One runnable single-MMA GEMM shows all three; the rest of the book is these design elements at scale.
:::

Part I explained what the hardware is; to make it compute anything, we need a way to program it. We could write raw CUDA or PTX, and many fast kernels are written exactly that way. But the decisions that actually determine a kernel's behavior — which threads run an operation, where each tile of data lives, which hardware path executes it — do not appear as such in that code. They are buried in intrinsic arguments, address arithmetic, and convention, scattered across the kernel where they are hard to see and harder to change. TIRx (Tensor IR neXt) is a Python DSL that lifts those three decisions into the open: **scope** (which threads run an operation), **layout** (where the operand tiles live), and **dispatch** (which hardware path executes it). It still names hardware concepts directly — threads, shared and tensor memory, barriers, `tcgen05` MMA — but as structured IR the compiler can see, so it can lower, check, and schedule the kernel rather than treat it as opaque intrinsic calls. Like the framework in *Dive into Deep Learning*, TIRx is the consistent medium through which every concept in this book becomes runnable code, and this chapter introduces it through one small end-to-end kernel.

Rather than introduce these ideas in the abstract, we will work from a single complete kernel: a minimal single-MMA GEMM. We get it running first, and only then read it back, line by line, to see how scope, layout, and dispatch each shape it and how the kernel is compiled. The tensor layout model that the kernel relies on is developed in its own right in {ref}`chap_data_layouts`, and the full language-feature set in {ref}`chap_language_reference`; here we keep the focus on the one kernel and the three design elements.

## A First Kernel: Single-MMA GEMM

Our example computes a single 128 x 128 output tile of `D = A B^T` with K = 64. The whole computation is expressed as one `Tx.gemm_async` tile operation, from end to end. (That one tile operation does not map to a single hardware instruction: because the hardware MMA K-atom is 16, the K=64 tile lowers to a short sequence of `tcgen05.mma` instructions stepping along K. The point of the DSL is precisely that we write the tile, not the sequence.) Around that operation, the kernel does the usual chores: it allocates shared memory (SMEM) and tensor memory (TMEM), copies A and B from global to shared memory, issues the tile MMA into a TMEM accumulator, reads that accumulator back out through registers, and stores the result. Small as it is, this kernel is Step 1 of the GEMM ladder we climb in {ref}`chap_gemm_basics`, where it returns with a full walkthrough.

Every TIRx kernel begins from the same handful of imports, so it is worth seeing them once up front:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

We wrap the kernel in a small builder, `hgemm_v1(M, N, K)`, that takes the problem shape and returns a `PrimFunc`. For our chosen shape, `M=N=128, K=64`, the launch happens to contain exactly one output tile, which is what keeps this first version simple enough to read in one sitting:

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

Before we read the kernel, let us make sure it works. We compile it and check its output against a torch reference. We do not have to spell out the exact architecture: the arch (e.g. `sm_100a`) is auto-detected from the device, so the target `"cuda"` is enough, and `tir_pipeline="tirx"` is what selects the TIRx lowering pipeline. Once compiled, `ex.mod(...)` takes torch tensors directly, with no manual conversion in between.

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
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")
```

## Scope, Layout, Dispatch

Now that the kernel runs, we can read it back and ask what its lines actually decide. Seen this way, the whole kernel is a set of choices along three design elements. Every operation in it answers the same three questions — *who* runs it, *where* its data lives, and *how* it executes — and those three answers are exactly scope, layout, and dispatch. The rest of this section takes the design elements one at a time; the interactive demo below lets you see which lines each design element controls.

```{raw} html
<iframe src="../demo/tirx_dispatch.html" title="TIRx: scope, layout, dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: click Scope / Layout / Dispatch to spotlight the lines of the kernel each design element controls.*

**Scope — which threads issue or cooperate.** Every operation names the group of threads that runs it, and different operations want different groups. The global-to-shared copy `Tx.cta.copy(...)` is CTA-scoped: all 128 threads pitch in, each carrying its own slice of the data. The MMA sits at the other extreme. A `tcgen05.mma` is a *single-instruction* cooperative op, so there is nothing to be gained by having every thread issue it — the kernel narrows the scope down to a single thread with `if warp_id == 0: if T.ptx.elect_sync():`, and that one elected thread issues both `Tx.gemm_async` and `tcgen05.commit`. The hardware still runs the MMA over the whole tile; issuing it once simply spares us from launching the same MMA 128 times. The readback `Tx.wg.copy_async(...)` lands in between, at warpgroup scope: the warpgroup's 128 threads split the 128 x 128 accumulator row by row, each thread pulling its own row out of TMEM.

**Layout — where the operands and accumulator live.** In TIRx no tile floats free of memory; each one carries an explicit placement, and the kernel above sets three of them. The A and B operands go into SMEM under a `tma_shared_layout` (`A_layout` and `B_layout`), a swizzled shared-memory layout that `tcgen05.mma` insists on. The accumulator lives in TMEM, declared with `T.decl_buffer(..., scope="tmem", ...)` and a `TileLayout` over `TLane`/`TCol`. And the register readback view `Dreg_wg` uses `TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])`, which maps the first tile axis onto warpgroup threads so that row *i* is owned by warpgroup thread *i*. These placements are not decoration: layout is part of an operation's contract. The MMA reads the SMEM layouts of its operands and the TMEM layout of its accumulator precisely so it knows how the tile is physically arranged.

**Dispatch — which hardware path executes it.** A tile operation can often be lowered in more than one way, and dispatch is how we pick. Here `Tx.gemm_async(..., dispatch="tcgen05", ...)` selects the Blackwell Tensor Core path, so the tile MMA lowers to `tcgen05.mma` writing into TMEM. Dispatch reaches the copies too: the synchronous `Tx.cta.copy` in this kernel lowers to plain thread copies, and later GEMM steps will swap that for TMA without touching the surrounding scope or layout at all. Because the path is named rather than implied, the same kernel shape can be retargeted to different engines just by changing this one design element.

## How Compilation Works

We already compiled the kernel above to test it; now we look a little closer at what that step does. The recipe is short: wrap the `PrimFunc` in an `IRModule` and hand it to `tvm.compile(mod, target=..., tir_pipeline="tirx")`. This runs the TIRx lowering pipeline and hands back an `Executable` that you call directly.

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

It is worth knowing, at least in outline, what `tir_pipeline="tirx"` sets in motion. The pipeline's central pass, `LowerTIRx`, resolves each tile primitive against its scope / layout / dispatch contract — this is where the three design elements we just discussed are actually cashed out into instructions. After that, the usual host/device split and a finalize step produce the launchable module. If you prefer, you can also compile inside a `with target:` block, which lets the kernel pick up the surrounding target context.

One nice property of this flow is that nothing is hidden from you: the result can be inspected at both levels. You can read the IR itself with `.show()` or `.script()`, and you can read the CUDA C that the compiler ultimately emitted straight off the compiled module.

```python
kernel.show()                          # pretty-print the TIRx (TVMScript)
print(kernel.script())                 # ... the same, as a string

# the generated CUDA C source, from the compiled Executable:
print(ex.mod.imports[0].inspect_source())
```

This is only a sketch. For the full lowering story — all of the passes, how tile-primitive dispatch is resolved, and how the host/device split is done — see {ref}`chap_arch`.

## Where to Go Next

- {ref}`chap_data_layouts` — the tensor layout model (`TileLayout`, named axes, swizzle) that the operand and accumulator placements above are built from. Start here if the layout design element felt like the most mysterious of the three.
- {ref}`chap_language_reference` — the full language-feature set, covering parser utilities, data types, buffers and memory, control flow, and thread synchronization, for when you want the complete vocabulary rather than the tour.
- {ref}`chap_gemm_basics` — this kernel as Step 1 of the GEMM optimization path, built up through K-loop accumulation, spatial tiling, TMA, and warp specialization. This is the natural next stop if you want to see the same three design elements scale up to a real kernel.
