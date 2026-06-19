(chap_tirx_primer)=
# Introduction to TIRx

TIRx (Tensor IR neXt) is a Python DSL for writing GPU kernels at the IR level. A TIRx kernel names hardware concepts directly — threads, shared and tensor memory, barriers, `tcgen05` MMA — but instead of scattering that intent across raw CUDA/PTX intrinsic arguments, it exposes the three things the compiler needs as structured IR: **scope** (which threads run an operation), **layout** (where the operand tiles live), and **dispatch** (which hardware path executes it). The compiler sees those knobs explicitly, so it can lower, check, and schedule the kernel instead of treating it as opaque intrinsic calls. Like the framework in *Dive into Deep Learning*, TIRx is the consistent medium through which every concept in this book becomes runnable code.

Rather than catalogue the language feature by feature, we start from one complete kernel — a minimal single-MMA GEMM — get it running, and then read it back to unpack scope / layout / dispatch and to see how compilation works. Once those three knobs are concrete, the rest of the language falls into place. The tensor layout model is covered in {ref}`chap_data_layouts`, and the full language-feature set in {ref}`chap_language_reference`.

## A First Kernel: Single-MMA GEMM

The smallest interesting GEMM is a single MMA, so that is where we begin. The example computes one 128 x 128 output tile of `D = A B^T` with K = 64: a single Blackwell `tcgen05` MMA, end to end. Following the data, it allocates SMEM and TMEM, copies A and B from global to shared memory, issues one MMA into a TMEM accumulator, reads that accumulator back through registers, and stores the result. This is Step 1 of the GEMM ladder built up in {ref}`chap_gemm_basics`; it reappears there with the full walkthrough.

All TIRx kernels start from the same imports:

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

The kernel is wrapped in a `hgemm_v1(M, N, K)` builder. With `M=N=128, K=64` the launch contains exactly one output tile:

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

Now we run it. Compile the kernel and check the output against a torch reference. The arch (e.g. `sm_100a`) is auto-detected from the device, so the target `"cuda"` is enough; `tir_pipeline="tirx"` selects the TIRx lowering pipeline. `ex.mod(...)` takes torch tensors directly.

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

With the kernel running, we can read it back as what it really is: a set of choices along three knobs. Every operation answers three questions — *who* runs it, *where* its data lives, and *how* it executes — and those answers are scope, layout, and dispatch.

```{raw} html
<iframe src="../demo/tirx_dispatch.html" title="TIRx: scope, layout, dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: click Scope / Layout / Dispatch to spotlight the lines of the kernel each knob controls.*

**Scope — which threads issue or cooperate.** Operations name the group of threads that runs them. `Tx.cta.copy(...)` is CTA-scoped: all 128 threads cooperate on the global-to-shared copy, each handling a slice. The MMA is the opposite extreme — `tcgen05.mma` is a *single-instruction* cooperative op, so the kernel narrows scope to one thread with `if warp_id == 0: if T.ptx.elect_sync():`, and that one elected thread issues `Tx.gemm_async` plus `tcgen05.commit`. The hardware still runs the whole tile's MMA; issuing it once avoids launching the same MMA 128 times. The readback `Tx.wg.copy_async(...)` is warpgroup-scoped: the 128 threads of the warpgroup split the 128 x 128 accumulator row-by-row, each thread pulling its own row out of TMEM.

**Layout — where the operands and accumulator live.** Each tile carries an explicit placement. The A and B operands go into SMEM under a `tma_shared_layout` (`A_layout` / `B_layout`) — a swizzled shared-memory layout that `tcgen05.mma` requires. The accumulator lives in TMEM, declared with `T.decl_buffer(..., scope="tmem", ...)` and a `TileLayout` over `TLane`/`TCol`. The register readback view `Dreg_wg` uses `TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])`, which maps the first tile axis onto warpgroup threads — row *i* is owned by warpgroup thread *i*. Layout is part of the operation's contract: the MMA reads the SMEM layouts of its operands and the TMEM layout of its accumulator to know how the tile is physically arranged.

**Dispatch — which hardware path executes it.** When more than one lowering exists, dispatch picks. `Tx.gemm_async(..., dispatch="tcgen05", ...)` selects the Blackwell Tensor Core path, so the tile MMA lowers to `tcgen05.mma` writing TMEM. The synchronous `Tx.cta.copy` here lowers to thread copies; later GEMM steps swap that for TMA without changing the surrounding scope or layout. Naming the path explicitly is what lets one kernel shape target different engines.

## How Compilation Works

Wrap the `PrimFunc` in an `IRModule` and compile with `tvm.compile(mod, target=..., tir_pipeline="tirx")`; it runs the TIRx lowering pipeline and returns an `Executable` you call directly:

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

`tir_pipeline="tirx"` selects the TIRx lowering pipeline: `LowerTIRx` resolves each tile primitive against its scope / layout / dispatch contract, then host/device split and finalize produce the launchable module. Compiling inside a `with target:` block also works and lets the kernel pick up the target context.

You can inspect the result at both levels. Read the IR with `.show()` / `.script()`, and read the generated CUDA from the compiled module:

```python
kernel.show()                          # pretty-print the TIRx (TVMScript)
print(kernel.script())                 # ... the same, as a string

# the generated CUDA C source, from the compiled Executable:
print(ex.mod.imports[0].inspect_source())
```

For the full lowering story — the passes, the tile-primitive dispatch, and the host/device split — see {ref}`chap_arch`.

## Where to Go Next

- {ref}`chap_data_layouts` — the tensor layout model (`TileLayout`, named axes, swizzle) that the operand and accumulator placements above are built from.
- {ref}`chap_language_reference` — the full language-feature set: parser utilities, data types, buffers and memory, control flow, and thread synchronization.
- {ref}`chap_gemm_basics` — this kernel as Step 1 of the GEMM optimization path, built up through K-loop accumulation, spatial tiling, TMA, and warp specialization.
