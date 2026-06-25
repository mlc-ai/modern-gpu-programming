(zh_chap_tirx_primer)=
# TIRx 简介

:::{admonition} 概览
:class: overview

- TIRx 是一个用于在 IR 层级编写 GPU kernel 的 Python DSL：你会直接命名硬件，但通过结构化 IR 来表达。
- 每个 tile operation 都由三个设计元素控制：*scope*（哪些 thread）、*layout*（tile 位于哪里）和 *dispatch*（哪条硬件路径）。
- 一个可运行的 single-MMA GEMM 会展示三者；本书其余内容就是把这些设计元素扩展到更大规模。
:::

:::{admonition} 运行示例
:class: note

这些示例需要 Blackwell GPU（`sm_100a`，例如 B200）。TIRx compiler 随 Apache TVM wheel 的
`tvm.tirx` 模块发布；请将它与 CUDA build 的 PyTorch 一起安装：

```bash
pip install apache-tvm==0.25.0
```

用 `python -c "import tvm, tvm.tirx; print(tvm.__version__)"` 确认它能 import。
同样的设置可以运行本书中每个可运行示例。
:::

第一部分解释了硬件是什么。要让它计算任何东西，我们还需要一种编程方式。

我们可以直接写 CUDA 或 PTX，许多高速 kernel 也正是这样写的。问题在于，真正决定 kernel 行为的决策在那里很难看清：
哪些 thread 运行某个操作、每个 data tile 位于哪里，以及由哪条硬件路径执行它。
这些选择被埋在 intrinsic 参数、地址算术和约定之中。

TIRx（Tensor IR neXt）是一个 Python DSL，它把这三个决策显式提到台面上：
**scope**（哪些 thread 运行操作）、**layout**（operand tile 位于哪里）和 **dispatch**（哪条硬件路径执行它）。
它仍然直接命名硬件概念，包括 thread、shared/tensor memory、barrier 和 `tcgen05` MMA。
不同之处在于，这些选择现在是结构化 IR，编译器可以 lower、check 和 schedule。

我们不会抽象地介绍这些思想，而是从一个完整 kernel 出发：一个最小 single-MMA GEMM。
我们先让它跑起来，然后再逐行读回去，观察 scope、layout 和 dispatch 各自如何塑造它，以及 kernel 如何被编译。
这个 kernel 依赖的 tensor layout model 会在 {ref}`zh_chap_tirx_layout_api` 中独立展开，
完整语言特性集见 {ref}`zh_chap_language_reference`；这里我们聚焦于一个 kernel 和三个设计元素。

## 第一个 Kernel：Single-MMA GEMM

我们承诺的 kernel 是一个最小 GEMM，缩减到仍然能使用 Tensor Core 的最小版本。它在 K = 64 时计算
`D = A B^T` 的单个 128 x 128 output tile。整个计算从头到尾表达为一个 `Tx.gemm_async` tile operation。
（这个 tile operation 并不映射到单条硬件指令：因为硬件 MMA 的 K-atom 是 16，K=64 的 tile 会 lower 成沿 K 方向前进的一小段
`tcgen05.mma` 指令序列。DSL 的意义恰恰在于我们写 tile，而不是写序列。）
围绕这个操作，kernel 做常规杂务：分配 shared memory（SMEM）和 tensor memory（TMEM），把 A 和 B 从 global copy 到 shared memory，
把 tile MMA 发射到 TMEM accumulator 中，通过寄存器把 accumulator 读回，并存储结果。
虽然它很小，这个 kernel 正是 {ref}`zh_chap_gemm_basics` 中 GEMM 阶梯的 Step 1，那里会完整走读它。

每个 TIRx kernel 都从同一组 import 开始，所以值得先看一次：

```python

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import tma_shared_layout, SwizzleMode
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

我们把 kernel 包在一个小 builder `hgemm_v1(M, N, K)` 中，它接收 problem shape 并返回一个 `PrimFunc`。
对于我们选择的 shape `M=N=128, K=64`，launch 恰好只包含一个 output tile，
这让第一个版本足够简单，可以一次读完：

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

在阅读 kernel 之前，先确保它能工作。我们编译它，并用 torch reference 检查输出。
我们不必写出精确架构：arch（例如 `sm_100a`）会从 device 自动检测，因此 target `"cuda"` 就足够；
`tir_pipeline="tirx"` 用来选择 TIRx lowering pipeline。编译完成后，`ex.mod(...)` 可以直接接收 torch tensor，
中间不需要手动转换。

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

# ex.mod(...) takes torch tensors directly, the same call form used in every chapter.
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")
```

## Scope、Layout、Dispatch

现在 kernel 已经能运行，我们可以读回它，并询问每一行实际上决定了什么。
这样看，整个 kernel 就是围绕三个设计元素作出的一组选择。其中每个操作都回答同三个问题：
*谁*运行它，数据*位于哪里*，以及它*如何*执行；这三个答案正是 scope、layout 和 dispatch。
本节剩余部分会逐一讨论这些设计元素；下面的交互式演示让你看到每个设计元素控制哪些代码行。

```{raw} html
<iframe src="../demo/tirx_dispatch.html" title="TIRx: scope, layout, dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互：点击 Scope / Layout / Dispatch，高亮 kernel 中由该设计元素控制的代码行。*

使用这个演示时，关注三个问题：

- **Scope：谁运行这个操作？** `Tx.cta.copy(...)` 是 CTA-scoped，因此全部 128 个 thread 都会帮助完成 GMEM -> SMEM copy。
  `Tx.gemm_async(...)` 由一个被选出的 thread 发射一次，因为每条 lowered `tcgen05.mma` 指令已经是一次 cooperative MMA launch。
  `Tx.wg.copy_async(...)` 是 warpgroup-scoped，因此 warpgroup 的 128 个 thread 会逐行切分 TMEM readback。
- **Layout：每个 tile 位于哪里？** A 和 B 使用 `tcgen05.mma` 期望的 swizzled SMEM layout。
  accumulator 在 `TLane`/`TCol` layout 下位于 TMEM。register readback view 把 row 映射到 `tid_in_wg`，
  因此每个 warpgroup thread 拥有一个 row fragment。
- **Dispatch：哪条硬件路径执行它？** `Tx.gemm_async(..., dispatch="tcgen05", ...)` 选择 Blackwell Tensor Core 路径。
  copy 操作也有 dispatch 选择：第一个 kernel 使用普通 thread copy，后续 GEMM step 会把这些 copy 换成 TMA，
  而不改变周围的 scope 或 layout。

**可以让你的 agent 试试**：从第一个 kernel 中挑三行：一个 copy、一个 MMA 和一个 TMEM readback。
让它为每行标注 scope、layout 和 dispatch，然后检查答案是否匹配代码中的 guard、buffer layout 和 `dispatch=` 参数。

## 编译如何工作

我们上面已经编译过 kernel 来测试它；现在稍微仔细看一下这一步做了什么。
流程很短：把 `PrimFunc` 包进 `IRModule`，并交给 `tvm.compile(mod, target=..., tir_pipeline="tirx")`。
这会运行 TIRx lowering pipeline，并返回一个可以直接调用的 `Executable`。

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

至少从轮廓上了解 `tir_pipeline="tirx"` 启动了什么，是很有价值的。pipeline 的核心 pass `LowerTIRx`
会根据每个 tile primitive 的 scope / layout / dispatch contract 来解析它：
我们刚刚讨论的三个设计元素正是在这里兑现成指令。之后，常规 host/device split 和 finalize 步骤会产生可 launch 的 module。
如果你愿意，也可以在 `with target:` block 内编译，这让 kernel 能继承外层 target context。

这个流程的一个好性质是，没有什么对你隐藏：结果可以在两个层级检查。
你可以用 `.show()` 或 `.script()` 阅读 IR 本身，也可以直接从 compiled module 读取编译器最终生成的 CUDA C。

```python
kernel.show()                          # pretty-print the TIRx (TVMScript)
print(kernel.script())                 # ... the same, as a string

# the generated CUDA C source, from the compiled Executable:
print(ex.mod.imports[0].inspect_source())
```

这只是一个概览。完整 lowering 故事，包括所有 pass、tile-primitive dispatch 如何解析，以及 host/device split 如何完成，
见 {ref}`zh_chap_arch`。

## 接下来读什么

一个 kernel 已经足够让我们认识 scope、layout 和 dispatch，并看到它们被编译和运行。
三个设计元素中的每一个，以及这个 kernel 本身，都会通向进一步展开的章节：

- {ref}`zh_chap_tirx_layout_api`：tensor layout model（`TileLayout`、named axes、swizzle），上面的 operand 和 accumulator placement 都建立在它之上。如果 layout 这个设计元素在三者中最神秘，请从这里开始。
- {ref}`zh_chap_language_reference`：完整语言特性集，覆盖 parser utilities、data types、buffers and memory、control flow 和 thread synchronization；当你想要完整词汇表，而不只是导览时，可以读这里。
- {ref}`zh_chap_gemm_basics`：把这个 kernel 作为 GEMM 优化路径的 Step 1，并通过 K-loop accumulation、spatial tiling、TMA 和 warp specialization 逐步扩展。如果你想看同样三个设计元素如何扩展到真实 kernel，这是自然的下一站。
