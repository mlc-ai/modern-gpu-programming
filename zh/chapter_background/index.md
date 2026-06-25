(zh_chap_background)=
# GPU 执行模型

:::{admonition} Overview
:class: overview

- 一个 kernel 会在一套线程层级（thread → warp → warpgroup → CTA → cluster → grid）上运行，并跨越不同的内存空间（寄存器、SMEM、GMEM、TMEM）。
- 计算被划分到 CUDA core 和 Tensor Core；像 TMA 这样的专用引擎负责搬运供它们消费的数据。
- 一个 kernel 本质上是一条 pipeline：它把数据暂存到这些内存空间中，并在彼此独立的计算引擎和数据移动引擎之间交接工作；反复出现的目标，是让这些引擎同时保持忙碌。
:::

要写出高速 GPU 程序，理解硬件本身以及代码如何在硬件上运行非常重要。本章概览 GPU 的执行模型：
执行工作的线程层级，存放和移动数据的内存空间，以及承担重活的计算引擎和数据移动引擎。
我们会先逐一介绍这些部件，然后把它们组合进一条 GEMM pipeline 中，从而看清数据和执行如何流经硬件。
本书后续几乎每一种优化，本质上都是以某种方式在这些相同部件之间安排工作。

现代 GPU 还包含许多专门化的硬件单元。为了先建立一个直观印象，在深入每个部件之前，下面的交互式演示展示了
Blackwell streaming multiprocessor 内部的主要元素。你可以点击各个部分查看细节。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互：Blackwell SM，展示其中的 warp/warpgroup、shared memory、Tensor Memory，以及
Tensor Core 和 TMA 引擎。*

## 执行层级

我们从真正执行工作的线程开始。GPU 并不会把成千上万个线程呈现为一个扁平的池子，而是把它们组织成嵌套层级。
这样做的原因是，协作会同时发生在几个不同尺度上。每一层的存在，都是为了让某个尺度上的协作更廉价。
下图展示了 Blackwell 上的线程层级；你可以点击每一层来高亮它。

```{raw} html
<iframe src="../demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; min-width:900px; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互：点击某一层：thread → warp → warpgroup → CTA → cluster → grid。*

- **Thread**：标量执行单元。每个 thread 都有自己的程序计数器和寄存器，并通过它在所属 warp 内的 lane ID 来标识。
- **Warp**：以 SIMT（*single instruction, multiple threads*）方式执行的 32 个 thread。一个 warp 的各个 lane 会一起发射同一条指令，
  但每个 lane 保留自己的寄存器，也可以被单独 mask 掉；这正是单个 warp 中不同 lane 能够走不同分支的原因。
- **Warpgroup**：四个连续的 warp，也就是 128 个 thread。Hopper 引入 warpgroup，作为发射 warpgroup 级 MMA（`wgmma`）的单位；
  在 Blackwell 上，它又承担了第二个角色：Tensor Memory 访问的协作单位。128 个 thread 会一起把一个 TMEM tile 移入或移出寄存器。
- **CTA**（*Cooperative Thread Array*，也就是 CUDA 所说的 thread block）：硬件调度的基本单位。
  一个 CTA 运行在单个 SM 上，并拥有该 SM 内一块私有的 shared-memory 分配。同一个 SM 上可以同时驻留多个 CTA；
  这种情况下，它们会瓜分该 SM 的 shared-memory 容量。
- **Cluster**：一组相互协作的 CTA，它们可能位于不同的 SM 上。cluster 内的 CTA 可以彼此同步，也可以读写彼此的 shared memory；
  这种能力称为 distributed shared memory。

这些层级值得多停留一下，因为不同于更早的架构，Blackwell 的关键操作**并不全都由同一组线程发射**。
TMA copy 由单个 thread 发起，随后由硬件执行。TMEM 到寄存器的 load 是 warpgroup-distributed 的：
四个 warp 共同协作，每个 warp 移动 TMEM tile 中属于自己的切片。`tcgen05` MMA 由一个被选出的 thread 提交，
而 clustered MMA 会一次跨越两个 CTA。因此，每种操作都有自己的自然粒度；运行该操作的线程集合，
就是我们所说的该操作的 **scope**。scope 是本书反复回到的三个设计元素（scope、layout 和 dispatch）中的第一个。

## 内存空间

这一层级中的线程能跑多快，取决于数据能多快到达它们手中。因此接下来我们看数据住在哪里。
不存在一种既大又快的单一内存；物理规律迫使容量和速度之间做取舍。所以 GPU 提供的不是一种内存，而是多种内存，
每一种都在不同的位置取得这种折中；kernel 的工作，就是让数据流经这些内存空间。每个空间都有自己的容量、
延迟，以及关于谁能访问它的规则。

| 内存 | 所属范围 | 作用 | 说明 |
|--------|-----------|------|-------|
| **Global (GMEM)** | 整个 device | 持久化 tensor 存储 | 大容量 HBM，由所有 SM 共享 |
| **Shared (SMEM)** | 每个 CTA（一个 SM） | tile 暂存 | 低延迟 scratchpad；B200 上最高 228 KB/SM |
| **Tensor Memory (TMEM)** | 每个 CTA | MMA accumulator 存储 | Blackwell 新增；供 `tcgen05` 使用 |
| **Register File (RF)** | 每个 thread | 标量和每线程 tile fragment | 很快；保存 epilogue/临时值 |

按顺序读，这些空间描述了一条路径。本书中几乎每个 kernel 的数据路径都是
**GMEM → SMEM →（compute）→ registers → SMEM → GMEM**；对于 tensor-core kernel，TMEM 位于这条路径中间，
在数学计算运行时保存 accumulator。

在这四者中，**Tensor Memory (TMEM)** 是唯一一个在 Blackwell 之前没有对应物的空间；完整细节会留到
{ref}`zh_chap_tensor_cores`。不过，现在先理解它的动机很有价值。早期 GPU 把大型 MMA accumulator 保存在寄存器中，
而寄存器是稀缺资源，accumulator 会与其他值竞争。Blackwell 则把 `tcgen05` 的 accumulator 输出写入 TMEM：
这是一个 CTA 作用域的二维 scratchpad，每个 CTA 有 128 个 lane，最多 512 个 32-bit column
（这个数组物理上位于 SM 上）。随后 kernel 必须在 epilogue 之前显式地把 TMEM 读回寄存器。
这个额外步骤并不是免费的，它带来的两个后果会贯穿全书。第一，TMEM read 是**显式且 warpgroup-distributed** 的，
由一个 warpgroup 的四个 warp 协作完成。第二，TMEM 不同于寄存器，必须被**显式分配和释放**。

### 跨 cluster 的 Distributed Shared Memory

cluster 是这个层级中唯一一个成员可以跨越多个 SM 的层级；这种可达范围带来了一种其他层级没有的内存能力。
一个 CTA 运行在一个 SM 上，并使用该 SM 的 shared memory，但单个 CTA 的 SMEM 预算有限，而大 tile 通常需要比一个 block
单独能提供的更多 operand 存储，或者更多复用。Hopper 给出的答案是 **thread block cluster**：
一组比独立 block 更紧密协作的 CTA；它们可以一起同步，也可以读写彼此的 shared memory，这种能力称为
**distributed shared memory (DSMEM)**。Blackwell 保留了 cluster，并在此基础上加入动态调度
（{ref}`zh_chap_clc`）和 2-CTA cooperative MMA。

DSMEM 允许一个 CTA 直接寻址并访问另一个 peer CTA 的 shared memory。一个 thread 可以命名 peer 的 SMEM 中的某个位置，
并把一个 tile 从自己的 SMEM 直接 bulk-copy 到对方 SMEM 中；当字节落地后，会触发 completion barrier
（{ref}`zh_chap_async_barriers`）。第三部分中的 2-CTA cluster GEMM 正是建立在这个机制之上：它利用 DSMEM 在一对 CTA
之间共享 operand tile，而不需要把数据绕回 global memory。

下图展示了 CTA cluster 让额外的 DSMEM hop 成为可能；点击某个部分，可以看到每个 CTA 拥有什么，以及 cross-CTA read
发生在哪里。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="A 2-CTA cluster sharing distributed shared memory" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互：一个 2-CTA cluster，其中每个 CTA 拥有 A 的一半和 B 的一半，通过 cluster（DSMEM）读取对方的 B，
两者共同产生一个 256×256 的输出 tile。*

## 计算：CUDA Core 与 Tensor Core

线程以及它们搬运的数据，最终必须在算术单元处相遇。一个 SM 提供的不是一种数学引擎，而是两种不同的数学引擎。
二者之间的分工塑造了几乎每个 kernel 的写法，并且它们扮演互补角色。

- **CUDA core** 是通用 SIMT ALU。它们运行标量和向量指令，用于处理索引算术、elementwise 计算、
  reduction 和控制流，也就是围绕重型矩阵工作的“胶水逻辑”。
- **Tensor Core** 是固定功能单元，在 *tile* 粒度执行 dense matrix multiply-accumulate，
  用一条指令计算 $D = AB + C$。

这种划分之所以重要，是因为 Tensor Core 提供的算术吞吐远高于 CUDA core，FLOP/s 通常高出一个数量级甚至更多。
因此，dense linear algebra（GEMM、convolution 和 attention）只有运行在 Tensor Core 上，才能接近峰值性能。
所以，获得性能在很大程度上就是让这些 Tensor Core 持续有数据可算。不同 GPU 世代之间变化的是 Tensor Core
**如何**被编程，以及它们的结果**落在哪里**。Hopper 引入了异步 warpgroup MMA（`wgmma.mma_async`）；
Blackwell 的第五代 Tensor Core，即 `tcgen05`，把 accumulator 放在 Tensor Memory 中，而不是寄存器中；
我们会用 {ref}`zh_chap_tensor_cores` 专门介绍它。

cluster 以两种方式扩展这些引擎，而这两种方式会在 GEMM 章节中反复出现。**2-CTA cooperative MMA**
让两个 CTA 各自贡献自己的 SMEM operand，共同形成一个更大的 Tensor Core MMA tile。**TMA multicast**
让数据移动引擎的一次 load 同时把同一个 GMEM tile 送到多个 CTA，消除本来由多次独立 load 造成的冗余 global traffic。
二者都建立在前面介绍的 distributed shared memory 之上。

## GEMM 数据 Pipeline

到目前为止，我们已经分别介绍了各个硬件单元。为了看清它们如何协同工作，可以用一条典型的通用矩阵乘法（GEMM）
pipeline 作为例子。下面的交互式演示展示了三阶段 GEMM tile pipeline 中涉及的单元；点击诸如 `tma load`
这样的动作，可以高亮它穿过各硬件单元时所走的数据路径。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互：Blackwell 上的 load → MMA → epilogue pipeline；点击一个动作，追踪它跨硬件单元的数据路径。*

一个 GEMM tile 会流经三个阶段。

1. **Load。** 一个 TMA copy（{ref}`zh_chap_tma`）把 A 或 B operand tile 从 GMEM 流式搬入 SMEM。
   一个 thread 发射这次 copy，并预先记录预计会到达多少字节。当字节落地时，TMA 引擎报告进度；
   只有当所有预期字节都已送达后，completion barrier 才会翻转。
2. **Compute。** 一个 `tcgen05` MMA（{ref}`zh_chap_tensor_cores`）从 SMEM 中读取 operand tile，
   并把乘积累加进一个 TMEM tile。一个被选出的 thread 发射它；数学计算完成后，它会 signal 一个 barrier。
3. **Epilogue。** warpgroup 把 TMEM accumulator 读回寄存器，把结果 cast 成输出 dtype，然后存到 GMEM；
   这通常会先暂存到 SMEM，再发射一次 TMA store。

这样写出来，三个阶段看上去是严格串行的；但慢 kernel 和快 kernel 的全部差异，就在于 **overlap**。
朴素 kernel 确实会按顺序执行这些步骤（load、wait、compute、wait、store），于是每个引擎在等待前一个引擎时都会闲置。
快速 kernel 则把它们 pipeline 起来：Tensor Core 正在计算 tile `k` 时，TMA 引擎已经在获取 tile `k+1`，
epilogue 也正在忙着排空 tile `k-1`，因此三个引擎可以同时保持占用。让三个异步引擎安全地相互交接工作，
正是 barrier 和 phase 模型（{ref}`zh_chap_async_barriers`）的职责；第三部分的 GEMM 阶梯就是建立在这个模型之上。

## 接下来读什么

现在我们已经看过高层图景，可以继续阅读深入解释主要机制的章节：

- {ref}`zh_chap_tensor_cores` 详细解释 `tcgen05` 计算和 Tensor Memory。
- {ref}`zh_chap_tma` 介绍基于 TMA 的异步数据移动。
- {ref}`zh_chap_async_barriers` 介绍用于协调这些引擎的 mbarrier 和 phase 模型。
