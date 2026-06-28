(chap_background)=
# GPU 执行模型

:::{admonition} 概览
:class: overview

- GPU kernel 的执行首先由线程层级决定：thread、warp、warpgroup、CTA、cluster 和 grid 分别对应不同的协作粒度。Blackwell 上的很多操作都有自己的天然 scope，例如 TMA copy 由单个 thread 发起，TMEM load 由 warpgroup 协作完成，而 2-CTA cooperative MMA 会跨越两个 CTA。
- 数据不会只停留在一个地方。GMEM、SMEM、TMEM 和寄存器分别服务于不同的容量、延迟和访问范围；cluster 还通过 DSMEM 让一个 CTA 可以访问另一个 CTA 的 shared memory。高性能 kernel 的核心任务之一，就是安排数据在这些空间之间高效移动。
- 计算和数据搬运由不同硬件引擎共同完成。CUDA Core 处理地址计算、控制流和标量逻辑，Tensor Core 执行主要矩阵计算，TMA 负责异步搬运数据。最后我们会用一条 GEMM 数据流水线把这些部件串起来，说明高性能 kernel 如何通过 overlap 让多个引擎同时保持忙碌。


:::

想要写出高性能 GPU kernel，首先要理解一次 kernel 执行时，线程如何被组织，数据放在哪里，以及不同硬件引擎如何协同工作。本章会围绕这三件事展开：先介绍 GPU 的线程层级，再介绍保存和搬运数据的内存空间，最后介绍承担计算和数据搬运的硬件引擎。随后，我们会用一条 GEMM 流水线把这些概念串起来，说明数据如何在内存空间之间移动，计算又如何与数据搬运重叠。后续章节中的许多优化，都会建立在这几个基本机制之上。

我们先从 Blackwell SM 的架构开始。下图展示了本章会用到的几个主要硬件单元。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*点击图中组件查看细节：Blackwell SM，包括 warp/warpgroup、共享内存、Tensor Memory，以及 Tensor Core 和 TMA 引擎。*

## 执行层级

GPU 不会把成千上万个线程当作一个扁平的线程集合来管理，而是把它们组织成嵌套的层级结构。每一层都对应一种协作粒度，让相应规模下的线程协作更加高效。下图展示了 Blackwell 上的线程层次结构。

```{raw} html
<iframe src="../demo_zh/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*点击图中组件查看细节：thread → warp → warpgroup → CTA → cluster → grid。*

* **Thread**：最小的执行单位。每个 thread 都有自己的程序计数器和寄存器，并通过它所在 warp 内的 lane ID 来标识。
* **Warp**：以 SIMT（*single instruction, multiple threads*）方式执行的一组 32 个 thread。一个 warp 中的 lane 会一起发出同一条指令，但每个 lane 保留自己的寄存器，也可以被单独 mask 掉。遇到分支时，warp 会通过不同的 mask 分别执行各条路径。
* **Warpgroup**：四个连续的 warp，也就是 128 个 thread。Hopper 架构引入了 warpgroup，作为发起 warpgroup-level MMA（`wgmma`）的单位；在 Blackwell 上，warpgroup 也是访问 Tensor Memory 的协作单位。
* **CTA**（*Cooperative Thread Array*，也就是 CUDA 中的 thread block）：硬件调度的基本单位。一个 CTA 运行在单个 SM 上，并拥有该 SM 中一块属于自己的 shared memory 分配。多个 CTA 可以同时驻留在同一个 SM 上，此时它们共享这个 SM 的 shared memory 总容量。
* **Cluster**：一组可以相互协作的 CTA。一个 cluster 中的 CTA 可以位于不同 SM 上；它们可以彼此同步，也可以通过 distributed shared memory（DSMEM）访问对方的 shared memory。

Blackwell 上的关键操作并不都由同一组线程发起。TMA copy 由单个 thread 发起，然后交给硬件执行；TMEM load 由一个 warpgroup 中的四个 warp 协作完成；`tcgen05` MMA 由一个被选出的 thread 提交；2-CTA cooperative MMA 则跨越两个 CTA。也就是说，每种操作都有自己的协作粒度。后文把执行某个操作所涉及的线程范围称为这个操作的 **scope**；它会和 layout、dispatch 一起，成为分析 kernel 写法的基本工具。


## 内存空间

线程层级决定了计算如何组织，但线程跑得再多，数据跟不上也无法获得高性能。因此，我们接下来讨论数据存放在哪里。GPU 不使用单一内存空间，而是提供多种内存空间，在容量、延迟和访问范围之间做不同取舍。kernel 的工作，就是让数据在这些空间之间高效移动。

| 内存 | 作用范围 | 用途 | 说明 |
|------|----------|------|------|
| **Global（GMEM）** | 整个 device | 存放输入/输出 tensor | 大容量 HBM，由所有 SM 共享 |
| **Shared（SMEM）** | 每个 CTA（一个 SM 内） | Tile 暂存 | 低延迟的临时存储区；B200 上最高可达 228 KB/SM |
| **Tensor Memory（TMEM）** | 每个 CTA | MMA accumulator 存储 | Blackwell 新增；供 `tcgen05` 使用 |
| **Register File（RF）** | 每个 thread | 标量和每个 thread 的 tile fragment | 很快；保存 epilogue/临时值 |

在这几类内存空间中，**Tensor Memory（TMEM）** 是 Blackwell 新引入的一类内存。在 Blackwell 之前，MMA accumulator 通常存放在寄存器中；但寄存器资源有限，大型 MMA 会带来很高的寄存器压力。Blackwell 的 `tcgen05` 改为把 accumulator 写入 TMEM，从而把这部分存储从寄存器中移出来。可以把 TMEM 理解为一个 CTA 作用域的二维临时存储区：每个 CTA 对应 128 个 lane，最多有 512 个 32-bit 列。这个数组在逻辑上归 CTA 使用，但物理上位于 SM 上。由于 accumulator 先写入 TMEM，kernel 在进入 epilogue 前，需要显式地把它读回寄存器。这种设计会带来两个后续章节反复出现的影响。第一，TMEM read 是显式的，并且是 warpgroup-distributed 的：它由一个 warpgroup 中的四个 warp 协作完成。第二，TMEM 不像寄存器那样由编译器自动分配和管理，而是需要程序显式地分配和释放。

### 跨 Cluster 的分布式共享内存

前面介绍的多数层级都限制在单个 SM 内，而 cluster 可以把多个 CTA 组织到不同 SM 上。这样一来，CTA 之间不仅可以同步，还可以跨 SM 访问彼此的 shared memory。这种跨 CTA 的 shared memory 访问能力称为分布式共享内存，也就是 distributed shared memory（DSMEM）。

DSMEM 的作用是避免不必要的 global memory 往返。借助 DSMEM，一个 CTA 可以把自己 SMEM 中的 tile 直接拷贝到同一 cluster 内另一个 CTA 的 SMEM 中，而不需要先写回 GMEM、再由对方重新读取。拷贝完成后，硬件会触发 completion barrier，通知后续计算这份数据已经可用。

第三部分的 2-CTA cluster GEMM 就建立在这个机制之上。两个 CTA 可以通过 DSMEM 共享 operand tile，从而减少 global memory 访问。这里的“共享”不是把两个 CTA 的 SMEM 合并成一块：Asmem 和 Bsmem 仍然分别属于各自的 CTA。DSMEM 提供的是跨 CTA 访问能力，使得 cluster 内的其他 CTA，或者 cta_group=2 的 cooperative MMA，能够读到对方 SMEM 中的数据。

下图展示了一个 2-CTA cluster 中的 DSMEM 访问路径：每个 CTA 仍然拥有自己的 SMEM，但可以通过 DSMEM 读取另一个 CTA 的 SMEM


```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/cta_cluster.html" title="A 2-CTA cluster sharing distributed shared memory" loading="lazy"
        style="width:100%; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*点击图中组件查看细节：一个 2-CTA cluster。每个 CTA 拥有 A 和 B 的一半，并通过 cluster（DSMEM）读取对方的 B；这一对 CTA 最终产生一个 256x256 的输出 tile。*

## 计算核心：CUDA Core 和 Tensor Core

线程层级决定计算如何组织，内存空间决定数据放在哪里；真正执行算术运算的，是 SM 内的计算单元。一个 SM 中主要有两类计算单元：CUDA Core 和 Tensor Core。

* **CUDA Core** 是通用的 SIMT ALU，负责执行标量和向量指令。kernel 中的地址计算、elementwise 运算、reduction 和控制流，大多由 CUDA Core 完成。
* **Tensor Core** 是面向矩阵计算的固定功能单元，以 *tile* 为粒度执行 dense matrix multiply-accumulate，在一条指令中计算 $D = AB + C$。

Tensor Core 的算术吞吐量远高于 CUDA Core，通常可以达到后者 10 倍或更多的 FLOP/s。GEMM、convolution 和 attention 这类 dense linear algebra 计算，只有尽量跑在 Tensor Core 上，才可能接近峰值性能。因此，写出高性能 kernel 的一个核心目标，就是让 Tensor Core 持续工作，而不是因为数据或依赖没准备好而空转。

不同 GPU 架构不仅改变 Tensor Core 的吞吐量，也改变它们的编程方式和 accumulator 的存放位置。Hopper 引入了异步 warpgroup MMA（`wgmma.mma_async`）；Blackwell 的第五代 Tensor Core，也就是 `tcgen05`，则把 accumulator 放入 Tensor Memory，而不是寄存器中。后续章节会专门讨论这一点。

Cluster 在 GEMM 中还会带来两个重要用法。**2-CTA cooperative MMA** 允许两个 CTA 各自提供一部分 SMEM operand，共同发起一个更大的 Tensor Core MMA tile。**TMA multicast** 允许一次 GMEM load 把同一个 tile 送到多个 CTA，避免每个 CTA 分别读取同一份数据造成冗余 global memory traffic。二者都依赖前面介绍的 cluster 和 DSMEM 机制。


## GEMM 数据流水线

前面几节分别介绍了线程层级、内存空间、数据搬运机制和计算单元。现在我们用一条 GEMM 流水线把它们串起来，看这些硬件结构如何协同工作。下图展示了一条三阶段 GEMM tile 流水线中涉及的主要单元。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo_zh/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```

*点击图中组件查看细节：Blackwell 上的 load → MMA → epilogue 流水线。*

单个 GEMM tile 通常会经过三个阶段。

1. **Load。** TMA copy 把 A 或 B 的 operand tile 从 GMEM 搬到 SMEM。一个 thread 发起这次 copy，并记录预计到达的字节数。随着数据写入 SMEM，TMA 引擎会更新进度；只有在所有预期字节都到达后，completion barrier 才会被触发。
2. **Compute。** `tcgen05` MMA 从 SMEM 读取 operand tile，并把乘积累加到 TMEM tile 中。一个被选出的 thread 提交这次 MMA；计算完成后，硬件会向对应的 barrier 发出完成信号。
3. **Epilogue。** Warpgroup 把 TMEM accumulator 读回寄存器，将结果转换成输出 dtype，然后写回 GMEM。这一步通常会先经过 SMEM staging，也可能使用 TMA store 完成最终写回。

把三个阶段这样列出来时，它们看起来像是严格顺序执行的。但慢 kernel 和快 kernel 的关键区别，正在于能否把这些阶段重叠起来。一个朴素的 kernel 会按顺序执行 load、wait、compute、wait、store；这样一来，每个引擎都会在等待前一个阶段完成时空转。高性能 kernel 则会把这些阶段组织成流水线：当 Tensor Core 正在计算第 `k` 个 tile 时，TMA 引擎已经在搬运第 `k+1` 个 tile，epilogue 也在处理第 `k-1` 个 tile 的输出。这样，多个引擎就能在同一时间保持忙碌。如何让这些异步引擎安全地交接工作，正是 barrier 和 phase 模型要解决的问题。第三部分的 GEMM 优化阶梯也正是建立在这一机制之上。
