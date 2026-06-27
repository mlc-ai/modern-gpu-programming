(zh_chap_performance)=
# 什么让 Kernel 变快

:::{admonition} 概览
:class: overview

- Roofline 模型给出 kernel 的性能上限。这个上限由内存带宽或计算吞吐决定。
- 算术强度决定适用哪一个上限。它表示每搬运一个字节所完成的有用算术工作量。
- 低算术强度意味着 kernel 是 memory-bound。主要出路是搬运更少字节、更多复用数据、融合操作，或使用更小的 dtype。
- 高算术强度意味着 kernel 可能是 compute-bound。此时主要任务就是让 Tensor Core 保持忙碌。
- 在现代 GPU kernel 中，最主要的杠杆是 overlap。只要依赖图允许，TMA、Tensor Core、epilogue 和 store 就应该同时运行。
:::

kernel 的快慢只有相对于某个上限才有意义。像 330 TFLOP/s 这样的数字本身看起来很大，
但如果放到一个 dense fp16 或 bf16 Tensor Core 工作能维持约 2 PFLOP/s 的 GPU 上，它的含义就完全不同了。
如果没有上限作为参照，就很难判断一个 kernel 是已经接近硬件极限，还是仍然让芯片的大部分能力闲着。

Roofline 模型给出的正是这个上限。它把 kernel 拆成两类基本活动：搬运字节，以及执行算术。
如果 kernel 不能足够快地移动数据，内存带宽就会设定上限。如果 kernel 有足够的数据复用和足够多的算术工作，
计算吞吐就会设定上限。

本章的数字以 NVIDIA B200 作为贯穿示例。沿用 {ref}`zh_chap_background` 中的约定，我们使用便于推理的整数上限：
dense fp16 或 bf16 Tensor Core 吞吐约为 2 PFLOP/s，HBM3e 带宽约为 8 TB/s。
精确值取决于具体设备、时钟、功耗限制和测量设置，因此这里应把它们理解为数量级上限，而不是 datasheet 常数。

## Roofline 模型

每个 kernel 都会移动数据并执行算术。Roofline 模型用这两条路径中更慢的一条来约束 kernel。

compute ceiling 是硬件的最大算术吞吐。对于 B200 上的 Tensor Core GEMM，相关上限就是 Tensor Core 吞吐。
对于标量或 elementwise kernel，相关上限可能是 CUDA core 吞吐，或另一个功能单元。

memory ceiling 是带宽乘以算术强度。如果一个 kernel 每搬运一个字节只做很少算术，内存带宽就会限制性能。
如果它每字节执行很多操作，内存就不太可能是限制因素。

基本的 roofline bound 是：

```text
可达到 FLOP/s <= min(峰值 FLOP/s, 内存带宽 * 算术强度)
```

算术强度是：

```text
算术强度 = 有用 FLOP 数 / 搬运的字节数
```

必须指定内存层级。对于 HBM roofline，这里的字节是 HBM 字节。对于 L2 roofline，它们是 L2 字节。
对于 SMEM roofline，它们是 shared memory 字节。在本章中，默认 roofline 是 HBM roofline。

在 roofline 图中，x 轴是算术强度，单位是 FLOP/byte。y 轴是可达到的性能。memory roof 是一条斜线：

```text
性能 = 带宽 * 算术强度
```

compute roof 是一条水平线：

```text
性能 = 峰值 FLOP/s
```

二者在 ridge point 处相交：

```text
ridge point = 峰值 FLOP/s / 带宽
```

对于这里使用的 B200 近似数字：

```text
ridge point ≈ 2000 TFLOP/s / 8 TB/s
            ≈ 250 FLOP/byte
```

算术强度低于这个值的 kernel，在 HBM roofline 下是 memory-bound。它无法达到 Tensor Core 峰值吞吐，
因为它无法每秒提供足够多的字节来喂饱这么多算术。

算术强度高于这个值的 kernel 可能是 compute-bound。此时，内存流量不再是一阶限制。
剩下的工作，是足够好地驱动计算单元，以接近那条水平 roof。

Roofline 模型真正有用的部分不是图本身，而是它告诉程序员哪个资源正在成为约束。
memory-bound kernel 不会因为数学指令稍微更好就变快。compute-bound kernel 也不会因为省下几个无关紧要的字节就变快。
第一步，是知道 kernel 位于 ridge 的哪一侧。

![包含示例工作负载的 B200 roofline，展示内存上限、计算上限和 ridge 点](../img/roofline.png)

## 常见工作负载的算术强度

算术强度通常首先是算法属性，其次才是实现细节。在编写 kernel 之前，通常就能做一个粗略估计。

### Elementwise 与 Reduction

elementwise kernel（如 GELU）和 reduction 风格的 kernel（如 RMSNorm）会读写大 tensor，
但每个元素只执行少量 FLOP。

它们的算术强度很低，位于 ridge point 的很左侧。这类 kernel 的最佳版本通常试图接近内存带宽 roof，
而不是 Tensor Core compute roof。

对于这些 kernel，重要问题很机械：

```text
加载和存储是否合并访问？
每个字节是否只搬运一次？
这个操作能否与生产者或消费者融合？
dtype 能否更小？
TMA 或向量化访问能否帮上忙？
```

如果没有复用，也没有 fusion 机会，memory roof 就是真正的上限。

### GEMM

GEMM 是相反的情况。它的算术强度会随问题规模增长，因为每个载入的 tile 都可以被复用到许多 multiply-accumulate 操作中。

对于 `M = N = K` 的方阵 fp16 matmul，理想算术强度大约是：

```text
AI ≈ 2N^3 / (3 * 2N^2)
   = N / 3 FLOP/byte
```

这个估计假设 A 和 B 各读一次，C 写一次，beta 为零，片上复用完美，并且没有额外 metadata、padding 或冗余流量。
真实 kernel 搬运的数据会比这个理想模型更多。但这个估计仍然有用。

当 `N = 4096` 时：

```text
AI ≈ 4096 / 3
   ≈ 1365 FLOP/byte
```

这个值远在 B200 约 250 FLOP/byte 的 ridge point 右侧。因此，大型 GEMM 在 HBM roofline 下是 compute-bound。
目标不只是减少 HBM traffic。目标是使用 Tensor Core、持续喂饱它们，并把数据移动与计算 overlap 起来，
从而让 compute roof 变得可达。

这就是为什么 GEMM 虽然有高算术强度，朴素 GEMM 仍然可能很慢。算法允许高性能，但实现可能让 Tensor Core 闲置。

### 注意力

Attention 位于这两个极端之间。它的算术强度取决于序列长度、head dimension、tiling、masking，
以及中间 tensor 是否被 materialize。

标准 attention 的关键问题是 score matrix。如果 kernel 把 score matrix 写入 HBM，随后又读回来，
它就通过内存移动了一个大型中间结果。Flash Attention（{ref}`zh_chap_flash_attention`）通过把相关 tile 留在片上，
避免这次 HBM 往返，从而提高算术强度。

因此，attention 优化一部分是 roofline 问题，一部分是调度问题。算法被改写，让更少字节进入 HBM。
随后调度 kernel，让剩余的数据移动和计算 overlap。

## 当算术强度较低时

如果一个 kernel 位于 ridge 左侧，它就是 memory-bound。Tensor Core 或 CUDA core 可能闲置，
因为瓶颈是字节，而不是算术指令。

有两类应对方式。

第一类是提高算术强度。这条路径杠杆更高，因为它可以把 kernel 推向 compute-bound 区域。

最重要的技术是 fusion。低算术强度的常见来源，是把中间 tensor 写入 HBM，并在下一个操作中立刻读回。
融合 producer 和 consumer 可以把这个中间结果留在寄存器、SMEM 或 TMEM 中。HBM 往返就消失了。

例子包括：

```text
带 elementwise epilogue 的 GEMM
把 normalization 折叠进相邻操作
在不 materialize 完整 score matrix 的情况下计算 attention
```

第二种技术是为了复用而 blocking。如果一个 tile 载入一次，并在被逐出之前使用很多次，每个字节就支撑了更多算术工作。
GEMM 的高算术强度正是来自这种复用。其他工作负载只要存在对 tile 的重复使用，也可以采用同样思想。

第三种技术是减少每个值占用的字节数。从 fp32 换成 fp16、fp8 或 fp4 会减少 traffic，并提高每字节 FLOP。
当格式需要 metadata、scale factor 或额外转换工作时，真实收益会小于原始 dtype 比例。block-scaled fp8 和 fp4 就是这样的例子。
即便如此，更小的 dtype 仍然常常是让 kernel 在 roofline 上向右移动的最直接方式之一。

第二类应对方式，是接受 memory roof，并试图抵达它。有些 kernel 没有足够工作可以 fusion，也没有足够复用可以利用。
纯 copy、简单 elementwise 操作，或对大 tensor 的单遍 reduction，可能在本质上就是 memory-bound。

在这种情况下，目标不是超过 roof，而是饱和它。

这意味着：

```text
每个字节只搬运一次
避免冗余读取
使用合并访问或向量化访问
对规则的大块 tile 使用 TMA
保持足够多未完成的内存请求
算法允许时使用更小的存储 dtype
```

一旦 memory-bound kernel 达到 memory roof，进一步优化计算就没有帮助。想更快，唯一办法是改变算法，让它搬运更少字节。

## 优化阶梯

roofline 说明什么是可能的，但并不说明达到那个上限有多容易。

大型 fp16 GEMM 理论上可能是 compute-bound。这只意味着 HBM roof 不是主要限制，
并不意味着任何实现都能达到 Tensor Core roof。要缩小差距，需要正确的指令、layout、staging、同步和调度。

第三部分的 GEMM kernel 会在 B200 上把这一点展示为一系列步骤（{ref}`zh_chap_gemm_advanced`）。
每一步都保留相同的基本算法，但改变 tile 的计算方式或调度方式。

GEMM 阶梯中第一次测得的大幅跃升，是从 thread-copy tiled 路径移动到 TMA-backed 路径。
TMA 把规则的 GMEM -> SMEM tile 移动从 CTA thread 手中拿走，让 kernel 通过硬件管理的 bulk copy 喂给 Tensor Core。

在第一次跃升之后，主要改进来自 overlap 和调度。TMA 把未来的 tile 带入 shared memory。`tcgen05.mma` 异步运行。
epilogue 排空先前结果。software pipelining 和 warp specialization 安排这些部件，让硬件引擎同时活跃。

也没有规则要求每个中间步骤本身都必须更快。像 warp specialization 这样的步骤，可能会暂时把资源花在一种结构上，
而这种结构不会立刻改善数字。但如果它能启用更简单结构无法表达的后续 overlap，它仍然可能是正确的一步。

![B200 上的 GEMM 优化旅程：从同步分块 baseline，到 TMA、warp 专门化、CTA 集群和多消费者执行的测量点](../img/gemm_perf.png)

## Overlap 是主要杠杆

一旦 GEMM 已经是 compute-bound，并且已经使用 Tensor Core，剩下的差距通常来自 idle time。

一个简单 kernel 可能这样做：

```text
载入 tile k
计算 tile k
存储 tile k
载入 tile k + 1
计算 tile k + 1
存储 tile k + 1
```

这种 schedule 会让硬件闲置。load 运行时，Tensor Core 在等待。Tensor Core 运行时，copy engine 可能闲置。
store 排空时，两者都可能在等待。

pipelined kernel 则试图把彼此独立的阶段一起运行：

```text
载入 tile k + 1
计算 tile k
存储 tile k - 1
```

这就是本书后面使用的 Blackwell kernel 结构背后的核心思想。TMA 处理异步数据移动。`tcgen05.mma` 处理异步 Tensor Core 工作。
epilogue 和 store 处理输出侧。`mbarrier` 对象连接各个阶段，让每个 consumer 只在真正需要数据时等待。

重点不是移除依赖，而是围绕依赖进行调度。tile `k` 的 MMA 必须等 tile `k` 载入后才能开始。
tile `k` 的 epilogue 必须等 tile `k` 的 MMA 完成后才能读取 accumulator。
但 tile `k + 1` 的 load 通常可以在 tile `k` 的 MMA 正在进行时运行，而 tile `k - 1` 的 store 通常也可以同时排空。

这就是为什么后面许多章节会聚焦异步机制：

```text
用 TMA 处理全局内存到共享内存的搬运
用 mbarrier 表示加载完成和资源交接
用 tcgen05 执行异步 Tensor Core 计算
用 TMEM 存放长生命周期的累加器
用 warp specialization 分离生产者和消费者角色
用 cluster 支持更大的协作 tile 和 multicast
```

它们是不同机制，但服务于同一个调度目标：让有用工作同时运行在不止一条硬件路径上。

## Occupancy 与资源压力

overlap 并不是唯一的 latency hiding 机制。更老也更通用的机制是 occupancy。

occupancy 是驻留在一个 SM 上的工作量。如果一个 warp stall，scheduler 可以运行另一个已经 ready 的 warp。
它通过保持一池可用的独立 warp 来隐藏延迟。

occupancy 受每个 SM 的资源限制。主要限制包括寄存器、shared memory、warp slot 和 CTA slot。
如果一个 kernel 每个 thread 使用很多寄存器，或每个 CTA 使用大量 shared memory，它可能 occupancy 很低，
因为 SM 上只能放下少量 CTA 或 warp。

许多现代 Tensor Core kernel 会有意以降低 occupancy 的方式消耗资源。multi-stage shared memory pipeline 会消耗 SMEM。
大型 register fragment 会消耗寄存器。TMEM 分配会消耗 Tensor Memory 容量。warp specialization 可能会为 producer
或 consumer 角色保留整个 warp。

这种取舍是刻意的。与其通过让许多无关 warp 驻留来隐藏延迟，这些 kernel 会在较少数量的驻留 CTA 内部通过显式 overlap
来隐藏延迟。如果 pipeline 能让 TMA、Tensor Core 和 store 保持忙碌，低 occupancy 的 kernel 仍然可以很快。

没有哪种方式总是更好。有些 kernel 需要高 occupancy，因为它们具有不规则内存访问，或显式 overlap 很有限。
另一些 kernel 需要深度 staging 和 specialization，因为那是高效喂饱 Tensor Core 的唯一方式。
正确的问题不是 occupancy 是否很高，而是活跃的硬件单元是否保持忙碌。

## 这对后续章节有什么帮助

本书后续会不断回到同一套诊断问题：

```text
这个 kernel 受哪条 roof 约束？
哪个资源正在成为瓶颈？
什么改动能让 kernel 更接近那条 roof？
```

对于 memory-bound kernel，答案通常是更少字节和更好的带宽使用。这意味着 fusion、coalescing、vectorized access、
适用时使用 TMA，以及更小的 dtype。

对于 compute-bound GEMM，答案是先使用 Tensor Core，然后做 overlap。kernel 必须 stage operand、发射异步 MMA 工作、
保持 pipeline 充满，并在不阻塞计算路径的情况下排空结果。

对于 Flash Attention，第一步是通过把 score 和 probability tile 留在片上来提高算术强度。
之后，它使用与 GEMM 相同的 overlap 工具：tiled data movement、shared memory staging、异步计算，以及谨慎的资源交接。

这给出了一个实用的优化流程：估计算术强度，定位 roof，判断 kernel 是 memory-bound 还是 compute-bound，
然后优化真正设定上限的资源。

如果没有这一步，kernel 优化就会变成猜谜。有了它，每一次修改都有理由：要么提高算术强度，
要么让内存路径更接近带宽峰值，要么减少 compute roof 下的 idle time。
