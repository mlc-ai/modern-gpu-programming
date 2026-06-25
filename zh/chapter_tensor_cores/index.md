(zh_chap_tensor_cores)=
# Tensor Core：`tcgen05`

:::{admonition} 概览
:class: overview

- `tcgen05` 是 Blackwell 的 Tensor Core 指令家族。它的 MMA 指令以协作方式执行 tile matrix-multiply-accumulate 工作，并由一个被选出的 thread commit。
- accumulator 位于 TMEM 中，而不是寄存器中。epilogue 稍后用 `tcgen05.ld` 把它带回寄存器。
- `cta_group::1` 和 `cta_group::2` 控制是一个 CTA 还是两个 CTA 协作完成 MMA。这个选择也会改变 M 维度如何映射到 TMEM。
- block-scaled MMA mode（如 `mxfp8` 和 `nvfp4`）会添加 scale-factor operand。数据 operand 位于 SMEM，而 scale factor 会通过 TMEM stage。
:::

dense linear algebra 是现代 GPU 花费大部分有用工作的地方。普通 CUDA-core matrix multiply 无法接近芯片标称峰值
（{ref}`zh_chap_background`）。快速 GEMM 和 attention kernel 通过以正确的 tile shape、layout 和 synchronization
喂给 Tensor Core，来接近这个峰值。

从 Volta 开始，基本操作在精神上没有改变。Tensor Core 消费 matrix tile，把它们相乘，并累加结果。
代际之间变化的是操作如何发射、operand 如何布局，以及 accumulator 位于哪里。

Blackwell 对最后一点做了重大改变。`tcgen05` 的 accumulator 不再作为长期存在的 register fragment 保存。
它被写入 Tensor Memory，也就是 TMEM（{ref}`zh_chap_tmem`）。这一个变化会影响整个 kernel：
MMA 写入 TMEM，completion 被异步追踪，epilogue 稍后从 TMEM 中 load accumulator，
并把它变回自己用于转换和 store 的 register fragment。

本章聚焦计算指令本身。TMA（{ref}`zh_chap_tma`）负责把 operand 移入 SMEM。
TMEM 负责保存 accumulator 以及某些 scale-factor operand。`tcgen05.mma` 是位于这两次内存移动之间的 Tensor Core 操作。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tcgen05_intro.html" title="tcgen05 and Tensor Memory" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互：`tcgen05` accumulator 行为。切换 A 或 B 的 transpose，选择输出宽度 `N`，并逐步走过 `K` 次迭代，观察 partial sum 如何在 TMEM 中累加。*

## `tcgen05` MMA

`tcgen05` MMA 是 Blackwell Tensor Core 的 matrix-multiply-accumulate 指令。它是一条协作指令：
工作面向一个 warpgroup 执行，在某些模式下还可以涉及同一个 cluster 中的两个 CTA。
这条指令不是由每个 thread 独立发射的；而是由一个被选出的 thread 代表参与组 commit 这个操作。

把 MMA 拆成三个问题会更清楚。

第一个问题是谁协作。普通模式使用一个 CTA，写作 `cta_group::1`。更大的模式使用 cluster 中的两个 CTA，写作 `cta_group::2`。
两种情况下，这条指令都表示对一个 tile 的一次 Tensor Core 操作，而不是一个 thread 的标量操作。

第二个问题是 operand 和结果住在哪里。数据 operand 通常位于 SMEM。某些变体也可以从 TMEM 读取 A operand。
accumulator 写入 TMEM。operand layout 必须匹配 Tensor Core 的期望，包括数据 operand 使用的 swizzled shared-memory layout
（{ref}`zh_chap_data_layout`）。

第三个问题是如何观察 completion。`tcgen05.mma` 是异步的。发射 MMA 并不意味着 multiply-accumulate 已经完成。
指令会在操作 commit 后返回，而 Tensor Core 继续运行。kernel 使用 commit group 和 `mbarrier` 来得知结果何时 ready
（{ref}`zh_chap_async_barriers`）。

正是这种异步行为让 overlap 成为可能。快速 kernel 不会发射 MMA 后立刻 stall 到它结束。
它可以发射 MMA，开始准备后续 tile，并且只在真正需要结果时等待。代价是每一次 handoff 都必须显式。
如果 epilogue 在 MMA completion barrier 触发之前读取 TMEM，那就是读得太早。

## Accumulator 位于 TMEM

在 Ampere 和 Hopper 上，accumulator 以寄存器形式暴露给程序。MMA 产生 per-lane register fragment，
epilogue 直接消费这个 fragment。这很简单，但它把 accumulator 大小绑定到了每个 thread 的寄存器预算上。

Blackwell 打破了这个链接。`tcgen05.mma` 把 accumulator 写入 TMEM，这是 Blackwell 上一个 CTA 作用域的内存空间。
accumulator 可以在整个 compute phase 中留在 TMEM，epilogue 稍后使用 `tcgen05.ld` 把它载回寄存器。

这改变了 kernel 的形态。register fragment 在边界处仍然重要。epilogue 仍然需要寄存器，以便转换、应用 elementwise work，
并存储结果。但长期存在的 accumulator state 不再是 register allocation 问题，而是 TMEM allocation 和 layout 问题
（{ref}`zh_chap_tmem`）。

这就是为什么 `tcgen05` 和 TMEM 必须放在一起理解。MMA 指令决定计算哪个 tile。
TMEM 决定 accumulator 落在哪里。epilogue 必须使用匹配的 load 路径，以自己期望的 register layout 恢复 accumulator。

## `cta_group::1` and `cta_group::2`

`tcgen05` MMA 可以运行在 `cta_group::1` 或 `cta_group::2` 模式。

在 `cta_group::1` 中，一个 CTA 拥有这次 MMA。它的 operand 位于该 CTA 的 SMEM 中，accumulator 写入该 CTA 的 TMEM。

在 `cta_group::2` 中，cluster 中的两个 CTA 协作处理一个 MMA tile。每个 CTA 都有自己的 SMEM 和 TMEM。
accumulator 并不是存储在一个跨越两个 CTA 的物理 TMEM 区域中，而是切分到两个 CTA 上，每个 CTA 持有自己的部分。
偶数 CTA 发射指令，并为这一对 CTA commit completion barrier。

这个选择很重要，因为它改变逻辑 accumulator tile `C(M, N)` 如何映射到 TMEM。
TMEM 有 128 个硬件 Lane row，以及最多 512 个硬件 Col column。在 TIRx layout 记法中，这些轴写作 `TLane` 和 `TCol`。
MMA mode 决定 `C` 的 row 和 column 如何放到这些 TMEM 轴上。

有四种有用情况值得记住。

下图沿用演示中的颜色约定：紫色表示 SMEM operand，橙色表示 TMEM accumulator state，绿色表示 Tensor Core MMA 路径。
CTA 身份通过标签和位置表示，而不是通过改变这些硬件颜色表示。

### `cta_group::1`, `M = 128`

这是最简单的情况。一个 CTA 计算 128-row tile。TMEM 也有 128 个 Lane row。
因此映射是直接的：accumulator 的 row `m` 映射到 Lane `m`，N 维度映射到 TMEM column。

结果填满 128 个 Lane row 和 N 个 Col column。这是 baseline 图景。CTA 在 SMEM 中拥有 A 和 B，
并在自己的 TMEM 中拥有完整 accumulator tile。

![cta_group::1, M=128：行 m 直接映射到 TMEM lane m](../img/mma_cg1_m128.svg)

### `cta_group::1`, `M = 64`

当 `M = 64` 时，accumulator 只有 64 行，但 TMEM 仍然有 128 个 Lane row。
硬件并不会简单地把 row 0 到 63 pack 到 lane 0 到 63。相反，它会把它们以四段 16-row run 的形式分散到 128 个 lane 中。

row 0 到 15 去 lane 0 到 15。row 16 到 31 去 lane 32 到 47。
row 32 到 47 去 lane 64 到 79。row 48 到 63 去 lane 96 到 111。

这会在 lane 16 到 31、48 到 63、80 到 95、112 到 127 留下空隙。这些空隙是有意的。
通过不同的 lane alignment，另一个独立的 `M = 64` MMA 可以占用互补 lane。
这让两个较小的 M tile 能共享 128-lane TMEM 结构，而不会彼此踩踏。

N 维度仍然映射到 TMEM column。不寻常的部分只在于 M row 在 Lane 上的 placement。

![cta_group::1, M=64：四段 16 行 run，lane stride 为 32，为另一个对齐的 M=64 tile 留出空间](../img/mma_cg1_m64.svg)

### `cta_group::2`, `M = 256`

当 M 维度大到一个 CTA 无法自然持有时，MMA 可以使用 `cta_group::2`。对于 `M = 256`，切分很直接：
CTA 0 持有 row 0 到 127，CTA 1 持有 row 128 到 255。

每个 CTA 使用自己的 TMEM Lane row 0 到 127，以及完整 N column。物理上，这是两个独立的 128-row TMEM region，
每个 CTA 一个。逻辑上，它们形成一个 256×N accumulator tile。

每个 CTA 也提供 A 中对应自己 M row 的部分。B 按该模式的要求对两个 CTA 可用。
偶数 CTA 负责发射 MMA，并为这对 CTA commit completion barrier。

这是 {ref}`zh_chap_gemm_advanced` 中 two-CTA cluster GEMM 使用的模式。

![cta_group::2, M=256：M 连续切分到两个 CTA 上，每个 CTA 128 行](../img/mma_cg2_m256.svg)

### `cta_group::2`, `M = 128`

`cta_group::2`、`M = 128` 模式仍然使用两个 CTA，但 M 维度更短。由于总共只有 128 行，
每个 CTA 接收 64 个 M row。

剩余的 lane 容量用于 pack N 维度。在每个 CTA 内，N 的一半占据 lane 0 到 63，
另一半占据 lane 64 到 127。这样即使每个 CTA 只拥有 64 个 M row，也能使用全部 128 个 Lane row。

因此这个切分有两部分。M 在 CTA pair 之间切分，每个 CTA 64 行。
随后 N 在每个 CTA 内部跨 TMEM Lane row 的 lower half 和 upper half 切分。

![cta_group::2, M=128：每个 CTA 64 个 M 行，N 的两半堆叠在下/上半 lane 上](../img/mma_cg2_m128.svg)

在这些模式中，原则相同。`tcgen05.mma` 计算一个逻辑 accumulator tile，但这个 tile 必须放入物理的
128 Lane × 最多 512 Col 的 TMEM 空间。mode 和 M shape 决定这种 placement。
kernel 后续在把 accumulator 读回时，必须使用同一种映射。

对于这里的 kernel，TMEM 中的 accumulator 通常是 f32。这是常见的高精度路径。
它不是唯一可能的 accumulator type。`.kind::f16` 路径可以用 f16 accumulate。

## Operand Placement

对于 dense MMA mode，A 和 B 会在 MMA 运行前准备在 SMEM 中。TMA 负责把 global memory tile 移入 SMEM。
kernel 会把这些 SMEM tile 安排成 Tensor Core 期望的 layout，包括任何必需的 swizzle。

accumulator C 写入 TMEM。这是与早期世代的主要区别。epilogue 不会直接把 accumulator 作为 MMA 指令输出接收。
它必须用 `tcgen05.ld` 从 TMEM 显式 load。

在 `cta_group::1` 中，一个 CTA 提供 operand 并拥有 accumulator。在 `cta_group::2` 中，
每个 CTA 从自己的 SMEM 提供自己一侧的 operand，并拥有 accumulator 中属于自己的 TMEM 部分。
当 A 按 M 切分时，每个 CTA 保留自己 M slice 的 A row。B 按 mode 共享，因为两个 M slice 都要乘以同一个 N×K tile。

阅读 kernel 时，这种分离很重要。SMEM placement 回答 Tensor Core 如何读取 A 和 B。
TMEM placement 回答 accumulator 去哪里。这两个 layout 由 MMA mode 联系起来，但它们不是同一个内存空间，不能互换看待。

## Block-Scaled MMA

dense mode 直接从 SMEM 读取数据 operand，并累加到 TMEM。Block-scaled MMA 增加两个 operand：A 和 B 的 scale-factor tensor。

这用于 `mxfp8` 和 `nvfp4` 这样的极低精度格式。低精度格式很高效，但动态范围很小。
单个 global scale 通常过于粗糙。如果 scale 按最大值选择，小值会损失精度；如果 scale 按小值选择，大值可能 clip。

block scaling 通过给小 K block 分配 scale factor 来修复这个问题。一组连续 K 元素共享一个 scale。
MMA 在概念上用对应 scale 对每个 block 做 dequantize，然后用 accumulator type 累加乘积。

对于 A 和 B，这会引入两个 scale-factor tensor：

```text
SFA(M, SFK)
SFB(N, SFK)
```

其中 `SFK = K / B`，而 `B` 是沿 K 的 block size。

精确 block size 取决于格式。重要的是，scale axis 以更粗粒度跟随 K。
每个 scale factor 描述的是一块 K 值，而不是单个元素，也不是整个矩阵。

数学形式是：

```text
acc += (Aq * scale_a) * (Bq * scale_b)
```

其中 `Aq` 和 `Bq` 是 quantized low-precision value，scale 在 accumulate 前恢复它们的近似幅度。

scale dtype 也很重要。使用 `e8m0` scale 时，每个 scale 实际上是 2 的幂。
使用 `nvfp4` 所采用的 `e4m3` scale 时，scale 是一个小浮点值，可以表示 2 的幂之间的值。

## Scale Factor 位于哪里

block-scaled `tcgen05.mma` 与 dense MMA 有一条重要 placement rule 不同：scale factor 从 TMEM 读取。

数据 operand A 和 B 仍然 stage 在 SMEM 中。scale factor SFA 和 SFB 通过 TMEM stage。
由于 TMA load 到 SMEM，scale factor 通常需要额外一步。kernel 先把它们 load 到 SMEM，
再用 `tcgen05.cp` 从 SMEM copy 到 TMEM。只有当 scale factor 位于 TMEM 中时，block-scaled MMA 才能读取它们。

这给 scale factor 带来了不同于数据 operand 的移动路径：

```text
A, B:     global memory to SMEM, then MMA reads SMEM
SFA, SFB: global memory to SMEM, then tcgen05.cp copies SMEM to TMEM, then MMA reads TMEM
```

scale factor 的 TMEM layout 很紧凑。一个 128-row scale vector 可以 pack 到 32 个 Lane row 中：
lane position 基于 `r % 32`，column 方向基于 `r / 32`。
数据随后可以 broadcast 到读取完整 128 Lane 空间的四个 warp 上（{ref}`zh_chap_layout_generations`）。

这是为什么 TMEM layout 必须显式的好例子。accumulator layout 和 scale-factor layout 都在 TMEM 中，
但它们不是同一个 layout。accumulator 使用 MMA output mapping，scale factor 使用 block-scaled MMA 期望的 compact layout。

## `cta_group::2` 中的 Scale Factor

在 two-CTA 情况下，scale factor 跟随它缩放的数据。

SFA 缩放 A。由于 A 按 M 在 CTA pair 之间切分，SFA 也按 M 切分。每个 CTA 持有与自己 A row 对应的 SFA row。

SFB 缩放 B。由于两个 CTA 都乘以同一个 B tile，SFB 必须对两个 CTA 可见。实践中，这意味着 SFB 会 multicast 到 CTA pair。

这就是 block-scaled cluster GEMM 中常见 load pattern 的来源。SFA 按 CTA load，使用该 CTA 自己 M slice 的 mask。
SFB 会 broadcast 到这一对 CTA，因为两个 CTA 都需要同一组 N-side scale factor。

![块缩放 MMA 放置：A 和 B 在 SMEM 中打包；SFA、SFB 和 C 位于 TMEM，其中 SFA 按 M 跨 CTA 切分，SFB 多播到 CTA 对](../img/mma_block_scaled.svg)

## 保持 MMA Contract 匹配

一个 Blackwell GEMM tile 会经过几条专门化路径。

TMA 把 A 和 B 从 global memory 带入 SMEM。对于 block-scaled mode，它也会把 scale factor 带入 SMEM。
需要时，`tcgen05.cp` 把这些 scale factor 移入 TMEM。`tcgen05.mma` 读取 operand，在 Tensor Core 上异步运行，
并累加到 TMEM 中。completion barrier 告诉 kernel 这个 accumulator 何时 ready。
epilogue 随后用 `tcgen05.ld` 把 accumulator 从 TMEM 载回寄存器，并存储最终输出。

跨越这些路径，kernel 必须保持三个 contract 匹配：SMEM operand layout、TMEM accumulator 或 scale-factor layout，
以及让下一个 consumer 可以安全运行的异步 completion signal。
