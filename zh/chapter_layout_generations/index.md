(zh_chap_layout_generations)=
# 跨 GPU 世代的 Tensor Core Operand Layout

:::{admonition} 概览
:class: overview

- 在 Ampere、Hopper 和 Blackwell 中，Tensor Core 执行的高层操作仍然相同：`D = A B + C`。
- 代际之间变化的是 operand 如何到达 Tensor Core、支持哪些 tile shape 和 dtype，以及 accumulator 位于哪里。
- Ampere 使用 warp 级 register fragment。shared memory tile 通过 `ldmatrix` 载入 fragment，accumulator 保留在寄存器中。
- Hopper 允许 `wgmma` 通过 matrix descriptor 直接从 shared memory 读取 operand。descriptor 会指定 Tensor Core 所期望的 shared-memory swizzle format。
- Blackwell 保留 shared-memory operand 路径，但把 accumulator 移入 TMEM。Block-scaled MMA 也通过 TMEM 暂存 scale factor。
- 有两个内存约束贯穿所有世代：global memory coalescing 和 shared memory bank conflict。
:::

从远处看，Tensor Core 操作似乎很稳定。它把 A 和 B 的 tile 相乘，加上 accumulator C，并产生 D。
从 Volta 开始，这个形式就没有变过。

但围绕这个操作的细节并没有固定不变。某一代上很快的 kernel，到了下一代可能会变慢。
使用错误 layout 的 kernel 也可能算出错误答案，即使逻辑数学仍然写作 `D = A B + C`。
原因在于 Tensor Core 消费的不是抽象矩阵，而是以非常具体的硬件 layout 排列的 operand。

本章沿着三代 GPU 追踪这个 layout contract。Ampere 通过 warp-level register fragment 暴露 Tensor Core。
Hopper 把输入 operand 移到 shared memory descriptor。Blackwell 保留 shared memory operand，
但把 accumulator 移入 TMEM。操作仍然是 matrix-multiply-accumulate，但进入和离开 Tensor Core 的路径每一代都在改变。

{ref}`Data Layout <zh_chap_data_layout>` 章节中的 layout 记法，是我们描述这些 contract 的语言。
Blackwell 的 TMEM 细节会在 {ref}`zh_chap_tmem` 中单独介绍。

## 两个从未消失的约束

在 Tensor Core 参与之前，两个普通内存约束就已经在塑造 GPU kernel 的 layout。

第一个是 global memory coalescing。当一个 warp 的 32 个 lane 发射 global memory load 时，
内存系统希望这些地址落入少量连续且对齐的 memory segment。如果地址分散，warp load 就会变成多次 memory transaction。
同样的逻辑数据移动会消耗更多带宽和更多时间。

第二个是 shared memory bank conflict。shared memory 被划分成 32 个 bank。
如果 warp 中的多个 lane 访问不同地址，但这些地址映射到同一个 bank，这些访问就无法同时被服务，硬件会把它们串行化。
因此，一个看起来只是平坦 shared memory array、似乎无害的 layout，可能会因为 bank pattern 而变慢。

swizzling 通常是修复 shared memory 侧问题的方法。逻辑 tile 保持不变，但物理地址映射会被置换，
让访问 pattern 分散到多个 bank 上，而不是堆叠到同一个 bank。

这两个约束甚至适用于完全不使用 Tensor Core 的 kernel。Tensor Core kernel 还会增加第三个约束：
operand 必须按 Tensor Core 指令本身期望的 layout 排列。本章剩余部分讨论这个第三约束如何在 Ampere、Hopper 和 Blackwell 间变化。

## Ampere：跨 Warp Lane 的 Register Fragment

在 Ampere 级 GPU 上，主要 Tensor Core 指令是 warp-level 的 `mma.sync.aligned.m16n8k*` 系列。
关键事实是这条指令从哪里读写数据：寄存器。

A、B，以及 C 或 D accumulator，都是分布在一个 warp 的 32 个 lane 上的 per-thread register fragment。
shared memory 只是 staging area。在 MMA 运行之前，operand tile 必须从 shared memory 移入该指令所期望的精确 register fragment layout。

数据路径如下：

```text
SMEM -> 寄存器，使用 ldmatrix
寄存器 -> 寄存器，使用 mma.sync
寄存器 -> SMEM，使用普通 store
```

Ampere 的大部分 layout 故事都来自这条路径。kernel 必须以一种能高效载入的形式把 tile 存进 shared memory，
然后使用 `ldmatrix` 产生 `mma.sync` 所需的 register fragment。

## Ampere Tensor Core 期望什么

Ampere Tensor Core 读取由 8×8 subtile unit 构成的 register fragment。这些 unit 是 `ldmatrix` 载入、MMA 消费的单位。

以 fp16 或 bf16 输入、fp32 accumulate 的 `mma.m16n8k16` 作为具体例子。accumulator tile 的 shape 是 `16 by 8`。
它按固定 pattern 分布在 32 个 lane 上。

对于 C 或 D accumulator，lane `l` 持有的 row 是：

```text
l / 4
l / 4 + 8
```

column 是：

```text
2 * (l % 4)
2 * (l % 4) + 1
```

因此每个 lane 拥有四个 fp32 accumulator 值：来自两个 8-row half 的两行，交叉上两个相邻列。
四个连续 lane 覆盖一行的八个 column。

A operand 使用同样的 M 侧 row carve。K 维度分布在 `l % 4` 以及该 lane 持有的寄存器上。
对于 fp16 或 bf16，每个 32-bit 寄存器会 pack 两个 K 值。

B operand 使用匹配的 K placement，并把 N 侧分散到 lane group 和寄存器上。

精确细节会随 instruction shape 和 dtype 变化，但原则固定：Tensor Core 期望某种特定的 per-lane register fragment。
如果值不在这些寄存器的这个 pattern 中，指令就会把错误元素相乘。

在 layout 记法中，m8n8 fragment 就是那种用 named lane axes 写出的 pattern，例如：

```text
S[(8, 4, 2) : (4@laneid, 1@laneid, 1@m)]
```

两个 `laneid` iter 一起描述 row 和 column 片段如何散布到 lane 上，而最后的 `m` 分量描述 per-lane register slot。

## `ldmatrix`：从 Shared Memory 到 Register Fragment

`ldmatrix` 是 Ampere 中连接 shared memory 和 Tensor Core register fragment 的指令。它是一个 warp-collective load。
一条指令会把一个或多个 8×8 的 16-bit matrix 从 shared memory 移入 `mma.sync` 期望的分布式 register layout。

指令形式是：

```text
ldmatrix.sync.aligned.m8n8.x1.shared.b16
ldmatrix.sync.aligned.m8n8.x2.shared.b16
ldmatrix.sync.aligned.m8n8.x4.shared.b16
```

并且可以带一个可选的 `.trans` qualifier。

`.x1`、`.x2` 和 `.x4` 形式分别载入一个、两个或四个 8×8 matrix。row base address 由 lane 提供。
对于 matrix `m` 和 row `r`，base address 来自 lane `m * 8 + r`。
这意味着 `.x1` 使用 lane 0 到 7 提供 row address，`.x2` 使用 lane 0 到 15，`.x4` 使用 lane 0 到 31。

结果会直接落入 MMA fragment。对于基本的 8×8 情况，lane `l` 会收到 Tensor Core 所期望的 row 和 column pair。
如果用普通的 per-lane `ld.shared` 指令循环，就必须手工复现这种 scatter。
`ldmatrix` 则把 shared-memory-to-fragment 的重排作为一条 warp-collective 指令完成。

`.trans` 形式会在 load 时转置每个 8×8 matrix。当 operand 的存储方向与 MMA 指令期望的方向相反时，就会使用它。

![ldmatrix 将一个 8x8 共享内存 tile 载入 warp 寄存器 fragment；Ampere 上的反向路径使用普通存储，专用 stmatrix 指令稍后才在 Hopper 出现](../img/ldstmatrix.svg)

## 把 Ampere Fragment 写回

`mma.sync` 完成后，accumulator 仍然是 register fragment。epilogue 必须把这个 fragment 移出去。

在 Ampere 上，没有 `ldmatrix` 的专用反向指令。kernel 使用普通 per-thread store，
有时在 store 之前配合 warp shuffle 或本地重排，把 accumulator 以有用 layout 写入 shared memory 或 global memory。

这让 Ampere 模型保持简单，但也把许多 layout 工作暴露给 kernel。输入侧使用 `ldmatrix` 创建 fragment。
计算指令读写 register fragment。输出侧则由从这些 fragment 发出的普通 store 处理。

## Ampere 上的 Swizzle

Ampere kernel 已经需要 shared memory swizzle。原因是 shared memory tile 通常以一种访问 pattern 写入，却以另一种 pattern 读取。

假设一个 tile 是从 global memory 按行填充的。row-major layout 会让这种写入 coalesced 且 bank-friendly。
但 `ldmatrix` 后面可能以一种实际沿列或跨 8×8 subtile 行走的 pattern 读取该 tile。
如果使用朴素 row-major layout，这些读取可能堆叠到同一个 shared memory bank 上。

对于一个简单的 `(8, 64)` float16 tile，一行是：

```text
64 * 2 bytes = 128 bytes
```

这刚好是一整条 shared memory bank line。沿固定 column 向下走时，每一行前进 128 字节，所以 bank index 会重复。
八行可能坍缩到同一个 bank 上，造成 8-way conflict。

改成朴素 column-major layout 并不能完整解决问题。它通常只是把 conflict 移到另一种访问上：
row write 变差，而 column-style read 变好。

XOR swizzle 通过让 physical column 依赖 row 来修复这个问题。一个简单版本是：

```text
physical_col = logical_col xor row
```

逻辑 tile 不变。shared memory 中的物理 placement 被置换，使 row-style write 和 Tensor Core read pattern 都能避免 bank conflict。

在 Ampere 上，这种 swizzle 通常通过手写 shared memory index math 表达。后续世代则把它变成硬件引擎使用的 descriptor format 的一部分。

![在朴素 row-major tile 上，行写入会分散到多个 bank，而列读取会在一个 bank 上碰撞；XOR swizzle 在不牺牲合并行写入的情况下，把列读取分散到多个 bank](../img/swizzle_conflict.svg)

## Hopper：`wgmma`、Shared Memory Descriptor 与 Swizzle 格式

Hopper 改变了 Tensor Core 路径的输入侧。它不再要求每个 operand 都用 `ldmatrix` 载入寄存器；
Hopper 的 `wgmma` 可以直接从 shared memory 读取 operand。

B operand 从 shared memory matrix descriptor 读取。A operand 可以从 shared memory descriptor 或寄存器读取，
对应 `.ss` 和 `.rs` 两种形式。

这移除了 SMEM-sourced operand 的显式 `ldmatrix` 步骤，但没有移除 layout requirement。
Tensor Core 仍然期望 operand 以精确的 shared memory format 存储。区别在于，这个 format 现在通过 matrix descriptor 描述给硬件。

## Hopper Tensor Core 期望什么

Hopper shared memory matrix descriptor 是 shared memory 中 matrix tile 的一种紧凑描述。
它告诉 `wgmma` 如何把逻辑 operand coordinate 转成 shared memory address。

descriptor 包含如下字段：

```text
起始地址
主维度偏移
步长维度偏移
swizzle 模式
基址偏移
```

精确解释取决于 operand major mode。对于 K-major tile，一个 stride 沿 K 前进，另一个沿 M 前进。
对于 MN-major tile，二者角色互换。

swizzle mode 是 shared memory descriptor format 之一，例如：

```text
SWIZZLE_NONE
SWIZZLE_32B
SWIZZLE_64B
SWIZZLE_128B
```

swizzle mode 决定两件事。它决定 descriptor 使用的 atom shape，也决定应用在该 atom 内部的 XOR permutation。
例如，128-byte swizzle mode 会把 operand 看作由 8-row × 128-byte atom 组成的网格，并在每个 atom 内应用 swizzle。

kernel 仍然必须正确放置字节。TMA 通常负责填充 shared memory tile，而 TMA descriptor 必须使用后续 `wgmma` descriptor 所指定的同一种 swizzle format。
如果 TMA 写入的是 128-byte swizzled tile，那么 `wgmma` descriptor 就必须把它作为 128-byte swizzled tile 来读取。
如果 descriptor 和数据不一致，Tensor Core 就会读到错乱的 operand。

这是相对于 Ampere 的主要变化。swizzle 不再只是藏在手写 shared memory indexing 中。
Hopper 把它提升为 first-class descriptor format。写入 tile 的 TMA load 和读取 tile 的 `wgmma` 指令，
都可以命名同一种 format。

![Hopper 共享内存 matrix descriptor 把 operand 坐标映射到 swizzled 共享内存 atom：descriptor stride 选择 atom，swizzle 选择 atom 内部的 byte 位置](../img/smem_descriptor.svg)

## Hopper 输出仍然使用寄存器

Hopper 改变了输入路径，但 accumulator 仍然位于寄存器中。

`wgmma` 指令把 accumulator 写入 per-thread register fragment。精确的 fragment 大小和寄存器数量取决于 instruction shape，
例如 `m64nNk16`，其中 N 会改变 accumulator register 数量。但基本思想和 Ampere 一样：epilogue 消费一个 register fragment。

因此 Hopper 有一个 mixed layout model。输入 operand 可以直接来自 shared memory descriptor，swizzle 由硬件描述。
输出 accumulator 仍然是 register layout 问题。

Blackwell 改变了输出侧。

## Blackwell：`tcgen05` 与 TMEM

Blackwell 为数据 operand 保留 shared memory descriptor 这一思想。A 和 B 仍然在 shared memory 中按 Tensor Core 期望的 layout 准备。
某些模式也可以从 TMEM 读取 A operand。

主要变化在 accumulator。`tcgen05.mma` 会把 accumulator 写入 Tensor Memory，也就是 TMEM，
而不是把它保留为长期存在的 register fragment。在 compute phase 中，accumulator 留在 TMEM。
epilogue 随后使用 `tcgen05.ld` 把它载回寄存器。

这把输出 layout 问题从寄存器移动到了 TMEM。kernel 必须分配 TMEM、选择正确的 TMEM layout、等待 MMA 完成，
然后使用匹配的 `tcgen05.ld` 路径，为 epilogue 恢复 accumulator fragment。

`cta_group::1` 和 `cta_group::2` 如何把 accumulator 分配到一个或两个 CTA 上，细节见 {ref}`zh_chap_tensor_cores`。
与早期世代差异最大的 layout，是 block-scaled scale-factor layout。

## TMEM 中的 Scale Factor Layout

block-scaled MMA mode（如 `mxfp8` 和 `nvfp4`）会加入 scale-factor operand。除了 A 和 B，MMA 还会读取：

```text
SFA(M, SFK)
SFB(N, SFK)
```

其中 `SFK` 是 K scale block 的数量。

数据 operand A 和 B 位于 shared memory。scale factor 位于 TMEM。因此它们有不同的数据移动路径。

TMA 从 global memory load 到 shared memory，并不会直接 load 到 TMEM。因此 scale factor 通常分两步移动：

```text
通过 TMA 从全局内存移到共享内存
通过 tcgen05.cp 从共享内存移到 TMEM
```

只有在这次 copy 之后，scale factor 才进入 `tcgen05.mma` 期望读取它们的内存空间。

TMEM scale-factor layout 使用 TMEM 的硬件坐标 Lane 和 Col。在 TIRx layout 记法中，这些轴写作 `TLane` 和 `TCol`。

一个 128-row scale vector 会被压缩到 32-lane group 中，然后复制到 TMEM 的四个 32-lane window 上。
在 layout 记法中，核心 pattern 是：

```text
S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
```

shard 放置 base 32-row group：

```text
TLane = r
TCol  = s
```

replica 项会在 lane offset 0、32、64 和 96 处添加副本：

```text
TLane = r + 32 * q，其中 q ∈ {0, 1, 2, 3}
TCol  = s
```

这就是 `warpx4` broadcast pattern。同一个紧凑 scale-factor group 会在完整的 128-lane TMEM 空间中变得可见。

32-bit `TCol` cell 内部还存在 byte packing。packing 取决于 `scale_vec` mode：

```text
1X：一个 scale 值在整个 32-bit 单元中广播
2X：打包两个 scale 值，且每个都复制一份
4X：打包四个 K-block scale 值
```

![scale_vec byte 打包：1X 在 4-byte cell 中广播一个缩放因子；2X 打包两个缩放因子且各复制一份；4X 打包四个 K-block 缩放因子](../img/sf_scale_vec.svg)

这种 packing 在 Ampere 或 Hopper 中没有直接对应物，因为那些世代没有供 `tcgen05` block-scaled MMA 使用的 TMEM scale-factor operand。

在 `cta_group::2` 中，scale factor 会跟随它所缩放的数据。SFA 缩放 A，因此它按 M 在两个 CTA 之间切分，
匹配每个 CTA 拥有的 A row。SFB 缩放 B，而 B 被计算中的两个 CTA half 共享，因此 SFB 会 multicast 到两个 CTA
（{ref}`zh_chap_tensor_cores`）。

## 反复出现的 Fragment

尽管周围的内存路径在变化，一个结构会不断回归：m8n8 风格的 register fragment。

在 Ampere 上，`ldmatrix` 构建这个 fragment，让 `mma.sync` 能读取它。

在 Hopper 上，`wgmma` 把它的 accumulator 写成 register fragment，供 epilogue 使用。

在 Blackwell 上，accumulator 在 compute 期间位于 TMEM，但在 epilogue 处理并存储它之前，
`tcgen05.ld` 会把它载回 register fragment（{ref}`zh_chap_tmem`）。

因此 fragment 并没有消失，只是角色改变了。早期世代会在整个 compute phase 中把 accumulator 保留在那里。
Blackwell 则主要在 TMEM 和 epilogue 的边界处使用它。

## 贯穿主线

在 Ampere 上，kernel 显式构建 Tensor Core register fragment。shared memory swizzle 主要由 kernel 通过 index math 负责。

在 Hopper 上，Tensor Core 可以通过 matrix descriptor 直接从 shared memory 读取 operand。
swizzle 变成 TMA 和 `wgmma` 共享的 named descriptor format。

在 Blackwell 上，输入侧仍然使用 shared memory operand，但 accumulator 移到 TMEM。
block-scaled MMA 还增加了必须 stage 到 TMEM 中的 scale-factor operand。

descriptor 并不会消除 layout 工作。它们只是把 contract 显式化。kernel 仍然必须确保数据移动路径、memory layout
和 Tensor Core 指令全部一致。写入 swizzled SMEM tile 的 TMA descriptor、读取该 tile 的 MMA descriptor，
以及附着在 buffer 上的 layout，都必须描述同一种物理排列。

如果其中任何一部分不一致，硬件仍然会运行。它只是会读到错误字节，或者读取得很慢。
这就是为什么 layout 不是 Tensor Core kernel 周围的装饰，而是 instruction interface 的一部分。
