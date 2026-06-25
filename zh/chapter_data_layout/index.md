(zh_chap_data_layout)=
# 数据布局及其记法

:::{admonition} 概览
:class: overview

- *数据布局*把 tensor 的逻辑索引映射到物理位置，并决定 coalescing、bank conflict，以及某个引擎能否读取一个 tile。
- 本书用一种记法书写布局：`S[(shape) : (strides)]`，并配合 named axes（`@laneid`、`@TLane` 等）以及用于 broadcast 或复制数据的 replication 项 `R[...]`。
- Swizzle 是一种对地址做 XOR 重映射的机制，用来消除 shared-memory bank conflict。
:::

同一组数字，如果以不同的物理排列写入内存，在同一块 GPU 上的运行速度可能相差一个数量级。

原因在于，tensor 的逻辑索引并不会说明它的字节实际位于哪里。硬件对这个位置极其敏感：
它决定了 32 个 lane 的 load 是 coalesce 成一次 transaction，还是散成 32 次；
决定了它们的地址是落在不同 memory bank 中，还是碰撞并串行化；
甚至决定了一个 tile 的字节排列是否能被 Tensor Core 读取。

机器学习程序通常用逻辑 shape 来描述 tensor。**数据布局**补上缺失的物理部分：
它说明具有逻辑索引 `(i, j, …)` 的元素住在哪里，是在内存、寄存器，还是某种其他硬件存储中。

本章介绍现代 GPU 编程中出现的主要布局。为了让讨论可控，我们发展出一种紧凑的**记法**，
用它描述机器学习系统会遇到的多种场景。最后我们会讨论 **swizzling**：
它是一种让同一个 tile 的按行访问和按列访问都能同时高效的机制。

## Shape–Stride 模型

在进入 GPU 特有布局之前，值得先从最简单的布局开始，因为本章后面的所有内容都建立在它之上。
从核心上说，一个 layout 只有两部分：一个 **shape**，以及一组匹配的 **strides**。
我们把这对信息写成 `S[(shape) : (strides)]`；要找到某个逻辑索引的位置，只需把该索引与 strides 做点积。
例如，一个 row-major 的 4×4 矩阵可以写成：

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1
```

这不过是经典 shape/stride 模型的一种紧凑写法（也是 CuTe 记法的 row-major 简化版），后续一切都从它构建出来。

事实上，你几乎肯定已经用过这个模型。任何写过 PyTorch 或 NumPy 的人都用过，因为这些库里的 tensor
本质上就是一个 shape，加上一组作用于扁平 storage buffer 的 stride：

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← exactly S[(3, 4) : (4, 1)]
```

一旦你这样看待 tensor，就会明白为什么许多“reshape”操作根本不触碰数据。
它们只是重写 strides，并返回同一份 storage 上的一个 **view**。最清楚的例子是 transpose，也就是 permute：

```python
tt = t.permute(1, 0)               # or t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← strides swapped, no data moved
tt.data_ptr() == t.data_ptr()      # True, same bytes
```

这里，`t.permute(1, 0)` 是同一块内存上的 `S[(4, 3) : (1, 4)]`：
transpose 纯粹是 stride 的变化，没有移动任何一个字节。对 contiguous tensor 做 `reshape` 或 `view` 也是同样故事：
在旧 storage 上给出新的 shape 和新的 strides。（NumPy 的行为完全相同；唯一差别是它的 `.strides`
以字节为单位，而不是以元素为单位。）

GPU 上的 layout 正是这样工作的。本章剩余内容其实都是同一个思想的各种变体：
一个 tile 的映射（无论映射到内存，还是通过稍后介绍的 named axes 映射到 lane 和寄存器）都是固定 buffer 上的一条 stride 规则，
所以重新排列 tile 通常是改变 *layout*，而不是 copy。不过，我们也要小心这种推理的边界。
zero-copy 的故事对于单一线性地址空间上的逻辑 view 非常清晰；但在 GPU 上，只有当新的 view 与既有字节排列和 ownership
安排兼容时才成立。一旦你改变某个元素由哪个 thread 或 register 拥有，或者改变 SMEM swizzle，
通常就需要真实的数据移动：load、store、shuffle、`ldmatrix`、transpose。

## Tile Layout

到目前为止，我们描述的是整个 tensor 的 layout。不过，GPU kernel 很少一次操作整张矩阵；
它们处理更小的 tile，而这些 tile 会由硬件的不同部分 load、transform 和 compute。
好消息是，tiling 并不需要新概念。它仍然只是一个 layout，只是现在多写了几个维度。
把一个 8×8 矩阵切成 2×4 的 tile，就得到一个 4-D layout，其坐标为
`(tile_row, row_in_tile, tile_col, col_in_tile)`，并选择 strides 让每个 tile 保持 contiguous：

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

一个逻辑 `(i, j)` 会先变成 `(i//2, i%2, j//4, j%4)`，然后通过 strides 计算地址。
值得注意的是，这个记法完全不需要特殊的“tile”概念就能表达 tiling：
它仍然是前面的 shape–stride 模型，只是把索引拆成了外层和内层坐标。

下面的交互式可视化展示了逻辑矩阵索引如何被分解成 tile 坐标，然后映射到物理地址。

```{raw} html
<iframe src="../demo/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互：点击一个单元格，查看它的 tiled index 和 address。*

## Named Axes

到目前为止，`S[...]` 中的每个 stride 都表示线性内存中的 offset，而我们也把 address 当成内存中的位置。
但在 GPU 上，数据可以住在不止一个地方：除了内存，一个 tile 也可能分散在 warp lane、thread register，
或者 TMEM lane 和 column 之中。为了统一描述这些情况，我们用 **named axes** 扩展记法。
思路是让每个 stride 系数携带一个轴标签，说明它沿着哪个空间移动：
`@m` 表示普通内存，`@laneid` 表示 warp lane，`@reg` 表示寄存器，`@warpid` 表示 warp，
`@TLane` / `@TCol` 表示 TMEM 坐标。有了这些标签，单个 layout 不仅能描述数据位于内存何处，
还能描述它如何分布在负责操作它的硬件资源上。

一旦显式标出 memory tag，内存中一个 row-major 的 8×16 tile 就只是：

```text
S[(8, 16) : (16@m, 1@m)]
```

当 layout 描述的不是内存中的排列，而是*跨 thread 分布*的数据时，这些 tag 就开始发挥价值。
以 `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]` 为例：它不是指向线性内存，
而是把行和列映射到 lane ID 以及每个 lane 的一个寄存器。这里的 `laneid` 表示 warp 内的 lane index，
即 `thread_index % warp_size`。这正是你会在 {ref}`zh_chap_layout_generations` 中遇到的 tensor-core register fragment。

下面的交互式可视化展示了 layout 如何把 tensor 元素分布到 warp lane 和 per-lane register 上，
而不是把它们放在线性内存中。

```{raw} html
<iframe src="../demo/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互：一个位于 `@laneid` 和 `@reg` 上的 layout；点击一个单元格，查看哪个 lane/register 持有它。*

## Distributed Layout

named axes 之所以有用，是因为它们让我们能在系统的许多层级上统一描述 placement，
甚至包括*跨整个 device* 的 placement。我们刚刚把它们用于单个 GPU 内部的 lane 和 register，
但同一个思想也可以向外延伸：像 `@gpuid_x` 和 `@gpuid_y` 这样的轴可以说明数据位于 GPU mesh 的哪里，
于是这个记法也能捕捉分布式训练和推理中出现的 sharding pattern。
这些轴尚未捕捉到的一件事是 *replication*，也就是数据被复制到不止一个位置。
因此我们加入记法 `R[n : stride]`，其中 `R` 标记 replicated dimension。
例如，`R[2 : 1@gpuid_x]` 描述沿 `@gpuid_x` 轴的 replication。把二者合在一起，
一个表达式就能同时把 tensor shard 到 2×2 GPU mesh 上，并沿一个轴复制它：

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

下面的演示在一个小型 GPU mesh 上展示这种 partition-and-replication 组合模式。
点击任意单元格，可以看到哪个 device 持有它；也可以观察 `@gpuid_x` replication 如何把相同副本放到配对 device 上。
按钮可以在 fully-sharded、shard + replica 和 shard + offset layout 之间切换。

```{raw} html
<iframe src="../demo/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互：一个分布在 2×2 GPU mesh 上的 layout；点击一个单元格，查看哪些 device 持有它。*

### Kernel 内部的 Replication Pattern：TMEM 中的 Scale Factor

我们刚刚为 GPU mesh 引入的 replication dimension `R[...]`，并不只与多个 device 有关。
同一个结构也能描述完全发生在单个 kernel 内部的事情：硬件把数据*跨 lane broadcast*。
Blackwell 的 block-scaled MMA（{ref}`zh_chap_layout_generations`）就是一个很好的例子。
它的 scale factor 位于 TMEM 中，其中一个 128-row scale vector 只存储在 **32 个 TMEM lane** 中：
逻辑行 `r` 会去到 TMEM lane `r % 32`，而 `r // 32` 沿 column 方向展开。
这 32 个已存储的 TMEM lane 随后会**沿 TMEM `TLane` 轴复制**，从 32 个扩展到 128 个 TMEM lane，
让读取 warpgroup 中四个 warp 的每一个，都能在自己的 32-lane TMEM window 中找到一份副本。
这是一种 `warpx4` broadcast，我们用 replication dimension 来书写它。读取本身由这些 warp 的 thread 执行：

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

这会给出四个副本，副本之间相隔 32 个 TMEM lane：TMEM lane `l`、`l+32`、`l+64` 和 `l+96`
都持有同一个 scale。和之前一样，replication dimension 不携带新数据；它只是说“同一个值，位于四个 TMEM-lane 位置上”，
就像刚才 `@gpuid_x` 把一行 broadcast 到 GPU mesh 上一样。

下面的交互式演示把两个步骤放在一起展示：先紧凑 pack 到 32 个 TMEM lane 中，然后通过 `warpx4`
broadcast 到 128 个读取 lane。

```{raw} html
<iframe src="../demo/sf_tmem.html" title="Scale factors in TMEM: packing and warpx4 replication" loading="lazy"
        style="width:100%; min-width:1040px; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互：点击一个 scale factor `SFA[m, sf]`；它会 pack 到 TMEM 的 lane `m mod 32`、column `(m // 32)·4 + sf`，
然后沿 `TLane` 轴通过 `warpx4` broadcast 到四个 lane 副本（`l`、`l+32`、`l+64`、`l+96`），每个 warp 的 32-lane window 各一份。*

每个 column 内部的 byte packing（`scale_vec` 的 1X/2X/4X 模式）以及 `cta_group::2` split
会在 {ref}`zh_chap_layout_generations` 中介绍。

已经熟悉 CuTe 的读者，可以把本章记法理解为它的一个 row-major 变体：
我们在其上扩展了显式的 hardware-named axes，以及专门的 replication 结构。

## Swizzle Layout

本章最后一种 layout 是为了解决一个具体的硬件问题。GPU 上的 shared memory 被组织成多个 memory bank；
当不同 lane 落到不同 bank 上时，访问最快。相反，如果多个 lane 访问的是*同一个* bank 内的不同地址，
硬件别无选择，只能把它们串行化，于是我们就要付出 **bank conflict** 的代价。

在 tensor 程序中，这很难避免，因为内存访问并不是纯线性顺序。处理矩阵时，我们经常需要读取同一个 tile 的行切片和列切片，
这就产生了真实的张力：对按行访问高效的 layout，往往会让按列访问产生 bank conflict；偏向列的 layout 又会伤害行访问。
**Swizzling** 正是为打破这种张力而设计的技术。

swizzle 背后的想法是置换地址映射，通常做法是把 column index 与 row 做 XOR，
让按行和按列访问最终都分散到多个 bank 上。它提供的 conflict-free 保证是有条件的：
只对匹配的元素宽度、swizzle mode 和访问模式（也就是某个引擎的 descriptor 所期望的模式）成立，
并不适用于任意元素宽度或对齐方式。

下面第一个交互式演示把这一点具体化。点击一个 column index，观察每个元素落到哪个 bank 中：
左侧朴素 row-major tile 中，一列会把全部八个元素汇入同一个 bank，因此 read 会串行化为八个 cycle；
右侧的 XOR-swizzled layout 中，同一列会分散到八个不同 bank，只需一个 cycle 即可读取。

```{raw} html
<iframe src="../demo/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互：一个 8×8 tile；朴素 row-major 中按列访问会产生 bank conflict，XOR swizzle 后则 conflict-free。*

这个小小的 8×8 例子抓住了核心思想，但真实 GPU 内存的 bank 数量远多于这个玩具图所暗示的数量。
为了让 swizzling 在完整尺度上工作，我们不会把整个 tile 当成一个单体对象。相反，我们把内存切成小 segment，
并在每个 segment 内应用 swizzle pattern。实践中最常见的情况是 `SWIZZLE_128B`，
它围绕 128-byte segment 组织，使同样的 row/column-remapping 技巧能够自然适配 32-bank memory system。

下面的交互式演示展示一个具体的硬件 swizzle：`SWIZZLE_128B`。这样在推广到多种格式之前，
你可以先看到逐 segment 重复的 pattern。

```{raw} html
<iframe src="../demo/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互：128-byte segment 内部的 `SWIZZLE_128B` pattern；逐步查看 read cycle，观察 `physical_sector = logical_sector XOR row` 如何把每一列分散到不同 bank。*

同一个想法可以扩展到这个 128-byte 情况之外。为了简化可视化，接下来我们会用一个单色块表示一个 segment，
而不是画出每个 bank。一般来说，硬件会定义一个小的重复 **atom**，permutation 会应用在这个 atom 上；
不同 swizzle mode 会选择不同 atom 大小。`SWIZZLE_128B` 使用 8 × 128 B atom，
`SWIZZLE_64B` 使用 8 × 64 B atom，`SWIZZLE_32B` 使用 8 × 32 B atom；
随后整个 tile 会由当前使用的 atom 平铺而成。

最后一个交互式演示允许你在这些格式之间切换（包括 16 B interleaved mode）、选择数据类型，
并悬停任意单元格，直接检查一个 atom 内部的元素排列。对于推理某条 load/store 指令期望哪种 swizzle，
这正是合适的细节层级。

```{raw} html
<iframe src="../demo/swizzle_atom_general.html" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*交互：选择一种 swizzle format（以及数据类型），查看它的 atom shape（8 × N B）；悬停某个单元格，查看其中元素如何被置换。*

应该选择哪种模式？经验法则是优先选择 tile 能填满的*最大* atom。一个 N-byte atom 要求 tile 的 contiguous dimension
至少为 N 字节，并且是 N 的倍数。因此，`SWIZZLE_128B` 只在一行跨度至少为 128 字节，
也就是 64 个 `float16` 元素时适用。如果能适配，它就是默认选择，因为它的 8 × 128 B atom 覆盖完整的 128-byte bank line，
从而一次把一列分散到全部 32 个 bank 上，在 fp16 中可以同时对 8 行和 8 列提供 conflict-free 访问。
不过，当问题 shape 迫使 contiguous dimension 较小时，tile 就无法再填满 128 B atom；
此时你会降到 `SWIZZLE_64B` 或 `SWIZZLE_32B`，也就是该行仍然能覆盖的最大 atom。

你永远不需要手工算出这些置换后的地址；但有必要精确说明 swizzle 与 `S[...]` 记法之间的关系：
它*不是*那个 affine map 的一部分，而是叠加在其上的一个独立、非仿射层。`S[...]` layout 把元素放到线性内存
（`@m`）地址上，随后 swizzle 置换该地址。在 TIRx layout API 中，这写作
`ComposeLayout(swizzle, tile)`（{ref}`zh_chap_tirx_layout_api`）。你的任务只是为每个会接触这个 tile 的 op
选择一种一致的模式，然后让 composed layout 完成其余工作。

硬件填充的也是这个 composed layout，这正是 swizzling 和 tiling 汇合的地方。TMA descriptor 是多维的，
所以单个三维 box 可以同时描述 tile 的 atom tiling 以及每个 atom 内的 swizzle；
一次 TMA load 随后会按 atom 布置 tile，并在写入 shared memory 时完成 swizzle（{ref}`zh_chap_tma`），
不需要单独的 swizzling pass。每个引擎要求*哪一种* swizzle 是 generation-specific 的，这正是下一章的主题。
