(zh_chap_tirx_layout_api)=
# TIRx Layout API

:::{admonition} 概览
:class: overview

- TIRx layout API 会把 {ref}`zh_chap_data_layout` 中的 layout 记法变成编译器对象。主要对象是 `TileLayout`、`SwizzleLayout` 和 `ComposeLayout`。
- `TileLayout` 描述 named hardware axes 上的仿射 placement。它由 shard spec `S[...]`、replica spec `R[...]` 和可选 offset 构成。
- 一个 layout 会把一个逻辑坐标映射到一个或多个物理坐标。`layout.apply()` 会求值这个映射。
- `SwizzleLayout` 描述用于避免 bank conflict 的、基于 XOR 的 shared-memory swizzle。`ComposeLayout` 会把 swizzle 叠加到 tile layout 上。
- `tmem_datapath_layout`、`tcgen05_atom_layout` 和 `wg_local_layout` 等现成 constructor 覆盖了 kernel 中反复出现的硬件 layout。
:::

{ref}`zh_chap_data_layout` 引入了本书通用的记法：一个 tile shape、一组位于 named axes 上的 stride，
以及一个可选 replication term，用来表示被复制而不是被 partition 的值。本章会把这个记法转成编译器使用的 API。

目标是让页面上的记法和 kernel 中的代码看起来几乎一样。当你写下这样的 layout：

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

你不只是在写解释。你正在构造一个可以附着到 buffer 上的 `TileLayout` 对象。
之后，每个接触这个 buffer 的 tile operation 都可以从 layout 中读取它的 placement。
placement 写一次、检查一次，然后由编译器复用。

layout 可以在从 pool 分配时附着，也可以在声明 buffer 时附着：

```python
pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)
```

从那一刻起，buffer 就携带自己的物理 placement。tile operation 不需要重复说明每个元素住在哪里。

layout object 位于同一个模块中：

```python
from tvm.tirx.layout import (
    TileLayout,
    SwizzleLayout,
    ComposeLayout,
    S,
    R,
    laneid,
    warpid,
    tid_in_wg,
    TLane,
    TCol,
    m,
    tcgen05_atom_layout,
    tmem_datapath_layout,
)
```

这个 API 背后有一个核心思想：layout 不必把逻辑索引映射到单个物理地址。
它会把逻辑索引映射到 named axes 上的一组物理坐标。通常情况下，这个集合只有一个元素。
当存在 replication 时，同一个逻辑元素会有多个物理 placement。

这就是为什么 layout model 有三部分：shard、replica 和 offset。shard 放置元素。
replica 把它复制到额外坐标。offset 平移整个 placement。

## 通过例子理解 Layout

下面的例子展示了 API 的基本形状。

TMEM 中的 accumulator 可以写成 TMEM axes 上的直接 placement：

```python
acc = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])
```

这里，逻辑 row 映射到 `TLane`，逻辑 column 映射到 `TCol`。
在 {ref}`zh_chap_tmem` 中，硬件坐标称为 Lane 和 Col。在 TIRx layout 记法中，这些硬件轴写作 `TLane` 和 `TCol`。

block-scaled MMA 的 scale-factor layout 使用 replication：

```python
scale_factor_layout = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
)
```

shard 会在 TMEM 中放置一个 32-row group。replica 以 32 lane 的 stride 把这个 group 重复四次，
因此这个 32-row group 在完整 128-lane TMEM 空间中可见。

tensor-core register fragment 可以分布在 lane 和 warp 上：

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

同一个物理轴可以出现多次。在这个例子中，两个不同 iter 都贡献到 `laneid`。
没有显式轴的 stride 会使用默认 memory axis `m`。

在真实 kernel 中，常见硬件 layout 通常来自 constructor：

```python
acc = tmem_datapath_layout("D", 128, 256)

ld = tcgen05_atom_layout("32x32b", (128, 64), "float32")
```

这些 constructor 返回普通 `TileLayout` 对象。它们只是便利函数，不是另一套机制。
你可以检查返回的 layout，把它与其他 layout compose，或者在 shape 比较特殊时手写底层 `S[...]` 和 `R[...]` 形式。

## 交互式演示

进入机制之前，有一个能动手戳的具体对象会很有帮助。下面的演示允许你选择 preset layout，
编辑 logical shape 和 `S` 或 `R` 项，选择 dtype 和 swizzle mode，并点击一个元素，查看哪个或哪些物理坐标拥有它。

```{raw} html
<p>
  <a class="reference external" href="../_static/tirx-layout-demo/index.html"
     target="_blank" rel="noopener"
     style="display:inline-block; padding:10px 18px; background:#3b82f6;
     color:#fff !important; font-weight:700; border-radius:8px;
     text-decoration:none;">▶ Open the demo full screen ↗</a>
</p>
<iframe id="tirx-layout-demo-frame" src="../_static/tirx-layout-demo/index.html?notitle"
        style="width:100%; height:1040px; border:1px solid #dfe1e6;
        border-radius:10px; margin:10px 0 6px; display:block;"
        title="TIRx interactive layout demo" loading="lazy"></iframe>
<script>
// The demo (viz-base.js) posts its content height; size the iframe to fit so
// there is no inner scrollbar. This demo is responsive (fills the width), so
// only the height follows content.
(function () {
  var f = document.getElementById('tirx-layout-demo-frame');
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    if (f && e.source === f.contentWindow) f.style.height = d.height + 'px';
  });
})();
</script>
```

这个演示很有用，因为 API 的大部分内容就是演示所展示过程的精确版本。
一个逻辑元素进入 layout。layout 把它 flatten，跨自己的 iter split，在 named axes 上累加坐标，
然后在需要时应用 replication。

## TileLayout

`TileLayout` 是主要的 affine layout object。它通常用正文中相同的记法书写：

```python
TileLayout(S[shape : strides])
```

`S` 项是 shard spec。你可以这样读它：取一个这种 shape 的逻辑 tile，并用这些 named axes 上的 strides 放置它。

当一个值需要出现在多个位置时，shard spec 会用 replica spec 扩展：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])
```

也可以加入可选 offset：

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride] + offset)
```

在表面之下，这些部分由 iter 表示。一个 iter 是三元组：

```text
(extent, stride, axis)
```

它描述了在一个 named axis 上的 strided walk。extent 表示这个 iter 有多少个位置。
stride 表示每一步移动多远。axis 表示哪个硬件坐标正在改变。

一个 layout 有三部分。

### Shard

shard，也就是 `D`，是由 `S[...]` 构建的部分。它把逻辑索引 partition 到一个或多个 iter 上，并产生 base physical coordinate。

For example:

```python
S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
```

有四个 shard iter。它们的 extent 是 `8`、`2`、`4` 和 `2`。
它们的 stride 分别把数据放到 `laneid`、`warpid`、再次 `laneid`，以及默认 memory axis `m` 上。

这推广了普通 shape-and-stride 规则。区别在于，stride 附着到 named hardware axes 上，而不是附着到单个 flat address 上。

### Replica

replica，也就是 `R`，描述同一个逻辑元素的额外物理副本。replica iter 与逻辑索引无关。
它们枚举硬件空间中的额外 offset。

For example:

```python
R[2 : 4@warpid]
```

会在 `warpid` 轴上创建两个相隔四个 warp 的副本。

replication 不是为了方便而设的技巧，它描述的是真实硬件行为。有些数据会跨 warp、lane 或 memory region broadcast。
logical-to-physical mapping 自然支持这一点，因为一个逻辑元素可以映射到一组物理坐标。

### Offset

offset，也就是 `O`，是加到每个结果上的固定坐标。

For example:

```python
5@warpid
```

会把整个 placement 在 `warpid` 轴上平移五个单位。

offset 用于把 tile 放到选定 base coordinate、为独占使用保留一段区域，
或描述同一资源中位于另一个 tile 之后开始的 tile。

### 把这些部分组合起来

layout 会按顺序应用这三部分。

首先，shard 计算 base coordinate。然后，replica 把这个 coordinate fan out 成零个或多个额外副本。
最后，offset 平移每个 coordinate。

对于逻辑坐标 `x`，结果是：

```text
L(x) = { D(x) + r + O | r in R }
```

如果没有 replica，`R` 只包含 zero offset，因此结果是 singleton set。如果存在 replica，
结果会为每个 replica position 包含一个 coordinate。

在 TIRx 语法中，一个完整 layout 可以这样写：

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

从左到右读，shard 放置逻辑 tile，replica 在相隔四个 warp ID 的地方创建第二份副本，
offset 把整个 placement 平移到从 `warpid = 5` 开始。

如果 iter 已经被构造成对象，同一个 layout 可以直接构造：

```python
TileLayout.from_iters(shard, replica, offset)
```

大多数用户代码使用 `S[...]` 和 `R[...]` 记法，因为它更接近数学形式。

## Named Axes

layout 中的 axis 不是匿名维度。每个 axis 都命名一个真实硬件坐标，或 compiler-level placement coordinate。

Examples include:

```text
bx, by, bz
cbx, cby, cbz
tx
warpid
laneid
wgid
tid_in_wg
wid_in_wg
m
P, F
Bank
TLane, TCol
```

`bx`、`by` 和 `bz` 等 grid axes 会把工作放置到 CTA 之间。
`cbx`、`cby` 和 `cbz` 等 cluster axes 会把工作放置在 CTA cluster 内部。
`tx`、`warpid`、`laneid`、`tid_in_wg` 和 `wid_in_wg` 等 thread axes 描述 CTA 或 warpgroup 内部的 ownership。
`m` 轴是默认 linear memory axis。`P` 和 `F` 用于二维 scratchpad-style placement。
`Bank` 命名 shared memory bank。`TLane` 和 `TCol` 是 TMEM Lane 和 Col 坐标在 TIRx layout 中的名字。

axis name 是 layout 的一部分。这很重要，因为整数值相同的两个坐标可能表示不同硬件事物。
`1@tx` 不等于 `1@tid_in_wg`。`1@laneid` 不等于 `1@TLane`。layout 会让这些含义保持显式。

## Forward Mapping

求值一个 layout，就是拿一个逻辑坐标并计算它物理上落在哪里。API 方法是：

```python
layout.apply(*coord)
```

对于没有 replication 的 layout，结果是一个 coordinate dictionary。带 replication 时，结果是一组 coordinate dictionary。
coordinate dictionary 会把 axis name 映射到整数位置，例如：

```python
{"laneid": 7, "warpid": 2, "m": 1}
```

求值规则有四步。

首先，按 row-major 顺序 flatten 逻辑坐标。对于逻辑坐标：

```text
x = (x0, x1, ..., xr-1)
```

inside a logical shape:

```text
(S0, S1, ..., Sr-1)
```

flat index 是：

```text
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1
```

第二，把这个 flat index 跨 shard extents split。如果 shard extents 是：

```text
(e0, e1, ..., en-1)
```

那么 split 会产生 components：

```text
c0, c1, ..., cn-1
```

使用 shard extents 上相同的 row-major 顺序。

第三，使用 stride 把每个 component 累加到它的 axis 上。如果 shard iter `k` 的 extent 为 `ek`、
stride 为 `sk`、axis 为 `ak`，那么 component `ck` 贡献：

```text
ck * sk @ ak
```

同一个 axis 上的所有贡献会加在一起。随后再加上 offset。

第四，应用 replica iter。每个 replica iter 都贡献一个与逻辑坐标无关的额外 offset。
如果有多个 replica iter，layout 会枚举所有组合。

这条规则的一个有用后果是，layout 不需要 hard-code 输入 shape。它需要的是逻辑 tile 的总元素数等于 shard extents 的乘积。
一旦满足这一点，flatten 和 splitting 就定义了映射。

## 案例研究：Tensor Core Register Tile

考虑一个逻辑 `(8, 16)` tile，它分布在两个 warp 上，每个 warp 有 32 个 lane。
每个 lane 拥有一个小 register fragment。register slot 由默认 memory axis `m` 表示。

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

取 `(8, 16)` tile 中的一个逻辑元素 `(i, j)`。

row-major flat index 是：

```text
flat = 16 * i + j
```

按 shard extents `(8, 2, 4, 2)` split 得到：

```text
c0 = i
c1 = floor(j / 8)
c2 = floor(j / 2) mod 4
c3 = j mod 2
```

shard contribution 是：

```text
laneid = 4 * c0 + c2
warpid = c1
m      = c3
```

加上 offset `5@warpid` 后，它变成：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5
m      = j mod 2
```

replica 项：

```python
R[2 : 4@warpid]
```

会给 `warpid` 加上 `0` 或 `4`。因此完整映射是：

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5 + 4 * r，其中 r ∈ {0, 1}
m      = j mod 2
```

shard 把 tile 放到 warp 5 和 6 上。replica 随后把它复制到 warp 9 和 10。
因此，同一个逻辑元素会出现在两个 warp 位置。

这个例子说明了为什么模型使用一组物理坐标。replication 并不适合自然地表示为从物理坐标到逻辑坐标的函数；
它更自然地表示为从一个逻辑坐标到多个物理坐标的函数。

## 案例研究：Blackwell Tensor Memory

同一个 layout model 也适用于 memory placement。axis 不必是 thread axis，也可以是 memory axis。

TMEM 通过硬件 Lane 和 Col 坐标寻址。在 TIRx layout 记法中，这些轴写作 `TLane` 和 `TCol`。

考虑这个 layout：

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)
```

如果逻辑 tile shape 是 `(2, 128, 112)`，split components 就是逻辑坐标本身。
对于元素 `(a, l, c)`，映射是：

```text
TLane = l
TCol  = 112 * a + c
```

extent-128 iter 以 stride `1@TLane` 填满 128 个 TMEM Lane row。
extent-2 iter 以 stride `112@TCol`、extent-112 iter 以 stride `1@TCol`，二者共同覆盖 224 个 column：

```text
TCol in [0, 224)
```

224-column span 是有意的。TMEM layout 不必是 2 的幂。block-scaled FP8 GEMM 可能会选择 224-column accumulator，
因为完整 256-column tile 将无法为两个 accumulator stage 加 scale factor 留出足够 TMEM 容量。
layout API 可以直接表达这个 shape。

## Scale Factor Layout

上面的 accumulator layout 是纯 placement。每个逻辑 accumulator 元素映射到一个 TMEM coordinate。
block-scaled MMA 的 scale factor 不同，因为同一个物理 group 可能需要在多个 warp window 中可见。
这正是 replication 变得有用的地方。

一个紧凑 scale-factor layout 可以写作：

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)
```

shard 会在 TMEM 中放置一个 32-row scale-factor group：

```text
TLane = r
TCol  = s
```

对于逻辑 scale coordinate `(r, s)`。

replica 项创建四个相隔 32 lane 的副本：

```text
TLane = r + 32 * q，其中 q ∈ {0, 1, 2, 3}
TCol  = s
```

因此，这个 32-row group 在 TMEM lane 0 到 31、32 到 63、64 到 95、96 到 127 上都可见。
这就是 `warpx4` 广播模式（{ref}`zh_chap_layout_generations`）。
四个 warp-sized TMEM lane window 中的每一个都会看到同一个 scale-factor group。

在完整 block-scaled MMA layout 中，这个 atom 会与 M row 和 K scale-factor group 上的 outer iter 结合。
根据 scale-factor dtype，多个 scale factor 还可能被 pack 到一个 32-bit `TCol` cell 中。
例如，fp8 scale factor 可以把四个值 pack 到一个 32-bit column cell 中。
可选的 stride-zero reuse 和 pipeline-depth iter 随后可以描述跨多个 MMA 的 scale reuse 以及 double buffering。

重要的是，同一个 `TileLayout` model 描述了这两种情况。accumulator 是 TMEM 中的单一 placement。
scale factor 是同一 TMEM address space 中的复制式 placement。

## 现成 Layout

大多数 kernel 不会手写每一种硬件 layout。TIRx 为经常出现的 layout 提供 constructor。

```python
tmem_datapath_layout(datapath, rows, cols)
```

返回由 `tcgen05.mma` 写入的 TMEM accumulator layout。`datapath` 参数选择 row placement pattern。
例如，`"D"` 对应 `M = 128` 的 identity-style placement，而 `"F"` 对应 `M = 64` 的 scattered placement。

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
```

返回由 `tcgen05.ld` 或 `tcgen05.st` atom 移动的 register tile layout。
instruction shape 的例子包括 `.32x32b`、`.16x64b`、`.16x128b` 以及相关形式。
在 DSL 层面，这是一个 warpgroup-distributed tile。lowering 期间，它会变成四条 warp-collective `tcgen05.ld`
或 `tcgen05.st` 指令，每个 warp 一条，每个 warp 处理自己的 32 个 TMEM lane。

```python
wg_local_layout(cols, rows=128)
```

返回 warpgroup-local register tile，通常在 `tid_in_wg` 上每个 thread 一行。

这些 helper 用来避免手写常见硬件映射。它们并不隐藏模型。
每个 helper 都返回普通 `TileLayout`，由上面描述的同一组 `S` 和 `R` 部分构成。

## SwizzleLayout and ComposeLayout

`TileLayout` 是 affine 的。它可以表达 named axes 上的 stride、replication 和 offset。
这足以覆盖许多 placement，包括 thread fragment、TMEM tile 和 compact scale-factor layout。

shared memory swizzle 需要别的东西。用于避免 bank conflict 的 swizzle 不是 affine stride pattern，
而是对线性 shared-memory address 做基于 XOR 的 permutation。

因此，TIRx 把 swizzling 保留为单独的 layout object：

```python
SwizzleLayout(...)
```

并把它与 tile layout compose：

```python
ComposeLayout(swizzle, tile)
```

tile layout 先产生一个 linear memory address。随后 swizzle 对这个地址做 permutation。
把这两层分开，比强行把 XOR permutation 塞进 affine layout model 更清晰。

## 为什么需要 Swizzle

shared memory 被划分成 32 个 bank，每个 bank word 持有 4 字节。
当一次访问中的多个 lane 触及同一个 bank 中的不同地址时，该访问会因为 bank conflict 被串行化。

朴素 row-major tile 会结构性地产生这种 conflict。考虑一个具有 row-major layout 的 `(8, 64)` float16 tile：

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])
```

逻辑元素 `(i, j)` 的 linear element address 是：

```text
m = 64 * i + j
```

每行是 64 个 float16 值，也就是 128 字节。这正好是一整条 shared memory bank line。
如果一个 warp 以固定 `j` 向下读一列，每一步 row 都会前进一整条 128-byte line。
bank index 会重复，因此 column read 会跨多个 row 坍缩到同一个 bank 上。

swizzle 通过让低地址位依赖更高的 row bit 来改变这一点。
原本会反复落在同一个 bank 上的一列，会被分散到不同 bank 上。

## Swizzle Transform

`SwizzleLayout` 由三个整数参数控制：

```text
per_element = M
swizzle_len = B
atom_len    = S
```

输入是一个 linear element address `m`。

`m` 的低 `M` 位保持不变。这会保留一个小的 contiguous element group。
更高位会右移到一个临时值中：

```text
x = m >> M
```

然后，`x` 中位置 `[S, S + B)` 的 bit group 会 XOR 到 `x` 中位置 `[0, B)` 的 bit group 上。
swizzled address 随后通过把未改变的低 `M` 位放回去形成。

等价地：

```text
mask = (1 << B) - 1

low  = m & ((1 << M) - 1)
x    = m >> M
x2   = x ^ ((x >> S) & mask)

addr = (x2 << M) | low
```

为了让 layout well formed，`S` 必须至少为 `B`。

这个 transform 的目的不是改变 tile 中有哪些逻辑元素，而是改变这些元素在 shared memory 中落在哪里。
MMA 仍然读取同一个逻辑 tile。swizzle 让物理 bank pattern 更好。

## 选择 Swizzle 参数

正常使用中，swizzle 参数由 dtype 和 shared-memory swizzle mode 选择。
常见 mode 是 32-byte、64-byte 和 128-byte swizzle。

`per_element` 参数的选择会让一个小的 vector-sized group 保持 contiguous。对于 float16，一个 16-byte vector 包含 8 个元素，因此：

```text
M = log2(8) = 3
```

使用 128-byte swizzle 时，layout 使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

这会保持 16-byte vector group 完整，同时仍然足够置换更大的 shared-memory address pattern，以打破 column bank conflict。

大多数代码不应该手工推导这些参数。dtype 和 descriptor mode 通常会决定它们。
对程序员来说，重要的是 TIRx layout 中的 swizzle、TMA descriptor 和 MMA expectation 三者匹配。

因此，一个 swizzled shared memory allocation 看起来像这样：

```python
tile = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

layout = ComposeLayout(swizzle, tile)
```

composed layout 会被附着到 shared memory buffer 上。

## 元素的 Bank 与 Line

要判断 swizzle 是否有帮助，可以把 swizzled element address 转回 shared memory bank。

令 `addr` 为 swizzled element address，`b` 为元素大小（字节）。byte address 是：

```text
byte = addr * b
```

bank 是：

```text
bank = floor(byte / 4) mod 32
```

128-byte bank line 是：

```text
line = floor(byte / 128)
```

对于 float16，`b = 2`，因此 bank 公式变成：

```text
bank = floor(addr / 2) mod 32
```

这是下面 worked example 中使用的公式。

## Worked Example：`(8, 64)` float16 Tile 上的 128B Swizzle

回到 row-major float16 tile：

```text
m = 64 * i + j
```

使用：

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

transform 变成：

```text
x    = m >> 3
addr = ((x ^ ((x >> 3) & 7)) << 3) | (m & 7)
```

由于：

```text
m = 64 * i + j
```

我们可以写成：

```text
q = floor(j / 8)
r = j mod 8
```

swizzled address 是：

```text
addr = 64 * i + 8 * (q xor i) + r
```

现在看 column `j = 0`。此时 `q = 0` 且 `r = 0`，所以：

```text
addr = 72 * i
```

对于 float16，bank 是：

```text
bank = floor(addr / 2) mod 32
```

因此八个 row 映射到：

```text
i = 0: bank 0
i = 1: bank 4
i = 2: bank 8
i = 3: bank 12
i = 4: bank 16
i = 5: bank 20
i = 6: bank 24
i = 7: bank 28
```

这一列现在触及八个不同 bank。conflict 消失了。

如果没有 swizzling，同一列的 address 是：

```text
m = 64 * i
```

因此：

```text
bank = floor(64 * i / 2) mod 32 = 0
```

每一行都落在 bank 0 上，因此访问会被串行化。swizzle 只改变物理 placement，
但这已经足够把 column access 变成 conflict-free。

这个保证依赖于按设计方式使用 swizzle。dtype、swizzle width 和 access shape 必须匹配 TMA 与 MMA descriptor mode。
128-byte float16 swizzle 是围绕相关 16-byte row chunk 和 Tensor Core access pattern 设计的。
它并不承诺任意 shared memory access 都会变成 conflict-free。
本章开头的演示会让这一点可见：选择 dtype 和 swizzle mode，观察没有 swizzle 时一列如何坍缩到一个 bank 上，
再观察应用匹配 swizzle 后它如何散布到 bank 视图中。

## 设计理由

layout API 遵循三项设计选择。

第一，它支持一般 shape。硬件 tile 不总是 2 的幂。global tensor、shared memory stage、TMEM accumulator
和 scale-factor buffer 的 shape，往往来自容量限制或算法选择。layout model 把这些 shape 视为普通情况。

第二，映射方向是从逻辑坐标到物理坐标。这个方向很重要，因为 replication 很常见。
一个逻辑元素可能住在多个物理位置。logical-to-physical map 会直接把它表示为一组坐标。

第三，hardware axes 是显式的。layout 不使用匿名维度，也不依赖稍后的上下文来解释它们。
`tx`、`tid_in_wg`、`laneid`、`warpid`、`TLane` 和 `TCol` 之间的差异，会写进 layout 本身。

legality 和 feasibility check 并不只是 layout object 的职责。layout 可以说明数据放在哪里。
更高层的 tile primitive 会决定某个给定操作能否合法且高效地使用这个 placement。
这种分离让 layout API 保持小巧，同时仍然给编译器足够信息来 dispatch 真实硬件操作。
