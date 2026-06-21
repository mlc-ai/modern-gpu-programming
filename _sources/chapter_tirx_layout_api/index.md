(chap_tirx_layout_api)=
# TIRx Layout API

:::{admonition} Overview
:class: overview

- The TIRx layout API turns the layout notation from {ref}`chap_data_layout` into compiler objects. The main objects are `TileLayout`, `SwizzleLayout`, and `ComposeLayout`.
- `TileLayout` describes affine placement over named hardware axes. It is built from shard specs `S[...]`, replica specs `R[...]`, and optional offsets.
- A layout maps one logical coordinate to one or more physical coordinates. `layout.apply()` evaluates that mapping.
- `SwizzleLayout` describes the XOR-based shared-memory swizzles used to avoid bank conflicts. `ComposeLayout` stacks a swizzle on top of a tile layout.
- Ready-made constructors such as `tmem_datapath_layout`, `tcgen05_atom_layout`, and `wg_local_layout` cover the hardware layouts that appear repeatedly in kernels.
:::

{ref}`chap_data_layout` introduced the notation used throughout this book: a tile shape, a set of strides over named axes, and an optional replication term for values that are copied rather than partitioned. This chapter turns that notation into the API used by the compiler.

The goal is that the notation on the page and the code in the kernel look almost the same. When you write a layout such as:

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

you are not just writing an explanation. You are constructing a `TileLayout` object that can be attached to a buffer. After that, every tile operation that touches the buffer can read its placement from the layout. The placement is written once, checked once, and reused by the compiler.

A layout is attached either when allocating from a pool or when declaring a buffer:

```python
pool.alloc(shape, dtype, layout=layout)

T.decl_buffer(shape, dtype, scope=scope, layout=layout)
```

From that point on, the buffer carries its physical placement. The tile operations do not need to repeat where each element lives.

The layout objects live in one module:

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

There is one central idea behind the API. A layout does not have to map a logical index to a single physical address. It maps a logical index to a set of physical coordinates over named axes. In the usual case that set has one element. When replication is present, the same logical element has several physical placements.

This is why the layout model has three pieces: shard, replica, and offset. The shard places the element. The replica copies it to additional coordinates. The offset shifts the whole placement.

## Layouts by Example

The examples below show the basic shape of the API.

An accumulator in TMEM can be written as a direct placement over the TMEM axes:

```python
acc = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])
```

Here the logical row maps to `TLane`, and the logical column maps to `TCol`. In {ref}`chap_tmem`, the hardware coordinates are called Lane and Col. In the TIRx layout notation, those hardware axes are written as `TLane` and `TCol`.

A block-scaled MMA scale-factor layout uses replication:

```python
scale_factor_layout = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
)
```

The shard places a 32-row group in TMEM. The replica repeats that group four times at a stride of 32 lanes, so the 32-row group is visible across the full 128-lane TMEM space.

A tensor-core register fragment can be distributed across lanes and warps:

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

The same physical axis can appear more than once. In this example, two different iters both contribute to `laneid`. A stride without an explicit axis uses the default memory axis `m`.

In real kernels, common hardware layouts usually come from constructors:

```python
acc = tmem_datapath_layout("D", 128, 256)

ld = tcgen05_atom_layout("32x32b", (128, 64), "float32")
```

These constructors return ordinary `TileLayout` objects. They are conveniences, not a separate mechanism. You can inspect the returned layout, compose it with other layouts, or write the underlying `S[...]` and `R[...]` form by hand when a shape is unusual.

## Interactive Demo

Before the mechanics, it helps to have something concrete to poke at. The demo below lets you choose a preset layout, edit the logical shape and the `S` or `R` terms, choose a dtype and swizzle mode, and click an element to see which physical coordinate or coordinates own it.

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

The demo is useful because most of the API is just a precise version of what the demo shows. A logical element enters the layout. The layout flattens it, splits it across its iters, accumulates coordinates on named axes, and then applies replication if needed.

## TileLayout

A `TileLayout` is the main affine layout object. It is usually written with the same notation used in the text:

```python
TileLayout(S[shape : strides])
```

The `S` term is the shard spec. You can read it as: take a logical tile of this shape and place it using these strides over named axes.

When a value needs to appear in multiple places, the shard spec is extended with a replica spec:

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])
```

An optional offset can also be added:

```python
TileLayout(S[shape : strides] + R[replica_shape : replica_stride] + offset)
```

Under the surface, these pieces are represented by iters. An iter is a triple:

```text
(extent, stride, axis)
```

It describes a strided walk over one named axis. The extent tells how many positions the iter has. The stride tells how far each step moves. The axis tells which hardware coordinate is being changed.

A layout has three parts.

### Shard

The shard, or `D`, is the part built by `S[...]`. It partitions the logical index across one or more iters and produces the base physical coordinate.

For example:

```python
S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
```

has four shard iters. Their extents are `8`, `2`, `4`, and `2`. Their strides place data on `laneid`, `warpid`, `laneid` again, and the default memory axis `m`.

This generalizes the ordinary shape-and-stride rule. The difference is that the strides are attached to named hardware axes instead of to a single flat address.

### Replica

The replica, or `R`, describes additional physical copies of the same logical element. The replica iters are independent of the logical index. They enumerate extra offsets in hardware space.

For example:

```python
R[2 : 4@warpid]
```

creates two copies separated by four warps on the `warpid` axis.

Replication is not a trick for convenience. It describes real hardware behavior. Some data is broadcast across warps, lanes, or memory regions. A logical-to-physical mapping naturally supports that because one logical element can map to a set of physical coordinates.

### Offset

The offset, or `O`, is a fixed coordinate added to every result.

For example:

```python
5@warpid
```

shifts the whole placement by five on the `warpid` axis.

Offsets are used to place a tile at a chosen base coordinate, reserve a region for exclusive use, or describe a tile that starts after another tile in the same resource.

### Putting the Pieces Together

A layout applies these three parts in order.

First, the shard computes the base coordinate. Then the replica fans that coordinate out into zero or more additional copies. Finally, the offset shifts every coordinate.

For a logical coordinate `x`, the result is:

```text
L(x) = { D(x) + r + O | r in R }
```

If there is no replica, `R` contains only the zero offset, so the result is a singleton set. If there is a replica, the result contains one coordinate for each replica position.

In TIRx syntax, a full layout can look like this:

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

Read left to right, the shard places the logical tile, the replica copies it across two warp groups, and the offset shifts the whole placement to start at `warpid = 5`.

If the iters have already been built as objects, the same layout can be constructed directly:

```python
TileLayout.from_iters(shard, replica, offset)
```

Most user code uses the `S[...]` and `R[...]` notation because it is closer to the mathematical form.

## Named Axes

The axes in a layout are not anonymous dimensions. Each axis names a real hardware coordinate or a compiler-level placement coordinate.

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

Grid axes such as `bx`, `by`, and `bz` place work across CTAs. Cluster axes such as `cbx`, `cby`, and `cbz` place work within a CTA cluster. Thread axes such as `tx`, `warpid`, `laneid`, `tid_in_wg`, and `wid_in_wg` describe ownership inside a CTA or warpgroup. The axis `m` is the default linear memory axis. `P` and `F` are used for two-dimensional scratchpad-style placement. `Bank` names shared memory banks. `TLane` and `TCol` are the TIRx layout names for the TMEM Lane and Col coordinates.

The axis name is part of the layout. This matters because two coordinates with the same integer value can mean different hardware things. `1@tx` is not the same as `1@tid_in_wg`. `1@laneid` is not the same as `1@TLane`. The layout keeps those meanings explicit.

## Forward Mapping

Evaluating a layout means taking a logical coordinate and computing where it lands physically. The API method is:

```python
layout.apply(*coord)
```

For a layout without replication, the result is one coordinate dictionary. With replication, the result is a set of coordinate dictionaries. A coordinate dictionary maps axis names to integer positions, such as:

```python
{"laneid": 7, "warpid": 2, "m": 1}
```

The evaluation rule has four steps.

First, flatten the logical coordinate in row-major order. For a logical coordinate:

```text
x = (x0, x1, ..., xr-1)
```

inside a logical shape:

```text
(S0, S1, ..., Sr-1)
```

the flat index is:

```text
flat = x0 * S1 * S2 * ... * Sr-1
     + x1 * S2 * ... * Sr-1
     + ...
     + xr-2 * Sr-1
     + xr-1
```

Second, split that flat index across the shard extents. If the shard extents are:

```text
(e0, e1, ..., en-1)
```

then the split produces components:

```text
c0, c1, ..., cn-1
```

using the same row-major order over the shard extents.

Third, accumulate each component onto its axis using its stride. If shard iter `k` has extent `ek`, stride `sk`, and axis `ak`, then component `ck` contributes:

```text
ck * sk @ ak
```

All contributions to the same axis are added together. The offset is then added.

Fourth, apply the replica iters. Each replica iter contributes an additional offset independent of the logical coordinate. If there are several replica iters, the layout enumerates all combinations.

One useful consequence of this rule is that the layout does not need to hard-code the input shape. What it needs is that the logical tile has the same total number of elements as the product of the shard extents. Once that holds, flattening and splitting define the mapping.

## Case Study: Tensor Core Register Tile

Consider a logical `(8, 16)` tile distributed across two warps of 32 lanes each. Each lane owns a small register fragment. The register slot is represented by the default memory axis `m`.

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

Take a logical element `(i, j)` from the `(8, 16)` tile.

The row-major flat index is:

```text
flat = 16 * i + j
```

Splitting by the shard extents `(8, 2, 4, 2)` gives:

```text
c0 = i
c1 = floor(j / 8)
c2 = floor(j / 2) mod 4
c3 = j mod 2
```

The shard contributions are:

```text
laneid = 4 * c0 + c2
warpid = c1
m      = c3
```

After adding the offset `5@warpid`, this becomes:

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5
m      = j mod 2
```

The replica term:

```python
R[2 : 4@warpid]
```

adds either `0` or `4` to `warpid`. So the full mapping is:

```text
laneid = 4 * i + floor(j / 2) mod 4
warpid = floor(j / 8) + 5 + 4 * r, where r in {0, 1}
m      = j mod 2
```

The shard places the tile on warps 5 and 6. The replica then copies it to warps 9 and 10. The same logical element therefore appears in two warp positions.

This example shows why the model uses a set of physical coordinates. Replication is not naturally represented by a function from physical coordinate to logical coordinate. It is naturally represented by a function from one logical coordinate to several physical coordinates.

## Case Study: Blackwell Tensor Memory

The same layout model works for memory placement. The axes do not have to be thread axes. They can be memory axes.

TMEM is addressed by hardware Lane and Col coordinates. In the TIRx layout notation, those axes are written as `TLane` and `TCol`.

Consider this layout:

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)
```

If the logical tile shape is `(2, 128, 112)`, the split components are just the logical coordinates themselves. For element `(a, l, c)`, the mapping is:

```text
TLane = l
TCol  = 112 * a + c
```

The extent-128 iter with stride `1@TLane` fills the 128 TMEM Lane rows. The extent-2 iter with stride `112@TCol` and the extent-112 iter with stride `1@TCol` together cover 224 columns:

```text
TCol in [0, 224)
```

The 224-column span is intentional. TMEM layouts do not have to be powers of two. A block-scaled FP8 GEMM may choose a 224-column accumulator because a full 256-column tile would not leave enough TMEM capacity for two accumulator stages plus scale factors. The layout API can express that shape directly.

## Scale Factor Layouts

The accumulator layout above is a pure placement. Each logical accumulator element maps to one TMEM coordinate. Scale factors for block-scaled MMA are different because the same physical group may need to be visible across several warp windows. This is where replication becomes useful.

A compact scale-factor layout can be written as:

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]
    + R[4 : 32@TLane]
)
```

The shard places a 32-row scale-factor group in TMEM:

```text
TLane = r
TCol  = s
```

for a logical scale coordinate `(r, s)`.

The replica term creates four copies separated by 32 lanes:

```text
TLane = r + 32 * q, where q in {0, 1, 2, 3}
TCol  = s
```

So the 32-row group is visible at TMEM lanes 0 through 31, 32 through 63, 64 through 95, and 96 through 127. This is the `warpx4` broadcast pattern ({ref}`chap_layout_generations`). Each of the four warp-sized TMEM lane windows sees the same scale-factor group.

In the full block-scaled MMA layout, this atom is combined with outer iters over M rows and K scale-factor groups. Several scale factors may also be packed into one 32-bit `TCol` cell, depending on the scale-factor dtype. For example, fp8 scale factors can pack four values into one 32-bit column cell. Optional stride-zero reuse and pipeline-depth iters can then describe scale reuse across multiple MMAs and double buffering.

The important part is that the same `TileLayout` model describes both cases. The accumulator is a single placement in TMEM. The scale factors are a replicated placement in the same TMEM address space.

## Ready-Made Layouts

Most kernels do not hand-write every hardware layout. TIRx provides constructors for the layouts that appear often.

```python
tmem_datapath_layout(datapath, rows, cols)
```

returns the TMEM accumulator layout written by `tcgen05.mma`. The `datapath` argument selects the row placement pattern. For example, `"D"` corresponds to the `M = 128` identity-style placement, while `"F"` corresponds to the `M = 64` scattered placement.

```python
tcgen05_atom_layout(instr_shape, tensor_shape, dtype)
```

returns the register tile layout moved by a `tcgen05.ld` or `tcgen05.st` atom. Examples of instruction shapes include `.32x32b`, `.16x64b`, `.16x128b`, and related forms. At the DSL level this is a warpgroup-distributed tile. During lowering it becomes four warp-collective `tcgen05.ld` or `tcgen05.st` instructions, one per warp, with each warp handling its own 32 TMEM lanes.

```python
wg_local_layout(cols, rows=128)
```

returns a warpgroup-local register tile, usually with one row per thread on `tid_in_wg`.

These helpers are there to avoid rewriting common hardware mappings by hand. They do not hide the model. Each helper returns an ordinary `TileLayout` built from the same `S` and `R` pieces described above.

## SwizzleLayout and ComposeLayout

A `TileLayout` is affine. It can express strides, replication, and offsets over named axes. That is enough for many placements, including thread fragments, TMEM tiles, and compact scale-factor layouts.

Shared memory swizzles need something else. The swizzle used to avoid bank conflicts is not an affine stride pattern. It is an XOR-based permutation of the linear shared-memory address.

TIRx therefore keeps swizzling as a separate layout object:

```python
SwizzleLayout(...)
```

and composes it with the tile layout:

```python
ComposeLayout(swizzle, tile)
```

The tile layout first produces a linear memory address. The swizzle then permutes that address. Keeping these two layers separate is cleaner than forcing the XOR permutation into the affine layout model.

## Why Swizzle

Shared memory is divided into 32 banks, with each bank word holding 4 bytes. When the lanes of one access touch different addresses in the same bank, the access is serialized by a bank conflict.

A plain row-major tile can create this conflict structurally. Consider an `(8, 64)` float16 tile with row-major layout:

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])
```

The logical element `(i, j)` has linear element address:

```text
m = 64 * i + j
```

Each row is 64 float16 values, or 128 bytes. That is exactly one full shared memory bank line. If a warp reads down a column with fixed `j`, each row step advances by one full 128-byte line. The bank index repeats, so the column read collapses onto the same bank across rows.

The swizzle changes this by making the low address bits depend on higher row bits. A column that would otherwise land repeatedly on the same bank is scattered across different banks.

## The Swizzle Transform

A `SwizzleLayout` is controlled by three integer parameters:

```text
per_element = M
swizzle_len = B
atom_len    = S
```

The input is a linear element address `m`.

The low `M` bits of `m` are left unchanged. This preserves a small contiguous group of elements. The higher bits are shifted down into a temporary value:

```text
x = m >> M
```

Then the bit group at positions `[S, S + B)` of `x` is XORed into the bit group `[0, B)` of `x`. The swizzled address is then formed by putting the unchanged low `M` bits back.

Equivalently:

```text
mask = (1 << B) - 1

low  = m & ((1 << M) - 1)
x    = m >> M
x2   = x ^ ((x >> S) & mask)

addr = (x2 << M) | low
```

For the layout to be well formed, `S` must be at least `B`.

The point of the transform is not to change which logical elements are in the tile. It changes where those elements land in shared memory. The MMA still reads the same logical tile. The swizzle makes the physical bank pattern better.

## Choosing Swizzle Parameters

In normal use, the swizzle parameters are chosen from the dtype and the shared-memory swizzle mode. The common modes are 32-byte, 64-byte, and 128-byte swizzles.

The `per_element` parameter is chosen so that a small vector-sized group stays contiguous. For float16, a 16-byte vector contains 8 elements, so:

```text
M = log2(8) = 3
```

With a 128-byte swizzle, the layout uses:

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

This keeps the 16-byte vector group intact while still permuting the larger shared-memory address pattern enough to break the column bank conflict.

Most code should not derive these parameters by hand. The dtype and descriptor mode usually determine them. What matters for the programmer is that the swizzle in the TIRx layout, the TMA descriptor, and the MMA expectation all match.

A swizzled shared memory allocation therefore looks like:

```python
tile = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

layout = ComposeLayout(swizzle, tile)
```

The composed layout is what gets attached to the shared memory buffer.

## Bank and Line of an Element

To see whether a swizzle helps, translate the swizzled element address back into a shared memory bank.

Let `addr` be the swizzled element address, and let `b` be the element size in bytes. The byte address is:

```text
byte = addr * b
```

The bank is:

```text
bank = floor(byte / 4) mod 32
```

The 128-byte bank line is:

```text
line = floor(byte / 128)
```

For float16, `b = 2`, so the bank formula becomes:

```text
bank = floor(addr / 2) mod 32
```

This is the formula used in the worked example below.

## Worked Example: 128B Swizzle on an `(8, 64)` float16 Tile

Return to the row-major float16 tile:

```text
m = 64 * i + j
```

Use:

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

The transform becomes:

```text
x    = m >> 3
addr = ((x ^ ((x >> 3) & 7)) << 3) | (m & 7)
```

Since:

```text
m = 64 * i + j
```

we can write:

```text
q = floor(j / 8)
r = j mod 8
```

and the swizzled address is:

```text
addr = 64 * i + 8 * (q xor i) + r
```

Now look at column `j = 0`. Then `q = 0` and `r = 0`, so:

```text
addr = 72 * i
```

For float16, the bank is:

```text
bank = floor(addr / 2) mod 32
```

So the eight rows map to:

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

The column now touches eight distinct banks. The conflict is gone.

Without swizzling, the same column has address:

```text
m = 64 * i
```

and therefore:

```text
bank = floor(64 * i / 2) mod 32 = 0
```

Every row lands on bank 0, so the access is serialized. The swizzle changes only the physical placement, but that is enough to turn the column access into a conflict-free one.

This guarantee depends on using the swizzle in the way it was designed. The dtype, swizzle width, and access shape have to match the TMA and MMA descriptor modes. A 128-byte float16 swizzle is designed around the relevant 16-byte row chunks and Tensor Core access pattern. It is not a promise that arbitrary shared memory accesses become conflict free. The demo at the top of this chapter makes this visible: choose a dtype and swizzle mode, and watch a column collapse onto one bank with no swizzle, then scatter across the bank view once the matching swizzle is applied.

## Design Rationale

The layout API follows three design choices.

First, it supports general shapes. Hardware tiles are not always powers of two. Global tensors, shared memory stages, TMEM accumulators, and scale-factor buffers often have shapes that come from capacity limits or algorithm choices. The layout model treats those shapes as normal.

Second, the mapping goes from logical coordinates to physical coordinates. This direction is important because replication is common. One logical element may live in several physical places. A logical-to-physical map represents that directly as a set of coordinates.

Third, hardware axes are explicit. The layout does not use anonymous dimensions and rely on context to explain them later. The difference between `tx`, `tid_in_wg`, `laneid`, `warpid`, `TLane`, and `TCol` is written into the layout itself.

Legality and feasibility checks are not the job of the layout object alone. A layout can say where data is placed. Higher-level tile primitives decide whether a given operation can legally and efficiently use that placement. This separation keeps the layout API small while still giving the compiler enough information to dispatch real hardware operations.
