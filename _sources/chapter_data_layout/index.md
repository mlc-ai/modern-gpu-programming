(chap_data_layout)=
# Introduction to Data Layout

:::{admonition} Overview
:class: overview

- A *data layout* maps a tensor's logical indices to physical locations, and it decides coalescing, bank conflicts, and whether an engine can read a tile.
- The book writes layouts in one notation: `S[(shape) : (strides)]`, with named axes (`@laneid`, `@TLane`, …) and a replication term `R[...]` for broadcast or copied data.
- Swizzle is an XOR remapping of addresses that removes shared-memory bank conflicts.
:::

Two kernels can compute the same result over the same numbers and yet differ in speed by an order
of magnitude, purely because of where those numbers sit in the hardware. That "where" is the *data
layout*: the map from a tensor's logical indices `(i, j, …)` to a physical location — which byte of
memory, which thread's register, which SMEM bank. The data and its logical shape do not change; the
layout is what decides whether an access coalesces, hits distinct banks, or matches the format an
engine expects, and so it is where much of GPU performance is won or lost. To talk about it
precisely we need a way to write layouts down. This chapter builds a compact **notation** for
layouts that the rest of Part I uses; {ref}`chap_layout_generations` then applies it to each GPU
generation's hardware requirements. Treat the notation here as plain mathematical notation — the
goal is simply to get fluent at reading a layout.

## The shape–stride model

We start from the simplest possible layout and build everything else on top of it. At its core a
layout is just two things: a **shape** and matching **strides**. We write it as
`S[(shape) : (strides)]`, and the address of a logical index is the dot product of the index with
the strides. A row-major 4×4 matrix, for instance, is

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1
```

This is the classic shape/stride model, written compactly (a row-major simplification of CuTe's
notation). Everything below is built from it, so it is worth dwelling on.

The reassuring part is that you have almost certainly used this model already. If you have written
PyTorch or NumPy, then you have, because a tensor there *is* exactly a shape plus a stride over a
flat storage buffer:

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← exactly S[(3, 4) : (4, 1)]
```

Seeing a tensor this way pays off immediately: it explains why many "reshaping" operations don't
touch the data at all. They just rewrite the strides and hand back a **view** over the same
storage. Transpose/permute is the clearest case:

```python
tt = t.permute(1, 0)               # or t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← strides swapped, no data moved
tt.data_ptr() == t.data_ptr()      # True — same bytes
```

`t.permute(1, 0)` is `S[(4, 3) : (1, 4)]` over the *same* memory: the transpose is purely a stride
change. `reshape`/`view` on a contiguous tensor are the same story — new shape and strides, same
storage. (NumPy is identical; its `.strides` are just counted in bytes rather than elements.)

This is exactly how layouts work on a GPU, and the rest of the chapter is really one idea
generalized: a tile's mapping — to memory, or via named axes to lanes and registers — is a stride
rule over a fixed buffer, so rearranging a tile is usually a change of *layout*, not a copy. Hold
on to that, because it is what makes the more exotic layouts ahead manageable.

## Tile layout

GPU kernels rarely work on a whole matrix at once; they work on tiles. The pleasant surprise is
that tiling needs no new machinery — it is just a layout with more dimensions. An 8×8 matrix cut
into 2×4 tiles becomes a 4-D layout — `(tile_row, row_in_tile, tile_col, col_in_tile)` — with
strides chosen so each tile is contiguous:

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

A logical `(i, j)` maps through `(i//2, i%2, j//4, j%4)` and then the strides. Notice that the
notation expresses tiling without any special "tile" concept at all — it is still the same
shape–stride model, just with the index split into outer and inner coordinates.

```{raw} html
<iframe src="../demo/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: click a cell to see its tiled index and address.*

## Named axes

So far an address has meant a byte in linear memory. But a GPU has more than one "address space":
besides memory there are the lanes of a warp, a thread's registers, and TMEM's lanes and columns,
and a tile can be spread across any of them. To say *which* space a stride moves through, we let a
stride coefficient carry an **axis tag** — `@m` for ordinary memory, and others like `@laneid`
(thread lane), `@reg` (register), `@warpid`, and TMEM's `@TLane` / `@TCol`. With memory tags
explicit, a row-major 8×16 tile in memory is just

```text
S[(8, 16) : (16@m, 1@m)]
```

The tags earn their keep when a layout describes data *spread across threads* rather than laid out
in memory. For instance, `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]` maps rows and columns onto
lane IDs and a per-lane register instead of linear memory — and this is not a contrived example: it
is exactly the tensor-core register fragment you will meet in {ref}`chap_layout_generations`.

```{raw} html
<iframe src="../demo/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: a layout over `@laneid` and `@reg` — click a cell to see which lane/register holds it.*

## Distributed layout

If a tag can name a lane or a register, nothing stops it from naming a whole GPU. The same idea
therefore extends across devices: axes like `@gpuid_x` / `@gpuid_y` place data on a GPU mesh. One
genuinely new thing appears at this scale, though — data that is *copied* rather than partitioned —
and we express it with a **replication** dimension `R[n : stride]` (stride 0 = broadcast):

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

```{raw} html
<iframe src="../demo/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: a layout distributed over a 2×2 GPU mesh.*

### Intra-kernel replication pattern: scale factors in TMEM

Replication might sound like a multi-GPU concern, but the same dimension describes something that
happens inside a single kernel: data the hardware *broadcasts across lanes*. Blackwell's
block-scaled MMA ({ref}`chap_layout_generations`) is the clean example, and it is worth tracing
because the layout looks surprising at first. Its scale factors live in TMEM, and a 128-row scale
vector is stored in only **32 TMEM lanes** (logical row `r` → lane `r % 32`, with `r // 32` running
along columns). Those 32 stored lanes are then **replicated along the lane axis** so that all 128
lanes of the reading warpgroup get a copy — a `warpx4` broadcast, written with a replication
dimension:

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

Four replicas at a stride of 32 lanes: lanes `l`, `l+32`, `l+64`, `l+96` all hold the same scale.
The replication dimension carries no new data — it says "the same value, in four lane positions,"
exactly as `@gpuid_x` broadcast a row across the GPU mesh above.

```{raw} html
<iframe src="../demo/sf_tmem.html" title="Scale factors in TMEM: packing and warpx4 replication" loading="lazy"
        style="width:100%; min-width:1040px; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: hover a TMEM lane — it stores 4 logical M-rows (`m // 32` → column) and is broadcast `warpx4` to 4 of the warpgroup's 128 reading lanes.*

The byte packing inside each column (the `scale_vec` 1X/2X/4X modes) and the `cta_group::2` split
are in {ref}`chap_layout_generations`.

## Swizzle layout

The last layout in this chapter exists to solve a specific hardware problem, so it helps to see the
problem first. Shared memory is split into **32 banks** that can be read in parallel, and a warp's
accesses are served in a single cycle only if they hit *distinct* banks; when several lanes land in
different addresses of the *same* bank, those accesses **serialize** into a bank conflict. This bites
in a very common situation: a layout where reading a *row* is conflict-free but reading a *column*
funnels every element into one bank, so the column access runs many times slower than the row
access over identical data.

**Swizzle** is the fix. It permutes the address mapping — typically by XOR-ing the column index with
the row — so that *both* row and column accesses spread across banks:

```{raw} html
<iframe src="../demo/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: an 8×8 tile, bank-conflicted by column in plain row-major, conflict-free after the XOR swizzle.*

In practice you do not invent a permutation per kernel; the hardware offers a fixed menu, named by
the granularity at which it shuffles — `SWIZZLE_NONE`, `SWIZZLE_32B`, `SWIZZLE_64B`, `SWIZZLE_128B`:

```{raw} html
<iframe src="../demo/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: the SWIZZLE_128B pattern.*

What the granularity refers to is the size of a small repeating **atom** on which the permutation is
defined: `SWIZZLE_128B` uses an 8 × 128 B atom, `SWIZZLE_64B` an 8 × 64 B atom, `SWIZZLE_32B` an
8 × 32 B atom, and the whole tile is then tiled by that atom. The demo shows the element arrangement
inside one atom for each format:

```{raw} html
<iframe src="../demo/swizzle_atom_general.html" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: pick a swizzle format to see its atom shape (8 × N B) and how elements are permuted inside it.*

The practical upshot is that swizzle is a remapping the hardware applies for you, so your job is not
to compute permuted addresses but to pick a consistent mode (e.g. `SWIZZLE_128B`) across all the
ops that touch a tile and let the hardware handle the addressing — `SWIZZLE_128B`, for example,
gives conflict-free access to 8 rows and 8 columns at a time in fp16. *Which* swizzle each engine
demands is generation-specific, and that is precisely the subject of the next chapter.
