(chap_data_layout)=
# Introduction to Data Layout

:::{admonition} Overview
:class: overview

- A *data layout* maps a tensor's logical indices to physical locations, and it decides coalescing, bank conflicts, and whether an engine can read a tile.
- The book writes layouts in one notation: `S[(shape) : (strides)]`, with named axes (`@laneid`, `@TLane`, …) and a replication term `R[...]` for broadcast or copied data.
- Swizzle is an XOR remapping of addresses that removes shared-memory bank conflicts.
:::

The same numbers, written into memory in a different physical arrangement, can run an order of
magnitude apart on the same GPU. The reason is that a tensor's logical indices say nothing about
where its bytes actually sit, and the hardware is highly sensitive to that placement: it
determines whether 32 lanes' loads coalesce into one transaction or scatter into 32, whether their
addresses land in distinct memory banks or collide and serialize, and even whether a tile matches
the byte arrangement a Tensor Core can read at all. When we work with machine learning programs,
however, we usually think in terms of multi-dimensional tensors. A **data layout** describes how a
tensor element with logical indices `(i, j, …)` is mapped to a physical location — in memory, in
registers, or in some other piece of hardware storage. In this chapter, we introduce the main data
layouts that arise in modern GPU programming. To keep the discussion tractable, we will develop a
single compact **notation** that describes all of them across the range of situations a machine
learning system runs into. We will close with **swizzling**, the mechanism that makes both
row-wise and column-wise access to a tile efficient at the same time.

## The Shape–Stride Model

It is worth starting from the simplest possible layout, because everything else in the chapter is
built on top of it. At its core, a layout is just two things: a **shape** and a matching set of
**strides**. We write the pair as `S[(shape) : (strides)]`, and to find where a logical index lives
we take the dot product of that index with the strides. A row-major 4×4 matrix, for instance, looks
like this:

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1
```

This is nothing more than the classic shape/stride model, written compactly — a row-major
simplification of CuTe's notation — and everything that follows is built from it.

In fact, you have almost certainly used this model already. Anyone who has written PyTorch or NumPy
has, because a tensor in those libraries *is* precisely a shape together with a stride over a flat
storage buffer:

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← exactly S[(3, 4) : (4, 1)]
```

Once you see a tensor this way, it becomes clear why so many "reshaping" operations never touch the
data at all. They simply rewrite the strides and hand back a **view** over the same storage, and the
clearest example is transpose, or permute:

```python
tt = t.permute(1, 0)               # or t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← strides swapped, no data moved
tt.data_ptr() == t.data_ptr()      # True — same bytes
```

Here `t.permute(1, 0)` is `S[(4, 3) : (1, 4)]` over the *same* memory: the transpose is purely a
change of strides, with not a single byte moved. The story is the same for `reshape` or `view` on a
contiguous tensor — a new shape and new strides over the old storage. (NumPy behaves identically; the
only difference is that its `.strides` are counted in bytes rather than elements.)

This is exactly how layouts work on a GPU, and the rest of the chapter is really a series of
variations on one idea: a tile's mapping — whether into memory, or, through the named axes we
introduce shortly, into lanes and registers — is a stride rule over a fixed buffer, so rearranging a
tile is usually a change of *layout* rather than a copy. We should be careful about the boundaries of
this reasoning, though. The zero-copy story holds cleanly for a logical view over a single linear
address space; on a GPU it applies only when the new view is compatible with the existing byte and
ownership arrangement. The moment you change which thread or register owns an element, or change the
SMEM swizzle, you generally need real data movement — loads, stores, shuffles, `ldmatrix`,
transposes.

## Tile Layout

So far we have described layouts for whole tensors. GPU kernels, however, rarely operate on an
entire matrix at once; they work on smaller tiles, which are loaded, transformed, and computed on by
different parts of the hardware. The good news is that tiling asks for nothing new. It is still
just a layout — only now written with a few more dimensions. Cut an 8×8 matrix into 2×4 tiles and we
get a 4-D layout, with coordinates `(tile_row, row_in_tile, tile_col, col_in_tile)` and strides
chosen so that each tile stays contiguous:

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

A logical `(i, j)` first becomes `(i//2, i%2, j//4, j%4)` and then runs through the strides. What is
worth noticing is that the notation expresses tiling without any special "tile" concept at all: it is
the same shape–stride model as before, with the index merely split into outer and inner coordinates.

The interactive visualization below shows how a logical matrix index is decomposed into tile
coordinates and then mapped to a physical address.

```{raw} html
<iframe src="../demo/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: click a cell to see its tiled index and address.*

## Named Axes

Up to this point we have treated an address as a location in linear memory. On a GPU, though, data
can live in more than one place: besides memory, a tile may be spread across warp lanes, across
thread registers, or across TMEM lanes and columns. To describe all of these uniformly, we extend
the notation with **named axes**. The idea is to let each stride coefficient carry an axis tag that
says which space it moves through — `@m` for ordinary memory, `@laneid` for warp lanes, `@reg` for
registers, `@warpid` for warps, and `@TLane` / `@TCol` for TMEM coordinates. With the tags in hand, a
single layout can describe not only where data sits in memory but also how it is distributed across
the hardware resources that operate on it.

Once the memory tags are made explicit, a row-major 8×16 tile in memory is simply

```text
S[(8, 16) : (16@m, 1@m)]
```

The tags start to earn their keep when a layout describes data *spread across threads* rather than
laid out in memory. Take `S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]`: instead of pointing into
linear memory, it maps rows and columns onto lane IDs and a per-lane register. This is exactly the
tensor-core register fragment you will meet in {ref}`chap_layout_generations`.

The interactive visualization below shows how a layout can distribute tensor elements across warp
lanes and per-lane registers, rather than placing them in linear memory.

```{raw} html
<iframe src="../demo/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: a layout over `@laneid` and `@reg` — click a cell to see which lane/register holds it.*

## Distributed Layout

What makes named axes so useful is that they let us describe placement uniformly across many levels
of the system. We have just used them for lanes and registers inside a single GPU, but the very same
idea reaches across devices: axes such as `@gpuid_x` and `@gpuid_y` can say where data lives in a GPU
mesh, and with them the notation captures the sharding patterns that show up in distributed training
and inference. One thing the axes do not yet capture is *replication* — data that is copied to more
than one place — so we add the notation `R[n : stride]`, where `R` marks the replicated dimension.
For example, `R[2 : 1@gpuid_x]` describes replication along the `@gpuid_x` axis. The interactive demo
below shows several ways to express sharding and replication on a 2×2 GPU mesh:

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

```{raw} html
<iframe src="../demo/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: a layout distributed over a 2×2 GPU mesh.*

### Intra-Kernel Replication Pattern: Scale Factors in TMEM

The same replication dimension turns out to describe something that happens entirely inside a single
kernel as well: data that the hardware *broadcasts across lanes*. Blackwell's block-scaled MMA
({ref}`chap_layout_generations`) is a good example. Its scale factors live in TMEM, where a 128-row
scale vector is stored in only **32 TMEM lanes** — logical row `r` goes to TMEM lane `r % 32`, with
`r // 32` running along the columns. Those 32 stored TMEM lanes are then **replicated along the TMEM
`TLane` axis**, from 32 up to 128 TMEM lanes, so that each of the four warps in the reading warpgroup
finds a copy in its own 32-lane TMEM window. This is a `warpx4` broadcast, and we write it with a
replication dimension. The reads themselves are carried out by those warps' threads:

```text
S[(32, …) : (1@TLane, …)] + R[4 : 32@TLane]
```

That gives four replicas at a stride of 32 TMEM lanes: TMEM lanes `l`, `l+32`, `l+64`, and `l+96` all
hold the same scale. As before, the replication dimension carries no new data — it simply says "the
same value, sitting in four TMEM-lane positions," in just the way `@gpuid_x` broadcast a row across
the GPU mesh a moment ago.

```{raw} html
<iframe src="../demo/sf_tmem.html" title="Scale factors in TMEM: packing and warpx4 replication" loading="lazy"
        style="width:100%; min-width:1040px; height:560px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: hover a TMEM lane — it stores 4 logical M-rows (`m // 32` → column) and is broadcast `warpx4` across the `TLane` axis to all 128 TMEM lanes, one copy per warp's 32-lane window.*

The byte packing inside each column (the `scale_vec` 1X/2X/4X modes) and the `cta_group::2` split are
covered in {ref}`chap_layout_generations`.

Readers who already know CuTe can think of the notation in this chapter as a row-major variant of it,
extended with explicit hardware-named axes and a dedicated replication structure.

## Swizzle Layout

The final layout in this chapter exists to solve one specific hardware problem. Shared memory on a
GPU is organized into memory banks, and accesses run fastest when different lanes land on different
banks. When several lanes instead reach different addresses within the *same* bank, the hardware has
no choice but to serialize them, and we pay the cost of a **bank conflict**.

In tensor programs this is hard to avoid, because memory is not accessed in a purely linear order.
Working with matrices, we routinely need to read both row slices and column slices of the same tile,
and that creates a genuine tension: a layout that is efficient for row-wise access tends to produce
bank conflicts for column-wise access, while one that favors columns hurts rows. **Swizzling** is the
technique designed to break this tension.

The idea behind swizzle is to permute the address mapping — typically by XOR-ing the column index
with the row — so that *both* row and column accesses end up spread across banks. The conflict-free
guarantee it provides is specific: it holds for the matching element width, swizzle mode, and access
pattern (the one an engine's descriptor expects), and not for arbitrary element widths or alignments.

```{raw} html
<iframe src="../demo/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: an 8×8 tile, bank-conflicted by column in plain row-major, conflict-free after the XOR swizzle.*

The little 8×8 example captures the core idea, but real GPU memories have many more banks than that
toy picture suggests. To make swizzling work at full scale, we do not treat the whole tile as one
monolithic object. Instead, we cut memory into small segments and apply the swizzle pattern within
each segment, and in this way the same row/column-remapping trick carries over to the full banked
memory system.

Different hardware modes pick different sizes for this basic segment, or **atom**. The common choices
are `SWIZZLE_NONE`, `SWIZZLE_32B`, `SWIZZLE_64B`, and `SWIZZLE_128B`, each applying the same general
idea at a different granularity:

```{raw} html
<iframe src="../demo/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: the SWIZZLE_128B pattern.*

What the granularity refers to is the size of the small repeating **atom** on which the permutation
is defined. `SWIZZLE_128B` uses an 8 × 128 B atom, `SWIZZLE_64B` an 8 × 64 B atom, and `SWIZZLE_32B`
an 8 × 32 B atom; the whole tile is then tiled by whichever atom is in use. The demo shows the
element arrangement inside one atom for each format:

```{raw} html
<iframe src="../demo/swizzle_atom_general.html" title="Swizzle atom layout per format (128B/64B/32B)" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: pick a swizzle format to see its atom shape (8 × N B) and how elements are permuted inside it.*

One reassuring point is that swizzle is a remapping the hardware applies for you. Your job is not to
compute permuted addresses by hand but to pick a single consistent mode — say `SWIZZLE_128B` —
across all the ops that touch a tile, and then let the hardware take care of the addressing.
`SWIZZLE_128B`, to make it concrete, gives conflict-free access to 8 rows and 8 columns at a time in
fp16. *Which* swizzle each engine demands is generation-specific, and that is the subject of the next
chapter.
