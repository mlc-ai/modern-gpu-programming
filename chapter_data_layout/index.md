(chap_data_layout)=
# Introduction to Data Layout

A *data layout* is the map from a tensor's logical indices `(i, j, …)` to a physical location —
which byte of memory, which thread's register, which SMEM bank. The data and its logical shape are
the same regardless; the layout decides whether an access coalesces, hits distinct banks, or matches
the format an engine expects. This chapter builds a compact **notation** for layouts that the rest
of Part I uses; {ref}`chap_layout_generations` then applies it to each GPU generation's hardware
requirements. Treat the notation here as plain mathematical notation — the goal is to get fluent at
reading a layout.

## The shape–stride model

A layout is a **shape** and matching **strides**. We write it as `S[(shape) : (strides)]`, and the
address of a logical index is the dot product of the index with the strides. A row-major 4×4 matrix
is

```text
S[(4, 4) : (4, 1)]        addr(i, j) = i·4 + j·1
```

This is the classic shape/stride model, written compactly (a row-major simplification of CuTe's
notation). Everything below is built from it.

If you have used PyTorch or NumPy, you have already used this model — a tensor *is* a shape plus a
stride over a flat storage buffer:

```python
import torch
t = torch.arange(12).reshape(3, 4)
t.shape        # torch.Size([3, 4])
t.stride()     # (4, 1)        ← exactly S[(3, 4) : (4, 1)]
```

The payoff: many "reshaping" operations don't touch the data at all — they just rewrite the strides
and hand back a **view** over the same storage. Transpose/permute is the clearest case:

```python
tt = t.permute(1, 0)               # or t.T
tt.shape                           # torch.Size([4, 3])
tt.stride()                        # (1, 4)        ← strides swapped, no data moved
tt.data_ptr() == t.data_ptr()      # True — same bytes
```

`t.permute(1, 0)` is `S[(4, 3) : (1, 4)]` over the *same* memory: the transpose is purely a stride
change. `reshape`/`view` on a contiguous tensor are the same story — new shape and strides, same
storage. (NumPy is identical; its `.strides` are just counted in bytes rather than elements.)

That is exactly how layouts work on a GPU, and the rest of this chapter generalizes it: a tile's
mapping — to memory, or via named axes to lanes and registers — is a stride rule over a fixed
buffer, so rearranging a tile is usually a change of *layout*, not a copy.

## Tile layout

Tiling is just a layout with more dimensions. An 8×8 matrix cut into 2×4 tiles is a 4-D layout —
`(tile_row, row_in_tile, tile_col, col_in_tile)` — with strides chosen so each tile is contiguous:

```text
S[(4, 2, 2, 4) : (16, 4, 8, 1)]
```

A logical `(i, j)` maps through `(i//2, i%2, j//4, j%4)` and then the strides. The notation expresses
tiling without any special "tile" concept — it is the same shape–stride model.

```{raw} html
<iframe src="../demo/tiled_layout.html" title="Tile layout: interactive address computation" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: click a cell to see its tiled index and address.*

## Named axes

A GPU has more than one "address space": linear memory, but also the lanes of a warp, a thread's
registers, and TMEM's lanes/columns. So a stride coefficient carries an **axis tag** — `@m` for
ordinary memory, and others like `@laneid` (thread lane), `@reg` (register), `@warpid`, and TMEM's
`@TLane` / `@TCol`.
A row-major 8×16 tile in memory is

```text
S[(8, 16) : (16@m, 1@m)]
```

Named axes let one layout describe data *spread across threads*. For instance,
`S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]` maps rows and columns onto lane IDs and a per-lane
register rather than linear memory — this is exactly the tensor-core register fragment you will meet
in {ref}`chap_layout_generations`.

```{raw} html
<iframe src="../demo/thread_register.html" title="Thread + register layout via named axes" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: a layout over `@laneid` and `@reg` — click a cell to see which lane/register holds it.*

## Distributed layout

The same idea extends across devices. Axes like `@gpuid_x` / `@gpuid_y` place data on a GPU mesh,
and a **replication** dimension `R[n : stride]` (stride 0 = broadcast) expresses data that is copied
rather than partitioned:

```text
S[(2, 4, 8) : (1@gpuid_y, 8@m, 1@m)] + R[2 : 1@gpuid_x]
```

```{raw} html
<iframe src="../demo/tile_distributed.html" title="Distributed layout across a GPU mesh" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: a layout distributed over a 2×2 GPU mesh.*

### Intra-kernel replication pattern: scale factors in TMEM

Replication is not only about multiple GPUs — it also describes data the hardware *broadcasts across
lanes*. Blackwell's block-scaled MMA ({ref}`chap_layout_generations`) is the clean example. Its
scale factors live in TMEM, and a 128-row scale vector is stored in only **32 TMEM lanes** (logical
row `r` → lane `r % 32`, with `r // 32` running along columns). Those 32 stored lanes are then
**replicated along the lane axis** so all 128 lanes of the reading warpgroup get a copy — a `warpx4`
broadcast, written with a replication dimension:

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

Shared memory is split into **32 banks** that can be read in parallel. A warp's accesses are served
in one cycle only if they hit *distinct* banks; if several lanes hit different addresses in the same
bank, the accesses **serialize** (a bank conflict). The classic problem: a layout where reading a
*row* is conflict-free but reading a *column* puts every element in one bank.

**Swizzle** fixes this by permuting the address mapping — typically an XOR of the column index with
the row — so that *both* row and column access spread across banks:

```{raw} html
<iframe src="../demo/swizzle_8x8.html" title="8x8 XOR swizzle" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: an 8×8 tile, bank-conflicted by column in plain row-major, conflict-free after the XOR swizzle.*

Real hardware names the permutation by its granularity — `SWIZZLE_NONE`, `SWIZZLE_32B`,
`SWIZZLE_64B`, `SWIZZLE_128B`:

```{raw} html
<iframe src="../demo/swizzle_128B.html" title="SWIZZLE_128B layout" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: the SWIZZLE_128B pattern.*

Swizzle is a remapping the hardware applies for you: pick a consistent mode (e.g. `SWIZZLE_128B`)
across the ops that touch a tile and let the hardware handle the addressing — `SWIZZLE_128B` gives
conflict-free access to 8 rows and 8 columns at a time (fp16). *Which* swizzle each engine demands
is generation-specific; that is the subject of the next chapter.
