(chap_tensor_cores)=
# Tensor Cores: `tcgen05`

:::{admonition} Overview
:class: overview

- The `tcgen05` MMA is Blackwell's tile matrix-multiply-accumulate: a single cooperative instruction issued by one elected thread.
- Its accumulator lives in TMEM, not registers, and `cta_group::1` vs `::2` sets whether one or two CTAs cooperate (and the M dimension).
- Block-scaled MMAs (mxfp8 / nvfp4) add per-block scale factors, also staged through TMEM.
:::

Dense linear algebra is the bulk of the work in a modern model, and a GPU reaches
its advertised peak on that work in exactly one place: the Tensor Core. A CUDA-core matrix multiply
leaves most of the chip idle, so every fast GEMM or attention kernel lives or dies by how well it
feeds and drives this one unit — see {ref}`chap_background` for what a Tensor Core is and how it
differs from a CUDA core. The tile-level multiply-accumulate $D = AB + C$ has stayed constant since
Volta (2017), but each generation changes *how* the Tensor Core is programmed and *where* its
results live, and on Blackwell those two questions are the whole game: the **`tcgen05`** MMA moves
the accumulator off-register and decides how each tile is issued, and getting both right is what
separates a peak kernel from a slow one. This chapter is about driving that unit: the `tcgen05` MMA
and its modes ({ref}`chap_tmem`), the accumulator's new home, and the operand layouts
({ref}`chap_data_layout`) and async completion ({ref}`chap_async_barriers`) the MMA depends on.

Blackwell's **fifth-generation** Tensor Core, exposed through the **`tcgen05`** instruction
family, is the latest such shift, and for kernel authors it is the most consequential one in years.
The reason is a single design decision: the accumulator no longer lives in registers. Instead it
moves out into a dedicated on-chip memory, **Tensor Memory (TMEM)** ({ref}`chap_tmem`). That one
relocation ripples through everything else about how a kernel is written, which is why the rest of
this chapter is built around the `tcgen05` MMA itself.

The interactive demo below gives a first look at `tcgen05` in a few different configurations. In
each case, the MMA writes its partial sums into TMEM rather than registers.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tcgen05_intro.html" title="tcgen05 and Tensor Memory" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: a few `tcgen05` configurations and their TMEM accumulator behavior.*

## The `tcgen05` MMA

`tcgen05` is Blackwell's tensor-core instruction family, and it is worth being precise about what it
does and does not cover. It is *only* the compute path — the unit that actually multiplies the
tiles. Getting the operands into place beforehand is a separate job, and that job belongs to TMA
({ref}`chap_tma`). So when you issue an MMA, what you are really configuring are the three knobs we
introduced earlier, and each one answers a different question:

1. **Scope — who cooperates.** An MMA is issued for a warpgroup. On Blackwell, some modes go a step
   further and let two CTAs in a cluster cooperate on one larger MMA tile. Either way, the
   instruction itself is *committed* by a single elected thread.
2. **Layout — where the operands and the result live.** The operands are tiles in SMEM (a few
   variants also read an A-operand straight from TMEM), and the accumulator is written to TMEM. The
   operand layouts have to match exactly what the Tensor Core expects to find, and that expectation
   is where swizzle ({ref}`chap_data_layout`) enters the picture.
3. **Dispatch — which path runs it.** `tcgen05` issues the MMA and then *returns before the math is
   done*: it is asynchronous. To know when the result is actually ready, you track completion with a
   barrier ({ref}`chap_async_barriers`).

That third knob deserves a moment on its own, because it is easy to gloss over. **Issuing an MMA is
not the same as finishing it.** The instruction hands control back immediately, and that is exactly
what makes overlap possible: a fast kernel issues the loads for the next tile while the current MMA
is still grinding away, and only waits on the barrier at the last moment, when it genuinely needs
the result in hand.

## The Accumulator Lives in TMEM

Of all the things Blackwell changes, where the accumulator lives is the defining one. On earlier
generations the running sum $C$ sat in registers, and that had a real cost: it tied how large an
accumulator you could keep to a thread's register budget, a budget that is always in short supply.
Blackwell breaks that link. Now `tcgen05.mma` writes the accumulator to **Tensor Memory (TMEM)** —
the per-SM 2D scratchpad we cover in {ref}`chap_tmem` — and the epilogue reads it back out with
`tcgen05.ld`. With that settled, the rest of this chapter can concentrate on the MMA itself: how it
is issued, and how its operands split across the various modes.

## `cta_group::1` vs `cta_group::2`, and the M Dimension

Why does a single instruction need several modes at all? Two reasons. The useful tile shapes do not
all fit into TMEM the same way, and one CTA is sometimes simply too small to hold a tile worth
computing. To accommodate both, a `tcgen05` MMA can run over one CTA (`cta_group::1`) or over a pair
of CTAs in a cluster (`cta_group::2`). Together, the mode and the tile's **M** dimension decide how
the (M, N) accumulator gets laid out in TMEM — that is, which lane and which column each logical
element `C[m, n]` ends up on (recall that TMEM is 128 lanes × up to 512 columns). In the single-CTA
modes, N maps onto TMEM columns and the only thing that varies from case to case is how the M rows
map onto lanes. The 2-CTA mode goes one step further and also splits N across lanes, as the table
makes clear. Note, too, that not every M is legal in every mode; here are the combinations that are:

| `cta_group` | M | CTAs | accumulator layout in TMEM |
|---|---:|:---:|---|
| 1 | 64 | 1 | four 16-row runs at lane stride 32 (one CTA) |
| 1 | 128 | 1 | identity: row `m` → lane `m` (one CTA) |
| 2 | 128 | 2 | 64 rows/CTA; N split across lanes 0–63 / 64–127 |
| 2 | 256 | 2 | 128 rows/CTA, contiguous |

The simplest case to start from is **`cta_group::1`, M = 128 (identity).** A single CTA holds
A (M, K) and B (N, K) in its SMEM and the accumulator in its TMEM. Because there happen to be exactly
as many rows as there are lanes, the mapping is the obvious one — row `m` goes to lane `m`, and
column `n` goes to a TMEM column. C fills all 128 lanes × N columns, with nothing left idle. This is
the picture to keep in your head as the default; every other mode is best understood as a variation
on what to do when the row count and the lane count no longer line up so neatly.

![cta_group::1, M=128: identity — row m maps to TMEM lane m](../img/mma_cg1_m128.svg)

**`cta_group::1`, M = 64.** Here the row count and the lane count come apart: there are only 64 rows
to place into 128 lanes. The naive thing would be to pack them into lanes 0–63, but that would leave
the other half of TMEM sitting idle. So the hardware does something less obvious and spreads them
out instead, as four runs of 16 rows at a **lane stride of 32**: rows 0–15 go to lanes 0–15, rows
16–31 to lanes 32–47, rows 32–47 to lanes 64–79, and rows 48–63 to lanes 96–111 — which leaves lanes
16–31, 48–63, 80–95, and 112–127 free. Those gaps are not waste; they are there on purpose. A
`lane_align` of 16 shifts the whole pattern up by 16, so **two independent M = 64 MMAs can share the
same 128 lanes**, one running at align 0 and the other at align 16. Throughout all of this, the
columns stay the full N.

![cta_group::1, M=64: four 16-row runs at lane stride 32, gaps free for a second M=64 tile](../img/mma_cg1_m64.svg)

When one CTA's TMEM simply cannot hold the tile — anything taller than 128 rows — we step up to
`cta_group::2` and let a pair of CTAs cooperate. **`cta_group::2`, M = 256** is the cleanest version
of this. An even/odd CTA pair in the cluster splits M contiguously: CTA 0 takes rows 0–127 and CTA 1
takes rows 128–255, and **each writes into its own TMEM lanes 0–127, across the full N columns**. It
helps to be explicit about what this means physically: the (256, N) accumulator is two separate
128-row TMEM regions, one per CTA — not a single buffer that somehow spans both. Each CTA keeps its
own 128 A-rows in SMEM, and the **even CTA** is the one that issues the instruction and commits the
completion barrier on behalf of the pair. This is exactly the mode the 2-CTA cluster GEMM in
{ref}`chap_gemm_advanced` is built on.

![cta_group::2, M=256: M split contiguously, 128 rows per CTA across the pair](../img/mma_cg2_m256.svg)

**`cta_group::2`, M = 128** is the same CTA pairing, but applied to a shorter tile, and the
interesting twist is what happens to the slack. With only 128 rows to share between two CTAs, there
is room to spare in M, and the hardware spends it on N instead. Each CTA takes only **64 M-rows**
(CTA 0 gets rows 0–63, CTA 1 gets rows 64–127), and N is split in half: within each CTA, the low-N
half occupies lanes 0–63 and the high-N half stacks on top into lanes 64–127. The result is that
both CTAs again use all 128 lanes — packing 64 rows × two N-halves — rather than leaving the upper
lanes empty.

![cta_group::2, M=128: 64 rows per CTA with N split across the lower/upper 64 lanes](../img/mma_cg2_m128.svg)

It is worth pausing on what stays the same across all four modes, since it is easy to lose sight of
amid the row-and-lane bookkeeping. In every one of them, B (N, K) lives in SMEM, and under
`cta_group::2` each CTA supplies its operands from its own SMEM (with A split by M, as we saw above).
The accumulator C (M, N) is f32 in TMEM for the kernels we work with here, though this is a choice
rather than a hard rule: the `.kind::f16` path can also accumulate in f16.

## Block-Scaled MMA (mxfp8 / nvfp4)

The real trouble with very low precision is dynamic range. An fp8 or fp4 element simply cannot span
the spread of magnitudes that a real matrix contains, so if you try to cover everything with a single
global scale, you are forced into a bad trade: either you clip the large values, or you flush the
small ones to zero. The way out is to stop scaling globally and start scaling finely. That is exactly
how low-precision formats hold onto their accuracy — they attach a **scale factor to each block of
K**. Every group of `B` consecutive K-elements shares one scale, and because the group is small, that
scale can keep its block's values comfortably inside the representable range. Mechanically, this means
a block-scaled MMA carries two extra operands beyond A and B: the scale-factor tensors
**SFA (M, SFK)** and **SFB (N, SFK)**, where `SFK = K / B`.

Just how fine the blocking is depends on the format, because the block size `B` is precisely the
granularity of the scale vector:

| Format | data dtype | scale dtype | block `B` |
|---|---|---|---|
| nvfp4 | fp4 | e4m3 | 16 |
| mxfp8 | fp8 | e8m0 | 32 |
| mxf4 | fp4 | e8m0 | 32 |

The math behind this is a per-block dequantize-then-accumulate. Before any product is summed in f32,
each quantized value is multiplied back by its block's scale, which restores it to roughly its true
magnitude:

$$D[m,n] \mathrel{+}= \sum_k \big(A_q[m,k]\cdot \mathrm{SFA}[m,\, k/B]\big)\,\big(B_q[n,k]\cdot \mathrm{SFB}[n,\, k/B]\big).$$

The choice of scale dtype shapes what those factors are even able to express. With e8m0 scales the
factor is always an exact power of two — the stored byte is a biased exponent, so the scale works out
to $2^{\text{byte}-127}$ — whereas nvfp4's e4m3 scales are small floats, which lets them land at
values in between powers of two.

Compared with the plain MMA, the placement of operands differs in just one respect, but it is an
important one: **the scale factors live in TMEM**, not SMEM. The reason is straightforward — the
block-scaled `tcgen05.mma` reads them from TMEM:

| Logical | dtype | Where |
|---|---|---|
| A (M, K) | fp8 / fp4 (packed) | SMEM |
| B (N, K) | fp8 / fp4 (packed) | SMEM |
| SFA (M, SFK) | e8m0 / e4m3 | **TMEM** |
| SFB (N, SFK) | e8m0 / e4m3 | **TMEM** |
| C (M, N) | f32 | TMEM |

Here is where a small wrinkle appears. TMA always delivers into SMEM, so this TMEM requirement forces
the scale factors onto a detour that the data operands never have to take: they are first TMA-loaded
into SMEM, and only then copied from SMEM to TMEM with `tcgen05.cp`, before the MMA is allowed to run.
The layout they take in TMEM is pleasingly compact — a 128-row scale vector packs into just 32 lanes
(`r % 32`, with `r // 32` running along the columns) and is then broadcast `warpx4` to all 128 reading
lanes. The full layout is spelled out in {ref}`chap_layout_generations`.

In the two-CTA case, the guiding principle is simple: a scale travels with whatever it scales. Under
**`cta_group::2`**, the scale factors split in exactly the same way as the data they describe.
**SFA follows A**, so each CTA holds the M-half that matches its own A rows. **SFB, by contrast, is
multicast to both CTAs**, because each CTA's M-half has to multiply against the very same per-N column
scales. In the kernels, this is what surfaces as the familiar "load SFA per-CTA (single-CTA mask),
broadcast SFB (pair mask)" pattern, with per-CTA `SFA` and multicast `SFB`.

![Block-scaled MMA placement: A/B packed in SMEM; SFA, SFB, and C in TMEM, with SFA split by M and SFB multicast across the CTA pair](../img/mma_block_scaled.svg)
