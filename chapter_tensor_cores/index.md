(chap_tensor_cores)=
# Tensor Cores: `tcgen05`

:::{admonition} Overview
:class: overview

- The `tcgen05` MMA is Blackwell's tile matrix-multiply-accumulate: a single cooperative instruction issued by one elected thread.
- Its accumulator lives in TMEM, not registers, and `cta_group::1` vs `::2` sets whether one or two CTAs cooperate (and the M dimension).
- Block-scaled MMAs (mxfp8 / nvfp4) add per-block scale factors, also staged through TMEM.
:::

**Motivation.** Dense linear algebra is the bulk of the work in a modern model, and a GPU reaches
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
family, is the latest such shift, and the most consequential one for kernel authors: it changes
where the accumulator lives, moving it out of registers and into a dedicated on-chip memory,
**Tensor Memory (TMEM)** ({ref}`chap_tmem`). That single relocation reshapes how a kernel is
written, so this chapter focuses on the `tcgen05` MMA itself.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tcgen05_intro.html" title="tcgen05 and Tensor Memory" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the `tcgen05` MMA writing its accumulator into Tensor Memory (TMEM).*

## The `tcgen05` MMA

`tcgen05` is Blackwell's tensor-core instruction family. It is *only* the compute path, the unit
that multiplies tiles; moving the operands into place is TMA's job ({ref}`chap_tma`). What you
configure when you issue an MMA are the three knobs from the introduction, each answering a
different question:

1. **Scope — who cooperates.** An MMA is issued for a warpgroup, and some Blackwell modes let two
   CTAs in a cluster cooperate on one larger MMA tile. The instruction itself is *committed* by a
   single elected thread.
2. **Layout — where operands and result live.** Operands are tiles in SMEM (some variants also
   read an A-operand from TMEM); the accumulator is written to TMEM. Operand layouts must match
   what the Tensor Core expects, which is where swizzle ({ref}`chap_data_layout`) comes in.
3. **Dispatch — which path runs it.** `tcgen05` issues the MMA and *returns before the math is
   done* — it is asynchronous. Completion is tracked with a barrier ({ref}`chap_async_barriers`).

On the third knob: **issuing an MMA is not the same as finishing it.** The instruction returns
immediately, which is what makes overlap possible — a fast kernel issues the next tile's loads
while the current MMA is still running, then waits on the barrier only when it needs the result.

## The accumulator lives in TMEM

Where the accumulator lives is the defining Blackwell change. On earlier generations the
running sum $C$ sat in registers, which tied accumulator capacity to a thread's register budget. On
Blackwell, `tcgen05.mma` instead writes it to **Tensor Memory (TMEM)** — the per-SM 2D scratchpad
covered in {ref}`chap_tmem` — and the epilogue reads it back with `tcgen05.ld`. The rest of this
chapter is the MMA itself: how it is issued and how its operands split across the various modes.

## `cta_group::1` vs `cta_group::2`, and the M Dimension

A single instruction needs several modes because the useful tile shapes do not all
fit the same way into TMEM, and one CTA is sometimes too small to hold a worthwhile tile. A
`tcgen05` MMA runs over one CTA (`cta_group::1`) or a pair of CTAs in a cluster
(`cta_group::2`). The mode and the tile's **M** together decide how the (M, N) accumulator is laid
out in TMEM — which lane and column each logical element `C[m, n]` lands on (TMEM is 128 lanes ×
up to 512 columns). In the single-CTA modes N maps to TMEM columns and only the *lane* mapping of
the M rows changes from case to case; the 2-CTA mode additionally splits N across lanes, as the
table shows.
Not every M is legal in every mode; the combinations are these:

| `cta_group` | M | CTAs | accumulator layout in TMEM |
|---|---:|:---:|---|
| 1 | 64 | 1 | four 16-row runs at lane stride 32 (one CTA) |
| 1 | 128 | 1 | identity: row `m` → lane `m` (one CTA) |
| 2 | 128 | 2 | 64 rows/CTA; N split across lanes 0–63 / 64–127 |
| 2 | 256 | 2 | 128 rows/CTA, contiguous |

The simplest case is **`cta_group::1`, M = 128 (identity).** One CTA holds A (M, K) and B (N, K) in
its SMEM and the accumulator in its TMEM, and because there are exactly as many rows as lanes the
mapping is the obvious one: row `m` → lane `m`, column `n` → a TMEM column. C occupies all 128 lanes
× N columns, with nothing wasted. This is the layout to picture by default; the others are
variations on what to do when the row count and the lane count no longer match.

![cta_group::1, M=128: identity — row m maps to TMEM lane m](../img/mma_cg1_m128.svg)

**`cta_group::1`, M = 64.** Now there are only 64 rows to place in 128 lanes, so packing
into lanes 0–63 would leave half of TMEM idle. The hardware spreads them out instead, as four runs
of 16 rows at a **lane stride of 32**: rows 0–15 → lanes 0–15, rows 16–31 → lanes 32–47, rows 32–47
→ lanes 64–79, rows 48–63 → lanes 96–111 — leaving lanes 16–31, 48–63, 80–95, 112–127 free. Those
gaps have a use: a `lane_align` of 16 shifts the whole pattern up by 16, so **two independent
M = 64 MMAs can share the 128 lanes** (one at align 0, one at align 16). Columns remain the full N.

![cta_group::1, M=64: four 16-row runs at lane stride 32, gaps free for a second M=64 tile](../img/mma_cg1_m64.svg)

When one CTA's TMEM is not enough — a tile taller than 128 rows — we move to `cta_group::2` and let
a pair cooperate. **`cta_group::2`, M = 256** is the clean version: an even/odd CTA pair in the
cluster splits M contiguously, CTA 0 taking rows 0–127 and CTA 1 rows 128–255, **each into its own
TMEM lanes 0–127, full N columns**. The (256, N) accumulator is two
separate 128-row TMEM regions, one per CTA — not one buffer spanning both. Each CTA holds its own
128 A-rows in SMEM, and the **even CTA** issues the instruction and commits the completion barrier
for the pair. This is the mode the 2-CTA cluster GEMM in {ref}`chap_gemm_advanced` uses.

![cta_group::2, M=256: M split contiguously, 128 rows per CTA across the pair](../img/mma_cg2_m256.svg)

**`cta_group::2`, M = 128** is the same pairing applied to a shorter tile, and here the slack in M
is spent on N instead. Each CTA takes only **64 M-rows** (CTA 0 rows 0–63, CTA 1 rows 64–127), and
N is split in half: within each CTA the low-N half occupies lanes 0–63 and the high-N half stacks
into lanes 64–127. Both CTAs end up using all 128 lanes, packing 64 rows × two N-halves rather than
leaving the upper lanes empty.

![cta_group::2, M=128: 64 rows per CTA with N split across the lower/upper 64 lanes](../img/mma_cg2_m128.svg)

What does *not* change across the four modes: in every case B (N, K) lives in
SMEM, and for `cta_group::2` each CTA supplies operands from its own SMEM (with A split by M as
above). The accumulator C (M, N) is f32 in TMEM for the kernels here; the `.kind::f16` path can also accumulate in f16.

## Block-Scaled MMA (mxfp8 / nvfp4)

The trouble with very low precision is dynamic range: an fp8 or fp4 element cannot represent
the spread of magnitudes a real matrix contains, so a single global scale either clips the large
values or flushes the small ones to zero. The fix is to scale more finely. Low-precision formats
hold accuracy by attaching a **scale factor to each block of K**: every group of `B` consecutive
K-elements shares one scale, which keeps each block's values inside the representable range. A
block-scaled MMA carries two operands beyond A and B — the scale-factor tensors
**SFA (M, SFK)** and **SFB (N, SFK)**, where `SFK = K / B`.

How fine the blocking is depends on the format, since the block size `B` is exactly the
scale-vector granularity:

| Format | data dtype | scale dtype | block `B` |
|---|---|---|---|
| nvfp4 | fp4 | e4m3 | 16 |
| mxfp8 | fp8 | e8m0 | 32 |
| mxf4 | fp4 | e8m0 | 32 |

The math is a per-block dequantize-then-accumulate:
each quantized value is multiplied back by its block's scale before the product is summed in
f32:

$$D[m,n] \mathrel{+}= \sum_k \big(A_q[m,k]\cdot \mathrm{SFA}[m,\, k/B]\big)\,\big(B_q[n,k]\cdot \mathrm{SFB}[n,\, k/B]\big).$$

The scale dtype shapes what those factors can express. For e8m0 scales the factor is an exact power
of two — the stored byte is a biased exponent, so the scale is $2^{\text{byte}-127}$ — while
nvfp4's e4m3 scales are small floats and can land between powers of two.

The placement differs from the plain MMA in one way:
**the scale factors live in TMEM**, not SMEM, because the block-scaled `tcgen05.mma` reads them from
TMEM:

| Logical | dtype | Where |
|---|---|---|
| A (M, K) | fp8 / fp4 (packed) | SMEM |
| B (N, K) | fp8 / fp4 (packed) | SMEM |
| SFA (M, SFK) | e8m0 / e4m3 | **TMEM** |
| SFB (N, SFK) | e8m0 / e4m3 | **TMEM** |
| C (M, N) | f32 | TMEM |

Because TMA delivers into SMEM, this TMEM requirement forces the scale factors to take a detour the
data operands never do: they are first TMA-loaded into SMEM, then copied SMEM → TMEM with
`tcgen05.cp` before the MMA can run. The TMEM layout is compact — a 128-row scale vector packs into
just 32 lanes (`r % 32`, with `r // 32` running along columns) and is broadcast `warpx4` to all 128
reading lanes; the full layout is in {ref}`chap_layout_generations`.

In the two-CTA case a scale travels with whatever it scales.
Under **`cta_group::2`** the scale factors split exactly like the data they describe. **SFA follows
A**, so each CTA holds the M-half matching its own A rows, whereas **SFB is multicast to both CTAs**,
because each CTA's M-half multiplies against the same per-N column scales. In the kernels this
shows up as the "load SFA per-CTA (single-CTA mask), broadcast SFB (pair mask)" pattern.

![Block-scaled MMA placement: A/B packed in SMEM; SFA, SFB, and C in TMEM, with SFA split by M and SFB multicast across the CTA pair](../img/mma_block_scaled.svg)
