(chap_tensor_cores)=
# Tensor Cores: `tcgen05`

Tensor Cores are not new. They have executed tile-level matrix multiply-accumulate
($D = AB + C$) since Volta (2017), and every generation since has carried them — see
{ref}`chap_background` for what a Tensor Core is and how it differs from a CUDA core. What
changes from generation to generation is *how the Tensor Core is programmed* and *where its
results live*.

Blackwell's **fifth-generation** Tensor Core, exposed through the **`tcgen05`** instruction
family, changes where the accumulator lives: it moves out of registers and into a dedicated
on-chip memory, **Tensor Memory (TMEM)** ({ref}`chap_tmem`). This chapter covers the `tcgen05`
MMA itself.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tcgen05_intro.html" title="tcgen05 and Tensor Memory" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the `tcgen05` MMA writing its accumulator into Tensor Memory (TMEM).*

## The `tcgen05` MMA

`tcgen05` is Blackwell's tensor-core instruction family. (It is *only* the compute path; data
movement is TMA's job, {ref}`chap_tma`.) Beyond the math, a `tcgen05` MMA is defined by the three
knobs from the introduction:

1. **Scope — who cooperates.** An MMA is issued for a warpgroup, and some Blackwell modes let two
   CTAs in a cluster cooperate on one larger MMA tile. The instruction itself is *committed* by a
   single elected thread.
2. **Layout — where operands and result live.** Operands are tiles in SMEM (some variants also
   read an A-operand from TMEM); the accumulator is written to TMEM. Operand layouts must match
   what the Tensor Core expects, which is where swizzle ({ref}`chap_data_layout`) comes in.
3. **Dispatch — which path runs it.** `tcgen05` issues the MMA and *returns before the math is
   done* — it is asynchronous. Completion is tracked with a barrier ({ref}`chap_async_barriers`).

**Issuing an MMA is not the same as finishing it.** A
fast kernel issues the next tile's loads while the current MMA is still running, then waits on the
barrier only when it actually needs the result.

## The accumulator lives in TMEM

The defining Blackwell change is *where the accumulator lives*: `tcgen05.mma` writes it to
**Tensor Memory (TMEM)** — the per-SM 2D scratchpad covered in {ref}`chap_tmem` — instead of to
registers, and the epilogue reads it back with `tcgen05.ld`. The rest of this chapter is the MMA
itself: how it is issued and how its operands split.

## `cta_group::1` vs `cta_group::2`, and the M Dimension

A `tcgen05` MMA runs over one CTA (`cta_group::1`) or a pair of CTAs in a cluster
(`cta_group::2`). The mode and the tile's **M** together decide how the (M, N) accumulator is laid
out in TMEM — which lane and column each logical element `C[m, n]` lands on. TMEM is 128 lanes × up
to 512 columns; **N always maps to columns**, so what changes across the cases is the *lane*
mapping of the M rows. Each mode supports only specific M:

| `cta_group` | M | CTAs | accumulator layout in TMEM |
|---|---:|:---:|---|
| 1 | 64 | 1 | four 16-row runs at lane stride 32 (one CTA) |
| 1 | 128 | 1 | identity: row `m` → lane `m` (one CTA) |
| 2 | 128 | 2 | 64 rows/CTA; N split across lanes 0–63 / 64–127 |
| 2 | 256 | 2 | 128 rows/CTA, contiguous |

**`cta_group::1`, M = 128 (identity).** One CTA holds A (M, K) and B (N, K) in its SMEM and the
accumulator in its TMEM, with row `m` → lane `m` and column `n` → a TMEM column. C occupies all 128
lanes × N columns.

![cta_group::1, M=128: identity — row m maps to TMEM lane m](../img/mma_cg1_m128.svg)

**`cta_group::1`, M = 64.** Still one CTA, but 64 rows don't fill 128 lanes. They are placed as
four runs of 16 rows at a **lane stride of 32**: rows 0–15 → lanes 0–15, rows 16–31 → lanes 32–47,
rows 32–47 → lanes 64–79, rows 48–63 → lanes 96–111 — leaving lanes 16–31, 48–63, 80–95, 112–127
free. A `lane_align` of 16 shifts the whole pattern up by 16, so **two independent M = 64 MMAs can
share the 128 lanes** (one at align 0, one at align 16). Columns are the full N.

![cta_group::1, M=64: four 16-row runs at lane stride 32, gaps free for a second M=64 tile](../img/mma_cg1_m64.svg)

**`cta_group::2`, M = 256.** An even/odd CTA pair in the cluster cooperates, and M splits
contiguously: CTA 0 takes rows 0–127, CTA 1 takes rows 128–255, **each into its own TMEM lanes
0–127, full N columns**. So the (256, N) accumulator is two separate 128-row TMEM regions, one per
CTA — not one buffer spanning both. Each CTA holds its own 128 A-rows in SMEM; the **even CTA**
issues the instruction and commits the completion barrier for the pair (the `cta_mask=3` /
`pair_mask` you see in the GEMM kernels). This is the mode the 2-CTA cluster GEMM in
{ref}`chap_gemm_advanced` uses.

![cta_group::2, M=256: M split contiguously, 128 rows per CTA across the pair](../img/mma_cg2_m256.svg)

**`cta_group::2`, M = 128.** Also a pair, but each CTA takes only **64 M-rows** (CTA 0 rows 0–63,
CTA 1 rows 64–127), and N is split in half: within each CTA the low-N half occupies lanes 0–63 and
the high-N half stacks into lanes 64–127. Both CTAs use all 128 lanes, packing 64 rows × two
N-halves.

![cta_group::2, M=128: 64 rows per CTA with N split across the lower/upper 64 lanes](../img/mma_cg2_m128.svg)

In every case B (N, K) lives in SMEM, and for `cta_group::2` each CTA supplies operands from its
own SMEM (A split by M as above). The accumulator C (M, N) is always f32 in TMEM.

## Block-Scaled MMA (mxfp8 / nvfp4)

Low-precision formats (fp8, fp4) hold accuracy by attaching a **scale factor to each block of K**:
every group of `B` consecutive K-elements shares one scale. A block-scaled MMA therefore has two
operands beyond A and B — scale-factor tensors **SFA (M, SFK)** and **SFB (N, SFK)**, where
`SFK = K / B`.

The block size `B` (the scale-vector granularity) depends on the format:

| Format | data dtype | scale dtype | block `B` |
|---|---|---|---|
| nvfp4 | fp4 | e4m3 | 16 |
| mxfp8 | fp8 | e8m0 | 32 |
| mxf4 | fp4 | e8m0 | 32 |

The math is a per-block dequantize-then-accumulate: the quantized value is multiplied by its
block's scale before the product is summed in f32,

$$D[m,n] \mathrel{+}= \sum_k \big(A_q[m,k]\cdot \mathrm{SFA}[m,\, k/B]\big)\,\big(B_q[n,k]\cdot \mathrm{SFB}[n,\, k/B]\big).$$

For e8m0 scales the factor is an exact power of two (the stored byte is a biased exponent, scale =
$2^{\text{byte}-127}$); nvfp4's e4m3 scales are small floats.

The placement has one new twist: **the scale factors live in TMEM**, not SMEM (the block-scaled
`tcgen05.mma` reads them from TMEM):

| Logical | dtype | Where |
|---|---|---|
| A (M, K) | fp8 / fp4 (packed) | SMEM |
| B (N, K) | fp8 / fp4 (packed) | SMEM |
| SFA (M, SFK) | e8m0 / e4m3 | **TMEM** |
| SFB (N, SFK) | e8m0 / e4m3 | **TMEM** |
| C (M, N) | f32 | TMEM |

So the scale factors take a detour the data operands don't: they are TMA-loaded into SMEM, then
copied SMEM → TMEM with `tcgen05.cp` (a `Tx.copy_async` whose destination is a `tmem` buffer)
before the MMA. In TMEM they are packed as `uint32` cells, with logical row `r` landing at
TMEM lane `r % 128`, column `col0 + r / 128`.

Under **`cta_group::2`** the scale factors split exactly like the data they describe: **SFA follows
A** — each CTA holds the M-half matching its A rows — while **SFB is multicast to both CTAs**, since
each CTA's M-half multiplies against the same per-N column scales. In the kernels this is the "load
SFA per-CTA (single-CTA mask), broadcast SFB (pair mask)" pattern.

![Block-scaled MMA placement: A/B packed in SMEM; SFA, SFB, and C in TMEM, with SFA split by M and SFB multicast across the CTA pair](../img/mma_block_scaled.svg)
