(chap_performance)=
# What Makes a Kernel Fast

:::{admonition} Overview
:class: overview

- The roofline model bounds a kernel by either memory bandwidth or compute, decided by its *arithmetic intensity* (FLOPs per byte moved).
- Low arithmetic intensity means memory-bound: raise it (bigger tiles, fusion) or you cannot reach peak FLOPs.
- The main lever for speed is *overlap* (running data movement and compute at once), limited by occupancy and resource pressure.
:::

You can pour weeks into a kernel and still not know whether it is any good, because
"fast" means nothing without a ceiling: 330 TFLOP/s sounds impressive until you learn the same GPU
can sustain on the order of 2 PFLOP/s, leaving that kernel at one-sixth of what the silicon allows.
The roofline model gives you that ceiling *before you write a line of code*: it pins down the
speed-of-light for this specific kernel and tells you which resource, memory bandwidth or compute,
is the one stopping it. With that in hand you optimize the resource that actually binds instead of
guessing: a memory-bound kernel will not get faster from better math, and a compute-bound one will
not get faster from fewer bytes. This chapter builds the three ideas the rest of the book leans on
to ask "is this fast?": arithmetic intensity, the roofline, and overlap.

The numbers here are for the NVIDIA B200. Following the convention of {ref}`chap_background`,
we use order-of-magnitude ceilings: **on the order of 2 PFLOP/s** dense fp16/bf16 tensor-core
throughput and **on the order of 8 TB/s** of HBM3e bandwidth. The exact values depend on the
specific GPU model and its clock, so treat them as round numbers for reasoning, not datasheet
guarantees.

## The Roofline Model

Every kernel does two things, no matter how complicated it looks: it moves bytes between memory and
the chip, and it does arithmetic on those bytes. At any moment one of these two activities is the
one holding the kernel back, and whichever it is sets the ceiling on how fast the kernel can run.
This simple observation is the heart of the *roofline* model, introduced by Williams, Waterman, and
Patterson ("Roofline: An Insightful Visual Performance Model for Multicore Architectures,"
*Communications of the ACM*, 2009). The model bounds attainable performance by the slower of the
two paths:

$$\text{attainable FLOP/s} = \min\big(\underbrace{\text{peak FLOP/s}}_{\text{compute roof}},\ \underbrace{\text{bandwidth} \times \text{AI}}_{\text{memory roof}}\big)$$

The quantity that decides which path wins is **arithmetic intensity (AI)**: the ratio of useful
floating-point work to the bytes moved at whichever memory level sets the roof. For an HBM roofline
those are HBM bytes; an L2 or SMEM roofline would count traffic at that level instead.

$$\text{AI} = \frac{\text{FLOPs}}{\text{bytes moved}} \quad [\text{FLOP/byte}]$$

If we plot attainable performance against arithmetic intensity, the two terms of the minimum become
two ceilings: a sloped *memory roof* that climbs with bandwidth × arithmetic intensity, and a flat
*compute roof* fixed at peak FLOP/s. The two lines cross at the **ridge point**.

![Roofline for the B200, with example workloads](../img/roofline.png)

That ridge point is where the model earns its keep, because it splits every kernel into one of two
regimes. For the B200 it sits at roughly `2000 / 8 ≈ 250` FLOP/byte. A kernel whose arithmetic
intensity falls **below** the ridge can never reach peak compute, no matter how clever the code is:
it is *memory-bound*, and the only lever left is to move fewer bytes, or to move them faster. A
kernel **above** the ridge is *compute-bound*. Here moving the bytes is no longer the limiting
factor (though it is never literally free), and the job becomes keeping the tensor cores busy. The
payoff is that knowing which side a kernel lands on tells you, before you write a single line of
code, which resource you are actually fighting.

## Arithmetic Intensity of Real Workloads

We have seen that arithmetic intensity decides which side of the ridge a kernel lands on, so the
natural next question is how to find a kernel's arithmetic intensity in the first place. The
encouraging answer is that it is mostly a property of the algorithm, not of the implementation. You
can often work it out from the math of what a kernel computes long before you write the kernel
itself, and that estimate already tells you which side of the ridge you will be fighting on. Let us
walk through where the workloads in this book fall.

- **Elementwise and reductions** (GELU, RMSNorm) read and write large tensors but do only a handful
  of FLOPs per element. Their arithmetic intensity lands far below the ridge point, deep in the
  memory-bound region, so the very best one of these kernels can hope for is to saturate HBM
  bandwidth. That sets the agenda: coalesced or TMA loads, and fusion to squeeze more math out of
  each byte loaded.

- **GEMM** is the opposite story, because its arithmetic intensity grows with problem size. For a
  square `M=N=K` fp16 matmul, the ideal works out to

  $$\text{AI} = \frac{2N^3}{(3N^2)\cdot 2\,\text{bytes}} = \frac{N}{3}\ \text{FLOP/byte}.$$

  Read this as an *ideal* upper bound rather than a promise: it assumes A and B are read exactly
  once, C is written once (β = 0), on-chip reuse is perfect, and no metadata or padding traffic
  creeps in, and a real kernel always moves somewhat more. Even with that caveat, at `N = 4096` the
  figure is ≈ 1365 FLOP/byte, far to the right of the ridge, so **GEMM at scale is compute-bound.**
  The ceiling is the tensor-core peak, which means the whole task is reaching that peak: use
  `tcgen05`, keep it fed, and overlap everything. This is also exactly why a naive GEMM disappoints.
  The problem permits peak performance, but a poor implementation leaves the tensor cores idle and
  sits orders of magnitude below the roof.

- **Attention** sits in between, with an arithmetic intensity that depends on sequence length and
  head dimension. Flash Attention ({ref}`chap_flash_attention`) is, at its core, an exercise in
  *raising* that arithmetic intensity by keeping intermediate tiles in TMEM/SMEM instead of
  streaming them out through HBM and back.

## When Arithmetic Intensity Is Low

Suppose your kernel sits left of the ridge, so it is memory-bound. The tensor cores will idle no
matter how good the compute code is, simply because the bottleneck is bytes, not FLOPs. You have two
responses available, and it helps to think of them in order: first try to escape the memory-bound
region entirely, and if you cannot, settle for making the most of it.

**The first response is to raise arithmetic intensity, to do more work per byte.** This is the
higher-leverage move, because if it succeeds it carries the kernel across the ridge and converts a
memory-bound problem into a compute-bound one. Three techniques get you there.

- *Fuse.* The single biggest source of low arithmetic intensity is writing an intermediate result
  out to global memory only to read it straight back. Fusing the producer and the consumer keeps
  that intermediate in registers or SMEM, so the bytes never touch HBM at all. Fusing an elementwise
  epilogue into a GEMM, or folding a normalization into the op that feeds it, eliminates entire HBM
  round trips. Flash Attention is the extreme case of this idea: it fuses $QK^\top$, softmax, and
  the $PV$ product so that the large score matrix `S` is never written to HBM, and that fusion is
  precisely what pushes attention rightward on the roofline.
- *Block for reuse.* Load a tile into SMEM or registers once, then use it for many operations before
  evicting it. Reuse is exactly what gives GEMM its high arithmetic intensity, and any op with reuse
  to exploit benefits the same way.
- *Use a smaller dtype.* Moving to fp16, fp8, or fp4 carries fewer bytes per element, which both
  raises arithmetic intensity (FLOPs per byte) and cuts the bandwidth the kernel demands. The one
  caveat is that block-scaled fp8/fp4 brings back some traffic for the scale-factor metadata and the
  dequantization, so the real gain is a little smaller than the raw byte ratio suggests.

**The second response applies when the arithmetic intensity is simply irreducible: accept the memory
roof and saturate it.** Sometimes there is no work left to add. A pure copy, or a single-pass
elementwise or reduction over a large tensor, has no reuse to exploit, and its best possible
performance genuinely *is* the memory roof. The goal then shifts from beating the roof to actually
reaching it, which comes down to a handful of mechanical concerns.

- Move each byte once, with no redundant reads. Pull shared inputs through L2, and when several CTAs
  need the same tile, multicast it once with a cluster/TMA load instead of letting each CTA re-read
  it on its own.
- Use wide, coalesced or vectorized loads and TMA bulk transfers, so the memory system runs at its
  full width.
- Keep enough memory requests in flight, through async copies and sufficient occupancy, to hide
  latency and actually reach peak bandwidth.
- Drop precision wherever the algorithm tolerates it, storing in fp8/fp4 to move fewer bytes.

Once a memory-bound kernel reaches the memory roof, it is done. No compute technique can help, and
the only way to go faster from there is to change the algorithm so that it moves less data in the
first place.

## The Optimization Ladder

The roofline tells us what is *possible*, but it says nothing about how hard the gap will be to
close. Theory tells us a 4096³ fp16 GEMM is compute-bound and could in principle approach the
~2 PFLOP/s ceiling; getting there is a wholly separate engineering problem. To see what closing the
gap actually looks like, consider what the implementations in Part III measure on a B200, where the
same algorithm climbs toward the roof one technique at a time.

The single biggest jump is the very first one, switching from CUDA-core tiling to the
**tensor-core + TMA** path. At `M=N=K=2048`:

| Implementation | TFLOP/s | vs. naive |
|---|---:|---:|
| Naive tiled GEMM (no tensor core) | 2.9 | 1× |
| TMA load + `tcgen05` MMA | 330 | **116×** |

That one step buys two orders of magnitude, and yet it lands at 330 TFLOP/s, still far from the
roof. From there it is asynchrony and scheduling that do the rest of the work. At `M=N=K=4096`:

| Step | Technique | TFLOP/s | % of ~2 PFLOP/s peak |
|---|---|---:|---:|
| 5 | Software pipeline (depth 2) | 639 | ~32% |
| 6 | Persistent kernel + tile scheduler | 723 | ~36% |
| 7 | Warp specialization | 603 | ~30% |
| 8 | 2-CTA cluster | 1057 | ~53% |
| 9 | Multi-consumer | 1145 | ~57% |

These 4096³ rates are the throughput form of the wall-clock times in the End-to-End table of
{ref}`chap_gemm_advanced` (Steps 7–9 here), so the two tables describe the same runs.

The figure below turns that table into the optimization trajectory we will revisit in Part III: one
kernel family, one added technique per step, and the throughput each step buys.

![GEMM optimization journey](../img/gemm_perf.png)

Two lessons from this ladder set up much of the rest of the book. The first is that **the ceiling is
real**. The optimized kernel reaches roughly 57% of the fp16 peak at 4096³ (and more at larger
sizes), which puts it close to the vendor library (cuBLAS sits at about 62% here), and the gap that
remains is hardware reality rather than some technique we forgot to apply. The second lesson is that
**the steps are not all monotone**. Warp specialization at step 7 momentarily *trades away*
throughput in exchange for a structure that the later steps go on to exploit, which is a reminder
that an optimization is justified by what it unlocks, not always by the number it posts on its own.

## Overlap Is the Lever

The ladder showed the gains piling up but not the single idea behind them; nearly every step past the
tensor-core switch is a different way of arranging one thing, and that thing is overlap. To see why,
return to a compute-bound GEMM that already uses tensor cores. There the remaining gap usually comes
from idle time between load, compute, and store stages rather than from weak arithmetic throughput. The raw
FLOP/s figure is fixed by the tensor cores (you cannot make them multiply any faster), so the only
remaining way to go faster is to stop *waiting*. A kernel that loads a tile, then computes on it,
then stores the result spends most of its life idle, with one hardware unit standing around while
another finishes its turn. The fix is **overlap**: while the tensor core works on tile `k`, the TMA
engine is already fetching tile `k+1`, and the epilogue is draining tile `k-1`, so that no unit
ever sits idle waiting on its neighbor.

This is exactly why Blackwell exposes the load (TMA), compute (`tcgen05`), and store paths as
*independent asynchronous engines*, coordinated by mbarriers ({ref}`chap_async_barriers`). Software
pipelining (step 5) and warp specialization (step 7) are the two structural patterns for arranging
that overlap, and Part III builds on both.

## Occupancy and Resource Pressure

Overlap is not the only way to hide latency, and it is worth knowing the classic alternative. That
alternative is **occupancy**: the number of warps or warpgroups an SM keeps resident, so that
whenever one warp stalls the SM can simply switch to another that is ready to run. Occupancy is
capped by the per-SM resources each warp consumes, chiefly registers and shared memory. The modern
kernels in this book make a deliberate trade against this: they spend a great deal of SMEM on
multi-stage tile buffers, along with plenty of registers, which means they often run at *low*
occupancy and hide latency through explicit async overlap rather than through a large pool of
resident warps. Both mechanisms are worth understanding, the classic occupancy-driven latency
hiding and the explicit pipelining this book emphasizes, because real kernels reach for whichever
one the resource budget happens to allow.

## What This Buys You Later

With these three ideas in hand, the rest of the book reads as a sequence of concrete answers to one
recurring question: which roof am I under, and what moves me toward it? The same pattern plays out
chapter after chapter, just with different workloads.

- For memory-bound kernels, the answer is coalesced or TMA loads together with fusion.
- For compute-bound GEMM in Part III, the answer is tensor cores to raise the ceiling, followed by
  TMA, pipelining, and specialization to reach it.
- For Flash Attention in Part IV, the answer is to raise arithmetic intensity by keeping tiles
  on-chip, and then to apply the very same overlap toolkit.

So whenever a kernel underperforms, the first move is not to start guessing at code changes. It is
to come back to this chapter's question: compute the arithmetic intensity, find the roof, and then
optimize the resource that is actually binding.
