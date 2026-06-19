(chap_performance)=
# What Makes a Kernel Fast

:::{admonition} Overview
:class: overview

- The roofline model bounds a kernel by either memory bandwidth or compute, decided by its *arithmetic intensity* (FLOPs per byte moved).
- Low arithmetic intensity means memory-bound: raise it (bigger tiles, fusion) or you cannot reach peak FLOPs.
- The main lever for speed is *overlap* — running data movement and compute at once — limited by occupancy and resource pressure.
:::

**Motivation.** You can pour weeks into a kernel and still not know whether it is any good, because
"fast" means nothing without a ceiling: 330 TFLOP/s sounds impressive until you learn the same GPU
can sustain on the order of 2 PFLOP/s, leaving that kernel at one-sixth of what the silicon allows.
The roofline model gives you that ceiling *before you write a line of code* — it pins down the
speed-of-light for this specific kernel and tells you which resource, memory bandwidth or compute,
is the one stopping it. With that in hand you optimize the resource that actually binds instead of
guessing: a memory-bound kernel will not get faster from better math, and a compute-bound one will
not get faster from fewer bytes. This chapter builds the three ideas the rest of the book leans on
to ask "is this fast?" — arithmetic intensity, the roofline, and overlap.

The numbers here are for the NVIDIA B200. Following the convention of {ref}`chap_background`,
we use order-of-magnitude ceilings: **on the order of 2 PFLOP/s** dense fp16/bf16 tensor-core
throughput and **on the order of 8 TB/s** of HBM3e bandwidth. The exact values depend on SKU
and clock, so treat them as round numbers for reasoning, not datasheet guarantees.

## The Roofline Model

A kernel does two things: it moves bytes and it does math. Whichever of those it cannot do fast
enough is the bottleneck. The *roofline* model — introduced by Williams, Waterman, and Patterson
("Roofline: An Insightful Visual Performance Model for Multicore Architectures," *Communications of
the ACM*, 2009) — bounds attainable performance by the slower of the two:

$$\text{attainable FLOP/s} = \min\big(\underbrace{\text{peak FLOP/s}}_{\text{compute roof}},\ \underbrace{\text{bandwidth} \times \text{AI}}_{\text{memory roof}}\big)$$

where **arithmetic intensity (AI)** is the ratio of useful floating-point work to bytes moved at
the memory level whose bandwidth sets the roof (HBM bytes for the HBM roofline; an L2 or SMEM
roofline counts traffic at that level instead):

$$\text{AI} = \frac{\text{FLOPs}}{\text{bytes moved}} \quad [\text{FLOP/byte}]$$

Plotting attainable performance against arithmetic intensity gives two ceilings — a sloped *memory roof*
(bandwidth × arithmetic intensity) and a flat *compute roof* (peak FLOP/s) — meeting at the **ridge point**:

![Roofline for the B200, with example workloads](../img/roofline.png)

The ridge point splits all kernels into two regimes. For the B200 it sits at roughly
`2000 / 8 ≈ 250` FLOP/byte. A kernel whose arithmetic intensity is **below** the ridge can never
reach peak compute no matter how good the code is — it is *memory-bound*, and the only lever is
moving fewer bytes (or moving them faster). A kernel **above** the ridge is *compute-bound* — moving the
bytes is no longer the bottleneck (not literally free), and the job is to keep the tensor cores busy. Knowing which side a
kernel falls on tells you, before you write a line of code, which resource you are fighting.

## Arithmetic Intensity of Real Workloads

Which side of the ridge a kernel lands on is mostly decided by the algorithm, not the code — you
can often read it off the math before implementing anything. Here is where the workloads in this
book fall:

- **Elementwise and reductions** (GELU, RMSNorm) read and write large tensors but do only a few FLOPs per element. Their
  arithmetic intensity is far below the ridge point — deep in the memory-bound region. The best such a kernel can do is
  saturate HBM bandwidth, so the design goals are coalesced/TMA loads and fusion (do more math
  per byte loaded).

- **GEMM** has arithmetic intensity that grows with size. For a square `M=N=K` fp16 matmul, the ideal is

  $$\text{AI} = \frac{2N^3}{(3N^2)\cdot 2\,\text{bytes}} = \frac{N}{3}\ \text{FLOP/byte}.$$

  This is an *ideal* upper bound — it assumes A and B are read once, C is written once (β = 0),
  on-chip reuse is perfect, and there is no metadata or padding traffic; a real kernel moves
  somewhat more. Even so, at `N = 4096` it is ≈ 1365 FLOP/byte — far to the right of the ridge. **GEMM at scale is
  compute-bound.** The ceiling is the tensor-core peak, so the task is reaching that peak (use
  `tcgen05`, keep it fed, overlap everything). It is also why a naive GEMM underperforms — the
  problem allows peak, but a poor implementation leaves the tensor cores idle, sitting orders of
  magnitude below the roof.

- **Attention** sits in between and depends on sequence length and head dimension; Flash
  Attention ({ref}`chap_flash_attention`) is largely an exercise in *raising* arithmetic intensity by keeping
  intermediate tiles in TMEM/SMEM instead of streaming them through HBM.

## When Arithmetic Intensity Is Low

When your kernel sits left of the ridge it is memory-bound: the Tensor Cores will idle no matter how
good the compute code is, because the bottleneck is bytes, not FLOPs. There are two responses, and
they apply in order.

**First, try to raise arithmetic intensity — do more work per byte.** This is the higher-leverage move because it
can move the kernel across the ridge, turning a memory-bound problem into a compute-bound one.
Three techniques do it:

- *Fuse.* The biggest source of low arithmetic intensity is writing an intermediate to global memory and reading it
  straight back. Fusing the producer and consumer keeps that intermediate in registers or SMEM, so
  the bytes never hit HBM. Fusing an elementwise epilogue into a GEMM, or a normalization into the
  op that feeds it, removes whole HBM round trips. Flash Attention is the extreme case: it fuses
  $QK^\top$, softmax, and the $PV$ product so the large score matrix `S` is never written to HBM —
  that fusion is what moves attention rightward on the roofline.
- *Block for reuse.* Load a tile into SMEM or registers and use it for many operations before
  evicting it. Reuse is exactly what gives GEMM its high arithmetic intensity; any op with reuse benefits the same way.
- *Use a smaller dtype.* fp16/fp8/fp4 move fewer bytes per element, which raises arithmetic intensity (FLOPs
  per byte) and cuts the bandwidth the kernel needs — though for block-scaled fp8/fp4 the scale-factor
  metadata and dequantization add some traffic back, so the gain is a little less than the raw byte ratio.

**Second, if the arithmetic intensity is irreducible — accept the memory roof and saturate it.** Sometimes there is
simply no work to add: a pure copy, or a single-pass elementwise or reduction over a large tensor,
has no reuse to exploit, and its best possible performance *is* the memory roof. The job then is
not to beat the roof but to actually reach it, which comes down to a few mechanical concerns:

- Move each byte once — no redundant reads. Read shared inputs through L2, and when several CTAs need
  the same tile, multicast it once with a cluster/TMA load instead of each CTA re-reading it.
- Use wide, coalesced/vectorized loads and TMA bulk transfers so the memory system runs at full width.
- Keep enough memory requests in flight (async copies, sufficient occupancy) to hide latency and
  actually reach peak bandwidth.
- Drop precision where the algorithm tolerates it (store in fp8/fp4) to move fewer bytes.

Once a memory-bound kernel is at the memory roof, it is done — no compute technique helps, and the
only further gains come from changing the algorithm so it moves less data.

## The Optimization Ladder

The roofline tells us what is *possible*; it says nothing about how hard the gap is to close.
Theory says a 4096³ fp16 GEMM is compute-bound and could approach the ~2 PFLOP/s ceiling, but
reaching that ceiling is a separate problem. Here is what the implementations in Part III measure on
a B200 — the same algorithm, climbing toward the roof one technique at a time.

The single biggest jump is the first one: switching from CUDA-core tiling to the **tensor-core +
TMA** path. At `M=N=K=2048`:

| Implementation | TFLOP/s | vs. naive |
|---|---:|---:|
| Naive tiled GEMM (no tensor core) | 2.9 | 1× |
| TMA load + `tcgen05` MMA | 330 | **116×** |

A single step buys two orders of magnitude — but it lands at 330 TFLOP/s, still far from the roof.
From there, asynchrony and scheduling do the rest of the work. At `M=N=K=4096`:

| Step | Technique | TFLOP/s | % of ~2 PFLOP/s peak |
|---|---|---:|---:|
| 5 | Software pipeline (depth 2) | 639 | ~32% |
| 6 | Persistent kernel + tile scheduler | 723 | ~36% |
| 7 | Warp specialization | 603 | ~30% |
| 8 | 2-CTA cluster | 1057 | ~53% |
| 9 | Multi-consumer | 1145 | ~57% |

These 4096³ rates are the throughput form of the wall-clock times in the End-to-End table of
{ref}`chap_gemm_advanced` (Steps 7–9 here), so the two tables describe the same runs.

![GEMM optimization journey](../img/gemm_perf.png)

Two lessons set up the rest of the book. First, **the ceiling is real**: the optimized kernel
reaches roughly 57% of the fp16 peak at 4096³ (and more at larger sizes), close to the vendor
library (cuBLAS is ~62% here) — the remaining gap is hardware reality, not missing techniques.
Second, **the steps are not all monotone** (warp specialization at step 7 momentarily *trades*
throughput for a structure that later steps exploit) — an optimization is justified by what it
unlocks, not always by its immediate number.

## Overlap Is the Lever

On the compute-bound side, raw FLOP/s is fixed by the tensor cores — you cannot make them multiply
faster — so the only way to climb is to stop *waiting*. A kernel that loads a tile, then computes on
it, then stores the result spends most of its time idle while one hardware unit waits for another.
The fix is **overlap**: while the tensor core works on tile `k`, the TMA engine is already fetching
tile `k+1`, and the epilogue is draining tile `k-1`, so no unit sits idle waiting on its neighbor.

This is why Blackwell exposes the load (TMA), compute (`tcgen05`), and store paths as
*independent asynchronous engines*, coordinated by mbarriers ({ref}`chap_async_barriers`).
Software pipelining (step 5) and warp specialization (step 7) are the two structural patterns
for arranging that overlap; Part III builds on both.

## Occupancy and Resource Pressure

Overlap is not the only way to hide latency. The classic alternative is **occupancy** — how many
warps/warpgroups an SM keeps resident so it can hide latency by switching to a ready warp whenever
one stalls. Occupancy is capped by per-SM resources, namely registers and shared memory, and the
modern kernels in this book make a deliberate trade: they spend a lot of SMEM (on multi-stage
tile buffers) and registers, so they often run at *low* occupancy and hide latency through explicit
async overlap instead of through many resident warps. Both mechanisms matter — classic
occupancy-driven latency hiding, and the explicit pipelining this book focuses on — because real
kernels lean on whichever the resource budget allows.

## What This Buys You Later

The rest of the book is a sequence of concrete answers to one recurring question — "which roof am I
under, and what moves me toward it?" The pattern repeats per chapter:

- Memory-bound kernels → coalesced/TMA loads and fusion.
- Compute-bound GEMM (Part III) → tensor cores to raise the ceiling, then TMA + pipelining +
  specialization to reach it.
- Flash Attention (Part IV) → raise arithmetic intensity by keeping tiles on-chip, then apply the same overlap
  toolkit.

So whenever a kernel underperforms, the first move is not to guess at code changes but to return to
this chapter's question: compute the arithmetic intensity, find the roof, and optimize the resource
that is actually binding.
