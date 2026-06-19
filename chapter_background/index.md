(chap_background)=
# GPU Execution Model

:::{admonition} Overview
:class: overview

- A kernel runs over a thread hierarchy (thread → warp → warpgroup → CTA → cluster → grid) across distinct memory spaces (registers, SMEM, GMEM, TMEM).
- Compute splits into CUDA cores and Tensor Cores; dedicated engines like TMA move the data that feeds them.
- Every later optimization serves one tile pipeline — load (GMEM → SMEM), compute (SMEM → TMEM), epilogue (TMEM → registers → GMEM) — and aims to keep the compute and data-movement engines busy at once.
:::

**Motivation.** Two kernels can compute the same result over the same numbers and
still differ in speed by an order of magnitude — and the gap is almost never the arithmetic, it is
how well the code fits the chip underneath. So before we write a single kernel, this chapter brings
that chip into focus: the hierarchy of threads that run the work, the distinct memory spaces they
move data through, and the handful of compute and data-movement engines that do the heavy lifting.
Nearly every optimization later in the book is a way of arranging work across those three things. We
assemble the picture here — the thread hierarchy, the compute units, the memory spaces, and how a
GEMM flows across them — and the `tcgen05` compute path ({ref}`chap_tensor_cores`), TMA data
movement ({ref}`chap_tma`), and the mbarrier model ({ref}`chap_async_barriers`) each build on it in
their own chapters.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the Blackwell SM — its warps/warpgroups, shared memory, Tensor Memory, and the
Tensor Core and TMA engines.*

## The Execution Hierarchy

A GPU groups threads at several levels because cooperation happens at different scales: the lanes
of a warp execute in lockstep, the threads of a CTA share that CTA's shared memory, and the CTAs of
a cluster synchronize across SMs. On Blackwell the levels are these:

```{raw} html
<iframe src="../demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; min-width:900px; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: click a level — thread → warp → warpgroup → CTA → cluster → grid.*

- **Thread** — the scalar unit of execution, identified by a lane ID within its warp.
- **Warp** — 32 threads executing in SIMT (*single instruction, multiple threads*): the lanes
  issue the same instruction together, but each has its own registers and can be masked off
  independently (so lanes can take different branches).
- **Warpgroup** — 4 consecutive warps (128 threads). Introduced on Hopper as the unit for
  warpgroup-level MMA (`wgmma`), it is also the cooperation unit for Blackwell Tensor Memory
  access, where the 128 threads cooperatively move a TMEM tile to or from registers.
- **CTA** (*Cooperative Thread Array*, a.k.a. a CUDA thread block) — the basic scheduling unit.
  A CTA runs on a single SM and owns a private shared-memory allocation within it; multiple
  resident CTAs share the SM's shared-memory capacity.
- **Cluster** — a group of cooperating CTAs (across SMs) that can synchronize and access each
  other's shared memory (distributed shared memory).

These levels matter because Blackwell operations are **not all issued by the same group of
threads**. A TMA copy is issued by one thread and
finished by hardware; a TMEM→register load is warpgroup-distributed (four warp-collective
`tcgen05.ld` instructions, one per warp's 32 TMEM lanes); a `tcgen05` MMA is committed
by one elected thread; a clustered MMA spans two CTAs. Each operation, in other words, has a
natural granularity, and which threads run it is the operation's **scope** — the first of the
book's three recurring knobs (scope, layout, dispatch).

## Compute: CUDA Cores and Tensor Cores

An SM has two kinds of math engine, and the split between them shapes how every kernel is written:

- **CUDA cores** — general-purpose SIMT ALUs that execute scalar/vector instructions for indexing,
  elementwise math, reductions, and control flow.
- **Tensor Cores** — fixed-function units that execute a dense matrix multiply-accumulate at *tile*
  granularity in a single instruction: $D = AB + C$.

Dense linear algebra (GEMM, convolution, attention) reaches peak throughput only on the Tensor
Cores, and what changes from one GPU generation to the next is *how* they are programmed and
*where* their results live: the asynchronous warpgroup MMA (`wgmma.mma_async`) arrived with Hopper, and
Blackwell's fifth-generation Tensor Core (`tcgen05`, with accumulators in Tensor Memory) is covered
in {ref}`chap_tensor_cores`.

## Memory Spaces

No single memory can be both large and fast, so a GPU offers several, and a kernel moves data
through them, each with its own capacity, latency, and access rules:

| Memory | Ownership | Role | Notes |
|--------|-----------|------|-------|
| **Global (GMEM)** | Device-wide | Persistent tensor storage | Large HBM, shared by all SMs |
| **Shared (SMEM)** | Per-CTA (one SM) | Tile staging | Low-latency scratchpad; up to 228 KB/SM on B200 |
| **Tensor Memory (TMEM)** | Per-CTA | MMA accumulator storage | New on Blackwell; used by `tcgen05` |
| **Register File (RF)** | Per-thread | Scalars and per-thread tile fragments | Fast; holds epilogue/temp values |

These spaces form a path. The data path of almost every kernel in this book is **GMEM → SMEM →
(compute) → registers → GMEM**, with TMEM holding accumulators in the middle for tensor-core
kernels.

![Memory dataflow across the hierarchy](../img/memory_dataflow.png)

**Tensor Memory (TMEM)** is the one memory here without a pre-Blackwell analog; its details are in
{ref}`chap_tensor_cores`. Earlier GPUs kept large MMA accumulators in registers; Blackwell instead
writes `tcgen05` accumulator output to TMEM, a CTA-scoped 2D scratchpad — 128 lanes × up to 512
32-bit columns per CTA (the array physically lives on the SM) — and the kernel then explicitly reads
TMEM into registers for the epilogue. That extra step is not free, and two consequences of it
show up everywhere later: TMEM reads are **explicit and warpgroup-distributed** (four warp-collective
`tcgen05.ld` instructions, one per warp's 32 TMEM lanes), and TMEM must be
**explicitly allocated and freed**.

## CTA Clusters

A CTA runs on one SM with its own shared memory, but a single CTA's SMEM budget is finite, and
large tiles often need more operand storage or reuse than one block can supply. Hopper's answer was
the **thread block cluster**: a group of CTAs that cooperate more tightly than independent blocks,
able to synchronize together and access each other's shared memory (distributed shared memory,
DSMEM). Blackwell builds on clusters with dynamic
scheduling ({ref}`chap_clc`) and 2-CTA cooperative MMA.

The key new capability is **distributed shared memory (DSMEM)** — the ability of cluster CTAs to
reach each other's shared memory — and the hardware exposes it in two parts. The first is an
*address*: the `mapa` instruction maps a local SMEM pointer to a peer CTA's rank, returning the
same offset in that CTA's SMEM. The second is a *transfer*: a single thread can bulk-copy a tile
from its own SMEM into a peer's, signalling a completion barrier ({ref}`chap_async_barriers`) when
the bytes land — a cluster-scoped `cp.async.bulk`. The 2-CTA cluster GEMM in Part III uses exactly
this to share operand tiles across the pair without a round trip through global memory.

![A CTA cluster sharing distributed shared memory](../img/cta_cluster.png)

Two cluster features built on DSMEM recur in the GEMM chapters: **2-CTA cooperative MMA**, where
two CTAs contribute SMEM operands to one larger MMA tile, and **TMA multicast**, where one TMA load
delivers the same GMEM tile to several CTAs and so cuts redundant global traffic.

## The GEMM Data Pipeline

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the load → MMA → epilogue pipeline on Blackwell, and how the stages overlap.*

With the thread hierarchy, the engines, and the memory spaces in place, we can trace how they work
together on a GEMM. A GEMM tile flows across the hardware in three stages:

1. **Load.** A TMA copy ({ref}`chap_tma`) streams an A/B operand tile from GMEM into SMEM. One
   thread issues it after `mbarrier.arrive.expect_tx(bytes)` records the expected byte (tx) count;
   the TMA engine issues `complete-tx` as bytes land, and the barrier's phase flips only once both
   the arrival count and the tx-count are satisfied.
2. **Compute.** A `tcgen05` MMA ({ref}`chap_tensor_cores`) reads the SMEM operands and
   accumulates into a TMEM tile. It is issued by one elected thread and signals a barrier when done.
3. **Epilogue.** The warpgroup reads the TMEM accumulator into registers, casts it to the output
   dtype, and stores it back to GMEM (often via SMEM staging + a TMA store).

These stages look sequential, but the difference between a slow and a fast kernel is **overlap**. A naive kernel runs the steps in order — load,
wait, compute, wait, store — leaving each engine idle while it waits for the previous one. A fast
kernel instead pipelines them: while the Tensor Core computes on tile `k`, the TMA engine is
already fetching tile `k+1` and the epilogue is draining tile `k-1`, so all three engines stay
busy at once. Making three asynchronous engines hand work to each other safely is the job of the
barrier/phase model ({ref}`chap_async_barriers`), and Part III's GEMM ladder is built on it.

## Numbers to Keep in Mind

Every design decision in this book comes down to a handful of capacities and speeds. The orders of
magnitude below (B200, approximate) explain why kernels stage only a few operand tiles, budget TMEM
carefully, and work hard to keep the Tensor Cores fed:

| Quantity | Value (B200, approx.) | Where it shows up |
|:--|:-:|:--|
| Streaming multiprocessors per GPU | ~148 | grid sizing, persistent schedulers |
| Shared memory per SM | up to 228 KB | SMEM budget for pipeline depth |
| Tensor Memory per SM | 128 rows × 512 cols (32-bit) | accumulator budget |
| Registers per thread (max) | 255 × 32-bit | why big accumulators don't stay in RF |
| Tensor Core peak @ fp16/bf16 (dense) | order of 2 PFLOP/s | the compute roof ({ref}`chap_performance`) |
| Tensor Core peak @ fp8 / fp4 (dense) | several PFLOP/s | mixed-precision GEMM |
| HBM3e bandwidth | order of 8 TB/s | the memory roof |
| One fp16 128×64 tile in SMEM | 16 KB | example staged operand-tile size |

Exact peak numbers depend on SKU, clock, and sparsity mode — use the table as *scale*, not a
performance model. The takeaway: SMEM holds only a handful of staged tiles, TMEM must be
budgeted, and tensor-core throughput is high enough that data movement *must* overlap compute.
