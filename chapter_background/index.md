(chap_background)=
# GPU Execution Model

:::{admonition} Overview
:class: overview

- A kernel runs over a thread hierarchy (thread → warp → warpgroup → CTA → cluster → grid) across distinct memory spaces (registers, SMEM, GMEM, TMEM).
- Compute splits into CUDA cores and Tensor Cores; dedicated engines like TMA move the data that feeds them.
- A kernel is a pipeline that stages data through these memory spaces and hands work between independent compute and data-movement engines; the recurring goal is to keep those engines busy at once.
:::

To write fast GPU programs, it is important to understand the hardware
itself and how code runs on that hardware. This chapter gives an overview of the GPU execution
model: the thread hierarchy that executes the work, the memory spaces that hold and move the data,
and the compute and data-movement engines that do the heavy lifting. We first introduce these
pieces one by one, then put them together in a GEMM pipeline so it is clear how data and execution
flow through the hardware. Nearly every optimization later in the book is some way of arranging
work across those same pieces.

Modern GPUs also contain many specialized hardware units. To give a first taste, the interactive
demo below shows the main elements inside a Blackwell streaming multiprocessor before we zoom in on each
part. You can click into each part to see its details.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the Blackwell SM, showing its warps/warpgroups, shared memory, Tensor Memory, and the
Tensor Core and TMA engines.*

## The Execution Hierarchy

We begin with the threads that do the work. A GPU does not present its thousands of threads as one
flat pool. Instead it groups them into a nested hierarchy, and it does so because cooperation happens
at several different scales at once. Each level exists to make cooperation cheap at one of those
scales. The following figure shows the hierarchy on Blackwell; you can click into each level to
highlight it.

```{raw} html
<iframe src="../demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; min-width:900px; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: click a level: thread → warp → warpgroup → CTA → cluster → grid.*

- **Thread**: the scalar unit of execution. Each thread has its own program counter and its own
  registers, and it is identified by a lane ID within its warp.
- **Warp**: 32 threads that execute in SIMT (*single instruction, multiple threads*). The lanes of
  a warp issue the same instruction together, yet each keeps its own registers and can be masked off
  on its own, which is what lets the lanes of a single warp follow different branches.
- **Warpgroup**: four consecutive warps, or 128 threads. Hopper introduced the warpgroup as the
  unit that issues warpgroup-level MMA (`wgmma`), and on Blackwell it takes on a second role: it is
  the cooperation unit for Tensor Memory access, where the 128 threads together move a TMEM tile into
  or out of registers.
- **CTA** (*Cooperative Thread Array*, what CUDA also calls a thread block): the basic unit the
  hardware schedules. A CTA runs on a single SM and owns a private shared-memory allocation inside
  it. Several CTAs can be resident on the same SM at once, and when they are, they divide up that
  SM's shared-memory capacity between them.
- **Cluster**: a group of cooperating CTAs that may live on different SMs. The CTAs in a cluster
  can synchronize with one another and can read and write each other's shared memory, a capability
  known as distributed shared memory.

These levels are worth dwelling on because, unlike on earlier architectures, Blackwell's key
operations are **not all issued by the same group of threads**. A TMA copy is launched by a single
thread and then carried out by hardware. A TMEM-to-register load is warpgroup-distributed: the four
warps cooperate, each moving its own slice of the TMEM tile. A `tcgen05` MMA is committed by one
elected thread, while a clustered MMA spans two CTAs at once. Each operation thus has its own natural granularity, and the set of threads that runs it is
what we call the operation's **scope**, the first of the three recurring design elements (scope, layout, and
dispatch) that this book returns to again and again.

## Memory Spaces

The threads in that hierarchy are only as fast as the data reaching them, so we turn next to where
that data lives. There is no single memory that is at once large and fast; physics forces a trade-off
between capacity and speed. A GPU therefore offers several memories rather than one, each striking that
trade-off at a different point, and a kernel works by moving data through them. Each space has its
own capacity, its own latency, and its own rules for who may access it.

| Memory | Ownership | Role | Notes |
|--------|-----------|------|-------|
| **Global (GMEM)** | Device-wide | Persistent tensor storage | Large HBM, shared by all SMs |
| **Shared (SMEM)** | Per-CTA (one SM) | Tile staging | Low-latency scratchpad; up to 228 KB/SM on B200 |
| **Tensor Memory (TMEM)** | Per-CTA | MMA accumulator storage | New on Blackwell; used by `tcgen05` |
| **Register File (RF)** | Per-thread | Scalars and per-thread tile fragments | Fast; holds epilogue/temp values |

Read in order, these spaces describe a path. The data path of almost every kernel in this book is
**GMEM → SMEM → (compute) → registers → SMEM → GMEM**, and for tensor-core kernels TMEM sits in the
middle of that path, holding the accumulators while the math runs.

Of the four, **Tensor Memory (TMEM)** is the only one with no analog on pre-Blackwell hardware, and
its full details wait until {ref}`chap_tensor_cores`. The motivation for it is worth understanding
now, though. Earlier GPUs kept large MMA accumulators in registers, where they competed for a scarce
resource. Blackwell instead writes `tcgen05` accumulator output to TMEM, a CTA-scoped 2D scratchpad
of 128 lanes by up to 512 32-bit columns per CTA (the array physically lives on the SM). The kernel
then has to read TMEM back into registers explicitly before the epilogue. That extra step is not
free, and two of its consequences will recur throughout the book. The first is that TMEM reads are
**explicit and warpgroup-distributed**, carried out cooperatively by the four warps of a warpgroup.
The second is that TMEM, unlike registers, must be **explicitly allocated and freed**.

### Distributed Shared Memory Across a Cluster

The cluster is the one level of the hierarchy whose members can span several SMs, and that reach buys
a memory capability the other levels lack. A CTA runs on one SM and works out of that SM's shared
memory, but a single CTA's SMEM budget is finite, and large tiles often demand more operand storage,
or more reuse, than one block alone can supply. Hopper's answer was the **thread block cluster**: a
group of CTAs that cooperate more tightly than independent blocks do, in that they can synchronize
together and read and write each other's shared memory, a capability called **distributed shared
memory (DSMEM)**. Blackwell keeps clusters and adds to them, with dynamic scheduling
({ref}`chap_clc`) and 2-CTA cooperative MMA.

DSMEM lets a CTA address and access a peer CTA's shared memory directly. A thread can name a location
in a peer's SMEM and bulk-copy a tile straight from its own SMEM into the peer's, raising a completion
barrier ({ref}`chap_async_barriers`) once the bytes have landed. The 2-CTA cluster GEMM in Part III is
built on exactly this mechanism, using it to share operand tiles across the pair of CTAs without ever
routing them back through global memory.

The figure below shows the extra DSMEM hop that a CTA cluster makes possible; click a piece to see
what each CTA owns and where the cross-CTA read happens.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="A 2-CTA cluster sharing distributed shared memory" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: a 2-CTA cluster, where each CTA owns half of A and half of B, reads the other's B across the
cluster (DSMEM), and the pair produces a 256×256 output tile.*

## Compute: CUDA Cores and Tensor Cores

The threads and the data they move have to meet at an arithmetic unit, and an SM offers two distinct
kinds of math engine rather than one. The division of labor between the two shapes how nearly every
kernel is written, and they play complementary roles.

- **CUDA cores** are general-purpose SIMT ALUs. They run the scalar and vector instructions that
  handle index arithmetic, elementwise math, reductions, and control flow, the glue logic that
  surrounds the heavy matrix work.
- **Tensor Cores** are fixed-function units that perform a dense matrix multiply-accumulate at *tile*
  granularity, computing $D = AB + C$ in a single instruction.

The reason this split matters is that the Tensor Cores deliver vastly more arithmetic throughput than
the CUDA cores, on the order of 10× or more in FLOP/s, so dense linear algebra (GEMM, convolution,
and attention) reaches peak performance only when it runs on the Tensor Cores. Getting performance is
therefore largely a matter of keeping those Tensor Cores fed. What shifts from one GPU generation to the next is *how* the Tensor Cores are
programmed and *where* their results come to rest. Hopper introduced the asynchronous warpgroup MMA
(`wgmma.mma_async`); Blackwell's fifth-generation Tensor Core, `tcgen05`, places its accumulators in
Tensor Memory instead of registers, and we devote {ref}`chap_tensor_cores` to it.

Clusters extend these engines in two ways that recur throughout the GEMM chapters. **2-CTA cooperative
MMA** lets two CTAs each contribute their SMEM operands to a single, larger Tensor Core MMA tile.
**TMA multicast** lets one load by the data-movement engine deliver the same GMEM tile to several CTAs
at once, eliminating the redundant global traffic that separate loads would otherwise incur. Both
build on the distributed shared memory introduced earlier.

## The GEMM Data Pipeline

So far we have introduced the hardware units individually. To see how they work together, we can
use a typical general-purpose matrix multiplication (GEMM) pipeline as an example. The
interactive demo below shows the units involved in a three-stage GEMM tile pipeline; click an action
such as `tma load` to highlight the data path it takes across the hardware units.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the load → MMA → epilogue pipeline on Blackwell; click an action to trace its data path across the hardware units.*

A single GEMM tile flows through three stages.

1. **Load.** A TMA copy ({ref}`chap_tma`) streams an A or B operand tile from GMEM into SMEM. One
   thread issues the copy, recording up front how many bytes are expected to arrive. As the bytes
   land, the TMA engine reports their progress, and a completion barrier flips only once all the
   expected bytes have been delivered.
2. **Compute.** A `tcgen05` MMA ({ref}`chap_tensor_cores`) reads the operand tiles out of SMEM and
   accumulates the product into a TMEM tile. One elected thread issues it, and it signals a barrier
   when the math is done.
3. **Epilogue.** The warpgroup reads the TMEM accumulator back into registers, casts the result to
   the output dtype, and stores it to GMEM, frequently by staging through SMEM and issuing a TMA
   store.

Written out this way the three stages look strictly sequential, but the whole difference between a
slow kernel and a fast one lies in **overlap**. A naive kernel really does run the steps in
order (load, wait, compute, wait, store), and so leaves each engine sitting idle while it waits on
the one before it. A fast kernel pipelines them instead: while the Tensor Core is computing on tile
`k`, the TMA engine is already fetching tile `k+1`, and the epilogue is busy draining tile `k-1`, so
all three engines stay occupied at the same time. Getting three asynchronous engines to hand work off
to one another safely is precisely the job of the barrier and phase model
({ref}`chap_async_barriers`), and the GEMM ladder of Part III is built on top of it.

## What to Read Next

Now that we have seen the high-level picture, we can move on to the chapters that dive deeper into
the main mechanisms:

- {ref}`chap_tensor_cores` explains `tcgen05` compute and Tensor Memory in detail.
- {ref}`chap_tma` covers TMA-based asynchronous data movement.
- {ref}`chap_async_barriers` introduces the mbarrier and phase model that coordinates these engines.
