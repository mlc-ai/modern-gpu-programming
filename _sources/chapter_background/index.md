(chap_background)=
# GPU Execution Model

:::{admonition} Overview
:class: overview

- A kernel runs over a thread hierarchy (thread → warp → warpgroup → CTA → cluster → grid) across distinct memory spaces (registers, SMEM, GMEM, TMEM).
- Compute splits into CUDA cores and Tensor Cores; dedicated engines like TMA move the data that feeds them.
- Every later optimization serves one tile pipeline — load (GMEM → SMEM), compute (SMEM → TMEM), epilogue (TMEM → registers → GMEM) — and aims to keep the compute and data-movement engines busy at once.
:::

Two kernels can compute the same result over the same numbers and
still differ in speed by an order of magnitude — and the gap is almost never the arithmetic, it is
how well the code fits the chip underneath. So before we write a single kernel, this chapter brings
that chip into focus: the hierarchy of threads that run the work, the distinct memory spaces they
move data through, and the handful of compute and data-movement engines that do the heavy lifting.
Nearly every optimization later in the book is a way of arranging work across those three things. We
assemble the picture here — the thread hierarchy, the compute units, the memory spaces, and how a
GEMM flows across them — and the `tcgen05` compute path ({ref}`chap_tensor_cores`), TMA data
movement ({ref}`chap_tma`), and the mbarrier model ({ref}`chap_async_barriers`) each build on it in
their own chapters.

The interactive below gives that whole Blackwell SM picture in one place before we zoom in on each part.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the Blackwell SM — its warps/warpgroups, shared memory, Tensor Memory, and the
Tensor Core and TMA engines.*

## The Execution Hierarchy

A GPU does not present its thousands of threads as one flat pool. Instead it groups them into a
nested hierarchy, and it does so because cooperation happens at several different scales at once. At
the finest scale, the lanes of a warp march through the same instruction in lockstep. A step coarser,
the threads of a CTA share a common pool of fast scratch memory. Coarser still, the CTAs of a
cluster can reach across physically separate SMs to synchronize and to read each other's memory. Each
level exists to make one of these forms of cooperation cheap, and on Blackwell the levels are the
following.

The interactive below names those levels from smallest to largest and shows how they nest.

```{raw} html
<iframe src="../demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; min-width:900px; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: click a level — thread → warp → warpgroup → CTA → cluster → grid.*

- **Thread** — the scalar unit of execution. Each thread has its own program counter and its own
  registers, and it is identified by a lane ID within its warp.
- **Warp** — 32 threads that execute in SIMT (*single instruction, multiple threads*). The lanes of
  a warp issue the same instruction together, yet each keeps its own registers and can be masked off
  on its own, which is what lets the lanes of a single warp follow different branches.
- **Warpgroup** — four consecutive warps, or 128 threads. Hopper introduced the warpgroup as the
  unit that issues warpgroup-level MMA (`wgmma`), and on Blackwell it takes on a second role: it is
  the cooperation unit for Tensor Memory access, where the 128 threads together move a TMEM tile into
  or out of registers.
- **CTA** (*Cooperative Thread Array*, what CUDA also calls a thread block) — the basic unit the
  hardware schedules. A CTA runs on a single SM and owns a private shared-memory allocation inside
  it. Several CTAs can be resident on the same SM at once, and when they are, they divide up that
  SM's shared-memory capacity between them.
- **Cluster** — a group of cooperating CTAs that may live on different SMs. The CTAs in a cluster
  can synchronize with one another and can read and write each other's shared memory, a capability
  known as distributed shared memory.

These levels are worth dwelling on because, unlike on earlier architectures, Blackwell's key
operations are **not all issued by the same group of threads**. A TMA copy is launched by a single
thread and then carried out by hardware. A TMEM-to-register load is warpgroup-distributed: it is
really four warp-collective `tcgen05.ld` instructions, one per warp, each handling that warp's 32
TMEM lanes. A `tcgen05` MMA is committed by one elected thread, while a clustered MMA spans two CTAs
at once. Each operation thus has its own natural granularity, and the set of threads that runs it is
what we call the operation's **scope** — the first of the three recurring knobs (scope, layout, and
dispatch) that this book returns to again and again.

## Compute: CUDA Cores and Tensor Cores

Inside each SM there are two distinct kinds of math engine rather than one, and the division of labor
between them shapes how nearly every kernel is written. The two play complementary roles.

- **CUDA cores** are general-purpose SIMT ALUs. They run the scalar and vector instructions that
  handle index arithmetic, elementwise math, reductions, and control flow — the glue logic that
  surrounds the heavy matrix work.
- **Tensor Cores** are fixed-function units that perform a dense matrix multiply-accumulate at *tile*
  granularity, computing $D = AB + C$ in a single instruction.

The reason this split matters is that dense linear algebra — GEMM, convolution, and attention —
reaches peak throughput only on the Tensor Cores, so getting performance is largely a matter of
keeping them fed. What shifts from one GPU generation to the next is *how* the Tensor Cores are
programmed and *where* their results come to rest. Hopper introduced the asynchronous warpgroup MMA
(`wgmma.mma_async`); Blackwell's fifth-generation Tensor Core, `tcgen05`, places its accumulators in
Tensor Memory instead of registers, and we devote {ref}`chap_tensor_cores` to it.

## Memory Spaces

There is no single memory that is at once large and fast; physics forces a trade-off between
capacity and speed. A GPU therefore offers several memories rather than one, each striking that
trade-off at a different point, and a kernel works by moving data through them. Each space has its
own capacity, its own latency, and its own rules for who may access it.

| Memory | Ownership | Role | Notes |
|--------|-----------|------|-------|
| **Global (GMEM)** | Device-wide | Persistent tensor storage | Large HBM, shared by all SMs |
| **Shared (SMEM)** | Per-CTA (one SM) | Tile staging | Low-latency scratchpad; up to 228 KB/SM on B200 |
| **Tensor Memory (TMEM)** | Per-CTA | MMA accumulator storage | New on Blackwell; used by `tcgen05` |
| **Register File (RF)** | Per-thread | Scalars and per-thread tile fragments | Fast; holds epilogue/temp values |

Read in order, these spaces describe a path. The data path of almost every kernel in this book is
**GMEM → SMEM → (compute) → registers → GMEM**, and for tensor-core kernels TMEM sits in the middle
of that path, holding the accumulators while the math runs.

The figure below summarizes that end-to-end data path and where each memory space sits in it.

![Memory dataflow across the hierarchy](../img/memory_dataflow.png)

Of the four, **Tensor Memory (TMEM)** is the only one with no analog on pre-Blackwell hardware, and
its full details wait until {ref}`chap_tensor_cores`. The motivation for it is worth understanding
now, though. Earlier GPUs kept large MMA accumulators in registers, where they competed for a scarce
resource. Blackwell instead writes `tcgen05` accumulator output to TMEM, a CTA-scoped 2D scratchpad
of 128 lanes by up to 512 32-bit columns per CTA (the array physically lives on the SM). The kernel
then has to read TMEM back into registers explicitly before the epilogue. That extra step is not
free, and two of its consequences will recur throughout the book. The first is that TMEM reads are
**explicit and warpgroup-distributed**, carried out by four warp-collective `tcgen05.ld`
instructions, one per warp's 32 TMEM lanes. The second is that TMEM, unlike registers, must be
**explicitly allocated and freed**.

## CTA Clusters

A CTA runs on one SM and works out of that SM's shared memory, but a single CTA's SMEM budget is
finite, and large tiles often demand more operand storage, or more reuse, than one block alone can
supply. Hopper's answer to this was the **thread block cluster**: a group of CTAs that cooperate more
tightly than independent blocks do, in that they can synchronize together and read and write each
other's shared memory. That last ability is called distributed shared memory, or DSMEM. Blackwell
keeps clusters and adds to them, with dynamic scheduling ({ref}`chap_clc`) and 2-CTA cooperative MMA.

The capability that makes clusters interesting is **distributed shared memory (DSMEM)**, and the
hardware exposes it in two parts that are easiest to understand together. The first part is an
*address*: the `mapa` instruction takes a local SMEM pointer and a peer CTA's rank and returns a
pointer to the same offset within that peer's SMEM. The second part is a *transfer*: armed with such
a pointer, a single thread can bulk-copy a tile out of its own SMEM and into a peer's, raising a
completion barrier ({ref}`chap_async_barriers`) once the bytes have landed. This is a cluster-scoped
`cp.async.bulk`. The 2-CTA cluster GEMM in Part III is built on exactly this mechanism, using it to
share operand tiles across the pair of CTAs without ever routing them back through global memory.

The figure below shows the extra DSMEM hop that a CTA cluster makes possible.

![A CTA cluster sharing distributed shared memory](../img/cta_cluster.png)

Two features that build on DSMEM will reappear throughout the GEMM chapters. The first is **2-CTA
cooperative MMA**, in which two CTAs each contribute their SMEM operands to a single, larger MMA
tile. The second is **TMA multicast**, in which one TMA load delivers the same GMEM tile to several
CTAs at once, eliminating the redundant global traffic that separate loads would otherwise incur.

## The GEMM Data Pipeline

The interactive below traces the three-stage GEMM tile pipeline that the rest of the book keeps optimizing.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the load → MMA → epilogue pipeline on Blackwell, and how the stages overlap.*

Now that the thread hierarchy, the compute engines, and the memory spaces are all on the table, we
can put them together and trace what actually happens when the hardware runs a GEMM. A single GEMM
tile flows through three stages.

1. **Load.** A TMA copy ({ref}`chap_tma`) streams an A or B operand tile from GMEM into SMEM. One
   thread issues the copy after `mbarrier.arrive.expect_tx(bytes)` has recorded how many bytes — the
   transaction, or tx, count — are expected to arrive. As the bytes land, the TMA engine issues
   `complete-tx`, and the barrier's phase flips only once both the arrival count and the tx-count
   have been satisfied.
2. **Compute.** A `tcgen05` MMA ({ref}`chap_tensor_cores`) reads the operand tiles out of SMEM and
   accumulates the product into a TMEM tile. One elected thread issues it, and it signals a barrier
   when the math is done.
3. **Epilogue.** The warpgroup reads the TMEM accumulator back into registers, casts the result to
   the output dtype, and stores it to GMEM — frequently by staging through SMEM and issuing a TMA
   store.

Written out this way the three stages look strictly sequential, but the whole difference between a
slow kernel and a fast one lies in **overlap**. A naive kernel really does run the steps in
order — load, wait, compute, wait, store — and so leaves each engine sitting idle while it waits on
the one before it. A fast kernel pipelines them instead: while the Tensor Core is computing on tile
`k`, the TMA engine is already fetching tile `k+1`, and the epilogue is busy draining tile `k-1`, so
all three engines stay occupied at the same time. Getting three asynchronous engines to hand work off
to one another safely is precisely the job of the barrier and phase model
({ref}`chap_async_barriers`), and the GEMM ladder of Part III is built on top of it.

## Numbers to Keep in Mind

In the end, nearly every design decision in this book comes back to a handful of capacities and
speeds. It helps to have their orders of magnitude in mind, because they are what explain why kernels
stage only a few operand tiles at a time, why they budget TMEM so carefully, and why they go to such
lengths to keep the Tensor Cores fed. The figures below are for the B200 and are approximate.

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

The exact peak numbers depend on the specific GPU model, its clock, and the sparsity mode, so treat
the table as a sense of *scale* rather than as a performance model. The lesson it teaches is a simple one. SMEM is
large enough to hold only a handful of staged tiles, TMEM is scarce enough that it must be budgeted,
and tensor-core throughput is so high that data movement *has* to overlap compute — there is no other
way to keep those engines busy.
