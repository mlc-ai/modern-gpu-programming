(chap_background)=
# GPU Execution Model

:::{admonition} Overview
:class: overview

- A GPU kernel's execution is shaped first by its thread hierarchy: thread, warp, warpgroup, CTA, cluster, and grid each correspond to a different scale of cooperation. Many Blackwell operations have their own natural scope: a TMA copy is launched by a single thread, a TMEM load is carried out by a warpgroup, and a 2-CTA cooperative MMA spans two CTAs.
- Data does not live in a single place. GMEM, SMEM, TMEM, and registers offer different tradeoffs in capacity, latency, and access scope; clusters also use DSMEM so one CTA can access another CTA's shared memory. A core task of a high-performance kernel is to move data efficiently between these spaces.
- Compute and data movement are handled by different hardware engines. CUDA cores handle address calculation, control flow, and scalar logic; Tensor Cores perform the main matrix computation; TMA moves data asynchronously. We end the chapter with a GEMM data pipeline that shows how overlap keeps multiple engines busy at the same time.
:::

To write high-performance GPU kernels, we first need to understand how threads are organized during
a kernel launch, where data lives, and how different hardware engines work together. This chapter is
organized around those three questions. We start with the GPU thread hierarchy, then introduce the
memory spaces that store and move data, and finally discuss the hardware engines responsible for
compute and data movement. We then connect these concepts with a GEMM pipeline, showing how data
moves between memory spaces and how compute overlaps with data movement. Many later optimizations in
the book build on these same basic mechanisms.

We begin with the Blackwell SM architecture. The figure below shows the main hardware units used in
this chapter.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Click a component to see details: the Blackwell SM, including warps/warpgroups, shared memory,
Tensor Memory, and the Tensor Core and TMA engines.*

## The Execution Hierarchy

A GPU does not manage thousands of threads as one flat collection. Instead, it organizes them into a
nested hierarchy. Each level corresponds to a different scale of cooperation and makes cooperation at
that scale efficient. The figure below shows the thread hierarchy on Blackwell.

```{raw} html
<iframe src="../demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; min-width:900px; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Click a component to see details: thread → warp → warpgroup → CTA → cluster → grid.*

- **Thread**: the scalar unit of execution. Each thread has its own program counter and its own
  registers, and it is identified by a lane ID within its warp.
- **Warp**: 32 threads that execute in SIMT (*single instruction, multiple threads*). The lanes of
  a warp issue the same instruction together, yet each keeps its own registers and can be masked off
  on its own, which is what lets the lanes of a single warp follow different branches.
- **Warpgroup**: four consecutive warps, or 128 threads. Hopper introduced the warpgroup as the
  unit that issues warpgroup-level MMA (`wgmma`), and on Blackwell it takes on a second role: it is
  also the cooperation unit for Tensor Memory access.
- **CTA** (*Cooperative Thread Array*, what CUDA also calls a thread block): the basic unit the
  hardware schedules. A CTA runs on a single SM and owns a private shared-memory allocation inside
  it. Several CTAs can be resident on the same SM at once, and when they are, they divide up that
  SM's shared-memory capacity between them.
- **Cluster**: a group of cooperating CTAs that may live on different SMs. The CTAs in a cluster
  can synchronize with one another and can read and write each other's shared memory, a capability
  known as distributed shared memory.

Blackwell's key operations are not all issued by the same group of threads. A TMA copy is launched by
a single thread and then carried out by hardware. A TMEM load is carried out cooperatively by the
four warps in a warpgroup. A `tcgen05` MMA is committed by one elected thread. A 2-CTA cooperative
MMA spans two CTAs. In other words, each operation has its own cooperation granularity. Later in the
book, we will call the set of threads involved in an operation its **scope**; together with layout and
dispatch, scope is one of the basic tools for analyzing kernel implementations.

## Memory Spaces

The thread hierarchy determines how computation is organized, but more threads do not help if data
cannot reach them fast enough. We therefore turn next to where data lives. A GPU does not use one
single memory space. Instead, it provides multiple memory spaces, each making a different tradeoff
between capacity, latency, and access scope. A kernel's job is to move data efficiently between these
spaces.

| Memory | Ownership | Role | Notes |
|--------|-----------|------|-------|
| **Global (GMEM)** | Device-wide | Persistent tensor storage | Large HBM, shared by all SMs |
| **Shared (SMEM)** | Per-CTA (one SM) | Tile staging | Low-latency scratchpad; up to 228 KB/SM on B200 |
| **Tensor Memory (TMEM)** | Per-CTA | MMA accumulator storage | Introduced with Blackwell; used by `tcgen05` |
| **Register File (RF)** | Per-thread | Scalars and per-thread tile fragments | Fast; holds epilogue/temp values |

Among these memory spaces, **Tensor Memory (TMEM)** was introduced with Blackwell. Before Blackwell, MMA
accumulators were usually stored in registers, but registers are limited and large MMAs can create
high register pressure. Blackwell's `tcgen05` instead writes accumulators into TMEM, moving that
storage out of registers. You can think of TMEM as a CTA-scoped 2D scratchpad: each CTA gets 128
lanes and up to 512 32-bit columns. Logically, this array belongs to the CTA, but physically it lives
on the SM. Because accumulators are first written to TMEM, the kernel has to read them back into
registers explicitly before the epilogue. This design has two implications that will appear
repeatedly in later chapters, especially {ref}`chap_tensor_cores`. First, TMEM reads are explicit and
warpgroup-distributed: they are carried out cooperatively by the four warps in a warpgroup. Second,
TMEM is not automatically allocated and managed by the compiler like registers; the program must
explicitly allocate and free it.

### Distributed Shared Memory Across a Cluster

Most of the levels introduced above are confined to a single SM, while a cluster can place multiple
CTAs on different SMs. This means CTAs can not only synchronize with one another, but also access one
another's shared memory across SMs. This cross-CTA shared-memory access capability is called
**distributed shared memory (DSMEM)**.

DSMEM helps avoid unnecessary round trips through global memory. With DSMEM, one CTA can copy a tile
directly from its own SMEM to another CTA's SMEM in the same cluster, without first writing it back
to GMEM and having the peer CTA read it again. Once the copy completes, hardware raises a completion
barrier to tell later computation that the data is ready.

The 2-CTA cluster GEMM in Part III is built on this mechanism. The two CTAs can share operand tiles
through DSMEM, reducing global-memory traffic. Here, "share" does not mean merging the two CTAs'
SMEM into one pool: Asmem and Bsmem still belong to their respective CTAs. DSMEM provides cross-CTA
access, allowing other CTAs in the cluster, or a `cta_group=2` cooperative MMA, to read data from a
peer CTA's SMEM.

The figure below shows the DSMEM access path in a 2-CTA cluster: each CTA still owns its own SMEM,
but it can read another CTA's SMEM through DSMEM.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/cta_cluster.html" title="A 2-CTA cluster sharing distributed shared memory" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Click a component to see details: a 2-CTA cluster. Each CTA owns half of A and B, reads the peer's
B through the cluster (DSMEM), and the pair produces a 256×256 output tile.*

## Compute: CUDA Cores and Tensor Cores

The thread hierarchy determines how computation is organized, and the memory spaces determine where
data lives. The arithmetic itself is carried out by compute units inside the SM. An SM mainly has two
kinds of compute units: CUDA cores and Tensor Cores.

- **CUDA cores** are general-purpose SIMT ALUs. They run the scalar and vector instructions that
  handle index arithmetic, elementwise math, reductions, and control flow. In Tensor Core kernels,
  they provide the glue logic around the heavy matrix work.
- **Tensor Cores** are fixed-function units that perform a dense matrix multiply-accumulate at *tile*
  granularity, computing $D = AB + C$ in a single instruction.

Tensor Cores provide much higher arithmetic throughput than CUDA cores, often 10× or more in
FLOP/s. Dense linear algebra workloads such as GEMM, convolution, and attention can approach peak
performance only when they run on Tensor Cores. A core goal of high-performance kernel programming is
therefore to keep Tensor Cores busy instead of leaving them idle because data or dependencies are not
ready.

Different GPU generations change not only Tensor Core throughput, but also how Tensor Cores are
programmed and where accumulators are stored. Hopper introduced asynchronous warpgroup MMA
(`wgmma.mma_async`). Blackwell's fifth-generation Tensor Core, `tcgen05`, stores accumulators in
Tensor Memory rather than registers. Later chapters, especially {ref}`chap_tensor_cores`, discuss
this in detail.

Clusters also introduce two important GEMM uses. **2-CTA cooperative MMA** allows two CTAs to each
provide part of the SMEM operands and jointly issue a larger Tensor Core MMA tile. **TMA multicast**
allows one GMEM load to deliver the same tile to multiple CTAs, avoiding redundant global memory
traffic from each CTA loading the same data separately. Both rely on the cluster and DSMEM mechanism
introduced earlier.

## The GEMM Data Pipeline

The previous sections introduced the thread hierarchy, memory spaces, data-movement mechanism, and
compute units separately. We now connect them with a GEMM pipeline and look at how these hardware
structures work together. The figure below shows the main units involved in a three-stage GEMM tile
pipeline.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Click a component to see details: the load → MMA → epilogue pipeline on Blackwell.*

A single GEMM tile usually flows through three stages.

1. **Load.** A TMA copy ({ref}`chap_tma`) moves an A or B operand tile from GMEM to SMEM. One thread
   issues the copy and records the expected number of arriving bytes. As data is written into SMEM,
   the TMA engine updates progress. Only after all expected bytes have arrived does the completion
   barrier fire.
2. **Compute.** A `tcgen05` MMA ({ref}`chap_tensor_cores`) reads operand tiles from SMEM and
   accumulates the product into a TMEM tile. One elected thread commits the MMA; when computation
   completes, hardware signals the corresponding barrier.
3. **Epilogue.** A warpgroup reads the TMEM accumulator back into registers, converts the result to
   the output dtype, and writes it back to GMEM. This step often stages through SMEM and may use TMA
   store for the final writeback.

Written this way, the three stages look strictly sequential. But the key difference between a slow
kernel and a fast kernel is whether these stages can overlap. A naive kernel runs load, wait,
compute, wait, and store in order; as a result, each engine sits idle while waiting for the previous
stage to finish. A high-performance kernel instead organizes these stages as a pipeline: while the
Tensor Core computes tile `k`, the TMA engine is already moving tile `k+1`, and the epilogue is
processing the output of tile `k-1`. This keeps multiple engines busy at the same time. Making these
asynchronous engines hand work off safely is exactly what the barrier and phase model is for, and
the GEMM optimization ladder in Part III is built on this mechanism.
