(chap_background)=
# GPU Execution Model

A GPU runs a kernel across a hierarchy of threads, a set of distinct memory spaces, and a few
compute and data-movement engines. This chapter covers those — the thread hierarchy, the compute
units, the memory spaces, and how a GEMM flows across them. The `tcgen05` compute path
({ref}`chap_tensor_cores`), TMA data movement ({ref}`chap_tma`), and the mbarrier coordination
model ({ref}`chap_async_barriers`) each get their own chapter; this one establishes the hierarchy
and dataflow they build on.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/sm_architecture.html" title="Blackwell SM architecture" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the Blackwell SM — its warps/warpgroups, shared memory, Tensor Memory, and the
Tensor Core and TMA engines.*

## The Execution Hierarchy

A GPU organizes threads into a nested hierarchy. On Blackwell the levels are:

```{raw} html
<iframe src="../demo/thread_hierarchy.html" title="Blackwell thread hierarchy" loading="lazy"
        style="width:100%; min-width:900px; height:520px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
```
*Interactive: click a level — thread → warp → warpgroup → CTA → cluster → grid.*

- **Thread** — the scalar unit of execution, identified by a lane ID within its warp.
- **Warp** — 32 threads executing in SIMT (*single instruction, multiple threads*): the lanes
  issue the same instruction together, but each keeps its own registers and data.
- **Warpgroup** — 4 consecutive warps (128 threads). Introduced on Hopper as the unit for
  warpgroup-level MMA (`wgmma`), it is also the cooperation unit for Blackwell Tensor Memory
  access, where the 128 threads cooperatively move a TMEM tile to or from registers.
- **CTA** (*Cooperative Thread Array*, a.k.a. a CUDA thread block) — the basic scheduling unit.
  A CTA runs on a single SM and owns that SM's shared memory.
- **Cluster** — a group of cooperating CTAs (across SMs) that can share memory and synchronize.

These levels matter because Blackwell operations are **not all issued by the same group of
threads**. A TMA copy is issued by one thread and finished by hardware; a TMEM→register load is
warpgroup-cooperative; a `tcgen05` MMA is committed by one elected thread; a clustered MMA spans
two CTAs. This is the **scope** knob from the introduction: which threads run an operation.

## Compute: CUDA Cores and Tensor Cores

An SM has two kinds of math engine:

- **CUDA cores** — general-purpose SIMT ALUs that execute scalar/vector instructions for indexing,
  elementwise math, reductions, and control flow.
- **Tensor Cores** — fixed-function units that execute a dense matrix multiply-accumulate at *tile*
  granularity in a single instruction: $D = AB + C$.

Tensor Cores are **not new** — they have performed tile-level MMA since Volta (2017), and dense
linear algebra (GEMM, convolution, attention) reaches peak throughput only on them. What changes
from one GPU generation to the next is *how* the Tensor Core is programmed and *where* its results
live: the asynchronous warpgroup MMA model arrived with Hopper, and Blackwell's fifth-generation
Tensor Core (`tcgen05`, with accumulators in Tensor Memory) is covered in {ref}`chap_tensor_cores`.

## Memory Spaces

A kernel moves data through a hierarchy of memories, each with its own capacity, latency, and
access rules:

| Memory | Ownership | Role | Notes |
|--------|-----------|------|-------|
| **Global (GMEM)** | Device-wide | Persistent tensor storage | Large HBM, shared by all SMs |
| **Shared (SMEM)** | Per-CTA (one SM) | Tile staging | Low-latency scratchpad; up to 228 KB/SM on B200 |
| **Tensor Memory (TMEM)** | Per-SM | MMA accumulator storage | New on Blackwell; used by `tcgen05` |
| **Register File (RF)** | Per-thread | Scalars and per-thread tile fragments | Fast; holds epilogue/temp values |

The data path of almost every kernel in this book is **GMEM → SMEM → (compute) → registers →
GMEM**, with TMEM holding accumulators in the middle for tensor-core kernels.

![Memory dataflow across the hierarchy](../img/memory_dataflow.png)

**Tensor Memory (TMEM)** is the one space without a pre-Blackwell analog, so it is worth naming
here even though its details belong to {ref}`chap_tensor_cores`. Earlier GPUs kept large MMA
accumulators in registers; Blackwell instead writes `tcgen05` accumulator output to TMEM, a
per-SM 2D scratchpad (128 rows × up to 512 32-bit columns). The kernel then explicitly reads
TMEM into registers for the epilogue. Two consequences show up everywhere later: TMEM reads are
**explicit and warpgroup-cooperative**, and TMEM must be **explicitly allocated and freed**.

## CTA Clusters

Hopper introduced **thread block clusters**: a group of CTAs that cooperate more tightly than
independent blocks — cluster CTAs can synchronize together and access each other's shared memory
(distributed shared memory, DSMEM). Blackwell builds on clusters with dynamic scheduling
({ref}`chap_clc`) and 2-CTA cooperative MMA.

**Distributed shared memory (DSMEM).** Cluster CTAs can reach each other's shared memory, and the
hardware exposes this in two parts. First, an *address*: the `mapa` instruction maps a local SMEM
pointer to a peer CTA's rank, returning the same offset in that CTA's SMEM. Second, a *transfer*: a
single thread can bulk-copy a tile from its own SMEM into a peer's, signalling a completion barrier
({ref}`chap_async_barriers`) when the bytes land — a cluster-scoped `cp.async.bulk`. The 2-CTA
cluster GEMM in Part III uses this to share operand tiles across the pair without a round trip
through global memory.

![A CTA cluster sharing distributed shared memory](../img/cta_cluster.png)

Clusters matter for large tiles because one CTA's SMEM budget is finite. Two cluster features
recur in the GEMM chapters: **2-CTA cooperative MMA** (two CTAs contribute SMEM operands to one
larger MMA tile) and **TMA multicast** (one TMA load delivers the same GMEM tile to several
CTAs, cutting redundant global traffic).

## The GEMM Data Pipeline

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/pipeline_arch.html" title="Blackwell GEMM data pipeline" loading="lazy"
        style="width:100%; min-width:1320px; height:680px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the load → MMA → epilogue pipeline on Blackwell, and how the stages overlap.*

Here is how a GEMM tile flows across the hardware:

1. **Load.** A TMA copy ({ref}`chap_tma`) streams an A/B operand tile from GMEM into SMEM. One
   thread issues it; the TMA engine does the transfer and signals an mbarrier when the bytes land.
2. **Compute.** A `tcgen05` MMA ({ref}`chap_tensor_cores`) reads the SMEM operands and
   accumulates into a TMEM tile. It is issued by one elected thread and signals a barrier when done.
3. **Epilogue.** The warpgroup reads the TMEM accumulator into registers, casts it to the output
   dtype, and stores it back to GMEM (often via SMEM staging + a TMA store).

The difference between a slow and a fast kernel is **overlap**. A naive kernel runs these steps
in sequence — load, wait, compute, wait, store — leaving each engine idle most of the time. A
fast kernel pipelines them: while the Tensor Core computes on tile `k`, the TMA engine is already
fetching tile `k+1` and the epilogue is draining tile `k-1`. Making three asynchronous engines
hand work to each other safely is the job of the barrier/phase model ({ref}`chap_async_barriers`);
Part III's GEMM ladder is built on it.

## Numbers to Keep in Mind

These orders of magnitude (B200, approximate) explain why kernels stage only a few operand tiles,
budget TMEM carefully, and work hard to keep the Tensor Cores fed:

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
