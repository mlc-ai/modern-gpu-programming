(chap_clc)=
# Advanced Topics: Cluster Launch Control

:::{admonition} Overview
:class: overview

- A persistent kernel keeps about one CTA resident per SM (not a guaranteed 1:1 mapping) and loops over output tiles instead of relaunching a CTA per tile.
- Cluster Launch Control is the hardware mechanism that hands each persistent CTA its next tile — a work-stealing loop driven by two instructions.
- The payoff is even SM utilization from launch to finish.
:::

**Motivation.** A persistent kernel keeps roughly one CTA resident per SM (residency is not a
guaranteed 1:1 mapping) and loops over many output tiles instead of relaunching a fresh CTA per
tile — the pattern Part III builds. The open question is how each CTA decides which tile to do next.
The simplest answer is a *static* formula, tile = f(grid index), computed up front. That works only
when every tile costs the same and the tile count divides evenly across the SMs; the moment it
doesn't, the schedule is locked in before any work runs, so a few unlucky SMs grind through the tail
while the rest finish their share and sit idle. What we want instead is to hand out the next tile
only when an SM is actually ready for it, so an SM that finishes early just pulls more. That is what
Cluster Launch Control provides, and the rest of this chapter follows it from the two PTX
instructions ({ref}`chap_async_barriers` for the barrier model they reuse) to the work-stealing loop
and back to the persistent GEMM of {ref}`chap_gemm_advanced`.

To fix imbalance, stop deciding the schedule in advance and hand work out only when an SM is ready
for it. **Cluster Launch Control (CLC)** is the Blackwell (`sm_100`) hardware mechanism that does
this. Instead of computing its next tile from a fixed formula, a persistent cluster asks the grid
scheduler — in hardware — for the next not-yet-launched cluster's work and takes it over. Because
work is handed out on demand, an SM that finishes early pulls more, and the tail flattens out.

Thread block clusters *themselves* — launching CTAs in clusters with DSMEM and cluster barriers —
are a Hopper feature ({ref}`chap_background`); CLC is not the clusters but the Blackwell addition
that makes their **scheduling** dynamic.

## The Two Instructions

The mechanism is exposed as two PTX instructions (`clusterlaunchcontrol`, PTX ISA 8.6): one asks
for the next tile, the other reads the answer.

- **`try_cancel`** — `clusterlaunchcontrol.try_cancel.async`. A single asynchronous request that
  asks the scheduler to *cancel the launch* of the next pending cluster and hand this cluster its
  coordinates instead. The 16-byte response is written to SMEM and an mbarrier is signalled on
  completion (`mbarrier::complete_tx::bytes`), multicast to every CTA in the cluster — so it is
  awaited with the same phase/barrier model as a TMA load ({ref}`chap_async_barriers`).
- **`query_cancel`** — reads that response once the barrier fires, in two steps:
  `clusterlaunchcontrol.query_cancel.is_canceled` returns a **predicate** (did a cancellation
  succeed?), and only when it did do you call `clusterlaunchcontrol.query_cancel.get_first_ctaid`
  to extract the cancelled cluster's first CTA id — a coordinate vector (`x`, `y`, `z`), the tile to
  process. A false predicate means *no work left*; you branch on the predicate, there is no numeric
  sentinel to compare against.

Stealing a tile is implemented as cancelling some other cluster's pending launch and inheriting its
coordinates, which is why the response is either a successful cancellation (carrying a real cluster
coordinate) or a failure that says the grid is exhausted.

## The Work-Stealing Loop

With those two instructions, the persistent kernel becomes a short loop that keeps asking for work
until there is none left:

1. `try_cancel` the next cluster — an asynchronous request that returns immediately.
2. Process the tile this cluster already has, while that request is in flight.
3. Wait on the request's mbarrier, then `query_cancel`: if `is_canceled` is true, read
   `get_first_ctaid`, decode it into tile coordinates, and loop; otherwise exit.

The ordering matters: the request goes out *before* the cluster computes its current tile, so the
asynchronous, barrier-tracked `try_cancel` overlaps that compute. The next tile's assignment is
already in hand by the time the current tile finishes, and the SM moves from one tile to the next
without stalling to ask what to do.

TIRx exposes CLC as a dynamic tile scheduler. The persistent GEMM shown in {ref}`chap_gemm_advanced`
(Step 6) uses the *static* `ClusterPersistentScheduler2D` — a grid-stride-style assignment — to keep
the walkthrough simple; CLC is the drop-in dynamic alternative that distributes tiles by hardware
work-stealing instead. The payoff shows up when per-tile cost varies or the tile count does not
divide evenly across SMs — the cases where a static schedule leaves SMs idle in the tail.
