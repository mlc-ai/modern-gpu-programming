(chap_clc)=
# Advanced: Cluster Launch Control

:::{admonition} Overview
:class: overview

- A persistent kernel keeps about one CTA resident per SM (not a guaranteed 1:1 mapping) and loops over output tiles instead of relaunching a CTA per tile.
- Cluster Launch Control is the hardware mechanism that hands each persistent CTA its next tile — a work-stealing loop driven by two instructions.
- The payoff is even SM utilization from launch to finish.
:::

A persistent kernel keeps roughly one CTA resident per SM (residency is not a
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

The cure for that imbalance is to stop deciding the schedule in advance and instead hand work out
only when an SM is ready for it. This is exactly what **Cluster Launch Control (CLC)**, the Blackwell
(`sm_100`) hardware mechanism, gives us. Rather than computing its next tile from a fixed formula, a
persistent cluster turns around and asks the grid scheduler — in hardware — for the next
not-yet-launched cluster's work, and then takes that work over. Because tiles are handed out on
demand, an SM that happens to finish early simply pulls more, and the long tail we worried about
flattens out on its own.

It is worth being precise about what is new here. Thread block clusters *themselves* — launching CTAs
in clusters that share DSMEM and synchronize through cluster barriers — are a Hopper feature
({ref}`chap_background`), and we already have them. CLC is not the clusters; it is the Blackwell
addition that makes their **scheduling** dynamic.

## The Two Instructions

We have said CLC hands tiles out on demand; now we can see exactly how a CTA asks for one. The whole
mechanism is exposed as just two PTX instructions (`clusterlaunchcontrol`, PTX ISA 8.6): one asks the
grid scheduler for the next tile, and the other reads back the answer. Let us look at each in turn.

- **`try_cancel`** — `clusterlaunchcontrol.try_cancel.async`. This is a single asynchronous request
  that asks the scheduler to *cancel the launch* of the next pending cluster and hand this cluster its
  coordinates instead. The 16-byte response is written into SMEM, and when it lands an mbarrier is
  signalled (`mbarrier::complete_tx::bytes`), multicast to every CTA in the cluster. The nice
  consequence is that we wait on the answer with the very same phase-and-barrier model we already use
  for a TMA load ({ref}`chap_async_barriers`) — there is nothing new to learn about how to await it.
- **`query_cancel`** — once that barrier fires, this reads the response, and it does so in two steps.
  First, `clusterlaunchcontrol.query_cancel.is_canceled` returns a **predicate**: did a cancellation
  actually succeed? Only when it did succeed do we go on to call
  `clusterlaunchcontrol.query_cancel.get_first_ctaid`, which extracts the cancelled cluster's first
  CTA id — a coordinate vector (`x`, `y`, `z`) that names the tile to process. A false predicate
  simply means there is *no work left*. Note that we branch on the predicate itself; there is no
  numeric sentinel value to compare against.

The reason the response takes this shape is that stealing a tile is, under the hood, cancelling some
other cluster's pending launch and inheriting its coordinates. So every reply is one of two things:
either a successful cancellation, carrying a real cluster coordinate, or a failure that tells us the
grid is exhausted.

## The Work-Stealing Loop

With `try_cancel` to request a tile and `query_cancel` to read the answer in hand, we can see how a
persistent kernel uses them. The two instructions collapse the kernel into a short loop that keeps
asking the scheduler for work until there is none left. Each iteration does three things:

1. It issues a `try_cancel` for the next cluster. This is an asynchronous request, so it returns
   immediately and leaves us free to do other work.
2. It processes the tile this cluster already holds, while that request is still in flight.
3. It waits on the request's mbarrier and then calls `query_cancel`. If `is_canceled` is true, it
   reads `get_first_ctaid`, decodes it into tile coordinates, and goes around again; otherwise it
   exits the loop.

The ordering in those steps is the whole trick, so it is worth dwelling on. The request goes out
*before* the cluster computes its current tile, which means the asynchronous, barrier-tracked
`try_cancel` overlaps that computation rather than blocking it. By the time the current tile is
finished, the assignment for the next tile is already in hand, and the SM glides from one tile
straight into the next without ever stalling to ask what to do.

TIRx exposes CLC as a dynamic tile scheduler. To keep its walkthrough simple, the persistent GEMM in
{ref}`chap_gemm_advanced` (Step 6) uses the *static* `ClusterPersistentScheduler2D`, a
grid-stride-style assignment computed up front. CLC is the drop-in dynamic alternative: same loop,
but tiles are distributed by hardware work-stealing instead of a fixed formula. The payoff becomes
visible precisely when per-tile cost varies, or when the tile count does not divide evenly across the
SMs — the cases where a static schedule would leave some SMs idle through the tail.
