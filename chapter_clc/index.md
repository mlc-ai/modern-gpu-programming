(chap_clc)=
# Advanced Topics: Cluster Launch Control

:::{admonition} Overview
:class: overview

- A persistent kernel keeps one CTA resident per SM and loops over output tiles instead of relaunching a CTA per tile.
- Cluster Launch Control is the hardware mechanism that hands each persistent CTA its next tile — a work-stealing loop driven by two instructions.
- The payoff is even SM utilization from launch to finish.
:::

Suppose we want to keep every SM busy from launch to finish. A *persistent* kernel takes a step
toward that: it keeps one CTA resident per SM and has it loop over many output tiles — rather than
launching a fresh CTA per tile — pulling each next tile from a scheduler (the pattern Part III
builds). But where does that next tile come from? The straightforward answer is a *static*
assignment, tile = f(grid index), and it is exactly here that the trouble starts. If tiles cost
different amounts, or the tile count doesn't divide evenly across the SMs, the work imbalances: some
SMs run out of tiles and sit idle while the unlucky few grind through the tail.

The honest way to fix imbalance is to stop deciding the schedule in advance and hand work out only
when an SM is ready for it. **Cluster Launch Control (CLC)** is the Blackwell (`sm_100`) hardware
mechanism that does this. Instead of computing its next tile from a fixed formula, a persistent
cluster asks the grid scheduler — in hardware — for the next not-yet-launched cluster's work and
takes it over. Because work is handed out on demand, an SM that finishes early simply pulls more,
and the tail flattens out.

One clarification keeps the lineage straight. Thread block clusters *themselves* — launching CTAs
in clusters with DSMEM and cluster barriers — are a Hopper feature ({ref}`chap_background`); CLC is
not the clusters but the Blackwell addition that makes their **scheduling** dynamic.

## The Two Instructions

The whole mechanism is exposed as just two PTX instructions (`clusterlaunchcontrol`, PTX ISA 8.6),
and they divide the work cleanly: one asks for the next tile, the other reads the answer.

- **`try_cancel`** — `clusterlaunchcontrol.try_cancel.async`. A single asynchronous request that
  asks the scheduler to *cancel the launch* of the next pending cluster and hand this cluster its
  coordinates instead. The 16-byte response is written to SMEM and an mbarrier is signalled on
  completion (`mbarrier::complete_tx::bytes`), multicast to every CTA in the cluster — so it is
  awaited with the same phase/barrier model as a TMA load ({ref}`chap_async_barriers`).
- **`query_cancel`** — decodes that response once the barrier fires. If a cluster was successfully
  cancelled (its work stolen), it returns the cancelled cluster's first `ctaid.x` — the tile to
  process; otherwise it returns a sentinel (`0xFFFFFFFF`) meaning *no work left*.

The word "cancel" is the key to the trick: stealing a tile is implemented as cancelling some other
cluster's pending launch and inheriting its coordinates, which is why the answer can be either a
real tile or the sentinel that says the grid is exhausted.

## The Work-Stealing Loop

With those two instructions, the persistent kernel becomes a short loop that keeps asking for work
until there is none left:

1. Process the tile this cluster was launched with.
2. `try_cancel` the next cluster; wait on its mbarrier.
3. `query_cancel`: if it returns a valid `ctaid`, decode it into tile coordinates and process that
   tile; if it returns the sentinel, exit.

The ordering of these steps is doing real work. Because step 2 is asynchronous and barrier-tracked,
it does not block — it overlaps with step 1's compute, so the next tile's assignment is already in
hand by the time the current tile finishes. The SM moves from one tile to the next without ever
stalling to ask what to do.

This is exactly the step the GEMM ladder ({ref}`chap_gemm_advanced`) reaches for: its
persistent-kernel stage uses CLC so output tiles are distributed by hardware work-stealing rather
than a static grid-stride loop. As the opening argued, the payoff shows up precisely when per-tile
cost varies or the tile count does not divide evenly across SMs — the cases where a static schedule
would have left SMs idle in the tail.
