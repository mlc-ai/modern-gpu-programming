(chap_clc)=
# Cluster Launch Control (CLC)

A persistent kernel launches one CTA per SM and loops over output tiles with a tile scheduler
(Part III). With a *static* assignment — tile = f(grid index) — the work can imbalance: if tiles
cost different amounts, or the tile count doesn't divide evenly across the SMs, some SMs finish
early and idle while others carry the tail.

**Cluster Launch Control (CLC)** is the Blackwell (`sm_100`) hardware mechanism that addresses
this. Instead of computing its next tile from a fixed formula, a persistent cluster asks the grid
scheduler — in hardware — for the next not-yet-launched cluster's work and takes it over. Work is
handed out dynamically, so an SM that finishes early immediately pulls more.

Thread block clusters *themselves* — launching CTAs in clusters with DSMEM and cluster barriers —
are a Hopper feature ({ref}`chap_background`). CLC is the Blackwell addition that makes their
**scheduling** dynamic.

## The Two Instructions

CLC is two PTX instructions (`clusterlaunchcontrol`, PTX ISA 8.6):

- **`try_cancel`** — `clusterlaunchcontrol.try_cancel.async`. A single asynchronous request that
  asks the scheduler to *cancel the launch* of the next pending cluster and hand this cluster its
  coordinates instead. The 16-byte response is written to SMEM and an mbarrier is signalled on
  completion (`mbarrier::complete_tx::bytes`), multicast to every CTA in the cluster — so it is
  awaited with the same phase/barrier model as a TMA load ({ref}`chap_async_barriers`).
- **`query_cancel`** — decodes that response once the barrier fires. If a cluster was successfully
  cancelled (its work stolen), it returns the cancelled cluster's first `ctaid.x` — the tile to
  process; otherwise it returns a sentinel (`0xFFFFFFFF`) meaning *no work left*.

## The Work-Stealing Loop

A CLC persistent kernel is a loop:

1. Process the tile this cluster was launched with.
2. `try_cancel` the next cluster; wait on its mbarrier.
3. `query_cancel`: if it returns a valid `ctaid`, decode it into tile coordinates and process that
   tile; if it returns the sentinel, exit.

Because step 2 is asynchronous and barrier-tracked, it overlaps with step 1's compute — the next
tile's assignment is ready by the time the current tile finishes, so the SM never stalls waiting
for work.

The persistent-kernel step of the GEMM ladder ({ref}`chap_gemm_advanced`) uses CLC so output
tiles are distributed by hardware work-stealing rather than a static grid-stride loop. The
payoff is best when per-tile cost varies or the tile count does not divide evenly across SMs —
exactly the cases where a static schedule leaves SMs idle in the tail.
