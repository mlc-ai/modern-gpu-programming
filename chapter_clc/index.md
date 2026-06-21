(chap_clc)=
# Advanced: Cluster Launch Control

:::{admonition} Overview
:class: overview

- A persistent kernel keeps a fixed set of CTAs or CTA clusters resident (often sized so there is roughly one active work owner per SM, though not relying on a guaranteed 1:1 mapping) and has them loop over many output tiles instead of launching one CTA per tile.
- Cluster Launch Control is the Blackwell hardware mechanism that lets a resident cluster ask for another tile at runtime. It is a hardware work-stealing path built around two PTX instructions: one instruction requests work, and the other reads back whether the request succeeded.
- The main benefit is better tail behavior. When tiles have uneven cost, or when the number of tiles does not divide evenly across the available SMs, CTAs that finish early can pull more work instead of sitting idle.
:::

A persistent GEMM does not treat the CUDA grid as a fixed one-CTA-per-output-tile launch. Instead, it launches a smaller set of long-lived CTAs or CTA clusters. Each one computes a tile, advances to another tile, computes again, and keeps going until the output space is finished. This is the execution pattern built up in {ref}`chap_gemm_advanced`.

Once the kernel is persistent, the main scheduling question becomes simple: after a CTA or cluster finishes its current tile, where does the next tile come from?

The simplest answer is a static formula. For example, the kernel can compute the tile coordinate from the CTA id, then advance by a grid stride. That is easy to implement, and it works well when all tiles have roughly the same cost and the tile count is evenly distributed across the GPU. But the schedule is decided before the work actually runs. If a few tiles take longer, or if the last few tiles are unevenly assigned, some SMs finish their share early while others are still working through the tail.

Cluster Launch Control, or CLC, changes that scheduling model. Instead of deciding the whole assignment up front, a persistent cluster can ask the hardware grid scheduler for another not-yet-launched cluster's work. If the request succeeds, the current cluster takes over that cluster coordinate and computes the corresponding tile. If the request fails, there is no more work to steal, and the loop exits.

This is not the same thing as thread block clusters themselves. Thread block clusters (CTAs launched together, with cluster-level synchronization and access to distributed shared memory) were introduced with Hopper ({ref}`chap_background`). CLC is the Blackwell addition that makes scheduling over those cluster coordinates dynamic. The cluster is already the unit of launch; CLC lets an already-running cluster cancel a pending launch and inherit its coordinates.

The rest of this chapter follows that mechanism in three steps: the two PTX instructions, the persistent work-stealing loop, and the connection back to the persistent GEMM scheduler used later.

## The Two Instructions

Cluster Launch Control is exposed through two PTX instructions. The first instruction sends an asynchronous request to the grid scheduler. The second instruction reads the response.

The request instruction is `clusterlaunchcontrol.try_cancel.async`.

A `try_cancel` asks the scheduler to cancel the launch of a pending cluster and return that cluster's coordinates to the caller. The response is written to shared memory as a 16-byte record. Since the request is asynchronous, the instruction does not wait for the response to arrive. Instead, completion is reported through an `mbarrier`, using the same barrier-and-phase model used by TMA.

This is an important detail because it means CLC does not introduce a new waiting model. The kernel issues the request, associates it with a barrier, and later waits on the barrier before reading the response. The response arrival is signaled through the barrier with byte-count completion, in the same general style as other asynchronous hardware operations (see {ref}`chap_async_barriers`).

Once the barrier has fired, the kernel uses the query instructions.

The first query is `clusterlaunchcontrol.query_cancel.is_canceled`. It returns a predicate telling the kernel whether the cancellation succeeded. A true predicate means the scheduler found a pending cluster launch, canceled it, and returned its coordinate. A false predicate means there was no pending work left to take.

Only when `is_canceled` is true should the kernel read the coordinate. It does that with `clusterlaunchcontrol.query_cancel.get_first_ctaid`, which extracts the first CTA id of the canceled cluster. That CTA id is a coordinate vector, usually read as `(x, y, z)`, and the kernel decodes it into the output tile it should compute next.

There is no numeric sentinel tile id in this protocol. The kernel branches on the predicate. If the predicate is true, the coordinate is valid. If the predicate is false, the work-stealing loop is done.

Under the hood, this shape follows directly from what CLC is doing. The hardware is not allocating an abstract task from a software queue. It is canceling a cluster launch that has not happened yet. A successful response therefore contains a real cluster coordinate. A failed response simply means the launch queue has been exhausted.

## The Work-Stealing Loop

With those two instructions, the persistent scheduler becomes a short loop.

At any point in the loop, the cluster has one tile it is responsible for computing. Before it starts that tile, it sends a `try_cancel` request for the next one. The request runs asynchronously. While the scheduler is working on that request, the cluster computes its current tile.

After the current tile is finished, the cluster waits on the `mbarrier` associated with the `try_cancel` response. Once the response is ready, it calls `query_cancel.is_canceled`. If the predicate is true, it calls `query_cancel.get_first_ctaid`, decodes the returned coordinate, and uses that as the next tile. If the predicate is false, there is no more work left, and the cluster exits.

In code shape, the loop is:

1. issue `try_cancel` for a possible next tile;
2. compute the current tile while the request is in flight;
3. wait for the response barrier;
4. query whether the cancellation succeeded;
5. either continue with the returned coordinate or exit.

The placement of the request is what makes the loop useful. The cluster does not wait until it has finished the current tile before asking for more work. It asks first, then computes. That overlaps the scheduler request with useful work. By the time the current tile is done, the answer for the next tile is often already available.

This is the same basic reason persistent kernels use asynchronous copies and tensor-core barriers elsewhere. The kernel avoids putting a long-latency operation directly on the critical path. CLC applies the same idea to tile scheduling: ask for the next unit of work early, compute the current unit, then consume the scheduling result when it is needed.

## Relation to Persistent GEMM

The persistent GEMM in {ref}`chap_gemm_advanced` uses a static scheduler for the main walkthrough. A static scheduler is easier to explain because the next tile can be computed directly from loop state. For example, a scheduler such as `ClusterPersistentScheduler2D` can assign tiles using a grid-stride pattern over the output tile space.

CLC is the dynamic replacement for that static assignment. The outer loop stays the same: each resident cluster repeatedly computes one output tile and then advances to another. What changes is where the next tile comes from. With the static scheduler, the next tile is computed by a formula. With CLC, the next tile is returned by hardware work stealing.

That difference matters most near the tail of the launch. In a static schedule, the remaining work may not be evenly distributed. Some SMs may run out of assigned tiles while others still have several left. With CLC, the cluster that finishes early asks for another pending cluster coordinate. As long as there is work left in the launch queue, early finishers keep pulling more tiles.

It also matters when tile cost is not uniform. Some GEMM tiles may take different paths because of boundaries, masking, sparsity, grouped scheduling, or fused work around the main matrix multiply. A static schedule assumes the tile assignment is good enough before any of those costs are observed. CLC does not need that assumption. It assigns more work only after a cluster becomes available.

In TIRx, CLC can therefore be exposed as a dynamic tile scheduler. The programming model does not need to change the computation of a tile. The tile body is the same persistent GEMM body used by the static scheduler. The scheduler changes from "compute my next tile coordinate from a formula" to "ask hardware for the next available cluster coordinate." The result is the same persistent loop, but with hardware-driven work distribution instead of a fixed launch-time schedule.
