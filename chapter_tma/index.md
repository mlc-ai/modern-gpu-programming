(chap_tma)=
# Async Data Movement: TMA

:::{admonition} Overview
:class: overview

- TMA is a hardware engine that asynchronously copies rectangular tiles between global and shared memory: one thread issues the copy, the engine moves the bytes.
- It can swizzle a tile as it writes SMEM so the layout matches what the Tensor Core expects.
- Loads complete through an mbarrier (byte-count tracked); stores through a commit/wait group.
:::

**Motivation.** A Tensor Core that can do 2 PFLOP/s is worthless the moment it sits idle waiting for
data, and at scale GEMM and attention are only compute-bound ({ref}`chap_performance`) when the cores
stay fed. The classic way to feed them is to have the threads loop over addresses and copy tiles in
themselves, but that spends the warp's instruction budget on bookkeeping — index arithmetic and
load/store issue — that has nothing to do with the math. The **Tensor Memory Accelerator (TMA)** is
the hardware engine that fixes this: one thread issues a copy, and the engine moves the rectangular
tile between global and shared memory on its own, leaving the threads free to compute. This chapter
covers how a single thread issues that copy, how TMA swizzles the tile so the layout matches what the
Tensor Core expects ({ref}`chap_data_layout`), and how loads and stores signal completion — loads
through an mbarrier ({ref}`chap_async_barriers`), stores through a commit/wait group.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the TMA engine copying a tile from global to shared memory.*

## One Thread Issues, Hardware Moves the Tile

The idea behind TMA is to split the work between the threads and the hardware. Instead of every
thread running its own load/store loop, a single thread issues one tile copy, and the engine carries
out the transfer in the background while the rest of the warp gets on with the math. To issue the
copy, that thread hands the engine a tensor-map descriptor — a compact record of the global tensor's
shape, strides, the tile (or box) to read, and the swizzle to apply — together with the destination
SMEM address it should copy into. From there the engine streams the bytes on its own.

It helps to notice that the same logical operation, "copy this tile," can be realized in two quite
different ways. One option is for the threads to cooperate, each pulling its share of the data; the
other is for a single thread to issue a TMA transfer and let the engine do the rest. The two paths
behave differently in both performance and synchronization, so picking between them is a genuine
*dispatch* decision — exactly the scope / layout / dispatch lens we introduced at the start of the
book.

## Swizzled Layouts

Moving the tile is only half the job; it also has to land in a form the Tensor Core can read
efficiently. This is where **swizzling** comes in. As TMA writes the tile into shared memory, it can
permute the layout so that the Tensor Core later reads it without bank conflicts — provided the
swizzle it applies matches the layout (and mode) the MMA expects. The swizzle pattern itself rides
along in the TMA descriptor, which is what lets one thread set it once and have the engine apply it to
the whole tile.

This is the point where {ref}`chap_data_layout` meets the hardware. The TMA descriptor, the SMEM
tile, and the MMA all have to agree on the same swizzle. If they disagree, the engine will still
faithfully deposit bytes into shared memory — they simply will not be arranged the way the Tensor Core
wants to read them, and the computation will be wrong.

## Completion: Loads vs. Stores

Asynchrony is what buys us the overlap, but it also raises a question the old synchronous loop never
had to worry about. Because the issuing thread returns immediately — long before the bytes have
actually arrived — how does the kernel know the transfer has finished before something tries to read
the data? The answer is that TMA needs an explicit completion signal. Loads and stores handle this
differently, so it is worth looking at each in turn.

![TMA load synchronization flow](../img/tma_sync_flow.png)

A **load (GMEM → SMEM)** ties into an **mbarrier** ({ref}`chap_async_barriers`). Before the transfer
begins, the issuing thread tells the barrier how many bytes to expect with
`mbarrier.arrive.expect_tx(bytes)`. As the data lands, the engine emits `complete-tx` signals that
account for the bytes that have arrived, and the barrier's phase only flips once both the arrival
count and the tx-count have been satisfied. Consumers wait on that barrier, so they never touch the
tile until it is fully in place. A **store (SMEM → GMEM)** is simpler, because nothing downstream is
waiting on the result the way a consumer waits on a freshly loaded tile. It uses a lighter-weight
**commit-group / wait-group** mechanism instead: the kernel commits the store group it issued and
later waits for that group to drain before it reuses the SMEM buffer.

Put together, TMA turns tile movement from a per-thread loop into an asynchronous hardware operation
with byte-count tracking and explicit completion. That is precisely the structure a pipelined kernel
is built on: while the Tensor Cores work through the current tile, the load of the next one can run in
the background, and by the time the cores are ready the data is already waiting for them.
