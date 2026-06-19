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

TMA divides the labor: a **single thread issues** a tile copy, and the **hardware performs the
transfer** in the background. There is no per-thread load/store loop. The issuing thread hands the
engine a tensor-map descriptor — the global tensor's shape, strides, and tile/box and swizzle — and a
destination SMEM address at the copy, and the engine streams the bytes on its own.

The same logical "copy this tile" can be realized two ways. The threads can cooperate, each pulling
its share of the data, or one thread can issue a TMA transfer and let the engine do the rest. The
two paths differ in both performance and synchronization, and choosing between them is a *dispatch*
decision — the scope / layout / dispatch lens from the book's introduction.

## Swizzled Layouts

The tile also has to land in a form the Tensor Core can read efficiently. TMA can apply layout
**swizzling** as it writes into shared memory, producing a SMEM layout that the Tensor Core reads
without bank conflicts when the swizzle matches the layout/mode the MMA expects. The swizzle pattern travels in the TMA descriptor, and it must match the
layout the MMA expects. This links {ref}`chap_data_layout` to the hardware: the TMA descriptor, the
SMEM tile, and the MMA must all agree on the same swizzle, or the bytes the engine deposits will not
be the bytes the core wants.

## Completion: Loads vs. Stores

Asynchrony buys the overlap, but it raises a question the synchronous loop never had to answer: since
the issuing thread returns immediately, how does the kernel know the transfer finished before
something reads the data? TMA needs an explicit completion signal, and loads and stores use different
mechanisms.

![TMA load synchronization flow](../img/tma_sync_flow.png)

A **load (GMEM → SMEM)** integrates with an **mbarrier** ({ref}`chap_async_barriers`). The issuing
thread records the expected byte (tx) count (`mbarrier.arrive.expect_tx(bytes)`); the engine issues
`complete-tx` as bytes land, and the phase flips only once both the arrival count and the tx-count are
satisfied; consumers wait on the barrier before touching the tile. A **store (SMEM → GMEM)** has no consumer waiting on the result, so it
uses a lighter-weight **commit-group / wait-group** mechanism instead: the kernel commits the issued
store group and later waits for it to drain before reusing the SMEM buffer.

TMA turns tile movement from a per-thread loop into an asynchronous hardware operation with
byte-count tracking and explicit completion. This is the structure a pipelined kernel needs: the
load of the next tile can run in the background while the Tensor Cores process the current one.
