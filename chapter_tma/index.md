(chap_tma)=
# Asynchronous Data Movement: TMA

GEMM and attention are compute-bound at scale ({ref}`chap_performance`), but only if the Tensor
Cores stay fed. Feeding them is the job of the **Tensor Memory Accelerator (TMA)** — a hardware
engine that copies rectangular tiles between global and shared memory asynchronously, so the
threads don't spend their instruction budget on address arithmetic and loads.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: the TMA engine copying a tile from global to shared memory.*

## One Thread Issues, Hardware Moves the Tile

The defining property of TMA: a **single thread issues** a tile copy, and the **hardware performs
the transfer** in the background. There is no per-thread load/store loop — the descriptor tells
the engine the tile shape, strides, and destination, and the engine streams the bytes.

This is the **dispatch** knob again: the same logical "copy this tile" becomes a TMA transfer
rather than a cooperative thread copy, with very different performance and synchronization.

## Swizzled Layouts

TMA can apply layout **swizzling** as it writes into shared memory, producing a SMEM layout that
the Tensor Core can read without bank conflicts. The swizzle pattern is part of the TMA descriptor
and must match the layout the MMA expects — the concrete link between {ref}`chap_data_layout` and
the hardware: the TMA descriptor, the SMEM tile, and the MMA must all agree on the same swizzle.

## Completion: Loads vs. Stores

Because TMA is asynchronous, the kernel needs an explicit way to know a transfer finished. The two
directions use different mechanisms:

![TMA load synchronization flow](../img/tma_sync_flow.png)

- **Loads (GMEM → SMEM)** integrate with an **mbarrier** ({ref}`chap_async_barriers`). The issuing
  thread tells the barrier how many bytes to expect (`arrive.expect_tx(total_bytes)`); the hardware
  arrives on the barrier when exactly that many bytes have landed; consumers wait on the barrier
  before reading the tile.
- **Stores (SMEM → GMEM)** use a **commit-group / wait-group** mechanism instead: the kernel
  commits the issued store group and later waits for it to drain before reusing the SMEM buffer.

TMA turns tile movement from a per-thread loop into an asynchronous hardware operation with
byte-count tracking and explicit completion — exactly the structure that lets the load of the next
tile overlap the compute on the current one.
