(chap_tma)=
# Async Data Movement: TMA

:::{admonition} Overview
:class: overview

- TMA is a hardware engine for asynchronous tile copies between global memory and shared memory. One thread issues the copy, and the engine moves the bytes.
- A TMA copy is described by a tensor-map descriptor. The descriptor tells the engine the global tensor shape, strides, tile coordinates, and shared-memory swizzle mode.
- On the load path, TMA can swizzle the tile as it writes shared memory, so the tile lands in the layout expected by the Tensor Core.
- TMA loads complete through an `mbarrier` with byte-count tracking. TMA stores use a commit group and wait group.
:::

A Tensor Core only helps if it has data ready to consume. In a GEMM or attention kernel, the math may be compute-bound once the pipeline is full ({ref}`chap_performance`), but the pipeline only stays full if the next operand tile arrives in time.

The older way to move a tile is to have threads copy it themselves. Each thread computes addresses, issues loads from global memory, and stores values into shared memory. That works, but it spends warp instructions on address arithmetic and copy bookkeeping instead of compute. It also makes the copy path visible in the instruction stream of the same warps that are supposed to feed the Tensor Core.

The Tensor Memory Accelerator, or TMA, moves this work into a hardware copy engine. One thread issues a tile copy. The copy engine then moves a rectangular tile between global memory and shared memory asynchronously. While the engine is moving bytes, the rest of the CTA can continue with other work.

TMA also handles part of the layout problem. A Tensor Core does not just need the right values in shared memory. It needs them in the right shared-memory layout. On the load path, TMA can apply a shared-memory swizzle as it writes the tile. That lets the tile land directly in the layout the later MMA expects.

This chapter covers the three pieces needed to use TMA in a kernel: how one thread issues the copy, how swizzled shared-memory layouts are applied, and how the kernel waits for completion.

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: TMA copying a tile from global memory to shared memory. Toggle the swizzle mode and hover a source cell to see where it lands in shared memory.*

## One Thread Issues, Hardware Moves the Tile

A TMA copy starts with one issuing thread. That thread does not loop over all elements in the tile. It gives the hardware a description of the copy, then the TMA engine performs the transfer.

The main input is a tensor-map descriptor. The descriptor describes the global tensor and how a tile should be read from it. It records information such as the tensor shape, strides, element size, tile shape, and swizzle mode. The issuing thread also provides the shared-memory address where the tile should land.

After the instruction is issued, the copy runs asynchronously. The issuing thread can continue. Other threads in the CTA can also continue. The transfer is now the responsibility of the TMA engine, not a loop of ordinary load and store instructions.

This gives the kernel two different ways to express the same logical operation, "copy this tile."

One path is a thread copy. Threads cooperate to load from global memory and store into shared memory. This gives the kernel direct control over every access, but it consumes thread instructions and registers for address calculations.

The other path is a TMA copy. One thread issues the transfer, and the hardware copy engine performs the rectangular copy. This is the natural path for large regular tiles, especially the operand tiles used by Tensor Core kernels.

These two paths have different synchronization rules and different performance behavior. Choosing between them is a dispatch decision. The layout tells the kernel what memory arrangement it wants. The scope tells it which threads or CTAs are participating. The dispatch decides whether the copy is implemented by ordinary thread code or by TMA.

## Swizzled Layouts

Moving the tile is not enough. The tile also has to be placed in shared memory in a layout that the Tensor Core can read efficiently.

This is where TMA swizzling is used. As TMA writes the tile into shared memory, it can permute the shared-memory address pattern. The global memory tile is still a logical rectangle, but the destination layout in shared memory can be swizzled.

The swizzle mode is part of the TMA descriptor. Once the descriptor is set up, the issuing thread does not have to manually apply the swizzle. The engine applies it as the bytes land in shared memory.

The important requirement is agreement. The TMA descriptor, the shared-memory tile layout, and the later MMA instruction must all describe the same layout ({ref}`chap_data_layout`). If TMA writes the tile with one swizzle but the MMA reads it as if it had another, the hardware will still do exactly what it was asked to do. The bytes will simply be arranged incorrectly for the computation.

This is the point where the layout notation becomes more than a bookkeeping device. The layout used by the DSL has to match the hardware layout used by the TMA descriptor and the Tensor Core instruction. For example, if the kernel says an operand tile is stored in a 128-byte swizzled layout, the TMA descriptor has to use the matching swizzle mode, and the MMA dispatch has to expect that same shared-memory arrangement. The demo above lets you toggle between no swizzle and 128-byte swizzle; hover a source element to see where it lands once the swizzle is applied.

A useful way to read the swizzle is that TMA is not changing the logical tile. It is changing where the logical elements land physically in shared memory. The later MMA still consumes the same logical A or B tile. The swizzle only decides how that tile is arranged across shared memory banks.

## 3D TMA for Tiling and Swizzling

A plain TMA copy moves a flat 2D tile, but the shared-memory layout the Tensor Core wants is usually *tiled* into swizzle atoms (the 8 x 128-byte atoms from {ref}`chap_data_layout`). TMA handles that with an extra descriptor dimension. A **3D TMA** describes the shared-memory box as `(group, row, col)`, where the group dimension walks across atoms and the inner two address within one atom. A single 3D copy then both lays the tile out atom by atom (tiling) and applies the swizzle inside each atom, so the data arrives already in the layout the MMA expects, with no separate tiling or swizzling pass.

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tma_3d.html" title="Tiling and swizzling with 3D TMA" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*Interactive: a 3D TMA copy, addressed as (group, row, col), tiling into swizzled shared memory.*

Choosing the swizzle *format* is tied to this tiling. A wider swizzle scatters a column across more banks, so 128-byte swizzle is the default when it fits, but an N-byte atom needs the tile's contiguous dimension to fill it. A tile that is small because of a shape constraint therefore cannot use 128-byte swizzle and must step down to 64-byte or 32-byte: the rule of thumb is to pick the largest swizzle the tile can fill ({ref}`chap_data_layout`). The demo below shows the constraint directly: a 128-byte swizzle on a 16 x 16 tile becomes conflict-free only once the tile is split into 16 x 8 groups that match the atom.

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tiling_constraint.html" title="Swizzle imposes a tiling constraint" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
<script>
(function () {
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    document.querySelectorAll('iframe.demo-tma3d').forEach(function (f) {
      if (e.source === f.contentWindow) f.style.height = d.height + 'px';
    });
  });
})();
</script>
```
*Interactive: a 128-byte swizzle on a 16 x 16 tile, conflict-free once tiled into 16 x 8 groups.*

## Completion: Loads

The copy is asynchronous, so issuing it is not enough. A consumer cannot read the shared-memory tile just because the TMA instruction has been issued. The tile is safe to read only after the engine has finished writing the bytes.

For TMA loads, the completion signal is an `mbarrier` ({ref}`chap_async_barriers`).

The usual sequence is:

1. initialize or reuse an `mbarrier` for the pipeline stage;
2. tell the barrier how many bytes the TMA transfer is expected to write;
3. issue the TMA load;
4. let the TMA engine update the barrier as bytes arrive;
5. have the consumer wait on the barrier phase before reading the shared-memory tile.

The byte count is set with an operation such as:

```text
mbarrier.arrive.expect_tx(bytes)
```

This does two jobs. It records the expected transfer size, and it also performs the issuing thread's arrival on the barrier. The barrier is not complete merely because this call happened. It still waits for the TMA engine to report that the expected bytes have arrived.

As the transfer progresses, the engine performs complete-tx updates against the barrier. The barrier phase flips only after both conditions are met: the arrival count is satisfied, and the pending byte count reaches zero.

The consumer then waits on that barrier. Once the wait completes for the expected phase, the shared-memory tile is ready. At that point the MMA path can safely read it.

![TMA load synchronization flow](../img/tma_sync_flow.png)

This is the same barrier model used by other asynchronous producer-consumer handoffs. The producer is the TMA engine. The consumer is the MMA path or any other code that reads the shared-memory tile. The barrier is the explicit handoff between them.

## Completion: Stores

TMA stores move data in the opposite direction, from shared memory to global memory. They are also asynchronous, but the completion mechanism is different.

A TMA load usually feeds a consumer inside the same kernel. The MMA path needs to know when the shared-memory tile is ready. That is why the load path uses an `mbarrier`.

A TMA store usually writes the final data out to global memory. There is often no immediate in-kernel consumer waiting on the stored result. The main thing the kernel needs to know is when it is safe to reuse the shared-memory buffer or finish the store sequence.

For that, TMA stores use a commit group and wait group. The kernel issues one or more stores, commits the group, and later waits for the group to drain. After the wait completes, the stores in that group have finished from the kernel's point of view, and the shared-memory region used by the store can be reused safely.

So the rule is simple:

```text
TMA load:  wait through an mbarrier with byte-count tracking
TMA store: wait through a commit group and wait group
```

The two mechanisms serve the same purpose at different handoff points. Loads need to make a shared-memory tile visible to later consumers. Stores need to make sure an outgoing transfer is complete before the kernel reuses the source storage or relies on the store having drained.

## Why TMA Matters for Pipelining

TMA is most useful when it is part of a pipeline. A kernel can issue the load for a future tile while the Tensor Core computes the current tile. The load runs in the background. The compute runs in the foreground. The barrier connects the two when the future tile becomes the current tile.

A typical GEMM loop uses this structure repeatedly. One stage of shared memory holds the tile currently consumed by MMA. Another stage is being filled by TMA. As the loop advances, the roles rotate. Before MMA reads a stage, it waits on that stage's load barrier. Before TMA overwrites a stage, the kernel makes sure the previous consumer is done with it.

This is why TMA and `mbarrier` usually appear together in Blackwell- and Hopper-style kernels. TMA gives the kernel an asynchronous copy engine. The barrier gives the kernel a precise way to know when the copied bytes are ready.

Putting the pieces together, the data path is straightforward. TMA moves a rectangular tile from global memory into shared memory. The descriptor controls the global indexing and the shared-memory swizzle. The load completes through an `mbarrier`. Once the barrier fires, the Tensor Core can read the tile from shared memory. For stores, TMA moves data from shared memory back to global memory and the kernel waits through a commit group and wait group.

That is the practical model to keep in mind. TMA turns tile movement from a per-thread copy loop into an asynchronous hardware operation. The kernel still owns the schedule, the layout, and the synchronization, but it no longer has to spend the main compute threads on the mechanics of moving every element by hand.
