(chap_tmem)=
# Special Memory: TMEM

:::{admonition} Overview
:class: overview

- TMEM is a Blackwell-only per-SM 2D scratchpad (128 lanes × up to 512 columns) that holds the `tcgen05` accumulator and its scale factors.
- It is addressed by `TLane` × `TCol` and must be explicitly allocated and freed by the kernel, in 32-column units.
- Ordinary shared-memory ld/st cannot reach it; three asynchronous `tcgen05` instructions move data in and out.
:::

Through Hopper, the Tensor Core's accumulator lived in registers — which works until the tiles grow
large enough that the accumulator crowds out everything else the threads need to hold. Blackwell's
answer is a memory space earlier GPUs do not have: **Tensor Memory (TMEM)**, a per-SM 2D scratchpad
where the `tcgen05` Tensor Core ({ref}`chap_tensor_cores`) keeps its accumulator and, for
block-scaled MMAs, its scale factors. This
chapter covers what TMEM is, how it is addressed and allocated, and how data moves in and out of it.

## A 2D address space

Earlier generations kept large MMA accumulators in registers throughout the compute phase; on
Blackwell `tcgen05.mma` instead writes them to TMEM — **128 rows × up to 512 32-bit columns** per
SM. The shape hints at how TMEM is addressed: not as a flat byte array, but as a genuine grid. Rows
are indexed by a hardware axis called `TLane` (128 lanes) and columns by `TCol` (up to 512), and a
TMEM buffer is declared with a layout over those two axes. An accumulator, for instance, is
`S[(128, N) : (1@TLane, 1@TCol)]` in the notation of {ref}`chap_data_layout`.

![TMEM 2D layout: TLane rows × TCol columns](../img/tmem_layout.png)

## Allocation

Unlike registers, which the compiler hands out automatically, TMEM must be explicitly **allocated
and freed** by the kernel. A single warp does the allocation, in units of 32 columns, with the
column count rounded up to a power of two. TMEM is a budgeted per-SM resource,
much like SMEM: a kernel sizes its TMEM the same way it sizes its SMEM ring buffers, and has to live
within the per-SM limit.

## Reading and writing TMEM

Because TMEM is its own address space, the ordinary `ld.shared` / `st.shared` instructions do not
reach it. Data moves in and out through three dedicated `tcgen05` instructions, one for each path the
accumulator and its scale factors need to travel.

The first, **`tcgen05.ld`**, moves **TMEM → registers**. It is a warpgroup-cooperative load with a
*fixed fragment layout* (the `.32x32b` / `.16x*b` datapath atoms) that distributes the TMEM tile
into registers as the **m8n8 register fragment** ({ref}`chap_layout_generations`) — lane `l` gets row
`l/4` and two columns. That fixed layout gives continuity: the epilogue pulls the
accumulator out of TMEM into the *same* per-lane fragment an Ampere `mma` or Hopper `wgmma` produces,
so it can cast and store the result with code that already exists. The second, **`tcgen05.st`**, is
the reverse — **registers → TMEM**, in that same fragment — used to stage data a thread already holds
in registers (an A operand, say) into TMEM. The third, **`tcgen05.cp`**, is a bulk **SMEM → TMEM**
copy (the `32x128b.warpx4` form); this is the instruction that stages a block-scaled MMA's scale
factors.

![tcgen05.ld / st move the TMEM accumulator to and from registers in the m8n8 fragment (lane l → row l/4, two columns)](../img/tcgen05_ldst.svg)

All three share TMA's defining trait: they are **asynchronous**. Like TMA and the MMA itself, they
return before the data has actually moved, so a `tcgen05.wait` / commit must gate any consumer that
depends on the result ({ref}`chap_async_barriers`).

These three instructions, together with TMA, carry a tile through its full life on Blackwell, the
path you will write directly in {ref}`chap_gemm_basics`: TMA stages the operands into
SMEM, `tcgen05.mma` accumulates into TMEM, and the epilogue `tcgen05.ld`s TMEM back into registers to
produce the output.
