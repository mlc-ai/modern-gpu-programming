(chap_tmem)=
# Tensor Memory (TMEM)

Blackwell adds a memory space that earlier GPUs do not have: **Tensor Memory (TMEM)**, a per-SM 2D
scratchpad where the `tcgen05` Tensor Core ({ref}`chap_tensor_cores`) keeps its accumulator and, for
block-scaled MMAs, its scale factors. Because it is a *memory*, it belongs here alongside TMA: this
chapter covers what TMEM is, how it is addressed and allocated, and how data moves in and out of it.

## A 2D address space

Earlier generations kept large MMA accumulators in registers throughout the compute phase; on
Blackwell `tcgen05.mma` instead writes them to TMEM — **128 rows × up to 512 32-bit columns** per
SM. TMEM is not addressed like linear memory: rows are indexed by a hardware axis called `TLane`
(128 lanes) and columns by `TCol` (up to 512). A TMEM buffer is declared with a layout over those
axes — e.g. an accumulator is `S[(128, N) : (1@TLane, 1@TCol)]` in the notation of
{ref}`chap_data_layout`.

![TMEM 2D layout: TLane rows × TCol columns](../img/tmem_layout.png)

## Allocation

TMEM must be explicitly **allocated and freed** by the kernel. Allocation is done by a single warp,
in units of 32 columns, with the column count rounded up to a power of two — so TMEM is a budgeted
per-SM resource, like SMEM. A kernel sizes its TMEM the way it sizes its SMEM ring buffers.

## Reading and writing TMEM

TMEM is not addressed like SMEM — there is no `ld.shared` / `st.shared` for it. Data moves in and
out through three dedicated `tcgen05` instructions:

- **`tcgen05.ld` — TMEM → registers.** A warpgroup-cooperative load with a *fixed fragment layout*
  (the `.32x32b` / `.16x*b` datapath atoms). It distributes the TMEM tile into registers in the
  **m8n8 register fragment** ({ref}`chap_layout_generations`) — lane `l` gets row `l/4`, two
  columns. So the epilogue pulls the accumulator out of TMEM into the *same* per-lane fragment an
  Ampere `mma` or Hopper `wgmma` produces, then casts and stores it. In TIRx:
  `Tx.wg.copy_async(reg, tmem[...])`.
- **`tcgen05.st` — registers → TMEM.** The reverse, in that same fragment — used to stage data a
  thread already holds in registers (e.g. an A operand) into TMEM. `Tx.wg.copy_async(tmem[...], reg)`.
- **`tcgen05.cp` — SMEM → TMEM.** A bulk copy (the `32x128b.warpx4` form) — this is what stages the
  block-scaled MMA's scale factors. `Tx.copy_async(tmem, smem)`.

![tcgen05.ld / st move the TMEM accumulator to and from registers in the m8n8 fragment (lane l → row l/4, two columns)](../img/tcgen05_ldst.svg)

All three are **asynchronous**: like TMA and the MMA, they return before the data has moved, so a
`tcgen05.wait` / commit gates any consumer ({ref}`chap_async_barriers`).

When you reach {ref}`chap_gemm_basics`, this is the path you will write directly: TMA stages the
operands into SMEM, `tcgen05.mma` accumulates into TMEM, and the epilogue `tcgen05.ld`s TMEM back
into registers to produce the output.
