(chap_tmem)=
# Special Memory: TMEM

:::{admonition} Overview
:class: overview

- TMEM is a Blackwell-only per-SM 2D scratchpad (128 lanes × up to 512 columns) that holds the `tcgen05` accumulator and its scale factors.
- It is addressed by `TLane` × `TCol` and must be explicitly allocated and freed by the kernel, in 32-column units.
- Ordinary shared-memory ld/st cannot reach it; three asynchronous `tcgen05` instructions move data in and out.
:::

**Motivation.** Through Hopper, the Tensor Core's accumulator lived in registers, and that works
until the tiles grow large enough that the accumulator crowds out everything else a thread needs to
hold. A bigger MMA tile is exactly what you want for throughput, but registers are a fixed,
per-thread budget, so past some tile size the two goals collide. Blackwell's answer is a memory space
earlier GPUs simply do not have: **Tensor Memory (TMEM)**, a CTA-scoped 2D scratchpad — 128 lanes ×
up to 512 32-bit columns per CTA — where the `tcgen05` Tensor Core ({ref}`chap_tensor_cores`) keeps
its accumulator and, for block-scaled MMAs, its scale factors. Unlike registers, which the compiler
hands out for you, TMEM is something the kernel must explicitly allocate, fill, and free, so you
cannot write a Blackwell GEMM without understanding it. This chapter covers what TMEM is, how it is
addressed and allocated, and how data moves in and out of it — the foundation you will then put to
work in {ref}`chap_gemm_basics`.

## A 2D address space

On earlier generations a large MMA accumulator stayed in registers for the whole compute phase. On
Blackwell `tcgen05.mma` writes it to TMEM instead, and the space it writes into is **128 rows × up to
512 32-bit columns**, scoped to the CTA. That shape is worth pausing on, because it tells you how
TMEM is addressed. It is not a flat byte array that you index with a single offset; it is a genuine
two-dimensional grid. The rows are indexed by a hardware axis called `TLane`, of which there are 128,
and the columns by `TCol`, of which there are up to 512. So when you declare a TMEM buffer, you give
it a layout over those two axes, just as you would for any other tile. An accumulator, for example,
is written `S[(128, N) : (1@TLane, 1@TCol)]` in the notation of {ref}`chap_data_layout`.

![TMEM 2D layout: TLane rows × TCol columns](../img/tmem_layout.png)

## Allocation

Registers come to you for free, in the sense that the compiler decides which ones to use and when to
release them. TMEM does not work that way: the kernel has to **allocate and free** it explicitly.
The allocation is a per-CTA affair. One warp in the CTA performs it, requesting columns in units of
32, and the column count is rounded up to a power of two. From there you can think of TMEM the same
way you think of shared memory. It is a budgeted resource that belongs to the CTA, so you size it
much as you would size your SMEM ring buffers, and you have to stay within the per-CTA limit the
hardware gives you.

## Reading and writing TMEM

Since TMEM is an address space of its own, the ordinary `ld.shared` and `st.shared` instructions
cannot reach into it. Data travels in and out through three dedicated `tcgen05` instructions, one for
each path the accumulator and its scale factors need to take.

The first of these, **`tcgen05.ld`**, moves data from **TMEM into registers**. At the DSL level a
single copy is warpgroup-distributed, and it lowers to four warp-collective `tcgen05.ld` instructions
— one per warp, each handling its own 32 TMEM lanes, so that the four warps together cover all 128.
The instruction is not a single fixed shape but one drawn from a *family* of datapath atoms: the
shapes `.16x64b`, `.16x128b`, `.16x256b`, `.32x32b`, and `.16x32bx2`, each carrying a repeat factor
from `.x1` up to `.x128`. This path picks one of them, and the number of registers each thread ends
up with follows from the shape and repeat that were chosen. Whichever atom is used, it distributes
the TMEM tile across registers ({ref}`chap_layout_generations`) so that lane `l` receives row `l/4`
and two columns. The reason this particular layout matters is continuity: the epilogue pulls the
accumulator out of TMEM into the *same* per-lane fragment that an Ampere `mma` or a Hopper `wgmma`
would produce, which means it can cast and store the result using code that already exists. The
second instruction, **`tcgen05.st`**, simply runs that path in reverse — from **registers back into
TMEM**, in the same fragment — and you reach for it when a thread already holds data in registers, an
A operand for instance, and you want to stage it into TMEM. The third, **`tcgen05.cp`**, is a bulk
copy from **SMEM into TMEM** (the `32x128b.warpx4` form); this is the one that stages the scale
factors for a block-scaled MMA.

![tcgen05.ld / st move the TMEM accumulator to and from registers in the m8n8 fragment (lane l → row l/4, two columns)](../img/tcgen05_ldst.svg)

What all three have in common is the trait that also defines TMA: they are **asynchronous**. Each
returns before the data has actually moved, so anything that depends on the result has to be gated by
an explicit synchronization ({ref}`chap_async_barriers`). How you wait, though, depends on the
instruction. A `tcgen05.ld` or `tcgen05.st` completes through `tcgen05.wait::ld` or
`tcgen05.wait::st`, whereas a `tcgen05.mma` or `tcgen05.cp` completes through a commit group together
with an mbarrier. And when the result is handed off between threads, you need fences on top of that.

Taken together with TMA, these three instructions carry a tile through its entire life on Blackwell —
the very path you will write out by hand in {ref}`chap_gemm_basics`. TMA stages the operands into
SMEM, `tcgen05.mma` accumulates into TMEM, and then the epilogue uses `tcgen05.ld` to bring TMEM back
into registers and produce the output.
