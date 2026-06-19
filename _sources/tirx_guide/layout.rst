..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _chap_data_layouts:

Tensor Layout
=============

.. admonition:: Overview
   :class: overview

   - The TIRx layout API: ``TileLayout`` (shard / replica / offset over named axes), ``SwizzleLayout``, and ``ComposeLayout``.
   - A layout maps each logical index to a *set* of physical coordinates; ``layout.apply()`` evaluates it.
   - Ready-made constructors (``tmem_datapath_layout``, ``tcgen05_atom_layout``, ``wg_local_layout``) cover the common hardware tiles.

Notation tells you what a layout *means*; it does not yet put a
single number anywhere on the chip. To write a kernel you need every element of a
tile pinned to a precise physical place — which thread holds it, which register or
TMEM lane it lands in — and you need that placement to be something the compiler
can read, check, and feed to each tile op rather than a convention you carry in
your head. That is what the TIRx layout API gives you: a ``TileLayout`` you
construct once and attach to a buffer, written in a fixed vocabulary of named
hardware axes and evaluated by a single rule. :ref:`chap_data_layout` introduced
the layout *notation* — the shape–stride pair ``S[shape : strides]``, strides
tagged with named **axes**, and the replication term ``R[n : stride]`` for data
the hardware copies rather than partitions; this chapter turns that notation into
the real objects you import from ``tvm.tirx.layout`` — ``TileLayout``,
``SwizzleLayout``, and ``ComposeLayout`` — together with the axis vocabulary they
are written in, the ready-made constructors TIRx ships for the common hardware
tiles, and the exact rule a layout evaluates to, run through two real hardware
tiles. If the notation below looks unfamiliar, read that chapter first; here we
assume it and build on it.

The workflow is deliberately simple. You build a layout once and attach it to a
buffer — either through ``pool.alloc(shape, dtype, layout=...)`` or through
``T.decl_buffer(shape, dtype, scope=..., layout=...)`` — and from then on every
tile op that touches that buffer reads its placement straight from the layout, so
you never repeat the placement at each use site. All of the objects you attach
live in a single module::

    from tvm.tirx.layout import (
        TileLayout, SwizzleLayout, ComposeLayout,    # the three layout classes
        S, R,                                        # shard / replica spec builders
        laneid, warpid, tid_in_wg, TLane, TCol, m,   # named axes (a few of many)
        tcgen05_atom_layout, tmem_datapath_layout,   # ready-made layout constructors
    )

There is really only one idea to carry over from :ref:`chap_data_layout`. A layout
does not map a logical index to a single address; it maps it to a *set* of
coordinates on the named axes, because one element can genuinely live in several
places at once. That map breaks cleanly into three parts — shard (``D``), replica
(``R``), and offset (``O``) — and the rest of the chapter builds it up one piece
at a time.

Interactive Demo
----------------

Before we get into the mechanics, it helps to have something concrete to poke at.
The demo below lets you pick a preset, edit the logical shape and the ``S/R/O``
layout, choose a dtype and a swizzle mode, and then click on an element to see
which physical thread or threads own it. It is worth coming back to as each piece
of notation is introduced: the math in the sections that follow is just a precise
description of what you can watch happen here.

.. raw:: html

   <p>
     <a class="reference external" href="../_static/tirx-layout-demo/index.html"
        target="_blank" rel="noopener"
        style="display:inline-block; padding:10px 18px; background:#3b82f6;
        color:#fff !important; font-weight:700; border-radius:8px;
        text-decoration:none;">▶ Open the demo full screen ↗</a>
   </p>
   <iframe id="tirx-layout-demo-frame" src="../_static/tirx-layout-demo/index.html?notitle"
           style="width:100%; height:1040px; border:1px solid #dfe1e6;
           border-radius:10px; margin:10px 0 6px; display:block;"
           title="TIRx interactive layout demo" loading="lazy"></iframe>
   <script>
   // The demo (viz-base.js) posts its content height; size the iframe to fit so
   // there is no inner scrollbar. This demo is responsive (fills the width), so
   // only the height follows content.
   (function () {
     var f = document.getElementById('tirx-layout-demo-frame');
     window.addEventListener('message', function (e) {
       var d = e.data;
       if (!d || d.type !== 'demoHeight' || !d.height) return;
       if (f && e.source === f.contentWindow) f.style.height = d.height + 'px';
     });
   })();
   </script>

TileLayout
----------

We write a **TileLayout** as ``S[shape : strides]``. The ``S`` marks a shard spec,
which you can read as "a tile of this *shape*, with each logical index mapped by
these per-axis *strides*"; when an element needs to be replicated, you extend it
with a replica set and write ``S[...] + R[...]``. Until replication enters the
picture this is nothing more than the familiar shape–stride pair, only with named
axes. Underneath, a layout is assembled from three parts taken in turn — shard
(``D``), replica (``R``), and offset (``O``) — so let us look at each in order.

The atom that every part is built from is the **iter**, a triple
``(extent, stride, axis)`` describing a single linear, strided walk along one axis.
You can think of an iter as the named-axis version of one shape/stride entry:
*extent* many steps, *stride* apart, along *axis*.

- **D (Shard).** This is a list of one or more iters, each carrying an extent and a
  stride on some axis. The shard partitions the logical index across these iters
  and produces a single base coordinate, generalizing the ordinary shape–stride
  rule to several axes at once. We write it in parentheses, for example
  ``S[(8,2,4,2):(4@laneid,1@warpid,1@laneid,1)]``.
- **R (Replica).** This is a set of replication iters that enumerate offsets in
  hardware space, independent of the logical index. Adding each element of that set
  to the shard result is what gives us replication, or broadcasting. We write it in
  square brackets, for example ``R[2:4@warpid]``.
- **O (Offset).** This is a fixed coordinate offset — one integer per axis — added
  to every result. It is how you place data at a chosen base position or reserve a
  resource for exclusive use.

The three parts compose in a natural order: the shard finds the base coordinate,
the replica fans it out, and the offset shifts the whole result. So for a logical
index ``x`` the layout produces

.. math::

   L(x) = \{\, D(x) + r + O \mid r \in R \,\},

Here ``D(x)`` is the base coordinate coming from the sharded iters, ``r`` ranges
over all combinations of the replica iters (collapsing to a single zero offset
when ``R`` is empty), and ``O`` is the constant offset. The set notation is what
lets ``L(x)`` stay a singleton in the common case while still being able to hold
several coordinates once ``R`` is non-empty. Each term is written ``n @ axis``, and
if a stride is not paired with an axis, the memory axis ``m`` is assumed by default.

Written out in TIRx, a shard together with a replica and an offset reads::

    TileLayout(S[(8,2,4,2) : (4@laneid, 1@warpid, 1@laneid, 1)] + R[2:4@warpid] + 5@warpid)

Reading this left to right: ``S[...]`` and ``R[...]`` build the spec, the ``+``
operator glues the shard, the replica, and a bare ``n @ axis`` offset together, and
``TileLayout(...)`` wraps the whole thing into the object you attach to a buffer.
(If you already have pre-built ``Iter(extent, stride, axis)`` objects, you pass them
through ``TileLayout.from_iters(shard, replica, offset)`` instead.)

What makes the axes carry real meaning is that each one names a specific hardware
resource, and that name is part of the layout itself rather than something inferred
from context. Blocks ``bx`` / ``by`` / ``bz`` and cluster blocks ``cbx`` / ``cby``
/ ``cbz`` place a tile across the grid; the thread axes ``tx`` (block-wide),
``warpid``, ``laneid``, ``wgid``, ``tid_in_wg``, and ``wid_in_wg`` spread it across
a CTA or warpgroup; ``m`` is linear memory; ``P`` / ``F`` are the partition and
free axes of a 2D scratchpad; ``Bank`` is a shared-memory bank; and ``TLane`` /
``TCol`` address Blackwell tensor memory.

Forward Mapping
~~~~~~~~~~~~~~~

Evaluating a layout means taking a logical coordinate and working out where it
lands. The method ``layout.apply(*coord)`` does exactly this: it hands you back a
dict from axis name to coordinate, such as ``{"laneid": …, "warpid": …, "m": …}``,
and the four steps below are precisely what it runs inside. Concretely, evaluating
``L(x)`` for a coordinate ``x = (x_0, …, x_{r-1})`` drawn from a shape
``(S_0, …, S_{r-1})`` comes down to four mechanical steps — flatten, split,
accumulate, broadcast — and we will lean on them for the rest of the chapter.

**1. Flatten** the coordinate row-major to a single index:

.. math::

   \mathrm{flat} = \sum_{d} x_d \prod_{e > d} S_e .

**2. Split** that index across the shard extents ``(e_0, …, e_{n-1})``
(one component ``c_k`` per shard iter, innermost-first):

.. math::

   c_k = \left\lfloor \mathrm{flat} \,\Big/ \textstyle\prod_{l > k} e_l \right\rfloor
         \bmod e_k .

**3. Accumulate** each component onto its axis with its stride to get the base
coordinate, then **add the offset**:

.. math::

   D(x)[a] = \sum_{k\,:\,a_k = a} c_k\, s_k ,
   \qquad
   \bigl(D(x) + O\bigr)[a] = D(x)[a] + O[a] .

**4. Broadcast** the replica iters: ``r`` ranges over
``∏_t [0, e_t)`` and adds, per replica iter ``(e_t, s_t, a_t)``, ``r_t s_t`` to
axis ``a_t`` — yielding the set ``L(x)``:

.. math::

   L(x)[a] = D(x)[a] + O[a] + \sum_{t\,:\,a_t = a} r_t\, s_t .

One consequence of doing flatten before split is worth pausing on: the layout never
hard-codes the input shape. A layout *admits* any shape whose total size equals
``∏_k e_k``, and once that holds the very same layout works for all of them, because
the flatten/split step simply re-derives the per-iter components from whatever shape
it happens to be handed.

Case Study: NVIDIA Tensor-Core Tile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quickest way to see that these four steps do something real is to run them on
an actual tensor-core tile and watch a hardware mapping fall out. Consider a logical
``(8, 16)`` tile distributed across 2 warps of 32 lanes each, where each lane holds
part of the tile in its own registers (the ``reg`` slot is just the default memory
axis ``m``)::

    TileLayout(S[(8,2,4,2):(4@laneid,1@warpid,1@laneid,1)] + R[2:4@warpid] + 5@warpid)

The shard factors the logical indices into four iters of extent ``8, 2, 4, 2`` over
the axes ``laneid, warpid, laneid, m``. Let us run the four steps on a generic
element ``(i, j)``:

#. **flatten:** ``flat = 16 i + j``.
#. **split** by ``(8, 2, 4, 2)``: ``c_0 = i``, ``c_1 = ⌊j/8⌋``,
   ``c_2 = ⌊j/2⌋ mod 4``, ``c_3 = j mod 2``.
#. **accumulate + offset:** ``laneid = 4 c_0 + c_2 = 4 i + ⌊j/2⌋ mod 4`` (two
   iters land on ``laneid``); ``warpid = c_1 + 5 = ⌊j/8⌋ + 5`` (offset
   ``5@warpid``); ``m = c_3 = j mod 2``.
#. **replica** ``R[2:4@warpid]``: ``r ∈ {0, 1}`` adds ``4r`` to ``warpid``, so
   each element lives on **two** warps.

So the full mapping is

.. math::

   \mathrm{laneid} = 4 i + \lfloor j/2 \rfloor \bmod 4, \quad
   \mathrm{warpid} = \lfloor j/8 \rfloor + 5 + 4 r\ (r \in \{0,1\}), \quad
   m = j \bmod 2 .

So the shard places the tile on warps ``{5, 6}`` (from ``⌊j/8⌋ + 5``), and the
replica then copies it to ``{9, 10}``. Spelling out a few elements makes the pattern
concrete:

.. list-table::
   :header-rows: 1
   :widths: 14 10 26 12 20 10

   * - ``(i, j)``
     - flat
     - ``(c0, c1, c2, c3)``
     - laneid
     - warpid (×2)
     - ``m``
   * - ``(0, 0)``
     - 0
     - ``(0, 0, 0, 0)``
     - 0
     - ``{5, 9}``
     - 0
   * - ``(0, 1)``
     - 1
     - ``(0, 0, 0, 1)``
     - 0
     - ``{5, 9}``
     - 1
   * - ``(0, 2)``
     - 2
     - ``(0, 0, 1, 0)``
     - 1
     - ``{5, 9}``
     - 0
   * - ``(1, 0)``
     - 16
     - ``(1, 0, 0, 0)``
     - 4
     - ``{5, 9}``
     - 0
   * - ``(0, 8)``
     - 8
     - ``(0, 1, 0, 0)``
     - 0
     - ``{6, 10}``
     - 0
   * - ``(7, 15)``
     - 127
     - ``(7, 1, 3, 1)``
     - 31
     - ``{6, 10}``
     - 1

Case Study: Blackwell Tensor Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous example bound its strides to thread axes, but the model does not
care what an axis means — bind the strides to different axes and everything works
the same way. To see that, we point the same model at Blackwell **tensor
memory**, a 2D address space addressed by ``TLane`` × ``TCol`` (both of them
*memory* axes). Now every stride binds to a memory axis, so the layout is a pure
placement: no threads, no replica, no offset, and ``L(x)`` is a plain singleton::

    TileLayout(S[(2,128,112):(112@TCol,1@TLane,1@TCol)])

If we take the logical tile shape to equal the shard extents, ``(2, 128, 112)``,
then the split step becomes the identity (``c_k = x_k``) and element ``(a, l, c)``
maps to:

.. math::

   \mathrm{TLane} = l, \qquad \mathrm{TCol} = 112\,a + c .

Reading off the iters, the extent-128 iter (``1@TLane``) lays the tile across
**128 lanes**, while the extent-2 iter (``112@TCol``) and the extent-112 iter
(``1@TCol``) together cover **224 columns** (``TCol = 112 a + c ∈ [0, 224)``). Once
again, a few elements make this concrete:

.. list-table::
   :header-rows: 1
   :widths: 28 14 14

   * - ``(a, l, c)``
     - TLane
     - TCol
   * - ``(0, 0, 0)``
     - 0
     - 0
   * - ``(0, 5, 3)``
     - 5
     - 3
   * - ``(1, 0, 0)``
     - 0
     - 112
   * - ``(1, 127, 111)``
     - 127
     - 223

Notice that the 224-wide span is intentionally **not a power of two**. A
block-scaled FP8 GEMM may reach for a 224-column tile precisely because tensor
memory cannot hold two accumulator stages plus the scale factors at a full 256
columns. This is exactly the kind of shape that general-shape support lets the
layout express directly, instead of forcing it through a power-of-two mold.

**Scale factors (SFA / SFB).** The accumulator above needed no replica; its scale
factors are where the replica earns its keep. A block-scaled MMA keeps its per-block
scale factors in tensor memory too, and there a single physical group has to feed
several warps at the same time — exactly the broadcast that a replica provides. The
atom looks like this::

    TileLayout(S[(32, sf_per_mma):(1@TLane, 1@TCol)] + R[4:32@TLane])

Here we have 32 rows on ``TLane`` and ``sf_per_mma`` scale factors on ``TCol``, and
the replica ``R[4:32@TLane]`` replicates that 32-row group along the TMEM ``TLane``
axis (32 → 128 TMEM lanes, with stride 32 covering ``TLane`` 0–127). This is the
"warpx4" router: each of the four warps' 32-lane TMEM window sees the same
scale-factor group, and the actual reads are then carried out by those warps'
threads. From there the atom is direct-summed with an outer iter over ``(M rows,
K scale-factor groups)``, packing ``epc = 32 / SF_bits`` scale factors into each
32-bit ``TCol`` cell (for instance four fp8 ``e8m0`` SFs per cell); optional
stride-0 ``reuse`` and outer ``pipe_depth`` iters then capture SF reuse across MMAs
and double-buffering. The payoff is that a single ``TileLayout`` model expresses
both the accumulator (a pure placement, no replica) and its scale factors (a
replicated, routed placement) within the very same tensor-memory address space.

Beyond GPU Registers
~~~~~~~~~~~~~~~~~~~~~~

The two case studies were not special cases at all; they were the one model pointed
at two different pieces of hardware, and that generality is the whole reason for
naming axes in the first place. Bind strides to the block and cluster axes (``bx``
… ``cbz``) and the layout **shards a tile across the grid**; bind them to on-chip
memory axes instead and it expresses the native accelerator memories — a
2D-partitioned scratchpad (partition ``P`` and free ``F`` axes), shared-memory banks
(``Bank``), or NVIDIA Blackwell tensor memory with its native 2D addressing
(``TLane`` × ``TCol``). The demo ships a preset for each of these, so you can swap
the target hardware without ever leaving the layout language.

Ready-Made Layouts
~~~~~~~~~~~~~~~~~~~

In practice you rarely hand-write the hardware tiles above. The placements that come
up again and again — a ``tcgen05`` accumulator, the register fragment a
``tcgen05.ld`` produces, a warpgroup-local register tile — already ship as
constructors in ``tvm.tirx.layout``, each one returning a ready-to-use
``TileLayout``:

- ``tmem_datapath_layout(datapath, rows, cols)`` gives you the logical-row →
  physical TMEM-lane placement that an MMA writes, for datapath ``"D"`` (M=128, an
  identity row→lane map) or ``"F"`` (M=64, warp-scattered).
- ``tcgen05_atom_layout(instr_shape, tensor_shape, dtype)`` gives you the
  per-warpgroup register tile that a ``tcgen05.ld`` / ``tcgen05.st`` atom
  (``.32x32b``, ``.16x64b``, …) moves to and from registers. At the DSL level the
  tile is warpgroup-distributed, but it lowers to four warp-collective
  ``tcgen05.ld`` / ``tcgen05.st`` instructions — each PTX instruction is issued by
  one warp, which moves its own 32 TMEM lanes.
- ``wg_local_layout(cols, rows=128)`` gives you a warpgroup-local register tile,
  with one row per thread on ``tid_in_wg``.

Every one of these returns an ordinary ``TileLayout`` built from the same ``S`` /
``R`` spec, so nothing is hidden: you can inspect what it produced, compose it
further, or drop down to writing the spec by hand whenever a shape is unusual —
which is exactly what the accumulator case study above did.

SwizzleLayout, ComposeLayout
----------------------------

Everything so far has been affine: a logical index multiplied by some strides, plus
an offset. That is enough to describe placement, but it runs into a wall at one of
the most common performance problems in shared memory — bank conflicts — because the
cure for those is a deliberately *non-linear* shuffle of addresses. This is why TIRx
introduces a second kind of layout.

A *swizzle* is an XOR-based permutation of the linear memory address. There is no
way to write it as a strided ``TileLayout``, so rather than twist the affine model
out of shape to accommodate it, TIRx keeps it separate and composes the two: a
``SwizzleLayout`` stacked on top of the tile layout, written
``ComposeLayout(swizzle, tile)``. The tile layout produces a linear memory address,
and the swizzle then permutes that address. The next few sections explain why the
permutation is needed in the first place and how it is defined.

Why Swizzle
~~~~~~~~~~~

Recall the mechanism from :ref:`chap_data_layout`. Shared memory is organized as
**32 banks of 4 bytes**, and a *bank conflict* occurs — serializing the access —
whenever the lanes of a single access touch different addresses that fall in the
same bank. With a plain row-major tile this conflict is not bad luck but
*structural*. Take the ``(8, 64)`` ``float16`` tile ``S[(8,64):(64@m,1@m)]``, where
element ``(i, j)`` sits at address ``m = 64i + j``. A single row spans
``64 × 2 = 128`` bytes, which is exactly one bank line, so walking *down a column*
(holding ``j`` fixed and increasing ``i``) jumps a full bank line on every step and
lands on the **same bank** each time — an 8-way column conflict (8 rows all funneled
into one bank). The transform below is what scatters those accesses back across the
banks, and we will close the loop on this very tile in the worked example.

The Transform
~~~~~~~~~~~~~

The idea behind the cure is simple: make a column's addresses depend on the row, so
that they scatter across banks instead of stacking on one. An XOR achieves this
cheaply, with no multiply and no table lookup. A ``SwizzleLayout`` is controlled by
three integer parameters — ``per_element`` (M), ``swizzle_len`` (B), and
``atom_len`` (S) — and maps a linear element address ``m`` as follows, leaving the
low ``M`` bits untouched while XOR-ing a higher group of bits down into a lower one:

.. math::

   \text{addr}(m) = \bigl(f(m \gg M)\bigr)\!\cdot\! 2^{M} + (m \bmod 2^{M}),
   \qquad
   f(x) = x \oplus \bigl((x \mathbin{\&} (\,(2^{B}-1)\ll S\,)) \gg S\bigr).

In words, the bits at positions ``[S, S+B)`` of ``x = m >> M`` are XOR-ed into the
bits ``[0, B)``. For the layout to be well-formed we need ``S ≥ B``.

Choosing the Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

You will seldom choose ``M``, ``B``, and ``S`` by hand. In practice they are
pinned down by two things you already know — the **element dtype** and the
**swizzle mode**, that is, the 32B / 64B / 128B shared-memory swizzle widths:

.. math::

   M = \operatorname{bitlen}\!\left(\frac{128}{\text{dtype bits}}\right) - 1,
   \qquad
   B = \begin{cases} 1 & 32\text{B} \\ 2 & 64\text{B} \\ 3 & 128\text{B} \end{cases},
   \qquad
   S = 3 .

As a concrete example, ``float16`` (16-bit) gives ``M = bitlen(8) - 1 = 3``, and
paired with a 128B swizzle that becomes
``SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)``. Intuitively, ``M``
keeps a 16-byte (128-bit) contiguous run unswizzled, which matches the minimum
vector access so the swizzle never splits a vectorized load. You attach the result
to a buffer by composing the two layers: ``ComposeLayout(SwizzleLayout(3, 3, 3),
tile)`` is what you pass to the ``layout=`` of a swizzled SMEM allocation.

Bank and Line of an Element
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before we can claim the swizzle removed the conflict, we need a way to translate a
swizzled address back into the bank it actually touches. Since a bank word is 4
bytes and an element occupies ``b = dtype_bytes`` bytes, the swizzled element
address ``a = addr(m)`` lands in

.. math::

   \text{bankword} = \left\lfloor \frac{a \cdot b}{4} \right\rfloor,
   \qquad
   \text{bank} = \text{bankword} \bmod 32,
   \qquad
   \text{line} = \left\lfloor \frac{\text{bankword}}{32} \right\rfloor .

Worked Example: 128B Swizzle, ``float16``, ``(8, 64)`` Tile
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us return to the conflicting tile from earlier and finish the story. Applying
``SwizzleLayout(3, 3, 3)`` over ``m = 64i + j``, the address simplifies to

.. math::

   \text{addr}(i, j) = 64\,i + 8\,(\lfloor j/8 \rfloor \oplus i) + (j \bmod 8),

and with ``b = 2`` we have ``bank = ⌊addr/2⌋ mod 32``. Now read column ``j = 0``
down the 8 rows: the addresses are ``addr = 72 i``, which fall on banks
``0, 4, 8, 12, 16, 20, 24, 28`` — **eight distinct banks, and no conflict**. Compare
that to the unswizzled tile, where the same column gives ``bank = ⌊j/2⌋`` for every
row — a single bank, fully serialized. The swizzle did exactly what we wanted.

The 128B/``float16`` case above is just one instance of a more general guarantee:
with swizzle, a read of any **8×16B column is conflict-free, under any format**
(32B / 64B / 128B). The catch is that this holds only when the element width,
swizzle mode, and access pattern all match the swizzle — the TMA/MMA descriptor mode
the parameters were chosen for — and not for an arbitrary access. The ``B``
parameter is tuned per swizzle width precisely so that the eight 16-byte rows of a
column always land on distinct banks.

You can watch all of this in the interactive demo. Pick a dtype and a swizzle mode
(``none`` / ``32B`` / ``64B`` / ``128B``) in the *Swizzle (SMEM)* control, and the
physical panel switches to a *line × bank* view, where each cell is one 4-byte bank
word holding ``4 / dtype_bytes`` elements side by side. Choose ``none`` and a column
collapses onto a single bank — the conflict, made visible; turn a swizzle on and the
same column scatters across the banks.

Design Rationale
----------------

It is worth closing by naming the three choices that pushed this model toward what
it is, rather than toward some plainer scheme.

- **General shape support.** Non-power-of-two shapes turn out to be everywhere — in
  global tensors, in multi-stage shared-memory buffers, and in capacity-limited
  on-chip scratchpads — so the layout treats general shapes as the default rather
  than bolting them on as a special case.
- **Logical-to-physical mapping.** The map runs from logical coordinates to a *set*
  of physical coordinates, not the other way around. This is what lets replication —
  one logical element living in several physical locations — be expressed cleanly,
  something a physical-to-logical formulation cannot always manage for strided
  patterns.
- **Explicit hardware axes.** Each axis carries its hardware meaning inside the
  layout itself, so an expression is unambiguous without any external context. For
  instance, ``1@tx`` (a block-wide thread id) and ``1@tid_in_wg`` (a thread id
  within a warpgroup) stay distinct, instead of collapsing into a generic ``t``
  whose meaning depends on where it was defined. Legality and feasibility checks,
  meanwhile, are deliberately left to the higher-level tile-primitive layer (Parts
  III–IV) rather than to the layout itself.
