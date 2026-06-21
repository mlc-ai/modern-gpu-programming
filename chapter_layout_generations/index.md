(chap_layout_generations)=
# Data Layout Across GPU Generations

:::{admonition} Overview
:class: overview

- Across Ampere → Hopper → Blackwell the MMA stays the same operation in form (`D = AB + C`); what changes is the shapes, dtypes, and accumulator it supports and how operands must be laid out to reach the Tensor Core.
- Ampere uses a per-lane register fragment, Hopper a SMEM matrix descriptor with swizzle formats, Blackwell SMEM operands plus a TMEM accumulator.
- Two constraints hold every generation: global-memory coalescing and shared-memory bank conflicts.
:::

The tensor core on every recent GPU performs the same kind of operation, the matrix
multiply-accumulate `D = AB + C`. So you might expect a kernel that hits peak throughput on one
generation to carry over to the next. It often does not: the same kernel can run silently slow, or
return silently wrong results, on the following chip. The reason is that the operation's high-level
form stays fixed while *how its operands reach the tensor core* does not, and neither do the shapes,
dtypes, and accumulator each generation supports. What each generation actually demands is a
*specific* operand layout, and a layout the hardware merely tolerates as ordinary memory can still be
the wrong one for the tensor core. This chapter follows that one moving part across **Ampere** →
**Hopper** → **Blackwell**, building on the layout notation from {ref}`chap_data_layout` (`S[...]`,
named axes, swizzle); the Blackwell TMEM specifics are in {ref}`chap_tmem`.

## Two Constraints That Never Went Away

Before any tensor core enters the picture at all, two layout rules already govern how a kernel moves
data, and they hold on every generation we will look at. The first is **global-memory coalescing**.
When the 32 lanes of a warp issue a load, the memory system wants those addresses to fall in one
contiguous, aligned segment, so that it can serve the whole warp in as few transactions as possible;
scatter the addresses around and the same load fragments into many transactions. The second is
**shared-memory bank conflicts**. SMEM is divided into 32 banks, and if several lanes in a warp
happen to address different rows of the *same* bank, the hardware cannot satisfy them at once and the
accesses serialize. The standard remedy here is **swizzle**: we permute the address mapping so that a
warp's lanes land in distinct banks instead of piling onto one.

Both of these are really about *ordinary* memory traffic; they would matter even in a kernel that
never touched a tensor core. What the rest of this chapter adds is a third demand layered on top of
them: the specific layout the *tensor core* itself insists on for its operands.

## Ampere: Register Fragment over Warp/Lane

With those two memory rules in place, we can turn to the third demand, the one the tensor core itself
makes, starting with the earliest of our three generations. On Ampere-class GPUs (`sm_80`) the
tensor-core instruction is the warp-level
`mma.sync.aligned.m16n8k*`, and the defining fact about it is where it reads its operands from:
**registers**. A, B, and the C/D accumulator are all per-thread register fragments, spread across the
warp's 32 lanes. Almost everything else about the Ampere data path follows from this one constraint,
because the data has to be shuffled into registers before the instruction and back out of them
afterward:

```text
SMEM --ldmatrix--> registers --mma.sync--> registers --st.shared--> SMEM
```

### What the Tensor Core Expects: an m8n8 Register Fragment

Because the Ampere operands live in registers, the first thing to pin down is their register layout.
To use the `mma.sync` instruction at all, we have to know precisely which value sits in which lane's register;
the layout is not an implementation detail we can ignore, it is part of the instruction's contract.
The register fragment is built from **8×8 ("m8n8") sub-tiles**, which are the unit that `ldmatrix`
moves and the tensor core reads. Take `mma.m16n8k16` (fp16/bf16 in, fp32 accumulate) as the concrete
case. The 32 lanes are carved **8 along M × 4 along N**, and each lane owns a handful of registers.
All three operands share that M carve, but they differ in what runs across the lanes and the
registers:

- **The C/D accumulator (M×N = 16×8).** Lane `l` holds rows `m ∈ {l/4, l/4 + 8}` and columns
  `n ∈ {2·(l%4), 2·(l%4)+1}`, that is, four fp32 values per lane, namely two 8-row halves crossed
  with two adjacent columns. Four consecutive lanes together cover one row's eight columns.
- **The A operand (M×K = 16×16).** It uses the same M carve as C/D, but now K runs across `l%4` and
  the registers (four b32 registers per lane, each packing two fp16 along K).
- **The B operand (K×N = 16×8).** Its K matches A's, and N is the 8-lane group (two b32 registers
  per lane).

This is exactly the concrete `m16n8k16` C/D fragment hiding behind the named-axes demo in
{ref}`chap_data_layout` (`S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]`), where each lane holds two
adjacent columns per 8×8 and four consecutive lanes cover one row.

### `ldmatrix`: SMEM → the Fragment

`ldmatrix` is the Ampere warp-collective instruction that moves an 8×8 shared-memory tile into the
fixed register fragment the tensor core expects.

The figure below shows that SMEM-to-fragment mapping. The reverse path on Ampere still uses ordinary
stores; `stmatrix` only arrives later, on Hopper and beyond.

![ldmatrix loads an 8x8 SMEM tile into the warp register fragment; stmatrix, the reverse, is a Hopper (sm_90+) instruction](../img/ldstmatrix.svg)

The job of `ldmatrix` is to get the tile out of SMEM and into that fragment. A single
`ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16` loads one, two, or four 8×8 16-bit matrices
from SMEM into the fragment, and it does so as one warp-collective instruction. Three details are
what make it line up with the MMA:

- **The addresses come from the lanes themselves.** Each source row's base address is supplied by one
  lane: matrix `m`, row `r` is addressed by lane `m·8 + r`. So `.x1` draws its row addresses from
  lanes 0–7, `.x2` from lanes 0–15, and `.x4` from all of lanes 0–31.
- **The result lands already distributed** the way the MMA wants it, so lane `l` ends up holding
  row `l/4` and columns `2·(l%4)` and `2·(l%4)+1` (one b32 packing the two adjacent fp16), precisely
  the fragment the MMA reads.
- **The optional `.trans` qualifier** transposes each 8×8 as it loads, so the two halves map *down a
  column* instead of across a row. This is how we feed the tensor core an operand that happens to be
  stored the opposite way from what the MMA expects.

A plain per-lane `ld.shared` loop simply cannot produce the MMA's scattered fragment cheaply, whereas
this one `ldmatrix` performs the entire SMEM→register shuffle the tensor core demands.

### Writing the Fragment Back

Once the MMA finishes, its result sits scattered across the lanes in the C/D layout we described
above, and we still have to get it out. On Ampere we write it back with ordinary per-thread
`st.shared` stores (optionally with a warp shuffle first to regather it) into a SMEM tile, from
which a coalesced `st.global` can finally reach GMEM. There is no dedicated reverse of `ldmatrix` on
Ampere: the `stmatrix` instruction, which gathers the register fragment straight back into SMEM, does
not exist on `sm_80` and only arrives with Hopper (`sm_90+`). So the Ampere story is symmetric and
simple: the register fragment is fixed by the hardware, `ldmatrix` bridges SMEM into it on the way in,
and plain stores bridge it back out on the way out.

### Swizzle: the Same Conflict, Already on Ampere

Ampere kernels already needed swizzle, and the reason is the conflicting access pattern we have just
set up. The same SMEM tile is *written* one way and *read* another: it is filled coalesced from GMEM
along a **row**, and then read by `ldmatrix` along a **column**. With a plain row-major tile the row
write hits 8 distinct banks and is conflict-free, but the column read hits one bank 8 times, an
8-way conflict. Switching to a col-major tile does not save us; it merely flips which of the two
accesses pays the price. Without a permutation, one side always loses: row-major favors the write,
while col-major favors the read. The figure below contrasts the two accesses on a plain row-major
tile, showing the row write spread across distinct banks while the column read collides on one.

![Row write hits 8 distinct banks (conflict-free); column read hits one bank 8 times (conflict)](../img/swizzle_conflict.svg)

The way out is the XOR **swizzle** from {ref}`chap_data_layout`: we store element `(r, c)` at column
`c ⊕ r`, and that single permutation makes the row write *and* the column read conflict-free at the
same time. As we will see in the next section, Hopper later folds this exact permutation into the TMA
and MMA descriptors so that nobody has to spell it out by hand; on Ampere, though, it still had to
live in hand-written index math.

## Hopper: `wgmma`, SMEM Descriptors, and Swizzle Formats

Ampere paid for its register fragment on every MMA. The next generation keeps the same `D = AB + C`
but rethinks where the operands come from, and that one decision reshapes the whole input path.

### What the Tensor Core Expects: a SMEM Matrix Descriptor

Recall that the Ampere data path spends real instructions shuffling operands through registers.
Hopper (`sm_90`) removes that cost on the *input* side. Its `wgmma` instruction reads its operands
**straight from SMEM**, with no `ldmatrix` in between: the **B** operand always comes from a SMEM
matrix descriptor, while the **A** operand can come from either a SMEM descriptor or a register
fragment; these are the `.ss` and `.rs` forms of the instruction. For a SMEM-sourced operand the
tensor core does not read just any SMEM, however. It reads through a 64-bit **matrix descriptor**,
which fixes the one format the operand is allowed to be stored in. The descriptor is what turns an
index `(m, k)` into an actual SMEM address, and it carries five fields:

| Field | Meaning |
|---|---|
| **start_address** | base of the tile in SMEM, 16-byte-aligned (stored as `addr ≫ 4`) |
| **swizzle** | the swizzle format, which sets the **atom shape** (8 × 128/64/32/16 B) and the XOR pattern inside it |
| **ldo** (leading byte offset) | stride to the next atom along the **major** dim |
| **sdo** (stride byte offset) | stride to the next atom along the **other** dim |
| **matrix base offset** | small offset (often 0) applied before swizzling, to align the tile's start within a swizzle atom |

With these fields in hand, the hardware views A(M×K) as a 2-D grid of **atoms**, and each field plays
its own part in resolving an index `(m, k)`. The swizzle format sets two things at once: the shape of
each atom (8 × 128 B for `SWIZZLE_128B`, and 8 × 64 / 32 / 16 B for the smaller modes) and how its
bytes are XOR-permuted inside, which is the very same swizzle from the Ampere section and is what
keeps the `wgmma` read free of bank conflicts. The two strides **ldo** and **sdo** then tell the
hardware how far to step *between* atoms, and which axis each one walks depends on the operand's
major-ness: **ldo** strides along the **major** dimension and **sdo** along the **other** one. For a
K-major tile (A stored K-contiguous), that puts `ldo` along K and `sdo` down M; an MN-major tile
simply swaps the two. So to resolve `A[m, k]` the hardware combines the two strides to land on the
right atom, and then applies the swizzle to find the exact byte inside it:

![A SMEM matrix descriptor (start_address, ldo, sdo, swizzle, base offset) tiles A(M×K) into 8×N B swizzle atoms, with ldo/sdo the strides between atoms](../img/smem_descriptor.svg)

It is worth being clear about what this buys us, because it relocates the kernel's job rather than
removing it. The kernel still has to write A into SMEM in exactly this atom-tiled, swizzled format
(the TMA load is what does that), and it still has to hand `wgmma` a descriptor whose `ldo`, `sdo`,
and `swizzle` match the data it wrote (in the kernels these end up as literal constants, for example
`ldo = 1`, `sdo = 64`, `swizzle = 64B`). What is new is that swizzle is now a first-class format on
Hopper (`SWIZZLE_NONE / 32B / 64B / 128B`), and the *same* format is named in both the TMA
descriptor that fills the tile and the `wgmma` descriptor that reads it. Because both sides quote the
same format, the load and the MMA agree by construction, instead of relying on the hand-written index
math that carried this on Ampere.

The element arrangement inside a single atom, for each of the formats (`SWIZZLE_128B` = 8 × 128 B,
`SWIZZLE_64B`, `SWIZZLE_32B`), is precisely the swizzle-atom demo in {ref}`chap_data_layout`.

The **output** side, by contrast, has not moved. The accumulator `D` of `wgmma` is still a per-thread
**register** fragment, built from the same 8×8 sub-tiles we met on Ampere, although now its register
count and exact lane mapping scale with the instruction's **N** (a `wgmma` is shaped `m64nNk16`), so
it is no longer a single fixed m8n8 tile. The upshot is that a Hopper GEMM reads its operands the new
way but still keeps the accumulator in registers and runs a register epilogue, just as on Ampere.
Moving the accumulator out of registers altogether is the change that waits for Blackwell's TMEM.

## Blackwell: `tcgen05` and TMEM

Hopper moved the operands into SMEM but left the accumulator in registers. The third generation
finishes the move, relocating the accumulator out of the register file and introducing a new operand
that has no place to live anywhere else.

### What the Tensor Core Expects: SMEM Operands and a TMEM Accumulator

Blackwell (`sm_100`) inherits Hopper's SMEM matrix descriptor for the A/B operands (and an A operand
may additionally be read from TMEM), so the input side will feel largely familiar. The real change is
on the output side: the **accumulator** now moves into TMEM. Unlike an Ampere `mma` accumulator,
which lives in a register fragment throughout, the Blackwell accumulator never visits registers during
the compute phase at all; it stays in TMEM until the epilogue reads it out. We leave the question of
how the (M, N) accumulator and the A/B operands split across one or two CTAs (`cta_group::1` vs
`cta_group::2`) to {ref}`chap_tensor_cores`. The layout that is genuinely new at this generation, and
the one we will focus on here, is the **scale factors** of a block-scaled MMA.

### Scale-Factor Layout in TMEM

The new operand promised above is the per-block scale. A block-scaled MMA (mxfp8, nvfp4) carries two
extra operands beyond A and B: `SFA (M, SFK)` and
`SFB (N, SFK)`, where `SFK = K / block`. The thing that sets them apart is where they live: unlike A
and B, **the scale factors reside in TMEM** rather than SMEM, so they cannot simply ride the ordinary
operand path. Instead they take a small SMEM→TMEM detour: a TMA load first brings them into SMEM, and
then `tcgen05.cp` copies them on into TMEM before the MMA runs.

Once they are in TMEM, their layout, the PTX *tcgen05 MMA scale-factor A layout*, turns out to be
exactly the lane-replication example from {ref}`chap_data_layout`. A 128-row scale vector packs into
32 lanes (row `r` goes to lane `r % 32`, with `r // 32` running along TMEM columns at stride
`epc = 4`) and is then broadcast `warpx4` to all 128 reading lanes (`R[4 : 32@TLane]`).

The one piece here that has no Ampere or Hopper analogue is the byte packing, that is, how many
distinct scales a single `uint32` column holds. This is set by the **scale_vec** mode, and it matches
the PTX *scale-factor A* 1x/2x/4x layouts:

![scale_vec byte packing: 1X (fp8) broadcasts one scale across 4 bytes; 2X (mxfp4) packs two scales each duplicated; 4X (nvfp4) packs four K-block scales](../img/sf_scale_vec.svg)

When the MMA spans two CTAs (`cta_group::2`), the scale factors split exactly the way their data
does: **SFA follows A**, so each CTA holds the M-half that matches its A rows, while **SFB is
multicast** to both CTAs ({ref}`chap_tensor_cores`).

For all that changes from one generation to the next, one structure quietly recurs in
every one of them: the **m8n8 register fragment**. It is what `ldmatrix` builds on Ampere, what
`wgmma` outputs on Hopper, and what `tcgen05.ld` reads TMEM into on Blackwell ({ref}`chap_tmem`): a
single register layout that survives every change to the surrounding hardware.

## The Throughline

Stepping back, there is a clear direction to the three generations: more and more of the layout gets
**described to the hardware** through descriptors, instead of being open-coded with shuffle
instructions:

| Generation | Operands read from | Layout described by |
|---|---|---|
| Ampere (`sm_80`) | registers | `ldmatrix` + hand-staged (swizzled) SMEM stores |
| Hopper (`sm_90`) | SMEM | `wgmma` matrix descriptor + TMA box & swizzle |
| Blackwell (`sm_100`) | SMEM / TMEM | `tcgen05` matrix descriptor + TMEM accumulator & scale-factor layouts |

It bears repeating that the descriptors do not remove the work; they relocate it. The kernel still
has to place the bytes in the exact format the engine reads, which means a TMA load, the matrix
descriptor it feeds, and the MMA that consumes it must all agree on one and the same swizzle. Let any
one of them fall out of step with the others and the tensor core will happily read scrambled data.
