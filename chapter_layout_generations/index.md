(chap_layout_generations)=
# Data Layout Through GPU Generations

Why organize a chapter by GPU generation at all? Because the thing that actually changed from
**Ampere** to **Hopper** to **Blackwell** is not the math the tensor core does — it is *how operands
reach the tensor core*. Each generation's memory and compute engines demand a *specific* operand
layout, and getting it wrong is silently slow, or silently wrong. So we trace that one moving part
across the three generations, building on the layout notation from {ref}`chap_data_layout`
(`S[...]`, named axes, swizzle). The Blackwell TMEM specifics are in {ref}`chap_tmem`.

## Two Constraints That Never Went Away

Before any tensor core enters the picture, two layout rules already hold on every generation, and
they will frame everything that follows. The first is **global-memory coalescing**: the 32 lanes of
a warp should address one contiguous, aligned segment so the memory system can serve them in as few
transactions as possible. The second is **shared-memory bank conflicts**: SMEM is divided into 32
banks, and if several lanes in a warp address different rows of the *same* bank, those accesses
serialize. The fix for the second is **swizzle** — permuting the address mapping so a warp's lanes
spread across distinct banks.

Both of these are about *ordinary* memory traffic. What the rest of this chapter adds is a third
demand layered on top of them: the layout the *tensor core* itself requires for its operands.

## Ampere — Register Fragment over warp/lane

On Ampere-class GPUs (`sm_80`) the tensor-core instruction is the warp-level
`mma.sync.aligned.m16n8k*`, and the decisive fact about it is where it gets its data: it reads its
operands from **registers**. A, B, and the C/D accumulator are all per-thread register fragments
spread across the warp's 32 lanes. Everything else on Ampere follows from this one constraint,
because the data has to be shuffled into and out of registers around the instruction:

```text
SMEM --ldmatrix--> registers --mma.sync--> registers --stmatrix--> SMEM
```

### What the Tensor Core expects: an m8n8 register fragment

To use the instruction we have to know precisely which value sits in which lane's register, so let
us pin that down. The register fragment is built from **8×8 ("m8n8") sub-tiles** — the unit
`ldmatrix` moves and the tensor core reads. For `mma.m16n8k16` (fp16/bf16 in, fp32 accumulate), the
32 lanes are carved **8 along M × 4 along N**, and each lane owns a few registers. The three operands
share that M carve but differ in what runs across the lanes and registers:

- **C/D accumulator (M×N = 16×8):** lane `l` holds rows `m ∈ {l/4, l/4 + 8}` and columns
  `n ∈ {2·(l%4), 2·(l%4)+1}` — four fp32 values per lane (two 8-row halves × two adjacent columns).
  Four consecutive lanes cover one row's eight columns.
- **A operand (M×K = 16×16):** same M carve as C/D; K runs across `l%4` and the registers — four
  b32 registers per lane, each packing two fp16 along K.
- **B operand (K×N = 16×8):** K matches A; N is the 8-lane group — two b32 registers per lane.

This is not an abstract layout: it is the concrete `m16n8k16` C/D fragment behind the named-axes
demo in {ref}`chap_data_layout` (`S[(8, 4, 2) : (4@laneid, 1@laneid, 1@reg)]`), where each lane
holds two adjacent columns per 8×8 and four consecutive lanes cover one row.

### `ldmatrix`: SMEM → the fragment

![ldmatrix loads an 8x8 SMEM tile into the warp register fragment; stmatrix is the reverse](../img/ldstmatrix.svg)

Now we need to get the tile from SMEM into that fragment, and this is the job `ldmatrix` exists for.
`ldmatrix.sync.aligned.m8n8.x{1,2,4}[.trans].shared.b16` loads one, two, or four 8×8 16-bit matrices
from SMEM into the fragment in a single warp-collective instruction. Three details make it work:

- **Addresses come from lanes.** Each source row's base address is supplied by one lane: matrix
  `m`, row `r` is addressed by lane `m·8 + r`. So `.x1` uses lanes 0–7, `.x2` lanes 0–15, and `.x4`
  lanes 0–31 as the row-address suppliers.
- **The result is distributed** so lane `l` ends up holding row `l/4`, columns `2·(l%4)` and
  `2·(l%4)+1` (one b32 packing the two adjacent fp16) — exactly the fragment the MMA reads.
- **`.trans`** transposes each 8×8 as it loads (the two halves map *down a column* instead of
  across a row), which is how you feed an operand stored the opposite way from what the MMA wants.

Putting those together explains why the instruction is in the ISA at all: a plain per-lane
`ld.shared` loop cannot cheaply produce the MMA's scattered fragment, but a single `ldmatrix`
performs the whole SMEM→register shuffle the tensor core demands.

### `stmatrix`: the fragment → SMEM

The output side has the mirror-image problem, and `stmatrix` solves it: it is the reverse —
register→SMEM, with the same lane/address mapping. After the MMA, the accumulator is scattered across
lanes in the C/D layout above, and `stmatrix` gathers it back into a SMEM tile, from which a
coalesced `st.global` (or, later, a TMA store) can reach GMEM. So the Ampere story is this same
shuffle running in both directions: the register fragment is fixed by hardware, and ldmatrix /
stmatrix are the two bridges between it and SMEM.

### Swizzle: the same conflict, already on Ampere

It is tempting to think of swizzle as a Hopper feature, but Ampere kernels already needed it, and the
reason is the conflicting access pattern we just set up. The SMEM tile is *written* one way (coalesced
from GMEM, along a **row**) and *read* another way (by `ldmatrix`, along a **column**). With a plain
row-major tile the row write hits 8 distinct banks (conflict-free), but the column read hits one bank
8 times — an 8-way conflict. Switching to col-major just flips which access pays the price, so no
unpermuted layout can satisfy both:

![Row write hits 8 distinct banks (conflict-free); column read hits one bank 8 times (conflict)](../img/swizzle_conflict.svg)

The way out is the XOR **swizzle** from {ref}`chap_data_layout`: store `(r, c)` at column `c ⊕ r`,
which makes the row write *and* the column read conflict-free at once. Hopper later folds this exact
permutation into the TMA and MMA descriptors (next section); on Ampere it had to live in hand-written
index math.

## Hopper — `wgmma`, SMEM Descriptors, and Swizzle Formats

### What the Tensor Core expects: a SMEM matrix descriptor

The Ampere data path spends real instructions shuffling operands through registers, and Hopper
(`sm_90`) attacks exactly that cost on the *input* side: `wgmma` reads its A and B operands
**directly from SMEM**, with no `ldmatrix` in between. That does not mean the Tensor Core reads
arbitrary SMEM, though. It reads through a 64-bit **matrix descriptor** that pins down the one format
the operand may be stored in. Think of the descriptor as the answer to "where does `A[m, k]` live?" —
it is the recipe that turns an index `(m, k)` into a SMEM address, and it has four parts:

| Field | Meaning |
|---|---|
| **start_address** | base of the tile in SMEM, 16-byte-aligned (stored as `addr ≫ 4`) |
| **swizzle** | the swizzle format — sets the **atom shape** (8 × 128/64/32/16 B) and the XOR pattern inside it |
| **ldo** — leading byte offset | stride to the next atom along the **major** dim |
| **sdo** — stride byte offset | stride to the next atom along the **other** dim |

With those four fields in hand, A(M×K) is laid out as a 2-D grid of **atoms**, and each field plays a
distinct role in resolving `(m, k)`. The swizzle format sets each atom's shape — 8 × 128 B for
`SWIZZLE_128B` (8 × 64 / 32 / 16 B for the smaller modes) — and how its bytes are XOR-permuted inside
(the Ampere section's swizzle), which is what keeps the `wgmma` read bank-conflict-free. **ldo** and
**sdo** are the byte strides *between* atoms, and which axis each walks depends on the operand's
major-ness: **ldo** strides along the **major** dimension and **sdo** along the **other**. For a
K-major tile (A stored K-contiguous) that puts `ldo` along K and `sdo` down M; an MN-major tile swaps
them. To resolve `A[m, k]`, the hardware combines the two strides to find the right atom and then the
swizzle to find the byte inside it:

![A SMEM matrix descriptor (start_address, ldo, sdo, swizzle) tiles A(M×K) into 8×N B swizzle atoms, with ldo/sdo the strides between atoms](../img/smem_descriptor.svg)

This reframes the kernel's job rather than removing it. The kernel must write A into SMEM in exactly
this atom-tiled, swizzled format — the TMA load does that — and hand `wgmma` a descriptor whose `ldo`
/ `sdo` / `swizzle` match (in the kernels these are literal constants, e.g. `ldo = 1`, `sdo = 64`,
`swizzle = 64B`). What makes this robust is that swizzle is a first-class format on Hopper —
`SWIZZLE_NONE / 32B / 64B / 128B` — and the *same* format is named in both the TMA descriptor that
fills the tile and the `wgmma` descriptor that reads it, so the load and the MMA agree by
construction. (On Ampere that same permutation lived in hand-written index math.)

The element arrangement inside one atom for each format (`SWIZZLE_128B` = 8 × 128 B, `SWIZZLE_64B`,
`SWIZZLE_32B`) is the swizzle-atom demo in {ref}`chap_data_layout`.

So much for the input side; the **output** side, by contrast, has not moved at all. `wgmma`'s
accumulator `D` is still a per-thread **register** fragment in the same m8n8 layout as Ampere
(above). A Hopper GEMM therefore reads its operands the new way but writes its accumulator and runs
its epilogue exactly as before. Moving the accumulator out of registers is the change that waits for
Blackwell's TMEM.

## Blackwell — `tcgen05` and TMEM

### What the Tensor Core expects: SMEM operands and a TMEM accumulator

Blackwell (`sm_100`) inherits Hopper's SMEM matrix descriptor for the A/B operands (an A operand may
also be read from TMEM), so the input side is largely familiar. The change is on the output side: the
**accumulator** moves into TMEM. It never visits a register fragment during the compute phase the way
an Ampere `mma` accumulator does — it stays in TMEM until the epilogue reads it out, which is the
whole point of the new memory space. How the (M, N) accumulator and the A/B operands split across one
or two CTAs (`cta_group::1` vs `cta_group::2`) is covered in {ref}`chap_tensor_cores`. The layout
that is genuinely new at this generation, and that we focus on here, is the **scale factors** of a
block-scaled MMA.

### Scale-factor layout in TMEM

A block-scaled MMA (mxfp8, nvfp4) carries two operands beyond A and B — `SFA (M, SFK)` and
`SFB (N, SFK)`, where `SFK = K / block`. The wrinkle is that, unlike A and B, **the scale factors
live in TMEM**, not SMEM, so they cannot ride the ordinary operand path. Instead they take a
SMEM→TMEM detour: a TMA load brings them into SMEM, and then `tcgen05.cp` copies them into TMEM
before the MMA.

Once there, their TMEM layout (the PTX *tcgen05 MMA scale-factor A layout*) is exactly the
lane-replication example from {ref}`chap_data_layout`: a 128-row scale vector packs into 32 lanes
(row → lane `r % 32`, `r // 32` along TMEM columns at stride `epc = 4`) and is broadcast `warpx4`
to all 128 reading lanes (`R[4 : 32@TLane]`).

The piece that has no Ampere or Hopper analogue is the byte packing — how many distinct scales a
single `uint32` column holds. That depends on the **scale_vec** mode, matching the PTX
*scale-factor A* 1x/2x/4x layouts:

![scale_vec byte packing: 1X (fp8) broadcasts one scale across 4 bytes; 2X (mxfp4) packs two scales each duplicated; 4X (nvfp4) packs four K-block scales](../img/sf_scale_vec.svg)

When the MMA spans two CTAs (`cta_group::2`), the scale factors split the way their data does:
**SFA follows A**, with each CTA holding the M-half matching its A rows, while **SFB is multicast** to
both CTAs ({ref}`chap_tensor_cores`).

Step back across all three generations and one structure keeps reappearing: the **m8n8 register
fragment**. It is what `ldmatrix` builds on Ampere, what `wgmma` outputs on Hopper, and what
`tcgen05.ld` reads TMEM into on Blackwell ({ref}`chap_tmem`) — a single register layout that survives
every change to the surrounding machinery.

## The Throughline

Reading the three generations in sequence, one trend stands out: progressively more of the layout is
**described to hardware** through descriptors, instead of being open-coded with shuffle instructions:

| Generation | Operands read from | Layout described by |
|---|---|---|
| Ampere (`sm_80`) | registers | `ldmatrix`/`stmatrix` + hand-staged (swizzled) SMEM |
| Hopper (`sm_90`) | SMEM | `wgmma` matrix descriptor + TMA box & swizzle |
| Blackwell (`sm_100`) | SMEM / TMEM | `tcgen05` matrix descriptor + TMEM accumulator & scale-factor layouts |

The lesson to carry forward is that the descriptors do not remove the work — they relocate it. The
kernel still has to place bytes in the exact format the engine reads: a TMA load, the matrix
descriptor it feeds, and the MMA that consumes it must all agree on the same swizzle. Get any one of
them out of step and the tensor core reads scrambled data.
