# TIRX Tile Primitives
:label:`chap_layouts`

:numref:`chap_tirx_primer` gave you the *language layer*: buffers, scopes, raw PTX intrinsics, the compile pipeline, and metaprogramming. In principle that is already enough to write a Blackwell GEMM — the hardware exposes TMA, `tcgen05.mma`, and `mbarrier` instructions, and all three are directly callable from TIRX as `Tx.ptx.*`. This chapter opens by doing exactly that, and then uses the resulting code to motivate the three additional interfaces that turn TIRX from a thin PTX wrapper into a compiler: **execution scope**, **tensor layout**, and **tile primitive dispatch**.

After this chapter, every `Tx.copy`, `Tx.gemm_async`, `TMABar`, and `layout=` expression you meet in the rest of the tutorial will feel like a direct substitution for a chunk of raw PTX — because that is precisely what it is.


## A No-Sugar GEMM
:label:`sec_no_sugar_gemm`

Our starting point is a *single-tile* GEMM written entirely with `Tx.ptx.*` intrinsics. One CTA, one warpgroup, no software pipelining, no cluster. Shapes are fixed at `M = N = 128, K = 64`, fp16 $\times$ fp16 $\to$ fp32, so the entire problem fits in one `tcgen05.mma` accumulator. The full listing lives at `tirx_tutorial/no_sugar_gemm.py`; we walk through it piece by piece.

### Host-side TMA descriptors

TMA does not take a raw `float*` — it takes a *tensor-map descriptor*, a 128-byte structure encoding global shape, stride, tile shape, swizzle mode, and out-of-bounds fill. The descriptor is built on the host and passed to the kernel as an extra argument. In TIRX the host-side call is made via `Tx.call_packed`:

```python
A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
Tx.call_packed(
    "runtime.cuTensorMapEncodeTiled",
    A_map, a_type, 2, A.data,
    K, M,          # global_shape  (innermost first)
    K * 2,         # global_stride (rank-1, in bytes)
    K, M,          # box_dim       (innermost first)
    1, 1,          # element_strides
    0, SWIZZLE, 2, 0,
)
```

Every field has to be correct: innermost-first, stride in bytes not elements, swizzle mode matching the SMEM layout used inside the kernel. A typo silently corrupts the load.

### SMEM buffers with an explicit swizzle layout

The destination of each TMA is an SMEM tile whose physical layout must match the descriptor. Blackwell SMEM has 32 banks of 4 bytes each; a naive row-major `(128, 64)` fp16 tile would have *all* 32 lanes of a warp hit the same bank on a column read. To avoid that, the `K`-inner direction is wrapped in a 128-byte swizzle pattern:

```python
A_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(M, 1, 64) : (64, M * 64, 1)]),
)
A_smem = Tx.alloc_buffer((M, K), a_type, scope="shared", layout=A_layout, align=128)
```

The three `3`s in `SwizzleLayout(3, 3, 3, ...)` index into a table of bit-fiddling recipes; picking one by hand is exactly as fun as it sounds.

### Thread hierarchy, TMEM allocation, and mbarrier init

Inside the kernel we declare the thread hierarchy explicitly and then reach for three hardware primitives:

```python
with Tx.kernel():
    Tx.cta_id([1])
    warp_id = Tx.warp_id([4])
    Tx.lane_id([32])
    tx      = Tx.thread_id([128])
    with Tx.cta():
        tmem_addr = Tx.shared_scalar("uint32")  # result of tcgen05.alloc
        bar_tma   = Tx.shared_scalar("uint64")  # mbarrier for TMA completion
        bar_mma   = Tx.shared_scalar("uint64")  # mbarrier for MMA completion

        with Tx.warp(Tx.filter(warp_id, 0, 1)):
            Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr),
                                 n_cols=N_COLS, cta_group=1)

        with Tx.thread():
            if tx == 0:
                Tx.ptx.mbarrier.init(Tx.address_of(bar_tma), 1)
                Tx.ptx.mbarrier.init(Tx.address_of(bar_mma), 1)
            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
```

Three separate gotchas are hiding here:

1. `tcgen05.alloc` must be issued by exactly one elected warp (`with Tx.warp(Tx.filter(warp_id, 0, 1))`) and the allocated address is **written into SMEM**, not returned. We have to reserve `tmem_addr` ahead of time as a `Tx.shared_scalar` so that every thread can read it later.
2. The two mbarriers must live at distinct SMEM addresses, so we allocate them as two separate `shared_scalar`s rather than slicing a shared buffer (which would share a base pointer).
3. The fence pair after `mbarrier.init` is required by the PTX model; forgetting it produces "works on my machine" kernels that break on the next driver update.

### Raw TMA load

```python
if tx == 0:
    Tx.ptx.cp_async.bulk.tensor.g2c(
        2, A_smem.data, Tx.address_of(bar_tma), Tx.address_of(A_map),
        0, 1, "", 0, 0,            # cta_mask, cta_group, cache_hint, coords...
    )
    Tx.ptx.cp_async.bulk.tensor.g2c(
        2, B_smem.data, Tx.address_of(bar_tma), Tx.address_of(B_map),
        0, 1, "", 0, 0,
    )
    Tx.ptx.mbarrier.arrive.expect_tx(
        Tx.address_of(bar_tma), TMA_TOTAL_BYTES,
    )
Tx.ptx.mbarrier.try_wait(Tx.address_of(bar_tma), tma_phase[0])
tma_phase[0] = tma_phase[0] ^ 1
```

Four operations that must agree on an invisible contract:

- Only one thread issues the copy (`if tx == 0`).
- We must manually sum `A_BYTES + B_BYTES` and pass it to `expect_tx`; one wrong byte and `try_wait` never returns.
- The phase bit has to be flipped by the user after every wait.
- `try_wait` is a suspend-and-resume: the compiler does not track which barrier is which — get the pointer wrong and you will wait on the next reuse.

### Raw `tcgen05.mma` loop

```python
if tx == 0:
    Tx.ptx.tcgen05.encode_instr_descriptor(
        descI.data, d_dtype="float32", a_dtype="float16", b_dtype="float16",
        M=128, N=128, K=16, trans_a=False, trans_b=False, n_cta_groups=1,
    )
    for k in range(K // MMA_K):
        Tx.ptx.tcgen05.encode_matrix_descriptor(descA.data, ..., ldo=1, sdo=64, swizzle=3)
        Tx.ptx.tcgen05.encode_matrix_descriptor(descB.data, ..., ldo=1, sdo=64, swizzle=3)
        Tx.ptx.tcgen05.mma(
            tmem_addr, descA[0], descB[0], descI[0],
            d_dtype="float32", a_dtype="float16", b_dtype="float16",
            use_a_tmem=False, cta_group=1,
            enable_input_d=(k != 0),  # False on first step: D = A*B, not D += A*B
        )
    Tx.ptx.tcgen05.commit(Tx.address_of(bar_mma), 1)
Tx.ptx.mbarrier.try_wait(Tx.address_of(bar_mma), mma_phase[0])
```

A single `tcgen05.mma` is one PTX instruction, but driving it takes *three* 64-bit bitfield descriptors:

- The *instruction* descriptor encodes $M$, $N$, $K$, $A$/$B$ transpose flags, and the CTA-group bit.
- The *matrix* descriptor for each operand encodes the SMEM base address, the leading-dimension offset `ldo`, the stride-dimension offset `sdo`, and the swizzle mode.
- The `ldo`, `sdo`, and `swizzle` fields **must** match the layout chosen for the SMEM allocation above; any disagreement produces scrambled reads that show up as `NaN`s in the output.

### TMEM → RF, element by element

```python
Tx.ptx.tcgen05.fence.after_thread_sync()
for i in range(N):
    Tx.ptx.tcgen05.ld(
        tmem_addr, reg[i],
        shape="32x32b", num=1, row=warp_id * 32, col=i,
    )
Tx.ptx.tcgen05.wait.ld()
for i in range(N):
    C[tx, i] = reg[i]
```

One `tcgen05.ld` loads a $32 \times 32$-bit slice (one column, 32 lanes) into a single register. A 128-column tile therefore takes 128 separate intrinsic calls in which we manually compute `row = warp_id * 32` and `col = i`. The warpgroup-cooperative nature of the instruction — each of the 128 lanes pulls a different TMEM row — is *not* expressed in the Python; it is left to the reader to trust.

### What this costs us

The kernel above is 130 lines of Python and compiles to ~420 lines of CUDA. It is not pleasant to maintain, but more importantly it is **fragile for reasons the compiler could have caught automatically**:

| Pain point | Root cause |
|:--|:--|
| Hand-built TMA descriptors | nothing in the DSL knows the global shape / stride of the tensor being copied |
| Hand-built matrix descriptors | the swizzle mode of `A_smem` is declared once in the allocator, then repeated by hand in the `ldo`/`sdo`/`swizzle` arguments to `encode_matrix_descriptor` |
| mbarrier state machine by hand | the user tracks phase bits, expected byte counts, and which barrier belongs to which producer |
| Element-by-element TMEM read | the compiler knows 128 warpgroup threads should pair with 128 `TLane`s, but we still spell out 128 intrinsic calls |
| Distinct SMEM scalars for mbarriers | the pool allocator is deliberately left out — see *Buffers and Scopes* in :numref:`chap_tirx_primer` |

Each pain point corresponds to a piece of information that *already exists somewhere* — in the global tensor shape, in the SMEM layout declaration, in the thread hierarchy — but has to be re-typed because no compiler is listening. The rest of this chapter introduces three declarative interfaces whose only job is to surface that information so the compiler can do the bookkeeping for us.


## Execution Scope
:label:`sec_exec_scope`

Every instruction issued by a GPU kernel has a *team kind*: the granularity of the thread group cooperating on that instruction. Blackwell recognises four levels and TIRX surfaces one `with`-block per level:

```python
with Tx.kernel():             # κ = kernel    (grid-wide setup)
    with Tx.cluster():        # κ = cluster   (grid → cluster of CTAs; optional)
        with Tx.cta():        # κ = cta       (one thread block — launch-sized)
            with Tx.warpgroup():  # κ = warpgroup (128 threads; TMEM reads live here)
                with Tx.warp():   # κ = warp      (32 threads)
                    with Tx.thread():  # κ = thread (one thread acts independently)
                        ...
```

A `with`-block does not change which threads are physically running — the grid launch configuration already decided that. What it changes is *how the compiler lowers whatever primitive appears next*. A `Tx.copy(Asmem, Agmem)` under `with Tx.cta():` becomes "all threads of the CTA cooperate on the copy"; the exact same call under `with Tx.thread():` becomes "each thread copies its own element".

### Narrowing the active thread set

A scope can be restricted with a predicate argument. Two forms are supported: a structural predicate (an ordinary boolean expression) or a `Tx.filter(coord, ...)` selector when the predicate needs an opaque PTX result like `elect_sync`:

```python
with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):  # one thread per warp
    Tx.copy_async(Asmem, Agmem, dispatch="tma")            # TMA issued from that thread

with Tx.warp(Tx.filter(warp_id, 0, 1)):                    # first warp of the CTA
    Tx.ptx.tcgen05.alloc(...)                              # TMEM alloc instruction

with Tx.thread(tid_in_wg == 0):                            # first thread of this warpgroup
    Tx.ptx.mbarrier.init(Tx.address_of(bar), 1)             # raw mbarrier.init(ptr, count)
```

`Tx.ptx.elect_sync()` is the most common filter: it returns a mask with exactly one thread selected per warp, which is the granularity TMA and `tcgen05.commit` demand. You saw the ad-hoc version of this in the no-sugar GEMM as `if tx == 0:` — a correct predicate, but one that guards individual intrinsic calls instead of a full block of code.

The `Tx.filter(var, ...)` form has two shapes: a range, `Tx.filter(var, lo, hi)`, and a predicate, `Tx.filter(var, cond_expr)`. The first argument is always a coordinate variable produced by `Tx.cta_id` / `Tx.warp_id` / `Tx.lane_id` / etc., so the compiler knows which axis is being narrowed.

### Symbolic coordinates

Inside a scope, threads usually need to know *which* of the cooperating threads they are. TIRX provides one symbolic id per level:

| Coordinate | Parent → child | Extent |
|:--|:--|:--|
| `Tx.cluster_id([...])` | kernel → cluster | cluster grid dims |
| `Tx.cta_id([Gx, Gy, ...])` | kernel → cta | `[Gx, Gy, ...]` — the grid launch |
| `Tx.cta_id_in_cluster([Cx, Cy])` | cluster → cta (clustered launch only) | one entry per cluster axis |
| `Tx.warpgroup_id([Nw])` | cta → warpgroup | 1 or 2 for the kernels in this tutorial |
| `Tx.warp_id([Nw])` | cta → warp | total warps in the CTA |
| `Tx.warp_id_in_wg([4])` | warpgroup → warp | always 4 |
| `Tx.thread_id([Nt])` | cta → thread | the launch's CTA size |
| `Tx.thread_id_in_wg([128])` | warpgroup → thread | 128 |
| `Tx.lane_id([32])` | warp → thread (laneid) | 32 |

Each call both declares the extent along that level *and* returns a symbolic variable that the compiler can reason about — it is not just a fancy `threadIdx.x` read. There is no `parent=...` keyword in the new API: the parent level is encoded in the function name (`*_in_cluster`, `*_in_wg`, `lane_id`).

### Why this is a pillar

Scope is what decides which lowering strategy is legal. The most common cases:

- `Tx.ptx.tcgen05.commit(bar)` is hard-coded to a single elected thread. Calling it under `with Tx.cta():` (no filter) is a compile-time error.
- `Tx.copy_async(reg, tmem)` lowers to a warpgroup-cooperative `tcgen05.ld`. Calling it from `with Tx.thread():` is rejected (wrong team kind); `Tx.copy` does not have a TMEM↔RF variant — `tcgen05.ld` is intrinsically async and lives under `copy_async`.
- `Tx.cuda.cta_sync()` is always a CTA-wide barrier; the compiler checks that the surrounding active thread set covers the whole CTA.

In other words, the execution scope is the *permission system* of the tile primitive dispatch. Before looking at what a primitive dispatches *into*, we need the second half of the permission system: tensor layout.


## Tensor Layout

A tensor on paper is indexed by logical coordinates such as `A[m, k]`. A tensor in a GPU is indexed by *physical* coordinates that depend on where it lives:

- In **global memory (GMEM)** a tensor is a flat byte array; `A[m, k]` becomes a single byte offset.
- In **shared memory (SMEM)** a tensor is also a flat per-CTA byte array, but the bytes are physically interleaved across 32 memory banks of 4 B each. `A[m, k]` is still a byte offset; the bank number is the low bits of that offset, and bad layouts show up as *bank conflicts* (a warp's 32 lanes colliding on the same bank).
- In **tensor memory (TMEM)** a tensor is a genuinely 2D scratchpad of 32-bit cells addressed by `(TLane, TCol)`; `A[m, k]` becomes a `(TLane, TCol)` pair.
- In **registers (RF)** the register file is per-thread, so there is no single address for `A[m, k]` at all. A logical tile must be *distributed* across the threads of a cooperating team, and the layout declares which thread holds which element and at which register offset.

A **layout** is the function that performs this translation. The same logical tile has four very different layouts across these spaces, and the same memory space can use many different layouts for the same tile. The choice is never cosmetic — a wrong layout produces one of three failures:

**Bank conflicts.** 32 threads of a warp hit the same SMEM bank and serialize. A naive row-major 128×64 fp16 tile has exactly this pathology when 32 threads read the same column from 32 different rows. The result is correct but 2–10× slower than necessary.

**Wrong MMA operands.** `tcgen05.mma` reads SMEM through a *matrix descriptor* that encodes a specific layout (leading dimension, stride, swizzle mode, base offset). If the SMEM layout does not match one of the descriptor-expressible patterns, the MMA silently reads scrambled bytes and the output is `NaN`.

**TMEM–RF mismatch.** When a warpgroup reads TMEM into registers via `Tx.copy_async(Dreg, tmem)`, the compiler matches the TMEM layout to the RF layout thread-by-thread. A 128×8 TMEM tile laid out as `(1@TLane, 1@TCol)` feeds naturally into a 128-thread warpgroup; `(1@TCol, 1@TLane)` loads the transpose.

All three failures happen because the user and the hardware disagree on *where data actually is*. Layouts give both sides the same vocabulary.

### Named Hardware Axes

Each axis is a concrete hardware resource with a fixed range:

| Named axis | Represents | Lives in |
|:-:|:-:|:-:|
| `TLane` | TMEM row index (0…127) | TMEM |
| `TCol` | TMEM column index (0…511) | TMEM |
| `tid_in_wg` | Thread index within a warpgroup (0…127) | RF |
| `laneid` | Thread index within a warp (0…31) | RF |
| `warpid` | Warp index within a CTA | RF |
| `cbx`, `cby` | CTA index within a cluster | Cross-CTA SMEM |
| `m` | Logical memory offset; the default axis used when a stride has no `@axis` suffix | GMEM, SMEM, RF |

When you tag a stride with `@TLane`, you tell the compiler: "this dimension maps one-for-one to the `TLane` axis of TMEM." The axes are a fixed vocabulary; you do not invent new ones.

### Shard, Replica, Offset

```python
from tvm.tirx.layout import TileLayout, S, R, TLane, TCol, tid_in_wg, warpid

layout = TileLayout(
    S[(128, 512) : (1@TLane, 1@TCol)]   # shard: where each logical cell lives
    + R[4 : 1@warpid]                    # replica: same data at 4 warp positions
    + 5@warpid                           # offset: start at warp 5
)
```

- **`S[shape : stride@axis]`** is the *shard*. It says how each logical dimension partitions over named axes: for every logical index `i` on dimension `d`, the physical coordinate gains `stride_d * i` on `axis_d`. If a dimension has no `@axis`, the default is `m` (the linear element offset within the buffer).
- **`+ R[extent : stride@axis]`** is the *replica*. It iterates over additional axis coordinates *without changing the logical index*, so the same data appears at multiple physical locations. Used for broadcasts and TMA multicast.
- **`+ k@axis`** is the *offset*. A constant shift on one axis — mostly invisible in user code; allocators generate it automatically.

Formally the layout $L$ maps each logical index $x$ to a set of hardware coordinates, $L(x) = \{\, S(x) + r + O \mid r \in \mathcal{R} \,\}$. When $\mathcal{R}$ is empty the map is single-valued (a function); otherwise multi-valued (broadcasting).

### Reading `stride@axis`

The expression `k@axis` reads as "advance by `k` units along `axis` when the logical index increases by 1." Two conventions worth noting:

- A bare integer `k` with no `@axis` is shorthand for `k@m`. So `S[(128, 8) : (1@tid_in_wg, 1)]` is identical to `S[(128, 8) : (1@tid_in_wg, 1@m)]`.
- When a whole `S[(shape...)]` has no stride list, strides default to row-major on `m`. For example `S[(PIPE_DEPTH, BLK_M, BLK_K)]` is an ordinary row-major flat buffer with that shape.

Multiple logical dimensions can target the same named axis to express a hierarchical partition. For instance `S[(4, 32) : (32@TLane, 1@TLane)]` means "outer warp of 4 × inner lane of 32", together tiling the 128-lane axis exactly once.

### A Tiny Visualization

The same 128×8 tile, through two layouts:

```python
layout_A = TileLayout(S[(128, 8) : (1@TLane,     1@TCol)])   # lives in TMEM
layout_B = TileLayout(S[(128, 8) : (1@tid_in_wg, 1)])        # lives in RF
```

Layout A places each logical cell `[r, c]` at TMEM coordinate `(TLane=r, TCol=c)` — one grid shared by the whole SM. Layout B places the same `[r, c]` in thread `r`'s register `c` — 128 private register files that, taken together, form the tile. A `Tx.copy_async(B, A)` pairs `TLane ↔ tid_in_wg` and `TCol ↔ m`; that pairing *is* the `tcgen05.ld` instruction.

### The Three Patterns You Will See Again and Again

Three layouts cover every Part III kernel: swizzled SMEM, 2D TMEM, and warpgroup-view RF.

**SMEM: the `tma_shared_layout` helper.** You almost never write an SMEM layout by hand. The helper chooses a TMA- and MMA-compatible swizzled layout from three inputs:

```python
from tvm.tirx.operator.tile_primitive.cuda.tma_utils import (
    tma_shared_layout, SwizzleMode,
)

A_layout = tma_shared_layout(
    dtype="float16",
    swizzle=SwizzleMode.SWIZZLE_128B_ATOM,
    shape=(BLK_M, BLK_K),
)
```

The helper emits a row-major shard wrapped in a 128-byte swizzle pattern. Both TMA (when generating the tensor descriptor) and `tcgen05.mma` (when encoding the matrix descriptor) understand this wrapper directly. Contrast with the no-sugar version (:numref:`sec_no_sugar_gemm`) where the user manually assembles a `Tx.ComposeLayout(Tx.SwizzleLayout(3, 3, 3, ...), Tx.TileLayout(...))`.

Pick the swizzle mode whose atom width equals `K_inner * sizeof(dtype)`. For fp16 with `BLK_K = 64`, one K-row is 128 B, so `SWIZZLE_128B_ATOM` is the natural choice. When in doubt, start there.

**TMEM: `(1@TLane, 1@TCol)`.** Every TMEM layout in this tutorial is a variation of

```python
TileLayout(S[(128, N) : (1@TLane, 1@TCol)])
```

Row dimension pinned to `TLane` with stride 1; column dimension pinned to `TCol` with stride 1. This is the only layout `tcgen05.mma` can write directly. Two rules:

- **The row extent is always 128.** `TLane` has 128 physical lanes, and TMEM allocations are always declared against this full 128-row backing. An `.m128n*` MMA instruction fills all 128 rows of its tile; an `.m64n*` MMA instruction only uses the first 64 `TLane` rows, leaving rows 64–127 untouched (available for an independent 64-row MMA to share the same allocation). You *slice* the column dimension to match an MMA instruction (`N = 128, 256, …`), but the row dimension of the declared TMEM buffer is always 128.
- **Axis order matters.** Swapping to `(1@TCol, 1@TLane)` transposes the tile in TMEM — correct-looking code, transposed output.

**RF: `(1@tid_in_wg, 1)`.** Registers are per-thread, but you can declare a local buffer with a warpgroup-wide logical shape through a layout:

```python
reg = Tx.alloc_buffer((128, N), "float32", scope="local",
                      layout=Tx.wg_local_layout(N))
# wg_local_layout(N) = TileLayout(S[(128, N) : (1@tid_in_wg, 1)])
```

Each thread's `N` physical registers become one logical row of the warpgroup-wide `(128, N)` tile. Now `Tx.copy_async(reg, tmem[:, n0:n0+N])` in every GEMM epilogue works because of **axis matching**:

| Source (data lives in) | Destination (data goes to) | Emitted instruction |
|---|---|---|
| GMEM (`m`) | SMEM (`m`, swizzled) | `cp.async.bulk.tensor` (TMA load) |
| SMEM (`m`, swizzled) | GMEM (`m`) | `cp.async.bulk.tensor` (TMA store) |
| SMEM × 2 operands | TMEM (`TLane`, `TCol`) | `tcgen05.mma` via matrix descriptor |
| TMEM (`TLane`, `TCol`) | RF (`tid_in_wg`, `m`) | `tcgen05.ld` (via `Tx.copy_async`) |
| RF (`tid_in_wg`, `m`) | TMEM (`TLane`, `TCol`) | `tcgen05.st` (via `Tx.copy_async`) |
| RF (`tid_in_wg`, `m`) | SMEM (`m`, swizzled) | shared-memory store |

Matching succeeds when (1) extents agree dimension-by-dimension and (2) the axis pair is in the table above. `TLane ↔ tid_in_wg` works because both have 128 elements and `tcgen05.ld` implements it; `TLane ↔ laneid` does not (`laneid` has only 32), so the compiler would reject it at compile time.

A realistic epilogue reads the 128×`MMA_N` accumulator `N` columns at a time in a loop so that each iteration fits comfortably in the 255-register per-thread budget. The dispatcher always lowers TMEM↔RF traffic to the async `tcgen05.ld` PTX intrinsic, so the user must follow each `copy_async` with `Tx.ptx.tcgen05.wait.ld()` before reading the destination registers:

```python
for no in Tx.unroll(MMA_N // N):
    Tx.copy_async(reg, tmem[:, no * N : (no + 1) * N])
    Tx.ptx.tcgen05.wait.ld()
    # cast / activation / store ...
```

### Where Replicas Show Up

You rarely write `R[...]` by hand. The kernels in this tutorial never spell out a replica; the two places replicas matter are both internal to the compiler:

- **TMA multicast across a CTA cluster.** A single `cp.async.bulk.tensor.multicast` PTX instruction can deliver the same tile to every CTA in the cluster. Internally the dispatcher represents the receiver set with a replica on a cluster axis (conceptually `… + R[2 : 1@cbx]` for a 2-CTA broadcast). At the *user* level you only see the `cta_mask=...` argument on a `Tx.copy_async(..., dispatch="tma")` call; the layout with the replica is synthesised inside the dispatcher. The clustered GEMM in :numref:`chap_gemm_advanced` (Step 8) uses `cta_group=2` for cooperative MMA but does *not* multicast the A / B tiles, so even this kernel does not spell the replica out.
- **Broadcast operands of cooperative primitives.** When a tile primitive consumes one operand that is "the same value across the cooperating team" — for instance a scalar fed into a per-lane RF tile — the dispatcher pads the smaller layout with an implicit replica to align extents before axis matching. This is again a compiler-side construction; the user only writes the natural one-dimensional layout.

If you want to see an explicit replica in source code, the cleanest example lives in the Axe Layout paper (Hou et al., 2026) §3, where TMA multicast is illustrated as `S[(3, 2, 128, 128) : (16384, 1@cbx, 128, 1)] + R[4 : 1@cby]`. In the kernels of this tutorial, the replica stays inside the compiler.


## Tile Primitive Dispatch
:label:`sec_tile_primitive_dispatch`

With execution scope and tensor layout in place, we can finally state what a *tile primitive* is and how TIRX lowers one. A **tile primitive** is a user-facing operation on tiles — `Tx.copy`, `Tx.gemm_async`, `Tx.cast`, `Tx.add`, `Tx.reduce`, `mbar.arrive`, and so on. Calling a primitive is *declarative*: it states what the operation does on the logical tensor values. The *how* — which instruction, at which thread granularity, using which addressing mode — is resolved by the compiler from three inputs:

$$
\underbrace{\text{primitive call}}_{\text{what}} \;+\; \underbrace{\text{execution scope}}_{\text{who}} \;+\; \underbrace{\text{operand layouts}}_{\text{where}} \;\;\longrightarrow\;\; \text{backend instruction(s)}
$$

The programmer never writes the dispatch logic; the compiler does, reusing a small set of layout operators — canonicalise, group, permute, slice — to match the primitive's shape requirements against the operand layouts and pick an efficient lowering. Two examples make the mechanism concrete.

### Example A: one `Tx.copy`, four different lowerings

A `Tx.copy(dst, src)` (synchronous) or `Tx.copy_async(dst, src)` (async, requires a user-issued wait) call is dispatched to different PTX depending on the primitive (`copy` vs `copy_async`), the enclosing scope, and the operand layouts:

| Primitive | Team kind | `dst` layout | `src` layout | Lowered to |
|:--|:--|:--|:--|:--|
| `copy_async` | thread (elected), `cta_group=2` | SMEM whose receiver set covers more than one CTA of the cluster (e.g. `R[2 : 1@cbx]` for a 2-CTA broadcast) | GMEM tile | **TMA multicast** (`cp.async.bulk.tensor.multicast`) |
| `copy_async` | thread (elected), single CTA | SMEM tile (swizzled) | GMEM tile | **TMA unicast** (`cp.async.bulk.tensor`) |
| `copy_async` | warpgroup | RF warpgroup view `(1@tid_in_wg, 1)` | TMEM tile `(1@TLane, 1@TCol)` | **`tcgen05.ld`**, warpgroup-cooperative async TMEM → RF — user must call `Tx.ptx.tcgen05.wait.ld()` to synchronise |
| `copy` | thread / warp | RF per-thread flat | GMEM / SMEM flat | plain vectorised **LDG / LDS / STS / STG** |

From the user's point of view the code in each row is essentially the same:

```python
Tx.copy(dst, src)                                       # synchronous (LDG/STG/LDS/STS)
# or, for any path that touches an async hardware engine:
Tx.copy_async(dst, src, dispatch="tma", cta_group=CTA_GROUP, mbar=...)   # TMA
Tx.copy_async(reg, tmem)                                 # tcgen05.ld
Tx.ptx.tcgen05.wait.ld()                                 # required before reading reg
```

What makes the first row lower to TMA multicast and the third row to `tcgen05.ld` is (a) the primitive choice (`copy_async`, because both TMA and `tcgen05.ld` are asynchronous PTX instructions), (b) the team kind flowing from the enclosing scope, and (c) the *named axes* in the destination layout — `cbx`/`cby` invites multicast, `TLane`/`TCol` invites `tcgen05.ld`. Declare a wrong layout and the dispatcher either picks a slower instruction or rejects the call with a compile-time error.

Contrast this with the no-sugar TMA load in :numref:`sec_no_sugar_gemm`: a single `Tx.copy_async(A_smem, A, dispatch="tma", mbar=bar.ptr_to([0]))` replaces the manual `cuTensorMapEncodeTiled` call on the host, the manual swizzle layout on SMEM, *and* the manual `expect_tx` byte counting — because all three pieces are recoverable from the scope plus the two operand layouts.

### Example B: `Tx.gemm_async` picks the MMA shape and CTA group

```python
Tx.gemm_async(tmem[:, :N], Asmem, Bsmem,
              accum=(k != 0), dispatch="tcgen05", cta_group=1)
```

The primitive says "accumulate $A B$ into `tmem`". The compiler consults:

- `tmem`'s layout `S[(128, N) : (1@TLane, 1@TCol)]` → the target lives in TMEM, so choose a `tcgen05.mma` variant.
- `Asmem` / `Bsmem` SMEM layouts (including swizzle mode) → build the SMEM matrix descriptors and set transpose flags required by `tcgen05.mma`.
- `cta_group=1` → single-CTA instance. Passing `cta_group=2` on a clustered launch (`Tx.cta_id_in_cluster([2, 1])`) produces the 2-CTA cooperative form, in which the MMA proxy reads the peer CTA's SMEM via cross-CTA access; the `M` axis doubles to up to $256$.
- Tile shape inferred from the layouts → select a valid `tcgen05.mma` M-N-K shape ($M \in \{64, 128\}$ for single-CTA or $\{128, 256\}$ with `cta_group=2`; $N \le 256$ with a step that depends on $M$ and CTA group — see :numref:`chap_background`; $K$ set by element type).

A single primitive call emits one — or, for large tiles, a short loop of — `tcgen05.mma` instructions with the right bit-packed instruction and SMEM descriptors. The descriptor encoding, the transpose handling, and the instruction loop all live in the dispatcher, not in the user kernel. That is why the 2-line MMA in :numref:`chap_gemm_basics` expands to 20+ lines of PTX without the user ever opening a PTX manual — the same expansion you saw spelled out by hand in :numref:`sec_no_sugar_gemm`.

### What to internalise

- **Declarativity.** The primitive says *what*; the scope and layouts say *for which team and where the data is*; the dispatcher says *how*. Changing the lowering strategy — for instance switching a TMA copy between unicast and multicast — is a layout edit, not a rewrite of the operation.
- **Layout selects the instruction family.** `cbx` / `cby` axes → cluster instructions; `TLane` / `TCol` → `tcgen05` instructions; `tid_in_wg` / `warpid` / `laneid` + `m` → ordinary per-thread LDG / STS / STG. The named-axis vocabulary is the user's way to *select* a lowering.
- **Scope gates which primitives are legal.** Some primitives only make sense under a specific team kind — e.g. `tcgen05.ld` requires a warpgroup-cooperative read, so calling it from `with Tx.thread():` is a compile-time error.


## The Sugared Rewrite
:label:`sec_sugared_rewrite`

With all three interfaces in hand, the no-sugar kernel of :numref:`sec_no_sugar_gemm` is replaceable by a much shorter program that emits the same PTX. The definitive reference is Step 1 of :numref:`chap_gemm_basics` — a runnable single-tile GEMM using only the TIRX abstractions introduced above, and no raw `Tx.ptx.*` calls for the core dataflow. Here is how each pain point we flagged at the end of :numref:`sec_no_sugar_gemm` is erased:

| No-sugar pain point | Sugared replacement | Lives in |
|:--|:--|:--|
| `cuTensorMapEncodeTiled` on the host + manual swizzle `ComposeLayout` on SMEM | `tma_shared_layout(...)` + `Tx.copy_async(dst, src, dispatch="tma")` | *Tensor Layout* and *Tile Primitive Dispatch* above |
| `tcgen05.encode_instr_descriptor` / `encode_matrix_descriptor` / `mma` / `commit` loop | `Tx.gemm_async(tmem, Asmem, Bsmem, dispatch="tcgen05", cta_group=1)` | *Tile Primitive Dispatch* |
| Three separate `Tx.shared_scalar`s for `tmem_addr`, `bar_tma`, `bar_mma` | `Tx.SMEMPool` + `TMABar(...)` / `TCGen05Bar(...)` helpers | *Buffers and Scopes* in :numref:`chap_tirx_primer` |
| 128 × `Tx.ptx.tcgen05.ld` calls with hand-rolled `row = warp_id * 32` | one `Tx.copy_async(reg, tmem[:, :N])` + `Tx.ptx.tcgen05.wait.ld()` under `with Tx.warpgroup():` (where `reg` has `layout=Tx.wg_local_layout(N)`) | *Tile Primitive Dispatch* |
| `if tx == 0:` + fences + phase-bit flipping | `with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):` + `TMABar.wait(stage, phase)` / `arrive(stage, tx_count=...)` | *Execution Scope* |

The runnable kernel in Step 1 of :numref:`chap_gemm_basics` is roughly half the length of the no-sugar version with none of the bitfield plumbing (descriptor bits, `expect_tx` byte counts, per-column `tcgen05.ld` calls). For a production-scale example with pipelining and a persistent scheduler, see `tirx-kernels-wt/tirx_kernels/gemm/hgemm_1consumer.py`.


## Summary

1. Everything in a Blackwell GEMM can be written in raw `Tx.ptx.*` intrinsics, and the resulting kernel is 130 lines of fragile boilerplate (:numref:`sec_no_sugar_gemm`).
2. **Execution scope** says *who cooperates*: a team kind (`cta` / `warpgroup` / `warp` / `thread`) plus an optional active-thread predicate. It is the permission system for tile primitives.
3. **Tensor layout** says *where data lives*: a shard / replica / offset expression over a fixed vocabulary of hardware axes (`TLane`, `TCol`, `tid_in_wg`, `cbx`, …). A wrong layout produces a performance collapse or a silently wrong result, never a crash.
4. **Tile primitive dispatch** combines the two — plus the primitive call itself — to pick a PTX instruction. A single `Tx.copy` / `Tx.copy_async` can lower to TMA multicast, TMA unicast, `tcgen05.ld`, or plain STG depending only on the enclosing scope and the operand layouts.

:numref:`chap_fused_gelu` uses a minimal elementwise operator to put the SMEM and RF patterns to work; :numref:`chap_gemm_basics` begins the GEMM journey where every layout in this chapter reappears in context.
