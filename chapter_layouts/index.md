# TIRX Layout: Mapping Tensors to Hardware
:label:`chap_layouts`

:numref:`chap_tirx_primer` introduced tensor layout as the first pillar of the TIRX programming model. This chapter opens up that interface. It answers three questions in order:

1. **Why do we need layouts at all?** — what goes wrong without them.
2. **What is a layout?** — the vocabulary and the three operators.
3. **What do real layouts look like?** — the three patterns used in every Part III kernel.

By the end you will be able to read every `layout=` expression in the GEMM, RMSNorm, and Flash Attention kernels, write your own for custom buffers, and diagnose bank-conflict or TMEM-matching bugs by inspecting the layout.


## Why Layouts Matter

A tensor on paper is indexed by logical coordinates such as `A[m, k]`. A tensor in a GPU is indexed by *physical* coordinates that depend on where it lives:

- In **global memory (GMEM)** a tensor is a flat byte array; `A[m, k]` becomes a byte offset.
- In **shared memory (SMEM)** a tensor lives in a 32-bank scratchpad with 4-byte banks; `A[m, k]` becomes a (bank, bank-offset) pair.
- In **tensor memory (TMEM)** a tensor is a 2D array addressed by `(TLane, TCol)`; `A[m, k]` becomes a `(TLane, TCol)` pair.
- In **registers (RF)** a tensor is sharded across 128 threads of a warpgroup; `A[m, k]` becomes a (thread-id, register-index) pair.

A **layout** is the function that performs this translation. The same logical tile has four very different layouts across these spaces, and the same memory space can use many different layouts for the same tile. The choice is never cosmetic — a wrong layout produces one of three failures:

**Bank conflicts.** 32 threads of a warp hit the same SMEM bank and serialize. A naive row-major 128×64 fp16 tile has exactly this pathology when 32 threads read the same column from 32 different rows. The result is correct but 2–10× slower than necessary.

**Wrong MMA operands.** `tcgen05.mma` reads SMEM through a *matrix descriptor* that encodes a specific layout (leading dimension, stride, swizzle mode, base offset). If the SMEM layout does not match one of the descriptor-expressible patterns, the MMA silently reads scrambled bytes and the output is `NaN`.

**TMEM–RF mismatch.** When a warpgroup reads TMEM into registers via `Tx.copy(Dreg, tmem)`, the compiler matches the TMEM layout to the RF layout thread-by-thread. A 128×8 TMEM tile laid out as `(1@TLane, 1@TCol)` feeds naturally into a 128-thread warpgroup; `(1@TCol, 1@TLane)` loads the transpose.

All three failures happen because the user and the hardware disagree on *where data actually is*. TIRX Layout gives both sides the same vocabulary.


## What a Layout Is

A TIRX layout maps logical indices to coordinates on **named hardware axes** via three operators: `S` (shard), `R` (replica), and `O` (offset).

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

Layout A places each logical cell `[r, c]` at TMEM coordinate `(TLane=r, TCol=c)` — one grid shared by the whole SM. Layout B places the same `[r, c]` in thread `r`'s register `c` — 128 private register files that, taken together, form the tile. A `Tx.copy(B, A)` pairs `TLane ↔ tid_in_wg` and `TCol ↔ m`; that pairing *is* the `tcgen05.ld` instruction.


## Layouts in Practice

Three patterns cover every Part III kernel: swizzled SMEM, 2D TMEM, and warpgroup-view RF. The axis-matching rule stitches them together.

### SMEM: the `tma_shared_layout` Helper

You almost never write an SMEM layout by hand. The helper chooses a TMA- and MMA-compatible swizzled layout from three inputs:

```python
from tvm.tirx.operator.scope_op_dispatch.cuda.tma_utils import (
    tma_shared_layout, SwizzleMode,
)

A_layout = tma_shared_layout(
    dtype="float16",
    swizzle=SwizzleMode.SWIZZLE_128B_ATOM,
    shape=(BLK_M, BLK_K),
)
```

The helper emits a row-major shard wrapped in a 128-byte swizzle pattern. Both TMA (generating the tensor descriptor) and `tcgen05.mma` (encoding the matrix descriptor) understand this wrapper directly.

**Why swizzle at all?** Blackwell SMEM has 32 banks of 4 bytes each. A naive row-major fp16 `(128, 64)` tile has `bank(m, k) = k/2 mod 32` — *independent of `m`* — so 32 threads reading the same column from 32 different rows all collide on one bank. The 128-byte swizzle scrambles the column index as a function of the row so the 32 accesses hit 32 distinct banks.

**Choosing a mode.** Pick the mode whose atom width equals `K_inner * sizeof(dtype)`. For fp16 with `BLK_K = 64`, one K-row is 128 B, so `SWIZZLE_128B_ATOM` is the natural choice. When in doubt, start with `SWIZZLE_128B_ATOM`.

Multi-stage SMEM for software pipelining is the same layout with an extra outer dimension:

```python
A_layout = tma_shared_layout(
    "float16", SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_M, BLK_K),
)
```

Swizzle wraps each stage independently; the stage dimension defaults to `m`, so the stride between stages is just the size of one tile.

### TMEM: `(1@TLane, 1@TCol)`

Every TMEM layout in this tutorial is a variation of:

```python
TileLayout(S[(128, N) : (1@TLane, 1@TCol)])
```

Row dimension pinned to `TLane` with stride 1; column dimension pinned to `TCol` with stride 1. This is the only layout `tcgen05.mma` can write directly. Two rules:

- **The row extent is always 128.** `TLane` has 128 physical lanes. Even a 64-row MMA instruction writes into 128 TMEM rows via a 64×N logical view over a 128-row physical backing. You *slice* the column dimension to match an MMA instruction (`N = 128, 256, …`), but the row dimension is always 128.
- **Axis order matters.** Swapping to `(1@TCol, 1@TLane)` transposes the tile in TMEM — correct-looking code, transposed output.

### RF: `(1@tid_in_wg, 1)` and the TMEM→RF Match

Registers are per-thread, but you can *view* a per-thread register buffer as a warpgroup-wide tile through a layout:

```python
from tvm.tirx.layout import wg_local_layout

reg_wg = reg.view(128, N, layout=wg_local_layout(N))
# wg_local_layout(N) expands to TileLayout(S[(128, N) : (1@tid_in_wg, 1)])
```

Each thread's `N` physical registers become one logical row. Now `Tx.copy(reg_wg, tmem[:, n0:n0+N])` in every GEMM epilogue works because of **axis matching**:

| Source (data lives in) | Destination (data goes to) | Emitted instruction |
|---|---|---|
| GMEM (`m`) | SMEM (`m`, swizzled) | `cp.async.bulk.tensor` (TMA load) |
| SMEM (`m`, swizzled) | GMEM (`m`) | `cp.async.bulk.tensor` (TMA store) |
| SMEM × 2 operands | TMEM (`TLane`, `TCol`) | `tcgen05.mma` via matrix descriptor |
| TMEM (`TLane`, `TCol`) | RF (`tid_in_wg`, `m`) | `tcgen05.ld` |
| RF (`tid_in_wg`, `m`) | TMEM (`TLane`, `TCol`) | `tcgen05.st` |
| RF (`tid_in_wg`, `m`) | SMEM (`m`, swizzled) | shared-memory store |

Matching succeeds when (1) extents agree dimension-by-dimension and (2) the axis pair is in the table above. `TLane ↔ tid_in_wg` works because both have 128 elements and `tcgen05.ld` implements it; `TLane ↔ laneid` does not (`laneid` has only 32), so the compiler would reject it at compile time.

A realistic epilogue reads the 128×`MMA_N` accumulator `N` columns at a time in a loop, so that each iteration fits comfortably in the 255-register per-thread budget:

```python
for no in Tx.unroll(MMA_N // N):
    Tx.copy(reg_wg, tmem[:, no * N : (no + 1) * N])
    # cast / activation / store ...
```

### Where Replicas Show Up

You rarely write `R[...]` from scratch — the tutorial uses it in one place, and in one more place it is conceptually implicit:

- **Scale / scalar broadcasts.** The softmax scale vector in Flash Attention (:numref:`chap_flash_attention`) is a length-`N` buffer broadcast to all 128 warpgroup threads via `S[(N,):(1,)] + R[128 : 1@tid_in_wg]`. This is the one explicit replica you will read in user code.
- **TMA multicast across a CTA cluster** (feature-level, not used by the kernels in this tutorial). A single TMA instruction can deliver the same tile to every CTA in the cluster; TIRX expresses the receiver set as a replica on a cluster axis such as `R[2 : 1@cbx]`, and lowers the copy to `cp.async.bulk.tensor.multicast`. In practice the user passes a cluster mask like `cta_mask=3` to `Tx.copy_async(..., dispatch="tma", ...)` and the dispatcher synthesises the layout; the clustered GEMM in :numref:`chap_gemm_advanced` (Step 8) uses `cta_group=2` for cooperative MMA but does *not* multicast the A / B tiles, so this layout pattern does not appear in its code.

The explicit replica you will actually see is therefore the Flash-Attention scale vector. For the clustered GEMM, `Tx.copy_async(..., cta_group=2)` and `smem_pipe.full.remote_view(...)` compose any cluster-aware layouts for you.


## Summary

1. **A layout maps logical indices to coordinates on named hardware axes.** The axes (`TLane`, `TCol`, `tid_in_wg`, `cbx`, …) are a hardware-fixed vocabulary; the three operators `S` / `R` / `O` are the only way to compose them.
2. **Compatibility between two layouts is an axis-matching problem.** `Tx.copy` succeeds when extents agree and the axis pair is one of the supported hardware pairings (`TLane ↔ tid_in_wg` → `tcgen05.ld`, etc.).
3. **The cost of a wrong layout is a performance collapse or a silently wrong result, not a crash.** Bank conflicts, swizzle mismatches, and transposed TMEM reads all show up this way. Check layouts first when a kernel is slow or subtly broken.

:numref:`chap_fused_gelu` uses a minimal elementwise operator to put the SMEM and RF patterns to work; :numref:`chap_gemm_basics` begins the GEMM journey where every layout in this chapter reappears in context.
