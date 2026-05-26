# Tile Primitive Mental Model
:label:`chap_layouts`

The background chapter introduced the Blackwell pieces: thread groups, memory spaces, TMA, `tcgen05` MMA, TMEM, mbarriers, and CTA clusters. This chapter shows how those pieces appear in TIRX code.

A **tile primitive** is one operation on a tile: copy this tile, multiply these tiles, read this accumulator tile, or store this output tile. It is smaller than a whole operator like GEMM, but larger than a scalar instruction.

The important point is that a tile primitive is read with its local context. For example, `Tx.copy_async(...)` is not one fixed instruction. Depending on the source, destination, scope, and dispatch, it can mean a TMA load, a TMA store, or a `tcgen05.ld` readback.

When reading a TIRX tile primitive, check three things around the call:

```text
scope    -> which threads cooperate on the tile operation
layout   -> how the logical tile maps to memory axes or thread axes
dispatch -> which hardware path implements the operation
```

For asynchronous primitives, synchronization is the handoff around the operation: a barrier, commit, wait, or fence tells the next operation when it is allowed to consume the result.

The rest of the chapter walks through these pieces in the order you usually read a kernel: scope first, then layout, then dispatch.


## Why Tile Primitives?
:label:`sec_no_sugar_gemm`

Use one GEMM stage as the running example. A stage moves operand tiles into shared memory, runs Tensor Core MMA on those tiles, reads accumulator tiles back, and stores the result. Attention and convolution kernels have different math, but they use the same kind of tile-level structure.

The math alone is not enough. For each tile operation, a Blackwell kernel also has to answer:

- Who cooperates on the operation?
- Where does the tile live?
- How is the tile laid out in that memory space?
- Which hardware path should run it: TMA, `tcgen05.mma`, `tcgen05.ld`, or ordinary loads and stores?
- For an asynchronous operation, what tells the next step that the result is ready?

Those questions are exactly what scope, layout, dispatch, and synchronization record in TIRX.

If you write directly against raw PTX-level intrinsics, those facts are spread across descriptors, elected-thread guards, mbarrier phases, TMEM addresses, and wait calls. Everything is explicit, but the tile structure is hard to read.

In a higher-level kernel DSL, the kernel may be shorter, but choices such as TMEM readback, TMA multicast, or 2-CTA cooperative MMA are often hidden behind a scheduler, template, or library call.

TIRX sits between those two styles. The hardware choices stay visible, but they sit next to the tile operation instead of being scattered through descriptor plumbing:

| Raw concern | Where it appears in TIRX |
|---|---|
| TMA descriptor fields | `Tx.copy_async(..., dispatch="tma")` plus source and destination layouts |
| `tcgen05` matrix descriptors | `Tx.gemm_async(..., dispatch="tcgen05")` plus SMEM and TMEM layouts |
| Elected or cooperative thread group | enclosing scope such as `Tx.cta()`, `Tx.warpgroup()`, or `Tx.thread(...)` |
| TMEM-to-register readback | `Tx.copy_async(reg, tmem)` under warpgroup scope |
| Async completion bookkeeping | barrier helpers, commits, waits, and fences around the primitive |

The point is not to hide the hardware. The point is to keep the tile structure readable while still exposing the Blackwell choices that matter.

## Execution Scope
:label:`sec_exec_scope`

Start with the first question: who cooperates on the operation? In TIRX, the answer comes from the enclosing scope.

TIRX uses `with` blocks for the execution levels used by Blackwell kernels:

```python
with Tx.cta():
    Tx.copy(Asmem[:, :], A[m:m + BLK_M, k:k + BLK_K])

with Tx.warpgroup():
    Tx.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])

with Tx.thread():
    Tx.copy(D[m, n], Dreg[i])
```

A scope block does not create new physical threads. The CUDA launch already did that. The scope tells TIRX how to interpret the primitive inside the block.

For example, a copy under `Tx.cta()` is a cooperative CTA-level copy. A copy under `Tx.thread()` is a per-thread copy. A TMEM-to-register readback must be under `Tx.warpgroup()` because the corresponding `tcgen05.ld` instruction is warpgroup-cooperative.

The scopes are not a fixed template that every kernel must nest exactly. A kernel uses the scopes that match the operations it performs. `Tx.kernel()` marks the whole kernel body; `Tx.cluster()` appears only when CTA clusters are used.

Some operations are issued by only part of a team. A TMA load, for example, may be issued by one selected thread:

```python
tid = Tx.thread_id_in_wg([128])
with Tx.thread(tid == 0):
    Tx.copy_async(Asmem[:, :], A[m:m + BLK_M, k:k + BLK_K], dispatch="tma")
```

For now, keep the simple rule: every tile primitive has a team, and sometimes only a selected member of that team issues the command on behalf of the team.

### Symbolic Coordinates

Scope tells you the team. Coordinates tell each member of that team where it is. A CTA needs to know which output tile it owns. A warpgroup needs to know which warpgroup it is. A thread needs its warp and lane id to decide which row or vector fragment it writes.

TIRX exposes those positions with symbolic coordinate calls. The GEMM chapters mostly use these:

| Coordinate | Meaning | Typical extent |
|---|---|---|
| `Tx.cta_id([Gx, Gy, ...])` | CTA id in the kernel launch | grid dims |
| `Tx.warpgroup_id([Nw])` | warpgroup id inside a CTA | 1 or 2 in these GEMM kernels |
| `Tx.warp_id([Nw])` | warp id inside a CTA | CTA warp count |
| `Tx.warp_id_in_wg([4])` | warp id inside a warpgroup | 4 |
| `Tx.thread_id([Nt])` | thread id inside a CTA | CTA thread count |
| `Tx.thread_id_in_wg([128])` | thread id inside a warpgroup | 128 |
| `Tx.lane_id([32])` | lane id inside a warp | 32 |

Cluster coordinates appear only in the clustered GEMM chapter:

| Coordinate | Meaning |
|---|---|
| `Tx.cluster_id([...])` | cluster id in the kernel launch |
| `Tx.cta_id_in_cluster([Cx, Cy])` | CTA id inside a cluster |

These calls are more than renamed `threadIdx` reads. The extent argument, such as `[4]` in `Tx.warp_id_in_wg([4])`, tells TIRX the size of that coordinate space. Later, layouts and tile primitives can be checked against those extents.


## Tensor Layout

After scope, look at layout. Layout answers the next question: **how does each element of a logical tile map to memory axes or thread axes?**

On paper, a tile is indexed by logical coordinates such as `A[m, k]`. On the GPU, the same logical tile may live in different physical spaces:

- In **GMEM**, it is a flat byte address in global memory.
- In **SMEM**, it is still a per-CTA byte address, but the address also determines shared-memory bank behavior.
- In **TMEM**, it is a two-dimensional tensor-memory coordinate, usually described with `TLane` and `TCol`.
- In **RF**, there is no single shared address. The tile is distributed across the private registers of many cooperating threads.

A layout is the mapping from logical tile indices to memory coordinates or thread owners. This is why layout changes behavior. A bad layout can create shared-memory bank conflicts, make `tcgen05.mma` read the wrong bytes, or make a TMEM-to-RF readback produce a transposed tile.

The easiest example is the accumulator tile. After MMA, the result lives in TMEM. During the epilogue, the warpgroup reads that same logical tile into registers:

```python
# TMEM accumulator: row r, column c lives at TMEM coordinate (TLane=r, TCol=c).
TileLayout(S[(128, N) : (1@TLane, 1@TCol)])

# RF readback view: logical row r is owned by warpgroup thread r.
TileLayout(S[(128, N) : (1@tid_in_wg, 1)])
```

Both layouts describe a logical `(128, N)` tile. The first says how the tile lives in TMEM. The second is a register view: it says how the same tile is split across the private registers of the 128 warpgroup threads.

### Named Axes

TIRX layout expressions use named hardware axes. The GEMM chapters mostly use:

| Axis | Meaning | Used for |
|---|---|---|
| `m` | ordinary linear memory offset | GMEM, SMEM, flat RF buffers |
| `TLane` | TMEM row coordinate, 0..127 | TMEM accumulator rows |
| `TCol` | TMEM column coordinate, 0..511 | TMEM accumulator columns |
| `tid_in_wg` | thread id inside a warpgroup, 0..127 | warpgroup-distributed RF tiles |

The notation `1@TLane` means: when the logical index increases by 1, advance by 1 along the `TLane` hardware axis. A bare stride such as `1` means `1@m`, the ordinary linear memory axis.

Other axes, such as `laneid`, `warpid`, `cbx`, and `cby`, appear when the code needs warp-level layouts or cluster-aware layouts.

The other recurring layout is the SMEM operand layout:

```python
# Swizzled SMEM layout for TMA and MMA operands.
A_layout = tma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
```

Read this as: store the SMEM tile in a swizzled form that the TMA descriptor path and the `tcgen05.mma` matrix-descriptor path can both understand.

### Common Layout Pairings

The GEMM chapters reuse a small number of layout-to-instruction pairings:

| Source and destination | Hardware path |
|---|---|
| GMEM tile -> swizzled SMEM tile | TMA load when `dispatch="tma"` |
| swizzled SMEM tile -> GMEM tile | TMA store when `dispatch="tma"` |
| two SMEM operand tiles -> TMEM accumulator tile | `tcgen05.mma` |
| TMEM tile `(TLane, TCol)` -> RF tile `(tid_in_wg, m)` | `tcgen05.ld` |
| RF tile -> GMEM or SMEM | ordinary thread stores |

For example, `TLane` and `tid_in_wg` both have extent 128, so a TMEM row can map naturally to one warpgroup thread during `tcgen05.ld`. Mapping the same TMEM rows to a 32-lane warp axis would not describe the warpgroup readback that this instruction expects.


## Tile Primitive Dispatch
:label:`sec_tile_primitive_dispatch`

After source and destination layouts, look at dispatch. Dispatch answers the next question: **which hardware path should run the tile operation?**

Start with copy. In TIRX, `copy` means "move this tile or fragment." The hardware path depends on the source, destination, scope, and sometimes an explicit `dispatch` argument:

| Code pattern | How to read it |
|---|---|
| `Tx.copy_async(Asmem[:, :], A[m:m + BLK_M, k:k + BLK_K], dispatch="tma")` | issue an async TMA load from GMEM into SMEM |
| `Tx.copy_async(D[m:m + BLK_M, n:n + BLK_N], Dsmem[:, :], dispatch="tma")` | issue an async TMA store from SMEM back to GMEM |
| `Tx.copy_async(Dreg_wg, tmem)` under `Tx.warpgroup()` | read TMEM into warpgroup registers with `tcgen05.ld` |
| `Tx.copy(D[m, n], Dreg[i])` under `Tx.thread()` | ordinary per-thread store |

The `dispatch="tma"` part is not just a performance hint. It is a request to use the TMA hardware path, so the source and destination must look like a legal TMA transfer. For example, the SMEM side needs a TMA-compatible layout such as `tma_shared_layout(...)`, and the async copy needs the matching barrier protocol.

For TMEM readback, the code usually has no `dispatch="tcgen05.ld"` string. The path is implied by the operands and scope: source in TMEM, destination in a warpgroup register view, and enclosing `Tx.warpgroup()`.

MMA is the same idea, but the operation is compute instead of copy:

```python
Tx.gemm_async(tmem, Asmem, Bsmem, accum=False, dispatch="tcgen05")
```

Read this as: use Blackwell `tcgen05` MMA to multiply the two SMEM operand tiles and write the accumulator tile to TMEM. The layouts, dtype, tile shape, and CTA-group setting determine which `tcgen05` form is legal and how the descriptors are built.

So when you read GEMM code, do not read `Tx.copy_async` or `Tx.gemm_async` in isolation. Check the surrounding scope and the source/destination layouts. That local context tells you whether the primitive is asking for TMA, `tcgen05.ld`, `tcgen05.mma`, or ordinary load/store code.


## How to Read the Code

The next chapter starts showing real kernels. You do not need the full language reference before reading them. The following vocabulary is enough to follow the first GEMM:

| Code | How to read it |
|---|---|
| `@Tx.prim_func` | defines one TIRX kernel function |
| `Tx.Buffer(shape, dtype)` | typed input or output tensor |
| `with Tx.kernel():` | begins the kernel body |
| `Tx.cta_id(...)`, `Tx.warpgroup_id(...)`, `Tx.warp_id_in_wg(...)`, `Tx.lane_id(...)` | symbolic launch and thread coordinates |
| `with Tx.cta()`, `with Tx.warpgroup()`, `with Tx.thread()` | which team cooperates on the code inside |
| `Tx.SMEMPool().alloc(...)` | allocate shared-memory storage for tiles and barriers |
| `Tx.alloc_local(...)` | allocate per-thread register storage |
| `Tx.decl_buffer(..., scope="tmem", allocated_addr=...)` | create a view of a TMEM allocation |
| `Tx.copy(...)` | synchronous copy or store |
| `Tx.copy_async(...)` | asynchronous tile movement; look nearby for the matching wait or barrier |
| `Tx.gemm_async(..., dispatch="tcgen05")` | tile-level Tensor Core MMA |
| `Tx.meta_var(expr)` | keep an address expression inline instead of creating a separate TIR variable |

Treat the code as a sequence of tile operations first. The parser details, dynamic buffers, and metaprogramming rules live in :numref:`chap_tirx_primer` for when you need a reference.


## Back to GEMM
:label:`sec_sugared_rewrite`

Now return to GEMM. The kernels will keep returning to the same data path:

```text
GMEM operand tiles
  -> SMEM operand tiles        via tile copy
  -> TMEM accumulator tile     via tcgen05 MMA
  -> RF tile                   via TMEM readback
  -> GMEM output tile          via stores or TMA store
```

Each arrow becomes one or more tile primitives. To understand a GEMM kernel, read each arrow as a local contract: what tile moves or computes, where it comes from, where it goes, which threads cooperate, which hardware path is requested, and, for async paths, what wait makes the result safe to use.

For example, the TMEM -> RF arrow may appear as:

```python
Tx.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
```

Do not stop at the function name. Read the surrounding context:

1. What operation is this? Here it is a copy.
2. What is the source? Here the source is TMEM.
3. What is the destination? Here the destination is a warpgroup register view.
4. Which threads run it? Here the surrounding `Tx.warpgroup()` means 128 threads cooperate.
5. Is there an async handoff? Here the code must call `Tx.ptx.tcgen05.wait.ld()` before reading `Dreg_wg`.

The next chapter starts with a simplified single-tile version of this data path, then adds K-loop accumulation and spatial tiling across CTAs. Later chapters replace the synchronous copy/writeback pieces with TMA pipelining, warp specialization, and CTA clusters.
