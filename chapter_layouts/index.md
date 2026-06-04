# Tile Primitive Mental Model
:label:`chap_layouts`

TIRX code is built around tile primitives. Before looking at a full GEMM kernel, it helps to know what one primitive means.

A **tile primitive** is one operation on a tile: copy this tile, multiply these tiles, read this accumulator tile, or store this output tile. It is smaller than a whole operator like GEMM, but larger than a scalar instruction.

In code, you write a tile primitive as a function call such as `Tx.copy_async(...)`. But the function name is only part of the meaning: `Tx.copy_async` is not one fixed instruction. Depending on the source, destination, scope, and dispatch, it can mean a TMA load, a TMA store, or a `tcgen05.ld` readback.

When reading a TIRX primitive, check the local contract around the call:

```text
scope    -> which threads cooperate on the tile operation
layout   -> how the logical tile maps to memory axes or thread axes
dispatch -> which hardware path implements the operation
```

For asynchronous primitives, a barrier, commit, wait, or fence records the handoff to the next operation.


## Why Tile Primitives?
:label:`sec_why_tile_primitives`

Use one GEMM stage as the running example. A stage moves operand tiles into shared memory, runs Tensor Core MMA on those tiles, reads accumulator tiles back, and stores the result. Attention and convolution kernels use different math, but they rely on the same tile-level structure — and that is no accident: the **tile is the unit Blackwell hardware actually operates on**. TMA moves tiles, `tcgen05` multiplies tiles, TMEM holds tiles. Writing at tile granularity matches the hardware while staying above scalar address-and-loop plumbing.

What we want is a kernel that runs near peak **and** stays readable — one you can follow, debug, and modify. On Blackwell those two goals pull against each other, because the math alone is not enough. For each tile operation the kernel must also answer:

- Who cooperates on the operation?
- Where does the tile live, and how is it laid out there?
- Which hardware path runs it: TMA, `tcgen05.mma`, `tcgen05.ld`, or ordinary loads and stores?
- For an asynchronous operation, what tells the next step that the result is ready?

The two existing styles each give up one side of that goal. **Raw PTX-level code** answers every question explicitly — and a single tile load drowns in the plumbing:

```text
# (schematic) one A-tile load, raw:
elect one lane  →  cp.async.bulk.tensor [Asmem], [A descriptor]
                →  mbarrier.arrive.expect_tx  (bytes)
                →  mbarrier.try_wait  (phase)
```

Everything is controllable, but the *operation* — "copy this A tile into SMEM" — is buried. A **higher-level DSL** goes the other way: the kernel gets short, but Blackwell-specific choices such as TMEM readback, TMA multicast, or 2-CTA cooperative MMA disappear behind a scheduler, template, or library call — so when the kernel is slow, you cannot see, let alone change, the decision that made it slow.

TIRX keeps the tile operation in view *and* the hardware contract explicit. The same load is one line:

```python
Tx.copy_async(Asmem[:, :], A[m:m+BLK_M, k:k+BLK_K], dispatch="tma")
```

It still reads as "copy this A tile into SMEM," yet `dispatch="tma"` names the hardware path and the SMEM layout names how the tile must sit for TMA to lower it. A `tcgen05` MMA stays a tile GEMM but exposes the SMEM operand and TMEM accumulator layouts; a TMEM readback stays a tile copy but its source is TMEM, its destination a warpgroup register view, and its scope `Tx.warpgroup()`.

That is the reason for tile primitives: the kernel stays written in tile operations — the granularity the hardware works at — while every operation still carries the **scope, layout, dispatch**, and synchronization facts the compiler needs. The payoff runs through the rest of this tutorial: you will read a real production GEMM and a Flash Attention kernel, and understand each optimization as a change to one of those facts.

## Execution Scope
:label:`sec_exec_scope`

Scope answers the first question: who cooperates on this operation?

TIRX uses `with` blocks for the execution levels used by Blackwell kernels. In the snippets below, focus on the `with` line that names the scope; the tile names (`Asmem`, `tmem`, `Dreg_wg`) are just stand-ins, defined for real in the GEMM chapters:

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

The scopes are not a fixed template that every kernel must nest exactly. A kernel uses the scopes that match the operations it performs. `Tx.device_entry()` marks the start of device code; `Tx.cluster()` appears only when CTA clusters are used.

Some operations are issued by only part of a team. A TMA load, for example, may be issued by one selected thread:

```python
tid = Tx.thread_id_in_wg([128])
with Tx.thread(tid == 0):
    Tx.copy_async(Asmem[:, :], A[m:m + BLK_M, k:k + BLK_K], dispatch="tma")
```

Keep the simple rule: every tile primitive has a team, and sometimes one selected member issues the command on behalf of that team.

### Symbolic Coordinates

Scope tells you the team. Coordinates tell the code where that team sits in the launch. A CTA needs to know which output tile it owns. A thread needs its warp and lane id to decide which row or vector fragment it writes.

Many GEMM kernels start by naming the CTA, warpgroup, warp, and lane:

```python
bx, by = Tx.cta_id([M // BLK_M, N // BLK_N])
wg_id = Tx.warpgroup_id([1])      # single warpgroup, so wg_id is always 0 (unused below)
warp_id = Tx.warp_id_in_wg([4])
lane_id = Tx.lane_id([32])
```

Read it from the outside in. `Tx.cta_id(...)` gives the CTA's position in the launch grid, so the kernel can choose an output tile such as `D[bx * BLK_M, by * BLK_N]`. `Tx.warpgroup_id([1])` declares the warpgroup coordinate for a single-warpgroup kernel; later kernels use more warpgroups when producer, consumer, and writeback roles are split. `Tx.warp_id_in_wg([4])` and `Tx.lane_id([32])` identify a thread inside one warpgroup, which is enough to assign output rows during writeback.

The number inside brackets is the extent of that coordinate space. For example, `[4]` in `Tx.warp_id_in_wg([4])` says there are four warps in the warpgroup, and `[32]` in `Tx.lane_id([32])` says each warp has 32 lanes. TIRX uses these extents when checking layouts and tile primitives. A warpgroup register layout that uses `tid_in_wg` has extent 128, so it matches a 128-row TMEM readback; a lane-only layout has extent 32, so it would describe a different mapping.

Other coordinates appear when the kernel needs them. `Tx.thread_id([Nt])` names a thread inside a CTA. `Tx.thread_id_in_wg([128])` names a thread inside a warpgroup. Clustered kernels add `Tx.cluster_id(...)` and `Tx.cta_id_in_cluster(...)` to identify the cluster and the CTA's position inside it.


## Tensor Layout

Layout answers the next question: when the code says "this tile", where do the elements of that tile physically live?

At the math level, a tile is indexed by logical coordinates such as `A[m, k]` or `D[m, n]`. Hardware instructions do not operate on that notation directly. They need a physical view of the tile: bytes in shared memory, coordinates in TMEM, or values owned by particular threads.

A layout is that physical view. It maps logical tile indices to memory axes or thread axes. This is why layout is not decoration: get it wrong and the kernel either won't compile — a layout the TMA or `tcgen05.mma` path cannot lower is rejected — or, worse, it compiles and silently produces wrong output, the way a readback layout with the wrong extent hands each thread the wrong row. The right layout is what lets a TMA copy write the tile, lets `tcgen05.mma` read the operands correctly, and gives each thread the row it is meant to store.

The GEMM kernels use layouts in three recurring places: SMEM operand tiles, the TMEM accumulator tile, and the register view used during writeback.

First, the A and B operands live in SMEM before MMA:

```python
A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
```

Read this as: store each SMEM operand tile in a swizzled form that the TMA path and `tcgen05.mma` can both understand. *Swizzling* just reorders the bytes within the shared-memory tile so the engines that read it land on different shared-memory banks instead of colliding on the same one — it is a hardware-access layout, not a change to the tile's logical shape. The A tile has logical shape `(BLK_M, BLK_K)`, and the B tile has logical shape `(BLK_N, BLK_K)`. The physical shared-memory placement is chosen for hardware access rather than for a simple row-major picture.

Second, the MMA accumulator lives in TMEM:

```python
TileLayout(S[(128, N) : (1@TLane, 1@TCol)])
```

Unlike the swizzled SMEM layout above, which comes from the `tma_shared_layout` helper, TMEM and register layouts are written directly with the `S[...]` shard spec — but both produce the same kind of layout object. `S[shape : mapping]` is that *shard spec* (`S` comes from `from tvm.tirx.layout import S`). Despite the slice-like syntax, the colon is not slicing here: `S` reads the part before the colon as the tile `shape` and the part after it as the per-axis `mapping`. Read it as "a tile of this `shape`, with each logical index mapped the way the right side says." Here the left side `(128, N)` is the logical tile shape — `N` is the tile's column count, for example `BLK_N`. The mapping tuple lines up position by position with the shape tuple: `128` ↔ `1@TLane` and `N` ↔ `1@TCol`. So the row index maps to `TLane`, the column index maps to `TCol`, and logical element `(r, c)` lives at TMEM coordinate `(TLane=r, TCol=c)`.

Third, the epilogue reads the same logical accumulator tile into registers:

```python
TileLayout(S[(128, N) : (1@tid_in_wg, 1)])
```

This is not a single shared array. It is a distributed register view. Logical row `r` is owned by warpgroup thread `r`, and the columns for that row live in that thread's private registers. For a `(128, N)` readback, the 128 rows match the 128 threads in a warpgroup.

### Named Axes

The names after `@` are hardware axes:

- `TLane` is the TMEM row axis (extent 128).
- `TCol` is the TMEM column axis.
- `tid_in_wg` is the thread index inside a warpgroup, from 0 to 127.

The number before `@` is a **stride**, and the axis after `@` is where that stride applies. `1@TLane` means each increment of the logical row index advances one position along the `TLane` axis; in general `s@axis` advances `s` positions. A physical coordinate is then the per-axis sum of logical-index × stride — the same shape-stride model used for ordinary memory, where a row-major `(R, C)` tile is `S[(R, C) : (C, 1)]` with `addr(r, c) = r·C + c·1`. A bare number with no `@`, as in the second slot of `(1@tid_in_wg, 1)`, targets the default memory axis (`m`): an ordinary linear step through contiguous storage, which is why the bare form is used for flat GMEM, flat SMEM, and simple local register buffers. (This shape-stride notation is a simplified, row-major take on CuTe layouts.)

So these two layouts describe the same logical tile shape but two different physical views:

```python
TileLayout(S[(128, N) : (1@TLane, 1@TCol)])    # TMEM view
TileLayout(S[(128, N) : (1@tid_in_wg, 1)])
```

The first says where the accumulator lives after MMA. The second says which warpgroup thread receives each row during readback. Other axes, such as `laneid`, `warpid`, `cbx`, and `cby`, appear when the code needs warp-level layouts or cluster-aware layouts.

The full `TileLayout` model also has optional replica and offset pieces. In source terms, a layout contains shard iterators, replica iterators, and per-axis offsets. The GEMM path here mostly uses the simple shard form shown above; replica and offset are useful for more specialized layouts and are left to the reference.

### Layouts Enable Hardware Paths

Layouts tell TIRX which hardware path is legal for a primitive. A TMA load needs a GMEM source and a TMA-compatible SMEM destination. `tcgen05.mma` needs SMEM operand layouts that match the matrix descriptors it will use, and it writes the accumulator into a TMEM layout. `tcgen05.ld` reads that TMEM layout into a warpgroup register layout.

The TMEM-to-register readback is the easiest place to see the check. `TLane` and `tid_in_wg` both have extent 128, so a TMEM row can map naturally to one warpgroup thread. Mapping those TMEM rows to a 32-lane warp axis would describe a different operation, not the warpgroup readback expected by `tcgen05.ld`.


## Tile Primitive Dispatch
:label:`sec_tile_primitive_dispatch`

After source and destination layouts, look at dispatch. Dispatch answers the next question: **which hardware path should run the tile operation?**

Start with copy. In TIRX, `copy` means "move this tile or fragment." It does not name one fixed instruction. The hardware path comes from the local context:

```python
Tx.copy_async(Asmem[:, :], A[m:m + BLK_M, k:k + BLK_K], dispatch="tma")
```

This requests a TMA load from GMEM into SMEM. The `dispatch="tma"` part is not just a performance hint; it asks for the TMA hardware path. That means the GMEM slice, the SMEM destination, the SMEM layout, and the barrier protocol all have to match what TMA can legally do.

The same primitive can also describe a TMA store when the direction is reversed:

```python
Tx.copy_async(D[m:m + BLK_M, n:n + BLK_N], Dsmem[:, :], dispatch="tma")
```

Here the source is SMEM and the destination is GMEM, so the requested TMA path is SMEM -> GMEM.

TMEM readback is different. The code usually does not write `dispatch="tcgen05.ld"`:

```python
with Tx.warpgroup():
    Tx.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
```

The path is implied by the operands and scope: source in TMEM, destination in a warpgroup-distributed register view, and enclosing `Tx.warpgroup()`. That combination lowers to the Blackwell TMEM load path, `tcgen05.ld`.

MMA follows the same idea, but the operation is compute instead of copy:

```python
Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :], accum=False, dispatch="tcgen05")
```

Read this as: use Blackwell `tcgen05` MMA to multiply the two SMEM operand tiles and write the accumulator tile to TMEM. The operand layouts, accumulator layout, dtype, tile shape, and CTA-group setting must describe a legal `tcgen05` operation.

So when you read GEMM code, do not read `Tx.copy_async` or `Tx.gemm_async` in isolation. Check the surrounding scope, the source and destination layouts, and the dispatch argument if one is present. That local context tells you whether the primitive is asking for TMA, `tcgen05.ld`, `tcgen05.mma`, or ordinary load/store code.

**Try with your agent**: For the four primitives shown in this section, ask it to classify the hardware path: TMA load, TMA store, TMEM readback, or `tcgen05` MMA. For each one, name the source tile, destination tile, local context that determines the path, and one synchronization fact that is not fully shown in the snippet.


## Core APIs for Reading GEMM

The next chapter starts showing complete kernels. Scope, coordinates, layouts, and dispatch were covered above. This section only introduces the remaining names that you need for a first read.

Read these API names literally:

| API | How to read it |
|-----|----------------|
| `@Tx.prim_func` | Define one TIRX kernel function. |
| `Tx.Buffer(shape, dtype)` | Declare a typed device buffer argument. |
| `Tx.SMEMPool()` | Create a shared-memory allocation pool for this kernel. |
| `pool.alloc(shape, dtype, ...)` | Allocate one object from the shared-memory pool. |
| `pool.move_base_to(offset)` | Move the next shared-memory allocation to a fixed byte offset. |
| `pool.commit()` | Finish the shared-memory allocation plan. |
| `Tx.ptx.tcgen05.alloc(...)` | Ask Blackwell hardware for a TMEM allocation. |
| `Tx.decl_buffer(..., scope="tmem", allocated_addr=...)` | Create an indexable TIRX view of an existing TMEM allocation. |
| `Tx.alloc_local(shape, dtype)` | Allocate per-thread register storage. |
| `Tx.wg_reg_tile(elem_per_thread, dtype=...)` | Allocate a warpgroup-distributed `(128, elem_per_thread)` register tile for TMEM readback. |
| `Tx.cuda.cta_sync()` | Synchronize all threads in the CTA and order shared-memory writes. |
| `Tx.ptx.mbarrier.*` | Use raw mbarrier operations for async completion. |
| `Tx.ptx.tcgen05.*` | Use raw `tcgen05` helpers such as TMEM allocation, MMA commit, and wait. |
| `Tx.ptx.fence.*` | Add ordering across async/proxy boundaries. |
| `Tx.meta_var(expr)` | Keep an index expression inline in generated TIR. |

Two patterns are worth spelling out before the GEMM code.

First, shared memory is packed through `Tx.SMEMPool()`. The first allocations are usually small control objects:

```python
pool = Tx.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")
mma_bar = pool.alloc((1,), "uint64", align=8)
```

`tmem_addr` is not TMEM itself. It is a shared-memory slot where `tcgen05.alloc` will write the TMEM address. `mma_bar` is shared-memory storage for an mbarrier.

Then the kernel allocates the real operand tiles:

```python
pool.move_base_to(1024)
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
pool.commit()
```

`pool.move_base_to(1024)` leaves space for metadata at the start of shared memory. `Asmem` and `Bsmem` are the SMEM operand tiles. `pool.commit()` means no more pool allocations are added after this point.

Second, TMEM allocation and TMEM views are separate. This call allocates TMEM:

```python
Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)
```

This line does not allocate TMEM. It creates a typed view over the allocation address stored in `tmem_addr`:

```python
tmem = Tx.decl_buffer(
    (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
    layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
)
```

After that declaration, the code can write `tmem[:, :BLK_N]` instead of manually doing TMEM address arithmetic.

Register storage comes in two forms. The TMEM readback target must carry the warpgroup-distributed `(128, N) : (1@tid_in_wg, 1)` layout — that layout is the contract, not the spelling. The GEMM chapters obtain it two equivalent ways:

```python
# explicit: allocate flat, then view it with the distributed layout (Steps 1-6)
Dreg = Tx.alloc_local((BLK_N,), acc_type)
Dreg_wg = Dreg.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

# shorthand: the same (128, BLK_N) distributed tile in one call (later kernels)
Dreg_wg = Tx.wg_reg_tile(BLK_N)

Dreg_f16 = Tx.alloc_local((BLK_N,), d_type)
```

Either way, `Dreg_wg` is the warpgroup-distributed register tile that receives the fp32 accumulator read from TMEM via `tcgen05.ld`; its `(1@tid_in_wg, 1)` layout is exactly the readback view described above. What the hardware path checks is that layout, not whether you wrote `Tx.wg_reg_tile` or `Tx.alloc_local(...).view(...)`. `Dreg_f16` is per-thread local storage for the casted fp16 values that will be stored to output memory.

`Tx.meta_var(...)` appears often in address calculations:

```python
m_st = Tx.meta_var(bx * BLK_M)
n_st = Tx.meta_var(by * BLK_N)
```

Read it as an inline alias for an index expression. It is not storage, and it does not allocate anything.

When reading the GEMM code, use the same order each time: storage first, then scope, then tile primitive, then the wait or barrier that makes the next step safe.

## Exercises

1. For a TMEM-to-register readback, name its scope, its source and destination layouts, and its dispatch path. Why must it run under `Tx.warpgroup()` and not `Tx.thread()`?
2. The same `Tx.copy_async` call can lower to a TMA load, a TMA store, or a `tcgen05.ld` readback. What in the local context (source, destination, scope, dispatch) decides which?
3. `TileLayout(S[(128, N) : (1@TLane, 1@TCol)])` and `TileLayout(S[(128, N) : (1@tid_in_wg, 1)])` describe the same logical tile shape. What does each say about where the tile physically lives, and which step of the GEMM path uses each?
