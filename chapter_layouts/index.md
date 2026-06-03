# Tile Primitive Mental Model
:label:`chap_layouts`

TIRX code is built around tile primitives. Before looking at a full GEMM kernel, it helps to know what one primitive means.

A **tile primitive** is one operation on a tile: copy this tile, multiply these tiles, read this accumulator tile, or store this output tile. It is smaller than a whole operator like GEMM, but larger than a scalar instruction.

The function name is only part of the meaning. For example, `Tx.copy_async(...)` is not one fixed instruction. Depending on the source, destination, scope, and dispatch, it can mean a TMA load, a TMA store, or a `tcgen05.ld` readback.

When reading a TIRX primitive, check the local contract around the call:

```text
scope    -> which threads cooperate on the tile operation
layout   -> how the logical tile maps to memory axes or thread axes
dispatch -> which hardware path implements the operation
```

For asynchronous primitives, a barrier, commit, wait, or fence records the handoff to the next operation.


## Why Tile Primitives?
:label:`sec_no_sugar_gemm`

Use one GEMM stage as the running example. A stage moves operand tiles into shared memory, runs Tensor Core MMA on those tiles, reads accumulator tiles back, and stores the result. Attention and convolution kernels use different math, but they still rely on the same kind of tile-level structure.

The math alone is not enough. For each tile operation, a Blackwell kernel also has to answer:

- Who cooperates on the operation?
- Where does the tile live?
- How is the tile laid out in that memory space?
- Which hardware path should run it: TMA, `tcgen05.mma`, `tcgen05.ld`, or ordinary loads and stores?
- For an asynchronous operation, what tells the next step that the result is ready?

Raw PTX-level code answers those questions with descriptor fields, elected-thread guards, mbarrier phases, TMEM addresses, and wait calls. Everything is explicit, but the tile operation is hard to see.

A higher-level DSL may make the kernel shorter, but Blackwell choices such as TMEM readback, TMA multicast, or 2-CTA cooperative MMA may be hidden behind a scheduler, template, or library call.

TIRX is meant to keep the tile operation readable without hiding the hardware contract. A TMA load is still a tile copy, but the code says `dispatch="tma"` and uses layouts that TMA can lower. A `tcgen05` MMA is still a tile GEMM, but the code exposes the SMEM operand layouts and the TMEM accumulator layout. A TMEM readback is still a tile copy, but the source is TMEM, the destination is a warpgroup register view, and the surrounding scope is `Tx.warpgroup()`.

That is the reason for tile primitives: keep the kernel written in tile operations while still giving the compiler the scope, layout, dispatch, and synchronization facts it needs.

## Execution Scope
:label:`sec_exec_scope`

Scope answers the first question: who cooperates on this operation?

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
_ = Tx.warpgroup_id([1])
warp_id = Tx.warp_id_in_wg([4])
lane_id = Tx.lane_id([32])
```

Read it from the outside in. `Tx.cta_id(...)` gives the CTA's position in the launch grid, so the kernel can choose an output tile such as `D[bx * BLK_M, by * BLK_N]`. `Tx.warpgroup_id([1])` declares the warpgroup coordinate for a single-warpgroup kernel; later kernels use more warpgroups when producer, consumer, and writeback roles are split. `Tx.warp_id_in_wg([4])` and `Tx.lane_id([32])` identify a thread inside one warpgroup, which is enough to assign output rows during writeback.

The number inside brackets is the extent of that coordinate space. For example, `[4]` in `Tx.warp_id_in_wg([4])` says there are four warps in the warpgroup, and `[32]` in `Tx.lane_id([32])` says each warp has 32 lanes. TIRX uses these extents when checking layouts and tile primitives. A warpgroup register layout that uses `tid_in_wg` has extent 128, so it matches a 128-row TMEM readback; a lane-only layout has extent 32, so it would describe a different mapping.

Other coordinates appear when the kernel needs them. `Tx.thread_id([Nt])` names a thread inside a CTA. `Tx.thread_id_in_wg([128])` names a thread inside a warpgroup. Clustered kernels add `Tx.cluster_id(...)` and `Tx.cta_id_in_cluster(...)` to identify the cluster and the CTA's position inside it.


## Tensor Layout

Layout answers the next question: when the code says "this tile", where do the elements of that tile physically live?

At the math level, a tile is indexed by logical coordinates such as `A[m, k]` or `D[m, n]`. Hardware instructions do not operate on that notation directly. They need a physical view of the tile: bytes in shared memory, coordinates in TMEM, or values owned by particular threads.

A layout is that physical view. It maps logical tile indices to memory axes or thread axes. This is why layout is not decoration: it decides whether a TMA copy can write the tile, whether `tcgen05.mma` can read the operands correctly, and whether a TMEM readback gives each thread the row it is supposed to store.

The GEMM kernels use layouts in three recurring places: SMEM operand tiles, the TMEM accumulator tile, and the register view used during writeback.

First, the A and B operands live in SMEM before MMA:

```python
A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_M, BLK_K))
B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM, (BLK_N, BLK_K))
```

Read this as: store each SMEM operand tile in a swizzled form that the TMA path and `tcgen05.mma` can both understand. The A tile has logical shape `(BLK_M, BLK_K)`, and the B tile has logical shape `(BLK_N, BLK_K)`. The physical shared-memory placement is chosen for hardware access rather than for a simple row-major picture.

Second, the MMA accumulator lives in TMEM:

```python
TileLayout(S[(128, N) : (1@TLane, 1@TCol)])
```

Read the left side, `(128, N)`, as the logical tile shape. Read the right side as the physical mapping: the row index maps to `TLane`, and the column index maps to `TCol`. In other words, logical element `(r, c)` lives at TMEM coordinate `(TLane=r, TCol=c)`.

Third, the epilogue reads the same logical accumulator tile into registers:

```python
TileLayout(S[(128, N) : (1@tid_in_wg, 1)])
```

This is not a single shared array. It is a distributed register view. Logical row `r` is owned by warpgroup thread `r`, and the columns for that row live in that thread's private registers. For a `(128, N)` readback, the 128 rows match the 128 threads in a warpgroup.

### Named Axes

The names after `@` are hardware axes:

- `TLane` is the TMEM row axis.
- `TCol` is the TMEM column axis.
- `tid_in_wg` is the thread index inside a warpgroup, from 0 to 127.

The notation `1@TLane` means that increasing the corresponding logical index by one advances by one step along the `TLane` axis. A bare `1` means the ordinary linear memory axis, written explicitly as `1@m`. That is used for flat GMEM, flat SMEM, and simple local register buffers.

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
Tx.gemm_async(tmem, Asmem, Bsmem, accum=False, dispatch="tcgen05")
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

Register storage is local to each thread:

```python
Dreg = Tx.alloc_local((BLK_N,), acc_type)
Dreg_f16 = Tx.alloc_local((BLK_N,), d_type)
```

In GEMM writeback, `Dreg` holds fp32 values read from TMEM. `Dreg_f16` holds the casted fp16 values that will be stored to output memory.

`Tx.meta_var(...)` appears often in address calculations:

```python
m_st = Tx.meta_var(bx * BLK_M)
n_st = Tx.meta_var(by * BLK_N)
```

Read it as an inline alias for an index expression. It is not storage, and it does not allocate anything.

When reading the GEMM code, use the same order each time: storage first, then scope, then tile primitive, then the wait or barrier that makes the next step safe.
