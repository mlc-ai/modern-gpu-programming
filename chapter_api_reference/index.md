# TIRX API Reference
:label:`chap_api_reference`

This chapter provides a comprehensive reference for the TIRX APIs used in this tutorial.


## Thread Hierarchy

| API | Description |
|-----|-------------|
| `@Tx.prim_func` | Declare a TIRX primitive function |
| `Tx.kernel()` | Kernel execution scope |
| `Tx.cluster()` | Cluster execution scope (clustered launches only) |
| `with Tx.cta():` | Scope: all threads in the CTA execute this block |
| `with Tx.warpgroup():` | Scope: all 128 threads in the warpgroup execute this block |
| `with Tx.warp():` | Scope: all 32 threads in the warp execute this block |
| `with Tx.thread():` | Scope: each thread executes independently |
| `with Tx.thread(cond):` | Scope: only threads where `cond` is true execute |
| `Tx.filter(var, lo, hi)` / `Tx.filter(var, cond)` | Active-thread filter — narrow the current scope to `var ∈ [lo, hi)` or `cond` true. Use as the scope argument: `with Tx.warp(Tx.filter(warp_id, 0, 1)):` |
| `Tx.cta_id([Gx, Gy, ...])` | CTA index in the grid (kernel → cta) |
| `Tx.cta_id_in_cluster([Cx, Cy])` | CTA index within the cluster (cluster → cta) |
| `Tx.warpgroup_id([Nw])` | Warpgroup index within the CTA (cta → warpgroup) |
| `Tx.warp_id([Nw])` | Warp index within the CTA (cta → warp) |
| `Tx.warp_id_in_wg([4])` | Warp index within the warpgroup (warpgroup → warp; always 4) |
| `Tx.thread_id([Nt])` | Thread index within the CTA (cta → thread) |
| `Tx.thread_id_in_wg([128])` | Thread index within the warpgroup (warpgroup → thread; always 128) |
| `Tx.lane_id([32])` | Thread index within a warp (warp → thread; always 32) |
| `Tx.cuda.thread_rank()` | Linear thread index within the CTA (i.e., `threadIdx.x + threadIdx.y * blockDim.x + ...`). Used as the default `leader` predicate for one-thread-per-CTA work such as `MBarrier.init`. |
| `Tx.ptx.elect_sync()` | Elect one thread *per warp* (so 4 threads in a 4-warp warpgroup). Good for `tcgen05.commit` and TMA stores; for TMA *loads* prefer `tid_in_wg == 0` so `expect_tx` is incremented exactly once. |


## Function Parameters

| API | Description |
|-----|-------------|
| `Tx.Buffer(shape, dtype)` | Buffer parameter with shape known at compile time. Use when dimensions are Python constants (e.g., GEMM with fixed M, N, K). |
| `Tx.handle` | Opaque buffer parameter with shape determined at runtime. Pair with `Tx.match_buffer` inside the kernel to bind the actual shape. |
| `Tx.match_buffer(handle, shape, dtype)` | Bind a `Tx.handle` parameter to a typed buffer with a specific shape. Shape dimensions can include `Tx.int32()` symbolic variables for runtime values. |
| `Tx.int32()` | Declare a symbolic integer dimension. The actual value is inferred at runtime from the buffer shape. |


## Memory

| API | Description |
|-----|-------------|
| `Tx.SMEMPool()` | Shared memory pool allocator |
| `pool.alloc(shape, dtype, layout=...)` | Allocate buffer from pool |
| `pool.move_base_to(offset)` | Set next allocation offset (for overlapping buffers) |
| `pool.commit()` | Finalize all allocations |
| `Tx.TMEMPool(smem_pool, total_cols, cta_group, tmem_addr)` | TMEM pool. Wraps one `tcgen05.alloc` of `total_cols` columns whose returned base address is written to `tmem_addr[0]`; subsequent `alloc(shape, dtype, layout=...)` calls hand out `Tx.decl_buffer(scope="tmem", allocated_addr=...)` slices into that region. |
| `Tx.alloc_local(shape, dtype)` | Allocate per-thread register buffer |
| `buf.view(shape, layout=...)` | Create a view of a register buffer with a different layout |
| `Tx.decl_buffer(shape, dtype, scope="tmem", ...)` | Declare a TMEM buffer |
| `Tx.address_of(buf)` | Get address of a buffer (used for TMEM alloc) |
| `buf.ptr_to([idx])` | Get pointer to the idx-th element (used for mbarrier access) |
| `tma_shared_layout(dtype, SwizzleMode, shape)` | Create TMA-compatible swizzled layout for SMEM |
| `Tx.ptx.tcgen05.alloc(addr, n_cols, cta_group)` | Allocate TMEM |
| `Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group)` | Release TMEM allocation permit |
| `Tx.ptx.tcgen05.dealloc(addr, n_cols, cta_group)` | Deallocate TMEM |


## Data Movement

| API | Description |
|-----|-------------|
| `Tx.copy(dst, src)` | Synchronous copy (LDG / STG / LDS / STS variants) |
| `Tx.copy_async(dst, src, dispatch="tma", ...)` | TMA async copy (load or store) |
| `Tx.copy_async(reg, tmem)` / `Tx.copy_async(tmem, reg)` | `tcgen05.ld` / `tcgen05.st` (warpgroup scope); follow with `Tx.ptx.tcgen05.wait.ld()` / `wait.st()` |
| `Tx.cast(dst, src)` | Element-wise type cast |
| `Tx.gemm_async(C, A, B, *, transA=False, transB=False, accum=False, dispatch="tcgen05", **kwargs)` | tcgen05 MMA. `accum` is keyword-only (default `False`); backend-specific options such as `cta_group=` flow through `**kwargs` into the op config. |


## Math

| API | Description |
|-----|-------------|
| `Tx.tanh(x)` | Element-wise tanh; preserves dtype of `x` |
| `Tx.rsqrt(x)` | Element-wise reciprocal square root, `1 / sqrt(x)`; preserves dtype of `x` |
| `Tx.exp(x)` / `Tx.log(x)` / `Tx.sqrt(x)` | Standard transcendentals lowered to their fast CUDA intrinsics |


## Control Flow

| API | Description |
|-----|-------------|
| `for i in Tx.unroll(N):` | Explicit unrolled loop with `i` usable for buffer slicing |
| `for i in Tx.serial(N):` | Sequential loop (not unrolled), `i` is a TIR variable |

For `Tx.meta_var` and `@Tx.inline`, see the *Syntactic Sugar* section at the end of this chapter — they are not part of the programming model.


## Synchronization

| API | Description |
|-----|-------------|
| `Tx.ptx.mbarrier.init(ptr, count)` | Initialize mbarrier with expected arrival count |
| `Tx.ptx.mbarrier.try_wait(ptr, phase)` | Wait for mbarrier phase |
| `Tx.ptx.mbarrier.arrive.expect_tx(ptr, bytes)` | Set expected TMA byte count |
| `Tx.ptx.tcgen05.commit(ptr, cta_group, cta_mask)` | tcgen05 commit (auto-arrive on completion) |
| `Tx.ptx.tcgen05.fence.after_thread_sync()` | Epilogue-only fence: order prior MMA writes to TMEM against subsequent `tcgen05.ld` reads |
| `Tx.ptx.fence.proxy_async("shared::cta")` | Shared memory fence |
| `Tx.ptx.fence.mbarrier_init()` | Fence after mbarrier initialization |
| `Tx.ptx.cp_async.bulk.commit_group()` | Commit pending TMA store operations |
| `Tx.ptx.cp_async.bulk.wait_group(n)` | Wait until at most `n` TMA store groups remain |
| `Tx.ptx.bar.sync(barrier_id, thread_count)` | Named-barrier `__syncthreads`. Synchronizes exactly `thread_count` threads on barrier `barrier_id` (0–15). Useful when only a subset of the CTA participates — e.g., a `bdx * bdy` grid that does not fill 128/256 threads — where plain `cta_sync()` would hang. |
| `Tx.cuda.cta_sync()` | CTA-wide barrier (like `__syncthreads`) |
| `Tx.cuda.warpgroup_sync(barrier_id)` | Warpgroup-level barrier |
| `Tx.cuda.cluster_sync()` | Cluster-wide barrier |


## High-Level Abstractions

These abstractions are introduced in Step 7 (Warp Specialization) and used through Step 9.

| API | Description |
|-----|-------------|
| `TMABar(pool, depth)` | Barrier signaled by TMA (`mbarrier.arrive.expect_tx`) |
| `TCGen05Bar(pool, depth)` | Barrier signaled by `tcgen05.commit` |
| `MBarrier(pool, depth, phase_offset=0, leader=None)` | Plain mbarrier — threads arrive explicitly |
| `bar.init(count)` | Initialize barrier with expected arrival count |
| `bar.wait(stage, phase)` | Wait for barrier at given stage and phase |
| `TMABar.arrive(stage, tx_count=, cta_id=, pred=)` | Arrive with expected transaction-byte count. `tx_count` is the total bytes TMA will deposit for this stage. |
| `TCGen05Bar.arrive(stage, cta_group=, cta_mask=)` | Arrive via `tcgen05.commit`. Must run under `elect_sync` (commit is per-thread). |
| `MBarrier.arrive(stage, cta_id=, pred=)` | Thread-level arrive |
| `bar.ptr_to([stage])` | Get pointer to barrier at given stage |
| `MBarrier.remote_view(rank)` / `TMABar.remote_view(rank)` | View a barrier mapped to another CTA's SMEM (cluster kernels) |
| `Pipe(pool, stages, full_type=, empty_type=, init_full=, init_empty=, ...)` | Full+empty barrier pair for a software-pipelined data flow |
| `Pipe.tma(pool, stages, ...)` / `Pipe.tcgen05(...)` / `Pipe.mbar(...)` | Factory shortcuts for the three common pipe topologies |
| `pipe.producer()` / `pipe.consumer()` | Return a `_PipeEndpoint` with `.wait()` / `.signal()` / `.advance()` |
| `RingState(depth, phase=)` | Tracks stage index and phase for a ring buffer (no barrier inside) |
| `RingState.init(phase)` | Initialize phase tracking (1 for producer, 0 for consumer) |
| `RingState.stage` / `.phase` | Current stage index and phase value |
| `RingState.advance()` | Advance to next stage; flip phase when wrapping |


## Tile Scheduling

| API | Description |
|-----|-------------|
| `ClusterPersistentScheduler2D(name, num_m_tiles, num_n_tiles, l2_group_size=, num_clusters=)` | L2-friendly persistent scheduler — assigns `(m_idx, n_idx)` blocks round-robin across `num_clusters` CTAs. |
| `scheduler.init(cta_id)` | Bind this CTA's starting tile |
| `scheduler.valid()` | True while this CTA still has tiles |
| `scheduler.m_idx` / `scheduler.n_idx` / `scheduler.tile_idx` | Current tile coordinates (mutable local scalars) |
| `scheduler.next_tile()` | Advance to this CTA's next tile |
| `FlashAttentionLinearScheduler(...)` | Per-task linear schedule for non-causal FA |
| `FlashAttentionLPTScheduler(...)` | Longest-processing-time-first for causal FA |


## Layout

The layout namespace is `tvm.tirx.layout` (also re-exported on `Tx` for the common helpers used in this tutorial).

| API | Description |
|-----|-------------|
| `TileLayout(spec)` | Build a layout from `S[...]` / `R[...]` / `O[...]` segments |
| `S[(extents...) : (strides...)]` | Sharded segment — distribute extents across the axes given by the stride list |
| `R[N : axis]` | Replica — broadcast across `N` lanes of `axis` |
| `O[offset]` | Offset segment (rare; used inside `TileLayout` to skip leading bytes) |
| `TLane`, `TCol` | Named TMEM hardware axes (rows × columns of tensor memory) |
| `tid_in_wg`, `laneid`, `warpid` | Named axes for the warpgroup-level thread coordinate, the in-warp lane, and the warp-in-CTA index |
| `cbx`, `cby` | Named cluster axes (cluster block x / y) |
| `wg_local_layout(cols, rows=128)` | Shorthand for `TileLayout(S[(rows, cols) : (1@tid_in_wg, 1)])` — a warpgroup-wide RF tile |
| `tma_shared_layout(dtype, swizzle_mode, shape)` | Produce a TMA-compatible swizzled SMEM layout. Wrapper around `tma_utils`. |
| `SwizzleMode.SWIZZLE_128B_ATOM` / `SWIZZLE_64B_ATOM` / etc. | TMA swizzle modes for SMEM atoms |


## Common Pitfalls

- **`accum` must be boolean-compatible**: Use `False` (not `0`) for the first MMA iteration.

- **Fence API**: Use `Tx.ptx.fence.proxy_async("shared::cta")` — positional argument, not keyword `scope=`.

- **Do NOT call `cta_sync()` inside an elected-thread scope** — only one thread is executing, `cta_sync()` requires all threads.

- **`alloc_local` vs `decl_buffer`**: Use `Tx.alloc_local` for register buffers. `Tx.decl_buffer` is only for hardware-managed memory like TMEM.

- **TMA store must be followed by `commit_group()` + `wait_group(0)`**: Without waiting, the next iteration may overwrite Dsmem.

- **`fence.after_thread_sync()` is rarely needed**: The MMA-completion mbarrier already carries release→acquire semantics, so most kernels — including the early Steps in this tutorial and most of CUTLASS — omit it entirely. The `tirx-kernels` reference kernels take a more conservative stance and insert it in the writeback path, after `mma2ld.wait(phase)` and immediately before the first `tcgen05.ld`, to make the ordering between prior tcgen05 writes to TMEM and the upcoming thread-proxy reads explicit. It is **never** needed on the TMA→MMA edge — the TMA-completion mbarrier and `cta_sync()` after a synchronous SMEM copy already cover that case.

- **Constants must be defined outside `@Tx.prim_func`**: Variables like `EPI_N`, `TMEM_LD_N`, `MMA_N` must be Python constants.

- **`pool.move_base_to(1024)`**: Reserves the first 1024 bytes for metadata (TMEM address, barriers). All data buffers (Asmem, Bsmem, Dsmem) are allocated sequentially after this offset.

- **GPU flakiness**: If tests fail intermittently, check `nvidia-smi` and switch to an idle GPU.


## Syntactic Sugar

*The constructs in this section are **not** part of the TIRX tile primitive layer* (see :numref:`chap_layouts` for execution scope, tensor layout, and tile primitive dispatch). They are parser-time convenience — removing them changes how the code looks, not what it means. They show up in Part III because real kernels use them to stay readable. A brief introduction to the metaprogramming trio lives in :numref:`chap_tirx_primer`; this section is the full reference.

### `Tx.meta_var`: compile-time alias

`Tx.meta_var(expr)` names a compile-time expression so it can be reused in buffer slicing without becoming a TIR dynamic variable:

```python
m_st = Tx.meta_var(bx * BLK_M)
n_st = Tx.meta_var(by * BLK_N)
Tx.copy(Asmem[:, :], A[m_st : m_st + BLK_M, k_st : k_st + BLK_K])
```

Any variable *assigned* inside `@Tx.prim_func` that is not wrapped with `meta_var` becomes a TIR dynamic variable. Constants such as `BLK_M`, `PIPE_DEPTH`, and `EPI_N` must stay Python constants defined outside the function.

### `@Tx.inline`: compile-time helper

`@Tx.inline` marks a helper as inlined at compile time. It can close over outer kernel variables, which is useful for packaging repeated load / MMA / writeback boilerplate. The conventional order in a TMA load is **issue the copies, then arrive with the byte count** — the TMA engine increments the barrier as bytes land, and `expect_tx` only sets the target count:

```python
@Tx.inline
def tma_load(stage, k):
    with Tx.thread(Tx.filter(lane_id, Tx.ptx.elect_sync())):
        Tx.copy_async(Asmem[stage], A[m_st : m_st + BLK_M, k : k + BLK_K],
                      dispatch="tma", mbar=tma_bar.ptr_to([stage]))
        Tx.copy_async(Bsmem[stage], B[n_st : n_st + BLK_N, k : k + BLK_K],
                      dispatch="tma", mbar=tma_bar.ptr_to([stage]))
        tma_bar.arrive(stage, tx_count=total_bytes)
```

There is no runtime function-call overhead and no new semantics — just a reusable code region.
