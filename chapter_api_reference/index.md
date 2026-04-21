# TIRX API Reference
:label:`chap_api_reference`

This chapter provides a comprehensive reference for the TIRX APIs used in this tutorial.


## Thread Hierarchy

| API | Description |
|-----|-------------|
| `Tx.prim_func(tirx=True)` | Declare a TIRX primitive function |
| `Tx.kernel()` | Kernel execution scope |
| `with Tx.cta():` | Scope: all threads in the CTA execute this block |
| `with Tx.warpgroup():` | Scope: all 128 threads in the warpgroup execute this block |
| `with Tx.warp():` | Scope: all 32 threads in the warp execute this block |
| `with Tx.thread():` | Scope: each thread executes independently |
| `with Tx.thread(parent="warp")[cond]:` | Scope: only threads where `cond` is true execute |
| `Tx.cta_id(shape, parent=...)` | CTA index in grid or cluster |
| `Tx.warpgroup_id(shape, parent=...)` | Warpgroup index within CTA |
| `Tx.warp_id(shape, parent=...)` | Warp index within warpgroup |
| `Tx.thread_id(shape, parent=...)` | Thread index within the scope specified by `parent` (e.g., `parent="warp"` gives lane ID 0-31; `parent="cta"` gives CTA-level thread ID 0-127) |
| `Tx.ptx.elect_sync()` | Elect one thread in a warp (for single-thread dispatch) |


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
| `Tx.PoolAllocator()` | Shared memory pool allocator |
| `pool.alloc(shape, dtype, layout=...)` | Allocate buffer from pool |
| `pool.move_base_to(offset)` | Set next allocation offset (for overlapping buffers) |
| `pool.commit()` | Finalize all allocations |
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
| `Tx.copy(dst, src)` | Synchronous copy |
| `Tx.copy_async(dst, src, dispatch="tma", ...)` | TMA async copy (load or store) |
| `Tx.cast(dst, src)` | Element-wise type cast |
| `Tx.gemm_async(C, A, B, *, transA=False, transB=False, accum=False, dispatch="tcgen05", **kwargs)` | tcgen05 MMA. `accum` is keyword-only (default `False`); backend-specific options such as `cta_group=` flow through `**kwargs` into the op config. |


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
| `Tx.cuda.cta_sync()` | CTA-wide barrier (like `__syncthreads`) |
| `Tx.cuda.warpgroup_sync(barrier_id)` | Warpgroup-level barrier |
| `Tx.cuda.cluster_sync()` | Cluster-wide barrier |


## High-Level Abstractions

These abstractions are introduced in Step 7 (Warp Specialization) and used through Step 9.

| API | Description |
|-----|-------------|
| `TMABar(pool, depth, name)` | TMA barrier array (auto-arrive via byte counting) |
| `TCGen05Bar(pool, depth, name)` | tcgen05 barrier array (auto-arrive via commit) |
| `MBarrier(pool, depth, name)` | Manual mbarrier array (threads arrive explicitly) |
| `bar.init(count)` | Initialize barrier with expected arrival count |
| `bar.wait(stage, phase)` | Wait for barrier at given stage and phase |
| `TMABar.arrive(stage, tx_count)` | Arrive with expected transaction-byte count (TMA load). `tx_count` is the total bytes the TMA engine will deposit into SMEM for this stage. |
| `TCGen05Bar.arrive(stage, cta_group=, cta_mask=)` | Arrive via tcgen05 commit |
| `MBarrier.arrive(stage, cta_id=, pred=)` | Thread-level arrive |
| `bar.ptr_to([stage])` | Get pointer to barrier at given stage |
| `TMABar.remote_view(cta_id)` | Access another CTA's barrier (cross-CTA) |
| `PipelineState(name, depth)` | Manages pipeline stage index and phase |
| `PipelineState.init(is_producer=)` | Initialize phase tracking |
| `PipelineState.stage` / `.phase` | Current stage index and phase value |
| `PipelineState.move_to_next_stage()` | Advance to next pipeline stage |
| `ClusterPersistentScheduler2D(...)` | L2-friendly tile scheduler for persistent kernels |


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

*The constructs in this section are **not** part of the TIRX programming model* (see :numref:`chap_tirx_primer` for the three pillars). They are convenience — removing them changes how the code looks, not what it means. They show up in Part III because real kernels use them to stay readable.

### `Tx.meta_var`: compile-time alias

`Tx.meta_var(expr)` names a compile-time expression so it can be reused in buffer slicing without becoming a TIR dynamic variable:

```python
m_st = Tx.meta_var(bx * BLK_M)
n_st = Tx.meta_var(by * BLK_N)
Tx.copy(Asmem[:, :], A[m_st : m_st + BLK_M, k_st : k_st + BLK_K])
```

Any variable *assigned* inside `@Tx.prim_func` that is not wrapped with `meta_var` becomes a TIR dynamic variable. Constants such as `BLK_M`, `PIPE_DEPTH`, and `EPI_N` must stay Python constants defined outside the function.

### `@Tx.inline`: compile-time helper

`@Tx.inline` marks a helper as inlined at compile time. It can close over outer kernel variables, which is useful for packaging repeated load / MMA / writeback boilerplate:

```python
@Tx.inline
def tma_load(stage, k):
    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
        tma_bar.arrive(stage, tx_count=total_bytes)
        Tx.copy_async(Asmem[stage], A[m_st : m_st + BLK_M, k : k + BLK_K],
                      dispatch="tma", mbar=tma_bar.ptr_to([stage]))
```

There is no runtime function-call overhead and no new semantics — just a reusable code region.
