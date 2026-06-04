# TIRX API Reference
:label:`chap_api_reference`

This page is a lookup table for the APIs used in the tutorial. It follows the current TVM/TIRX style from the local `tvm` and `tirx-kernels` repositories.

## Kernel Entry and Coordinates

| API | Meaning |
|-----|---------|
| `@Tx.prim_func` | Define a TIRX primitive function. Use this when the function signature is already concrete. |
| `@Tx.jit(...)` | Define a constexpr-specialized TIRX function. Current reference kernels use this for shape-specialized kernels such as Flash Attention 4. |
| `Tx.device_entry()` | Mark the start of device code. Put thread coordinates and device allocations after this line. |
| `Tx.cta_id([Gx, Gy, ...])` | CTA index in the launch grid. The list gives the grid extent. |
| `Tx.cluster_id([...])` | Cluster index in the launch grid, when CTA clusters are used. |
| `Tx.cta_id_in_cluster([Cx, Cy])` | CTA coordinate inside a cluster. |
| `Tx.warpgroup_id([Nw])` | Warpgroup id inside a CTA. |
| `Tx.warp_id([Nw])` | Warp id inside a CTA. |
| `Tx.warp_id_in_wg([4])` | Warp id inside a warpgroup. A warpgroup has 4 warps. |
| `Tx.thread_id([Nt])` | Thread id inside a CTA. |
| `Tx.thread_id_in_wg([128])` | Thread id inside a warpgroup. |
| `Tx.lane_id([32])` | Lane id inside a warp. |

Coordinate APIs encode the parent level in the name, for example `cta_id_in_cluster` and `warp_id_in_wg`.

## Execution Scopes

| API | Meaning |
|-----|---------|
| `with Tx.cluster():` | Cluster-cooperative region. Used only by clustered kernels. |
| `with Tx.cta():` | CTA-cooperative region. |
| `with Tx.warpgroup():` | Warpgroup-cooperative region, 128 threads. |
| `with Tx.warp():` | Warp-cooperative region, 32 threads. |
| `with Tx.thread():` | Per-thread region. |
| `with Tx.thread(cond):` | Per-thread region narrowed by a predicate. |
| `Tx.filter(var, pred)` | Predicate helper when a scope needs to select active lanes from a symbolic variable. |

Example:

```python
tid = Tx.thread_id_in_wg([128])
with Tx.thread(tid == 0):
    Tx.copy_async(Asmem[:, :], A_tile, dispatch="tma")
```

`Tx.ptx.elect_sync()` elects one participating thread per warp. It is useful for warp-level issue paths, but it is not the same as `tid == 0`.

## Buffers and Storage

| API | Meaning |
|-----|---------|
| `Tx.Buffer(shape, dtype)` | Typed kernel argument in a concrete signature. |
| `Tx.match_buffer(ptr, shape, dtype)` | Bind an opaque pointer argument to a typed buffer view. |
| `Tx.alloc_local(shape, dtype)` | Per-thread register/local storage. |
| `Tx.wg_reg_tile(elem_per_thread, dtype=...)` | Warpgroup-distributed `(128, elem_per_thread)` register tile for TMEM readback; equivalent to `Tx.alloc_local(...).view(..., S[(128, N) : (1@tid_in_wg, 1)])`. |
| `Tx.SMEMPool()` | Shared-memory allocation pool. |
| `pool.alloc(shape, dtype, layout=..., align=...)` | Allocate shared-memory storage from the pool. |
| `pool.alloc_mma(shape, dtype)` | Allocate shared-memory storage using the MMA-friendly helper used by current kernels. |
| `pool.move_base_to(offset)` | Advance the pool base before later allocations. |
| `pool.commit()` | Close the shared-memory allocation plan. |
| `Tx.decl_buffer(..., scope="tmem", allocated_addr=...)` | Create an indexable view over a TMEM allocation. It does not allocate TMEM by itself. |
| `Tx.TMEMPool(pool, total_cols=..., cta_group=..., tmem_addr=...)` | Manage a TMEM allocation through a pool object. |
| `tmem_pool.alloc(shape, dtype, layout=..., cols=..., datapath=...)` | Allocate a TMEM view from the pool. |
| `tmem_pool.commit()` | Commit the TMEM allocation. |
| `tmem_pool.dealloc()` | Release the TMEM allocation. |
| `Tx.TMEMStages(tmem, col_start=..., width=..., stages=..., stride=...)` | Describe repeated TMEM stage regions inside one allocation. |

The common manual pattern is:

```python
tmem_addr = pool.alloc([1], "uint32")
Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)
tmem = Tx.decl_buffer(
    (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
    layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
)
```

Current higher-level kernels usually use `Tx.TMEMPool(...)`, then call `commit()` and `dealloc()`.

## Layouts

| API | Meaning |
|-----|---------|
| `TileLayout(S[(shape) : (mapping)])` | Map logical tile indices to memory or thread axes. |
| `tma_shared_layout(dtype, swizzle, shape)` | Shared-memory layout compatible with TMA and `tcgen05.mma` descriptors. |
| `TLane`, `TCol` | TMEM row and column axes. |
| `tid_in_wg` | Warpgroup thread axis, extent 128. |
| `laneid`, `warpid` | Warp-level axes used in lower-level layouts. |
| `cbx`, `cby` | CTA-in-cluster axes used by clustered layouts. |

Common GEMM layouts:

```python
# TMEM accumulator.
TileLayout(S[(128, N) : (1@TLane, 1@TCol)])

# Warpgroup-distributed register view.
TileLayout(S[(128, N) : (1@tid_in_wg, 1)])
```

## Tile Primitives

| API | Meaning |
|-----|---------|
| `Tx.copy(dst, src, ...)` | Synchronous copy/store. Lowering depends on scope and layouts. |
| `Tx.copy_async(dst, src, dispatch="tma", ...)` | Request a TMA tile load or store. The SMEM side must use a compatible layout. |
| `Tx.copy_async(reg_view, tmem_view)` | TMEM-to-register readback under warpgroup scope. Follow with `Tx.ptx.tcgen05.wait.ld()`. |
| `Tx.copy_async(tmem_view, reg_view)` | Register-to-TMEM writeback used by Flash Attention softmax/correction paths. Follow with `Tx.ptx.tcgen05.wait.st()` before a later consumer depends on the store. |
| `Tx.gemm_async(dst, a, b, dispatch="tcgen05", ...)` | Blackwell Tensor Core MMA tile operation. |
| `Tx.cast(dst, src)` | Cast a register/local fragment, commonly fp32 accumulator values to fp16 output values. |

Read a primitive with its local context: source, destination, scope, layouts, dispatch, and the wait or barrier that makes the result safe to consume.

## Synchronization and Hardware Helpers

| API | Meaning |
|-----|---------|
| `Tx.cuda.cta_sync()` | CTA-wide barrier and shared-memory ordering point. |
| `Tx.cuda.cluster_sync()` | Cluster-wide synchronization. |
| `Tx.cuda.warpgroup_sync(id)` | Synchronize only the current warpgroup using a named barrier id. |
| `Tx.ptx.mbarrier.init(ptr, count)` | Initialize an mbarrier. |
| `Tx.ptx.mbarrier.arrive.expect_tx(ptr, bytes)` | Tell a TMA load barrier how many bytes to expect. |
| `Tx.ptx.mbarrier.try_wait(ptr, phase)` | Wait until the barrier reaches the expected phase. |
| `Tx.ptx.tcgen05.commit(bar, cta_group=...)` | Commit issued MMA work to a completion barrier. |
| `Tx.ptx.tcgen05.wait.ld()` | Wait for TMEM-to-register readback. |
| `Tx.ptx.tcgen05.wait.st()` | Wait for register-to-TMEM stores. |
| `Tx.ptx.cp_async.bulk.commit_group()` | Commit a TMA store group. |
| `Tx.ptx.cp_async.bulk.wait_group(n)` | Wait until enough TMA store groups have drained. |
| `Tx.ptx.fence.proxy_async(...)` | Ordering fence across proxy boundaries, often before async engines read thread-written SMEM. |
| `Tx.ptx.fence.mbarrier_init()` | Make initialized mbarriers visible before later use. |

## Pipeline Helpers

| API | Meaning |
|-----|---------|
| `PipelineState(depth, phase=None)` | Track the current stage and phase for a ring buffer. |
| `state.stage` / `state.phase` | Current stage index and phase. |
| `state.init(phase)` | Initialize phase tracking. |
| `state.advance()` | Advance to the next stage and flip phase when the stage wraps. |
| `Pipeline(pool, stages, full=..., empty=..., init_full=..., init_empty=..., empty_phase_offset=...)` | Allocate matching full/empty barrier arrays for a software pipeline. |
| `pipe.full` / `pipe.empty` | Barrier endpoints used by producer and consumer roles. |
| `MBarrier(pool, depth)` | Explicit thread-arrival barrier wrapper. |
| `TMABar(pool, depth)` | TMA byte-counting barrier wrapper. |
| `TCGen05Bar(pool, depth)` | MMA completion barrier wrapper, signaled by `tcgen05.commit`. |

These five helpers are *not* on the `Tx` namespace (unlike `Tx.SMEMPool`/`Tx.TMEMPool`). Import them explicitly from `tvm.tirx.lang.pipeline`:

```python
from tvm.tirx.lang.pipeline import PipelineState, Pipeline, MBarrier, TMABar, TCGen05Bar

kv_state = PipelineState(SMEM_PIPE_DEPTH_KV)
kv_load = Pipeline(pool, SMEM_PIPE_DEPTH_KV, full="tma", empty="tcgen05", empty_phase_offset=1)
```

The state tracks which stage is current; the pipeline object owns the barriers that protect each stage.

## Compile and Inspect

```python
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

cuda_source = ex.mod.imports[0].inspect_source("cuda")
print(cuda_source)
```

Use generated CUDA to verify that scope guards and hardware instructions match the intended contract.

## Common Footguns

- Use `@Tx.jit(...)` for constexpr-specialized kernels; plain `@Tx.prim_func` otherwise.
- `Tx.filter` does not have a range form. Pass a variable and a predicate.
- `Tx.ptx.elect_sync()` elects one thread per warp, not one thread per CTA or warpgroup.
- `Tx.decl_buffer(..., scope="tmem")` creates a view. TMEM must already have been allocated by `tcgen05.alloc` or `Tx.TMEMPool`.
- TMA stores use commit/wait-group completion, not the TMA-load mbarrier byte-counting protocol.
- An async copy or MMA is not safe to consume until the matching wait or barrier has completed.
- `tcgen05.commit` / `TCGen05Bar.arrive` must be guarded so only the intended MMA issuer calls it. Empty commit groups from non-issuer lanes can signal a barrier too early.
