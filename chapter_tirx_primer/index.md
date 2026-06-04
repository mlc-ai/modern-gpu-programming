# TIRX Language and Compile Pipeline
:label:`chap_tirx_primer`

This appendix explains how TIRX source becomes generated CUDA. Use it when a tutorial kernel contains an unfamiliar parser feature, buffer form, or generated-code check.

## What TIRX Source Represents

A TIRX function is Python syntax that builds TVM IR. The Python code is not executed as a normal GPU program. The parser reads the function body and turns supported constructs into IR nodes: buffers, coordinates, scopes, allocations, loops, conditions, layouts, and tile primitives.

The current entry pattern is:

```python
@Tx.prim_func
def kernel(a_ptr: Tx.handle, b_ptr: Tx.handle):
    n = Tx.int32()
    A = Tx.match_buffer(a_ptr, [n], "float32")
    B = Tx.match_buffer(b_ptr, [n], "float32")

    Tx.device_entry()
    bx = Tx.cta_id([(n + 255) // 256])
    tid = Tx.thread_id([256])

    with Tx.thread():
        i = bx * 256 + tid
        if i < n:
            B[i] = A[i] * 2.0
```

Read the structure as:

1. Host-visible arguments and buffer binding.
2. `Tx.device_entry()` starts device-side IR.
3. Coordinate calls name the launch and thread positions.
4. Scope blocks tell TIRX which execution team owns the operation.
5. Primitive calls describe tile movement, compute, or scalar/register work.

## `@Tx.prim_func` and `@Tx.jit`

Use `@Tx.prim_func` when the kernel shape is already concrete or when dynamic arguments are represented with `Tx.int32()` and `Tx.match_buffer`.

Use `@Tx.jit(...)` when the kernel has compile-time parameters. Current `tirx-kernels` reference kernels use `@Tx.jit(persistent=True)` for shape-specialized persistent kernels:

```python
@Tx.jit(persistent=True)
def _kernel(
    Q: Tx.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
    O: Tx.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
    *,
    BATCH_SIZE: Tx.constexpr,
    SEQ_LEN_Q: Tx.constexpr,
    NUM_QO_HEADS: Tx.constexpr,
    HEAD_DIM: Tx.constexpr,
):
    Tx.device_entry()
    ...
```

A JIT function is specialized before compile:

```python
kernel = _kernel.specialize(
    BATCH_SIZE=batch_size,
    SEQ_LEN_Q=seq_len_q,
    NUM_QO_HEADS=num_heads,
    HEAD_DIM=head_dim,
)
```

## Coordinates and Scopes

Coordinates name positions. Scopes name the team that cooperates on the following operation.

```python
Tx.device_entry()
bx, by = Tx.cta_id([M // BLK_M, N // BLK_N])
wg_id = Tx.warpgroup_id([2])
warp_id = Tx.warp_id_in_wg([4])
lane_id = Tx.lane_id([32])
tid_in_wg = Tx.thread_id_in_wg([128])
```

The relationship is in the name: `warp_id_in_wg` means warp inside a warpgroup, and `cta_id_in_cluster` means CTA inside a cluster.

Scope examples:

```python
with Tx.cta():
    Tx.copy(Asmem[:, :], A_tile)

with Tx.warpgroup():
    Tx.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
    Tx.ptx.tcgen05.wait.ld()

with Tx.thread():
    Tx.copy(D[row, :], Dreg_f16[:])
```

A scope block does not create threads. The launch already did that. The scope tells TIRX how to lower the operation inside it.

## Buffers

There are two common ways to describe kernel arguments.

The handle form is useful when a runtime dimension appears in the buffer shape:

```python
def kernel(a_ptr: Tx.handle, b_ptr: Tx.handle):
    n = Tx.int32()
    A = Tx.match_buffer(a_ptr, [n], "float32")
    B = Tx.match_buffer(b_ptr, [n], "float32")
```

The typed-buffer form is useful for constexpr-specialized kernels:

```python
def kernel(A: Tx.Buffer((M, K), "float16"), D: Tx.Buffer((M, N), "float16")):
    ...
```

Inside the kernel, these are device buffers. They are not Python arrays.

## Shared Memory and TMEM

Shared memory is usually allocated through `Tx.SMEMPool()`:

```python
pool = Tx.SMEMPool()
tmem_addr = pool.alloc([1], "uint32")
mma_bar = pool.alloc([1], "uint64", align=8)
pool.move_base_to(1024)
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
pool.commit()
```

The pool is a shared-memory allocation plan. It is not an automatic live-range optimizer. Put control objects and tile storage in the order you need, then call `commit()`.

TMEM can be managed manually. The layout vocabulary (`TileLayout`, `S`, `TLane`, `TCol`) lives in `tvm.tirx.layout`, not on `Tx`, so import it explicitly:

```python
from tvm.tirx.layout import S, TileLayout, TLane, TCol

Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=512, cta_group=1)
tmem = Tx.decl_buffer(
    (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
    layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
)
```

Or through `Tx.TMEMPool(...)`, which is the style used in current larger kernels:

```python
tmem_pool = Tx.TMEMPool(pool, total_cols=N_COLS_TMEM, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()
...
tmem_pool.dealloc()
```

`Tx.decl_buffer(..., scope="tmem")` gives TIRX a typed view of already allocated TMEM. It does not allocate TMEM by itself.

## Pipeline Objects

Software-pipelined kernels need stage and phase tracking. Current TIRX uses `PipelineState` and `Pipeline`. These helpers are *not* part of the `Tx` namespace; import them explicitly from `tvm.tirx.lang.pipeline` (the same module also exports `MBarrier`, `TMABar`, and `TCGen05Bar`):

```python
from tvm.tirx.lang.pipeline import Pipeline, PipelineState

kv_state = PipelineState(SMEM_PIPE_DEPTH_KV)
kv_load = Pipeline(pool, SMEM_PIPE_DEPTH_KV, full="tma", empty="tcgen05", empty_phase_offset=1)
```

`PipelineState` carries `stage` and `phase`. Calling `advance()` moves to the next stage and flips the phase when the ring wraps.

`Pipeline` allocates the barriers that protect each stage. The `full` endpoint is usually waited by the consumer; the `empty` endpoint is usually waited by the producer before reusing a stage. The exact names depend on the producer-consumer direction.

## Metaprogramming Helpers

`Tx.meta_var(expr)` is a parser-only value: the wrapped expression is substituted inline wherever the name is used, so it never becomes a separate IR variable in the final TIR:

```python
m_st = Tx.meta_var(bx * BLK_M)
n_st = Tx.meta_var(by * BLK_N)
```

Use it for index arithmetic and small compile-time-derived expressions, not for storage allocation.

`@Tx.inline` marks a helper that is expanded at the call site:

```python
@Tx.inline
def tma_load(stage, k_start):
    with Tx.thread(tid == 0):
        Tx.copy_async(Asmem[stage, :, :], A[:, k_start:k_start + BLK_K], dispatch="tma")
```

Inlining is useful when the same tile operation appears in prefetch and loop bodies.

## Compile Pipeline

Compile TIRX kernels with the TIRX pipeline selected:

```python
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
```

At a high level:

```text
TIRX Python DSL
  -> TVM IR with TIRX-specific nodes
  -> TIRX lowering passes
  -> ordinary TIR
  -> CUDA host/device code
  -> compiled GPU module
```

The default TIR pipeline is not a replacement for `tir_pipeline="tirx"` when the function contains TIRX tile primitives.

## Generated CUDA Inspection

Generated CUDA is the fastest way to check whether scopes and hardware helpers lowered as intended:

```python
cuda_source = ex.mod.imports[0].inspect_source("cuda")
print(cuda_source)
```

Useful checks:

| TIRX intent | Generated-code clue |
|-------------|---------------------|
| CTA coordinate | `blockIdx.x`, `blockIdx.y` |
| CTA thread coordinate | `threadIdx.x` |
| Warpgroup branch | `warp_id_in_cta >> 2` |
| Warp branch | `warp_id_in_cta & 3` |
| Lane-0 branch | `threadIdx.x % 32 == 0` |
| Elected issue | `tvm_builtin_elect_one_sync_op()` |
| TMA path | `cp.async.bulk` |
| Blackwell MMA | `tcgen05.mma` |
| mbarrier | `mbarrier.init`, `mbarrier.arrive`, `mbarrier.try_wait` |

When the TIRX source and generated CUDA disagree, debug the generated CUDA first. It shows which threads actually issue an operation and which waits are present.
