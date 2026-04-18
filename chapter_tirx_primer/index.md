# TIRX Programming Model
:label:`chap_tirx_primer`

The TIRX programming model is built out of three explicit interfaces that together answer every question the compiler needs to lower a kernel: *where each tensor element lives in hardware* (tensor layout), *which threads cooperate at what granularity* (execution scope), and *what instruction a high-level primitive call turns into* (tile primitive dispatch). This chapter introduces those three pillars and nothing else. Memory hierarchy details live in :numref:`chap_background`; full layout semantics live in :numref:`chap_layouts`; concrete kernel walk-throughs start in :numref:`chap_gemm_basics`.


## Import Convention

This tutorial uses the following import:

```python
from tvm.script import tirx as Tx
```

`Tx` is the TIRX namespace used throughout the tutorial. It provides the TIRX DSL APIs, including kernel constructs, thread-hierarchy scopes, memory helpers, tile primitives such as `Tx.copy` / `Tx.gemm_async`, and low-level PTX intrinsics under `Tx.ptx.*`.


## Overview: Three Pillars

A TIRX kernel is built out of three orthogonal ingredients. Each addresses one question the compiler must answer when it lowers a primitive call:

| Pillar | Answers the question | What the user declares |
|:--|:--|:--|
| **Tensor layout** | *Where does each logical element live in hardware?* | a per-buffer layout, e.g. `S[(128, 512) : (1@TLane, 1@TCol)]` |
| **Execution scope** | *Which threads cooperate, and at what granularity?* | a team kind (`with Tx.warpgroup():`) plus an active thread set (`[predicate]`) |
| **Tile primitive dispatch** | *What backend instruction implements this primitive under the current scope and operand layouts?* | the primitive call itself, e.g. `Tx.copy(dst, src)` or `Tx.gemm_async(...)` |

These three pillars are what make TIRX a compiler rather than a thin wrapper over PTX. We introduce them one at a time, then show how they line up in a minimal kernel template. Everything outside the three pillars — syntactic sugar such as `Tx.meta_var` and `@Tx.inline`, and plumbing utilities such as the `PoolAllocator` — is cataloged in :numref:`chap_api_reference`.


## Pillar 1: Tensor Layout

TIRX extends the classical shape-and-stride layout of NumPy and PyTorch with two additions, forming a *shard + replica + offset* triple. Every operand buffer carries such a layout, declaring how its logical indices map onto **named hardware axes**.

- **S (Shard).** A list of `(extent, stride, axis)` iters. This is the familiar shape-and-stride, except each stride is tagged with the axis it steps along. For example `S[(128, 512) : (1@TLane, 1@TCol)]` says logical dim 0 runs along the `TLane` (TMEM row) axis with stride 1, and logical dim 1 runs along `TCol` (TMEM column) with stride 1.
- **R (Replica).** A *set* of iters that enumerate broadcast offsets independent of the logical index. `R[4 : 1@warp]` replicates the same data across 4 consecutive warps. This is what makes Blackwell scale-factor tiles, cluster-wide multicast buffers, and distributed tensors expressible without shape hacks.
- **O (Offset).** A constant coordinate offset added to every result. Typical use: pin a TMEM buffer to a specific base column.

The tutorial uses the symbol `S` for the shard list, matching :numref:`chap_layouts`. The TIRX research paper calls it `D` in its formal notation; the two are interchangeable.

The axes available in TIRX form a fixed vocabulary — the compiler understands their semantics and emits the right lowering when it sees each axis name:

| Axis | Meaning |
|:--|:--|
| `m` | ordinary memory offset (SMEM / GMEM / RF element stride) |
| `tid`, `tid_in_wg`, `lane`, `warp`, `warp_in_wg`, `wg_id` | thread-hierarchy axes |
| `TLane`, `TCol` | Blackwell TMEM 2D axes |
| `cta`, `cbx`, `cby` | CTA and cluster axes (for TMA multicast, DSMEM, cluster-wide tiles) |
| `pid` | device-mesh axis for multi-GPU tensors |

A tensor's layout is the *declarative* part of the programming model. Pillars 2 and 3 read the layout to decide what instructions to emit; the programmer never derives offsets or thread-work-assignments by hand. The three layouts you will see repeatedly in Part III are:

```python
tma_shared_layout(dtype, SWIZZLE_128B_ATOM, shape)    # SMEM, swizzled for TMA
TileLayout(S[(128, N) : (1@TLane, 1@TCol)])           # TMEM operand
TileLayout(S[(128, N) : (1@tid_in_wg, 1)])            # RF, warpgroup view
```

Full semantics — how axes compose, how to pick a swizzle mode, how TMEM↔RF axis matching lets `Tx.copy` lower to `tcgen05.ld`, how cluster axes drive TMA multicast — are in :numref:`chap_layouts`. For this chapter the important thing is only that every buffer carries a layout, and that Pillars 2 and 3 consult it.


## Pillar 2: Execution Scope

At every program point, two orthogonal facts must be known:

1. The **team kind** $\kappa$ — at what granularity is the next primitive expected to run? One of `cta`, `warpgroup`, `warp`, `thread`. This is chosen by `with Tx.<team>():`. It does *not* change which threads are physically executing; it changes how the compiler lowers whatever comes next.
2. The **active thread set** $A$ — which physical threads actually enter the block? This is narrowed by appending a bracket predicate `[cond]` to a thread-level scope.

The separation matters because the same 128 warpgroup threads might cooperatively do a `Tx.copy` of a TMEM tile into registers (team kind = warpgroup, active set = all 128), then one of them might issue a TMA store (team kind = thread, active set = just the elected thread). The thread count is unchanged; only the granularity flipped.

### Changing the team kind

```python
with Tx.kernel():              # κ = kernel    (grid-wide setup)
    with Tx.cta():             # κ = cta       (all threads in this CTA cooperate)
        with Tx.warpgroup():   # κ = warpgroup (128-thread team, used for TMEM reads)
            with Tx.warp():    # κ = warp      (32-thread team)
                with Tx.thread():   # κ = thread (each thread acts independently)
                    ...
```

A `Tx.copy(Asmem, Agmem)` under `with Tx.cta():` means "all threads in the CTA cooperate on the copy". The same call under `with Tx.thread():` means "each thread copies its own element". The primitive call is identical; only the team kind changed. Pillar 3 turns this into different instructions.

### Narrowing the active thread set

Appending `[predicate]` to a thread-level scope restricts which threads enter. The predicate may be a single-point condition or a contiguous range:

```python
with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:   # exactly one thread per warp
    Tx.copy_async(Asmem, Agmem, dispatch="tma")        # TMA issued from that thread

with Tx.warp()[0:1]:                                   # first warp only
    Tx.ptx.tcgen05.alloc(...)                          # TMEM alloc instruction

with Tx.thread(parent="warpgroup")[warp_id == 0 and lane_id == 0]:  # single thread in a warpgroup
    mma_bar.init(arrive_count=1)
```

`elect_sync()` is the most common filter — paired with `Tx.thread(parent="warp")` it gives exactly one thread per warp, which matches how TMA issues and `tcgen05.commit` want to be dispatched.

Team switches must align to hardware boundaries — e.g. switching to `warpgroup` scope requires the current active warp range to be a multiple of 4 warps starting at a warpgroup boundary. Misaligned switches are rejected at compile time. The exact rules are in :numref:`chap_api_reference`.

### Scope, at a glance

| Construct | Effect |
|:--|:--|
| `with Tx.cta()` / `Tx.warpgroup()` / `Tx.warp()` / `Tx.thread()` | change the **team kind** (the granularity at which the next primitive is lowered) |
| `[predicate]` appended to a thread-level scope | narrow the **active thread set**; predicate is single-point or a contiguous range |
| `Tx.cta_id([...], parent=...)`, `Tx.warpgroup_id(...)`, `Tx.warp_id(...)`, `Tx.thread_id(...)` | symbolic coordinates within the current team |

Some primitives are hard-coded to a particular team kind — for instance `Tx.cuda.cta_sync()` is always a CTA-wide barrier, and `Tx.cuda.warpgroup_sync(id)` is always warpgroup-wide. The compiler checks that the surrounding active thread set is compatible with such fixed-scope operations.


## Pillar 3: Tile Primitive Dispatch

A **tile primitive** is a user-facing operation on tiles — `Tx.copy`, `Tx.gemm_async`, `Tx.cast`, `Tx.add`, `Tx.reduce`, `mbar.arrive`, and so on. Calling a primitive is *declarative*: it states what the operation does on the logical tensor values. The *how* — which instruction, at which thread granularity, using which addressing mode — is resolved by the compiler from three inputs:

$$
\underbrace{\text{primitive call}}_{\text{what}} \;+\; \underbrace{\text{execution scope}}_{\text{who}} \;+\; \underbrace{\text{operand layouts}}_{\text{where}} \;\;\longrightarrow\;\; \text{backend instruction(s)}
$$

This is the third pillar. The programmer never writes the dispatch logic; the compiler does, reusing a small set of layout operators — canonicalise, group, permute, slice — to match the primitive's shape requirements against the operand layouts and pick an efficient lowering. Two examples make the mechanism concrete.

### Example A: one `Tx.copy`, four different lowerings

The same `Tx.copy(dst, src)` call is dispatched to different PTX depending on the scope and the operand layouts:

| Team kind | `dst` layout | `src` layout | Lowered to |
|:--|:--|:--|:--|
| thread (elected), `cta_group=2` | SMEM with cluster axes (`1@cbx` shard + `R[4 : 1@cby]`) | GMEM tile | **TMA multicast** (`cp.async.bulk.tensor.multicast`) |
| thread (elected), single CTA | SMEM tile (swizzled) | GMEM tile | **TMA unicast** (`cp.async.bulk.tensor`) |
| warpgroup | RF warpgroup view `(1@tid_in_wg, 1)` | TMEM tile `(1@TLane, 1@TCol)` | **`tcgen05.ld`** (warpgroup-cooperative TMEM → RF) |
| thread | RF per-thread flat | GMEM / SMEM flat | plain vectorised **LDG / LDS / STS / STG** |

From the user's point of view the code in each row is essentially the same:

```python
Tx.copy(dst, src)
# or, for the TMA variants:
Tx.copy_async(dst, src, dispatch="tma", cta_group=CTA_GROUP, mbar=...)
```

What makes the first row lower to TMA multicast and the third row to `tcgen05.ld` is (a) the team kind flowing from the enclosing scope, and (b) the *named axes* in the destination layout — `cbx`/`cby` invites multicast, `TLane`/`TCol` invites `tcgen05.ld`. Declare a wrong layout and the dispatcher either picks a slower instruction or rejects the call with a compile-time error.

### Example B: `Tx.gemm_async` picks the MMA shape and CTA group

```python
Tx.gemm_async(tmem[:, :N], Asmem, Bsmem,
              accum=(k != 0), dispatch="tcgen05", cta_group=1)
```

The primitive says "accumulate $A B$ into `tmem`". The compiler consults:

- `tmem`'s layout `S[(128, N) : (1@TLane, 1@TCol)]` → the target lives in TMEM, so choose a `tcgen05.mma` variant.
- `Asmem` / `Bsmem` SMEM layouts (including swizzle mode) → build the SMEM matrix descriptors and set transpose flags required by `tcgen05.mma`.
- `cta_group=1` → single-CTA instance. Passing `cta_group=2`, together with SMEM layouts that carry `1@cbx` + `R[... : 1@cby]`, produces the 2-CTA cooperative form that reaches up to $M=256$.
- Tile shape inferred from the layouts → select a valid `tcgen05.mma` M-N-K shape ($M \in \{64, 128, 256\}$, $N$ multiple of 8 up to 256, $K$ set by element type).

A single primitive call emits one — or, for large tiles, a short loop of — `tcgen05.mma` instructions with the right bit-packed instruction and SMEM descriptors. The descriptor encoding, the transpose handling, and the instruction loop all live in the dispatcher, not in the user kernel. That is why the 2-line MMA in :numref:`chap_gemm_basics` expands to 20+ lines of PTX without the user ever opening a PTX manual.

### What to internalise

- **Declarativity.** The primitive says *what*; the scope and layouts say *for which team and where the data is*; the dispatcher says *how*. Changing the lowering strategy — for instance switching a TMA copy between unicast and multicast — is a layout edit, not a rewrite of the operation.
- **Layout selects the instruction family.** `cbx` / `cby` axes → cluster instructions; `TLane` / `TCol` → `tcgen05` instructions; `tid` / `tid_in_wg` + `m` → ordinary per-thread LDG / STS / STG. The named-axis vocabulary is the user's way to *select* a lowering.
- **Scope gates which primitives are legal.** Some primitives only make sense under a specific team kind — e.g. `tcgen05.ld` requires a warpgroup-cooperative read, so calling it from `with Tx.thread():` is a compile-time error.


## Memory Spaces at a Glance

The three pillars above define *what the programming model is*. They say little about *where memory physically lives*; one paragraph is enough here, with pointers elsewhere for the details:

- **Global memory** — declared via `Tx.Buffer((M, N), dtype)`. No TIRX-specific layout is required; accessed through `Tx.copy_async(..., dispatch="tma")` or ordinary indexing.
- **Shared memory** — allocated out of a `PoolAllocator` and given a layout, usually via `tma_shared_layout(...)` for swizzled TMA-compatible tiles. Size and swizzle budget are covered in :numref:`chap_background` and :numref:`chap_layouts`.
- **Tensor memory (TMEM)** — declared with `Tx.decl_buffer(..., scope="tmem")` and a `TLane` / `TCol` layout. Lifecycle (`tcgen05.alloc` / `dealloc`) is shown in :numref:`chap_gemm_basics`.
- **Register file** — declared with `Tx.alloc_local((...,), dtype)`; a per-thread buffer can be `.view(...)`-ed into a warpgroup-level logical tile through a layout such as `S[(128, N) : (1@tid_in_wg, 1)]`.


## A Minimal Kernel Template

The three pillars map onto a kernel's four regions as follows (the right-hand comments name the dispatched instruction for each primitive call):

```python
@Tx.prim_func(tirx=True)
def hgemm(A: Tx.Buffer((M, K), "float16"),
          B: Tx.Buffer((N, K), "float16"),
          D: Tx.Buffer((M, N), "float16")):
    with Tx.kernel():
        # ① Setup — layouts declared, SMEM / TMEM allocated
        bx, by = Tx.cta_id([grid_m, grid_n], parent="kernel")
        wg_id  = Tx.warpgroup_id([num_wg], parent="cta")
        pool   = Tx.PoolAllocator()
        Asmem  = pool.alloc(shape_a, layout=A_layout)   # SMEM layout
        tmem   = Tx.decl_buffer(..., scope="tmem", layout=tmem_layout)

        # ② Load — κ = thread (elected); Tx.copy_async → TMA
        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
            Tx.copy_async(Asmem, A[...], dispatch="tma")

        # ③ Compute — κ = thread (elected); Tx.gemm_async → tcgen05.mma
        with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
            Tx.gemm_async(tmem, Asmem, Bsmem, dispatch="tcgen05", cta_group=1)

        # ④ Writeback — κ = warpgroup for TMEM → RF; κ = thread for RF → GMEM
        with Tx.warpgroup():
            Tx.copy(Dreg_wg, tmem[:, :])                # tcgen05.ld
        Tx.copy(D[...], Dreg_f16)                       # STG
```

Each chapter of Part III elaborates one piece: :numref:`chap_gemm_basics` introduces ① and ②; :numref:`chap_gemm_async` adds TMA and software pipelines to ② and ③; :numref:`chap_gemm_advanced` warp-specialises all four regions across consumer warpgroups; :numref:`chap_flash_attention` composes the full stack.


## Where to Find More

- Full op catalog, barrier helpers, and PTX intrinsics: :numref:`chap_api_reference`.
- Layout semantics in depth (shard / replica / offset, swizzling, axis matching): :numref:`chap_layouts`.
- Memory-hierarchy numbers (SMEM / TMEM / RF capacities, bandwidths): :numref:`chap_background`.
- Syntactic sugar used throughout Part III (`Tx.meta_var`, `@Tx.inline`): :numref:`chap_api_reference`, section *Syntactic Sugar*.
