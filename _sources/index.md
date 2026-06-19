# Modern GPU Programming For MLSys

This book teaches modern GPU kernel programming as a progression: **understand the GPU
hardware → learn to program it → write state-of-the-art kernels.** It assumes you have seen
CUDA basics (grid/block/thread, shared memory, a naive tiled GEMM), but it treats the modern,
Blackwell-class GPU as the real subject — its memory hierarchy and Tensor Memory, its
tensor-core and asynchronous data-movement engines, warpgroups and clusters — rather than as a
quick review.

The vehicle is **TIRx** (Tensor IR neXt), a Python DSL for writing GPU kernels at the IR
level. TIRx sits between high-level kernel DSLs and raw CUDA/PTX: the kernel still names
hardware concepts directly, while the compiler sees scope, layout, and dispatch as structured
IR instead of scattered intrinsic arguments. Like the framework in *Dive into Deep Learning*,
TIRx is the consistent medium through which every concept becomes runnable code.

A **tile primitive** is a structured operation on tile values, and its lowering is controlled
by three knobs that recur throughout the book:

- **Scope** — which group of threads issues or cooperates on the operation.
- **Layout** — how the operand tiles map to GMEM, SMEM, TMEM, or registers.
- **Dispatch** — which hardware path is intended when there is a choice, such as TMA or `tcgen05`.

Asynchronous primitives add one more concern — *coordination*: a barrier, commit, wait, or fence
marks each handoff between tile operations.

## How This Book Is Organized

- **Part I — Understanding the GPU.** What the hardware *is*: the execution and memory model and
  the performance model (roofline, overlap) that defines "fast"; then a deep dive into data
  layout, the memory engines (TMA and Tensor Memory), the Tensor Core, the barrier/phase
  coordination model, and advanced scheduling (CLC). Everything later is programming *this* hardware.
- **Part II — TIRx Overview.** An introduction to TIRx through one runnable
  single-MMA GEMM — scope, layout, and dispatch in action, and how compilation works — and the
  tensor layout model (`TileLayout`, named axes, swizzle). The full language-feature set lives in
  the Reference.
- **Part III — GEMM: Tiled to SOTA.** The optimization spine — a tiled GEMM built up through
  TMA pipelining, persistent scheduling, warp specialization, and 2-CTA clusters.
- **Part IV — Flash Attention 4.** A complete attention kernel built from the Part III techniques:
  two MMAs with softmax between them, online-softmax rescaling, causal masking, and GQA.
- **Part V — Workflow & Practice.** Profiling/debugging and writing kernels with agents.
- **Reference.** TIRx language reference, the API cheat sheet, compiler internals, and the full
  Flash Attention 4 source.

```{toctree}
:caption: Part I — Understanding the GPU
:maxdepth: 1

chapter_background/index
chapter_performance/index
chapter_data_layout/index
chapter_layout_generations/index
chapter_tma/index
chapter_tensor_cores/index
chapter_tmem/index
chapter_async_barriers/index
chapter_clc/index
```

```{toctree}
:caption: Part II — TIRx Overview
:maxdepth: 1

chapter_intro_tirx/index
tirx_guide/layout
```

```{toctree}
:caption: Part III — GEMM: Tiled to SOTA
:maxdepth: 2

chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
```

```{toctree}
:caption: Part IV — Flash Attention 4
:maxdepth: 2

chapter_flash_attention/index
```

```{toctree}
:caption: Part V — Workflow & Practice
:maxdepth: 2

chapter_profiling/index
chapter_ai_assisted/index
```

```{toctree}
:caption: Reference
:maxdepth: 1

appendix/index
tirx_guide/arch/index
tirx_guide/language_reference/index
chapter_api_reference/index
chapter_fa4_source/index
```
