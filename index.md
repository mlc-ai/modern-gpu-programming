# Modern GPU Programming For MLSys

Machine learning systems sit at the heart of modern AI workloads. In these systems, performance
often comes down to the quality of a small number of GPU kernels. Attention kernels, LLM prefill
and decode kernels, low-precision block-scaled GEMMs, fused MoE layers, and other large fused
kernels all directly shape end-to-end speed in both training and serving.

To make these kernels fast, however, we need more than a list of optimization tricks. Modern GPUs
are no longer simple variations of the same old design. Recent architectures introduce richer
memory spaces, new access patterns, and increasingly specialized execution units. To program them
well, we need both a clear mental model of the hardware and a practical understanding of how
high-performance kernels are built. This book is about developing both.

The book follows a simple progression: first understand the GPU hardware, then learn the
programming model we will use, and finally build state-of-the-art kernels step by step. Our main
target is the Blackwell generation, and our main running examples are fast matrix multiplication
(GEMM) and FlashAttention. Along the way, we will also study the core ingredients behind GPU
optimization: data layout, asynchronous data movement, and asynchronous coordination.

The material grows out of the [Machine Learning Systems](https://mlsyscourse.org/) course series
at Carnegie Mellon University. To make the ideas easier to study and easier to run, this book uses
the **TIRx** Python DSL to build real GPU kernel examples step by step. TIRx stays close to the
hardware, which lets us reason about low-level control while still learning through runnable code.

## How This Book Is Organized

- **Part I, Understanding the GPU.** This part introduces the overall organization of the GPU,
  general recipes for writing fast kernels, and key concepts such as data layout, asynchronous
  memory operations, and coordination. It builds the hardware intuition that the rest of the book
  relies on.
- **Part II, TIRx Overview.** This part introduces the key elements of TIRx, which serve as the
  foundation for the code examples throughout the book.
- **Part III, GEMM: Tiled to SOTA.** A complete guide to optimizing a tiled GEMM, built up through
  TMA pipelining, persistent scheduling, warp specialization, and 2-CTA clusters.
- **Part IV, Flash Attention 4.** A complete attention kernel built from the Part III techniques:
  two MMAs with softmax between them, online-softmax rescaling, causal masking, and GQA.
- **Reference.** TIRx language reference and compiler internals.

```{toctree}
:caption: Part I, Understanding the GPU
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
:caption: Part II, TIRx Overview
:maxdepth: 1

chapter_intro_tirx/index
chapter_tirx_layout_api/index
```

```{toctree}
:caption: "Part III, GEMM: Tiled to SOTA"
:maxdepth: 2

chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
```

```{toctree}
:caption: Part IV, Flash Attention 4
:maxdepth: 2

chapter_flash_attention/index
```

```{toctree}
:caption: Reference
:maxdepth: 1

appendix/index
appendix/troubleshooting
tirx_guide/arch/index
tirx_guide/language_reference/index
```
