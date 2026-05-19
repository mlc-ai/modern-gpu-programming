# TIRX: High-Performance GPU Kernel Programming

TIRX (Tensor IR neXt) is a Python DSL for writing high-performance GPU kernels on modern NVIDIA GPUs. It extends TensorIR with first-class support for hardware features such as TMA, tcgen05 MMA, mbarrier synchronization, and CTA clusters, making it easier to write kernels that map directly to hardware.


## Get Started

New here? Follow this path:

1. **Ch 2 (Setup)** — Install TIRX and run your first GPU kernel in under 5 minutes

2. **Ch 3 (Language Basics)** — Buffers, scopes, raw PTX, the compile pipeline, and metaprogramming

3. **Ch 4 (Tile Primitives)** — Start from a 130-line no-sugar Blackwell GEMM, then learn the three interfaces (scope / layout / dispatch) that cut it down to 40

4. **Ch 5 (Fused GELU)** — Write your first real operator kernel

5. **Ch 7 (GEMM Steps 1–3)** — Start the GEMM optimization journey: 70 ms → 0.12 ms, within 10% of cuBLAS (0.11 ms)


## Who Is This For

- **Systems engineers** who want to write high-performance GPU kernels for LLM training and inference.

- **ML engineers** who want to understand what happens under the hood in frameworks like PyTorch and libraries like cuBLAS.

- **Students** taking a GPU programming or ML systems course.

**Prerequisites**: Familiarity with CUDA concepts (threads, warps, shared memory). No TVM experience needed. Requires an NVIDIA Blackwell GPU (B200/B100) to run the code.


## Tutorial Structure

- **Part I: Fundamentals** (Ch 1–4) — Blackwell GPU architecture, environment setup, the TIRX language layer (buffers, scopes, raw PTX, compile pipeline, metaprogramming), and the TIRX tile primitive layer (execution scope, tensor layout, tile primitive dispatch) motivated by a no-sugar Blackwell GEMM.

- **Part II: Entry-Level Kernels** (Ch 5–6) — Learn TIRX basics through two self-contained operators: Fused GELU (elementwise + fusion) and RMSNorm (warp reduction + shared memory).

- **Part III: GEMM Deep Dive** (Ch 7–9) — A single GEMM kernel, optimized step by step across 9 steps in 3 chapters, from 70 ms down to ~0.12 ms (within ~10% of cuBLAS at 0.11 ms).

- **Part IV: Advanced Kernels** (Ch 10) — Flash Attention on Blackwell.

- **Part V: AI-Assisted Development** (Ch 11) — Using LLMs to generate boilerplate, debug kernels, and produce test harnesses.

- **Appendix** (Ch 12) — Full API reference and common pitfalls.


```toc
:maxdepth: 2
:numbered:

chapter_background/index
chapter_setup/index
chapter_tirx_primer/index
chapter_layouts/index
chapter_fused_gelu/index
chapter_rmsnorm/index
chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
chapter_flash_attention/index
chapter_ai_assisted/index
chapter_api_reference/index
```
