# Modern GPU Programming For MLSys — Book Outline (planning doc)

*Title:* **Modern GPU Programming For MLSys** · *Vehicle:* **TIRx** · *Model:* Dive-into-Deep-Learning
(runnable, build-it-up, one framework as the spine).

**Audience assumption (locked):** reader *knows some CUDA* — grid/block/thread, SMEM,
coalescing, a naive tiled GEMM — so we don't re-teach SIMT from zero. But **Part I is a real
hardware foundation, not a refresher**: it treats the modern Blackwell-class GPU (memory
hierarchy + TMEM, the tensor-core compute engine, asynchronous TMA, the barrier/phase model,
warpgroups/clusters) as the actual subject. All of that hardware comes **before** the
TIRx programming part.

**Structure note (revised):** the former "Part III — Async Tensor-Core Machine" (Tensor Cores
& TMEM, TMA, mbarriers) is now **folded into Part I**, so the GPU is fully introduced before
Part II teaches how to program it. Parts after it shifted up by one. The per-chapter notes
below still describe each chapter correctly; only the Part grouping/numbering changed — Part I
now holds Ch1–2 + the three hardware chapters + "What Makes a Kernel Fast"; GEMM is Part III,
Flash Attention Part IV, Workflow Part V.

**Spine / through-line**
- One framework: every kernel is read through **scope · layout · dispatch** (introduced in Part II, reused everywhere).
- One case study: the **GEMM optimization ladder** (steps 1→10) is the backbone of the GEMM part; each step applies one hardware concept from Part I. The `summary_journey` figure is the recurring map.
- Rhythm: Part I teaches a hardware capability (slide-derived) → the GEMM part uses it (concept → code).
- Capstone: Flash Attention reuses *all* prior machinery.
- Generation-agnostic seams: hardware specifics are flagged so **Rubin** slots in later without a rewrite.

**Legend:** `[have]` exists · `[grow]` expand existing · `[NEW]` write new.
Slide decks: **DL** = data-layout, **GG** = modern-gpu-gemm, **BT** = blackwell-tirx.

---

## Slide-deck → chapter feed (quick map)

| Slide content | Feeds chapter(s) |
|---|---|
| BT: sm_architecture, pipeline_arch | 2 Modern GPU at a Glance |
| GG: GEMM Optimization, End-to-End GEMM Example | 3 What Makes a Kernel Fast |
| DL: Tile Layout, Named Axes, Distributed Axes | 7 Data Layouts |
| DL+GG: Memory Banks, Simple Swizzle, SWIZZLE_128B, Bank Sector View, Swizzle Atoms, Tiling Constraint | 7 Data Layouts (swizzle) |
| BT: tirx_raw/why_cuda_ptx/however_painful/tirx_operator_dispatch, core_apis, kernel_structure | 6 Tile Primitives |
| GG: Tensor Core, Tensor Core Benefit from Swizzling · BT: tcgen05_intro | 9 Tensor Cores & TMEM |
| GG: TMA Global→Shared, 3D TMA, Why 128B Swizzle · BT: tma_intro | 10 TMA |
| BT: barrier_intro, mbarrier_mechanism, mbarrier_*_timeline, phase_tracking | 11 mbarriers & Phases |
| BT: step1/step4/step7/step9 demos, summary_journey | 12–14 GEMM, recurring |

---

## Part I — The Modern GPU (fast refresher + what's new)

### Introduction — DROPPED as a separate chapter
The landing page (`index.md`) serves as the introduction: it carries the abstraction-ladder
framing, TIRx-as-vehicle, the scope/layout/dispatch knobs, and "How This Book Is Organized."
Part I therefore opens directly with the GPU Execution and Memory Model.

### 2. The Modern GPU at a Glance `[grow background]`
**Goal:** refresh the hierarchy fast, then install the *new* mental furniture.
- 1-page recap (assumed known): grid/block/warp/thread, GMEM/L2/SMEM/registers, coalescing.
- New since "classic" CUDA: **warpgroup** (4 warps / 128 threads), **CTA cluster + DSMEM**, **Tensor Memory (TMEM)**, the async execution units (Tensor Core engine + TMA engine) as first-class actors.
- The async dataflow picture: who issues, who computes, who waits (producer/consumer at a glance).
- Feeds: BT `sm_architecture`, `pipeline_arch`; existing `background` (Thread Hierarchy, Memory Spaces).
- New: reframe `background` from reference → refresher; add the warpgroup/cluster/TMEM deltas; generational note.

### 3. What Makes a Kernel Fast `[NEW]`
**Goal:** the "why" engine for every later optimization.
- Roofline: arithmetic intensity, memory- vs compute-bound; where GEMM/attention sit.
- Why moving bytes dominates: bandwidth vs FLOPs on Blackwell; the case for TMA + tiling.
- Overlap as the lever: latency hiding via async + pipelining (preview of Part IV).
- Occupancy, register/SMEM pressure — just enough to reason about tradeoffs.
- Feeds: GG `GEMM Optimization`, `End-to-End GEMM Example` (perf data), `speed.csv`.
- New: prose + one roofline figure + a small measured table (reuse the GEMM ladder numbers).

---

## Part II — Programming a GPU with TIRx

### 4. Setup & Your First Kernel `[have setup]`
**Goal:** environment works; reader compiles+runs+inspects a 1-D kernel.
- Requirements/install, the `double_it` minimal kernel, run on B200, inspect generated CUDA, troubleshooting.
- Mostly as-is; verify install instructions (currently flagged "being updated").

### 5. TIRx Language & Compile Pipeline `[have tirx_primer]`
**Goal:** the source model and how it lowers.
- `@T.prim_func` vs `@T.jit`; coordinates & scopes; buffers; SMEM/TMEM; `T.meta_var`; the tirx compile pipeline.
- As-is; add a short "what the compiler does with scope/layout/dispatch" bridge into Ch6.

### 6. Tile Primitives: Scope · Layout · Dispatch `[have layouts → split]`
**Goal:** install the three-knob framework (the book's recurring lens).
- Why tile primitives (vs raw PTX pain); the three knobs; reading a primitive in context (the op + its wait/barrier).
- Feeds: BT `why_cuda_ptx_native`, `however_cuda_cpp_painful`, `tirx_raw_cuda_ptx_tcgen05`, `tirx_operator_dispatch`, `core_apis`, `kernel_structure`.
- Action: keep the mental-model half here; **move the layout taxonomy out to Ch7**.

### 7. Data Layouts `[grow layouts + DL/GG slides]`
**Goal:** layouts as a first-class concept; bank conflicts & swizzle understood.
- Logical→physical mapping; `TileLayout` / `S[...]` notation; named axes (`@m`, `@laneid`, `@reg`, `tid_in_wg`).
- Tiled & thread layouts; distributed (cluster) layouts; register views.
- Memory banks → bank conflicts → **swizzle**; SWIZZLE_128B, swizzle atoms, the tiling constraint.
- Feeds: DL (Tile Layout, Named Axes, Distributed Axes, Memory Banks, Simple Swizzle, SWIZZLE_128B, Swizzle as optimization); GG (Bank Sector View, Swizzle Atoms, Tiling Constraint, Why 128B). Existing `layouts` Tensor-Layout section + `axe_layout` demos.
- New: substantial — promote slide concepts into prose+figures; tie to `tma_shared_layout` used later.

### 8. First Real Kernels: Elementwise & Reductions `[promote rmsnorm + fused_gelu]`
**Goal:** complete, simple kernels before the tensor-core machinery.
- Thread mapping & vectorized loads (GELU gate); warp-shuffle + cross-warp SMEM reduction (RMSNorm).
- Establishes per-thread vs cooperative scope, register tiles, the verify-vs-reference loop.
- Action: promote the two "practice" chapters into a taught Part-II chapter; light edits + motivation.

---

## Part III — The Async Tensor-Core Machine (concept chapters)

### 9. Tensor Cores & TMEM `[NEW from slides]`
**Goal:** understand `tcgen05` MMA and Tensor Memory.
- What a Tensor Core MMA is; the Blackwell `tcgen05` model; matrix/instruction descriptors; **TMEM** (128 rows × cols, Tensor-Core-private); why accumulators live in TMEM.
- Swizzle ↔ tensor-core operand layout link (closes the loop with Ch7).
- Feeds: GG (Tensor Core, Tensor Core Benefit from Swizzling); BT `tcgen05_intro`. Existing `background` MMA section.
- New: concept chapter; small TIRx snippet issuing one MMA into TMEM.

### 10. Asynchronous Data Movement: TMA `[NEW from slides]`
**Goal:** understand the Tensor Memory Accelerator.
- One thread issues, hardware moves the tile; TMA descriptors; 2D/3D TMA; swizzle-for-TMA; load vs store completion protocols.
- Feeds: GG (TMA Global→Shared, 3D TMA, Why 128B Swizzle); BT `tma_intro`. Existing `background` TMA section.
- New: concept chapter; snippet of `Tx.copy_async(..., dispatch="tma")` + `expect_tx`.

### 11. Async Coordination: mbarriers & Phases `[grow background]`
**Goal:** the synchronization model behind all async kernels.
- mbarrier data structure & APIs; arrive / `expect_tx` / `try_wait`; **phase tracking** (the ping-pong); signaling TMA vs `tcgen05` completion; producer/consumer handoff.
- Feeds: BT `barrier_intro`, `mbarrier_mechanism`, `mbarrier_arrive_timeline`, `mbarrier_tma_timeline`, `mbarrier_tcgen05_timeline`, `phase_tracking`. Existing `background` mbarrier section.
- New: promote the timeline figures into teaching; this chapter is the prereq the GEMM pipeline leans on.

---

## Part IV — GEMM: Tiled → SOTA (the spine) `[have]`

### 12. A Tiled GEMM `[have gemm_basics]` — steps 1–3
**Goal:** a correct single-tile → K-loop → spatially-tiled GEMM.
- **Add up front:** naive→SMEM-tiled→tensor-core *motivation* (why we jump to `tcgen05`), so the leap isn't abrupt. Optional: show `no_sugar_gemm.py` once to reveal what the sugar expands to.
- Feeds: BT `step1_*`, `summary_journey`; `tirx_tutorial/no_sugar_gemm.py`.

### 13. Pipelining with TMA `[have gemm_async]` — steps 4–6
**Goal:** async load → 2-stage software pipeline → persistent kernel + tile scheduler.
- Each step cites its Part-III concept (TMA → Ch10, mbarrier/phase → Ch11).
- Feeds: BT `step4_*`, `pipeline_dynamic`.

### 14. Warp Specialization & Clusters `[have gemm_advanced]` — steps 7–10
**Goal:** producer/consumer warp specialization, 2-CTA cluster (DSMEM), multi-consumer.
- Feeds: BT `step7_*`, `step9_*`, `summary_journey`.

---

## Part V — Capstone

### 15. Flash Attention 4 `[have flash_attention + fa4_source]`
**Goal:** compose everything — online softmax, two MMA phases, correction, epilogue; full warp-role machine.
- Keep the conceptual chapter + the annotated source listing. (Source listing now compiles — recently fixed.)

---

## Part VI — Workflow & Practice

### 16. Profiling & Debugging `[NEW + setup/ai_assisted bits]`
**Goal:** measure and fix kernels.
- Nsight Compute basics, reading generated CUDA/PTX/SASS, occupancy/stall analysis, debugging barriers & correctness (the elected-commit bug case study).
- Feeds: `setup` (Inspect Generated CUDA), `ai_assisted` (Case Study). New: profiling workflow + a worked perf-debug example.

### 17. Writing Kernels with Agents `[have ai_assisted]`
**Goal:** use the scope/layout/dispatch contract as agent prompt structure.
- As-is; re-point cross-refs to the new structure.

---

## Appendix `[have]`
API reference · FA4 full source · `no_sugar_gemm` · practice kernels · **hardware quick-ref tables (Blackwell now, Rubin later)** · (optional) numerics/precision note (fp16/bf16/fp8/nvfp4, accumulation, scaling).

---

## What's genuinely NEW to write (priority)

**P0 — makes it a "book," not a manual**
- ~~Ch1 Introduction~~ — dropped; the landing page (`index.md`) is the introduction.
- Ch3 What Makes a Kernel Fast (roofline / the "why").
- Ch12 naive→tiled→tensor-core motivation paragraph(s).

**P1 — front-half foundations from slides**
- Ch2 refactor `background` → modern-GPU refresher (warpgroup/cluster/TMEM deltas).
- Ch7 Data Layouts (promote DL + GG swizzle slides) — largest single new chapter.
- Ch9 Tensor Cores & TMEM, Ch10 TMA, Ch11 mbarriers (split `background` reference into 3 teaching chapters).
- Ch8 promote rmsnorm/gelu to a taught chapter.

**P2 — polish/completeness**
- Ch16 Profiling & Debugging.
- Numerics/precision appendix note.
- Rubin/portability seams + hardware quick-ref tables.

## Sequencing suggestion
1. Lock this outline + the new `toctree` (Parts I–VI).
2. Draft **Ch3** and **Ch1** first (they set tone + the perf vocabulary everything references).
3. Then Ch7 (layouts) and the Part-III split (9/10/11) — most reuse from slides.
4. Re-slot existing GEMM/FA chapters under the new parts; add the Ch12 motivation.
5. Ch16 + appendix polish last.

## Open questions for you
- **Slide assets:** port the interactive HTML demos into the book (as static figures? embedded iframes? regenerated images?), or re-draw key ones as static diagrams?
- **Depth of Part I:** is a 3-chapter refresher right, or fold Ch2+Ch3 into one to get to TIRx faster?
- **Numerics:** dedicate a short chapter (fp16/bf16/fp8/nvfp4 matter for SOTA), or keep as an appendix note?
- **Rubin:** include forward-looking "what changes on Rubin" callouts now, or defer entirely?
