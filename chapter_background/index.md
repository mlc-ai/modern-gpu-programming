# Background: Blackwell GPU Architecture
:label:`chap_background`

Before diving into the code, let's understand the key hardware features of NVIDIA Blackwell GPUs. If you're familiar with CUDA, you already know about threads, warps, shared memory, and global memory. Blackwell introduces several new hardware units and memory spaces that are essential for achieving peak performance.


## Thread Hierarchy

Blackwell organizes threads into a nested hierarchy:

![*Thread Hierarchy*](../img/thread_hierarchy.png)

- **CTA** — short for *Cooperative Thread Array*, the same concept that CUDA programmers know as a thread block. It is the basic scheduling unit: a CTA runs on a single SM and has access to that SM's shared memory. A CTA contains one or more warpgroups of 128 threads each; most kernels in this tutorial use one warpgroup per CTA, and the warp-specialized kernels in :numref:`chap_gemm_advanced` and :numref:`chap_flash_attention` use up to four.

- **Warpgroup**: 4 consecutive warps (128 threads). On Blackwell, the warpgroup is the cooperation unit for TMEM reads — the 128 `TLane` rows match the 128 threads one-to-one. `tcgen05.mma` itself is issued by a single elected thread, but its tile granularity along M matches a warpgroup, so the warpgroup remains the natural scope for a GEMM stage (see the *Tensor Core Generational Evolution* subsection below for details).

- **Warp**: A warp contains 32 threads and is the basic SIMT execution unit.

- **Thread**: Each thread is identified by a lane ID within its warp.

- **Cluster**: A group of cooperating CTAs.

## Memory Hierarchy

For TIRX kernels on Blackwell, it is useful to think about memory as a hierarchy spanning GMEM, SMEM, TMEM, and registers:

| Memory | Ownership | Latency | Description |
|--------|-----------|---------|-------------|
| **Global Memory (GMEM)** | Device-wide | High | Large capacity (HBM), bandwidth-limited |
| **Shared Memory (SMEM)** | Per-CTA (on one SM) | Low | 228 KB per SM, low-latency scratchpad |
| **Tensor Memory (TMEM)** | Per-SM | Very low | Private high-bandwidth Tensor-Core scratchpad; holds MMA accumulators and, for some `tcgen05.mma` variants, the A operand |
| **Register File (RF)** | Per-thread | Lowest | Fastest storage, per-thread |

The typical data flow for a kernel that uses Tensor Cores:

![*Memory Data Flow*](../img/memory_dataflow.png)

The key difference from earlier GPU generations is that tcgen05 writes accumulator results to TMEM first, and software later moves them to registers or shared memory as needed.

**Tensor Memory (TMEM)** is new in Blackwell. It is a high-bandwidth scratchpad memory private to the Tensor Cores. The tcgen05 MMA unit writes its accumulator output directly to TMEM, not to registers or shared memory.

- **TMEM reads are explicit and cooperative.** To consume MMA results, software must explicitly load from TMEM into the register file. These reads are performed cooperatively by the full warpgroup (128 threads).

- **TMEM uses its own addressing and allocation model.** It is not accessed like normal memory; instead, it uses a special 2D address space whose rows are indexed by a hardware axis called `TLane` (128 lanes) and whose columns are indexed by `TCol` (up to 512 columns). TMEM must be explicitly allocated and deallocated. The `TLane` axis has the same 128-element cardinality as a warpgroup, which is why a `tcgen05.ld` can pair TMEM rows with warpgroup threads one-to-one (see :numref:`chap_layouts` for the exact matching rule).


## TMA (Tensor Memory Accelerator)

TMA is a hardware unit that asynchronously copies rectangular tiles between global memory and shared memory. TMA lets software describe a tile copy, and hardware performs it asynchronously in the background.

- **Single-thread launch**: One thread issues the TMA operation, and the hardware performs the actual tile transfer asynchronously in the background. The remaining threads are free to do other work.

- **Swizzled layouts**: TMA can apply layout swizzling during the transfer, helping produce shared-memory layouts that are friendly to Tensor Core access.

- **Barrier integration**: TMA works with mbarriers. The programmer specifies the expected byte count with `arrive.expect_tx`, and the hardware signals completion when that amount of data has arrived.

- **Store support**: TMA is not only for GMEM→SMEM loads; it can also store data from SMEM back to GMEM. TMA stores use a commit-group / wait-group mechanism for completion tracking.

In short, TMA turns tiled GMEM↔SMEM movement into an asynchronous hardware operation with built-in layout and synchronization support.


## Tensor Cores: The Compute Engine

Before looking at the Blackwell-specific `tcgen05` instruction, it is worth stepping back to understand what a Tensor Core is and why the MMA API looks the way it does. If you are already comfortable with Hopper-era `wgmma`, you can skim this section.

A **Tensor Core** is a dedicated hardware unit that, in one instruction, performs a full tile-granularity matrix multiply-accumulate:

$$D = AB + C, \quad A \in \mathbb{R}^{M \times K},\ B \in \mathbb{R}^{K \times N},\ C, D \in \mathbb{R}^{M \times N}.$$

This is in contrast to **CUDA cores**, the GPU's classic general-purpose ALUs that execute scalar fused multiply-add (FMA) — one $d = a \times b + c$ on 32-bit operands, per thread, per cycle. A warp of 32 threads therefore retires at most 32 scalar FMAs per cycle. A single Tensor Core MMA, by contrast, retires the hundreds-to-thousands of FMAs that make up an $M \times N \times K$ tile as one pipelined asynchronous operation. The two engines live on the same SM and complement each other:

| | **CUDA core** | **Tensor Core** ($\texttt{tcgen05}$ on Blackwell) |
|:--|:--|:--|
| Granularity | 1 scalar FMA per instruction | $M \times N \times K$ FMAs per instruction |
| Execution model | SIMT: 32 threads per warp, each issues its own instruction | Single thread (or 2-CTA pair) issues one MMA; hardware runs the tile asynchronously |
| Operands live in | Register file | SMEM (via matrix descriptors) and/or TMEM |
| Accumulator in | Register file | **TMEM** on Blackwell (on-chip SM-local scratchpad, separate from the register file) |
| Typical dtypes | fp32 / fp64 / int32 — general-purpose | fp16 / bf16 / fp8 / fp4 / tf32 — low-precision GEMM-friendly |
| Used for | elementwise, reductions, index math, control flow, everything non-GEMM | dense matrix multiply, convolution, attention $QK^{\top}$ and $PV$ |
| Peak on HGX B200 | ~75 TFLOPS (fp32) | ~2.25 PFLOPS (fp16/bf16) &rarr; up to ~9 PFLOPS (fp4) |

The architectural motivation is simple: dense matrix multiplication dominates deep-learning workloads, has high arithmetic intensity ($O(N^3)$ FLOPs on $O(N^2)$ data), and is therefore compute-bound on a well-fed GPU. A dedicated MMA unit reaches peak throughput that scalar FMAs never can.

### Why You Cannot Ignore Them

On a Blackwell B200 there is a single Tensor Core unit per SM (the `tcgen05` engine), but its peak throughput depends on the operand precision — narrower operands pack more MMAs per cycle:

| Unit | Peak throughput (per GPU, approx.) | Ratio vs FP32 CUDA |
|:-:|:-:|:-:|
| FP32 CUDA cores | ~75 TFLOPS | 1× |
| Tensor Core @ FP16 / BF16 (dense) | ~2.25 PFLOPS | ~30× |
| Tensor Core @ FP8 (dense) | ~4.5 PFLOPS | ~60× |
| Tensor Core @ FP4 (dense) | ~9 PFLOPS | ~120× |

A kernel that does not use the Tensor Core throws away over 95% of the chip. Every GEMM, attention, and convolution kernel in this tutorial is written around it.

### Instruction Shape ($M \times N \times K$)

A single MMA instruction operates on a fixed-shape tile. On Blackwell `tcgen05.mma` with fp16/bf16 operands:

- $M \in \{64, 128\}$ (row tile; the $M$ dimension of $A$ and $D$). With 2-CTA cooperation $M$ doubles to 128 or 256.
- $N \le 256$ (column tile; the most flexible dimension). The exact step depends on $M$ and the CTA group: $M=64$ single-CTA allows multiples of $8$; $M=128$ single-CTA requires multiples of $16$; 2-CTA cooperative mode requires multiples of $32$.
- $K = 16$ (inner-product dimension, fixed for fp16/bf16 operands).

A GEMM loop builds a larger computation by stepping along $K$ and accumulating. Kernels written later in this tutorial choose $M_{\text{mma}} = 128$ or $256$ (with 2-CTA cooperation) and $N_{\text{mma}} = 128, 256$, but always $K_{\text{mma}} = 16$. The block-level tile `BLK_M × BLK_N × BLK_K` is then a multiple of the instruction tile along each dimension.

### Supported Data Types

Tensor Cores support a rich set of input types with accumulation usually in wider precision:

| Input ($A$, $B$) | Accumulator ($C$, $D$) | Primary use |
|:-:|:-:|:-:|
| FP16, BF16 | FP32 | Training; still common for inference |
| TF32 | FP32 | "Full-precision" training |
| FP8 (E4M3, E5M2) | FP32 | Large-model training and inference |
| FP6, FP4 | FP32 | Inference on weight-quantized models |
| INT8, INT4 | INT32 | Legacy inference; sparsity |

Blackwell additionally supports structured 2:4 sparsity on several of the above dtypes, which can up to double effective throughput by skipping half the multiplies. You will not use sparse MMAs in this tutorial, but you should know they exist.

### Generational Evolution

Understanding why Blackwell's MMA API looks the way it does is easier once you see the previous three generations side by side:

| Gen | Instruction | Who issues | Tile scope | A source | B source | Accumulator | Async? |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Volta (V100) | `mma.sync` (wmma) | all 32 threads of a warp | warp | RF | RF | RF | no |
| Ampere (A100) | `mma.sync` (e.g. `m16n8k16`) | all 32 threads of a warp | warp | RF | RF | RF | no |
| Hopper (H100) | `wgmma.mma_async` | **all 128 threads of a warpgroup** | warpgroup | RF **or SMEM** | **SMEM** | RF | **yes** |
| Blackwell (B200) | `tcgen05.mma` | **1 elected thread** (via `elect.sync`) | **warpgroup, or 2×warpgroup with 2-CTA** | **SMEM or TMEM** | **SMEM** | **TMEM** | **yes** |

Three trends drive the API you will use in TIRX:

1. **Tile scope grew from warp to warpgroup to 2-CTA; issue scope shrank to a single thread.** Volta and Ampere treated MMA as a per-warp operation: all 32 threads in a warp collectively execute the instruction and pool their registers to hold A, B, and D. Hopper moved the *tile* scope to a warpgroup — a single `wgmma` spans 128 threads' worth of data — but the *issue* was still collective: all 128 threads had to execute the instruction. Blackwell decouples the two. `tcgen05.mma` is issued by a **single elected thread**, which frees the other 127 threads of the warpgroup to do TMA issuing, softmax, or the epilogue in parallel. The MMA tile still matches warpgroup granularity along $M$ (128 rows of `TLane`), and in *2-CTA* cooperative mode two SMs jointly produce a 256-row tile.

2. **Operands migrated from RF to SMEM (and TMEM).** Reading A and B from registers forced every thread to hold a share of the tile in its register file. As tile sizes grew, RF pressure became the binding constraint: Hopper's `wgmma` already reads B from SMEM, and one variant reads A from SMEM too. Blackwell completes the move by reading A from either SMEM or TMEM. The result: no matter how large the MMA tile, RF usage does not grow with it.

3. **Execution became asynchronous.** Starting with Hopper, MMA issues a work request and returns immediately; completion is signaled through a separate barrier. This is essential for overlap — the issuing thread is free to prepare the next tile while the current MMA runs in the background.

### Accumulator in TMEM (Blackwell-Specific)

The most radical change on Blackwell is the accumulator location. A Hopper `wgmma` tile with $M=64, N=256$ and FP32 accumulation produces $64 \times 256 = 16384$ accumulator values, which at 128 threads per warpgroup means **128 FP32 registers per thread**. The architectural register file is capped at 255 × 32-bit registers per thread; roughly half of that budget is already gone, and we have not started loading operands or intermediate results. In practice Hopper kernels spend substantial effort managing this pressure (multiple smaller MMAs, register reuse tricks).

Blackwell solves the problem structurally by adding **TMEM** — a 128 × 512 32-bit scratchpad per SM, primarily used as the MMA accumulator (and, for `tcgen05.mma` variants with operand A in TMEM, as an A-operand staging area; Flash Attention in :numref:`chap_flash_attention` exploits this to keep the P matrix between the two MMAs inside TMEM). `tcgen05.mma` writes its accumulator output to TMEM, not to registers. The warpgroup only pulls data into registers (`tcgen05.ld`) during the writeback epilogue, when it actually needs to apply activation or cast to the output dtype. Registers are now free for operand pipelining.

This is why every GEMM kernel in Part III follows the same three-stage data flow already sketched in the *Memory Hierarchy* figure above: **TMA** moves GMEM → SMEM, `tcgen05.mma` computes into TMEM, `tcgen05.ld` pulls results back into RF, and a final store returns them to GMEM. Each arrow is a different hardware unit with its own synchronization primitive. The next two sections introduce the Blackwell MMA (`tcgen05`) and the barrier system (`mbarrier`) that stitch this pipeline together.


## tcgen05 (Tensor Core MMA)

`tcgen05` is Blackwell's matrix multiply-accumulate (MMA) unit. Depending on the MMA variant, operand A may come from TMEM or SMEM, while operand B is read from SMEM via matrix descriptors. The result is written to TMEM.

- **Asynchronous execution**: The MMA instruction returns immediately while the computation continues in the background.

- **Single-thread issue**: Only one elected thread issues the MMA and its corresponding commit. The operation itself is carried out by hardware on behalf of the warpgroup.

- **TMEM accumulation**: tcgen05 writes its output to TMEM. It can either overwrite the destination (`accum=False`) or accumulate into existing TMEM values (`accum=True`).

- **Commit and completion tracking**: After issuing one or more MMAs, software uses `commit` to group them. The hardware signals the associated mbarrier when the group completes.

- **CTA-group execution**: `cta_group` controls whether one CTA or multiple CTAs cooperate on the same MMA operation. This becomes important for clustered kernels later in the tutorial.

In short, tcgen05 moves MMA accumulation out of registers and into TMEM, which changes both the dataflow and the synchronization model compared with earlier generations.


## Why TMA and Tensor Cores Come Together

The Tensor Core and the TMA are introduced separately, but in practice they solve two halves of one problem. A rigorous way to see why needs to separate two questions that are often conflated:

1. Is the GEMM *memory-bound as a whole problem*? — a question about the global arithmetic intensity of the problem vs the machine's ops-to-byte ratio.
2. Can the Tensor Core be kept fully busy *cycle by cycle inside the SM*? — a question about whether each tile's operands arrive before the MMA on the previous tile finishes.

Confusing these two is the single most common mistake in back-of-the-envelope GEMM roofline reasoning.

### The whole-problem roofline

For a GEMM $C = AB$ with $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$ at fp16, total work is $2 M N K$ FLOPs and the *minimum* bytes that must cross HBM — every element read at most once — is $2(MK + KN + MN)$. The global arithmetic intensity is therefore

$$
\text{AI}_{\text{GEMM}} = \frac{2 M N K}{2(M K + K N + M N)} = \frac{M N K}{M K + K N + M N},
$$

which grows roughly as $\min(M, N, K)$ (see NVIDIA's *Matrix Multiplication Background User's Guide* for the same formula). For $M=N=K=4096$ this is $\approx 1365$ FLOP/B, about $5\times$ the HBM break-even of $2.25\,\text{PFLOPS}/8\,\text{TB/s} \approx 281$ FLOP/B. Typical LLM GEMMs (K in the thousands) are not even close to memory-bound in the whole-problem sense.

### Why the problem-level answer is not the kernel-level answer

Being compute-bound as a problem does not automatically make a kernel compute-bound. What the kernel actually has to do is feed the SM's Tensor Core one tile after another, and if the *time to bring the next tile's operands into SMEM* exceeds the *time to compute on the current tile*, the Tensor Core stalls regardless of how favourable the global intensity looks.

For a single $M \times N$ output tile with inner-K chunk $K_c$, the tile-level intensity is

$$
\text{AI}_{\text{tile}} = \frac{2 M N K_c}{2(M + N) K_c} = \frac{M N}{M + N}.
$$

For $128\times128$ that is 64 FLOP/B; for $256\times256$, 128 FLOP/B. Neither clears 281. So if *every* tile load came as a cold miss from HBM, you would have to grow $M = N$ to roughly $562$ just to break even — impossible in practice.

### The L2 cache is what makes small tiles viable

Blackwell's B200 has on the order of 100 MB of L2 (NVIDIA's Blackwell tuning guide lists 126 MB on GB200), and the L2's delivered bandwidth into the SM array is several times HBM's. A sensibly scheduled GEMM reuses heavily at L2: along a row of the output grid every tile shares the same A-rows; along a column every tile shares the same B-columns. Persistent tile schedulers (introduced in :numref:`chap_persistent_kernel`) visit tiles in an order that keeps these hot operands resident. Measured L2 hit rates for well-scheduled GEMMs on B200 are typically 70–90%, so the *effective* bandwidth an SM sees sits much closer to L2 bandwidth than to HBM bandwidth.

The practical consequence is that the $MN/(M+N) \geq 281$ inequality is the wrong condition to design around. It is the HBM-cold-miss worst case. The condition that actually matters is: can the *L2-hit* tile-load time be hidden behind the tile-compute time? Because L2 bandwidth per SM is substantially higher than HBM bandwidth per SM, 128×128 tiles (with 64 FLOP/B) are fully viable once L2 reuse is working.

### What TMA adds on top of the L2

Even when operands are in L2, moving them into SMEM still takes real time, and the Tensor Core cannot consume operands until they live in SMEM (or, for the accumulator, TMEM). Without asynchronous copy the kernel has to serialize:

```
load tile k -> MMA on tile k -> load tile k+1 -> MMA on tile k+1 -> ...
```

and the Tensor Core sits idle during every load interval. With TMA the kernel can overlap:

```
time t   : MMA(k)       || TMA(k+1 into stage s+1)
time t+1 : MMA(k+1)     || TMA(k+2 into stage s)
...
```

and Tensor Core utilization is governed by $\max(\text{compute time}, \text{load time})$ rather than their sum. On top of this, TMA is efficient per transaction: a single `cp.async.bulk.tensor` moves a full $128\times 64$ fp16 tile (16 KB) in one instruction issued by one elected thread, which leaves the other 127 threads in the warpgroup free to run a different MMA, an epilogue, or a softmax on a previous tile. Software pipelining (:numref:`chap_gemm_async` Step 5) and warp specialization (:numref:`chap_gemm_advanced` Step 7) are the software patterns that turn this hardware capability into sustained throughput. By the end of Part III you will see a kernel issuing a TMA load for tile $k+2$ while the MMA runs on tile $k+1$ and the writeback reads tile $k$ from TMEM — three hardware units working simultaneously on three consecutive tiles.

### Takeaways

- **Problem-level AI grows with $\min(M,N,K)$.** Large GEMMs are compute-bound against the HBM roofline; the 281 FLOP/B break-even tells you when a problem *could* be compute-bound, not whether a given kernel *is*.
- **Tile-level AI of $MN/(M+N)$** is the intensity you would see if every tile were a cold HBM miss; it is a worst-case bound, not the operating point.
- **L2 reuse is the main lever** that lets 128×128 tiles (64 FLOP/B) approach peak throughput. Tile-size alone is not the knob that decides whether the kernel is bandwidth-bound.
- **TMA is what turns "compute-bound problem" into "compute-bound kernel"** by overlapping whatever tile-load latency remains with Tensor Core compute.


## mbarrier (Memory Barrier)

mbarriers are hardware synchronization primitives stored in shared memory. They combine a counter with a phase bit to enable reusable, asynchronous synchronization.

**Lifecycle of an mbarrier:**

1. **Init**: Set the expected number of arrivals. The barrier starts at phase 0.
2. **Arrive**: Each arrival decrements the counter. There are three ways to arrive:
   - **TMA auto-arrive**: The hardware arrives automatically once the byte transfer completes (tracked via `expect_tx`).

   - **tcgen05 auto-arrive**: The hardware arrives once committed MMAs complete.

   - **Thread arrive**: A thread arrives explicitly (used by writeback threads to signal "TMEM is free").
3. **Wait**: A consumer waits until the barrier reaches the expected phase for that iteration, which indicates that all required arrivals have completed.
4. **Phase flip**: Once all arrivals are done, the barrier automatically toggles its phase (0 → 1 → 0 → ...). This lets the same barrier be reused across loop iterations: iteration 0 uses phase 0, iteration 1 uses phase 1, iteration 2 uses phase 0 again, etc.

This is the basic producer-consumer pattern used throughout the tutorial: a producer (for example, TMA) arrives on a barrier when data is ready, and the consumer waits on that barrier before using the data.


## Synchronization Rules

Blackwell has multiple asynchronous hardware units (threads, TMA, tcgen05 MMA) that read and write different memory spaces. Whenever data crosses from one unit or memory space to another, you need explicit synchronization:

| Data flow | Synchronization needed |
|:-:|:-:|
| Threads write SMEM → MMA reads SMEM (sync copy) | `cta_sync()` |
| TMA writes SMEM → MMA reads SMEM (async copy) | `mbarrier.try_wait` on the TMA completion barrier |
| MMA writes TMEM → Threads read TMEM (epilogue) | `mbarrier.try_wait` (optionally followed by `fence.after_thread_sync()`) |
| Threads write SMEM → TMA reads SMEM (store) | `fence.proxy_async("shared::cta")` |
| Alloc barriers/TMEM → Use them | `fence.proxy_async` + `fence.mbarrier_init` + `cta_sync()` |
| All work done → Deallocate TMEM | `cta_sync()` (single-CTA) or `cluster_sync()` (cluster) |

A few words on why this table is shorter than it looks. `cta_sync()` is a thread barrier *and* a release/acquire fence over shared memory, so the subsequent MMA issued by the same CTA sees earlier SMEM writes. An mbarrier arrival from an async engine (TMA completion, or `tcgen05.commit` from an MMA) likewise carries release→acquire semantics: once `try_wait` returns, the waiter sees all of the producer's memory effects — including MMA writes to TMEM. In practice the mbarrier alone is enough to make the early-step kernels in this tutorial (and most of CUTLASS) correct without any explicit fence. `fence.after_thread_sync()` is a conservative extra between the MMA-completion wait and the first `tcgen05.ld`, added in some reference kernels (the `tirx-kernels` writeback path used from Step 8 onward) to make the ordering between prior tcgen05 writes and the upcoming thread-proxy reads explicit; it is not needed on the TMA→MMA edge. `fence.proxy_async("shared::cta")` plays the analogous role on the thread→TMA-store edge in SMEM, where no mbarrier is involved.

Before deallocating TMEM, all CTAs that may still read or write it must be finished. For a single-CTA kernel, `cta_sync()` is sufficient; for clustered kernels, `cluster_sync()` is required.


## CTA Clusters

Blackwell supports **CTA clusters**: groups of CTAs that can cooperate more tightly than independent thread blocks. Key capabilities include:

- **shared::cluster memory**: CTAs in the same cluster can access each other's shared memory through the `shared::cluster` address space, typically using `map_shared_rank` to obtain the remote pointer.

- **Multicast TMA**: A single TMA command can deliver the same data to multiple CTAs simultaneously, reducing global memory bandwidth.

- **Cross-CTA barrier signaling**: mbarrier arrive/wait can be used across CTAs in the same cluster, enabling producer-consumer synchronization beyond a single CTA.

Clustering gives hardware units such as tcgen05 MMA access to every CTA's shared memory in the cluster, so a single MMA instance can read an operand tile that spans multiple SMs' SMEM and grows the effective MMA working set beyond one SM's 228 KB budget. The global-memory bandwidth used is still the sum of each CTA's own loads — cooperative MMA itself does not reduce GMEM traffic. TMA *multicast* is the complementary feature that actually does save GMEM bandwidth, by delivering one GMEM tile to every CTA in the cluster from a single load.


## Numbers to Remember

Throughout Part III we benchmark kernels against the Blackwell hardware limits. Having these reference numbers in working memory makes "is this kernel fast?" a calculation you can do in your head.

| Quantity | Value (B200, approx.) | Where it shows up |
|:--|:-:|:--|
| Streaming multiprocessors per GPU | 148 | grid sizing, persistent schedulers |
| Shared memory per SM | 228 KB | SMEM budget for pipeline depth |
| Tensor memory per SM | 128 lanes × 512 cols (32-bit) | TMEM accumulator budget |
| Registers per thread (max) | 255 × 32-bit | why accumulators moved off-RF |
| FP16/BF16 Tensor Core throughput | ~2.25 PFLOPS (dense) | GEMM roof |
| FP8 / FP4 Tensor Core throughput | ~4.5 / ~9 PFLOPS (dense) | mixed-precision GEMM |
| HBM3e bandwidth | ~8 TB/s | bandwidth roof |
| L2 cache | ~100 MB | where tile operands actually live after warm-up |
| HBM cold-miss break-even AI | ~281 FLOP/B | only binds if L2 reuse fails |
| One fp16 128×64 tile in SMEM | 16 KB | TMA transfer granularity |

Two rules of thumb follow directly from the table:

- **Tile choice.** A $128 \times 128$ fp16 tile has tile-level intensity $MN/(M+N) = 64$ FLOP/B. That is well below the 281 HBM cold-miss break-even, but this is only a problem when L2 reuse is absent; in a properly scheduled persistent kernel most tile loads hit L2, so 128×128 tiles routinely approach peak. Larger tiles ($256 \times 256$ via 2-CTA clustering) help mainly by raising occupancy of each MMA issue and reducing control overhead, not by crossing a magic AI threshold.

- **SMEM budget.** A 4-stage pipeline with two operand tiles plus a (smaller, non-pipelined) output tile easily consumes 150–220 KB, out of 228 KB per SM, once barriers and metadata are included. That fits but leaves limited slack; this is why Step 5 of the GEMM journey has to think carefully about `PIPE_DEPTH`.

We will revisit these numbers whenever a kernel's measured throughput is reported. If you find yourself without intuition for a benchmark result, come back here first.
