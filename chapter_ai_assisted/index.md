# AI-Assisted Kernel Development
:label:`chap_ai_assisted`

*Prerequisites: You should have completed at least through Step 7 (Warp Specialization). The examples in this chapter reference barrier patterns, tcgen05 MMA, and TMA pipelining from earlier chapters.*

---

Writing GPU kernels is slow, detail-heavy work. A single misplaced barrier arrive or a wrong descriptor field can produce silent corruption that takes hours to debug. Large language models (LLMs) like Claude can accelerate parts of this workflow --- generating boilerplate, spotting known bug patterns, and producing test harnesses --- but they also have limitations that you should understand before relying on them.

This chapter is practical, not theoretical. We walk through specific workflows where LLMs help, show a concrete debugging example, and catalog common failure modes so you know when to be cautious with LLM suggestions and when to consult the hardware manual directly.


## What LLMs Tend to Do Well — and Where They Need More Care

| Works Well | Needs Careful Verification |
|-----------|-----------|
| Generating boilerplate (e.g., SMEM allocation, barrier init, TMA config dictionaries) | Reasoning about barrier phase state across loop iterations (easy to get off by one) |
| Spotting common bug patterns (e.g., missing `elect_sync` guard, wrong barrier count, `cta_sync` inside a warpgroup branch) | Predicting SMEM bank conflicts or performance behavior |
| Explaining existing kernel code (e.g., "what does this writeback section do?") | Designing pipeline topologies from scratch (e.g., barrier grouping and direction) |
| Generating NumPy/Torch reference implementations for verification | Getting TMEM layout details exactly right (TileLayout, TLane/TCol, allocated_addr) |
| Translating between kernel styles (e.g., CUTLASS-style pseudocode to TIRX) | Handling TIRX-specific API quirks without explicit context |

The general pattern is that LLMs are strong at **matching familiar code patterns**, but less dependable for **multi-step reasoning over concurrent state**. Barrier protocols, phase tracking, and TMEM layout arithmetic are exactly the kinds of stateful details that usually require manual checking.


## Effective Prompting for TIRX Kernels

A key practical insight is that **output quality depends heavily on context quality**. A vague prompt ("my kernel is broken, help") gets a vague answer. A structured prompt with hardware constraints, code, and symptoms gets a precise diagnosis.

Here is a prompt template that works well for TIRX debugging:

```
I'm writing a TIRX kernel for [operation]. Target: Blackwell B200 (SM100a).

Hardware constraints:
- TMEM: 128 rows x 512 cols, layout TileLayout(S[(128,512) : (1@TLane, 1@TCol)])

- tcgen05.commit is per-thread; must be inside elect_sync() guard

- MBarrier.init() checks threadIdx.x < 1 -- only works from warp 0

- mbarrier arrives from async engines (TMA completion, tcgen05.commit) carry release→acquire ordering; after try_wait returns, the waiter sees the producer's memory effects. Extra fences on the TMA→MMA edge are unnecessary.

- `fence.after_thread_sync()` is rarely needed. The MMA-completion mbarrier already orders MMA writes against downstream reads, so most kernels (including CUTLASS) omit it. The `tirx-kernels` reference kernels take a conservative stance and insert it in the writeback path after `mma2ld.wait(phase)` and before the first `tcgen05.ld`; if you are mirroring that style, add it there.

Here is my kernel:
[paste code]

Symptom: [zeros / NaN / deadlock / wrong values at specific indices]
Question: [specific question]
```

Why each piece matters:

- **"Target: B200 SM100a"** prevents the model from suggesting Hopper (SM90) or Ampere (SM80) patterns. Without this, you may get `cp.async` suggestions instead of `tcgen05`.

- **Hardware constraints** help prevent the model from suggesting wrong API semantics. LLMs have limited TIRX training data and may generate plausible-sounding but incorrect behavior.

- **Symptom** narrows the search space dramatically. "Output is zeros" points to TMEM read-before-write. "Deadlock" points to barrier mismatch. "NaN" points to uninitialized memory or softmax overflow. "Wrong values at specific row offsets" points to descriptor or swizzle misconfiguration.

- **Specific question** forces a focused answer instead of a broad code review that buries the real issue.


## Symptom-to-Cause Quick Reference

Before prompting an LLM, it helps to narrow down the category of bug yourself. This table maps common symptoms to likely root causes:

| Symptom | Likely Cause | What to Check |
|---------|-------------|---------------|
| Output is all zeros | TMEM read before MMA completes | `tcgen05.commit` outside `elect_sync`; missing `mma2ld.wait()` on the writeback path |
| Output is random garbage | SMEM overwritten before MMA reads it | Barrier count too high (premature arrive); wrong pipeline stage index |
| Deadlock (kernel hangs) | Barrier wait with no matching arrive | Missing arrive on one code path; phase init mismatch; `MBarrier.init` from wrong warp |
| NaN in output | Uninitialized memory or overflow | Missing `accum=False` on first MMA iteration; softmax without max subtraction |
| Correct values in wrong positions | Descriptor or swizzle misconfiguration | Wrong SwizzleLayout parameters; SMEM alignment < 1024; wrong TMA descriptor shape |
| Correct for small sizes, wrong for large | Off-by-one in tile/block indexing | Boundary condition in K-loop count; wrong tile scheduler dimensions |
| Non-deterministic correctness | Race condition | Missing barrier between producer and consumer; `cta_sync` vs `cluster_sync` confusion |

Include this table (or a simplified version) in your prompt when debugging. It helps the LLM focus on the right area.


## Demo: Debugging a Broken Kernel

Consider this MMA loop from a warp-specialized GEMM (Step 7 style). This code runs inside the MMA consumer warp (`wg_id == 1, warp_id == 0`). There is one deliberate bug:

```python
# Broken MMA loop -- one bug
for k in range(K_TILES):
    tma2mma.wait(mma_ps.stage, mma_ps.phase)
    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
        Tx.gemm_async(
            tmem[:, :BLK_N],
            Asmem[mma_ps.stage],
            Bsmem[mma_ps.stage],
            accum=(k != 0),
            dispatch="tcgen05",
            cta_group=1,
        )
    # BUG: arrive is outside the elected-thread scope
    mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
    mma_ps.move_to_next_stage()
```

**Symptom:** Output is all zeros or random garbage, non-deterministically.

**Root cause:** `mma2tma.arrive()` internally calls `tcgen05.commit`, which creates a commit group for the calling thread. But `tcgen05.commit` is a **per-thread** instruction, not a warp-cooperative one. Only the elected thread (the one that issued the MMA via `elect_sync`) has a non-empty commit group. The other 31 threads in the warp each create an empty commit group that signals the mbarrier immediately.

The result: 31 empty commit groups fire the mbarrier almost instantly, well before the MMA completes. The TMA producer sees the barrier as arrived and starts overwriting SMEM with the next tile. The MMA is still reading the old tile from SMEM --- data corruption.

**How to prompt the LLM:** Paste the snippet above and add:

```
Output is zeros. The MMA uses elect_sync but the arrive() call does not.
Is this correct?
```

This gives the model the symptom (zeros), the structural observation (mismatch between elect_sync scope and arrive scope), and a yes/no question. A good model should be much more likely to identify the commit-group issue with this level of context.

**Fix:** Move the arrive inside the `elect_sync` guard:

```python
# Fixed MMA loop
for k in range(K_TILES):
    tma2mma.wait(mma_ps.stage, mma_ps.phase)
    with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
        Tx.gemm_async(
            tmem[:, :BLK_N],
            Asmem[mma_ps.stage],
            Bsmem[mma_ps.stage],
            accum=(k != 0),
            dispatch="tcgen05",
            cta_group=1,
        )
        # Now only the elected thread (with a non-empty commit group) arrives
        mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
    mma_ps.move_to_next_stage()
```

This is a very common bug pattern in Blackwell kernel development. It is also one of the easier cases for an LLM to catch, because the task is closer to recognizing a familiar structural mismatch than to reasoning through full phase-state behavior.


### Real comparison: weak prompt vs strong prompt

We tested the same bug with two different prompts using Claude Opus 4.6. The results illustrate why context matters.

**Weak prompt** (code only, vague question):

```
My GEMM kernel output is all zeros. Here's the MMA loop:
[paste code]
What's wrong?
```

**Model response** (incorrect): The model guessed that `cta_group=1` was wrong and suggested changing it to `cta_group=2`. It never identified that `mma2tma.arrive()` was outside the `elect_sync` scope — the actual bug.

**Strong prompt** (with hardware constraints template from this chapter):

```
I'm writing a TIRX kernel for GEMM. Target: Blackwell B200 (SM100a).

Hardware constraints:
- tcgen05.commit is per-thread; must be inside elect_sync() guard
- ...

Symptom: Output is all zeros or random garbage, non-deterministically.
Question: Is mma2tma.arrive() in the correct scope? It uses tcgen05.commit internally.
```

**Model response** (correct): *"Yes, the scope is wrong. `mma2tma.arrive()` uses `tcgen05.commit` internally, and `tcgen05.commit` is per-thread and must be inside an `elect_sync()` guard. In your code, `mma2tma.arrive()` is outside the elected-thread block, so all 32 threads create empty commit groups that signal the mbarrier immediately."*

The weak prompt led the model down a wrong path (`cta_group` mismatch). The strong prompt — with the hardware constraint that `tcgen05.commit` is per-thread — let the model immediately identify the real issue. **The quality of your context determines whether the model finds the bug or invents a plausible-sounding wrong answer.**


### LLM output red flags

When reviewing LLM-generated TIRX code, watch for these signals that the model is guessing rather than reasoning from correct knowledge:

1. **Mixing Hopper and Blackwell APIs** — suggests `cp.async` (Hopper) instead of `tcgen05` / TMA (Blackwell), or references `wgmma` instead of `tcgen05.mma`.
2. **Treating `elect_sync()` as warpgroup-level** — `elect_sync()` elects one thread per *warp* (32 threads), not per warpgroup (128 threads). With 4 warps you get 4 elected threads.
3. **Inventing TIRX APIs** — names like `Tx.barrier()`, `Tx.async_copy()`, or `Tx.shared_memory()` that don't exist. Always verify function names against the API Reference.
4. **Wrong barrier arrival counts** — e.g., suggesting `init(128)` for a barrier that only one elected thread arrives on, or `init(1)` for a barrier where all 128 warpgroup threads arrive.
5. **Sprinkling `fence.after_thread_sync()` everywhere** — models trained on older examples often insert this fence on every MMA boundary (e.g. between TMA wait and MMA issue). That is wrong: the TMA-completion and MMA-completion mbarriers already carry release→acquire ordering, and the fence is not required for correctness on those edges. The fence *only* appears in the writeback path of some conservative reference kernels (between `mma2ld.wait` and the first `tcgen05.ld`); if the model adds it elsewhere, strip it.


## Generating Test Harnesses

One of the most reliable uses of LLMs in kernel work is generating reference implementations. The model does not need full hardware understanding to be useful here --- it just needs to implement math correctly.

**Prompt example:**

```
Generate a numpy reference for this TIRX kernel signature:
- Operation: RMSNorm

- Input: x [batch, hidden_size] float16, weight [hidden_size] float16

- Output: out [batch, hidden_size] float16

- Formula: out[i] = x[i] / RMS(x) * weight[i], RMS(x) = sqrt(mean(x^2) + eps)
Use float64 for intermediate computation to avoid numerical issues.
```

**Expected output:**

```python
import numpy as np

def rmsnorm_ref(x_f16, weight_f16, eps=1e-6):
    """Numpy reference for RMSNorm. Computes in float64 for precision."""
    x = x_f16.astype(np.float64)
    w = weight_f16.astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    out = (x / rms) * w
    return out.astype(np.float16)
```

This is straightforward for most LLMs to produce correctly. The float64 intermediate trick avoids false mismatches from fp16 rounding in the reference.

For GEMM verification, a torch reference on CUDA is even simpler:

```python
import torch

# Reference for GEMM verification
M, N, K = 128, 256, 64
A = torch.randn(M, K, dtype=torch.float16, device="cuda")
B = torch.randn(N, K, dtype=torch.float16, device="cuda")
D_ref = A @ B.T  # cuBLAS reference, float16 accumulation
```

The torch path calls cuBLAS under the hood, giving you a hardware-verified reference.

For FP8 kernels with block scaling, the reference is more involved. This is where LLM assistance pays off --- ask it to handle the dtype conversion boilerplate:

```python
import ml_dtypes
import numpy as np

def fp8_block_scaled_gemm_ref(A_fp8, B_fp8, sfa, sfb, block_size=128):
    """Reference for FP8 block-scaled GEMM. A_fp8 and B_fp8 are int8 views."""
    A = A_fp8.view(ml_dtypes.float8_e4m3fn).astype(np.float64)
    B = B_fp8.view(ml_dtypes.float8_e4m3fn).astype(np.float64)
    # Apply per-block scale factors
    M, K = A.shape
    N = B.shape[0]
    for m in range(0, M, block_size):
        for k in range(0, K, block_size):
            A[m:m+block_size, k:k+block_size] *= sfa[m // block_size, k // block_size]
    for n in range(0, N, block_size):
        for k in range(0, K, block_size):
            B[n:n+block_size, k:k+block_size] *= sfb[n // block_size, k // block_size]
    return (A @ B.T).astype(np.float16)
```

For higher precision, accumulate in float32:

```python
D_ref = (A.float() @ B.float().T).half()  # float32 accumulation, cast back
```

Use these references with three-way comparison: your TIRX kernel output vs numpy reference vs torch/cuBLAS. If all three agree, you have high confidence. If your kernel disagrees with both references, the bug is in your kernel. If numpy and torch disagree with each other, check your accumulation precision.


## When to Be Cautious with LLM Suggestions

These are specific failure modes we have observed in practice. Treat them as a checklist.

**1. Phase tracking.**
Current LLMs often struggle with barrier phase initialization. The `is_producer` flag, the initial phase value, and the skip-first-wait pattern interact subtly. If the model suggests a phase init, always verify with this test: "If both producer and consumer start with the same phase, do they deadlock on the first iteration?" If yes, one of them needs to skip the first wait or use a different initial phase.

**2. TMEM layout arithmetic.**
LLMs may suggest wrong TileLayout parameters, forget `allocated_addr`, or confuse physical TMEM rows with logical rows. TMEM has a fixed 128-row x 512-column structure, and the mapping from logical indices to physical rows is hardware-defined. If the model suggests a TMEM layout, verify that `allocated_addr` is set correctly and that the `TileLayout` dimensions match the declared buffer shape.

**3. TIRX API quirks.**
LLMs have limited TIRX training data and may confuse TIRX APIs with CUTLASS or Triton equivalents. Examples of details that may not be represented reliably in the model's prior knowledge:
- `MBarrier.init()` internally uses `Tx.thread()[0:1]`, which checks `threadIdx.x < 1`. This is always false when called from a non-warp-0 context.

- `Tx.copy_async` from TMEM to local dispatches `tcgen05.ld` but does NOT emit `tcgen05.wait.ld()`. If you read the register immediately, you get stale data.

- `elect_sync()` elects one thread per **warp**, not per warpgroup. With 4 warps, you get 4 elected threads. LLMs sometimes conflate this with `Tx.thread()[0:1]` (one per warpgroup).

If an API call in the model's suggestion looks unfamiliar, grep the TIRX codebase to verify it exists before using it.

**4. Performance advice.**
"Use `PIPE_DEPTH=8` for better performance" --- LLMs generally cannot reason about the SMEM budget vs occupancy tradeoff. Deeper pipelines use more SMEM, which may reduce occupancy, which may hurt performance. Always profile with `torch.cuda.Event` timing or Proton. Performance intuition from an LLM should be verified empirically.


## Building a Persistent Context File

One of the most effective techniques for working with LLMs on kernel code is maintaining a **persistent context file** --- a plain text file that accumulates hard-won debugging lessons. Every time you discover a non-obvious bug, add a short entry:

```
### tcgen05.commit Must Use elect_sync
tcgen05.commit is per-thread, not warp-cooperative. If all 32 threads call it,
only the elected thread has a non-empty commit group. The other 31 create empty
commit groups that signal the mbarrier immediately.
Fix: Always wrap TCGen05Bar.arrive() calls inside an elected-thread scope:
with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:
```

Then include this file (or relevant excerpts) in your prompt context. This has two benefits:

1. **The LLM learns from your past bugs.** It will not suggest patterns you have already proven incorrect.
2. **You build institutional knowledge.** Six months later, when you hit a similar bug in a different kernel, the context file reminds both you and the LLM of the solution.

Here is a minimal starter template (`tirx_bugs.md`):

```markdown
# TIRX Bug Log (Blackwell SM100a)

### tcgen05.commit outside elect_sync
Symptom: zeros or garbage, non-deterministic.
Cause: tcgen05.commit is per-thread. 31 empty commit groups signal mbarrier early.
Fix: wrap TCGen05Bar.arrive() inside `with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:`.

### MBarrier.init from wrong warp
Symptom: kernel hangs (deadlock).
Cause: init() uses threadIdx.x < 1 internally. If called from wg_id==1, thread 0 is in WG0.
Fix: call init at CTA level (before any wg_id branch), or ensure warp_id==0 in WG0.

### fence.after_thread_sync is rarely needed (don't over-add it)
Symptom: model-generated code sprinkles fence.after_thread_sync() around every
         MMA, or you see it on the TMA->MMA edge.
Cause: mbarrier arrives from async engines already carry release->acquire ordering,
       so fences on the TMA->MMA edge and after cta_sync-guarded SMEM copies are
       unnecessary. Most production kernels (including CUTLASS) omit the fence
       entirely; only tirx-kernels' conservative writeback path inserts it.
Fix: strip stray fence.after_thread_sync() calls. If you are mirroring tirx-kernels,
     keep exactly one - after mma2ld.wait() and before the first tcgen05.ld.
```

Format tips:
- One bug per entry, under 5 lines each
- Include **symptom**, **root cause**, and **fix**
- Include hardware-specific details (SM version, instruction name)
- Paste relevant entries into your prompt when debugging similar issues


## Practical Workflow Summary

![AI-Assisted Kernel Development Workflow](../img/ai_assisted_workflow.png)

A workflow that combines human expertise with LLM assistance:

1. **Design** the kernel yourself: warp roles, barrier topology, pipeline depth. Kernel design should still be driven by the programmer, with the LLM used mainly as an assistant rather than the source of the design.
2. **Generate boilerplate** with the LLM: SMEM allocation, TMA descriptor setup, barrier initialization. These are tedious and mechanical.
3. **Write the core logic** yourself: MMA dispatch, softmax, reductions. You need to understand what every line does.
4. **Generate test harnesses** with the LLM: numpy/torch references, input generation, comparison logic.
5. **Debug** with the LLM: paste the failing code + symptom. LLMs tend to be good at pattern-matching known bugs (missing `elect_sync`, wrong barrier count, fence ordering), but less reliable for novel bugs.
6. **Profile** yourself: LLMs generally cannot tell you whether your kernel is fast. Use timing tools.

The goal is not to have the LLM write your kernel. The goal is to spend your time on the hard parts (design, core logic, performance) and let the LLM handle the tedious parts (boilerplate, references, known-bug detection).


## Exercises

1. Take your Step 7 kernel and introduce the `elect_sync` bug from this chapter: move `mma2tma.arrive()` outside the `with Tx.thread(parent="warp")[Tx.ptx.elect_sync()]:` block. Run the test to confirm it produces wrong output. Then paste the buggy code to an LLM with the symptom description. Does the model identify the root cause?

2. Ask an LLM to generate a numpy reference for the RMSNorm kernel from the RMSNorm chapter. Compare its output with your kernel's output on the same input. Do the results match to within fp16 tolerance?

3. Ask an LLM to explain the barrier flow in Step 8 (2-CTA cluster): "Each CTA's TMA warp loads both its own A tile and B tile into local SMEM, then CTA-0's MMA warp issues the cooperative MMA that reads B from both CTAs' SMEM. Trace which barriers fire in what order." Verify the explanation against the actual code.

4. Try asking an LLM to write a complete Step 7 kernel from scratch. Give it only the API reference from the API Reference chapter and the warp role table from Step 7 (Warp Specialization). What does it get right? What does it get wrong? This exercise illustrates the boundary between what LLMs tend to do well and where they need more care.

> **Try with your agent**: Paste your most confusing kernel bug (the one that took you the longest to find) and describe only the symptom. See if the model can identify the root cause. Then try again with the hardware constraints template from this chapter. Compare the quality of the two responses.
