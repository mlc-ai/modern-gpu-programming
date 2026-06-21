(chap_ai_assisted)=
# Writing TIRx Kernels with Agents

:::{admonition} Overview
:class: overview

- Coding agents get TIRx wrong by default, so you must ground them in the real `tvm` / `tirx-kernels` source and a contract prompt.
- The chapter gives a workflow plus five concrete use cases (explain, review, debug, generate tests, inspect CUDA) and where agent review stops being trustworthy.
:::

A coding agent is only as good as what it knows about your problem, and TIRx is exactly the kind of niche, fast-moving target a general model gets wrong by default. Ask one to write a Blackwell kernel cold and it will produce confident, plausible-looking code: intrinsics that do not exist, barrier patterns borrowed from Hopper, dispatch paths that never lower the way it claims. The model has read far more generic CUDA and Triton than TIRx, so that is what it falls back on. The payoff comes from closing that gap: ground the agent in the real `tvm` and `tirx-kernels` source and hand it a tight contract, and the same model becomes a fast, reliable partner for reading and changing Blackwell kernels. This chapter shows how to do that grounding, walks through five concrete use cases where it pays off, and marks the line where agent help stops being trustworthy.

Once the agent can read the right code, there are two ways to talk to it, and the difference matters more than it first appears. The first is delegation. You hand over a broad goal, "make the FA4 barrier section easier to understand," and let the agent decide how to get there. This is fine for mechanical edits once you already know the direction you want, but it teaches you very little, because all the interesting choices stay hidden inside the agent. The second way is learning-oriented. Here you sharpen the broad goal into a specific instruction, something like "explain `softmax_corr.full` and `softmax_corr.empty` as a mailbox-slot lifecycle, keep the value-MMA gate in a separate diagram, then rebuild the tutorial." For TIRx the second approach is almost always the better investment. The real work is not changing text or code; it is learning which choices are even possible and which hardware contracts each of those choices implies.

Of course, you cannot write a specific instruction when you do not yet know what the right instruction is. When that happens, turn the discovery itself over to the agent and ask it for candidates first. A prompt like this works well:

```text
I want to make the FA4 barrier section clearer, but I do not know the best form yet.
Give me 3 candidate rewrites. For each one, say what it explains well,
what it hides, and what code or diagram evidence I should verify.
Do not edit yet.
```

Once you have read the candidates, you pick one and turn it into an explicit instruction. The whole point of this round-trip is that the next time you meet the same kind of problem, you can skip it and write the instruction yourself. That is what makes agents valuable while you are still learning TIRx: they help you discover what is possible, and they convert vague goals into reusable engineering moves.

To do this reliably, though, you need a unit of interaction that both you and the agent can be precise about. The previous chapters built up a consistent way to read Blackwell kernels, identify the tile path, then check scope, layout, dispatch, and synchronization, and that same structure is exactly what carries you from a vague goal to a useful agent instruction. The unit is not a Python function, and it is not a whole CUDA file; it is a TIRx kernel contract. TIRx kernels are full of these local contracts:

- which scope owns a tile operation,
- where each tile lives and how it is laid out,
- which dispatch path is requested,
- which barrier proves an async producer is complete,
- and whether the generated CUDA matches the intended hardware path.

These contracts, together with the source references above, are the context an agent actually needs. Give it that, and it can help you draft, compare, and execute choices, but the programmer still owns the final contract. Keep that division of labor in mind for the rest of the chapter.

## Workflow

![Writing TIRx Kernels with Agents Workflow](../img/ai_assisted_tirx_workflow.png)

Put these ideas together and they form a loop:

1. Point the agent at `tvm` and `tirx-kernels`.
2. Start with the goal, but do not stop there.
3. If the path is unclear, ask for candidate strategies and tradeoffs.
4. Choose one strategy and rewrite it as a concrete TIRx instruction: tile path, roles, layouts, barriers, expected checks.
5. Ask the agent to execute, explain, or review one local contract at a time.
6. Confirm with generated CUDA, tests, and benchmarks.
7. Record the instruction pattern you learned so future prompts start more precise.

The agent is at its best when it helps you sharpen the instruction. Let it propose possibilities, but never let it silently decide the hardware contract.

## From Goal to Instruction

The whole loop hinges on one step: turning a goal into an instruction sharp enough to act on. For tutorial writing, the progression looks like this:

```text
Broad goal:
Make the FA4 barrier section easier to understand.

Candidate prompt:
Give me three ways to explain softmax_corr.empty:
1. a full barrier DAG,
2. a mailbox slot lifecycle,
3. a wait/arrive table.
For each option, say which misunderstanding it prevents and which detail it hides.
Do not edit yet.

Final instruction after choosing:
Use the mailbox slot lifecycle. Explain that softmax_corr.full means
WG2 may read acc_scale or row_sum, while softmax_corr.empty means
softmax may reuse that SMEM slot. Keep p_o_rescale separate
because it gates value MMA, not the scale slot.
```

Kernel work follows the very same pattern:

```text
Broad goal:
Make this GEMM faster.

Candidate prompt:
List three possible next optimizations for this Step-4-style GEMM.
For each one, name the tile path change, the new barriers, the SMEM/TMEM cost,
and one generated-CUDA check. Do not edit yet.

Final instruction after choosing:
Implement PIPE_DEPTH=2 software pipelining. Add one SMEM stage dimension
to A/B, use one TMA barrier per stage, prefetch the first two stages,
then in the K loop wait current stage, run MMA, and prefetch the next tile
into the stage being released. Explain the phase flips before editing.
```

In both cases the broad goal is identical to the one you started with. What changed is that you now name the tile path, the barriers, and the checks. This is what turns the agent into a learning tool: you walk away not just with a patch, but with a feel for the vocabulary of patches that were available to you.

## The TIRx Contract Prompt

The next step is to make that vocabulary explicit in the prompt itself. When you ask an agent about a TIRx kernel, resist the urge to start with a code dump. Start with the kernel contract instead.

Every field in the contract earns its place. The tile path lays out the data flow. After it come the same three pillars you saw on the per-step cards of the GEMM and Flash Attention chapters: **scope** (roles) says who executes each tile operation, **layout** says where its tiles live, and **dispatch** says which hardware path it lowers to. **Barriers**, finally, say which producer-consumer edges make the async work safe to consume. The example below is the kind of prompt you can derive directly from the warp-specialized GEMM chapter.

```text
Target: NVIDIA Blackwell SM100a.
Kernel: multi-consumer warp-specialized GEMM (Step 9, {ref}`chap_gemm_advanced`).

Tile path:
GMEM -> SMEM by TMA.
SMEM -> TMEM by tcgen05 MMA.
TMEM -> RF -> SMEM -> GMEM by epilogue.

Scopes and roles:
WG2 warp 3: TMA producer.
WG2 warps 0-1: MMA consumers.
WG0-WG1: writeback.

Layouts:
SMEM A/B use tma_shared_layout(...).
TMEM accumulator uses TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]).
RF readback uses TileLayout(S[(128, N) : (1@tid_in_wg, 1)]).

Dispatch:
TMA for GMEM<->SMEM loads and stores.
tcgen05 for the MMA; tcgen05.ld for the TMEM->RF readback.

Barriers (the names from {ref}`chap_gemm_advanced`):
tma2mma: TMA -> MMA (A/B SMEM stage loaded).
mma2tma: MMA -> TMA (SMEM stage free to refill).
mma2ld: MMA -> writeback (accumulator ready).
ld2mma: writeback -> MMA (accumulator slot free).

Symptom:
kernel hangs after the first output tile.

Question:
Which handoff should I inspect first?
```

A prompt like this hands the agent the same map a human reviewer would build in their head before reading a line of the code. Without it, the agent is left to guess from generic CUDA, Triton, or Hopper patterns, and it will usually miss the issue that is specific to Blackwell.

## Case Study: Elected MMA Commit Bug

To see why the contract matters, consider a bug that it catches but a raw code dump does not, a case where the prompt needs a hardware fact, not just code. The broken loop below comes from a warp-specialized GEMM. The MMA is issued by a single elected thread, but the barrier arrive has slipped outside the elected-thread scope:

```python
for k in range(K_TILES):
    tma2mma.wait(mma_ps.stage, mma_ps.phase)
    if T.ptx.elect_sync():
        Tx.gemm_async(
            tmem[:, :BLK_N],
            Asmem[mma_ps.stage],
            Bsmem[mma_ps.stage],
            accum=(k != 0),
            dispatch="tcgen05",
            cta_group=1,
        )

    # Broken: this commits from every lane in the warp.
    mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
    mma_ps.advance()
```

In these TIRx wrappers, `TCGen05Bar.arrive()` lowers to `tcgen05.commit`, so the caller has to guard it and make sure only the intended issuer ever calls it. Only the elected thread carries the real MMA work in its commit group. The other lanes create empty commit groups, and those empty groups can signal the mbarrier before the MMA has finished. The TMA producer then feels free to overwrite SMEM while the MMA still needs it, and the result is corruption that looks nothing like the obvious cause.

A useful prompt for the agent is not this:

```text
Why is my GEMM wrong?
```

It is this:

```text
The MMA issue uses elect_sync, but mma2tma.arrive is outside that elected scope.
In these TIRx wrappers, TCGen05Bar.arrive lowers to tcgen05.commit.
Can empty commit groups signal the barrier early?
```

The fix follows naturally: keep the arrive inside the same elected-thread scope as the MMA issue.

```python
for k in range(K_TILES):
    tma2mma.wait(mma_ps.stage, mma_ps.phase)
    if T.ptx.elect_sync():
        Tx.gemm_async(
            tmem[:, :BLK_N],
            Asmem[mma_ps.stage],
            Bsmem[mma_ps.stage],
            accum=(k != 0),
            dispatch="tcgen05",
            cta_group=1,
        )
        mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
    mma_ps.advance()
```

This is the pattern for the rest of the chapter. State the contract, ask the agent to check one edge, then verify the answer against the source, the generated CUDA, and a runnable test. The sections that follow walk through the recurring jobs you will hand an agent (explaining, reviewing, debugging, testing, and reading generated code), and each one is just another instance of this same pattern.

## Use Case 1: Explain a Kernel as Tile Primitives

Explanation is the most basic job you can give an agent, and it also happens to be the one that builds your own understanding fastest. Ask the agent to turn a region of code into a tile-primitive table:

```text
Read this TIRx code and make a table with:
primitive, scope, source tile, destination tile, dispatch path,
barrier waited before the primitive, and barrier signaled after it.
```

A good explanation of a GEMM writeback, for instance, should come back looking like this:

| Primitive | Scope | Source | Destination | Handoff |
|-----------|-------|--------|-------------|---------|
| `Tx.copy_async(Dreg_wg, tmem)` | warpgroup | TMEM accumulator | warpgroup RF tile | wait for MMA first, then `wait.ld()` |
| `Tx.cast(reg_f16, reg_f32)` + `Tx.copy(Dsmem, reg_f16)` | thread | per-thread registers | SMEM staging tile | fence before TMA store |
| `Tx.copy_async(D, Dsmem, dispatch="tma")` | selected TMA issuer | SMEM staging tile | GMEM output | commit the TMA store group and wait before SMEM reuse |

This is a sharper question than "explain this code," because it forces the answer onto the tutorial's mental model. And it gives you a built-in quality check: if the agent cannot fill in the scope, layout, dispatch, or handoff, you know the explanation is incomplete.

Take a single line like `Tx.copy_async(Dreg_wg, tmem)`. A weak explanation says "this copies data" and leaves it there. A useful explanation says far more: this runs under warpgroup scope, reads a TMEM accumulator tile, writes a warpgroup-distributed register view, lowers to four warp-collective `tcgen05.ld` instructions (each warp moves its own 32 TMEM lanes, so together they cover all 128 lanes), and needs `wait.ld()` before the registers can be consumed. That second answer is the one that teaches you something.

## Use Case 2: Review a Kernel Change

Explanation reads code that already works. Review asks a different question: does a change still honor the contract? It works best when the change has a small, checkable contract behind it. A few examples:

- change `PIPE_DEPTH` from 2 to 4,
- change `cta_group=1` to `cta_group=2`,
- add a second MMA consumer,
- replace synchronous copy with TMA,
- move writeback through SMEM and TMA store.

The trick is to ask for invariant checks rather than a broad, open-ended review:

```text
I changed a warp-specialized single-CTA GEMM into the clustered form from {ref}`chap_gemm_advanced` with CTA_GROUP=2.
Check only the cluster invariants:
- tcgen05.alloc / gemm_async / commit / dealloc cta_group values
- scheduler tile shape
- TMA byte count
- remote barrier view
- cluster_sync before cleanup
```

For warp-specialized GEMM, the most useful review questions group by the kind of change you made:

| Change | What to ask the agent to check |
|--------|--------------------------------|
| Increase pipeline depth | barrier array depth, stage index, phase flip point, SMEM budget |
| Add TMA store | RF -> SMEM fence, warpgroup sync, commit group, store wait before SMEM reuse |
| Add CTA cluster | `cta_group`, remote barriers, cluster tile shape, cleanup sync |
| Add consumer | per-consumer barrier slots, TMEM ranges, scheduler M tile factor |

Notice that the agent is not deciding the design here. It is checking whether the code still matches a design you already chose. Take `cta_group=2`: it is not just one keyword on `Tx.gemm_async`. It also ripples through TMEM allocation, MMA issue, commit, deallocation, remote barriers, and the scheduler tile shape, exactly the kind of fan-out that is easy to half-finish and tedious to verify by hand.

## Use Case 3: Debug from Symptoms

Review assumes you already know what changed. Debugging starts from the opposite end: you have a failure, and you work backward to its cause. The discipline, though, is the same one. Classify the symptom first, then ask the agent to map it back to the nearest producer-consumer handoff. The table below is the prompt-level version of that mapping; when you need to inspect the generated CUDA itself, turn to the *TIRx Language and Compile Pipeline* appendix page.

| Symptom | Likely area | First checks |
|---------|-------------|--------------|
| Kernel hangs | missing arrive, wrong phase, wrong init count | barrier producer/consumer pair |
| Output all zeros | TMEM read before MMA completes, early empty commit | MMA commit guard, `mma2ld.wait` |
| Wrong rows in chunks of 128 | warpgroup handoff bug | TMEM readback, writeback barrier, per-consumer slot |
| Random garbage | SMEM reused too early | TMA/MMA empty barrier, TMA store drain |
| NaN everywhere | invalid input to math or wrong descriptor | softmax max subtraction, masks, SMEM layout, MMA descriptor |
| Works for small sizes only | scheduler or boundary bug | tile count, phase reset, tail guard |

A good debugging prompt carries both the symptom and the code around the relevant barrier. For example:

```text
Symptom: output is zeros or random garbage.
The MMA issue uses elect_sync, but mma2tma.arrive is outside that elected scope.
Does this allow an empty commit group to signal the barrier early?
```

This is precisely the kind of question agents handle well. The local contract is spelled out, so the agent only has to check whether the code violates it, with no invention required.

## Use Case 4: Generate Reference Tests

Debugging needs something to debug against, and that is where reference tests come in. They also happen to be one of the safest jobs you can give an agent. The rule of thumb is simple: let the agent write the reference, never the kernel. The agent does not need to understand a single Blackwell barrier to produce a correct PyTorch or NumPy reference. What it does need is the layout convention, stated explicitly; leave it out and the reference will silently disagree with the kernel.

Here is a prompt that works:

```text
Generate a PyTorch reference for this TIRx kernel.
Operation: D = A @ B.T.
A shape: [M, K], fp16.
B shape: [N, K], fp16.
D shape: [M, N], fp16.
Use fp32 accumulation and cast to fp16.
Return max absolute error and relative error.
```

And here is the shape of reference you should expect back:

```python
D_ref = (A.float() @ B.float().T).half()
abs_err = (D - D_ref).abs()
max_err = float(abs_err.max())
rel_err = float((abs_err / D_ref.abs().clamp_min(1e-3)).max())
```

Flash Attention raises the stakes on layout, so spell the exact tensor convention out for the agent:

```text
Q, K, V are [batch, seq, heads, dim].
The kernel supports GQA: num_qo_heads may be larger than num_kv_heads.
Convert Q/K/V to [batch, heads, seq, dim] before calling scaled_dot_product_attention.
For GQA, repeat K/V along the head dimension after the transpose.
Transpose the output back to [batch, seq, num_qo_heads, dim].
If seq_len_q and seq_len_kv differ, verify causal-mask alignment explicitly.
```

Even so, the generated test still deserves a careful read: watch the dtype, the accumulation precision, the head layout, and the tolerance in particular. But reviewing a reference is far less risky than trusting the agent to write the kernel itself.

## Use Case 5: Inspect Generated CUDA

A reference can tell you the math came out wrong. What it cannot tell you is whether the right threads actually ran the operation. For that, the generated CUDA is the ground truth. The TIRx source expresses your intent; the generated CUDA shows the guards and instructions that will really run on the chip. Agents can help you read it, as long as you ask a concrete question:

```text
Here is the generated CUDA guard around tcgen05_alloc.
Does it require all lanes of one warp, or only lane 0?
```

A handful of patterns are worth checking, because each TIRx intent leaves a recognizable fingerprint in the generated code:

| TIRx intent | Generated-code clue |
|-------------|---------------------|
| warpgroup branch | `(warp_id_in_cta >> 2) == ...` |
| warp branch | `(warp_id_in_cta & 3) == ...` |
| lane 0 branch | `threadIdx.x % 32 == 0` |
| barrier init default leader | `threadIdx.x < 1` |
| elected issue | `tvm_builtin_elect_one_sync_op()` |

And one rule overrides everything: if the agent's explanation disagrees with the generated code, trust the generated code.

For a broader generated-CUDA workflow, turn to the *TIRx Language and Compile Pipeline* appendix page.

## Agent Review Boundaries

All five use cases rest on the same precondition: the contract is already on the table for the agent to check. The risky cases are the mirror image, the ones where the agent has to invent the hardware contract instead of verifying it. The difference shows up clearly in how you phrase the question.

- Do not ask "fix the barriers." Ask "who arrives at this barrier, and how many arrivals does init expect?"
- Do not ask "add the right fence." Ask "which producer-consumer edge does this fence order?"
- Do not ask "is this phase right?" Ask "what happens on the first wait, and when does the phase flip?"
- Do not accept invented TIRx APIs. Verify unfamiliar names with the API reference or `rg`.
- Do not accept an `elect_sync()` explanation unless it says the election is one thread per warp.
- Do not accept performance advice without measurement. Pipeline depth, SMEM use, occupancy, and Tensor Core utilization must be profiled.

So the real boundary is not "agent versus human" in some abstract sense. It is whether the contract is already explicit. When the contract is explicit, the agent can check it for you. When the contract is missing, do not let the agent quietly invent one. Ask it for candidate contracts, weigh their tradeoffs, and then choose and state the contract yourself.

## Project Context File

Drawing that boundary once is not enough, because every new conversation begins with an agent that remembers nothing of the last one. A project context file closes that gap. It is a short bug log you paste into future prompts, so the agent never has to rediscover, or worse, contradict, a lesson you already paid for once.

For example:

```markdown
# TIRx Blackwell Bug Notes

### tcgen05.commit Outside Elected Scope
Symptom: zeros or random garbage.
Cause: only the elected thread has a non-empty commit group; other lanes can signal early.
Fix: keep TCGen05Bar.arrive inside the same elect_sync scope as gemm_async.

### MBarrier.init from Wrong Warpgroup
Symptom: deadlock.
Cause: default leader is CTA thread 0. If init is inside wg_id == 1, no thread runs it.
Fix: initialize barriers at CTA level before role branches.

### TMA Store Reused Dsmem Too Early
Symptom: intermittent wrong rows.
Cause: the TMA store group was not committed or waited far enough before Dsmem reuse.
Fix: commit the store group and wait far enough for the staging SMEM buffer's reuse pattern.
```

Keep each entry short: symptom, cause, fix, and nothing more. A few notes in this form are worth far more to an agent than a long, rambling chat history.

## Exercises

1. Take the warp-specialized GEMM from {ref}`chap_gemm_advanced` and ask an agent to produce a tile-primitive table: primitive, scope, layout, dispatch, wait-before, signal-after. Which entries did it miss?
2. Move `mma2tma.arrive()` outside the elected MMA issue scope in a local experiment. Ask the agent to diagnose the failure using only the symptom, then ask again with the `tcgen05.commit` hardware constraint. Compare the answers.
3. Ask for a PyTorch reference for the Flash Attention kernel with GQA. Check whether the agent handles `repeat_interleave` for K/V heads correctly.
4. Give the agent the generated CUDA for a warp-specialized kernel and ask it to identify the TMA, MMA, and writeback branches. Verify against the source.
