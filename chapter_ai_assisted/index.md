# Writing TIRX Kernels with Agents
:label:`chap_ai_assisted`

Before asking an agent to work on TIRX, give it the source code it must reason from. It should be able to read the `tvm` codebase for the TIRX DSL, layout objects, tile primitives, and lowering rules, and the `tirx-kernels` codebase for real kernels, scheduler helpers, and barrier patterns. Without those references, the agent will fall back to generic CUDA, Triton, or Hopper assumptions, which are often wrong for Blackwell TIRX.

There are two ways to use an agent. The first is delegation: give it a broad goal, such as "make the FA4 barrier section easier to understand," and let it choose the method. That is useful for mechanical edits after you already know the direction, but it teaches you less because the important choices stay hidden. The second is learning-oriented: turn the broad goal into a specific instruction, such as "explain `bar_softmax_corr_full` and `bar_softmax_corr_empty` as a mailbox-slot lifecycle, keep the value-MMA gate in a separate diagram, then rebuild the tutorial." This is usually more effective for TIRX because the important work is not just changing text or code; it is learning what choices are possible and which hardware contracts those choices imply.

When you do not yet know the right instruction, ask the agent for candidates first. A good prompt is:

```text
I want to make the FA4 barrier section clearer, but I do not know the best form yet.
Give me 3 candidate rewrites. For each one, say what it explains well,
what it hides, and what code or diagram evidence I should verify.
Do not edit yet.
```

After reading the candidates, choose one and turn it into an explicit instruction. The next time you face the same kind of problem, you should be able to write that instruction yourself. This is the real value of using agents while learning TIRX: they help you discover "what is possible" and convert vague goals into reusable engineering moves.

The previous chapters built up a way to read Blackwell kernels: identify the tile path, then check scope, layout, dispatch, and synchronization. That same structure is also the right way to move from a vague goal to a useful agent instruction. The unit of interaction is not a Python function or a whole CUDA file. It is a TIRX kernel contract. TIRX kernels are full of local contracts:

- which scope owns a tile operation,
- where each tile lives and how it is laid out,
- which dispatch path is requested,
- which barrier proves an async producer is complete,
- and whether the generated CUDA matches the intended hardware path.

Those contracts, plus the source references above, are the context an agent needs.

The agent can help you draft, compare, and execute choices. The programmer still owns the final contract.

## Workflow

![Writing TIRX Kernels with Agents Workflow](../img/ai_assisted_tirx_workflow.png)

A practical workflow is:

1. Point the agent at `tvm` and `tirx-kernels`.
2. Start with the goal, but do not stop there.
3. If the path is unclear, ask for candidate strategies and tradeoffs.
4. Choose one strategy and rewrite it as a concrete TIRX instruction: tile path, roles, layouts, barriers, expected checks.
5. Ask the agent to execute, explain, or review one local contract at a time.
6. Confirm with generated CUDA, tests, and benchmarks.
7. Record the instruction pattern you learned so future prompts start more precise.

The agent is strongest when it helps you sharpen the instruction. Let it propose possibilities; do not let it silently decide the hardware contract.

## From Goal to Instruction

For tutorial writing, the difference looks like this:

```text
Broad goal:
Make the FA4 barrier section easier to understand.

Candidate prompt:
Give me three ways to explain bar_softmax_corr_empty:
1. a full barrier DAG,
2. a mailbox slot lifecycle,
3. a wait/arrive table.
For each option, say which misunderstanding it prevents and which detail it hides.
Do not edit yet.

Final instruction after choosing:
Use the mailbox slot lifecycle. Explain that bar_softmax_corr_full means
WG2 may read acc_scale or row_sum, while bar_softmax_corr_empty means
softmax may reuse that SMEM slot. Keep bar_p_full_o_rescaled separate
because it gates value MMA, not the scale slot.
```

For kernel work, the same pattern applies:

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

This pattern matters because it turns the agent into a learning tool. You are not only getting a patch. You are learning the vocabulary of possible patches.

## The TIRX Contract Prompt

When asking an agent about a TIRX kernel, do not start with only a code dump. Start with the kernel contract.

Each field has a job. The tile path gives the data flow. Scopes and roles say who executes each tile operation. Layouts and dispatch say which lowering is legal. Barriers say which producer-consumer edges make the async work safe to consume. The example below is the kind of prompt you can derive from the warp-specialized GEMM chapter.

```text
Target: NVIDIA Blackwell SM100a.
Kernel: warp-specialized GEMM from Chapter 5.

Tile path:
GMEM -> SMEM by TMA.
SMEM -> TMEM by tcgen05 MMA.
TMEM -> RF -> SMEM -> GMEM by epilogue.

Scopes and roles:
WG1 warp 3: TMA producer.
WG1 warp 0: MMA consumer.
WG0: writeback.

Layouts:
SMEM A/B use tma_shared_layout(...).
TMEM accumulator uses TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]).
RF readback uses TileLayout(S[(128, N) : (1@tid_in_wg, 1)]).

Barriers:
tma2mma: TMA -> MMA.
mma2tma: MMA -> TMA.
mma2ld: MMA -> writeback.
ld2mma: writeback -> MMA.

Symptom:
kernel hangs after the first output tile.

Question:
Which handoff should I inspect first?
```

This prompt gives the agent the same map a human reviewer would build before reading the code. Without it, the agent may guess from generic CUDA, Triton, or Hopper patterns and miss the Blackwell-specific issue.

## Case Study: Elected MMA Commit Bug

This case shows why the prompt needs a hardware fact, not just code. The broken loop comes from a warp-specialized GEMM. The MMA is issued by one elected thread, but the barrier arrive is outside the elected-thread scope:

```python
for k in range(K_TILES):
    tma2mma.wait(mma_ps.stage, mma_ps.phase)
    with Tx.thread(Tx.ptx.elect_sync()):
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

In these TIRX wrappers, `TCGen05Bar.arrive()` lowers to `tcgen05.commit`, and the caller must guard it so only the intended issuer calls it. Only the elected thread has the real MMA work in its commit group. The other lanes create empty commit groups, and those empty groups can signal the mbarrier before the MMA finishes. The TMA producer may then overwrite SMEM while the MMA still needs it.

A useful agent prompt is not:

```text
Why is my GEMM wrong?
```

It is:

```text
The MMA issue uses elect_sync, but mma2tma.arrive is outside that elected scope.
In these TIRX wrappers, TCGen05Bar.arrive lowers to tcgen05.commit.
Can empty commit groups signal the barrier early?
```

The fix is to keep the arrive in the same elected-thread scope as the MMA issue:

```python
for k in range(K_TILES):
    tma2mma.wait(mma_ps.stage, mma_ps.phase)
    with Tx.thread(Tx.ptx.elect_sync()):
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

This is the pattern for the rest of the chapter: state the contract, ask the agent to check one edge, then verify the answer against source, generated CUDA, and a runnable test.

## Use Case 1: Explain a Kernel as Tile Primitives

The first useful task is explanation. Ask the agent to convert a code region into a tile-primitive table:

```text
Read this TIRX code and make a table with:
primitive, scope, source tile, destination tile, dispatch path,
barrier waited before the primitive, and barrier signaled after it.
```

For example, a good explanation of a GEMM writeback should look like:

| Primitive | Scope | Source | Destination | Handoff |
|-----------|-------|--------|-------------|---------|
| `Tx.copy_async(Dreg_wg, tmem)` | warpgroup | TMEM accumulator | warpgroup RF tile | wait for MMA first, then `wait.ld()` |
| `Tx.cast(reg_f16, reg_f32)` + `Tx.copy(Dsmem, reg_f16)` | thread | per-thread registers | SMEM staging tile | fence before TMA store |
| `Tx.copy_async(D, Dsmem, dispatch="tma")` | selected TMA issuer | SMEM staging tile | GMEM output | commit the TMA store group and wait before SMEM reuse |

This is a better question than "explain this code" because it forces the answer to use the tutorial's mental model. If the agent cannot identify the scope, layout, dispatch, or handoff, that is a sign the explanation is incomplete.

For a line like `Tx.copy_async(Dreg_wg, tmem)`, a weak explanation is "this copies data." A useful explanation says: this runs under warpgroup scope, reads a TMEM accumulator tile, writes a warpgroup-distributed register view, lowers to a `tcgen05.ld`-style readback, and needs `wait.ld()` before the registers are consumed.

## Use Case 2: Review a Kernel Change

Agent review works best when the change has a small, checkable contract. Examples:

- change `PIPE_DEPTH` from 2 to 4,
- change `cta_group=1` to `cta_group=2`,
- add a second MMA consumer,
- replace synchronous copy with TMA,
- move writeback through SMEM and TMA store.

Ask for invariant checks, not a broad review:

```text
I changed a warp-specialized single-CTA GEMM into the clustered Chapter 5 form with CTA_GROUP=2.
Check only the cluster invariants:
- tcgen05.alloc / gemm_async / commit / dealloc cta_group values
- scheduler tile shape
- TMA byte count
- remote barrier view
- cluster_sync before cleanup
```

For warp-specialized GEMM, useful review questions are:

| Change | What to ask the agent to check |
|--------|--------------------------------|
| Increase pipeline depth | barrier array depth, stage index, phase flip point, SMEM budget |
| Add TMA store | RF -> SMEM fence, warpgroup sync, commit group, store wait before SMEM reuse |
| Add CTA cluster | `cta_group`, remote barriers, cluster tile shape, cleanup sync |
| Add consumer | per-consumer barrier slots, TMEM ranges, scheduler M tile factor |

The agent is not deciding the design. It is checking whether the code still matches the design. For example, `cta_group=2` is not just one keyword on `Tx.gemm_async`; it also changes TMEM allocation, MMA issue, commit, deallocation, remote barriers, and scheduler tile shape.

## Use Case 3: Debug from Symptoms

When a kernel fails, first classify the symptom. Then ask the agent to map the symptom back to the nearest producer-consumer handoff. The table here is the prompt-level version; use the TIRX Language and Compile Pipeline appendix page when you need generated-CUDA inspection.

| Symptom | Likely area | First checks |
|---------|-------------|--------------|
| Kernel hangs | missing arrive, wrong phase, wrong init count | barrier producer/consumer pair |
| Output all zeros | TMEM read before MMA completes, early empty commit | MMA commit guard, `mma2ld.wait` |
| Wrong rows in chunks of 128 | warpgroup handoff bug | TMEM readback, writeback barrier, per-consumer slot |
| Random garbage | SMEM reused too early | TMA/MMA empty barrier, TMA store drain |
| NaN everywhere | invalid input to math or wrong descriptor | softmax max subtraction, masks, SMEM layout, MMA descriptor |
| Works for small sizes only | scheduler or boundary bug | tile count, phase reset, tail guard |

The prompt should include both the symptom and the code around the relevant barrier. For example:

```text
Symptom: output is zeros or random garbage.
The MMA issue uses elect_sync, but mma2tma.arrive is outside that elected scope.
Does this allow an empty commit group to signal the barrier early?
```

This is the kind of question agents handle well: the local contract is explicit, and the agent only needs to check whether the code violates it.

## Use Case 4: Generate Reference Tests

Test generation is one of the safest agent use cases. Let the agent write the reference, not the kernel. It does not need to understand Blackwell barriers to write a PyTorch or NumPy reference, but the prompt must still state the layout convention.

Good prompt:

```text
Generate a PyTorch reference for this TIRX kernel.
Operation: D = A @ B.T.
A shape: [M, K], fp16.
B shape: [N, K], fp16.
D shape: [M, N], fp16.
Use fp32 accumulation and cast to fp16.
Return max absolute error and relative error.
```

Expected reference shape:

```python
D_ref = (A.float() @ B.float().T).half()
abs_err = (D - D_ref).abs()
max_err = float(abs_err.max())
rel_err = float((abs_err / D_ref.abs().clamp_min(1e-3)).max())
```

For Flash Attention, ask for the exact tensor layout:

```text
Q, K, V are [batch, seq, heads, dim].
The kernel supports GQA: num_qo_heads may be larger than num_kv_heads.
Convert Q/K/V to [batch, heads, seq, dim] before calling scaled_dot_product_attention.
For GQA, repeat K/V along the head dimension after the transpose.
Transpose the output back to [batch, seq, num_qo_heads, dim].
If seq_len_q and seq_len_kv differ, verify causal-mask alignment explicitly.
```

The generated test still needs review, especially dtype, accumulation precision, head layout, and tolerance. But this is far less risky than asking the agent to write the kernel itself.

## Use Case 5: Inspect Generated CUDA

For scope guards and issued intrinsics, the generated CUDA is the ground truth. TIRX source expresses intent; generated CUDA shows the guards and instructions that will actually run. Agents can help read it if you ask a concrete question:

```text
Here is the generated CUDA guard around tcgen05_alloc.
Does it require all lanes of one warp, or only lane 0?
```

Useful patterns to check:

| TIRX intent | Generated-code clue |
|-------------|---------------------|
| warpgroup branch | `(warp_id_in_cta >> 2) == ...` |
| warp branch | `(warp_id_in_cta & 3) == ...` |
| lane 0 branch | `threadIdx.x % 32 == 0` |
| barrier init default leader | `threadIdx.x < 1` |
| elected issue | `tvm_builtin_elect_one_sync_op()` |

If the agent's explanation disagrees with the generated code, trust the generated code.

For a broader generated-CUDA workflow, use the TIRX Language and Compile Pipeline appendix page.

## Agent Review Boundaries

The risky cases are the ones where the agent has to invent the hardware contract.

- Do not ask "fix the barriers." Ask "who arrives at this barrier, and how many arrivals does init expect?"
- Do not ask "add the right fence." Ask "which producer-consumer edge does this fence order?"
- Do not ask "is this phase right?" Ask "what happens on the first wait, and when does the phase flip?"
- Do not accept invented TIRX APIs. Verify unfamiliar names with the API reference or `rg`.
- Do not accept an `elect_sync()` explanation unless it says the election is one thread per warp.
- Do not accept performance advice without measurement. Pipeline depth, SMEM use, occupancy, and Tensor Core utilization must be profiled.

The boundary is not "agent versus human" in the abstract. It is whether the contract is already explicit. If the contract is explicit, the agent can check it. If the contract is missing, do not let the agent silently choose one. Ask for candidate contracts, compare their tradeoffs, then choose and state the contract yourself.

## Project Context File

A project context file is a short bug log that you paste into future prompts. It prevents the agent from rediscovering or contradicting lessons you already learned.

Example:

```markdown
# TIRX Blackwell Bug Notes

### tcgen05.commit outside elected scope
Symptom: zeros or random garbage.
Cause: only the elected thread has a non-empty commit group; other lanes can signal early.
Fix: keep TCGen05Bar.arrive inside the same elect_sync scope as gemm_async.

### MBarrier.init from wrong warpgroup
Symptom: deadlock.
Cause: default leader is CTA thread 0. If init is inside wg_id == 1, no thread runs it.
Fix: initialize barriers at CTA level before role branches.

### TMA store reused Dsmem too early
Symptom: intermittent wrong rows.
Cause: the TMA store group was not committed or waited far enough before Dsmem reuse.
Fix: commit the store group and wait far enough for the staging SMEM buffer's reuse pattern.
```

Keep entries short: symptom, cause, fix. These notes are more useful to an agent than a long chat history.

## Exercises

1. Take the warp-specialized GEMM from Chapter 5 and ask an agent to produce a tile-primitive table: primitive, scope, layout, dispatch, wait-before, signal-after. Which entries did it miss?
2. Move `mma2tma.arrive()` outside the elected MMA issue scope in a local experiment. Ask the agent to diagnose the failure using only the symptom, then ask again with the `tcgen05.commit` hardware constraint. Compare the answers.
3. Ask for a PyTorch reference for the Flash Attention kernel with GQA. Check whether the agent handles `repeat_interleave` for K/V heads correctly.
4. Give the agent the generated CUDA for a warp-specialized kernel and ask it to identify the TMA, MMA, and writeback branches. Verify against the source.
