(chap_flash_attention)=
# Flash Attention 4

:::{admonition} Overview
:class: overview

- Attention runs two MMAs with softmax wedged between them, so it cannot just repeat one MMA the way GEMM does.
- The kernel composes everything from Part III (TMA, `tcgen05`, TMEM, barriers) with warp roles, online-softmax rescaling, causal masking, and GQA.
:::

**Motivation.** Attention is the kernel that decides whether a transformer runs at all, and it is also where everything we built so far finally has to work together. Every piece we assembled for GEMM — TMA tile movement, `tcgen05` MMA, TMEM, warpgroup-local register tiles, explicit barriers — carries over here unchanged, so it is fair to ask why attention deserves a chapter of its own rather than a footnote to GEMM. The reason is that attention is not one MMA repeated: it is *two* MMAs with real work wedged between them — online softmax, causal masking, and the rescaling that keeps earlier and later blocks in a common scale. That middle stage is where all the new difficulty lives, because nothing about a plain matmul prepares you for state that has to be corrected on the fly while the GPU streams more data in. This is the chapter where everything composes into one real, end-to-end kernel: the sections below keep just enough of the Flash Attention 4 algorithm to make the kernel readable, then trace a single tile from GMEM through both MMAs and the softmax between them.

This chapter keeps just enough of the Flash Attention 4 algorithm to make the kernel readable, then focuses on how that algorithm becomes TIRx.

We follow one tile as it flows through the kernel. `Q`, `K`, and `V` enter as input tiles loaded from GMEM into SMEM. The score MMA multiplies `Q` and `K` into a score tile `S` in TMEM. Softmax turns `S` into a numerator tile `P`, and the value MMA combines `P` and `V` to update the output accumulator `O`. There is one twist that GEMM never had: when the running softmax maximum changes, the `O` accumulated so far is in the wrong scale and must be rescaled before the next value MMA can add into it. The sections below trace this path first, then show how TIRx hands each stage to a warpgroup and wires the stages together.

## Algorithm Shape

For one query block, Flash Attention computes:

$$O = \text{softmax}(QK^{\top} / \sqrt{d})V$$

The naive way is to form the full `S = QKᵀ`, softmax it, then multiply by `V`. That is a non-starter on-chip: at seq=4096 the full `S` is ~16M elements per head (~64 MB in fp32), orders of magnitude past SMEM or the single 128×512 TMEM region. Flash Attention never materializes `S` at all — it streams `K/V` in blocks and carries three per-row running states that summarize everything seen so far:

- `row_max`: the maximum score seen so far.
- `row_sum`: the running denominator of softmax.
- `O`: the running output accumulator.

The streaming update keeps those states correct as new blocks arrive. Each time a block is processed, the running max may rise, and everything computed under the old max has to be pulled back into the new scale before the new contribution is added:

```text
S = Q_block @ K_block.T
m_new = max(row_max, rowmax(S))
scale = exp((row_max - m_new) / sqrt(d))
P = exp((S - m_new) / sqrt(d))
row_sum = row_sum * scale + rowsum(P)
O = O * scale + P @ V_block
row_max = m_new
```

The `scale` factor rescales both the running denominator and the running output so the contributions from earlier and later blocks end up in a common scale.

The pseudocode above uses natural `exp` and an explicit `/sqrt(d)` for readability, but the kernel takes a cheaper route. It folds both `1/sqrt(d)` and `log2(e)` into one constant `scale_log2 = log2(e)/sqrt(d)` and evaluates every exponential with the hardware `exp2` on raw scores, using the identity `exp(x/sqrt(d)) = exp2(x · scale_log2)`, because `exp2` is faster than a natural `exp`.

Note that `P` here is *not* the final normalized attention matrix. It is only the softmax numerator for the current K/V block; the normalization is deferred, and after the last block the kernel writes `O / row_sum`.

For TIRx, knowing what the algorithm computes is only half the picture; the other half is *where each tile lives* as the kernel runs, because that dictates the layout and barrier code. `S`, `P`, and `O` are all tile values, and each has a home:

- `S` is the score tile. The score MMA writes it to TMEM.
- `P` is the softmax numerator tile. Softmax reads `S` from TMEM into registers, computes `P = exp((S - m_new) / sqrt(d))`, and writes `P` back to TMEM.
- `O` is the output accumulator tile. The value MMA reads `P` from TMEM and `V` from SMEM, then accumulates into `O` in TMEM.

The rescale flagged earlier is also a tile operation: when `row_max` changes, the old `O` is read from TMEM, multiplied in registers, and written back to TMEM before the next value MMA accumulates into it. Every step is a tile moving between SMEM, TMEM, and registers, and reading the rest of the chapter through that view keeps it legible.

## Tile-Primitive Graph

For one K/V block, the kernel walks this tile path top to bottom:

```text
Q, K, V in GMEM
  -> Q, K, V in SMEM        by TMA load
  -> S in TMEM              by score MMA: QK^T
  -> P in TMEM              by softmax numerator: TMEM -> RF -> TMEM
  -> O in TMEM              by value MMA: P V
  -> O in GMEM              by normalization, SMEM staging, and TMA store
```

The difference from GEMM is one line: GEMM is a single repeated MMA chain, while FA4 has two MMA phases with softmax in the middle of the chain. Everything else that follows is a consequence of that one extra stage.

Expanding the short path into producer-consumer edges gives the full graph:

| Stage | Tile movement or compute | TIRx primitive | Hardware path |
|-------|--------------------------|----------------|---------------|
| Load Q/K/V | GMEM tiles -> SMEM tiles | `Tx.copy_async(..., dispatch="tma")` | TMA load |
| Score MMA | Q in SMEM and K in SMEM -> score tile `S` in TMEM | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` |
| Softmax read | `S` in TMEM -> warpgroup register tile | `Tx.wg.copy_async(reg, tmem)` | `tcgen05.ld` |
| Softmax write | numerator tile `P` in registers -> fp16 TMEM view | `Tx.copy_async(tmem_as_f16, reg)` | TMEM store, followed by `tcgen05.wait.st()` |
| Value MMA | `P` in TMEM and V in SMEM -> output accumulator `O` in TMEM | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` with a TMEM operand |
| Correction | `O` in TMEM -> registers -> `O` in TMEM | TMEM readback, register multiply, TMEM store | `tcgen05.ld` / TMEM store |
| Epilogue | final `O` in TMEM -> registers -> SMEM -> GMEM | TMEM readback, `Tx.copy`, TMA store | `tcgen05.ld` + TMA store |

The unfamiliar middle steps are still tile operations. Softmax reads a score tile from TMEM into warpgroup registers, does row-wise math, and writes a `P` tile back to TMEM; correction reads an `O` tile from TMEM, rescales it, and writes it back. Same SMEM/TMEM/register vocabulary as GEMM, just more of it.

**Try with your agent**: Ask it to trace only the short path above. For each arrow, name the producer role, consumer role, source tile, destination tile, and hardware path. Then ask which arrows did not exist in the GEMM chapters.

## Warp Roles and Scopes

With the data path fixed, the next question is who runs each stage. Each CTA here has 4 warpgroups, 512 threads in all, split by *what kind of work* a warpgroup does:

- WG3 drives the hardware engines: TMA load, MMA, and TMA store.
- WG0, WG1, and WG2 do the register-heavy math that happens between those engine calls: softmax, correction, and epilogue.

The exact role table is:

| Owner | Role | What it does |
|-------|------|--------------|
| WG3, warp 1 | TMA load | Loads Q, K, and V tiles from GMEM to SMEM |
| WG3, warp 0 | MMA | Issues both score MMA and value MMA |
| WG3, warp 2 | TMA store | Stores final O tiles from SMEM to GMEM |
| WG0 | Softmax for Q stage 0 | Reads S from TMEM, computes P, writes P to TMEM |
| WG1 | Softmax for Q stage 1 | Same work for the second Q pipeline stage |
| WG2 | Correction and epilogue | Rescales O in TMEM, normalizes, stages output |

The "two Q stages" are not two attention heads. They are two slots in the Q pipeline, WG0 owning one and WG1 the other, so two Q tiles can be in flight at once. This is why the softmax work is duplicated across WG0 and WG1.

The code picks these roles out with symbolic coordinates:

```python
wg_id = T.warpgroup_id([4])
warp_id = T.warp_id_in_wg([4])
```

Find the role branch first; it tells you which team owns every tile primitive nested inside it.

- WG3 warp 1 starts TMA load commands. One elected lane issues the copy, and the TMA engine moves the tile.
- WG3 warp 0 issues the `tcgen05.mma` instructions.
- WG0 and WG1 run softmax under full warpgroup scope.
- WG2 runs correction and epilogue work under full warpgroup scope.

One asymmetry shapes the barrier graph: *every* MMA, both score and value, issues from WG3 warp 0 alone. WG0 and WG1 never issue an MMA — they only consume the score tile, run softmax, and write `P` back to TMEM.

That separation is why softmax needs barriers around it. `s_ready` carries the score tile from the MMA warp to softmax; `p_o_rescale` carries `P` and a rescaled `O` from softmax and correction back to the value MMA. The rest of the chapter keeps returning to those two names.

## Reading the Fragments

The fragments in this chapter are excerpts from `flash_attention4.py`, so they reference names defined in parts of the kernel we do not reproduce. The self-describing ones — `wg_id`, `warp_id`, `BLK_M`/`BLK_N`, `HEAD_DIM`, `kv_stage`, the `SMEM_PIPE_DEPTH_*` / `TMEM_PIPE_DEPTH` depths, `should_accumulate`, and `CTA_GROUP` (1 here) — are introduced where they first matter below. The rest get a one-line gloss here, for lookup the moment a fragment puts an unfamiliar name in front of you:

| Name | Meaning |
|------|---------|
| `q_stage`, `i_q` | Q pipeline stage, 0 or 1 — which Q tile slot (`SMEM_PIPE_DEPTH_Q = 2`). Inside WG0/WG1 softmax the warpgroup's own `wg_id` (0 or 1) *is* this same stage index, so `S_region[q_stage]`, `P_region[wg_id]`, and `O_region[i_q]` all select the same Q stage |
| `MMA_N` | score/output tile width in TMEM columns (128) |
| `MMA_K` | MMA inner-K step in `P`/`V` columns (16); `K_SPLIT = 6 * MMA_K = 96` |
| `K_SPLIT` | split point of the value-MMA schedule (see *The Two MMA Phases*); the first value MMA covers columns `0:K_SPLIT` (`6 * MMA_K = 96`) |
| `should_rescale` | WG2 per-row flag: whether the old `O` needs rescaling before the next value MMA (reduced across the warpgroup with `any_sync`) |
| `rescale_threshold` | when the scaled row-max change is small enough, `acc_scale` is clamped to exactly 1.0 and the rescale is skipped (8.0) |
| `scale_log2` | the softmax scale in log2 units, `log2(e)/√d`, so `P = exp2((S - m) · scale_log2)` |
| `acc_scale` | per-row rescale factor softmax passes to WG2 through the SMEM mailbox |
| `chunk_start`/`chunk_end`, `p_start`/`p_end` | column range of the 32-wide softmax chunk being read / written |

## The Two MMA Phases

For each streamed K/V tile, Flash Attention runs two MMA phases with softmax bridging them:

```text
Q, K -> score MMA -> S
S    -> softmax   -> P
P, V -> value MMA -> O
```

This is a pipeline of three producers: the first MMA produces attention scores `S`, softmax turns `S` into the numerator `P`, and the second MMA consumes `P` to update the output accumulator `O`. Normalization by `row_sum` is held back to the epilogue, after every K/V tile has contributed.

Each tile op below gets the same **scope / layout / dispatch** card as the GEMM steps, with one extra line — **Handoff** — naming the barrier(s) that pass the tile to the next role.

The compute code does not use raw TMEM column numbers. The kernel carves its single TMEM allocation into per-stage views — `S_region`, `P_region`, `O_region` — and indexes them by pipeline stage (`S_region[q_stage]`, `O_region[i_q]`, `P_region[i_q, 0:K_SPLIT]`). Those views are defined with `T.TMEMStages` in the [TMEM Layout and Reuse](#tmem-layout-and-reuse) section; for now treat each region as a named slice of the same physical TMEM.

### Score MMA

The score MMA computes:

$$S = Q_{\text{block}}K_{\text{block}}^{\top}$$

and writes the `128 x 128` score tile to TMEM:

```python
Tx.warp.gemm_async(
    S_region[q_stage],
    Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
    K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
if T.ptx.elect_sync():
    s_ready.arrive(q_stage)
```

We ask the same four questions the GEMM chapters asked of every tile op — who runs it, where the tiles live, how it dispatches, and how it hands off:

> **Tile-primitive readout — Score MMA**
> - Scope: WG3 warp 0 issues it; one elected lane arrives `s_ready`.
> - Layout: Q, K in SMEM → `S` in TMEM (`S_region[q_stage]`).
> - Dispatch: `tcgen05`.
> - Handoff: `s_ready` (→ softmax).

The single elected thread arriving on `s_ready` is the whole handoff: it announces that this score tile is finished and the softmax warpgroup may read it.

### Softmax Between MMAs

> **Tile-primitive readout — Softmax**
> - Scope: WG0 (Q stage 0) / WG1 (Q stage 1), full warpgroup.
> - Layout: `S` in TMEM → registers → `P` in fp16 TMEM (`P_region[wg_id]`).
> - Dispatch: `tcgen05.ld` to read, TMEM store to write; row-wise math in registers between them.
> - Handoff: waits `s_ready`; arrives `p_o_rescale` (first 96 columns) and `p_ready_2` (last 32).

This stage has no GEMM counterpart. WG0/WG1 wait for the score tile on `s_ready`, then read it out of TMEM in register-sized chunks:

```python
Tx.copy_async(
    s_chunk[:, chunk_start : chunk_end],
    S_region[wg_id, chunk_start : chunk_end],
)
```

That is a TMEM-to-register tile read under warpgroup scope. With the scores in registers, the softmax warpgroup does three things in order:

1. computes the row max and row sum,
2. computes the softmax numerator tile `P`,
3. writes `P` back to TMEM as fp16.

The last step looks like:

```python
Tx.copy_async(
    P_region[wg_id, p_start : p_end],
    p_chunk[:, p_start : p_end],
)
```

`P` is written back to TMEM rather than kept in the registers it was just computed in because the value MMA needs `P` as a *tile operand*, and an MMA cannot read scattered per-thread scalar registers as a matrix. The MMA-readable form of `P` in this kernel is `P_region`, a view over the fp16 TMEM alias `tmem_as_f16`. The writeback puts `P` into the only shape the next MMA can consume.

### Value MMA

The value MMA computes:

$$O = O + P_{\text{block}}V_{\text{block}}$$

By the time this runs, `O` has already been initialized (first block) or rescaled (later blocks) for the current K/V block, so the MMA accumulates. What differs from GEMM is where the operands live: the A operand is `P` in TMEM, the B operand is `V` in SMEM, and the accumulator `O` is also in TMEM:

```python
# First sub-MMA: columns 0:K_SPLIT (the first 96 of P / rows of V).
Tx.warp.gemm_async(
    O_region[i_q],
    P_region[i_q, 0:K_SPLIT],
    V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
    transB=True,
    accum=should_accumulate,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
# The second sub-MMA (same form, accum=True, gated on p_ready_2) covers the
# remaining columns K_SPLIT:BLK_N.
```

> **Tile-primitive readout — Value MMA**
> - Scope: WG3 warp 0.
> - Layout: `P` in TMEM + V in SMEM → `O` in TMEM (`O_region[i_q]`).
> - Dispatch: `tcgen05` with a TMEM operand.
> - Handoff: waits `p_o_rescale`, `p_ready_2`, `kv_load.full`; arrives `o_ready` (→ epilogue).

That operand placement is the hardware difference between the two MMAs:

- Score MMA reads both operands from SMEM: Q and K.
- Value MMA reads one operand, `P`, from TMEM.
- Value MMA reads the other operand, V, from SMEM.
- The result accumulates into `O` in TMEM.

The `accum=should_accumulate` flag is what implements the "initialize or add" choice from the algorithm: false on the first K/V tile of a query block, true thereafter.

The value MMA is split into a `96 + 32` schedule rather than run as one shot:

1. Softmax writes `P` in four 32-column chunks.
2. After the first three chunks are ready, the value MMA starts on the first 96 columns of `P` and the matching rows of `V`.
3. The final 32 columns wait for `p_ready_2`.
4. A second MMA consumes that final chunk and finishes the tile.

The split keeps the Tensor Core busy. Run it as one MMA and the value phase would stall until all four 32-column `P` chunks were exponentiated and stored. By firing on the first three chunks immediately, the kernel overlaps the last chunk's `exp` and TMEM write with a 96-wide MMA that is already running — work that would otherwise be idle time.

## TMEM Layout and Reuse

All of `S`, `P`, and `O` have to fit in one `128 x 512` TMEM allocation, and how they are packed is why barriers and layout are inseparable in this kernel:

![TMEM Layout](../img/tmem_layout_v3.png)

The figure reads as a set of tile slots:

- Score slots hold `S = QK^T`.
- Numerator slots hold the `P` tile after the softmax exponentiation step.
- Output slots hold the fp32 `O` accumulator.

These are not independent buffers — they are regions of the *same* allocation, and the sharing is forced. With Q-pipeline depth 2, the two `S` slots (2 × MMA_N = 256 columns) plus the two `O` slots (2 × MMA_N = 256 columns) already consume all 512 fp32 columns. There are no columns left for `P`, so `P` has to alias the same bytes through a narrower fp16 view. This works only because each region is reused strictly after its previous consumer finishes, which is what the barriers guarantee. In FA4, barriers are not just scheduling; they are what makes the layout legal.

The aliasing trick is set up through a `T.TMEMPool`. The kernel takes one fp32 view (`tmem`) for the score and output accumulators, then rewinds the pool base to 0 and takes a second, fp16 view (`tmem_as_f16`) over the *same* physical bytes:

```python
tmem_pool = T.TMEMPool(pool, total_cols=N_COLS_TMEM, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_pool.move_base_to(0)
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()
```

Because fp16 elements are half the width, the fp16 view exposes twice as many indexable columns over those same bytes, which lets `P` live in space the fp32 layout had no room for. With both views in hand, the `S`, `P`, and `O` slots are carved out as staged regions with `T.TMEMStages`, so compute code indexes by pipeline stage rather than raw columns:

```python
S_region = T.TMEMStages(tmem,        col_start=0,                       width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
O_region = T.TMEMStages(tmem,        col_start=MMA_N * SMEM_PIPE_DEPTH_Q, width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
P_region = T.TMEMStages(tmem_as_f16, col_start=MMA_N,                   width=BLK_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N * 2)
```

The `* 2` in `P_region`'s `col_start` and `stride` is the one place the aliasing leaks into the code: `S_region` and `O_region` are measured in fp32 `tmem` columns, but `P_region` is measured in fp16 `tmem_as_f16` columns, which are half as wide. With the regions defined, the compute code stays clean — it writes `S_region[q_stage]`, reads `S_region[wg_id, ...]`, writes `P_region[wg_id, ...]`, and accumulates into `O_region[i_q]`, never touching a raw column index.

**Try with your agent**: Ask it to explain the fp32 (`tmem`) and fp16 (`tmem_as_f16`) views in this FA4 kernel. Which physical TMEM regions hold `S`, `P`, and `O`, why does `P_region`'s stride use `MMA_N * 2`, and which consumers must finish before each region can be reused?

## How Barriers Connect the Roles

This is the hardest part of the kernel. Start with the handful of barriers that move data along the main compute path, and treat the rest as bookkeeping you can look up later. The data-ready handoffs are:

| Handoff | Meaning |
|---------|---------|
| TMA load -> score/value MMA | Q, K, or V has arrived in SMEM and can feed MMA |
| score MMA -> softmax | `S` is ready in TMEM |
| softmax/correction -> value MMA | `P` is ready in TMEM, and `O` is safe for accumulation |
| value MMA -> epilogue | final `O` is ready in TMEM |
| epilogue -> TMA store | `O_smem` is ready to store |

Everything not in that list is pipeline bookkeeping — barriers that release an SMEM, TMEM, or staging buffer so another role can reuse it. Any barrier, data or bookkeeping, reads as a tile handoff: who produced data, who consumes it, and which buffer becomes free once they are done.

![Flash Attention 4 MMA Input Gates](../img/flash_attention_main_handoff.png)

Read this diagram as correctness gates, not a schedule — it answers "what must be true before this MMA may fire," nothing about timing. Score MMA waits for Q and K in SMEM, then produces `S`. Value MMA waits on three things at once: V in SMEM, the `P` tile from softmax, and an `O` slot that WG2 has either released or rescaled. The softmax-to-value gate is split for the reason we met earlier — value MMA may begin after the first 96 columns of `P`, and `p_ready_2` releases the final 32.

One handoff does not fit the tile-readiness mold: the softmax-to-correction edge. Instead of passing a tile, softmax passes a single scalar — `acc_scale` during the K/V loop, or the final `row_sum` in the epilogue — through a one-slot SMEM mailbox to WG2. Because the slot is reused every iteration, a `full`/`empty` barrier pair guards it:

![Flash Attention 4 Softmax Scale-Slot Handshake](../img/flash_attention_softmax_correction.png)

Read `softmax_corr.full` and `softmax_corr.empty` as a producer-consumer pair:

1. Softmax waits for `softmax_corr.empty` before reusing the scale/sum slot.
2. Softmax writes `acc_scale` or final `row_sum` into that slot.
3. Softmax arrives on `softmax_corr.full`.
4. WG2 waits on `softmax_corr.full`, then reads the slot.
5. WG2 arrives on `softmax_corr.empty`.
6. The softmax warpgroup may reuse the slot in the next phase.

`softmax_corr.empty` means only that WG2 has consumed the scale/sum slot. It says nothing about whether `P` is ready, and it is *not* the gate that lets value MMA start. That gate is `p_o_rescale`, which fires when the first 96 columns of `P` are written and the `O` slot is safe to accumulate into. Confusing the two is a common source of wrong-result bugs.

With the main path understood, the full barrier list is a reference:

| Barrier | Producer -> consumer | What becomes safe |
|---------|----------------------|-------------------|
| `q_load.full` | TMA load -> score MMA | Q SMEM tile can feed MMA |
| `q_load.empty` | all score MMAs for this Q stage -> TMA load | Q SMEM stage can be reused for the next task |
| `kv_load.full` | TMA load -> score/value MMA | K or V SMEM tile can feed MMA |
| `kv_load.empty` | score/value MMA -> TMA load | K/V SMEM stage can be reused |
| `s_ready` | score MMA -> softmax | S TMEM tile can be read |
| `p_o_rescale` | softmax + WG2 -> value MMA | first 96 columns of P are in TMEM, and the O slot is safe for value MMA |
| `p_ready_2` | softmax -> value MMA | final quarter of P is in TMEM |
| `o_ready` | value MMA -> epilogue | final O accumulator is ready |
| `softmax_corr.full` | softmax -> WG2 | `acc_scale` or final `row_sum` is ready in the SMEM mailbox |
| `softmax_corr.empty` | WG2 -> softmax | the same SMEM mailbox slot can be reused after WG2 reads it |
| `corr_epi.full` | epilogue -> TMA store | O_smem is ready to store |
| `corr_epi.empty` | TMA store -> epilogue | O_smem stage can be reused |

As in GEMM, you can predict a barrier's type from who produces the signal:

- TMA loads use `TMABar`, because the TMA engine byte-counts its own completion.
- MMA completion uses `TCGen05Bar`, because `tcgen05.commit` signals the completion group.
- Pure thread-to-thread handoffs use `MBarrier`, where the participating threads arrive explicitly.

The split softmax-to-value handoff is worth a closer look. `p_o_rescale` lets the value MMA start once the first 96 columns of `P` are written and the `O` tile is safe to accumulate into. On the first K/V block, WG2 pre-arrives this barrier because there is no old `O` to rescale. On later K/V blocks, WG2 arrives after it has either skipped an unnecessary rescale or finished rescaling the old `O`. The rescale is skipped when the per-row scale is effectively 1.0 — softmax clamps `acc_scale` to exactly 1.0 when the row max barely moves, specifically when the log2-scaled max delta `(m_old - m_new) * scale_log2` stays within `rescale_threshold` so the `exp2` rescale factor rounds to 1.0, and WG2 reduces `should_rescale` across the warpgroup with `any_sync`, so it only touches `O` when at least one row needs it — the rescale it skips is a full TMEM → RF → TMEM read-modify-write over the whole `O` accumulator, wasted work once the max has stabilized and the scale rounds to 1.0. `p_ready_2` releases the last 32 columns of `P`. This matches the `96 + 32` value-MMA schedule from the previous section.

The new barriers all cluster in one place: `s_ready`, `p_o_rescale`, `p_ready_2`, and the softmax/correction pair are the barriers around softmax. They exist because the score MMA and value MMA are no longer adjacent. Register math, TMEM rewrites, and output rescaling sit between them, and every one of those steps needs a handoff.

**Try with your agent**: Ask it to trace one K/V block through `s_ready`, `p_o_rescale`, `p_ready_2`, and `o_ready`. For each barrier, ask who waits, who arrives, what tile becomes safe to read, and what storage can be reused afterward.

## Pipelining Structure

The barriers told us what must be *ready* before a role consumes a tile. They did not tell us what runs *concurrently*, which is the question here. The two are different: a correctness gate can be satisfied long before or long after the producer actually runs.

There is no single pipeline depth. Different tile streams move at different rates, so the kernel keeps a separate ring for each:

- Q pipeline depth 2: one CTA works on two Q stages. WG0 handles one stage, and WG1 handles the other.
- KV pipeline depth 3: K and V blocks stream through the inner loop while the same Q stages are reused.
- TMEM pipeline depth 2: each Q stage has its own S/P/O TMEM slots, and those slots are reused after the matching barriers fire.

![Flash Attention 4 Pipeline Structure](../img/flash_attention_pipeline_v2.png)

Read this as a timeline, not a barrier graph: it shows which roles are active at roughly the same moment, while the earlier barrier-flow figure is where you check the exact producer-consumer waits. The two figures answer the two different questions from the start of this section.

Each row matches one of the code's role branches:

- WG3 warp 1 issues TMA loads.
- WG3 warp 0 issues both score MMA and value MMA.
- WG0 and WG1 run softmax for the two Q stages.
- WG2 releases or rescales `O`, then later normalizes the final output.
- WG3 warp 2 issues the TMA store.

Following the figure left to right shows one representative pipeline wave. The load warp begins with `Q0`, `K[n-1]`, `Q1`, `V[n-1]`, then keeps streaming lower-index K/V blocks. The MMA warp issues the first score MMAs to produce `S0` and `S1`, and WG0/WG1 turn those into `P0` and `P1`.

The MMA warp does *not* run all score MMAs and then all value MMAs. Once both Q stages are primed it interleaves the two kinds — a value MMA for the current `V` block, then a score MMA for the next `K` block, and so on:

```text
score Q0*K[n-1]
score Q1*K[n-1]
value P0*V[n-1]
score Q0*K[n-2]
value P1*V[n-1]
score Q1*K[n-2]
value P0*V[n-2]
...
```

That interleaving is why the score, softmax, correction, and value rows all overlap in the figure rather than running in tidy succession.

The WG2 row reads `release / rescale`: on the first K/V block there is no old `O` yet, so WG2 only participates in the handoff that lets value MMA proceed; on later blocks it may rescale the old `O` before value MMA accumulates into it. Normalization and the TMA store happen only once, after the final K/V block of the attention task.

A single GEMM-style pipeline cannot describe FA4 because Q, K/V, and TMEM slots advance on independent schedules. TIRx keeps those schedules explicit — as separate tile buffers, `PipelineState` cursors, and barrier phases — rather than hiding the kernel behind one monolithic primitive. The complexity stays visible and inspectable.

## Rescaling and Writeback

The rescale is mandatory, not optional. Online softmax can raise the per-row maximum with each new score tile, and when it does, the `O` accumulated from earlier blocks was scaled by the *old* maximum. Each earlier term is therefore too large by a factor of `exp(m_new - m_old)`. Skip the correction and those blocks are over-weighted, so the final output is wrong. The fix is a TMEM → registers → TMEM tile operation:

$$O_{\text{old}} \leftarrow O_{\text{old}} \cdot e^{(m_{\text{old}} - m_{\text{new}}) / \sqrt{d}}$$

The work is split across two roles. Softmax computes the per-row scale and drops it in the SMEM mailbox; WG2 waits on `softmax_corr.full`, reads the current `O` from TMEM, multiplies by that scale, and writes `O` back:

```python
RESCALE_TILE = T.meta_var(16)
o_row = T.wg_reg_tile(RESCALE_TILE)
Tx.copy_async(o_row, O_region[i_q, d_start : d_start + RESCALE_TILE])
Tx.mul(o_row, o_row, acc_scale)
Tx.copy_async(O_region[i_q, d_start : d_start + RESCALE_TILE], o_row)
T.ptx.tcgen05.wait.st()
```

This is a full TMEM → registers → TMEM tile operation over the whole `O` accumulator, not scalar bookkeeping, with the same readout card as every other stage:

> **Tile-primitive readout — Correction (rescale)**
> - Scope: WG2, full warpgroup.
> - Layout: `O` in TMEM → registers → `O` in TMEM (`O_region[i_q]`).
> - Dispatch: `tcgen05.ld` to read, TMEM store to write; register multiply between them.
> - Handoff: waits `softmax_corr.full`; arrives `p_o_rescale` (→ value MMA) and `softmax_corr.empty` (→ softmax).

Tracing the synchronization end to end:

1. Softmax writes the scale value to SMEM.
2. WG2 waits on `softmax_corr.full`.
3. WG2 rescales `O` in TMEM.
4. WG2 arrives on `p_o_rescale`.
5. WG3's value MMA can now consume `P` and accumulate into the rescaled `O` tile.

The loop closes when `softmax_corr.empty` releases the SMEM slot after WG2 has read it, freeing softmax to reuse the mailbox next iteration.

Once the K/V loop ends, WG2 switches from correction to epilogue. It waits for the final `row_sum` and `o_ready`, reads the final `O` from TMEM, multiplies by `1 / row_sum` — the normalization deferred at the start — casts to fp16, and writes `O_smem`. WG3's TMA store warp carries `O_smem` back to GMEM.

One limitation matters for anyone extending this. The kernel computes the forward output only; a training forward pass would normally also store the log-sum-exp (LSE) needed by the backward pass. Adding that has a scaling detail to mind: this kernel keeps `row_max` as the maximum of the *raw*, unscaled `QK^T` scores, while `row_sum` accumulates `exp((S - row_max) / sqrt(d))`. So the `1/\sqrt{d}` factor has to be reapplied to `row_max` when forming the natural-log LSE:

$$\mathrm{LSE}_i = \log(\mathrm{row\_sum}_i) + \mathrm{row\_max}_i / \sqrt{d}$$

This implementation is forward-output only and does not write LSE.

## Causal Masking

Causal attention adds a constraint — a query may attend only to keys at or before its own position — and the kernel honors it in two complementary ways, one cheap and one precise.

The cheap one skips work entirely. Many K/V blocks sit fully above the diagonal and contribute nothing to a given Q block, so `get_n_block_max(...)` computes the last block that block could need, and the loop never loads or computes the rest.

The precise one handles the blocks that straddle the diagonal, where some columns are valid and some are not. Those blocks still run the score MMA, but softmax masks the invalid columns before exponentiation: for each row it derives a column limit from the row's query position and the block offset, keeps columns at or below it, and sets every column past it to `-inf` in registers so it contributes nothing to either the row max or the `exp2` numerator.

Rather than branch per element, the implementation applies the limit with `mask_r2p(...)`, which converts it into a bit mask over the whole 32-wide score chunk and masks the chunk in one shot. Blocks fully below the diagonal keep every column and need no mask at all.

From the tile-primitive view, causal mode does not rewrite the data path. It trims the K/V trip count and inserts a masking step into the register-resident softmax, between score MMA and `P` writeback.

## GQA Support

Grouped Query Attention lets several query heads share a single K/V head, which saves memory bandwidth but raises a packing question: how to keep one K/V tile and still feed many query heads through it. The kernel processes a whole group of query heads against one scheduled `kv_head_idx` at once:

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

The kernel reinterprets the 128 Q-tile rows. For `GQA_RATIO=4` they no longer mean 128 sequence positions; they mean 32 sequence positions times 4 query heads, packed together so all four heads ride the same K/V tile. The row decoding is:

```text
seq_pos = row // GQA_RATIO
q_head  = row % GQA_RATIO
```

The Q load expresses this packing with a 3D view. The source is the natural `Q[batch, seq, qo_head, dim]` layout, while the destination is the very same SMEM tile the score MMA later reads as a flat `128 x HEAD_DIM` operand — the view reconciles the two without copying:

```python
Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
Tx.copy_async(
    Q_smem_3d[i_q, :, :, :],
    Q[batch_idx,
      m_start : m_start + SEQ_Q_PER_TILE,
      kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
      :],
    **tma_copy_q,
)
```

K and V are never expanded in memory, which is the point of GQA. The single K/V tile for `kv_head_idx` is reused by all `GQA_RATIO` query heads packed into the Q rows. The output side mirrors the input: after the epilogue a matching 3D view stores the packed rows back to `O[batch, seq, qo_head, dim]`.

GQA lives entirely at the Q-load and O-store boundaries. Inside the compute path the score MMA still sees a plain `128 x HEAD_DIM` Q tile, and the rest of the tile-primitive graph is untouched — a feature confined to the edges of the kernel.

## Tile Scheduling

The scheduler's job is to map each CTA to a `(batch, kv_head, m_block)` attention task, and the right strategy depends on whether the masking makes tasks equal-cost:

- Non-causal mode uses `FlashAttentionLinearScheduler`. Every task does the same work, so a fixed CTA pool advancing by `num_ctas` is enough to spread them evenly.
- Causal mode uses `FlashAttentionLPTScheduler`, because causal masking makes the work wildly uneven — a Q block near the start attends to ~1 K/V block, one near the end to all of them. A naive split would leave some CTAs finishing long after others, so the longest-processing-time scheduler front-loads the heavy blocks to even out finish times (while keeping nearby batch/head tasks together for L2 locality).

Despite the different policies, the loop interface is identical:

```python
while scheduler.valid():
    m_block_idx = scheduler.m_block_idx
    batch_idx = scheduler.batch_idx
    kv_head_idx = scheduler.head_idx
    # process one Q block against its K/V block range
    scheduler.next_tile()
```

The only behavioral difference is what `next_tile()` does: in non-causal mode it advances the CTA to another task, while in causal mode it ends the loop after the current one. That is a scheduling decision only — it chooses *which* attention tile the CTA owns, never how the tile is computed. Inside the loop the same local primitives run regardless: TMA load, score MMA, softmax, value MMA, correction, TMA store.

## Compile and Verify

Everything above has been excerpts. To run the kernel you import the real thing from `tirx-kernels`, compile it, and check it against a torch reference. Two things differ from the GEMM verify cell: Flash Attention has a richer entry point (`get_flash_attention4_kernel`), and it takes an extra `profiler_buf` argument for its built-in profiler. This is the one cell to run for the whole chapter:

```python
import torch
import torch.nn.functional as F
import tvm
from tirx_kernels.attention.flash_attention4 import (
    get_flash_attention4_kernel, PROFILER_BUFFER_SIZE)

B, S, Hq, Hkv, D = 1, 1024, 32, 8, 128   # GQA: 32 query heads share 8 KV heads
Q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
O = torch.empty(B, S, Hq, D, dtype=torch.float16, device="cuda")
prof = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")

kernel = get_flash_attention4_kernel(B, S, S, Hq, Hkv, D, is_causal=False)
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
ex.mod(Q, K, V, O, prof)   # ex.mod takes torch tensors directly, like every other chapter
torch.cuda.synchronize()

# torch reference; enable_gqa lets the 32 query heads share the 8 KV heads
qt, kt, vt = (x.transpose(1, 2).float() for x in (Q, K, V))
ref = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True).transpose(1, 2).half()
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)
print(f"FA4: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}, non-causal -> PASS")
```

**Expected output**: `... -> PASS`. The kernel accumulates the online softmax in fp32, but several approximations still separate its result from a high-precision reference: fp16 storage and rounding of the inputs and operands, the `exp2`-based softmax reformulation (the `scale_log2 = log2(e)/√d` reframing of every exponential), the online-softmax reordering and per-row rescaling that sums the blocks in a running scale rather than all at once, and the final fp16 cast of `O` on writeback. The `rtol`/`atol` here — the same tolerance the source kernel's own test uses — is sized to cover all of these against the torch reference, not fp16 rounding alone. Treat a genuine failure here, not a borderline near-miss, as a signpost to the softmax path: a dropped `s_ready` / `p_o_rescale` / `p_ready_2` wait, or a `row_max` / `row_sum` update the rescale step failed to apply. Those are exactly the handoffs this chapter spent its barriers on.

## Differences from GEMM

| Aspect | GEMM | Flash Attention 4 |
|--------|------|-------------------|
| MMA phases | one repeated MMA | score MMA and value MMA |
| Work between MMAs | none beyond pipeline handoffs | online softmax, masking, and O rescaling |
| Running state | accumulator only | row max, row sum, O accumulator |
| Main intermediate | accumulator TMEM tile | S, P, and O TMEM tile regions |
| Warp roles | TMA producer, MMA consumer, writeback | TMA load, MMA, softmax, correction, TMA store |
| Barriers | mostly load/compute/writeback handoffs | additional score/softmax/value/correction handoffs |
| Scheduling unit | output matrix tile | attention task: `(batch, kv_head, m_block)` |

The differences all trace back to the structural change this chapter opened with: a second MMA with softmax between the two. The underlying TIRx contracts never changed:

- the tile primitive says what tile moves or computes,
- the surrounding scope says which threads cooperate,
- the layout says where the tile lives,
- the barrier says when the next role may consume it.

FA4 is harder than GEMM not because it uses different machinery, but because there are more tile values and more handoffs between them. The lens is the same; there is just more to look at.

## Exercises

1. Compared with GEMM, what new tile handoff appears between the two MMA phases in FA4? Name the producer, the TMEM tile, and the consumer.
2. Why does softmax write the numerator tile `P` back to TMEM instead of keeping it only in registers for the value MMA?
3. Pick `p_o_rescale` or `p_ready_2`. What exactly does the barrier prove, and what could go wrong if the value MMA skipped that wait?

**Try with your agent**: Pick one tile primitive this chapter did *not* walk through — for example a `Tx.copy_async` in the epilogue, the fp32→fp16 `Tx.cast`, or the second `gemm_pv` sub-MMA — and ask it to write that primitive's full scope / layout / dispatch / handoff card from the source alone: which threads issue or cooperate on it, where each operand tile lives (SMEM / TMEM / registers), which hardware path it lowers to, and which barrier makes its result safe to consume. Then audit the card against the kernel yourself — does the scope match the `wg_id` / `warp_id` guard the call sits under, and the layout match where that tile was allocated? Leaving this chapter able to read a primitive nobody annotated for you is the whole point of the scope/layout/dispatch lens.
