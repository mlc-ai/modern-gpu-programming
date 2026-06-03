# Flash Attention 4
:label:`chap_flash_attention`

We now move from GEMM to a more complex kernel: Flash Attention. It still uses the same tile-primitive machinery as the GEMM chapters: TMA tile movement, `tcgen05` MMA, TMEM, warpgroup-local register tiles, and explicit barriers. The extra complexity comes from the algorithm between the two MMA phases: online softmax, masking, rescaling, and final normalization.

This chapter keeps enough of the Flash Attention 4 algorithm to make the kernel readable, then focuses on how that algorithm is expressed in TIRX.

The easiest way to read the kernel is to follow the tile path. `Q`, `K`, and `V` start as input tiles loaded from GMEM into SMEM. The score MMA consumes `Q` and `K` to create the score tile `S` in TMEM. Softmax turns `S` into a numerator tile `P`, and the value MMA consumes `P` and `V` to update the output accumulator `O`. When the running softmax maximum changes, the old `O` tile must be rescaled before the next value MMA can accumulate into it. The sections below first explain this path, then show how TIRX assigns the work to warpgroups and connects the stages.

## Algorithm Shape

For one query block, Flash Attention computes:

$$O = \text{softmax}(QK^{\top} / \sqrt{d})V$$

without materializing the full attention matrix. The kernel streams K/V blocks and keeps three per-row running states:

- `row_max`: the maximum score seen so far.
- `row_sum`: the running denominator of softmax.
- `O`: the running output accumulator.

For each K/V block, the update is:

```text
S = Q_block @ K_block.T
m_new = max(row_max, rowmax(S))
scale = exp((row_max - m_new) / sqrt(d))
P = exp((S - m_new) / sqrt(d))
row_sum = row_sum * scale + rowsum(P)
O = O * scale + P @ V_block
row_max = m_new
```

Here `P` is not the final normalized attention matrix. It is the softmax numerator tile for the current K/V block. After all K/V blocks, the kernel writes `O / row_sum`.

For TIRX, the key question is not only what the algorithm computes, but where each tile lives while the kernel runs. `S`, `P`, and `O` are tile values:

- `S` is the score tile. The score MMA writes it to TMEM.
- `P` is the softmax numerator tile. Softmax reads `S` from TMEM into registers, computes `P = exp((S - m_new) / sqrt(d))`, and writes `P` back to TMEM.
- `O` is the output accumulator tile. The value MMA reads `P` from TMEM and `V` from SMEM, then accumulates into `O` in TMEM.

When `row_max` changes, the old `O` tile has to be rescaled before the next value MMA accumulates into it. That rescaling is also a tile operation: read `O` from TMEM, multiply in registers, and write `O` back to TMEM.

## Tile-Primitive Graph

Start with the short version. For one K/V block, the kernel follows this tile path:

```text
Q, K, V in GMEM
  -> Q, K, V in SMEM        by TMA load
  -> S in TMEM              by score MMA: QK^T
  -> P in TMEM              by softmax numerator: TMEM -> RF -> TMEM
  -> O in TMEM              by value MMA: P V
  -> O in GMEM              by normalization, SMEM staging, and TMA store
```

This is the FA4 version of the GEMM data path. GEMM has one repeated MMA chain. FA4 has two MMA phases, and the middle of the chain is softmax.

The full graph below expands that short path into producer-consumer edges:

| Stage | Tile movement or compute | TIRX primitive | Hardware path |
|-------|--------------------------|----------------|---------------|
| Load Q/K/V | GMEM tiles -> SMEM tiles | `Tx.copy_async(..., dispatch="tma")` | TMA load |
| Score MMA | Q in SMEM and K in SMEM -> score tile `S` in TMEM | `Tx.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` |
| Softmax read | `S` in TMEM -> warpgroup register tile | `Tx.copy_async(reg, tmem)` under `Tx.warpgroup()` | `tcgen05.ld` |
| Softmax write | numerator tile `P` in registers -> fp16 TMEM view | `Tx.copy_async(tmem_as_f16, reg)` | TMEM store, followed by `tcgen05.wait.st()` |
| Value MMA | `P` in TMEM and V in SMEM -> output accumulator `O` in TMEM | `Tx.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` with a TMEM operand |
| Correction | `O` in TMEM -> registers -> `O` in TMEM | TMEM readback, register multiply, TMEM store | `tcgen05.ld` / TMEM store |
| Epilogue | final `O` in TMEM -> registers -> SMEM -> GMEM | TMEM readback, `Tx.copy`, TMA store | `tcgen05.ld` + TMA store |

The middle steps are still tile operations. Softmax reads a score tile from TMEM into warpgroup registers, does row-wise math, and writes a `P` tile back to TMEM. Correction reads an `O` tile from TMEM, rescales it, and writes it back.

**Try with your agent**: Ask it to trace only the short path above. For each arrow, name the producer role, consumer role, source tile, destination tile, and hardware path. Then ask which arrows did not exist in the GEMM chapters.

## Warp Roles and Scopes

Each CTA in this FA4 kernel has 4 warpgroups, 512 threads total. The split is easier to read as two groups:

- WG3 drives the hardware engines: TMA load, MMA, and TMA store.
- WG0, WG1, and WG2 do the register-heavy work between hardware operations: softmax, correction, and epilogue.

The exact role table is:

| Owner | Role | What it does |
|-------|------|--------------|
| WG3, warp 1 | TMA load | Loads Q, K, and V tiles from GMEM to SMEM |
| WG3, warp 0 | MMA | Issues both score MMA and value MMA |
| WG3, warp 2 | TMA store | Stores final O tiles from SMEM to GMEM |
| WG0 | Softmax for Q stage 0 | Reads S from TMEM, computes P, writes P to TMEM |
| WG1 | Softmax for Q stage 1 | Same work for the second Q pipeline stage |
| WG2 | Correction and epilogue | Rescales O in TMEM, normalizes, stages output |

The two Q stages are the two entries in the Q pipeline. WG0 handles one stage while WG1 handles the other. They are not different attention heads; they are two pipeline slots for different Q tiles.

The code selects these roles with symbolic coordinates:

```python
wg_id = Tx.warpgroup_id([4])
warp_id = Tx.warp_id_in_wg([4])
```

When reading the code, first identify the role branch. That branch tells you which execution team owns the tile primitive inside it:

- WG3 warp 1 starts TMA load commands. One elected lane issues the copy, and the TMA engine moves the tile.
- WG3 warp 0 issues the `tcgen05.mma` instructions.
- WG0 and WG1 run softmax under full warpgroup scope.
- WG2 runs correction and epilogue work under full warpgroup scope.

All MMA instructions are issued from WG3 warp 0. WG0 and WG1 do not issue MMA. They consume the score tile, run softmax, and write the `P` tile back to TMEM.

This matters for barriers. `s_ready` connects score MMA to softmax. `p_o_rescale` connects softmax and correction back to the value MMA.

## The Two MMA Phases

For each streamed K/V tile, Flash Attention runs two MMA phases with softmax in between:

```text
Q, K -> score MMA -> S
S    -> softmax   -> P
P, V -> value MMA -> O
```

The first MMA produces attention scores. The second MMA consumes the softmax numerator tile and updates the output accumulator. Final normalization by `row_sum` happens in the epilogue.

### Score MMA

The score MMA computes:

$$S = Q_{\text{block}}K_{\text{block}}^{\top}$$

and writes the `128 x 128` score tile to TMEM:

```python
with Tx.warp():
    Tx.gemm_async(
        tmem[0:128, tmem_col_s : tmem_col_s + MMA_N],
        Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
        K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
        dispatch="tcgen05",
        cta_group=CTA_GROUP,
    )
if Tx.ptx.elect_sync():
    s_ready.arrive(q_stage)
```

Read this with the same rules as GEMM:

- source tiles: Q and K in SMEM,
- destination tile: `S` in TMEM,
- dispatch path: `tcgen05`,
- handoff after compute: `s_ready`.

The elected thread arrival on `s_ready` says this score tile is ready for the softmax warpgroup.

### Softmax Between MMAs

Softmax is the part that makes FA4 different from GEMM. WG0/WG1 wait for the score tile, then read it from TMEM in register chunks:

```python
Tx.copy_async(
    s_chunk[:, chunk_start : chunk_end],
    tmem[:, tmem_col_s + chunk_start : tmem_col_s + chunk_end],
)
```

This is a TMEM-to-RF tile read under warpgroup scope. After the read, the softmax warpgroup does three things:

1. computes the row max and row sum,
2. computes the softmax numerator tile `P`,
3. writes `P` back to TMEM as fp16.

The writeback looks like:

```python
Tx.copy_async(
    tmem_as_f16[:, tmem_col_p * 2 + p_start : tmem_col_p * 2 + p_end],
    p_chunk[:, p_start : p_end],
)
```

The writeback matters because the value MMA needs `P` as a tile operand. It cannot consume `P` as unrelated per-thread scalar registers. In this kernel, the MMA-readable form of `P` is the fp16 TMEM view `tmem_as_f16`.

### Value MMA

The value MMA computes:

$$O = O + P_{\text{block}}V_{\text{block}}$$

Here `O` has already been initialized or rescaled for this K/V block. The A operand is `P` in TMEM, the B operand is `V` in SMEM, and the output accumulator is `O` in TMEM:

```python
with Tx.warp():
    Tx.gemm_async(
        tmem[0:128, tmem_col_o : tmem_col_o + MMA_N],
        tmem_as_f16[0:128, tmem_col_p * 2 : tmem_col_p * 2 + K_SPLIT],
        V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
        transB=True,
        accum=should_accumulate,
        dispatch="tcgen05",
        cta_group=CTA_GROUP,
    )
```

This is the main hardware difference from the score MMA:

- Score MMA reads both operands from SMEM: Q and K.
- Value MMA reads one operand from TMEM: `P`.
- Value MMA reads the other operand from SMEM: V.
- The result accumulates into `O` in TMEM.

`accum=should_accumulate` controls whether this K/V tile initializes the output accumulator or adds into the existing `O` tile.

The value MMA is split into a `96 + 32` schedule:

1. Softmax writes `P` in four 32-column chunks.
2. After the first three chunks are ready, the value MMA starts on the first 96 columns of `P` and the matching rows of `V`.
3. The final 32 columns wait for `p_ready_2`.
4. A second MMA consumes that final chunk and finishes the tile.

This lets value MMA begin before the last part of the softmax-to-TMEM writeback is done.

## TMEM Layout and Reuse

The kernel uses one `128 x 512` TMEM allocation:

![TMEM Layout](../img/tmem_layout_v3.png)

The figure is easiest to read as a set of tile slots:

- Score slots hold `S = QK^T`.
- Numerator slots hold the `P` tile after the softmax exponentiation step.
- Output slots hold the fp32 `O` accumulator.

These are not independent buffers in global memory. They are regions of the same TMEM allocation. The schedule is valid because each region is reused only after the previous consumer has finished. That is why barriers are part of the layout story: TMEM reuse is safe only when the producer-consumer handoff is complete.

`tmem` is declared as fp32 for score and output accumulator views:

```python
tmem = Tx.decl_buffer(
    (128, N_COLS_TMEM),
    "float32",
    scope="tmem",
    allocated_addr=0,
    layout=TileLayout(S[(128, N_COLS_TMEM) : (1 @ TLane, 1 @ TCol)]),
)
```

The same physical allocation also has an fp16 view for `P`:

```python
tmem_as_f16 = Tx.decl_buffer(
    (128, N_COLS_TMEM * 2),
    "float16",
    scope="tmem",
    allocated_addr=0,
    layout=TileLayout(S[(128, N_COLS_TMEM * 2) : (1 @ TLane, 1 @ TCol)]),
)
```

The fp16 view has twice as many indexable columns. This is why the code indexes `P` with `tmem_col_p * 2`: `tmem_col_p` is the base slot used in the TMEM layout, while `tmem_as_f16` is indexed through the fp16 view.

**Try with your agent**: Ask it to explain the fp32 and fp16 TMEM views in this FA4 kernel. Which physical TMEM regions hold `S`, `P`, and `O`, why is `P` indexed with `tmem_col_p * 2`, and which consumers must finish before each region can be reused?

## How Barriers Connect the Roles

The barrier graph is the hardest part of the kernel. Do not try to memorize the full table first. Start with the data-ready handoffs on the main compute path:

| Handoff | Meaning |
|---------|---------|
| TMA load -> score/value MMA | Q, K, or V has arrived in SMEM and can feed MMA |
| score MMA -> softmax | `S` is ready in TMEM |
| softmax/correction -> value MMA | `P` is ready in TMEM, and `O` is safe for accumulation |
| value MMA -> epilogue | final `O` is ready in TMEM |
| epilogue -> TMA store | `O_smem` is ready to store |

The rest of the barriers are mostly pipeline bookkeeping: they release SMEM, TMEM, or staging buffers so another role can reuse them.

Read each barrier as a tile handoff: which role produced data, which role consumes it, and which buffer becomes reusable afterward.

![Flash Attention 4 MMA Input Gates](../img/flash_attention_main_handoff.png)

This diagram is about correctness gates, not scheduling. It shows what must be ready before each MMA phase may run. Score MMA waits for Q and K in SMEM, then produces `S`. Value MMA waits for V in SMEM, the `P` tile from softmax, and an `O` slot that WG2 has either released or rescaled. The softmax-to-value gate is split because value MMA can start after the first 96 columns of `P`, while the final 32 columns are released by `p_ready_2`.

The softmax/correction handoff needs a different view. It uses a small SMEM slot as a mailbox between the softmax warpgroup and WG2. That mailbox carries either `acc_scale` during the K/V loop or final `row_sum` during the epilogue. The `full` and `empty` barriers protect that mailbox slot:

![Flash Attention 4 Softmax Scale-Slot Handshake](../img/flash_attention_softmax_correction.png)

Read `softmax_corr.full` and `softmax_corr.empty` as a producer-consumer pair:

1. Softmax waits for `softmax_corr.empty` before reusing the scale/sum slot.
2. Softmax writes `acc_scale` or final `row_sum` into that slot.
3. Softmax arrives on `softmax_corr.full`.
4. WG2 waits on `softmax_corr.full`, then reads the slot.
5. WG2 arrives on `softmax_corr.empty`.
6. The softmax warpgroup may reuse the slot in the next phase.

That is all `softmax_corr.empty` means: WG2 has consumed the SMEM scale/sum slot. It does not mean `P` is ready, and it does not mean value MMA may start. The value-MMA gate is `p_o_rescale`: the first 96 columns of `P` are ready, and WG2 has made the `O` slot safe to accumulate into.

The full barrier list is still useful as a reference:

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

The barrier type follows the producer:

- TMA loads use `TMABar`, because completion is byte-counted by the TMA engine.
- MMA completion uses `TCGen05Bar`, because `tcgen05.commit` signals the completion group.
- Pure thread-to-thread handoffs use `MBarrier`, where the participating threads arrive explicitly.

Two barriers split the softmax-to-value handoff. `p_o_rescale` lets the value MMA start once the first 96 columns of `P` are written and the `O` tile is safe to accumulate into. On the first K/V block, WG2 pre-arrives this barrier because there is no old `O` to rescale. On later K/V blocks, WG2 arrives after it has either skipped an unnecessary rescale or finished rescaling the old `O`. `p_ready_2` releases the last 32 columns of `P`. This matches the `96 + 32` value-MMA schedule from the previous section.

Compared with GEMM, the new barriers are the ones around softmax: `s_ready`, `p_o_rescale`, `p_ready_2`, and the softmax/correction pair. They exist because the score MMA and value MMA are separated by register math, TMEM rewrites, and output rescaling.

**Try with your agent**: Ask it to trace one K/V block through `s_ready`, `p_o_rescale`, `p_ready_2`, and `o_ready`. For each barrier, ask who waits, who arrives, what tile becomes safe to read, and what storage can be reused afterward.

## Pipelining Structure

The previous section answered: what must be ready before a role may consume a tile? This section answers a different question: which roles can run at the same time?

The kernel does not have one single pipeline depth. It has separate rings for the tile streams that move at different rates:

- Q pipeline depth 2: one CTA works on two Q stages. WG0 handles one stage, and WG1 handles the other.
- KV pipeline depth 3: K and V blocks stream through the inner loop while the same Q stages are reused.
- TMEM pipeline depth 2: each Q stage has its own S/P/O TMEM slots, and those slots are reused after the matching barriers fire.

![Flash Attention 4 Pipeline Structure](../img/flash_attention_pipeline_v2.png)

This figure is a timeline, not a barrier graph. Use it to see which role is active at roughly the same time; use the previous barrier-flow figure to check the exact producer-consumer waits.

The rows in the figure match the code's role branches:

- WG3 warp 1 issues TMA loads.
- WG3 warp 0 issues both score MMA and value MMA.
- WG0 and WG1 run softmax for the two Q stages.
- WG2 releases or rescales `O`, then later normalizes the final output.
- WG3 warp 2 issues the TMA store.

Read the figure left to right as a representative pipeline wave. The load warp first loads `Q0`, `K[n-1]`, `Q1`, and `V[n-1]`, then keeps streaming lower-index K/V blocks. The MMA warp first issues score MMAs to produce `S0` and `S1`. WG0 and WG1 turn those score tiles into `P0` and `P1`.

After the first two score MMAs, the MMA warp does not switch into a separate value-only phase. It interleaves the two kinds of MMA: value MMA for the current `V` block, then score MMA for the next `K` block. A typical sequence is:

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

That interleaving is why the score, softmax, correction, and value rows overlap in the figure.

The WG2 row says `release / rescale` because the first K/V block has no old `O` to rescale, but WG2 still participates in the handoff that lets value MMA proceed. On later K/V blocks, WG2 may actually rescale the old `O` tile before value MMA accumulates into it. Normalization and TMA store happen only after the last K/V block for the current attention task.

This is why a single GEMM-style pipeline is not enough. Q, K/V, and TMEM slots advance on different schedules. TIRX keeps those schedules visible as separate tile buffers, `PipelineState` cursors, and barrier phases instead of hiding the whole attention kernel behind one monolithic primitive.

## Rescaling and Writeback

Online softmax can change the per-row maximum after each new score tile. When that happens, the output accumulated from earlier K/V blocks is in the old scale and must be rescaled before the next value MMA adds into it:

$$O_{\text{old}} \leftarrow O_{\text{old}} \cdot e^{(m_{\text{old}} - m_{\text{new}}) / \sqrt{d}}$$

Softmax computes the per-row scale and writes it to SMEM. WG2 waits for that scale through `softmax_corr.full`, reads the current `O` accumulator from TMEM, multiplies by the per-row scale, and writes `O` back to TMEM:

```python
Tx.copy_async(o_row_wg, tmem[:, tmem_col_o_stage + d_start : tmem_col_o_stage + d_start + 16])
with Tx.thread():
    Tx.mul(o_row_buf, o_row_buf, acc_scale)
Tx.copy_async(tmem[:, tmem_col_o_stage + d_start : tmem_col_o_stage + d_start + 16], o_row_wg[:, 0:16])
Tx.ptx.tcgen05.wait.st()
```

This is not scalar bookkeeping. It is another TMEM -> RF -> TMEM tile operation.

The synchronization is:

1. Softmax writes the scale value to SMEM.
2. WG2 waits on `softmax_corr.full`.
3. WG2 rescales `O` in TMEM.
4. WG2 arrives on `p_o_rescale`.
5. WG3's value MMA can now consume `P` and accumulate into the rescaled `O` tile.

After WG2 reads the scale value, `softmax_corr.empty` releases that SMEM slot so the softmax warpgroup can reuse it.

At the end of the K/V loop, WG2 switches from correction to epilogue. It waits for the final `row_sum` and `o_ready`, reads the final `O` accumulator from TMEM, multiplies by `1 / row_sum`, casts to fp16, and writes `O_smem`. WG3's TMA store warp then moves `O_smem` back to GMEM.

The current kernel computes the forward output only. A training forward kernel would normally also store log-sum-exp for backward:

$$\mathrm{LSE}_i = \log(\mathrm{row\_sum}_i) + \mathrm{row\_max}_i$$

This implementation is forward-output only and does not write LSE.

## Causal Masking

Causal attention changes which score elements are valid: a query position may only attend to keys at or before that position. In this kernel, causal handling appears in two places.

First, the K/V loop can stop early for each Q block. `get_n_block_max(...)` computes the last K/V block that this Q block may need, so the kernel does not load or compute K/V blocks that are entirely above the causal diagonal.

Second, blocks that cross the diagonal still run score MMA, but the softmax stage masks invalid columns before exponentiation. For each row, the code computes a column limit and sets scores beyond that limit to `-inf` in registers:

```python
mask = (1 << col_limit) - 1
in_bound = mask & (1 << i)
s_chunk[c] = s_chunk[c] if in_bound else -inf
```

The real implementation applies this in chunks with `mask_r2p(...)`, which builds a bit mask instead of branching on every score element. Blocks fully below the diagonal do not need this register mask; blocks crossing the diagonal do.

From the tile-primitive point of view, causal mode does not replace the main data path. It changes the K/V trip count and inserts masking into the RF softmax step between score MMA and `P` writeback.

## GQA Support

Grouped Query Attention shares one K/V head across multiple query heads. For a scheduled `kv_head_idx`, the kernel processes the corresponding group of query heads together:

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

For `GQA_RATIO=4`, the 128 rows of the Q tile represent 32 sequence positions times 4 query heads. The packed row mapping is:

```text
seq_pos = row // GQA_RATIO
q_head  = row % GQA_RATIO
```

The Q TMA load uses a 3D view to describe that packing. The source is `Q[batch, seq, qo_head, dim]`, while the destination is the same physical SMEM tile that the score MMA reads as a flat `128 x HEAD_DIM` operand:

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

K and V are not expanded in memory. The same K/V tile for `kv_head_idx` is reused by all `GQA_RATIO` query heads packed into the Q rows.

The output side mirrors the input side. After the epilogue, the kernel uses a matching 3D view to store the packed rows back to `O[batch, seq, qo_head, dim]`.

So GQA mainly changes the interpretation at the Q load and O store boundaries. Inside the compute path, the score MMA still sees a regular `128 x HEAD_DIM` Q tile, and the rest of the tile-primitive graph stays the same.

## Tile Scheduling

The scheduler maps each CTA to a `(batch, kv_head, m_block)` attention task. The kernel has two scheduling modes:

- Non-causal mode uses `FlashAttentionLinearScheduler`. The launch uses a fixed pool of CTAs, and each CTA advances by `num_ctas` to process multiple tasks.
- Causal mode uses `FlashAttentionLPTScheduler`. The launch uses one CTA per task, but the task order is rearranged so heavier Q blocks appear earlier and nearby batch/head tasks keep better L2 locality.

The code interface has the same shape in both modes:

```python
while scheduler.valid():
    m_block_idx = scheduler.m_block_idx
    batch_idx = scheduler.batch_idx
    kv_head_idx = scheduler.head_idx
    # process one Q block against its K/V block range
    scheduler.next_tile()
```

In non-causal mode, `scheduler.next_tile()` advances to another task for the same CTA. In causal mode, it ends the loop after the current task. Either way, scheduling only decides which attention tile the CTA owns. The tile primitives inside the loop remain the same local operations: TMA load, score MMA, softmax, value MMA, correction, and TMA store.

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

FA4 still uses the same local TIRX contracts:

- the tile primitive says what tile moves or computes,
- the surrounding scope says which threads cooperate,
- the layout says where the tile lives,
- the barrier says when the next role may consume it.

FA4 is harder than GEMM because there are more tile values and more producer-consumer handoffs between them.

## Exercises

1. Compared with GEMM, what new tile handoff appears between the two MMA phases in FA4? Name the producer, the TMEM tile, and the consumer.
2. Why does softmax write the numerator tile `P` back to TMEM instead of keeping it only in registers for the value MMA?
3. Pick `p_o_rescale` or `p_ready_2`. What exactly does the barrier prove, and what could go wrong if the value MMA skipped that wait?
