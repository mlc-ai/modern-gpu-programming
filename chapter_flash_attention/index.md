# Flash Attention 4
:label:`chap_flash_attention`

*Prerequisites: You should be comfortable with Step 7 (Warp Specialization) — specifically warp specialization, the 4-barrier producer-consumer pattern, PipelineState, and TMA/tcgen05 pipelining. If Step 7 still feels unclear, go back and re-read it before continuing.*

*This chapter explains how the same TIRX patterns from GEMM generalize to attention kernels. The focus is on architecture and algorithm design rather than line-by-line code.*

---

Flash Attention 4 is the Blackwell-native implementation of Flash Attention, designed specifically for SM100a GPUs. It fuses the entire attention computation ($QK^T \to \text{softmax} \to \times V$) into a single kernel, avoiding materializing the full $N \times N$ attention matrix. This chapter explores how Flash Attention 4 maps to Blackwell hardware using TIRX.


## What You Will Learn

- How online softmax works, step by step with a numeric example

- How Flash Attention 4's tiled algorithm maps to TIRX's warp-specialized architecture

- The two MMA phases (Score and Value) and the dataflow between them

- The barrier topology connecting 4 warpgroups (512 threads)

- Writeback and adaptive rescaling

- Causal masking and Grouped Query Attention (GQA)


## Online Softmax

Standard softmax requires two passes over the data: one to find the maximum (for numerical stability), and one to compute the exponentials and their sum. For a vector $x = [x_1, x_2, \ldots, x_N]$:

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\ \sum_j e^{x_j - \max(x)}}$$

This is problematic for attention because the score vector has length $N$ (the sequence length), and materializing it requires $O(N^2)$ memory across all query positions. Online softmax eliminates this by processing the scores in chunks, maintaining a running maximum and sum that get updated as each new chunk arrives.

### The core idea

For each row, we maintain two running values: a maximum $m$ and an exponential sum $\ell$. When a new chunk of scores arrives, we update both:

1. Compute the new maximum: $m_{\text{new}} = \max(m_{\text{old}}, \max(\text{chunk}))$
2. Rescale the old sum: $\ell = \ell \cdot e^{m_{\text{old}} - m_{\text{new}}}$
3. Add the new terms: $\ell = \ell + \sum_j e^{\text{chunk}_j - m_{\text{new}}}$

The rescaling in step 2 is the key insight. Since $m_{\text{old}} \le m_{\text{new}}$ (the maximum can only increase), the exponent $m_{\text{old}} - m_{\text{new}} \le 0$, so the rescaling factor $e^{m_{\text{old}} - m_{\text{new}}} \le 1$. This means we only ever multiply by values in $[0, 1]$, which is numerically stable.

### Numeric example

Consider computing softmax over 4 elements `[2, 5, 1, 8]` processed in two chunks of size 2.

**Chunk 1:** [2, 5]

$$m = 5, \quad \ell = e^{2-5} + e^{5-5} = e^{-3} + 1 \approx 0.050 + 1.0 = 1.050$$

**Chunk 2:** [1, 8]

$$m_{\text{new}} = \max(5, 8) = 8$$
$$\ell = 1.050 \cdot e^{5-8} + e^{1-8} + e^{8-8} = 1.050 \cdot e^{-3} + e^{-7} + 1$$
$$\approx 0.0523 + 0.000912 + 1.0 = 1.053$$

The final softmax values are $e^{x_i - 8} / 1.053$ for each element, which matches the two-pass result.

### Extending to Flash Attention 4

In Flash Attention 4, we track not just $(m, \ell)$ but also an output accumulator $O$. When the maximum changes, $O$ must be rescaled by the same factor:

$$O = O \cdot e^{m_{\text{old}} - m_{\text{new}}} + P_{\text{new}} \times V_{\text{block}}$$

where $P_{\text{new}} = e^{S_{\text{block}} - m_{\text{new}}}$ are the softmax weights for the current block. After processing all KV blocks, the final output is $O / \ell$ (dividing each row by its accumulated sum).

This is the complete Flash Attention 4 loop:

```
For each Q block (BLK_M rows):
    Initialize: max = -inf, sum = 0, O = 0
    For each KV block (BLK_N columns):
        S = Q_block @ K_block^T              # Scores: [BLK_M, BLK_N]
        m_new = max(max, rowmax(S))          # Update running max
        P = exp(S - m_new)                   # Softmax numerator
        sum = sum * exp(max - m_new) + rowsum(P)  # Rescale and update sum
        O = O * exp(max - m_new) + P @ V_block    # Rescale and accumulate
        max = m_new
    O = O / sum                              # Final normalization
```

All operations on `max`, `sum`, and `O` are **per-row** — each of the BLK_M rows maintains its own running state independently.


## Kernel Architecture

The Blackwell Flash Attention 4 kernel uses 4 warpgroups (512 threads) with full warp specialization:

| Warpgroup | Role | Description |
|-----------|------|-------------|
| **WG3, warp 0** | MMA warp | Issues **both** Score MMA (Q@K^T) and Value MMA (P@V) via `tcgen05.mma` |
| **WG3, warp 1** | TMA Load | Loads Q, K, V tiles from GMEM to SMEM via TMA |
| **WG3, warp 2** | TMA Store | Stores O tiles from SMEM to GMEM via TMA |
| **WG0** | Softmax | Runs online softmax on scores (Q stage 0), writes P to TMEM |
| **WG1** | Softmax | Same as WG0 but handles Q stage 1 (double-buffered Q) |
| **WG2** | Correction + Epilogue | Rescales O accumulator in TMEM, normalizes, writes to SMEM |

**All MMA instructions are issued by WG3's warp 0** — this is different from the warp role table description in some papers. WG0 and WG1 are purely softmax warpgroups: they read scores from TMEM, compute softmax (row_max, row_sum, exp), and write the probability matrix P back to TMEM. They do **not** issue any MMA instructions.

WG0 and WG1 each handle one of two Q pipeline stages. While WG0 processes Q stage 0 through softmax, WG1 processes Q stage 1. This is a form of **Q double-buffering** at the warpgroup level.

**Key constants** (from `flash_attention4.py`):

```python
BLK_M, BLK_N, BLK_K = 128, 128, 64    # Tile dimensions
WG_NUMBER = 4                           # 4 warpgroups = 512 threads
TMEM_PIPE_DEPTH = 2                     # Double-buffered TMEM
SMEM_PIPE_DEPTH_Q = 2                   # Q pipeline stages
SMEM_PIPE_DEPTH_KV = 3                  # KV pipeline stages
SM_NUMBER = 148                         # B200 SM count
N_COLS_TMEM = 512                       # Total TMEM columns allocated
```


## The Two MMA Phases

Flash Attention 4 requires two distinct MMA operations per KV block, creating a pipeline that is fundamentally different from GEMM's single-MMA pattern.

### Phase 1: Score MMA (Q @ K^T -> S)

The first MMA computes attention scores:

$$S = Q_{\text{block}} \times K_{\text{block}}^T \quad [\text{BLK}_M \times \text{BLK}_N]$$

This writes the score matrix $S$ into TMEM. WG3's MMA warp (warp 0) issues the Score MMA for both Q stages sequentially, then the `bar_s_full` barrier signals that scores are ready for WG0/WG1 to run softmax.

Simplified excerpt (the real implementation wraps `arrive` in an `elect_sync` guard and handles both Q stages in an unrolled loop — see `flash_attention4.py` for full details):

```python
# WG3, warp 0: Issues Score MMA (simplified)
def gemm_qk(q_stage, kv_stage, tmem_col_s, bar_s_full):
    with Tx.warp():
        Tx.gemm_async(
            tmem[0:128, tmem_col_s : tmem_col_s + MMA_N],
            Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
            K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
            dispatch="tcgen05", cta_group=CTA_GROUP,
        )
    # In the real code: wrapped in elect_sync()
    bar_s_full.arrive(q_stage)  # Signal: scores ready in TMEM
```

### Between phases: Softmax in registers

WG0/WG1 wait on `bar_s_full`, then read $S$ from TMEM into registers and run online softmax:

1. **Read S from TMEM** in chunks of 32 columns via `Tx.copy_async` (TMEM -> registers)
2. **Compute row max** and update the running maximum
3. **Compute rescaling factor** $e^{m_{\text{old}} - m_{\text{new}}}$ using an exp2 polynomial approximation
4. **Write acc_scale to SMEM** for WG2 (the correction warpgroup) to rescale O in TMEM
5. **Compute $P = e^{(S - m_{\text{new}}) \cdot \text{scale\_log2}}$** using exp2 emulation, cast to float16
6. **Write P to TMEM** (as float16) via `Tx.copy_async` (registers -> TMEM)
7. **Signal `bar_p_full_o_rescaled`** so that the Value MMA can begin

The exp2 emulation is a custom polynomial approximation (degree 3 Horner evaluation) implemented with packed f32x2 PTX instructions for maximum throughput. The kernel computes $2^{x}$ rather than $e^x$ and adjusts the scale factor accordingly ($\text{scale\_log2} = \log_2(e) / \sqrt{d}$).

### Phase 2: Value MMA (P @ V -> O)

The second MMA accumulates the weighted values:

$$O = O + P_{\text{block}} \times V_{\text{block}} \quad [\text{BLK}_M \times \text{HEAD}_{\text{DIM}}]$$

This is issued by WG3's MMA warp (warp 0). P lives in TMEM as float16 (written there by WG0/WG1), while V lives in SMEM. The result accumulates into O in TMEM as float32.

Simplified excerpt (barrier waits and elect_sync guards omitted — see `flash_attention4.py:gemm_pv` for the full version):

```python
# WG3, warp 0: Issues Value MMA (simplified)
# P is read from TMEM (A operand), V from SMEM (B operand, transposed)
K_SPLIT = 6 * MMA_K  # 96 — first 6 of 8 MMA iterations

# Part 1: P columns 0..95 x V rows 0..95 (can start before softmax finishes last quarter)
Tx.gemm_async(
    tmem[0:128, tmem_col_o : tmem_col_o + MMA_N],
    tmem_as_f16[0:128, tmem_col_p*2 : tmem_col_p*2 + K_SPLIT],
    V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
    transB=True, accum=should_accumulate, dispatch="tcgen05", cta_group=CTA_GROUP,
)

# Wait for softmax to finish writing last quarter of P (barrier omitted here)

# Part 2: P columns 96..127 x V rows 96..127
Tx.gemm_async(
    tmem[0:128, tmem_col_o : tmem_col_o + MMA_N],
    tmem_as_f16[0:128, tmem_col_p*2 + K_SPLIT : tmem_col_p*2 + BLK_N],
    V_smem[kv_stage, K_SPLIT:BLK_N, 0:HEAD_DIM],
    transB=True, accum=True, dispatch="tcgen05", cta_group=CTA_GROUP,
)
```

Notice the split: P@V is broken into two parts (first 96 columns, then last 32 columns). This allows the Value MMA to start before the softmax warpgroup finishes writing the last quarter of P to TMEM, overlapping computation with the TMEM store.

### TMEM Layout

The kernel uses a single TMEM allocation of 128 rows x 512 columns, partitioned as:

![TMEM Layout](../img/tmem_layout.png)

Each Q stage uses `tmem_offset=128` columns of f32 (or 256 columns of f16 for P). The S and P regions partially overlap in TMEM column space, which is safe because S is fully consumed before P is written.


## Barrier Topology

Flash Attention 4's barrier system is more complex than GEMM's because of the two MMA phases and the softmax computation between them. Below is every barrier in the kernel.

### Barrier Table

| Barrier | Type | Direction | Init Count | Purpose |
|---------|------|-----------|------------|---------|
| `bar_load_q_full` | ExpectTx | TMA -> MMA warp | 1 | Q tile loaded into SMEM |
| `bar_load_q_empty` | Commit | MMA warp -> TMA | 1 | Q SMEM buffer consumed by Score MMA |
| `bar_load_kv_full` | ExpectTx | TMA -> MMA warp | 1 | K or V tile loaded into SMEM |
| `bar_load_kv_empty` | Commit | MMA warp -> TMA | 1 | K/V SMEM buffer consumed by MMA |
| `bar_s_full` | Commit | MMA warp -> WG0/WG1 | 1 | Score MMA complete, S in TMEM |
| `bar_p_full_o_rescaled` | Arrive | WG0/WG1 + WG2 -> MMA warp | 256 | P in TMEM + O rescaled, ready for Value MMA |
| `bar_p_full_2` | Arrive | WG0/WG1 -> MMA warp | 128 | Last quarter of P written to TMEM |
| `bar_o_full` | Commit | MMA warp -> WG2 | 1 | Value MMA complete (last KV block), O in TMEM |
| `bar_softmax_corr_full` | Arrive | WG0/WG1 -> WG2 | 128 | acc_scale / row_sum written to SMEM |
| `bar_softmax_corr_empty` | Arrive | WG2 -> WG0/WG1 | 128 | WG2 done reading acc_scale from SMEM |
| `bar_corr_epi_full` | Arrive | WG2 -> TMA store | 128 | O normalized and in O_smem, ready for TMA store |
| `bar_corr_epi_empty` | Arrive | TMA store -> WG2 | 32 | TMA store complete, O_smem buffer free |

**Barrier types**: `Commit` barriers use `tcgen05.commit` (signals when MMA completes asynchronously). `ExpectTx` barriers track TMA byte counts. `Arrive` barriers use simple `mbarrier.arrive`.

### Barrier Flow Diagram

![Flash Attention 4 Barrier Flow](../img/flash_attention_barrier_flow.png)

### Comparison with GEMM barriers

In GEMM, the barrier system is simple: `full` and `empty` barriers between TMA producer and MMA consumer, with the epilogue being a straightforward TMEM->SMEM->TMA store. Flash Attention 4 adds:

- **bar_s_full / bar_p_full_o_rescaled**: An intermediate producer-consumer pair *within the MMA pipeline itself* (Score MMA produces S, softmax consumes it and produces P, Value MMA consumes P)
- **bar_softmax_corr_full / bar_softmax_corr_empty**: Coordination between softmax warpgroups and the correction warpgroup for O rescaling
- **bar_o_full**: Only fires on the last KV block iteration (unlike GEMM where every MMA signals completion)


## Writeback and Rescaling

After all KV blocks have been processed, the O accumulator in TMEM contains unnormalized values. The writeback process involves three steps:

### 1. Final normalization

WG0/WG1 write the final `row_sum` to SMEM. WG2 reads it and computes `norm_scale = 1.0 / row_sum` (with a guard for zero/NaN rows):

```python
# WG2 epilogue: normalize O
acc_O_is_zero_or_nan = (row_sum == 0.0) or (row_sum != row_sum)
norm_scale = rcp(1.0 if acc_O_is_zero_or_nan else row_sum)
```

### 2. TMEM -> SMEM transfer

WG2 reads O from TMEM in chunks of 16 columns, multiplies each element by `norm_scale`, casts from float32 to float16, and writes to O_smem:

```python
# WG2: Read O from TMEM, normalize, write to SMEM
for d_tile in range(HEAD_DIM // TMEM_EPI_LD_SIZE):
    o_row_f32 = tmem_load(tmem_col_o + d_tile * 16, size=16)
    o_row_f32 *= norm_scale           # Normalize
    o_row_f16 = cast_to_f16(o_row_f32)
    O_smem[i_q, tid_in_wg, d_tile*16 : d_tile*16+16] = o_row_f16
```

### 3. TMA store

WG3's TMA store warp (warp 2) waits on `bar_corr_epi_full`, then issues a TMA store from O_smem to global memory. This follows the same pattern as GEMM's epilogue.

### O rescaling during iteration

Between KV blocks, the O accumulator must be rescaled when the running maximum changes. This is done by **WG2** (the correction warpgroup), not by the softmax warpgroups. Here is why: WG0/WG1 are busy computing softmax and writing P to TMEM, while O is also in TMEM. Having a separate warpgroup read O, multiply by acc_scale, and write it back allows the rescaling to overlap with softmax computation.

```python
# WG2: Rescale O accumulator in TMEM (only when acc_scale < 1.0)
if any_thread_needs_rescale:
    for d_tile in range(HEAD_DIM // 16):
        o_row = tmem_load(tmem_col_o + d_tile * 16, size=16)
        o_row *= acc_scale
        tmem_store(tmem_col_o + d_tile * 16, o_row)
```

The kernel optimizes this by checking `any_needs_rescale` via `Tx.ptx.any_sync` — if no thread in the warp needs rescaling (because the maximum did not change), the entire TMEM read-modify-write is skipped.

**Adaptive rescaling threshold**: Unlike earlier Flash Attention versions that rescale O every time a new maximum appears, Flash Attention 4 only applies the rescaling when the maximum changes by more than `rescale_threshold` (set to 8.0 in the implementation). In practice, the running maximum stabilizes quickly after the first few KV blocks, so most iterations skip rescaling entirely. This reduces the number of expensive TMEM read-modify-write correction operations.

### LSE for backward pass

In a full training pipeline, the log-sum-exp $\text{LSE} = \log(\ell) + m$ would be stored as an auxiliary output for the backward pass. After the last KV block, $\text{LSE}_i = \log(\text{row\_sum}_i) + \text{row\_max}_i$ for each row $i$. **The current tutorial kernel does not output LSE** — it only computes the forward attention output O. A backward kernel would need an additional output buffer for LSE.


## Pipelining Structure

The kernel uses **three independent pipeline depths**:

- **Q pipeline** (depth 2): Q tiles change infrequently (only when moving to the next Q block). WG0 and WG1 each handle one Q stage.

- **KV pipeline** (depth 3): K and V tiles stream continuously as we iterate over the sequence length. Depth 3 provides enough buffering to hide TMA latency while MMA is in progress.

- **TMEM pipeline** (depth 2): Double-buffered TMEM for Score/P/O across the two Q stages.

![Flash Attention 4 Pipeline Structure](../img/flash_attention_pipeline.png)

Note that the KV blocks are iterated in **reverse order** (from `n_block_max - 1` down to 0). This is an optimization for causal attention where the last block is most likely to need masking.


## Causal Masking

For causal (autoregressive) attention, positions can only attend to earlier positions. The kernel implements this at two granularities:

### Block-level skipping

Entire KV blocks that are fully above the diagonal are skipped:

```python
def get_n_block_max(m_block_idx, causal):
    """Maximum KV block index (exclusive) for this Q block."""
    if not causal:
        return ceildiv(SEQ_LEN_KV, BLK_N)
    m_idx_max = (m_block_idx + 1) * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
    n_idx = m_idx_max + SEQ_LEN_KV - SEQ_LEN_Q
    return min(num_kv_blocks, ceildiv(n_idx, BLK_N))
```

This provides up to 2x speedup for causal attention — the first Q block processes only 1 KV block, the last processes all of them, averaging half.

### Three-phase iteration

KV blocks are processed in three phases:

1. **Phase 1**: The last KV block (`n_block_max - 1`), always with causal mask applied
2. **Phase 2**: Blocks that straddle the causal boundary (`n_block_min_causal` to `n_block_max - 2`), with per-element masking via R2P (Register-to-Predicate) bit manipulation
3. **Phase 3**: Blocks fully below the diagonal (`0` to `n_block_min_causal - 1`), no masking needed

### R2P masking

Rather than per-column comparisons, the kernel uses bit manipulation that compiles to the PTX `R2P` instruction:

```python
# Create bitmask: col_limit=5 -> 0b11111 (bits 0-4 set)
mask = (1 << col_limit) - 1
# Test bit i: compiles to R2P instruction
in_bound = mask & (1 << i)
s_chunk[c] = s_chunk[c] if in_bound else -inf
```


## GQA Support

Grouped Query Attention (GQA) shares K/V heads across multiple Q heads. The kernel handles this by packing multiple Q heads into the same BLK_M tile:

```python
GQA_RATIO = num_qo_heads // num_kv_heads  # e.g., 4
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO       # e.g., 32 sequence positions per tile
```

With `GQA_RATIO=4` and `BLK_M=128`, each tile holds 32 sequence positions x 4 Q heads = 128 rows. The SMEM row index `i` maps to `(seq_pos = i // GQA_RATIO, head = i % GQA_RATIO)`. TMA loads use a 3D view to pack the GQA heads:

```python
Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
Tx.copy_async(
    Q_smem_3d[i_q, :, :, :],
    Q[batch_idx, m_start : m_start + SEQ_Q_PER_TILE,
      kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO, :],
    **tma_copy_q,
)
```


## Tile Scheduling

The kernel supports two scheduling modes with different CTA launch strategies:

- **Non-causal (persistent)**: Launches `min(148, num_tasks)` CTAs that cycle through `(batch, kv_head, m_block)` tasks round-robin using a `LinearScheduler`. Each CTA picks up the next available task.
- **Causal (non-persistent)**: Launches one CTA per task (`cta_count = num_tasks`). Because causal attention gives later Q blocks fewer KV blocks to process, the workload is naturally unbalanced. An `LPTScheduler` (Longest Processing Time first) assigns heavier tasks first to balance load. An L2 swizzle factor optimizes cache reuse.

This distinction matters: causal mode does **not** use a persistent kernel, since the varying workload per tile makes persistent scheduling less effective.

```python
scheduler = (
    FlashAttentionLPTScheduler(...) if is_causal
    else FlashAttentionLinearScheduler(...)
)
while scheduler.valid():
    m_block_idx = scheduler.m_block_idx
    batch_idx = scheduler.batch_idx
    kv_head_idx = scheduler.head_idx
    # ... process tile ...
    scheduler.next_tile()
```


## Key Differences from GEMM

| Aspect | GEMM | Flash Attention 4 |
|--------|------|----------------|
| MMA operations | 1 type (A@B) | 2 types (Q@K^T, P@V) |
| Between-MMA work | None | Softmax (exp, max, sum, rescale) |
| State across tiles | Sum accumulation | (max, sum, O) triple |
| Masking | None | Causal block skipping + R2P masking |
| Thread count | 128-256 | 512 (4 warpgroups) |
| Pipeline structure | Uniform depth | Heterogeneous (Q depth 2, KV depth 3, TMEM depth 2) |
| Barrier count | 4 (full/empty x 2) | 12+ barriers with mixed types |
| Epilogue complexity | O = TMEM -> SMEM -> TMA | Rescale O, normalize by sum, then TMEM -> SMEM -> TMA |
| Warp roles | TMA producer, MMA consumer | TMA, Score MMA, Softmax, Value MMA, Correction, Writeback |


## Exercises

1. Why does the KV pipeline use depth 3 while the Q pipeline uses depth 2? What determines the optimal depth for each?
2. In online softmax, the rescaling $O = O \cdot e^{m_{\text{old}} - m_{\text{new}}}$ applies to the entire accumulated output. Why doesn't this cause precision loss? (Hint: what is the sign of the exponent?)
3. For causal attention with `seq_len=4096` and `BLK_N=128`, how many KV blocks does the first Q block process vs the last Q block? What is the average workload?
4. How does GQA_RATIO affect the effective tile size and MMA utilization? What happens when `GQA_RATIO > BLK_M`?
5. The Value MMA splits P into two parts (96 + 32 columns). Why? What would happen if it waited for all of P before starting?
6. Why is O rescaling done by a separate warpgroup (WG2) rather than by the softmax warpgroups (WG0/WG1) that computed the rescaling factor?

> **Try with your agent**: Describe your Flash Attention 4 kernel's pipeline structure (barrier types, warp roles, pipeline depths) and ask: *"Is there a deadlock scenario in this pipeline topology? Check that every wait has a matching arrive, and that no circular dependency exists between barrier groups."*


## Running the Kernel

The complete Flash Attention 4 kernel (~1000 lines) is in `tirx_tutorial/flash_attention4.py`. Below we compile it, verify correctness against PyTorch's scaled dot-product attention, and benchmark.

Note: The kernel signature has 5 arguments `(Q, K, V, O, profiler_buffer)` — the extra `profiler_buffer` is used for optional profiling instrumentation and must always be provided.

```python
import torch
import numpy as np
import tvm
from tirx_tutorial.flash_attention4 import (
    get_flash_attention4_kernel, prepare_data, PROFILER_BUFFER_SIZE
)

# Configuration
batch_size = 4
seq_len = 2048
num_qo_heads = 32
num_kv_heads = 8
head_dim = 128
is_causal = False

# Compile
kernel = get_flash_attention4_kernel(
    batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal
)
target = tvm.target.Target("cuda -arch=sm_100a")
with target:
    lib = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

# Prepare data (Q, K, V, O are CPU tensors)
Q, K, V, O = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
device = torch.device('cuda')  # gpu(0)
Q, K, V, O = Q.to(device), K.to(device), V.to(device), O.to(device)
profiler_buf = tvm.runtime.tensor(
    np.zeros(PROFILER_BUFFER_SIZE, dtype=np.uint64), tvm.cuda(0)
)

# Run — note the 5th argument (profiler_buffer)
f = lib["main"]
f(tvm.runtime.from_dlpack(Q), tvm.runtime.from_dlpack(K),
  tvm.runtime.from_dlpack(V), tvm.runtime.from_dlpack(O), profiler_buf)

# Verify against PyTorch SDPA
Q_ref = Q.transpose(1, 2).float()  # [batch, heads, seq, dim]
K_ref = K.transpose(1, 2).float()
K_ref = K_ref.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
V_ref = V.transpose(1, 2).float()
V_ref = V_ref.repeat_interleave(num_qo_heads // num_kv_heads, dim=1)
O_ref = torch.nn.functional.scaled_dot_product_attention(Q_ref, K_ref, V_ref, is_causal=is_causal)
O_ref = O_ref.transpose(1, 2).half()

max_err = float((O.cpu() - O_ref.cpu()).abs().max())
print(f"Flash Attention 4: batch={batch_size}, seq={seq_len}, heads={num_qo_heads}/{num_kv_heads}, dim={head_dim}")
print(f"Max error vs PyTorch SDPA: {max_err:.6f}")
assert max_err < 0.05, f"FAIL: max_err={max_err}"
print("PASS")

# Benchmark
args = [tvm.runtime.from_dlpack(Q), tvm.runtime.from_dlpack(K),
        tvm.runtime.from_dlpack(V), tvm.runtime.from_dlpack(O), profiler_buf]
for _ in range(10):
    f(*args)
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    f(*args)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 100
print(f"Performance: {ms:.3f} ms")
```
