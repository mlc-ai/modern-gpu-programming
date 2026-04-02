# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import math
from enum import Enum

import numpy as np
import torch

import tvm
import tvm.testing
from tvm.script import tirx as Tx
try:
    from tvm.tirx.bench import (  # fmt: skip
        CudaProfiler,
        ProtonContext,
        bench,
    )
except ImportError:
    CudaProfiler = None
    ProtonContext = None
    bench = None
from tvm.tirx.lang.pipeline import MBarrier, PipelineState, TCGen05Bar, TMABar
from tvm.tirx.lang.tile_scheduler import (  # fmt: skip
    FlashAttentionLinearScheduler,
    FlashAttentionLPTScheduler,
)
from tvm.tirx.lang.warp_role import WarpgroupRole, WarpRole
from tvm.tirx.layout import S, TCol, TileLayout, TLane
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg

M_CLUSTER = 1
N_CLUSTER = 1
SM_NUMBER = 148

NUM_GROUPS = 6
PROFILER_BUFFER_SIZE = int(2e6)
PROFILER_WRITE_STRIDE = SM_NUMBER * NUM_GROUPS
PROFILER_ON = False


class ProfileEventType(Enum):
    IssueTMA_Q = 0
    IssueTMA_K = 1
    IssueTMA_V = 2
    IssueMMA_QK = 3
    IssueMMA_PV = 4
    Softmax_MAX = 5
    Softmax_FMA = 6
    Softmax_EXP2 = 7
    Softmax_TMEM_ST = 8
    Softmax_SUM = 9
    Correction = 10
    EpiLDTMEM = 11
    TMAStore = 12


event_type_names = [
    "issue-tma-q",
    "issue-tma-k",
    "issue-tma-v",
    "issue-mma-qk",
    "issue-mma-pv",
    "softmax-max",
    "softmax-fma",
    "softmax-exp2",
    "softmax-tmem-st",
    "softmax-sum",
    "correction",
    "epi-ld-tmem",
    "tma-store",
]

WG_NUMBER = 4
WARP_NUMBER = 4
NUM_THREADS = (32 * WARP_NUMBER) * WG_NUMBER

N_COLS_TMEM = 512
TMEM_PIPE_DEPTH = 2
SMEM_PIPE_DEPTH_Q = 2
SMEM_PIPE_DEPTH_KV = 3

BLK_M = 128
BLK_N = 128
BLK_K = 64
SOFTMAX_LD_CHUNK = 32
SOFTMAX_ST_CHUNK = 32
EPI_TILE = 64
TMEM_EPI_LD_SIZE = 16
USE_S0_S1_BARRIER = False


MMA_M = 128
MMA_N = 128
MMA_K = 16

F16_BYTES = 2
F32_BYTES = 4
F128_BYTES = 16
a_type_qk = tvm.DataType("float16")
b_type_qk = tvm.DataType("float16")
d_type_qk = tvm.DataType("float32")
a_type_pv = tvm.DataType("float16")
b_type_pv = tvm.DataType("float16")
d_type_pv = tvm.DataType("float32")


# fmt: off
def get_flash_attention4_kernel(batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim, is_causal=False):

    BATCH_SIZE = batch_size
    SEQ_LEN_Q = seq_len_q
    SEQ_LEN_KV = seq_len_kv
    NUM_QO_HEADS = num_qo_heads
    NUM_KV_HEADS = num_kv_heads
    HEAD_DIM = head_dim

    # GQA parameters
    GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS  # e.g., 4 for num_qo_heads=32, num_kv_heads=8
    SEQ_Q_PER_TILE = BLK_M // GQA_RATIO       # e.g., 32 sequence positions per tile

    HEAD_DIM // MMA_K
    NUM_BLK_K = HEAD_DIM // BLK_K
    NUM_EPI_TILE = HEAD_DIM // EPI_TILE
    CTA_GROUP = 1

    # Block info for causal masking (following flash_attn/cute/block_info.py)
    def get_n_block_max(m_block_idx, causal):
        """Maximum KV block index (exclusive) for this Q block."""
        n_block_max = ceildiv(SEQ_LEN_KV, BLK_N)
        if not causal:
            return n_block_max
        # For causal: only process KV blocks up to diagonal
        # SEQ_Q_PER_TILE is already BLK_M // GQA_RATIO, so already in sequence coordinates
        m_idx_max = (m_block_idx + 1) * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
        n_idx = m_idx_max + SEQ_LEN_KV - SEQ_LEN_Q
        return Tx.min(n_block_max, ceildiv(n_idx, BLK_N))

    def get_n_block_min_causal_mask(m_block_idx):
        """KV block index where causal masking stops being needed.
        Blocks with index < this value don't need causal masking.
        """
        # SEQ_Q_PER_TILE is already in sequence coordinates (BLK_M // GQA_RATIO)
        m_idx_min = m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q
        n_idx = m_idx_min + SEQ_LEN_KV - SEQ_LEN_Q
        return Tx.max(0, n_idx // BLK_N)


    # L2 cache optimization for LPT scheduling (causal attention)
    L2_SIZE = 50 * 1024 * 1024  # 50MB L2 cache
    SIZE_ONE_KV_HEAD = SEQ_LEN_KV * HEAD_DIM * 2 * F16_BYTES  # K+V size per head
    L2_SWIZZLE = 1 if L2_SIZE < SIZE_ONE_KV_HEAD else (1 << int(math.log2(L2_SIZE // SIZE_ONE_KV_HEAD)))

    SSCALE_TOTAL_SIZE = 2 * SMEM_PIPE_DEPTH_Q * BLK_M
    assert TMEM_PIPE_DEPTH * MMA_N <= N_COLS_TMEM, "TMEM columns exceeded"

    def ceildiv(a, b):
        return (a + b - 1) // b

    def combine_int_frac_ex2(x_rounded, frac_ex2):
        func_name = "combine_int_frac_ex2"
        source_code = f"""
__device__ __forceinline__ float {func_name}(float x_rounded, float frac_ex2) {{
  float out;
  asm volatile(
    "{{\\n\\t"
    ".reg .s32 x_rounded_i, frac_ex_i, x_rounded_e, out_i;\\n\\t"
    "mov.b32 x_rounded_i, %1;\\n\\t"
    "mov.b32 frac_ex_i, %2;\\n\\t"
    "shl.b32 x_rounded_e, x_rounded_i, 23;\\n\\t"
    "add.s32 out_i, x_rounded_e, frac_ex_i;\\n\\t"
    "mov.b32 %0, out_i;\\n\\t"
    "}}\\n"
    : "=f"(out) : "f"(x_rounded), "f"(frac_ex2));
  return out;
}}
"""
        return Tx.cuda.func_call(
            func_name, x_rounded, frac_ex2, source_code=source_code, return_type="float32"
        )

    @Tx.inline
    def ex2_emulation_2(out, idx, x, y):
        # Polynomial coefficients for exp2 approximation (degree 3)
        poly_ex2_deg3 = Tx.meta_var(
            (
                1.0,
                0.695146143436431884765625,
                0.227564394474029541015625,
                0.077119089663028717041015625,
            )
        )
        fp32_round_int = Tx.meta_var(float(2**23 + 2**22))

        # Clamp inputs to avoid overflow (we assume x, y <= 127.0)
        xy_clamped: Tx.f32[2]
        xy_clamped[0] = Tx.max(x, -127.0)
        xy_clamped[1] = Tx.max(y, -127.0)

        # Round down to get integer part (stored as float with integer in lower bits)
        xy_rounded: Tx.f32[2]
        with Tx.thread():
            Tx.add(xy_rounded, xy_clamped, fp32_round_int, rounding_mode="rm")

        # Subtract to get the rounded-back value (round to nearest even)
        xy_rounded_back: Tx.f32[2]
        with Tx.thread():
            Tx.sub(xy_rounded_back, xy_rounded, fp32_round_int, rounding_mode="rn")

        # Compute fractional part: xy_frac = xy_clamped - xy_rounded_back
        xy_frac: Tx.f32[2]
        with Tx.thread():
            Tx.sub(xy_frac, xy_clamped, xy_rounded_back, rounding_mode="rn")

        # Horner's method: ((poly[3]*x + poly[2])*x + poly[1])*x + poly[0]
        xy_frac_ex2: Tx.f32[2]
        xy_frac_ex2[0] = poly_ex2_deg3[3]
        xy_frac_ex2[1] = poly_ex2_deg3[3]
        with Tx.thread():
            Tx.fma(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[2])
            Tx.fma(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[1])
            Tx.fma(xy_frac_ex2, xy_frac_ex2, xy_frac, poly_ex2_deg3[0])

        # Combine integer and fractional parts: shift integer left by 23 bits and add to fractional exp2
        out[idx] = combine_int_frac_ex2(xy_rounded[0], xy_frac_ex2[0])
        out[idx + 1] = combine_int_frac_ex2(xy_rounded[1], xy_frac_ex2[1])

    @Tx.meta_class
    class SmemDescriptor:
        def __init__(self, prefix: str):
            self.desc = Tx.local_scalar("uint64", name=prefix + "sdesc")

        @Tx.inline
        def init(self, smem_ptr, ldo, sdo, swizzle):
            Tx.ptx.tcgen05.encode_matrix_descriptor(
                Tx.address_of(self.desc), smem_ptr, ldo, sdo, swizzle
            )
            # Make the lo part (containing start_address) warp-uniform in-place
            self._make_lo_uniform()

        def _make_lo_uniform(self):
            """Shuffle the lower 32 bits of the descriptor to ensure warp-uniformity."""
            func_name = "smem_desc_make_lo_uniform"
            source_code = f"""
__forceinline__ __device__ void {func_name}(uint64_t* desc) {{
    SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);
    d->lo = __shfl_sync(0xffffffff, d->lo, 0);
}}
"""
            return Tx.cuda.func_call(
                func_name, Tx.address_of(self.desc),
                source_code=source_code, return_type="void"
            )

        def add_16B_offset(self, offset):
            """Add 16B-aligned offset to lower 32 bits only."""
            func_name = "tvm_builtin_smem_desc_add_16B_offset"
            source_code = f"""
__forceinline__ __device__ uint64_t {func_name}(uint64_t desc_base, int32_t offset) {{
    SmemDescriptor desc;
    desc.desc_ = desc_base;
    desc.lo += static_cast<uint32_t>(offset);
    return desc.desc_;
}}
"""
            return Tx.cuda.func_call(
                func_name, self.desc, offset, source_code=source_code, return_type="uint64"
            )

    Q_layout = Tx.ComposeLayout(Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True), Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH_Q, BLK_M, NUM_BLK_K, BLK_K) : (BLK_M * HEAD_DIM, BLK_K, BLK_M * BLK_K, 1)]))
    K_layout = Tx.ComposeLayout(Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True), Tx.TileLayout(Tx.S[(SMEM_PIPE_DEPTH_KV, BLK_N, NUM_BLK_K, BLK_K) : (BLK_N * HEAD_DIM, BLK_K, BLK_N * BLK_K, 1)]))
    O_layout = Tx.ComposeLayout(Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True), Tx.TileLayout(Tx.S[(TMEM_PIPE_DEPTH, BLK_M, NUM_EPI_TILE, EPI_TILE) : (BLK_M * HEAD_DIM, EPI_TILE, BLK_M * EPI_TILE, 1)]))

    @Tx.prim_func(tirx=True, persistent=True)
    def flash_attention4(
        Q: Tx.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),
        K: Tx.Buffer((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"),
        V: Tx.Buffer((BATCH_SIZE, SEQ_LEN_KV, NUM_KV_HEADS, HEAD_DIM), "float16"),
        O: Tx.Buffer((BATCH_SIZE, SEQ_LEN_Q, NUM_QO_HEADS, HEAD_DIM), "float16"),  # noqa: E741
        profiler_buffer: Tx.Buffer((PROFILER_BUFFER_SIZE,), "uint64"),
    ):
        # For GQA: each tile processes SEQ_Q_PER_TILE seq positions (not BLK_M)
        num_q_blocks_total = Tx.meta_var(ceildiv(SEQ_LEN_Q, SEQ_Q_PER_TILE))
        num_q_blocks_per_cta = Tx.meta_var(SMEM_PIPE_DEPTH_Q)
        num_q_blocks = Tx.meta_var(ceildiv(num_q_blocks_total, num_q_blocks_per_cta))

        # Task scheduling
        num_total_tasks = Tx.meta_var(BATCH_SIZE * NUM_KV_HEADS * num_q_blocks)

        # use non-persistent kernel for causal attention
        max_ctas: Tx.let = 148
        cta_count: Tx.let = Tx.min(max_ctas, num_total_tasks) if not is_causal else num_total_tasks

        with Tx.kernel():
            bx = Tx.cta_id([cta_count], parent="kernel")
            wg_id = Tx.warpgroup_id([4], parent="cta")
            warp_id = Tx.warp_id([4], parent="warpgroup")
            lane_id = Tx.thread_id([32], parent="warp")
            tid_in_wg = Tx.thread_id([128], parent="warpgroup")
            with Tx.cta():
                pool = Tx.PoolAllocator()
                # Allocate Q buffer with alignment
                Q_smem = pool.alloc((SMEM_PIPE_DEPTH_Q, BLK_M, HEAD_DIM), "float16", layout=Q_layout, align=1024)
                # Allocate K and V buffers (they share the same offset)
                K_smem = pool.alloc((SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM), "float16", layout=K_layout, align=1024)
                V_smem = K_smem.view(SMEM_PIPE_DEPTH_KV, BLK_N, HEAD_DIM)
                # Allocate O buffer
                O_smem = pool.alloc((TMEM_PIPE_DEPTH, BLK_M, HEAD_DIM), "float16", layout=O_layout, align=1024)
                # Allocate sScale buffer (ACC_SCALE/ROW_SUM shared + ROW_MAX)
                sScale = pool.alloc((SSCALE_TOTAL_SIZE,), "float32", align=1024)
                tmem_addr = pool.alloc([1], "uint32")

                ACC_SCALE_BASE: Tx.let = 0
                ROW_SUM_BASE: Tx.let = 0  # Shares with ACC_SCALE


                # Phase/stage scalars
                kv_pipe = PipelineState("kv", SMEM_PIPE_DEPTH_KV)
                phase_q: Tx.int32
                phase_s_full: Tx.int32
                phase_tmem: Tx.int32
                phase_s0_s1: Tx.int32
                phase_q_load: Tx.int32

                bar_load_q_full = TMABar(pool, SMEM_PIPE_DEPTH_Q)
                bar_load_q_empty = TCGen05Bar(pool, SMEM_PIPE_DEPTH_Q, phase_offset=1)

                bar_load_kv_full = TMABar(pool, SMEM_PIPE_DEPTH_KV)
                bar_load_kv_empty = TCGen05Bar(pool, SMEM_PIPE_DEPTH_KV, phase_offset=1)

                bar_p_full_o_rescaled = MBarrier(pool, 2)

                bar_s_full = TCGen05Bar(pool, 2)

                bar_o_full = TCGen05Bar(pool, 2)

                bar_softmax_corr_full = MBarrier(pool, 2)
                bar_softmax_corr_empty = MBarrier(pool, 2, phase_offset=1)

                bar_corr_epi_full = MBarrier(pool, TMEM_PIPE_DEPTH)
                bar_corr_epi_empty = MBarrier(pool, TMEM_PIPE_DEPTH, phase_offset=1)
                bar_p_full_2 = MBarrier(pool, 2)

                bar_s0_s1_sequence = MBarrier(pool, 8)

                bar_tmem_dealloc = MBarrier(pool, 1)
                pool.commit()

                profiler = CudaProfiler(profiler_buffer, write_stride=PROFILER_WRITE_STRIDE, num_groups=NUM_GROUPS, profiler_enabled=PROFILER_ON)

                if wg_id == 0 and warp_id == 0:
                    Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr[0]), n_cols=N_COLS_TMEM, cta_group=CTA_GROUP)
                    Tx.cuda.trap_when_assert_failed(tmem_addr[0] == Tx.uint32(0))

                tmem = Tx.decl_buffer((128, N_COLS_TMEM), "float32", scope="tmem", allocated_addr=0, layout=TileLayout(S[(128, N_COLS_TMEM) : (1 @ TLane, 1 @ TCol)]))
                tmem_as_f16 = Tx.decl_buffer((128, N_COLS_TMEM * 2), "float16", scope="tmem", allocated_addr=0, layout=TileLayout(S[(128, N_COLS_TMEM * 2) : (1 @ TLane, 1 @ TCol)]))

                # Create appropriate scheduler based on causal mode
                scheduler = (
                    FlashAttentionLPTScheduler(
                        "fa_scheduler",
                        num_batches=BATCH_SIZE,
                        num_heads=NUM_KV_HEADS,
                        num_m_blocks=num_q_blocks,
                        l2_swizzle=L2_SWIZZLE,
                    ) if is_causal else FlashAttentionLinearScheduler(
                        "fa_scheduler",
                        num_batches=BATCH_SIZE,
                        num_heads=NUM_KV_HEADS,
                        num_m_blocks=num_q_blocks,
                        num_ctas=cta_count,
                    )
                )

                scheduler.init(bx)  # Initialize with CTA ID

                if wg_id == 3 and warp_id == 1:
                    profiler.init(0)
                elif wg_id == 3 and warp_id == 2:
                    profiler.init(1)
                elif wg_id == 3 and warp_id == 0:
                    profiler.init(2)
                elif wg_id <= 1:
                    profiler.init(3 + wg_id)
                elif wg_id == 2:
                    profiler.init(5)

                kv_pipe.init(is_producer=False)
                phase_q = 0
                phase_tmem = 0
                phase_s_full = 0
                if USE_S0_S1_BARRIER:
                    phase_s0_s1 = Tx.if_then_else(wg_id == 1, 0, 1)
                phase_q_load = 0

                bar_load_q_full.init(1)
                bar_load_q_empty.init(1)
                bar_load_kv_full.init(1)
                bar_load_kv_empty.init(1)
                bar_p_full_o_rescaled.init(256)
                bar_p_full_2.init(128)
                bar_s_full.init(1)
                bar_o_full.init(1)
                bar_softmax_corr_full.init(128)
                bar_softmax_corr_empty.init(128)
                bar_corr_epi_full.init(128)
                bar_corr_epi_empty.init(32)
                bar_s0_s1_sequence.init(32)
                bar_tmem_dealloc.init(1)

                Tx.ptx.fence.proxy_async("shared::cta")
                Tx.ptx.fence.mbarrier_init()
                Tx.cuda.cta_sync()
                if wg_id == 2:
                    for i_q in Tx.unroll(2):
                        bar_p_full_o_rescaled.arrive(i_q)

                num_kv_blocks: Tx.let = ceildiv(SEQ_LEN_KV, BLK_N)
                tmem_s_base: Tx.let = 0
                tmem_o_base: Tx.let = 256
                tmem_p_base: Tx.let = 64
                tmem_offset: Tx.let = 128

                while scheduler.valid():
                    # Extract indices from scheduler
                    m_block_idx = Tx.meta_var(scheduler.m_block_idx)
                    batch_idx = Tx.meta_var(scheduler.batch_idx)
                    kv_head_idx = Tx.meta_var(scheduler.head_idx)
                    # m_start refers to SEQ_Q positions (not BLK_M rows)
                    m_start = Tx.meta_var(m_block_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q)
                    with Tx.cta():
                        # Tx.attr({"tirx.scope_partition": True})

                        if wg_id == 3:
                            Tx.ptx.setmaxnreg(False, 48)
                            with WarpRole(warp_id, 1):

                                    @Tx.inline
                                    def load_q(i_q):
                                        # Use phase_q_load for Q prefetch barrier synchronization
                                        bar_load_q_empty.wait(i_q, phase_q_load)
                                        # stage_q[0] ->  0 -> 1 -> 0 -> 1 -> ...

                                        tma_copy_q = Tx.meta_var({"dispatch": "tma", "mbar": bar_load_q_full.buf.ptr_to([i_q]), "cta_group": CTA_GROUP})
                                        # GQA: Load each qo_head with 2D TMA copy
                                        # SMEM layout: row i corresponds to (seq = i // GQA_RATIO, head = i % GQA_RATIO)
                                        profiler.start(ProfileEventType.IssueTMA_Q, lane_id == 0)
                                        Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
                                        with Tx.elected():
                                            Tx.copy_async(
                                                Q_smem_3d[i_q, :, :, :], Q[batch_idx, m_start + i_q * SEQ_Q_PER_TILE : m_start + (i_q + 1) * SEQ_Q_PER_TILE, kv_head_idx * GQA_RATIO: (kv_head_idx + 1) * GQA_RATIO, :],
                                                **tma_copy_q,
                                            )
                                            bar_load_q_full.arrive(i_q, CTA_GROUP * BLK_M * HEAD_DIM * F16_BYTES)  # ar(0,x)
                                        profiler.end(ProfileEventType.IssueTMA_Q, lane_id == 0)

                                    @Tx.inline
                                    def load_k(i_kv):
                                        bar_load_kv_empty.wait(kv_pipe.stage, kv_pipe.phase)
                                        tma_copy_k = Tx.meta_var({"dispatch": "tma", "mbar": bar_load_kv_full.buf.ptr_to([kv_pipe.stage]), "cta_group": CTA_GROUP})
                                        profiler.start(ProfileEventType.IssueTMA_K, lane_id == 0)
                                        with Tx.elected():
                                            Tx.copy_async(K_smem[kv_pipe.stage, :, :], K[batch_idx, i_kv * BLK_N : (i_kv + 1) * BLK_N, kv_head_idx, :],
                                                **tma_copy_k,
                                            )
                                            bar_load_kv_full.arrive(kv_pipe.stage, CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES)
                                        profiler.end(ProfileEventType.IssueTMA_K, lane_id == 0)
                                        kv_pipe.move_to_next_stage()

                                    @Tx.inline
                                    def load_v(i_kv):
                                        bar_load_kv_empty.wait(kv_pipe.stage, kv_pipe.phase)
                                        tma_copy_v = Tx.meta_var({"dispatch": "tma", "mbar": bar_load_kv_full.buf.ptr_to([kv_pipe.stage]), "cta_group": CTA_GROUP})
                                        profiler.start(ProfileEventType.IssueTMA_V, lane_id == 0)
                                        with Tx.elected():
                                            Tx.copy_async(
                                                V_smem[kv_pipe.stage, :, :],
                                                V[batch_idx, i_kv * BLK_N : (i_kv + 1) * BLK_N, kv_head_idx, :],
                                                **tma_copy_v,
                                            )
                                            bar_load_kv_full.arrive(kv_pipe.stage, CTA_GROUP * BLK_N * HEAD_DIM * F16_BYTES)
                                        profiler.end(ProfileEventType.IssueTMA_V, lane_id == 0)
                                        kv_pipe.move_to_next_stage()

                                    # For causal, compute reduced trip count for loads
                                    load_trip_count: Tx.int32
                                    load_trip_count = get_n_block_max(m_block_idx, is_causal) if is_causal else num_kv_blocks

                                    load_q(0)
                                    load_k(load_trip_count - 1)
                                    load_q(1)
                                    # Flip phase_q_load after Q stages complete (for persistent kernel)
                                    phase_q_load ^= 1
                                    load_v(load_trip_count - 1)
                                    for _i in Tx.serial(load_trip_count - 1, unroll=False):
                                        i_kv: Tx.let = load_trip_count - 2 - _i
                                        load_k(i_kv)
                                        load_v(i_kv)

                            with WarpRole(warp_id, 2):
                                    for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):  # stage=0,1
                                        bar_corr_epi_full.wait(i_q, phase_tmem)
                                        if i_q == 0:
                                            profiler.start(ProfileEventType.TMAStore, lane_id == 0)
                                        # GQA: m_start_global refers to SEQ_Q positions
                                        m_start_global = Tx.meta_var(m_start + i_q * SEQ_Q_PER_TILE)
                                        # TMA O store: Store each qo_head with 2D TMA copy
                                        # SMEM layout: row i corresponds to (seq = i // GQA_RATIO, head = i % GQA_RATIO)
                                        O_smem_3d = O_smem.view(TMEM_PIPE_DEPTH, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
                                        with Tx.elected():
                                            Tx.copy_async(
                                                O[batch_idx, m_start_global : m_start_global + SEQ_Q_PER_TILE, kv_head_idx * GQA_RATIO: (kv_head_idx + 1) * GQA_RATIO, :],
                                                O_smem_3d[i_q, :, :, :],
                                                dispatch="tma",
                                            )
                                        Tx.ptx.cp_async.bulk.commit_group()
                                    for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):
                                        Tx.ptx.cp_async.bulk.wait_group(1 - i_q)
                                        bar_corr_epi_empty.arrive(i_q)
                                    profiler.end(ProfileEventType.TMAStore, lane_id == 0)
                                    phase_tmem ^= 1

                            with WarpRole(warp_id, 0):
                                    acc: Tx.int32
                                    acc = 0

                                    @Tx.inline
                                    def gemm_qk(q_stage, kv_stage, tmem_col_s, bar_s_full):
                                        with Tx.warp():
                                            Tx.gemm_async(
                                                tmem[0:128, tmem_col_s : tmem_col_s + MMA_N],
                                                Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
                                                K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
                                                dispatch="tcgen05",
                                                cta_group=CTA_GROUP,
                                            )
                                        if Tx.ptx.elect_sync():
                                            bar_s_full.arrive(q_stage)

                                    @Tx.inline
                                    def gemm_pv(i_q, kv_stage, tmem_col_o, tmem_col_p, should_accumulate, bar_p_full_2):
                                        # TODO: gemm_async causes more spills
                                        K_SPLIT = Tx.meta_var(6 * MMA_K)  # 96 — first 6 MMA iterations
                                        # First part: k=0..5 (P cols 0..95, V rows 0..95)
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
                                        # Wait for last 1/4 of P
                                        bar_p_full_2.wait(i_q, phase_tmem)
                                        # Second part: k=6..7 (P cols 96..127, V rows 96..127)
                                        with Tx.warp():
                                            Tx.gemm_async(
                                                tmem[0:128, tmem_col_o : tmem_col_o + MMA_N],
                                                tmem_as_f16[0:128, tmem_col_p * 2 + K_SPLIT : tmem_col_p * 2 + BLK_N],
                                                V_smem[kv_stage, K_SPLIT:BLK_N, 0:HEAD_DIM],
                                                transB=True,
                                                accum=True,
                                                dispatch="tcgen05",
                                                cta_group=CTA_GROUP,
                                            )

                                    for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):
                                        tmem_col_s: Tx.let = tmem_s_base + i_q * tmem_offset
                                        bar_load_q_full.wait(i_q, phase_q_load)
                                        if i_q == 0:
                                            # for 2 q, confirm k is loaded
                                            bar_load_kv_full.wait(kv_pipe.stage, kv_pipe.phase)
                                        gemm_qk(i_q, kv_pipe.stage, tmem_col_s, bar_s_full)
                                        if i_q == 1:
                                            # finish twice qk mma
                                            if Tx.ptx.elect_sync():
                                                bar_load_kv_empty.arrive(kv_pipe.stage)
                                    kv_pipe.move_to_next_stage()

                                    # For causal, compute reduced trip count
                                    mma_trip_count: Tx.int32
                                    mma_trip_count = get_n_block_max(m_block_idx, is_causal) if is_causal else num_kv_blocks

                                    for i_kv in Tx.serial(
                                        mma_trip_count - 1, unroll=False
                                    ):
                                        stage_v: Tx.let = kv_pipe.stage
                                        phase_v: Tx.let = kv_pipe.phase
                                        kv_pipe.move_to_next_stage()
                                        stage_k = Tx.meta_var(kv_pipe.stage)
                                        phase_k = Tx.meta_var(kv_pipe.phase)

                                        for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):
                                            tmem_col_s: Tx.let = tmem_s_base + i_q * tmem_offset
                                            tmem_col_p: Tx.let = tmem_p_base + i_q * tmem_offset
                                            tmem_col_o: Tx.let = tmem_o_base + i_q * tmem_offset
                                            if i_q == 0:
                                                # wait for v is loaded
                                                bar_load_kv_full.wait(stage_v, phase_v)
                                            # wait for o_full to be ready
                                            bar_p_full_o_rescaled.wait(i_q, phase_tmem)
                                            gemm_pv(i_q, stage_v, tmem_col_o, tmem_col_p, acc, bar_p_full_2)
                                            if i_q == 1:
                                                # finish twice pv mma
                                                if Tx.ptx.elect_sync():
                                                    bar_load_kv_empty.arrive(stage_v)
                                            if i_q == 0:
                                                # for 2 q, confirm k is loaded
                                                bar_load_kv_full.wait(stage_k, phase_k)
                                            gemm_qk(i_q, stage_k, tmem_col_s, bar_s_full)
                                            if i_q == 1:
                                                # finish twice qk mma
                                                if Tx.ptx.elect_sync():
                                                    bar_load_kv_empty.arrive(stage_k)
                                        acc = 1
                                        kv_pipe.move_to_next_stage()
                                        phase_tmem ^= 1

                                    for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):
                                        tmem_col_p: Tx.let = tmem_p_base + i_q * tmem_offset
                                        tmem_col_o: Tx.let = tmem_o_base + i_q * tmem_offset
                                        if i_q == 0:
                                            # wait for v is loaded
                                            bar_load_kv_full.wait(kv_pipe.stage, kv_pipe.phase)
                                        # wait for o_full to be ready
                                        bar_p_full_o_rescaled.wait(i_q, phase_tmem)
                                        gemm_pv(i_q, kv_pipe.stage, tmem_col_o, tmem_col_p, acc, bar_p_full_2)
                                        if i_q == 1:
                                            # finish twice pv mma
                                            if Tx.ptx.elect_sync():
                                                bar_load_kv_empty.arrive(kv_pipe.stage)
                                        if Tx.ptx.elect_sync():
                                            bar_o_full.arrive(i_q)
                                    kv_pipe.move_to_next_stage()
                                    phase_tmem ^= 1

                                    for i_q in Tx.unroll(SMEM_PIPE_DEPTH_Q):
                                        if Tx.ptx.elect_sync():
                                            bar_load_q_empty.arrive(i_q)

                                    # Flip phase_q_load after Q stages complete (for persistent kernel)
                                    phase_q_load ^= 1

                        elif wg_id < 2:
                            with Tx.warpgroup():
                                # here phase_q and stage_q represent phase_tmem and stage_tmem

                                Tx.ptx.setmaxnreg(True, 200)

                                scale_log2 = Tx.meta_var(math.log2(math.e) / math.sqrt(HEAD_DIM))
                                rescale_threshold = Tx.meta_var(8.0)

                                row_max: Tx.f32[1]
                                row_sum: Tx.f32[1]

                                @Tx.inline
                                def mask_r2p(s_chunk_buf, col_limit, ncol: Tx.int32):
                                    """Apply mask using R2P-style bit manipulation.

                                    Optimizes: for j in range(N): buf[j] = -inf if j >= col_limit else buf[j]
                                    Into: bitmask operations that compile to R2P PTX instruction.

                                    Following flash_attn/cute/mask.py mask_r2p() lines 13-40:
                                    Process in 24-element chunks because shift by 31+ bits is problematic.
                                    For ncol=128: chunks 0-4 have 24 elements, chunk 5 has 8 elements.

                                    The bit test `mask & (1 << i)` compiles to the R2P (Register to Predicate)
                                    PTX instruction, which is more efficient than per-column comparisons.
                                    """
                                    ncol = Tx.meta_var(ncol)
                                    CHUNK_SIZE: Tx.let = 24  # Max safe shift amount (< 32)
                                    num_chunks: Tx.let = ceildiv(ncol, CHUNK_SIZE)

                                    for s in Tx.unroll(num_chunks):
                                        # Compute col_limit for this chunk (clamped to [0, chunk_cols])
                                        col_limit_s: Tx.let = Tx.max(col_limit - s * CHUNK_SIZE, 0)
                                        mask: Tx.uint32
                                        # Create bitmask: col_limit=5 -> 0b11111 (bits 0-4 set)
                                        mask = Tx.shift_left(Tx.int32(1), col_limit_s) - 1

                                        # Apply mask to each column in this chunk
                                        for i in Tx.unroll(CHUNK_SIZE):
                                            if i < ncol - s * CHUNK_SIZE:
                                                c: Tx.let = s * CHUNK_SIZE + i
                                                in_bound: Tx.let = Tx.bitwise_and(mask, Tx.shift_left(Tx.int32(1), i))
                                                s_chunk_buf[c] = Tx.Select(Tx.cast(in_bound, "bool"), s_chunk_buf[c], Tx.float32(-float("inf")))

                                @Tx.inline
                                def apply_causal_mask(s_chunk_buf, m_blk_idx, n_blk_idx):
                                    """Apply causal mask to attention scores.

                                    Following flash_attn/cute/mask.py apply_mask_sm100() lines 384-400:
                                    causal_row_offset = 1 + seqlen_k - n_block * tile_n - seqlen_q
                                    row_idx = thread_row + m_block * tile_m
                                    col_limit_right = row_idx + causal_row_offset
                                    Mask if col >= col_limit_right

                                    Coordinate Mapping:
                                    - BLK_M = 128 packed rows per tile
                                    - SEQ_Q_PER_TILE = BLK_M // GQA_RATIO (e.g., 32 for GQA_RATIO=4)
                                    - Each warpgroup handles one Q stage with SEQ_Q_PER_TILE sequence positions
                                    - tid_in_wg (0-127) maps to packed rows: (seq_pos, head) = (tid//GQA_RATIO, tid%GQA_RATIO)
                                    """
                                    # Convert thread index to sequence position within warpgroup
                                    seq_pos_in_wg: Tx.let = tid_in_wg // GQA_RATIO

                                    # Global sequence position
                                    # wg_id 0/1 handles different Q stages (each stage has SEQ_Q_PER_TILE positions)
                                    # m_block covers SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q sequence positions
                                    row_idx: Tx.let = (m_blk_idx * SEQ_Q_PER_TILE * SMEM_PIPE_DEPTH_Q +
                                               wg_id * SEQ_Q_PER_TILE +
                                               seq_pos_in_wg)

                                    # Causal row offset (from mask.py:385)
                                    # For seq_len_q == seq_len_kv: causal_row_offset = 1 - n_block * BLK_N
                                    causal_row_offset: Tx.let = 1 + SEQ_LEN_KV - n_blk_idx * BLK_N - SEQ_LEN_Q

                                    # Column limit: mask if col >= col_limit_right
                                    col_limit_right: Tx.let = row_idx + causal_row_offset

                                    # Use R2P-style masking instead of per-column comparison
                                    mask_r2p(s_chunk_buf, col_limit_right, BLK_N)

                                @Tx.inline
                                def softmax_step(i_kv, apply_mask=False, is_first=False):
                                    s_chunk_buf: Tx.f32[BLK_N]
                                    s_chunk = s_chunk_buf.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1 @ axis_tid_in_wg, 1)]))

                                    p_chunk_buf_f32: Tx.f32[BLK_N // 2]
                                    p_chunk_buf = Tx.decl_buffer((BLK_N,), dtype="float16", data=p_chunk_buf_f32.data)
                                    p_chunk = p_chunk_buf.view(128, BLK_N, layout=TileLayout(S[(128, BLK_N) : (1 @ axis_tid_in_wg, 1)]))

                                    tmem_col_s = Tx.meta_var(tmem_s_base + wg_id * tmem_offset)
                                    tmem_col_p = Tx.meta_var(tmem_p_base + wg_id * tmem_offset)

                                    bar_s_full.wait(wg_id, phase_s_full)  # noqa: F823
                                    profiler.start(ProfileEventType.Softmax_MAX, tid_in_wg == 0)
                                    tile_max: Tx.f32[1]
                                    for chunk_idx in Tx.unroll(BLK_N // SOFTMAX_LD_CHUNK):
                                        Tx.copy_async(s_chunk[:, chunk_idx * SOFTMAX_LD_CHUNK : (chunk_idx + 1) * SOFTMAX_LD_CHUNK], tmem[:, tmem_col_s + chunk_idx * SOFTMAX_LD_CHUNK : tmem_col_s + chunk_idx * SOFTMAX_LD_CHUNK + SOFTMAX_LD_CHUNK])

                                    # Apply causal mask if needed
                                    if apply_mask:
                                        apply_causal_mask(s_chunk_buf, m_block_idx, i_kv)

                                    row_max_old: Tx.f32
                                    row_max_old = row_max[0]
                                    with Tx.thread():
                                        if is_first:
                                            Tx.max(tile_max, s_chunk_buf)
                                        else:
                                            tile_max[0] = row_max_old
                                            Tx.max(tile_max, s_chunk_buf, accum=True)
                                    row_max_new: Tx.f32
                                    acc_scale: Tx.f32
                                    acc_scale_: Tx.f32  # For slack check
                                    row_max_safe: Tx.f32
                                    row_max_new = tile_max[0]
                                    row_max_safe = Tx.if_then_else(tile_max[0] == -float("inf"), 0.0, tile_max[0])

                                    if is_first:
                                        acc_scale = Tx.float32(1.0)
                                    else:
                                        acc_scale_ = (row_max_old - row_max_safe) * scale_log2

                                        # if the difference is too small, don't rescale
                                        if acc_scale_ >= -rescale_threshold:
                                            row_max_new = row_max_old
                                            row_max_safe = row_max_old
                                            acc_scale = Tx.float32(1.0)
                                        else:
                                            acc_scale = Tx.ptx.exp2(acc_scale_)

                                    # row_max is the max value of the tile
                                    # and row_max_scaled is the max value of the tile after scaled
                                    # scale_log2 is the log2 of the scale factor
                                    row_max[0] = row_max_new
                                    row_max_scaled: Tx.let = row_max_safe * scale_log2
                                    profiler.end(ProfileEventType.Softmax_MAX, tid_in_wg == 0)

                                    # Write acc_scale to sScale and arrive immediately (no wait here)
                                    if tid_in_wg < BLK_M and not is_first:
                                        sScale_idx: Tx.let = ACC_SCALE_BASE + tid_in_wg + wg_id * BLK_M
                                        sScale[sScale_idx] = acc_scale
                                    bar_softmax_corr_full.arrive(wg_id)
                                    profiler.start(ProfileEventType.Softmax_FMA, tid_in_wg == 0)
                                    with Tx.thread():
                                        Tx.fma(s_chunk_buf, s_chunk_buf, scale_log2, -row_max_scaled)
                                    profiler.end(ProfileEventType.Softmax_FMA, tid_in_wg == 0)
                                    if USE_S0_S1_BARRIER:
                                        bar_s0_s1_sequence.wait(wg_id * 4 + warp_id, phase_s0_s1)  # noqa: F823
                                    profiler.start(ProfileEventType.Softmax_EXP2, tid_in_wg == 0)
                                    for frag_idx in Tx.unroll(4):
                                        for i in Tx.unroll(BLK_N // 4 // 2):
                                            idx = Tx.meta_var(frag_idx * BLK_N // 4 + 2 * i)
                                            if i * 2 % 16 < 16 - 4 or frag_idx >= 4 - 1 or apply_mask:
                                                s_chunk_buf[idx] = Tx.ptx.exp2(s_chunk_buf[idx])
                                                s_chunk_buf[idx + 1] = Tx.ptx.exp2(s_chunk_buf[idx + 1])
                                            else:
                                                ex2_emulation_2(s_chunk_buf, idx, s_chunk_buf[idx], s_chunk_buf[idx + 1])
                                        with Tx.thread():
                                            Tx.cast(p_chunk_buf[frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4], s_chunk_buf[frag_idx * BLK_N // 4 : (frag_idx + 1) * BLK_N // 4])
                                    if USE_S0_S1_BARRIER:
                                        bar_s0_s1_sequence.arrive((1 - wg_id) * 4 + warp_id)
                                    profiler.end(ProfileEventType.Softmax_EXP2, tid_in_wg == 0)
                                    profiler.start(ProfileEventType.Softmax_TMEM_ST, tid_in_wg == 0)
                                    for i in Tx.unroll(3):
                                        Tx.copy_async(tmem_as_f16[:, tmem_col_p * 2 + i * BLK_N // 4 : tmem_col_p * 2 + (i + 1) * BLK_N // 4], p_chunk[:, i * BLK_N // 4 : (i + 1) * BLK_N // 4])
                                    Tx.ptx.tcgen05.wait.st()
                                    bar_p_full_o_rescaled.arrive(wg_id)
                                    Tx.copy_async(tmem_as_f16[:, tmem_col_p * 2 + 3 * BLK_N // 4 : tmem_col_p * 2 + BLK_N], p_chunk[:, 3 * BLK_N // 4 : BLK_N])
                                    Tx.ptx.tcgen05.wait.st()
                                    bar_p_full_2.arrive(wg_id)

                                    profiler.end(ProfileEventType.Softmax_TMEM_ST, tid_in_wg == 0)

                                    # Wait for correction warp to finish reading previous acc_scale
                                    bar_softmax_corr_empty.wait(wg_id, phase_q)  # noqa: F823

                                    profiler.start(ProfileEventType.Softmax_SUM, tid_in_wg == 0)
                                    phase_s_full ^= 1
                                    phase_q ^= 1
                                    with Tx.thread():
                                        if is_first:
                                            Tx.sum(row_sum, s_chunk_buf)
                                        else:
                                            row_sum[0] = row_sum[0] * acc_scale
                                            Tx.sum(row_sum, s_chunk_buf, accum=True)
                                    profiler.end(ProfileEventType.Softmax_SUM, tid_in_wg == 0)
                                    if USE_S0_S1_BARRIER:
                                        phase_s0_s1 ^= 1

                                bar_softmax_corr_empty.wait(wg_id, phase_q)
                                phase_q ^= 1
                                # Compute block ranges for this Q block
                                n_block_max: Tx.let = get_n_block_max(m_block_idx, is_causal)
                                n_block_min_causal: Tx.let = get_n_block_min_causal_mask(m_block_idx) if is_causal else n_block_max

                                # Phase 1: Last KV block (n_block_max - 1) with causal mask
                                # This block may have both seqlen boundary AND causal masking
                                softmax_step(n_block_max - 1, apply_mask=is_causal, is_first=True)

                                # Update n_block_max after Phase 1
                                n_block_max_after_p1: Tx.let = n_block_max - 1

                                # Phase 2: Blocks with partial causal masking
                                # These are blocks in [n_block_min_causal, n_block_max - 1)
                                num_phase2_blocks: Tx.let = Tx.max(n_block_max_after_p1 - n_block_min_causal, 0)
                                for i in Tx.serial(num_phase2_blocks, unroll=False):
                                    n_block: Tx.let = n_block_max_after_p1 - 1 - i
                                    softmax_step(n_block, apply_mask=True)

                                # Update n_block_max after Phase 2
                                n_block_max_after_p2: Tx.let = Tx.min(n_block_max_after_p1, n_block_min_causal)

                                # Phase 3: Unmasked blocks (no causal mask overhead)
                                # These are blocks in [0, n_block_min_causal)
                                for i in Tx.serial(n_block_max_after_p2, unroll=False):
                                    n_block: Tx.let = n_block_max_after_p2 - 1 - i
                                    softmax_step(n_block, apply_mask=False)
                                if tid_in_wg < BLK_M:
                                    sScale[ROW_SUM_BASE + tid_in_wg + wg_id * BLK_M] = row_sum[0]
                                bar_softmax_corr_full.arrive(wg_id)
                        with WarpgroupRole(wg_id, 2, regs=64):

                                bar_softmax_corr_full.wait(0, phase_q)
                                bar_softmax_corr_empty.arrive(0)
                                bar_softmax_corr_full.wait(1, phase_q)
                                phase_q ^= 1

                                # For causal, compute reduced trip count for correction warp
                                corr_trip_count: Tx.let = get_n_block_max(m_block_idx, is_causal) if is_causal else num_kv_blocks

                                for i_kv in Tx.serial(corr_trip_count - 1, unroll=False):
                                    for i_q in Tx.unroll(2):
                                        bar_softmax_corr_full.wait(i_q, phase_q)
                                        profiler.start(ProfileEventType.Correction, tid_in_wg == 0)
                                        acc_scale: Tx.f32
                                        should_rescale: Tx.i32

                                        if tid_in_wg < BLK_M:
                                            acc_scale = sScale[ACC_SCALE_BASE + tid_in_wg + i_q * BLK_M]
                                            should_rescale = Tx.Select(acc_scale < Tx.float32(1.0), 1, 0)
                                        else:
                                            should_rescale = 0

                                        any_needs_rescale: Tx.let = Tx.ptx.any_sync(0xFFFFFFFF, should_rescale)
                                        if any_needs_rescale != 0:
                                            if tid_in_wg < BLK_M:
                                                tmem_col_o_stage: Tx.let = tmem_o_base + i_q * tmem_offset
                                                RESCALE_TILE: Tx.let = 16

                                                o_row_buf: Tx.f32[16]
                                                o_row_wg = o_row_buf.view(128, 16, layout=TileLayout(S[(128, 16) : (1 @ axis_tid_in_wg, 1)]))

                                                for d_tile in Tx.unroll(ceildiv(HEAD_DIM, RESCALE_TILE)):
                                                    d_start: Tx.let = d_tile * RESCALE_TILE
                                                    if d_start < HEAD_DIM:
                                                        Tx.copy_async(o_row_wg, tmem[:, tmem_col_o_stage + d_start : tmem_col_o_stage + d_start + 16])
                                                        with Tx.thread():
                                                            Tx.mul(o_row_buf, o_row_buf, acc_scale)
                                                        Tx.copy_async(tmem[:, tmem_col_o_stage + d_start : tmem_col_o_stage + d_start + 16], o_row_wg[:, 0:16])
                                                Tx.ptx.tcgen05.wait.st()

                                        bar_p_full_o_rescaled.arrive(i_q)
                                        bar_softmax_corr_empty.arrive(1 - i_q)
                                        profiler.end(ProfileEventType.Correction, tid_in_wg == 0)
                                    # flip epi producer phase
                                    phase_q ^= 1
                                bar_softmax_corr_empty.arrive(1)

                                for i_q in Tx.unroll(2):
                                    # 1. Wait for softmax to signal row_sum is ready
                                    bar_softmax_corr_full.wait(i_q, phase_q)

                                    # 2. Read row_sum and release softmax_corr_empty immediately
                                    row_sum: Tx.let = sScale[ROW_SUM_BASE + tid_in_wg + i_q * BLK_M]
                                    bar_softmax_corr_empty.arrive(i_q)

                                    # 3. Wait for O_full and epi_empty (after releasing softmax)
                                    bar_o_full.wait(i_q, phase_tmem)
                                    bar_corr_epi_empty.wait(i_q, phase_tmem)

                                    profiler.start(ProfileEventType.EpiLDTMEM, tid_in_wg == 0)
                                    acc_O_mn_row_is_zero_or_nan: Tx.let = tvm.tirx.any(row_sum == Tx.float32(0.0), row_sum != row_sum)
                                    norm_scale: Tx.let = Tx.ptx.rcp(Tx.Select(acc_O_mn_row_is_zero_or_nan, Tx.float32(1.0), row_sum))
                                    tmem_col_o_stage: Tx.let = tmem_o_base + i_q * tmem_offset
                                    o_row_f32_buf: Tx.f32[TMEM_EPI_LD_SIZE]
                                    o_row_f32_wg = o_row_f32_buf.view(128, TMEM_EPI_LD_SIZE, layout=TileLayout(S[(128, TMEM_EPI_LD_SIZE) : (1 @ axis_tid_in_wg, 1)]))
                                    o_row_f16: Tx.f16[TMEM_EPI_LD_SIZE]

                                    for d_tile in Tx.unroll(ceildiv(HEAD_DIM, TMEM_EPI_LD_SIZE)):
                                        d_start: Tx.let = d_tile * TMEM_EPI_LD_SIZE
                                        if d_start < HEAD_DIM:
                                            Tx.copy_async(o_row_f32_wg, tmem[:, tmem_col_o_stage + d_start : tmem_col_o_stage + d_start + TMEM_EPI_LD_SIZE])
                                            with Tx.thread():
                                                Tx.mul(o_row_f32_buf, o_row_f32_buf, norm_scale)
                                            with Tx.thread():
                                                Tx.cast(o_row_f16, o_row_f32_buf)
                                                Tx.copy(O_smem[i_q, tid_in_wg, d_tile * TMEM_EPI_LD_SIZE : d_tile * TMEM_EPI_LD_SIZE + TMEM_EPI_LD_SIZE], o_row_f16, vec_len=8)

                                        profiler.end(ProfileEventType.EpiLDTMEM, tid_in_wg == 0)
                                    Tx.ptx.fence.proxy_async("shared::cta")

                                    # arrive epi_full
                                    bar_corr_epi_full.arrive(i_q)
                                    # Signal for the next work tile that O buffers in tmem are already read
                                    bar_p_full_o_rescaled.arrive(i_q)
                                phase_tmem ^= 1
                                phase_q ^= 1

                    scheduler.next_tile()

                # Deallocate TMEM after all tasks complete
                if wg_id == 0 and warp_id == 0:
                    Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
                    Tx.ptx.tcgen05.dealloc(0, n_cols=N_COLS_TMEM, cta_group=CTA_GROUP)

                Tx.cuda.cta_sync()

    return flash_attention4
# fmt: on


def prepare_data(batch_size, seq_len_q, seq_len_kv, num_qo_heads, num_kv_heads, head_dim):
    torch.manual_seed(0)
    Q = torch.randn((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)
    K = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    V = torch.randn((batch_size, seq_len_kv, num_kv_heads, head_dim), dtype=torch.float16)
    O = torch.zeros((batch_size, seq_len_q, num_qo_heads, head_dim), dtype=torch.float16)  # noqa: E741

    return Q, K, V, O


# ── Standard kernel interface ──────────────────────────────────────────

KERNEL_META = {
    "name": "flash_attention4",
    "category": "attention",
    "compute_capability": 10,
}

CONFIGS = [
    {
        "batch_size": 1,
        "seq_len": sl,
        "num_qo_heads": 32,
        "num_kv_heads": kv,
        "head_dim": 128,
        "is_causal": causal,
        "label": f"s{sl}_h32kv{kv}{'_causal' if causal else ''}",
    }
    for sl in [1024, 2048, 4096, 8192]
    for kv in [4, 8, 16, 32]
    for causal in [False, True]
]


def get_kernel(
    batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=False, **kwargs
):
    return get_flash_attention4_kernel(
        batch_size,
        seq_len,
        seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        is_causal=is_causal,
    )


def run_test(batch_size, seq_len, num_qo_heads, num_kv_heads, head_dim, is_causal=False, **kwargs):
    """Compile, run, and verify flash attention 4 kernel."""
    from tirx_kernels.runner import compile_kernel

    Q, K, V, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    prim_func = get_flash_attention4_kernel(
        batch_size,
        seq_len,
        seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        is_causal=is_causal,
    )
    ex = compile_kernel(prim_func)

    dev = tvm.cuda(0)
    Q_tvm = tvm.runtime.tensor(Q.cpu().numpy(), device=dev)
    K_tvm = tvm.runtime.tensor(K.cpu().numpy(), device=dev)
    V_tvm = tvm.runtime.tensor(V.cpu().numpy(), device=dev)
    O_tvm = tvm.runtime.tensor(
        np.zeros((batch_size, seq_len, num_qo_heads, head_dim), dtype=np.float16), dev
    )
    profiler_buf = tvm.runtime.tensor(np.zeros(PROFILER_BUFFER_SIZE, dtype=np.uint64), dev)

    ex(Q_tvm, K_tvm, V_tvm, O_tvm, profiler_buf)
    torch.cuda.synchronize()

    # Reference: naive scaled-dot-product attention
    Q_t = Q.float().transpose(1, 2)
    K_t = K.float().transpose(1, 2)
    V_t = V.float().transpose(1, 2)
    if num_qo_heads != num_kv_heads:
        repeat_factor = num_qo_heads // num_kv_heads
        K_t = K_t.repeat_interleave(repeat_factor, dim=1)
        V_t = V_t.repeat_interleave(repeat_factor, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) * scale
    if is_causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    ref = torch.matmul(attn, V_t).transpose(1, 2).to(torch.float16)
    np.testing.assert_allclose(O_tvm.numpy(), ref.cpu().numpy(), rtol=1e-2, atol=1e-2)


def run_bench(
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    is_causal=False,
    warmup=10,
    repeat=30,
    **kwargs,
):
    """Benchmark flash attention 4 with Proton profiling."""
    from tirx_kernels.runner import compile_kernel

    Q, K, V, _ = prepare_data(batch_size, seq_len, seq_len, num_qo_heads, num_kv_heads, head_dim)
    prim_func = get_flash_attention4_kernel(
        batch_size,
        seq_len,
        seq_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        is_causal=is_causal,
    )
    ex = compile_kernel(prim_func)

    dev = tvm.cuda(0)
    Q_tvm = tvm.runtime.tensor(Q.cpu().numpy(), device=dev)
    K_tvm = tvm.runtime.tensor(K.cpu().numpy(), device=dev)
    V_tvm = tvm.runtime.tensor(V.cpu().numpy(), device=dev)
    O_tvm = tvm.runtime.tensor(
        np.zeros((batch_size, seq_len, num_qo_heads, head_dim), dtype=np.float16), dev
    )
    profiler_buf = tvm.runtime.tensor(np.zeros(PROFILER_BUFFER_SIZE, dtype=np.uint64), dev)

    with ProtonContext("flash_attention4") as ctx:
        tir_ms = bench(
            lambda: ex(Q_tvm, K_tvm, V_tvm, O_tvm, profiler_buf),
            warmup=warmup,
            repeat=repeat,
            proton_name="tir",
        )

        # CuTeDSL Blackwell FMHA baseline
        try:
            import os
            import sys

            import cutlass
            import cutlass.cute as cute
            import cutlass.torch as cutlass_torch

            current_dir = os.path.dirname(os.path.abspath(__file__))
            tvm_root = os.environ.get(
                "TVM_HOME",
                os.path.abspath(os.path.join(current_dir, "../../../../tir")),
            )
            blackwell_path = os.path.join(
                tvm_root, "3rdparty/cutlass/examples/python/CuTeDSL/blackwell"
            )
            sys.path.insert(0, blackwell_path)
            from fmha import BlackwellFusedMultiHeadAttentionForward, MaskType

            Q_cute = Q.cuda()
            K_cute = K.cuda()
            V_cute = V.cuda()
            O_cute = torch.zeros_like(Q_cute)

            q_tensor, q_torch = cutlass_torch.cute_tensor_like(
                Q_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            k_tensor, k_torch = cutlass_torch.cute_tensor_like(
                K_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            v_tensor, v_torch = cutlass_torch.cute_tensor_like(
                V_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            o_tensor, o_torch = cutlass_torch.cute_tensor_like(
                O_cute, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            q_torch.copy_(Q_cute)
            k_torch.copy_(K_cute)
            v_torch.copy_(V_cute)

            mma_tiler = (128, 128, head_dim)
            fmha = BlackwellFusedMultiHeadAttentionForward(
                cutlass.Float32,
                cutlass.Float32,
                mma_tiler,
                is_persistent=True,
                mask_type=MaskType.CAUSAL_MASK if is_causal else MaskType.NO_MASK,
            )
            current_stream = cutlass_torch.default_stream()
            scale_softmax_log2 = (1.0 / math.sqrt(head_dim)) * math.log2(math.exp(1.0))
            scale_output = 1.0
            problem_size = (
                batch_size,
                seq_len,
                seq_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
            )
            compiled_fmha = cute.compile(
                fmha,
                q_tensor.iterator,
                k_tensor.iterator,
                v_tensor.iterator,
                o_tensor.iterator,
                problem_size,
                None,
                None,
                scale_softmax_log2,
                scale_output,
                current_stream,
            )

            def _run_cutedsl():
                compiled_fmha(
                    q_tensor.iterator,
                    k_tensor.iterator,
                    v_tensor.iterator,
                    o_tensor.iterator,
                    problem_size,
                    None,
                    None,
                    scale_softmax_log2,
                    scale_output,
                    current_stream,
                )

            bench(_run_cutedsl, warmup=warmup, repeat=repeat, proton_name="cutedsl_fa4")
        except Exception as e:
            import sys

            print(f"BASELINE_ERROR: cutedsl_fa4: {e}", file=sys.stderr)

        # Flash-Attention SM100 baseline
        try:
            import cutlass
            import cutlass.cute as cute
            import cutlass.torch as cutlass_torch
            from flash_attn.cute.flash_fwd_sm100 import FlashAttentionForwardSm100

            Q_fa = Q.cuda()
            K_fa = K.cuda()
            V_fa = V.cuda()
            O_fa = torch.zeros_like(Q_fa)

            q_t, q_th = cutlass_torch.cute_tensor_like(
                Q_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            k_t, k_th = cutlass_torch.cute_tensor_like(
                K_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            v_t, v_th = cutlass_torch.cute_tensor_like(
                V_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            o_t, o_th = cutlass_torch.cute_tensor_like(
                O_fa, cutlass.Float16, is_dynamic_layout=True, assumed_align=16
            )
            q_th.copy_(Q_fa)
            k_th.copy_(K_fa)
            v_th.copy_(V_fa)

            fa_fwd = FlashAttentionForwardSm100(
                head_dim=head_dim,
                head_dim_v=head_dim,
                qhead_per_kvhead=num_qo_heads // num_kv_heads,
                is_causal=is_causal,
                is_local=False,
                pack_gqa=False,
                m_block_size=128,
                n_block_size=128,
                is_persistent=True,
            )
            stream = cutlass_torch.default_stream()
            scale = 1.0 / math.sqrt(head_dim)
            compiled_fa = cute.compile(
                fa_fwd,
                q_t,
                k_t,
                v_t,
                o_t,
                None,
                scale,
                stream,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

            def _run_fa_sm100():
                compiled_fa(
                    q_t,
                    k_t,
                    v_t,
                    o_t,
                    None,
                    scale,
                    stream,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

            bench(
                _run_fa_sm100,
                warmup=warmup,
                repeat=repeat,
                proton_name="flashattn_sm100",
            )
        except Exception as e:
            import sys

            print(f"BASELINE_ERROR: flashattn_sm100: {e}", file=sys.stderr)

        # FlashInfer baseline
        try:
            import flashinfer

            workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
            prefill_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                workspace_buffer, "NHD", backend="cutlass"
            )
            qo_indptr = torch.tensor([0, seq_len], device="cuda:0", dtype=torch.int32)
            kv_indptr = torch.tensor([0, seq_len], device="cuda:0", dtype=torch.int32)
            prefill_wrapper.plan(
                qo_indptr,
                kv_indptr,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
            )
            q_fi = Q.clone().reshape(-1, num_qo_heads, head_dim).cuda()
            k_fi = K.clone().reshape(-1, num_kv_heads, head_dim).cuda()
            v_fi = V.clone().reshape(-1, num_kv_heads, head_dim).cuda()
            bench(
                lambda: prefill_wrapper.run(q_fi, k_fi, v_fi),
                warmup=warmup,
                repeat=repeat,
                proton_name="flashinfer",
            )
        except Exception as e:
            import sys

            print(f"BASELINE_ERROR: flashinfer: {e}", file=sys.stderr)

    return {"impls": ctx.get_impl_times()}
