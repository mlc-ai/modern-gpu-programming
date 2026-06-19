"""Full no-sugar single-tile GEMM: raw TMA + raw MMA + raw TMEM->RF->GMEM.

Every bit of TIRx sugar is unfolded here:

  - No ``Tx.copy`` / ``Tx.copy_async``
        -> raw ``Tx.ptx.cp_async.bulk.tensor.g2c``
  - No ``Tx.gemm_async``
        -> raw ``Tx.ptx.tcgen05.{encode_instr_descriptor,
                                  encode_matrix_descriptor, mma, commit}``
  - No ``Tx.copy(reg, tmem)`` sugar
        -> raw ``Tx.ptx.tcgen05.ld`` per element
  - No ``SMEMPool``
        -> independent ``Tx.alloc_buffer(scope="shared")`` per tile
  - No ``Pipe`` / ``TCGen05Bar`` helpers
        -> raw ``Tx.ptx.mbarrier.{init, arrive.expect_tx, try_wait}``

Single tile (M = N = 128, K = 64), no pipelining, no cluster, single CTA,
single warpgroup. Used by :numref:`chap_data_layouts` to motivate every TIRx
abstraction introduced in the rest of the chapter.
"""
import numpy as np
import tvm
from tvm.script import tirx as Tx


d_type, a_type, b_type = "float32", "float16", "float16"
M, N, K = 128, 128, 64
MMA_K = 16
N_COLS = 512
cta_group = 1
SWIZZLE = 3  # 128B swizzle mode id used by cuTensorMapEncodeTiled

A_BYTES = M * K * 2
B_BYTES = N * K * 2
TMA_TOTAL_BYTES = A_BYTES + B_BYTES

# SMEM layout: row-major 2D with a 128B swizzle wrapper (one "128B atom"
# == 64 fp16 elements). Both TMA and tcgen05.mma understand this wrapper
# directly; see :numref:`chap_data_layouts` for the derivation.
A_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(M, 1, 64) : (64, M * 64, 1)]),
)
B_layout = Tx.ComposeLayout(
    Tx.SwizzleLayout(3, 3, 3, swizzle_inner=True),
    Tx.TileLayout(Tx.S[(N, 1, 64) : (64, N * 64, 1)]),
)
ldo, sdo = 1, 64  # matrix-descriptor leading / stride dim (in 128B atoms)


# fmt: off
@Tx.prim_func(check_well_formed=False)
def gemm_nosugar(A: Tx.Buffer((M, K), a_type, layout=Tx.TileLayout(Tx.S[M, K])),
                 B: Tx.Buffer((N, K), b_type, layout=Tx.TileLayout(Tx.S[N, K])),
                 C: Tx.Buffer((M, N), d_type)):
    # -- host-side: build TMA tensor-map descriptors for A and B ----------
    A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
    B_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
    Tx.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        A_map, a_type, 2, A.data,
        K, M,          # global_shape  (innermost first)
        K * 2,         # global_stride (rank-1, in bytes)
        K, M,          # box_dim       (innermost first)
        1, 1,          # element_strides
        0, SWIZZLE, 2, 0,
    )
    Tx.call_packed(
        "runtime.cuTensorMapEncodeTiled",
        B_map, b_type, 2, B.data,
        K, N, K * 2, K, N, 1, 1, 0, SWIZZLE, 2, 0,
    )

    # -- device side -------------------------------------------------------
    Tx.device_entry()
    Tx.cta_id([1])
    Tx.warpgroup_id([1])
    warp_id = Tx.warp_id_in_wg([4])
    Tx.lane_id([32])
    tx = Tx.thread_id([128])
    with Tx.cta():
        # Three independent uint32/uint64 SMEM scalars (NOT sub-offsets
        # of a shared buffer) so that mbarrier addresses are unique.
        tmem_addr = Tx.shared_scalar("uint32")
        bar_tma = Tx.shared_scalar("uint64")
        bar_mma = Tx.shared_scalar("uint64")
        A_smem = Tx.alloc_buffer((M, K), a_type, scope="shared", layout=A_layout, align=128)
        B_smem = Tx.alloc_buffer((N, K), b_type, scope="shared", layout=B_layout, align=128)

        reg = Tx.alloc_buffer((N,), d_type, scope="local")
        descA = Tx.alloc_buffer((1,), "uint64", scope="local")
        descB = Tx.alloc_buffer((1,), "uint64", scope="local")
        descI = Tx.alloc_buffer((1,), "uint32", scope="local")
        tma_phase = Tx.alloc_buffer((1,), "int32", scope="local")
        mma_phase = Tx.alloc_buffer((1,), "int32", scope="local")

        # (1) allocate 512 TMEM columns from warp 0.
        with Tx.warp(warp_id == 0):
            Tx.ptx.tcgen05.alloc(Tx.address_of(tmem_addr), n_cols=N_COLS, cta_group=cta_group)

        with Tx.thread():
            # (2) init two mbarriers with expected-arrival count 1.
            if tx == 0:
                Tx.ptx.mbarrier.init(Tx.address_of(bar_tma), 1)
                Tx.ptx.mbarrier.init(Tx.address_of(bar_mma), 1)
            Tx.ptx.fence.proxy_async("shared::cta")
            Tx.ptx.fence.mbarrier_init()
            tma_phase[0] = 0
            mma_phase[0] = 0

            for i in range(N):
                reg[i] = 0.0
            Tx.cuda.cta_sync()

            # (3) TMA load A and B tiles; one fixed CTA thread issues the
            #     copy and bumps the expected byte counter on bar_tma.
            if tx == 0:
                # g2c(dim, dst, bar, tensormap_ptr, cta_mask, cta_group,
                #     cache_hint, *coords). cta_mask=0 and cta_group=1
                # mean unicast TMA to this CTA.
                Tx.ptx.cp_async.bulk.tensor.g2c(
                    2, A_smem.data, Tx.address_of(bar_tma), Tx.address_of(A_map),
                    0, 1, "", 0, 0,
                )
                Tx.ptx.cp_async.bulk.tensor.g2c(
                    2, B_smem.data, Tx.address_of(bar_tma), Tx.address_of(B_map),
                    0, 1, "", 0, 0,
                )
                Tx.ptx.mbarrier.arrive.expect_tx(Tx.address_of(bar_tma), TMA_TOTAL_BYTES)
            Tx.ptx.mbarrier.try_wait(Tx.address_of(bar_tma), tma_phase[0])
            tma_phase[0] = tma_phase[0] ^ 1

            # (4) Issue tcgen05.mma from one fixed CTA thread. Build the
            #     instruction descriptor once, then one matrix descriptor
            #     per K-inner step (MMA_K = 16 fp16 columns per step).
            if tx == 0:
                Tx.ptx.tcgen05.encode_instr_descriptor(
                    descI.data, d_dtype=d_type, a_dtype=a_type, b_dtype=b_type,
                    M=M, N=N, K=MMA_K, trans_a=False, trans_b=False,
                    n_cta_groups=cta_group,
                )
                for k in range(K // MMA_K):
                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                        descA.data,
                        A_smem.access_ptr("r", offset=A_smem.elem_offset_of([0, k * MMA_K])),
                        ldo=ldo, sdo=sdo, swizzle=SWIZZLE,
                    )
                    Tx.ptx.tcgen05.encode_matrix_descriptor(
                        descB.data,
                        B_smem.access_ptr("r", offset=B_smem.elem_offset_of([0, k * MMA_K])),
                        ldo=ldo, sdo=sdo, swizzle=SWIZZLE,
                    )
                    Tx.ptx.tcgen05.mma(
                        tmem_addr, descA[0], descB[0], descI[0],
                        d_dtype=d_type, a_dtype=a_type, b_dtype=b_type,
                        use_a_tmem=False, cta_group=cta_group,
                        enable_input_d=(k != 0),
                    )
                Tx.ptx.tcgen05.commit(Tx.address_of(bar_mma), cta_group)
            Tx.ptx.mbarrier.try_wait(Tx.address_of(bar_mma), mma_phase[0])
            mma_phase[0] = mma_phase[0] ^ 1
            Tx.cuda.cta_sync()

            # (5) TMEM -> RF, then RF -> GMEM (one column at a time).
            Tx.ptx.tcgen05.fence.after_thread_sync()
            for i in range(N):
                Tx.ptx.tcgen05.ld(
                    tmem_addr, reg[i],
                    shape="32x32b", num=1, row=warp_id * 32, col=i,
                )
            Tx.ptx.tcgen05.wait.ld()
            for i in range(N):
                C[tx, i] = reg[i]

        # (6) release TMEM back to the hardware pool.
        with Tx.warp(warp_id == 0):
            Tx.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
            Tx.ptx.tcgen05.dealloc(tmem_addr, n_cols=N_COLS, cta_group=cta_group)
# fmt: on


def build():
    """Compile the no-sugar GEMM for sm_100a and return (source, module)."""
    target = tvm.target.Target("cuda")  # defaults to sm_100a on Blackwell
    mod = tvm.IRModule({"main": gemm_nosugar})
    with target:
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    return mod.mod.imports_[0].inspect_source("cuda"), mod


def run_correctness():
    """Build, run, and torch-compare the no-sugar GEMM on the current GPU."""
    import torch

    torch.manual_seed(42)
    dev = tvm.cuda(0)
    _, mod = build()
    A_t = torch.rand((M, K), dtype=torch.float16)
    B_t = torch.rand((N, K), dtype=torch.float16)
    C_t = torch.zeros((M, N), dtype=torch.float32)
    A = tvm.runtime.tensor(A_t, device=dev)
    B = tvm.runtime.tensor(B_t, device=dev)
    C = tvm.runtime.tensor(C_t, device=dev)
    mod(A, B, C)
    ref = torch.matmul(A_t, B_t.T).to(torch.float32)
    np.testing.assert_allclose(C.numpy(), ref.numpy(), rtol=1e-3, atol=1e-2)
    return True


if __name__ == "__main__":
    src, _ = build()
    print(f"Generated CUDA: {len(src.splitlines())} lines")
    assert run_correctness()
    print("[PASS] no-sugar GEMM")
