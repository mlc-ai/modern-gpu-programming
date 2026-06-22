..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

CUDA C++/PTX intrinsics
=======================

When no tile primitive covers what you need, two escape hatches reach the hardware
directly: **call a backend intrinsic** (the ``T.cuda.*`` / ``T.ptx.*`` namespaces
from ``tvm.backend.cuda``), or **inline raw CUDA** source.

Calling backend intrinsics
--------------------------

``T.cuda.*`` and ``T.ptx.*`` expose the CUDA backend's device intrinsics directly —
synchronization, mbarriers, reductions, and the PTX data-movement / MMA families:

.. code-block:: python

    T.cuda.cta_sync()                    # block barrier (__syncthreads)
    T.cuda.warp_sync()                   # __syncwarp
    T.cuda.warpgroup_sync(8)             # warpgroup barrier
    T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))   # block-level reduction

    bar = T.alloc_shared((1,), "uint64")
    T.ptx.mbarrier.init(bar.data, 1)     # mbarrier for async completion
    T.ptx.mbarrier.try_wait(bar.data, phase)

A complete, runnable example — a warp all-reduce via ``T.tvm_warp_shuffle_xor``:

.. code-block:: python

    @T.prim_func
    def warp_reduce(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (32,), "float32", align=16)
        T.device_entry()
        cta_id = T.cta_id([1]); warp_id = T.warp_id([1]); lane_id = T.lane_id([32])
        v = T.alloc_local((1,), "float32"); i = T.alloc_local((1,), "int32")
        v[0] = T.float32(31 - lane_id)
        i[0] = 16
        while i[0] >= 1:
            v[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
            i[0] = i[0] // 2
        A[lane_id] = v[0]

The shuffle lowers straight to ``__shfl_xor_sync``:

.. code-block:: c++

    v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);

Other families under ``T.ptx.*`` / ``T.cuda.*``: ``cp_async`` (LDGSTS),
``cp_async.bulk.tensor`` (TMA), ``ldmatrix`` / ``stmatrix``, ``tcgen05.*``
(Blackwell MMA), ``atomic_add``, ``fence`` … See the backend API reference for the
full ``tvm.backend.cuda`` reference.

Synchronization semantics
-------------------------

Four synchronization mechanisms come up constantly in the GEMM and Flash Attention
kernels. Because they control asynchronous engines and parallel thread groups,
misusing any of them usually leads to silent corruption or deadlock.

**Mbarrier Phases.** Mbarriers track arrivals using a single internal phase bit.
The ``T.ptx.mbarrier.try_wait(bar, phase)`` intrinsic blocks until the barrier's
internal phase *differs* from the ``phase`` argument provided by the caller.
Consequently, when reusing a barrier across loop iterations, the caller must flip
its local phase tracker (``phase ^= 1``) after every wait. Failing to do so causes
subsequent waits to return immediately, allowing the engine to read half-written
memory. :ref:`chap_gemm_basics` walks through the full phase-tracking table.

**Election.** ``T.ptx.elect_sync()`` elects a *single active lane within a warp*,
not lane 0, and not one thread per CTA. To narrow an issuer down to exactly one
thread, you must pair it with a warp-level guard. The pattern ``if warp_id == 0:``
followed by ``if T.ptx.elect_sync():`` is used to issue ``Tx.gemm_async`` and
``tcgen05.commit`` in :ref:`chap_gemm_basics`.

**Named Warpgroup Barriers.** ``T.cuda.cta_sync()`` maps to ``__syncthreads()`` and
requires *every* CTA thread to arrive. Once warpgroups specialize onto different
code paths, placing a ``cta_sync()`` inside a warpgroup branch deadlocks the kernel
because the other warpgroups never reach it. The hardware provides 16 named
barriers (IDs 0 to 15); ``T.cuda.warpgroup_sync(10)`` synchronizes only the threads
of one warpgroup. Distinct warpgroups take distinct IDs (e.g.,
``warpgroup_sync(wg_id + 10)``) so they never collide on the same hardware barrier.
See :ref:`chap_gemm_advanced`.

**Fences.** Fences order a producer's writes before a consumer (often an
asynchronous engine) reads them:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Fence
     - Orders
   * - ``T.ptx.fence.proxy_async("shared::cta")``
     - thread-written shared memory before an async proxy (TMA store / MMA) reads it
   * - ``T.ptx.fence.mbarrier_init()``
     - mbarrier initialization before later arrivals or waits use the barrier
   * - ``T.ptx.tcgen05.fence.after_thread_sync()``
     - a conservative ordering fence on the ``tcgen05`` writeback edge (Steps 8 and 9 add it; it is not needed on the TMA-to-MMA path)

Inlining raw CUDA
-----------------

For something with no intrinsic at all, inject a ``__device__`` function from a
source string with ``T.cuda.func_call(name, *args, source_code=..., return_type=...)``:

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """

    @T.prim_func
    def k(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = T.cuda.func_call("my_relu", A[tx], source_code=SRC, return_type="float32")

The source is emitted verbatim and the call is wired in:

.. code-block:: c++

    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    // ...
    B_ptr[tx] = my_relu(A_ptr[tx]);
