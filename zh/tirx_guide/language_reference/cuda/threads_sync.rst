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

CUDA C++/PTX intrinsic
======================

当没有现成 tile primitive 覆盖你的需求时，有两条 escape hatch 可以直接触达
硬件：**调用 backend intrinsic**（来自 ``tvm.backend.cuda`` 的 ``T.cuda.*`` /
``T.ptx.*`` 命名空间），或者 **内联原始 CUDA** 源码。

调用 backend intrinsic
----------------------

``T.cuda.*`` 和 ``T.ptx.*`` 会直接暴露 CUDA backend 的 device intrinsic：
同步、mbarrier、reduction，以及 PTX data-movement / MMA 家族：

.. code-block:: python

    T.cuda.cta_sync()                    # block barrier (__syncthreads)
    T.cuda.warp_sync()                   # __syncwarp
    T.cuda.warpgroup_sync(8)             # warpgroup barrier
    T.cuda.cta_sum(val, num_warps, scratch.ptr_to([0]))   # block-level reduction

    bar = T.alloc_shared((1,), "uint64")
    T.ptx.mbarrier.init(bar.data, 1)     # mbarrier for async completion
    T.ptx.mbarrier.try_wait(bar.data, phase)

下面是一个完整可运行的例子：通过 ``T.tvm_warp_shuffle_xor`` 做 warp all-reduce：

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

shuffle 会直接 lower 成 ``__shfl_xor_sync``：

.. code-block:: c++

    v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);

``T.ptx.*`` / ``T.cuda.*`` 下还有其他家族：``cp_async``（LDGSTS）、
``cp_async.bulk.tensor``（TMA）、``ldmatrix`` / ``stmatrix``、``tcgen05.*``
（Blackwell MMA）、``atomic_add``、``fence`` 等。完整 ``tvm.backend.cuda``
参考请见 backend API reference。

同步语义
--------

GEMM 和 Flash Attention kernel 中反复出现四种同步机制。它们控制异步引擎和
并行线程组，所以任何一种用错，通常都会导致静默数据损坏或 deadlock。

**Mbarrier phase。** Mbarrier 使用一个内部 phase bit 跟踪 arrival。
``T.ptx.mbarrier.try_wait(bar, phase)`` intrinsic 会阻塞，直到 barrier 的内部
phase 与调用者提供的 ``phase`` 参数 *不同*。因此，当跨 loop iteration 复用
barrier 时，调用者必须在每次 wait 之后翻转自己的本地 phase tracker
（``phase ^= 1``）。如果忘了翻转，后续 wait 会立即返回，导致引擎读取半写入
的 memory。:ref:`zh_chap_gemm_basics` 中完整走了一遍 phase-tracking 表。

**Election。** ``T.ptx.elect_sync()`` 会在一个 warp 内选出 *单个 active lane*，
不是 lane 0，也不是每个 CTA 一个线程。若要把 issuer 缩小到精确一个线程，
必须配合 warp-level guard。:ref:`zh_chap_gemm_basics` 中使用
``if warp_id == 0:`` 后接 ``if T.ptx.elect_sync():`` 的模式来发出
``Tx.gemm_async`` 和 ``tcgen05.commit``。

**Named Warpgroup Barrier。** ``T.cuda.cta_sync()`` 会映射到 ``__syncthreads()``，
并要求 *每个* CTA 线程都到达。一旦 warpgroup 被 specialize 到不同代码路径，
把 ``cta_sync()`` 放进某个 warpgroup 分支就会让 kernel deadlock，因为其他
warpgroup 永远到不了它。硬件提供 16 个 named barrier（ID 0 到 15）；
``T.cuda.warpgroup_sync(10)`` 只同步一个 warpgroup 的线程。不同 warpgroup
使用不同 ID（例如 ``warpgroup_sync(wg_id + 10)``），避免撞到同一个硬件
barrier。见 :ref:`zh_chap_gemm_advanced`。

**Fence。** Fence 保证 producer 的写入排在 consumer（通常是异步引擎）读取
之前：

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Fence
     - 保证的顺序
   * - ``T.ptx.fence.proxy_async("shared::cta")``
     - 线程写入 shared memory，先于 async proxy（TMA store / MMA）读取它
   * - ``T.ptx.fence.mbarrier_init()``
     - mbarrier 初始化，先于后续 arrival 或 wait 使用该 barrier
   * - ``T.ptx.tcgen05.fence.after_thread_sync()``
     - ``tcgen05`` writeback 边上的保守 ordering fence（Steps 8 和 9 会加入；TMA-to-MMA 路径不需要）

内联原始 CUDA
-------------

如果某个功能完全没有 intrinsic，可以用
``T.cuda.func_call(name, *args, source_code=..., return_type=...)`` 从源码字符串
注入一个 ``__device__`` 函数：

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

源码会原样发出，并把调用接进去：

.. code-block:: c++

    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    // ...
    B_ptr[tx] = my_relu(A_ptr[tx]);
