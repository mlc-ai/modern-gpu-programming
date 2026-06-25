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

Buffer 与内存
=============

参数 buffer 通过 ``T.match_buffer`` 绑定；scratch buffer 则在函数体里用下面
两类声明 API 之一创建。可以用 ``A[i, j]`` 索引 buffer，用
``A[m0:m0+BM, 0:BK]`` 切片（得到 ``BufferRegion``），也可以用
``A.ptr_to([i, j])`` 取得指针，或用 ``A.data`` 取得原始 data pointer。

声明 buffer
-----------

创建 buffer 有两个基础 API：

- ``T.alloc_buffer(shape, dtype, scope=..., ...)`` — **分配新的存储空间**
  （发出一个 ``AllocBuffer`` 节点）并返回 ``Buffer``。``T.alloc_shared`` /
  ``T.alloc_local`` 只是 ``alloc_buffer`` 加上 ``scope="shared"`` /
  ``scope="local"`` 的简写。
- ``T.decl_buffer(shape, dtype, data=..., ...)`` — 在已有指针 ``data`` 上
  **声明一个 view** （不分配）；用于 alias 或 reinterpret 存储空间，例如 pool
  的子区域或 tensor-memory address。若 ``data=None``，它会像 ``alloc_buffer``
  一样分配。

buffer 的 ``data`` pointer 是一个 immutable ``Var``（``alloc_buffer`` 会定义它；
``decl_buffer`` 接收它）。如果要让 buffer 背后使用一个指针 *表达式*，请先
绑定该表达式——见 :doc:`data_types`。

二者共享同一种 descriptor；最重要的参数如下：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 参数
     - 含义
   * - ``dtype``
     - 元素类型，例如 ``"float32"``、``"float16"``、``"float4_e2m1fn"`` 等
   * - ``shape``
     - 逻辑形状（一组 extent）
   * - ``layout``
     - 物理映射（:ref:`TileLayout <zh_chap_tirx_layout_api>`）；``"default"`` = dense
       row-major
   * - ``elem_offset`` / ``allocated_addr``
     - ``elem_offset``（或 ``byte_offset``）把一个 *view* 放到 ``data`` 内部的
       某个偏移；``allocated_addr`` 携带预先分配的地址（tensor memory）
   * - ``align``
     - data pointer 的对齐字节数

``scope`` 参数选择内存空间：

.. list-table::
   :header-rows: 1
   :widths: 26 22 52

   * - Scope
     - 简写
     - 内存
   * - ``"global"``
     - (default)
     - device global memory
   * - ``"shared"``
     - ``T.alloc_shared``
     - static shared memory（``__shared__``）
   * - ``"shared.dyn"``
     - (pool)
     - dynamic shared memory（pooled，见下文）
   * - ``"local"``
     - ``T.alloc_local``
     - per-thread register
   * - ``"tmem"``
     - (TMEM pool)
     - Blackwell tensor memory（见下文）

.. code-block:: python

    A = T.match_buffer(A_ptr, (M, K), "float16", align=16)   # parameter buffer
    As = T.alloc_shared((BM, BK), "float16")                 # new shared tile
    acc = T.alloc_local((4,), "float32")                     # register accumulator
    view = T.decl_buffer((BM, BK), "float16", data=As.data)  # a view over As

**基于 pointer 的 buffer 本质上只是 pointer 上的一层 metadata。** 对任何
非 tmem buffer，声明都只是一个 pointer 加一个 layout；索引会解析成地址：

    addr(buffer[coord]) = buffer.data + elem_offset + layout.apply(coord, shape=shape)["m"]

（``layout.apply`` 返回每个轴的映射；其中 ``"m"`` 分量是元素偏移。）因此
*同一个* 逻辑访问会完全根据 buffer metadata 编译成不同的地址算术。对一个
4×8 区域写 ``B[i, j] = A[i, j] + 1``，并用四种方式声明 ``B``：

.. code-block:: python

    from tvm.tirx.layout import TileLayout, S

    B = T.match_buffer(p, (4, 8), "float32")                                       # row-major
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(1, 4)]))  # column-major
    B = T.match_buffer(p, (4, 8), "float32", elem_offset=64)                       # shifted view
    B = T.match_buffer(p, (4, 8), "float32", layout=TileLayout(S[(4, 8):(16, 1)])) # row stride 16

每种声明都会让 ``B[i, j]`` 在生成的 CUDA 中 lower 成不同索引（``A[i, j]``
load 仍是 ``i*8 + j``，只有 ``B`` 的 metadata 改了）：

.. code-block:: c++

    B_ptr[((i * 8) + j)]        = ...;   // row-major:        i*8 + j
    B_ptr[((j * 4) + i)]        = ...;   // column-major:     j*4 + i
    B_ptr[(((i * 8) + j) + 64)] = ...;   // elem_offset=64:   i*8 + j + 64
    B_ptr[((i * 16) + j)]       = ...;   // row stride 16:    i*16 + j

Shared memory
-------------

shared memory 有两种形式：**static** （编译期固定大小）和 **dynamic** （launch
时确定大小）；此外还有一个 pool helper 用来管理 dynamic 情况。

Static
~~~~~~

最简单的 shared buffer 是 **static** 形式：``T.alloc_shared``（也就是
``scope="shared"``），大小在编译期确定。把数据 stage 进去，调用
``cta_sync`` 让整个 block 都看到这些写入，然后再读出：

.. code-block:: python

    @T.prim_func
    def smem_demo(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (128,), "float32")
        B = T.match_buffer(B_ptr, (128,), "float32")
        T.device_entry()
        bx = T.cta_id([1])
        tx = T.thread_id([128])
        sm = T.alloc_shared((128,), "float32")   # static shared memory
        sm[tx] = A[tx]
        T.cuda.cta_sync()
        B[tx] = sm[tx] * T.float32(2.0)

它会 lower 成普通 ``__shared__`` array（省略生成 CUDA 的样板部分）：

.. code-block:: c++

    extern "C" __global__ void __launch_bounds__(128)
    smem_demo_kernel(float* __restrict__ A_ptr, float* __restrict__ B_ptr) {
      int tx = ((int)threadIdx.x);
      __shared__ alignas(64) float sm_ptr[128];      // T.alloc_shared
      sm_ptr[tx] = A_ptr[tx];
      __syncthreads();                               // T.cuda.cta_sync()
      B_ptr[tx] = sm_ptr[tx] * 2.0f;
    }

Dynamic
~~~~~~~

**Dynamic** shared memory（``scope="shared.dyn"``）的大小按 launch 确定（即
``sharedMemBytes`` launch 参数），不是编译期确定。一个 kernel **只能有一个**
dynamic-shared allocation，也就是 *arena*。因此你只分配一次 arena，再用
``T.decl_buffer`` 把每个 buffer 声明成它内部的一个 view：``data=`` 传 arena
pointer，并设置 ``elem_offset``：

.. code-block:: python

    arena = T.alloc_buffer((128,), "float32", scope="shared.dyn")   # the one arena
    As = T.decl_buffer((64,), "float32", data=arena.data, scope="shared.dyn")                 # offset 0
    Bs = T.decl_buffer((64,), "float32", data=arena.data, elem_offset=64, scope="shared.dyn") # offset 64
    As[tx] = A[tx]
    Bs[tx] = B[tx]
    T.cuda.cta_sync()
    C[tx] = As[tx] + Bs[tx]

两个 view 共享同一个 ``extern __shared__`` arena（省略生成 CUDA 的样板部分；
为了清楚起见，arena 命名为 ``smem``）：

.. code-block:: c++

    extern __shared__ __align__(64) float smem[];   // the one dynamic-shared arena
    smem[tx]      = A_ptr[tx];                       // As — view at offset 0
    smem[tx + 64] = B_ptr[tx];                       // Bs — view at offset 64
    __syncthreads();
    C_ptr[tx] = smem[tx] + smem[tx + 64];

（两次单独调用 ``alloc_buffer(scope="shared.dyn")`` 是错误的——*只允许一个
dynamic shared memory allocation*。）因此 static shared memory 在编译期定大小
（``__shared__ T x[N];``）；dynamic shared memory 则是这个 launch-sized arena，
其内部不同偏移处声明多个 view。

.. note::

   **TVM 如何标注 dynamic-shared 大小。** arena 的大小在编译期已知（这里
   ``128`` 个 float = ``512`` bytes）。lowering 期间，TVM 会向 device kernel 的
   ``tirx.kernel_launch_params`` 追加一个 ``"tirx.use_dyn_shared_memory"`` tag；
   host launcher 会计算总字节数，并把它作为最后一个 launch 参数传入：

   .. code-block:: python

       # device kernel attribute:
       "tirx.kernel_launch_params": ["blockIdx.x", "threadIdx.x", "tirx.use_dyn_shared_memory"]

       # host-side launch call  (..., gridDim.x, blockDim.x, dyn_shared_bytes):
       T.call_packed("dyn_kernel", A.data, B.data, C.data, 1, 64, 512)

   运行时，这个 ``512`` 会成为 ``cuLaunchKernelEx`` 调用中的
   ``config.sharedMemBytes``。你不需要手动设置它；它由 ``shared.dyn``
   allocation 的大小推导而来。

Pool sugar
~~~~~~~~~~

``T.SMEMPool`` 会自动处理 arena bookkeeping：它用 bump allocator 分配偏移，
因此你不用手写 ``decl`` view。除了 ``alloc`` / ``commit`` 之外，它还提供
per-buffer ``align=``、一个为你构建 MMA-compatible swizzle layout 的
``alloc_mma`` helper，以及用于回退 cursor、复用空间的 ``move_base_to``：

.. code-block:: python

    pool = T.SMEMPool()                          # bump allocator over shared.dyn
    As = pool.alloc((BM, BK), "float16", align=128)   # carve a tile
    Bs = pool.alloc((BK, BN), "float16", align=128)
    Cs = pool.alloc_mma((BM, BN), "float16")     # MMA-compatible, swizzle inferred
    pool.commit()                                 # finalize the pool's size
    # pool.move_base_to(offset) rewinds the cursor to reuse space

TMEM pool（见下文 `Tensor memory`_）构建在 ``SMEMPool`` 之上。

Registers
---------

per-thread scratch 存在 register 中。用 ``T.alloc_local(shape, dtype)``
（也就是 ``scope="local"``）分配：它对每个线程私有，并 lower 成保存在
register 中的 local array。

.. code-block:: python

    r = T.alloc_local((4,), "float32")   # per-thread register array
    for k in T.unroll(4):
        r[k] = A[tx, k]
    # ... compute on r[0..3] ...

.. code-block:: c++

    alignas(64) float r_ptr[4];          // per-thread, register-resident
    r_ptr[0] = A_ptr[tx * 4 + 0];
    r_ptr[1] = A_ptr[tx * 4 + 1];
    // ...

.. note::

   ``alignas(64)`` 是 *默认* buffer alignment：buffer 的 ``data_alignment``
   默认是 ``runtime::kAllocAlignment``（64 bytes），CUDA codegen 会把它印到每个
   allocation 上，包括这种对齐没有意义的 per-thread ``local`` array。对于这些
   register-resident array，它 **没有性能影响**：带静态可解析索引的 thread-local
   array 会被 nvcc/ptxas 提升到 register（scalar replacement of aggregates，SROA），
   因此它从不进入可寻址 local memory，对齐就是 no-op。（如果动态索引 array
   spill 到 local memory，它确实会带上这种过度对齐，但那是不常见情况。）这种
   register local 的过度对齐是一个已知粗糙边角，我们计划修掉（对 ``local``
   scope 使用 dtype 的自然对齐）。

Scalar
~~~~~~

scalar 只是一个 **单元素** register array；严格来说，不需要单独概念。你可以
分配一个 size-1 的 ``local`` buffer，并用 ``[0]`` 索引：

.. code-block:: python

    phase = T.alloc_local((1,), "int32")   # 1-element register array
    phase[0] = 0
    while phase[0] < 4:
        acc = acc + A[tx, phase[0]]
        phase[0] += 1

但到处写 ``phase[0]`` 很笨重，所以 **scalar** 正是这件事的语法糖：一个可以
**按名字** 读写的单元素 register buffer：

.. code-block:: python

    phase: T.int32 = 0                 # mutable scalar (sugar for the above)
    while phase < 4:
        acc = acc + A[tx, phase]
        phase += 1

    s = T.local_scalar("int32")        # explicit form; assign by name (s = ..., not s[0])
    acc: T.float32 = 0.0               # a type-annotated assignment also makes one

二者不只是相似，而是会 parse 成 **结构完全相同的 TIRx**。这个语法糖完全在
parser 中消解：``phase: T.int32`` *就是* 那个单元素 ``local`` buffer，
``phase`` / ``phase += 1`` *就是* ``phase[0]`` / ``phase[0] += 1``。两个
kernel 上的 ``tvm.ir.assert_structural_equal`` 会通过，printer 甚至会把显式
``alloc_local`` + ``[0]`` 形式 **重新打印回** scalar 形式；因此 parsing 完成
后没有任何差别。二者都会 lower 成同一个
``alignas(64) int phase_ptr[1];``；scalar 只是让你省掉 ``[0]``。
（``T.local_scalar`` / ``T.shared_scalar`` / ``T.alloc_scalar`` 可以显式选择 scope。）

.. note::

   **为什么不用** ``Var``\ **？** TIRx ``Var`` 是 *immutable* 的：它是一个单次
   static binding（也就是下面的 ``T.let`` 产生的东西）。scalar 需要是
   *mutable* 的，因为你会在循环和 accumulator 中反复给它赋值；所以它必须
   由一个可重复 store 的单元素 buffer 支撑，而不是 ``Var``。

``let``
~~~~~~~

``T.let`` binding 是 **immutable** 的：一个单独的 ``LetStmt`` （有名字的值，
不是 buffer）。它适合派生常量：

.. code-block:: python

    n: T.let = M * K               # immutable binding (LetStmt)
    half: T.let[T.int32] = N // 2  # ... with an explicit type

它会 lower 成 **普通 scalar C 变量**，不是 buffer（没有 array，没有 ``[0]``）。
对于 ``half: T.let = m * 2``（其中 ``m`` 是运行时值）：

.. code-block:: c++

    int half = m * 2;     // the `let` -> a const-like local

因为值是 immutable 的，simplifier 可以自由传播并对它做 CSE，所以在使用点你
经常会直接看到 ``m * 2`` 被替换进去（或通过 common-subexpression 临时变量
共享），而不是引用 ``half``。

.. note::

   **为什么还需要 immutable binding？** 因为值不会变化，arithmetic analyzer 可以
   把 var 绑定到这个值上（化简 ``LetStmt`` 时调用
   ``analyzer.Bind(var, value)``），所以关于这个值证明出的事实——常量边界、
   modular set（可整除性 / 对齐）、范围——都会 **传播到每次使用**。这会反过来
   支持 index simplification、bounds-check elimination，以及 alignment/vectorization
   决策。*mutable* scalar 是一次 memory load（``buf[0]``）：analyzer 不能假设它
   保持不变，所以这些性质不会传递。``let`` 也是一个纯值：没有 allocation，
   可以自由 inline / substitute / CSE；而 scalar 是一个带 load/store 语义的
   单元素 buffer。

Tensor memory
-------------

Blackwell *tensor memory* 不是普通 scratch scope：它必须用 warp-uniform 的
``T.ptx.tcgen05.alloc`` / ``tcgen05.dealloc`` intrinsic 显式 reserve 和 free；
每个 tensor 都是其中的一个 view，通过
``T.decl_buffer(..., scope="tmem", allocated_addr=<column>, layout=<tmem layout>)``
声明。``allocated_addr``（列偏移）是必需的，tensor-core dispatch 会 assert 它；
因此 ``T.alloc_buffer(scope="tmem")``（不会设置它）不能工作。不同于 shared
memory，tensor memory 不能直接寻址：只能通过 ``tcgen05`` 的 ``mma`` / ``ld`` /
``st`` / ``cp`` 读写。

手写时，一个 warp 会把 allocation 发到一个 shared slot 中；你再把每个 tensor
``decl`` 成某个列偏移处的 view；最后由一个 warp 释放它：

.. code-block:: python

    addr = T.alloc_shared((1,), "uint32")             # slot for the allocated base
    if warp_id == alloc_warp:                         # tcgen05.alloc is warp-uniform
        T.ptx.tcgen05.alloc(T.address_of(addr), n_cols=512, cta_group=cta_group)
    acc = T.decl_buffer((CTA_M, 512), "float32", scope="tmem",
                        allocated_addr=0, layout=tmem_layout)   # view at column 0
    # ... use acc as a gemm_async / copy_async operand ...
    if warp_id == alloc_warp:
        T.ptx.tcgen05.relinquish_alloc_permit(cta_group=cta_group)
        T.ptx.tcgen05.dealloc(addr, n_cols=512, cta_group=cta_group)

列偏移和 ``tmem_layout`` （datapath D/F layout）由你自己管理。下面的 pool 发出
的正是这套序列。

Pool
~~~~

``T.TMEMPool`` 把这些全部封装起来：warp-uniform alloc/dealloc、列方向
bump-allocation，以及 datapath layout：

.. code-block:: python

    tmem_addr = pool.alloc((1,), "uint32")          # pool = the kernel's smem pool
    tmem_pool = T.TMEMPool(pool, total_cols=512, cta_group=cta_group,
                           tmem_addr=tmem_addr)
    acc = tmem_pool.alloc((CTA_M, 512), "float32")  # allocated_addr set for you
    tmem_pool.commit()                               # emits tcgen05.alloc (one warp)
    # ... use acc ...
    tmem_pool.dealloc()                              # emits tcgen05.dealloc (one warp)

完整示例见 Part III 的 GEMM kernel。

Buffer API
----------

``Buffer`` 是 pointer 上的 metadata（见上文 *声明 buffer*），因此它的大部分
方法都是 *compile-time* reshape/reinterpret：要么改变索引算术，要么把指针交给
你；它们本身不会发出 runtime op。常见方法如下：

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - 方法
     - 含义
   * - ``B.data``
     - 原始 data pointer（一个 ``Var``）；打印为 ``B_ptr``
   * - ``B.ptr_to([i, j])``
     - 指向某个元素的 typed pointer（``address_of``）；打印为 ``&B_ptr[…]``
   * - ``B.vload([i], dtype="float32x4")`` / ``B.vstore([i], v)``
     - vectorized load / store；打印为 ``*(float4*)(B_ptr + …)``
   * - ``B.view(*shape, layout=…)``
     - 以新的 shape/layout reinterpret 同一份存储（不 copy）
   * - ``B.local(*shape, layout=…)``
     - 调用线程在 ``local`` buffer 中的私有 register slice
   * - ``B.permute(*dims)``
     - 轴被 permute 后的 view（转置 layout）
   * - ``B.access_ptr(mask, …)``
     - masked access pointer（``tvm_access_ptr`` builtin），用于把 region 传给
       intrinsic

**Pointer — ``ptr_to`` / ``data``。** ``ptr_to`` 用来把元素地址交给 intrinsic
或 inline function；``data`` 是 base pointer：

.. code-block:: python

    B[tx] = T.cuda.func_call("ld", A.ptr_to([tx]), source_code=SRC, return_type="float32")

.. code-block:: c++

    B_ptr[tx] = ld(&A_ptr[tx]);          // ptr_to([tx]) -> &A_ptr[tx];  A.data -> A_ptr

**Vectorized access — ``vload`` / ``vstore``。** 把多个元素作为一次宽传输来
移动（另见 :doc:`data_types`）：

.. code-block:: python

    B.vstore([tx * 4], A.vload([tx * 4], dtype="float32x4"))

.. code-block:: c++

    *(float4*)(B_ptr + tx * 4) = *(float4*)(A_ptr + tx * 4);

**Reshape / reinterpret — ``view`` / ``permute``。** 二者都是纯 metadata；
data pointer 不变，只有索引算术不同。``A.view(64, 4)`` 会把 256 元素 buffer
看成 ``64×4``；``A.permute(1, 0)`` 会转置轴：

.. code-block:: python

    A2 = A.view(64, 4);     y = A2[tx, 0] + A2[tx, 3]   # A2[tx, j] -> A_ptr[tx*4 + j]
    At = A.permute(1, 0);   z = At[i, j]                # At[i, j]  -> A_ptr[j*4 + i]

.. code-block:: c++

    A2_ptr[tx * 4]  /* +3 */                 // view: row-major 64x4 index
    At_ptr[(j * 4) + i]                       // permute: swapped strides

**Register — ``local``。** 把 thread-axis ``local`` layout 分解成调用线程的
flat register bundle（tile primitive 中大量使用）：

.. code-block:: python

    R  = T.alloc_buffer((32, 8), "float32", scope="local", layout=TileLayout(S[(32, 8) : (1 @ laneid, 1)]))
    Rl = R.local(8)          # this lane's 8 registers

.. code-block:: c++

    alignas(64) float Rl_ptr[8];             // the lane's private registers
