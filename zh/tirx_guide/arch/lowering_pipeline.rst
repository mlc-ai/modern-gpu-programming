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

TIRx lowering pipeline
======================

``tvm.compile(mod, target, tir_pipeline="tirx")`` 会把你写好的 TIRx module
送过 **tirx pipeline**。这是一串有序的 TIR pass，会把你写的高层结构
（tile primitive、带 ``TileLayout`` 类型的 buffer、execution-scope id）转换成
拆分后的 **host** + **device** 函数，然后由 CUDA backend 渲染成源码。
pipeline 定义在 ``python/tvm/tirx/compilation_pipeline.py``（``tirx_pipeline``）
中；本页按顺序走过这些 pass。

它所在的位置
------------

``tvm.compile`` 会先绑定 target，运行 **tirx pipeline**（下面这些 module-level
pass），然后分别对 host 和 device 函数应用 **finalization** pass，最后把每个
device 函数交给 CUDA code generator：

.. code-block:: text

    authored TIRx  ──BindTarget──▶  tirx_pipeline  ──▶  host func  ──host finalize──▶  C/LLVM
                                          │
                                          └──────────▶  device func ──device finalize──▶  CUDA

Pass 列表
---------

``tirx_pipeline`` module pass 会应用下面这个精确顺序（其中少数 pass 受
``PassContext`` config 控制）：

.. list-table::
   :header-rows: 1
   :widths: 6 32 62

   * - #
     - Pass
     - 作用
   * - 1
     - ``LowerTIRx``
     - 核心 lowering，见下文 `Inside LowerTIRx`_
   * - 2
     - ``UnifyThreadBinding``
     - 合并等价的 thread-axis binding，让每个 ``threadIdx`` / ``blockIdx``
       轴只声明一次
   * - 3
     - ``StmtSimplify``
     - 语句级算术化简（arith analyzer）
   * - 4
     - ``LowerTIRxOpaque``
     - 将剩余 opaque TIRx construct lower 成普通 TIR
   * - 5
     - ``FlattenBuffer``
     - 把多维 ``BufferLoad`` / ``BufferStore`` flatten 成 1-D
   * - 6
     - ``BF16ComputeLegalize``
     - 把 ``bfloat16`` compute 重写成合法形式（上转为 f32）
   * - 7
     - ``NarrowDataType(32)``
     - 在可证明安全时，把 index/loop ``PrimExpr`` dtype 缩窄到 32-bit
   * - 8
     - ``VectorizeLoop``
     - 把 ``T.vectorized`` loop 转成 vector op（若设置 ``tir.disable_vectorize``
       则跳过）
   * - 9
     - ``UnrollLoop``
     - 展开标记为 ``T.unroll`` 的 loop（以及小的常量 loop）
   * - 10
     - ``StmtSimplify``
     - 再次化简，因为 vectorize/unroll 暴露了常量
   * - 11
     - ``CommonSubexprElim``
     - 把重复子表达式 hoist 成临时变量（若设置 ``tir.disable_cse_tir``
       则跳过）
   * - 12
     - ``FP8ComputeLegalize``
     - 把 ``float8`` compute 重写成合法形式
   * - 13
     - ``VerifyMemory``
     - 检查 host-side 代码不会直接解引用 device memory（安全闸门）
   * - 14
     - ``AnnotateEntryFunc``
     - 将单个 PrimFunc 标记为 module entry point
   * - 15
     - ``SplitHostDevice``
     - 在 ``launch_thread`` 边界处，把每个 kernel 拆成 **host** 函数和
       **device** 函数
   * - 16
     - ``MakePackedAPI``
     - 将 host 函数重写成 packed-func ABI（TVM launcher 调用的形式）
   * - 17
     - ``FP8StorageLegalize``
     - legalize ``float8`` storage（打包进受支持的 container type）
   * - 18
     - ``BF16StorageLegalize``
     - legalize ``bfloat16`` storage

随后 **Finalization** 会按函数类型运行：

- **host**：``LowerTVMBuiltin``（lower ``tvm_*`` builtin）、``LowerIntrin``
  （target-specific intrinsic）
- **device**：``LowerWarpMemory``（warp-scoped buffer → shuffle）、``StmtSimplify``、
  ``LowerIntrin``

Inside LowerTIRx
----------------

``LowerTIRx`` 本身也是一个小序列（``src/tirx/transform/lower_tirx.cc``）：

.. code-block:: text

    LowerTIRx = Sequential([ TilePrimitiveDispatch, LowerTIRxCleanup ])

- **``TilePrimitiveDispatch``** 会把每个 ``TilePrimitiveCall``（``copy``、
  ``gemm``、``reduction`` 等）替换成所选 backend dispatch 发出的 body，
  也就是它的 variant-selection 和 codegen。
- **``LowerTIRxCleanup``** 会运行 ``LayoutApplier``：把每个带
  ``TileLayout`` 类型的 buffer access 解析成具体物理地址算术
  （``addr = data + elem_offset + layout.apply(coord)``），flatten buffer，并
  lower execution-scope id（``T.cta_id`` / ``T.thread_id`` / … 通过
  ``launch_thread`` 变成 ``blockIdx`` / ``threadIdx``）。

因此经过 ``LowerTIRx`` 后，module 就是普通 TIR：不再有 tile primitive，
不再有 ``TileLayout`` 间接层，scope id 也解析成了 thread axis。

一个完整例子
------------

来看一个一行 scale kernel：

.. code-block:: python

    @T.prim_func
    def scale(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        B = T.match_buffer(B_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([256])
        B[tx] = A[tx] * T.float32(2.0)

**经过 ``LowerTIRx`` 后**，scope id 已经是真实 thread axis，layout 也已经应用
（``A_1`` / ``B_1`` 是 flattened 1-D view）：

.. code-block:: python

    with T.launch_thread("blockIdx.x", 1) as blockIdx_x:
        threadIdx_x = T.launch_thread("threadIdx.x", 256)
        bx: T.let = blockIdx_x
        tx: T.let = threadIdx_x
        B_1[threadIdx_x] = A_1[threadIdx_x] * T.float32(2.0)

**经过 ``SplitHostDevice`` + ``MakePackedAPI`` 后**，一个函数变成两个：
一个 host launcher 和一个 device kernel：

.. code-block:: python

    @I.ir_module
    class Module:
        def main(...):          # host: packed-API launcher (computes the grid/block, launches)
            ...
        def scale_kernel(...):  # device: the __global__ body, run on the GPU

随后 CUDA backend 会把 ``scale_kernel`` 渲染成 ``__global__`` 函数
（``B_ptr[threadIdx.x] = A_ptr[threadIdx.x] * 2.0f``）。

自己复现
--------

你可以手动运行 pipeline 的任意前缀来检查某个阶段；这些文档中的 IR snippet
就是这样生成的：

.. code-block:: python

    from tvm.tirx import transform as TT

    target = tvm.target.Target("cuda")
    mod = TT.BindTarget(target.with_host("llvm"))(tvm.IRModule({"main": scale}))
    mod = TT.LowerTIRx()(mod)         # tile primitives dispatched, layouts applied
    print(mod.script())               # inspect the lowered TIRx IR

或者编译整个 module，然后读取生成的 CUDA：

.. code-block:: python

    exe = tvm.compile(tvm.IRModule({"main": scale}), target=target, tir_pipeline="tirx")
    print(exe.mod.imports[0].inspect_source())
