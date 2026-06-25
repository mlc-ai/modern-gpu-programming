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

数据类型与表达式
================

每个 TIRx 表达式都同时携带一个低层 **dtype** 和一个高层 **type**。

表达式 dtype
------------

``PrimExpr`` 的 ``.dtype`` 是它的标量（或向量）元素类型，例如 ``float32``、
``float16``、``bfloat16``、``int32``、``uint8``、``bool``、低精度的
``float8_e4m3fn`` / ``float4_e2m1fn``、``handle``（指针），以及
``float32x4`` 这类向量形式。每种 dtype 都会打印成对应的 CUDA 类型。下面
示例展示跨多种 dtype 分配 local/shared buffer，以及一次 vectorized
``float32x4`` load/store：

.. code-block:: python

    @T.prim_func
    def dtypes(A_ptr: T.handle, O_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), "float32")
        O = T.match_buffer(O_ptr, (256,), "float32")
        T.device_entry(); bx = T.cta_id([1]); tx = T.thread_id([64])
        f16  = T.alloc_local((1,), "float16")        # register scalars ...
        bf16 = T.alloc_local((1,), "bfloat16")
        i32  = T.alloc_local((1,), "int32")
        u8   = T.alloc_local((1,), "uint8")
        b1   = T.alloc_local((1,), "bool")
        sm   = T.alloc_shared((64,), "float16")      # ... and a shared tile
        v    = T.alloc_local((1,), "float32x4")      # a vector-dtype register (float4)
        v[0] = A.vload([tx * 4], dtype="float32x4")  # vectorized load
        O.vstore([tx * 4], v[0])                     # vectorized store
        # ... (use f16/bf16/i32/u8/b1/sm) ...

会 lower 成下面的代码（省略生成 CUDA 的样板部分）：

.. code-block:: c++

    half          f16_ptr[1];               // float16
    nv_bfloat16   bf16_ptr[1];              // bfloat16
    int           i32_ptr[1];               // int32
    uchar         u8_ptr[1];                // uint8
    signed char   b1_ptr[1];                // bool
    __shared__ alignas(64) half sm_ptr[64]; // shared float16
    float4        v_ptr[1];                 // float32x4  (vector)
    v_ptr[0]                  = *(float4*)(A_ptr + tx * 4);   // vectorized load
    *(float4*)(O_ptr + tx * 4) = v_ptr[0];                   // vectorized store

buffer 的 dtype 本身也可以是 **vector type**：``T.alloc_local((1,), "float32x4")``
会直接声明一个 ``float4`` register（用 ``v[0]`` 访问），而 ``float32x4``
的 ``vload`` / ``vstore`` 会把它作为一次 16-byte 访问来搬运。vector dtype
并不绑定在 ``vload`` 上；任意 buffer 或 scalar 都可以携带它。

因此 dtype → CUDA 的映射如下：

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - dtype → CUDA
     - dtype → CUDA
     - dtype → CUDA
   * - ``float32`` → ``float``
     - ``float16`` → ``half``
     - ``bfloat16`` → ``nv_bfloat16``
   * - ``int32`` → ``int``
     - ``uint8`` → ``uchar``
     - ``bool`` → ``signed char``
   * - ``float32x4`` → ``float4``
     - ``handle`` → ``T*`` (pointer)
     - (vector dtypes → CUDA vector types)

dtype 与 type
----------------

``dtype`` 是 *低层* 信息，说明“这些 bit 如何解释”。此外，值还拥有高层
**type**：标量是 ``PrimType(dtype)``，指针是
``PointerType(PrimType(dtype), scope)``。大多数表达式都是标量
（``PrimType``）；类型系统主要在 **指针** 上变得重要。

指针（``handle``）
------------------

buffer 的 ``data``，也就是它的指针，是一个 pointer type 的 ``Var``，并且
它是 **immutable** 的（指针不会被重新赋值）。这决定了你如何获得它：

- ``T.alloc_buffer(...)`` 会分配存储空间，**并** 定义它的 ``data`` 指针。
- ``T.decl_buffer(..., data=ptr)`` 会在已有指针 ``Var`` ``ptr`` 上声明一个
  buffer。
- 如果要让 buffer 背后使用一个指针 **表达式**，例如 ``T.ptx.map_shared_rank``
  （PTX ``mapa``）返回另一个 cluster CTA 的 shared address，你必须先用
  ``PointerType`` 的 ``T.let`` 把该表达式绑定成一个指针 ``Var``（``data``
  必须是 ``Var``，不能是表达式）：

  .. code-block:: python

      from tvm.ir.type import PointerType, PrimType

      ptr: T.let[T.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = \
          T.reinterpret("handle", T.ptx.map_shared_rank(mbar.ptr_to([0]), 0))
      remote_mbar = T.decl_buffer([1], "uint64", data=ptr, scope="shared")
