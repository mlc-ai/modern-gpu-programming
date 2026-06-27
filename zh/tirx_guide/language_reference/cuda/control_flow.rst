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

控制流
======

控制流包括 ``if``、循环家族和 ``while``；它们都会映射到直观对应的
CUDA 结构。

if
--

Python 的 ``if`` / ``else`` 会变成 CUDA 的 ``if`` / ``else``。可以用
thread/lane 比较来保护某段工作，也可以用 ``T.ptx.elect_sync()`` 选出
一个发出指令的线程：

.. code-block:: python

    if tx < 128:
        A[tx] = A[tx] * T.float32(2.0)
    else:
        A[tx] = A[tx] + T.float32(1.0)

    if T.ptx.elect_sync():
        ...                              # one elected lane (e.g. to issue TMA/MMA)

.. code-block:: c++

    if (((int)threadIdx.x) < 128) {
      A_ptr[tx] = A_ptr[tx] * 2.0f;
    } else {
      A_ptr[tx] = A_ptr[tx] + 1.0f;
    }

如果只是表达式级选择（不产生分支），使用 ``T.if_then_else(cond, a, b)``。
它会 lower 成三元表达式，因此不会引入 control-flow divergence：

.. code-block:: c++

    O_ptr[tx] = (A_ptr[tx] > 0.0f) ? A_ptr[tx] : 0.0f;

Uniform 与 divergent 控制流
---------------------------

像 ``if tx < 128`` 这样的 per-thread guard 对普通工作没有问题，但
**collective** 操作必须被它要同步的所有线程 *一致地* 到达。

例如，``T.cuda.cta_sync()`` 会映射到 ``__syncthreads()``，它要求 thread block
中的所有线程都到达。它绝不能放在 thread-divergent 或 warpgroup-divergent
分支里：如果放进 ``if wg_id == 0:``，其他 warpgroup 永远不会到达，kernel
就会 deadlock。若只需要同步一个 warpgroup，请使用 warpgroup-scoped
``T.cuda.warpgroup_sync(id)`` （见 :ref:`zh_chap_gemm_advanced` 和
:doc:`threads_sync`）。

barrier 初始化也要同样小心。``mbarrier`` 的 ``.init()`` 会 lower 成一个
single-thread guard（``if (threadIdx.x < 1)``）。如果再把它嵌进另一个
divergent 分支，barrier 可能保持未初始化，导致未定义的 launch failure。

loop
----

循环有四种形式；普通 Python ``range`` 会变成 ``T.serial``：

- ``T.serial(n)`` — 顺序循环（ptxas 仍可能 unroll 它）。
- ``T.unroll(n)`` — 完全 unroll（展开成直线语句）。
- ``T.vectorized(n)`` — vectorized loop。
- ``T.grid(*extents)`` — 嵌套循环。

循环内部可以使用 ``break`` / ``continue``。

.. code-block:: python

    for i, j in T.grid(8, 8):
        B[i, j] = T.max(A[i, j], T.float32(0.0))

.. code-block:: c++

    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 8; ++j)
        B_ptr[i * 8 + j] = max(A_ptr[i * 8 + j], 0.0f);

``T.unroll(4)`` 则会展开成四条直线语句，不再保留循环。

while
-----

``while`` 循环会一直运行到条件为 false。请使用 mutable scalar counter
（见 :doc:`buffers`）：

.. code-block:: python

    i: T.int32 = 0
    while i < 64:
        A[i] = A[i] + T.float32(1.0)
        i += 1

它会 lower 成一个带 early-exit ``break`` 的 ``while (1)`` （counter 是一个
单元素 register buffer）：

.. code-block:: c++

    int i_ptr[1];
    i_ptr[0] = 0;
    while (1) {
      if (!(i_ptr[0] < 64)) { break; }
      A_ptr[i_ptr[0]] = A_ptr[i_ptr[0]] + 1.0f;
      i_ptr[0] = i_ptr[0] + 1;
    }
