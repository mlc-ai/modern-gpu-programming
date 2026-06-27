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

Parser 工具
===========

有几个 helper 会在 **parse time** 生效（也就是 TVMScript 转成 TIRx 时），
让你内联 Python 计算出的值、抽出可复用片段，并打包 parser 侧状态。

``T.meta_var`` — 内联 Python 值
----------------------------------------

``T.meta_var(x)`` 会告诉 parser：把 ``x`` 这个由 **Python** 计算出的值当作
compile-time *meta* 值，直接内联进 IR，而不是把它解析成 script 变量。这样
可以避免一个临时 local，也能驱动 metaprogramming：围绕 meta value 的普通
Python ``for`` 会在 parser 中展开。

.. code-block:: python

    n = T.meta_var(4)              # n is a Python int, inlined
    for j in range(n):            # unrolled at parse time
        acc[0] = acc[0] + A[tx, j]

``@T.inline`` — 内联函数
--------------------------------

``@T.inline`` 定义的函数会在 parsing 期间把函数体 **内联到每个调用点**，
生成代码中不会出现调用。它遵循 Python 的 lexical（LEGB）scope 和 late
binding，因此参数会遮蔽外层变量：

.. code-block:: python

    @T.inline
    def add_into(acc, x):
        acc[0] = acc[0] + x

    add_into(acc, A[tx, j])       # inlined -> acc[0] = acc[0] + A[tx, j]

``@T.meta_class`` — parser 侧状态对象
---------------------------------------------

``@T.meta_class`` 标记一个普通 Python class，使其 **实例成为 parser meta
value**：字段可以持有 buffer 和 scalar，因此你可以把相关 allocation 与状态
打包进一个对象，并在 kernel body 中使用它。

.. code-block:: python

    @T.meta_class
    class State:
        def __init__(self, smem):
            self.acc = T.alloc_local([1], "float32")
            self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")

    s = State(smem.data)
    s.acc[0] = T.float32(0.0)     # use its fields like ordinary buffers
    # ... s.buf[i] ...

这很适合把 kernel 的 pipeline state（barrier、accumulator、scratch view）
组织到一起，而不是让许多分散的 local 穿过整个 body。

``T.constexpr``
---------------

``T.constexpr`` 标记 compile-time kernel 参数，它会通过 ``@T.jit`` 的
``.specialize(...)`` 固化进去。细节见 :ref:`zh_chap_tirx_primer`。
