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

CUDA C++/PTX
============

.. note::

   Native-level kernel authoring for the **CUDA backend** (the ``"cuda"``
   target) — the thread hierarchy, memory scopes, the ``T.cuda.*`` / ``T.ptx.*``
   intrinsics, and the compile / run / inspect loop. For what "native level"
   means in general, see :doc:`../native_basics`. The complete kernels in these
   chapters (``scale``, ``add``, ``smem_demo``, ``block_sum``, and the warp
   all-reduce) are tested end-to-end on a CUDA GPU.

All native authoring uses these imports. The ``__future__`` import lets ``@T.jit``
kernels reference compile-time parameters inside type annotations (see
:doc:`cuda/functions`); it is harmless for ordinary kernels::

    from __future__ import annotations
    import tvm
    from tvm.script import tirx as T

.. toctree::
   :maxdepth: 1

   cuda/first_kernel
   cuda/functions
   cuda/parser_utils
   cuda/data_types
   cuda/buffers
   cuda/control_flow
   cuda/threads_sync
   cuda/compiling
