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

.. _chap_tirx_primer:

TIRx Basics: Native Level
=========================

.. note::
   This part assumes a working TIRx install (``from tvm.script import tirx as T``); see the
   project README for current setup.

What "native level" means
-------------------------

A native-level TIRx kernel reads like a structured device kernel: you place
threads yourself, allocate shared/register buffers, write loops and barriers, and
call device intrinsics directly. There is no automatic scheduling — what you write
is what is emitted. It is the foundation the higher-level tile-primitive abstractions (used in the
GEMM and Flash Attention kernels of Parts III–IV) lower to, and where you go when a hardware
feature has no higher-level primitive yet.

The shared model
----------------

The authoring model is the same across backends:

- ``@T.prim_func`` (or ``@T.jit`` for compile-time-specialized) kernels, written
  with ``from tvm.script import tirx as T``;
- ``T.device_entry()`` plus *scope-id* intrinsics for thread binding;
- ``T.match_buffer`` parameters and ``T.alloc_*`` scratch buffers;
- ordinary loops, branches, and scalar math;
- ``tvm.compile(mod, target=..., tir_pipeline="tirx")`` to build, then call the
  result directly.

What differs per backend is the concrete set of memory scopes, synchronization
and device intrinsics, and the generated source. Pick your backend:

.. toctree::
   :maxdepth: 1

   native_basics/cuda

(Support for additional backends, e.g. ROCm, will appear here as it lands.)
