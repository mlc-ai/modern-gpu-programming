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

.. _zh_chap_language_reference:

TIRx 语言参考
==============

这里收录编写 TIRx device kernel 所需的完整语言特性，并从
:ref:`zh_chap_tirx_primer` walkthrough 中拆分出来：parser 工具、数据类型与表达式、
buffer 与内存、控制流，以及线程同步。当你需要确认某个特性的精确写法或语义时，
可以查阅这些页面。

.. toctree::
   :maxdepth: 1

   cuda/parser_utils
   cuda/data_types
   cuda/buffers
   cuda/control_flow
   cuda/threads_sync
