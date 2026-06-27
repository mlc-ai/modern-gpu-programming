---
orphan: true
---

# 面向 MLSys 的现代 GPU 编程

机器学习系统位于现代 AI 工作负载的核心。在这些系统中，性能往往取决于少数几个
GPU kernel 的质量。注意力 kernel、LLM prefill 和 decode kernel、低精度块缩放
GEMM、融合 MoE 层，以及其他大型融合 kernel，都会直接影响训练和服务中的端到端速度。

然而，要让这些 kernel 跑得快，仅有一串优化技巧还不够。现代 GPU 已经不再只是旧式设计的
简单变体。近年的架构引入了更丰富的内存空间、新的访问模式，以及越来越专门化的执行单元。
要写好它们，我们既需要对硬件形成清晰的心智模型，也需要实际理解高性能 kernel 是如何构建出来的。
本书的目标正是同时培养这两种能力。

本书遵循一条简单的路线：先理解 GPU 硬件，再学习我们将使用的编程模型，最后一步步构建
state-of-the-art 的 kernel。我们的主要目标是 Blackwell 这一代 GPU，贯穿全书的主要例子是
高速矩阵乘法（GEMM）和 FlashAttention。在这个过程中，我们也会研究 GPU 优化背后的核心要素：
数据布局、异步数据移动和异步协同。

这些材料源自卡内基梅隆大学的 [Machine Learning Systems](https://mlsyscourse.org/) 课程系列。
为了让这些思想更容易学习、也更容易运行，本书使用 **TIRx** Python DSL，一步步构建真实的
GPU kernel 示例。TIRx 贴近硬件，因此我们既能通过可运行代码学习，又能推理底层控制细节。

## 本书结构

- **第一部分，理解 GPU。** 本部分介绍 GPU 的整体组织方式、编写高速 kernel 的通用方法，以及
  数据布局、异步内存操作和协同等关键概念。它建立了后续章节都依赖的硬件直觉。
- **第二部分，TIRx 概览。** 本部分介绍 TIRx 的关键组成，它们是全书代码示例的基础。
- **第三部分，GEMM：从分块到 SOTA。** 这是优化 tiled GEMM 的完整指南，逐步引入
  TMA 流水线、持久化调度、warp specialization 和 2-CTA cluster。
- **第四部分，Flash Attention 4。** 使用第三部分技术构建完整的注意力 kernel：两个 MMA，
  中间插入 softmax，包含 online-softmax rescaling、causal masking 和 GQA。
- **参考。** TIRx 语言参考和编译器内部机制。

```{toctree}
:caption: 第一部分，理解 GPU
:maxdepth: 1

chapter_background/index
chapter_performance/index
chapter_data_layout/index
chapter_layout_generations/index
chapter_tma/index
chapter_tensor_cores/index
chapter_tmem/index
chapter_async_barriers/index
chapter_clc/index
```

```{toctree}
:caption: 第二部分，TIRx 概览
:maxdepth: 1

chapter_intro_tirx/index
chapter_tirx_layout_api/index
```

```{toctree}
:caption: "第三部分，GEMM：从分块到 SOTA"
:maxdepth: 2

chapter_gemm_basics/index
chapter_gemm_async/index
chapter_gemm_advanced/index
```

```{toctree}
:caption: 第四部分，Flash Attention 4
:maxdepth: 2

chapter_flash_attention/index
```

```{toctree}
:caption: 参考
:maxdepth: 1

appendix/index
appendix/debugging_warp_specialized
tirx_guide/arch/index
tirx_guide/language_reference/index
```
