(zh_chap_clc)=
# 进阶：Cluster Launch Control

:::{admonition} 概览
:class: overview

- persistent kernel 会保持一组固定 CTA 或 CTA cluster 驻留（通常让规模大致达到每个 SM 一个活跃 work owner，但不依赖保证的 1:1 映射），并让它们循环处理许多 output tile，而不是每个 tile 启动一个 CTA。
- Cluster Launch Control 是 Blackwell 的硬件机制，允许驻留 cluster 在运行时请求另一个 tile。它是一条围绕两条 PTX 指令构建的硬件 work-stealing 路径：一条指令请求工作，另一条读回请求是否成功。
- 主要收益是更好的 tail behavior。当 tile 成本不均，或者 tile 数量不能均匀分配到可用 SM 时，提前完成的 CTA 可以拉取更多工作，而不是闲置。
:::

persistent GEMM 不会把 CUDA grid 当成固定的“每个 output tile 一个 CTA”的 launch。
相反，它启动一组更小的、长生命周期的 CTA 或 CTA cluster。每一个计算一个 tile，前进到另一个 tile，再次计算，
并持续执行，直到输出空间完成。这正是 {ref}`zh_chap_gemm_advanced` 中逐步构建的执行模式。

一旦 kernel 是 persistent 的，主要调度问题就变得很简单：当一个 CTA 或 cluster 完成当前 tile 后，下一个 tile 从哪里来？

最简单的答案是静态公式。例如，kernel 可以从 CTA id 计算 tile coordinate，然后按 grid stride 前进。
这很容易实现，并且当所有 tile 成本大致相同、tile 数量能均匀分布到 GPU 上时效果很好。
但 schedule 是在实际工作运行前决定的。如果少数 tile 花费更久，或者最后几个 tile 分配不均，
有些 SM 会提前完成自己的份额，而另一些仍在处理 tail。

Cluster Launch Control，即 CLC，会改变这个调度模型。persistent cluster 不再预先决定整个 assignment，
而是可以向硬件 grid scheduler 请求另一个尚未 launch 的 cluster 的工作。如果请求成功，当前 cluster 接管那个 cluster coordinate，
并计算对应 tile。如果请求失败，就没有更多工作可偷，loop 退出。

这与 thread block cluster 本身不是一回事。thread block cluster（一起 launch 的 CTA，具有 cluster-level synchronization，
并能访问 distributed shared memory）是在 Hopper 中引入的（{ref}`zh_chap_background`）。
CLC 是 Blackwell 增加的机制，让这些 cluster coordinate 上的调度变为动态。
cluster 已经是 launch 单位；CLC 让已经运行的 cluster 可以取消一个 pending launch，并继承它的坐标。

## 两条指令

Cluster Launch Control 通过两条 PTX 指令暴露。第一条指令向 grid scheduler 发送异步请求，第二条指令读取响应。

请求指令是 `clusterlaunchcontrol.try_cancel.async`。

`try_cancel` 会要求 scheduler 取消一个 pending cluster 的 launch，并把该 cluster 的坐标返回给调用者。
响应会作为 16-byte record 写入 shared memory。由于请求是异步的，指令不会等待响应到达。
completion 会通过 `mbarrier` 报告，使用与 TMA 相同的 barrier-and-phase 模型。

这是一个重要细节，因为它意味着 CLC 没有引入新的等待模型。kernel 发射请求，把它关联到一个 barrier，
随后在读取响应前等待 barrier。响应到达通过带 byte-count completion 的 barrier signal，
整体风格与其他异步硬件操作相同（见 {ref}`zh_chap_async_barriers`）。

一旦 barrier 触发，kernel 使用 query 指令。

第一个 query 是 `clusterlaunchcontrol.query_cancel.is_canceled`。它返回一个 predicate，告诉 kernel cancel 是否成功。
predicate 为 true 表示 scheduler 找到了一个 pending cluster launch、取消了它，并返回了其坐标。
predicate 为 false 表示没有剩余 pending work 可取。

只有当 `is_canceled` 为 true 时，kernel 才应该读取 coordinate。它通过
`clusterlaunchcontrol.query_cancel.get_first_ctaid` 完成这件事，该指令提取被取消 cluster 的第一个 CTA id。
这个 CTA id 是 coordinate vector，通常读作 `(x, y, z)`，kernel 会把它 decode 成接下来应该计算的 output tile。

这个协议里没有数值形式的 sentinel tile id。kernel 根据 predicate 分支。如果 predicate 为 true，coordinate 有效。
如果 predicate 为 false，work-stealing loop 结束。

在底层，这个形状直接来自 CLC 正在做的事。硬件不是从软件队列中分配一个抽象 task；
它是在取消一个尚未发生的 cluster launch。因此，成功响应包含一个真实 cluster coordinate；
失败响应只是表示 launch queue 已经耗尽。

## Work-Stealing Loop

有了这两条指令，persistent scheduler 就变成一个短 loop。

在 loop 的任意时刻，cluster 都有一个自己负责计算的 tile。在开始这个 tile 之前，它会为下一个 tile 发送 `try_cancel` 请求。
请求异步运行。当 scheduler 处理这个请求时，cluster 计算自己的当前 tile。

当前 tile 完成后，cluster 会等待与 `try_cancel` 响应关联的 `mbarrier`。
响应 ready 后，它调用 `query_cancel.is_canceled`。如果 predicate 为 true，它调用 `query_cancel.get_first_ctaid`，
decode 返回的 coordinate，并把它作为下一个 tile。如果 predicate 为 false，就没有剩余工作，cluster 退出。

代码形态上，这个 loop 是：

1. 为可能的下一个 tile 发射 `try_cancel`；
2. 在请求 in flight 时计算当前 tile；
3. 等待 response barrier；
4. 查询 cancellation 是否成功；
5. 要么用返回的 coordinate 继续，要么退出。

请求的位置正是这个 loop 有用的原因。cluster 不会等当前 tile 完成后才请求更多工作。
它先请求，再计算。这样就把 scheduler request 与有用工作 overlap 起来。
当当前 tile 完成时，下一个 tile 的答案往往已经可用。

这与 persistent kernel 在其他地方使用异步 copy 和 tensor-core barrier 的基本原因相同：
kernel 避免把长延迟操作直接放到 critical path 上。CLC 把同样想法应用到 tile scheduling：
提前请求下一份工作，计算当前工作，然后在需要时消费调度结果。

## 与 Persistent GEMM 的关系

{ref}`zh_chap_gemm_advanced` 中的 persistent GEMM 在主线讲解中使用 static scheduler。
static scheduler 更容易解释，因为下一个 tile 可以直接从 loop state 计算出来。
例如，像 `ClusterPersistentScheduler2D` 这样的 scheduler 可以在 output tile space 上用 grid-stride pattern 分配 tile。

CLC 是这种 static assignment 的动态替代。outer loop 保持不变：每个 resident cluster 反复计算一个 output tile，
然后前进到另一个 tile。变化的是下一个 tile 从哪里来。使用 static scheduler 时，下一个 tile 由公式计算。
使用 CLC 时，下一个 tile 由硬件 work stealing 返回。

这种差异在 launch tail 附近最重要。在 static schedule 中，剩余工作可能分布不均。
有些 SM 可能已经耗尽 assigned tile，而其他 SM 仍有几个 tile。使用 CLC 时，提前完成的 cluster 会请求另一个 pending cluster coordinate。
只要 launch queue 中还有工作，提前完成者就会继续拉取更多 tile。

当 tile cost 不均匀时，这也很重要。一些 GEMM tile 可能因为边界、masking、sparsity、grouped scheduling，
或主矩阵乘法周围的 fused work 而走不同路径。static schedule 在观察到这些成本之前，就假设 tile assignment 足够好。
CLC 不需要这个假设。它只在某个 cluster 变得可用之后，才分配更多工作。

因此，在 TIRx 中，CLC 可以暴露为 dynamic tile scheduler。编程模型不需要改变 tile 的计算。
tile body 仍然是 static scheduler 使用的同一个 persistent GEMM body。scheduler 从“用公式计算我的下一个 tile coordinate”
变成“向硬件请求下一个可用 cluster coordinate”。结果仍然是同一个 persistent loop，
但工作分布由硬件驱动，而不是由固定 launch-time schedule 决定。
