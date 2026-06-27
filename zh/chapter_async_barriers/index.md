(zh_chap_async_barriers)=
# 异步协同：mbarrier

:::{admonition} 概览
:class: overview

- TMA 和 Tensor Core 都是异步的，因此发射工作并不等于完成工作，consumer 需要显式 completion signal。
- mbarrier 就是这个 signal：producer arrive，consumer wait，它会追踪 arrival count，以及（对 TMA 而言）byte count。
- 每个 barrier 都携带一个 *phase*，每一轮都会翻转；等待正确 phase，才能安全地门控 consumer。
:::

TMA（{ref}`zh_chap_tma`）和 Tensor Core（{ref}`zh_chap_tensor_cores`）操作是异步的。
当 kernel 发射 TMA load 或 `tcgen05` MMA 时，issuing thread 不会等待操作完成。
指令只是被提交给硬件引擎；实际的数据移动或矩阵操作会与程序其余部分并行继续。

这很有用，因为它让内存移动和计算可以 overlap。但这也意味着 program order 不足以证明数据已经 ready。
后续指令可能在较早的异步操作完成前运行。如果 TMA 仍在写 shared-memory tile 时 MMA 就开始读取，
MMA 会读到不完整数据。如果 epilogue 在 Tensor Core 完成 accumulator 写入前读取 TMEM，它会读到错误值。
如果 kernel 等待了错误条件，它可能永远无法前进。

因此，kernel 在每个异步 handoff 处都需要显式 completion signal。`mbarrier` 就是这个 signal。
producer 在自己的工作完成时 arrive 到 barrier；consumer 在使用产出的数据前 wait 这个 barrier。
同一机制用于 TMA-to-MMA handoff、MMA-to-epilogue handoff，以及跨 pipeline stage 的 buffer reuse。

barrier 不只是一个 one-shot flag。它携带 phase bit，而这个 phase bit 会在 barrier 完成一轮 arrival 后改变。
phase 让一个 barrier 可以跨许多 loop iteration 复用，而不会把一次 iteration 的 completion 与另一次混淆。

## The mbarrier

`mbarrier` 是 memory barrier 的缩写，是存储在 shared memory 中的硬件同步对象。
概念上，它包含两份状态：arrival counter 和 phase bit。counter 告诉 barrier 当前轮还缺多少个 arrival。
phase bit 告诉 kernel 这个 barrier 当前处在哪一轮。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_mechanism.html" title="mbarrier data structure and APIs" loading="lazy"
        style="width:100%; min-width:1320px; height:620px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互：一个 `mbarrier` 状态视图，展示 arrival counter、phase bit，以及 `init`、`arrive` 和 `wait` 操作；点击字段可以聚焦。*

barrier 从初始化开始。在 `init` 期间，kernel 设置这个 barrier 应该期望多少个 arrival。
barrier 从 phase 0 开始，counter 被加载为预期 arrival count。从那时起，barrier 就在等待所有必需的 producer
或某个资源的使用者报告自己已经完成。

arrival 会减少 barrier 仍在等待的工作量。kernel 的不同部分可以用不同方式 arrive 到 barrier，而这种区别很重要。

对于 TMA load，常见 arrival path 是 tx-count arrival。像 `mbarrier.arrive.expect_tx(bytes)` 这样的操作会做两件事：
第一，它算作 issuing thread 在 barrier 上的 arrival。第二，它记录 TMA engine 预计传输的字节数。
barrier 不会仅仅因为 issuing thread 已经 arrive 就完成。它还会等待 TMA engine 随着传输结束把 byte count drain 掉。
只有两个条件都满足时，phase 才会翻转：普通 arrival count 到达零，pending tx byte count 也到达零。

这就是为什么不应该把 `expect_tx` 理解为“又一个普通 arrival”。它为异步 copy 设置 byte budget。
硬件稍后通过 complete-tx update 记账实际 copy completion。只有 arrival 和 byte transfer 都完成时，barrier 才完成。

对于 Tensor Core 工作，arrival path 不同。`tcgen05` MMA 不会仅仅因为 MMA 已经发射就自动推进 barrier。
kernel 必须显式地把 barrier arrival 附着到 commit path 上，例如使用 `tcgen05.commit.mbarrier::arrive` 操作。
当这个 committed group 完成时，Tensor Core 侧会执行 barrier arrival。如果 kernel 忘了这个 commit arrival，
等待 barrier 的 consumer 会永远等下去。

普通 thread 也可以直接 arrive 到 barrier。当普通 thread code 是 producer，或一组 thread 在宣布自己已经用完某个资源时，会使用这种方式。
例如，consumer 读完 shared-memory buffer 后，可以 arrive 到一个 barrier，告诉 producer 这个 buffer 可以复用了。

waiting 是同一协议的 consumer 侧。consumer 会等待，直到 barrier 完成当前 iteration 所期望的 phase。
只有这时，读取数据或复用该 barrier 保护的资源才是安全的。

重要的一点是，异步硬件不仅会跑在程序前面；它也会通过 barrier 把 completion 报告回来。
TMA 可以 signal 一个 shared-memory tile 已经 ready。Tensor Core 工作可以 signal TMEM 结果已经 ready。
普通 thread 可以 signal 某个 buffer 不再被使用。barrier 给这些情况统一了 producer-consumer 形状：
producer arrive，consumer wait。

## Phase Tracking

barrier 通常不会只为一次使用而分配。pipelined K-loop 可能执行同一个 handoff 数百次，
如果每次 iteration 都分配新的 shared-memory barrier，就不现实。因此，kernel 会保留一小组固定 barrier，
并随着 loop 推进反复使用它们。

phase bit 让这种复用变得安全。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/phase_tracking.html" title="mbarrier phase tracking" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互：一个在多个 pipeline iteration 中复用的 barrier，展示 phase bit 如何在每个完成轮次后翻转。*

每当 barrier 完成当前轮的所有 arrival，它都会翻转 phase：phase 0 变成 phase 1，phase 1 变成 phase 0，如此往复。
wait 操作会检查 consumer 期望的 phase。这个 expected phase 由 kernel 保存在寄存器中。
当某个 stage 成功等待一轮后，kernel 会在下一轮使用该 barrier 前切换自己的本地 phase value。

这防止 kernel 把旧 completion 误认为新 completion。假设某个 barrier 用于一次 TMA load 并已经完成。
如果下一个 loop iteration 复用同一个 barrier 却不追踪 phase，consumer 可能观察到上一次 completion，
并错误地认为新的 load 已经 ready。phase bit 把这两轮分开：iteration 0 等待一个 phase，
iteration 1 等待相反 phase，iteration 2 再次等待第一个 phase，模式持续下去。

在真实 pipeline 中，bookkeeping 通常按 stage 进行。kernel 有固定数量的 shared-memory stage，
匹配固定数量的 barrier，以及寄存器中一小组 phase value。随着 loop 前进，每个逻辑 iteration 映射到一个物理 stage，
phase value 告诉 wait 操作它正在等待这个物理 barrier 的哪一轮。

这就是为什么后面的 GEMM 代码不需要每个 K tile 一个 barrier（{ref}`zh_chap_gemm_async`）。
它需要每个 reusable stage 一个 barrier，再加上 phase tracking。stage index 选择 shared-memory buffer 和 barrier。
phase value 区分这个 stage 当前使用和上一次使用。

**可以让你的 agent 试试**：给它一个 two-stage pipeline，让它追踪四次 iteration。
对每次 iteration，列出 stage index、本地 phase value、barrier 何时翻转，以及如果 stage 复用前没有切换 phase 会出什么问题。

## 同步规则

一旦 barrier 和 phase 机制清楚了，tensor-core kernel 中的同步 pattern 就相当机械。
每当一条路径产生数据，或释放另一条路径将要消费的资源时，handoff 都必须显式完成。

常见有三种情况。

第一种情况是 thread code 为异步引擎产生数据。如果 thread 写 shared memory，后续 TMA store 或 MMA 指令会读取这块 shared memory，
kernel 就必须在引擎读取前让 thread 写入可见。这需要合适的 thread-level synchronization 或 fence。
精确指令取决于 handoff 的 scope，但原因始终相同：引擎不能在 producer thread 完成写入前观察 shared-memory buffer。

第二种情况是 TMA 为 MMA 产生数据。TMA load 会异步填充 shared-memory tile。
MMA 路径不能只因为 TMA 指令已经发射，就推断 tile 已经 ready。
TMA 操作必须关联一个 `mbarrier`，而 MMA 路径必须在读取 tile 前 wait 这个 barrier。

第三种情况是 MMA 为 epilogue 产生数据。`tcgen05` MMA 会异步把结果写入 TMEM。
在 Tensor Core 完成相关工作之前，epilogue 不能安全读取 accumulator。
因此 MMA commit path 会 arrive 到一个 completion barrier，epilogue 在读取 TMEM 前 wait 这个 barrier。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/mbarrier_tma_timeline.html" title="mbarrier signalling TMA completion" loading="lazy"
        style="width:100%; min-width:1320px; height:700px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互：TMA load 通过 `mbarrier` signal completion。MMA 路径在读取 shared-memory tile 前等待 barrier。
Tensor Core 到 epilogue 的 handoff 形状相同，只是执行 arrival 的不是 TMA，而是 Tensor Core commit path。*

同一个思想也适用于 resource reuse。barrier 不只是 data-ready signal，也可以是“resource is free” signal。
在旧 tile 的所有 consumer 都用完它之前，shared-memory stage 不能被覆写。
在前一个使用者完成读写之前，TMEM region 不能被复用。在这些情况下，arrival 表示“我用完这个资源了”，
wait 表示“现在可以安全地为下一个 stage 复用这个资源了”。

这正是阅读 pipelined GEMM kernel 中同步逻辑的正确方式。wait 和 arrive 并不是作为 defensive programming 四处散落。
每一个都标记一次具体 ownership transfer：tile 变得 ready、accumulator 变得可读，或 buffer 变得可复用。
一旦识别出这些 handoff，control flow 就会容易跟随得多。
