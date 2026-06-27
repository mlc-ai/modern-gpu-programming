(zh_chap_tma)=
# 异步数据移动：TMA

:::{admonition} 概览
:class: overview

- TMA 是一个硬件引擎，用于在 global memory 和 shared memory 之间异步复制 tile。一个 thread 发射 copy，引擎负责移动字节。
- TMA copy 由 tensor-map descriptor 描述。descriptor 会告诉引擎 global tensor 的 shape、strides、tile coordinates，以及 shared-memory swizzle mode。
- 在 load 路径上，TMA 可以在写入 shared memory 时对 tile 做 swizzle，让 tile 直接落到 Tensor Core 所期望的 layout 中。
- TMA load 通过带 byte-count tracking 的 `mbarrier` 完成。TMA store 使用 commit group 和 wait group。
:::

只有当 Tensor Core 有准备好的数据可消费时，它才有帮助。在 GEMM 或 attention kernel 中，一旦 pipeline 填满，
数学部分可能是 compute-bound（{ref}`zh_chap_performance`），但只有下一个 operand tile 按时到达，pipeline 才能保持填满。

移动 tile 的旧方法是让 thread 自己复制。每个 thread 计算地址，从 global memory 发出 load，并把值存进 shared memory。
这可行，但它会把 warp 指令花在地址算术和 copy bookkeeping 上，而不是计算上。
它还会让 copy 路径出现在同一批本应喂给 Tensor Core 的 warp 的指令流中。

Tensor Memory Accelerator，即 TMA，会把这项工作移入硬件 copy engine。一个 thread 发射一次 tile copy。
随后 copy engine 在 global memory 和 shared memory 之间异步移动一个矩形 tile。
当引擎正在搬运字节时，CTA 的其余部分可以继续做其他工作。

TMA 也处理一部分 layout 问题。Tensor Core 不只是需要 shared memory 中有正确的值；
它还需要这些值处在正确的 shared-memory layout 中。在 load 路径上，TMA 可以在写入 tile 时应用 shared-memory swizzle。
这让 tile 可以直接落到后续 MMA 期望的 layout 中。

```{raw} html
<div style="overflow-x:auto;">
<iframe src="../demo/tma_intro.html" title="TMA: the Tensor Memory Accelerator" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互：TMA 将 tile 从 global memory 复制到 shared memory。切换 swizzle mode，并悬停 source cell，查看它落到 shared memory 中的哪里。*

## 一个 Thread 发射，硬件移动 Tile

TMA copy 从一个 issuing thread 开始。这个 thread 不会循环遍历 tile 中的所有元素。
它把 copy 的描述交给硬件，然后由 TMA engine 执行传输。

主要输入是 tensor-map descriptor。descriptor 描述 global tensor，以及应如何从中读取一个 tile。
它记录 tensor shape、strides、element size、tile shape 和 swizzle mode 等信息。
issuing thread 还会提供 tile 应该落到的 shared-memory address。

指令发射后，copy 会异步运行。issuing thread 可以继续执行，CTA 中的其他 thread 也可以继续执行。
传输现在由 TMA engine 负责，而不是由普通 load/store 指令循环负责。

这给了 kernel 两种不同方式来表达同一个逻辑操作：“复制这个 tile”。

一种路径是 thread copy。thread 协作地从 global memory load，并 store 到 shared memory。
这让 kernel 能直接控制每一次访问，但会消耗 thread 指令和用于地址计算的寄存器。

另一种路径是 TMA copy。一个 thread 发射传输，由硬件 copy engine 执行矩形 copy。
对于大型规则 tile，尤其是 Tensor Core kernel 使用的 operand tile，这是自然的路径。

这两条路径有不同的同步规则和性能行为。在二者之间选择，是一个 dispatch decision。
layout 告诉 kernel 它想要哪种内存排列。scope 告诉它哪些 thread 或 CTA 参与其中。
dispatch 则决定这个 copy 是由普通 thread code 实现，还是由 TMA 实现。

## Swizzled Layout

移动 tile 还不够。tile 还必须以 Tensor Core 能高效读取的 layout 放入 shared memory。

这正是 TMA swizzling 的用武之地。当 TMA 把 tile 写入 shared memory 时，它可以置换 shared-memory address pattern。
global memory tile 仍然是一个逻辑矩形，但 shared memory 中的 destination layout 可以是 swizzled 的。

swizzle mode 是 TMA descriptor 的一部分。一旦 descriptor 设置好，issuing thread 就不必手工应用 swizzle。
引擎会在字节落入 shared memory 时应用它。

重要要求是一致性。TMA descriptor、shared-memory tile layout 和后续 MMA 指令必须都描述同一个 layout（{ref}`zh_chap_data_layout`）。
如果 TMA 用一种 swizzle 写入 tile，而 MMA 却按另一种 swizzle 去读，硬件仍然会精确执行它被要求做的事；
只是这些字节对计算来说会排列错误。

在这一点上，layout 记法就不再只是 bookkeeping device。DSL 使用的 layout 必须匹配 TMA descriptor 和 Tensor Core 指令使用的硬件 layout。
例如，如果 kernel 说某个 operand tile 存储在 128-byte swizzled layout 中，TMA descriptor 就必须使用匹配的 swizzle mode，
而 MMA dispatch 也必须期望同一个 shared-memory arrangement。上面的演示允许你在 no swizzle 和 128-byte swizzle 之间切换；
悬停某个 source element，可以查看应用 swizzle 后它落在哪里。

理解 swizzle 的一种有用方式是：TMA 并没有改变逻辑 tile。它改变的是逻辑元素在 shared memory 中的物理落点。
后续 MMA 消费的仍然是同一个逻辑 A 或 B tile。swizzle 只决定这个 tile 如何排列在 shared memory bank 上。

## 用于 Tiling 和 Swizzling 的 3D TMA

普通 TMA copy 移动的是平坦 2D tile，但 Tensor Core 想要的 shared-memory layout 通常会被 *tiled* 成 swizzle atom
（来自 {ref}`zh_chap_data_layout` 的 8 x 128-byte atom）。TMA 用额外的 descriptor 维度来处理这一点。
**3D TMA** 把 shared-memory box 描述为 `(group, row, col)`，其中 group 维度跨 atom 行走，内部两个维度则在一个 atom 内寻址。
一次 3D copy 随后既会按 atom 布置 tile（tiling），又会在每个 atom 内应用 swizzle，
因此数据到达时已经处在 MMA 期望的 layout 中，不需要单独的 tiling 或 swizzling pass。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tma_3d.html" title="Tiling and swizzling with 3D TMA" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
```
*交互：一个 3D TMA copy，以 (group, row, col) 寻址，并 tiled 到 swizzled shared memory 中。*

选择 swizzle *format* 与这种 tiling 绑定在一起。更宽的 swizzle 会把一列分散到更多 bank 上，
所以能适配时默认选择 128-byte swizzle；但一个 N-byte atom 需要 tile 的 contiguous dimension 能填满它。
因此，如果某个 tile 因 shape 约束而较小，就不能使用 128-byte swizzle，必须降到 64-byte 或 32-byte：
经验法则是选择 tile 能填满的最大 swizzle（{ref}`zh_chap_data_layout`）。下面的演示直接展示了这个约束：
16 x 16 tile 上的 128-byte swizzle，只有当 tile 被切成匹配 atom 的 16 x 8 group 后，才会 conflict-free。

```{raw} html
<div style="overflow-x:auto;">
<iframe class="demo-tma3d" src="../demo/tiling_constraint.html" title="Swizzle imposes a tiling constraint" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--pst-color-border, #d0d0d0); border-radius:6px;"></iframe>
</div>
<script>
(function () {
  window.addEventListener('message', function (e) {
    var d = e.data;
    if (!d || d.type !== 'demoHeight' || !d.height) return;
    document.querySelectorAll('iframe.demo-tma3d').forEach(function (f) {
      if (e.source === f.contentWindow) f.style.height = d.height + 'px';
    });
  });
})();
</script>
```
*交互：16 x 16 tile 上的 128-byte swizzle；一旦 tiled 成 16 x 8 group，就会 conflict-free。*

## Completion：Load

copy 是异步的，所以仅仅发射还不够。consumer 不能只因为 TMA 指令已经发射，就去读取 shared-memory tile。
只有当引擎已经完成字节写入后，tile 才能安全读取。

对于 TMA load，完成信号是一个 `mbarrier`（{ref}`zh_chap_async_barriers`）。

通常序列如下：

1. 为 pipeline stage 初始化或复用一个 `mbarrier`；
2. 告诉 barrier 这次 TMA transfer 预计写入多少字节；
3. 发射 TMA load；
4. 让 TMA engine 在字节到达时更新 barrier；
5. 在读取 shared-memory tile 之前，让 consumer 等待对应 barrier phase。

byte count 通过如下操作设置：

```text
mbarrier.arrive.expect_tx(bytes)
```

这做了两件事。它记录预期 transfer size，同时也执行 issuing thread 在 barrier 上的 arrival。
barrier 不会仅仅因为这个调用发生就完成。它仍然等待 TMA engine 报告预期字节已经到达。

随着 transfer 进行，引擎会对 barrier 执行 complete-tx update。只有两个条件都满足时，barrier phase 才会翻转：
arrival count 已满足，并且 pending byte count 到达零。

consumer 随后等待这个 barrier。一旦对预期 phase 的 wait 完成，shared-memory tile 就 ready 了。
此时 MMA 路径可以安全读取它。

![TMA 加载同步流程](../img/tma_sync_flow.png)

这是其他异步 producer-consumer handoff 使用的同一个 barrier 模型。producer 是 TMA engine。
consumer 是 MMA 路径，或任何读取 shared-memory tile 的其他代码。barrier 是二者之间的显式 handoff。

## Completion：Store

TMA store 按相反方向移动数据：从 shared memory 到 global memory。它们同样是异步的，但 completion mechanism 不同。

TMA load 通常会喂给同一个 kernel 内部的 consumer。MMA 路径需要知道 shared-memory tile 何时 ready。
这就是 load 路径使用 `mbarrier` 的原因。

TMA store 通常把最终数据写出到 global memory。通常没有立即的 in-kernel consumer 在等待被存储的结果。
kernel 主要需要知道的是：什么时候可以安全复用 shared-memory buffer，或结束 store sequence。

为此，TMA store 使用 commit group 和 wait group。kernel 发射一个或多个 store，commit 这个 group，
稍后等待这个 group drain。wait 完成后，从 kernel 的角度看，该 group 中的 store 已经完成，
store 使用的 shared-memory region 可以安全复用。

所以规则很简单：

```text
TMA 加载：通过带字节计数跟踪的 mbarrier 等待
TMA 存储：通过 commit group 和 wait group 等待
```

这两种机制在不同 handoff point 服务于同一个目的。load 需要让 shared-memory tile 对后续 consumer 可见。
store 需要确保 outgoing transfer 已完成，然后 kernel 才能复用 source storage，或依赖 store 已经 drain。

## 为什么 TMA 对 Pipelining 很重要

当 TMA 成为 pipeline 的一部分时，它最有用。kernel 可以在 Tensor Core 计算当前 tile 的同时，发射未来 tile 的 load。
load 在后台运行，compute 在前台运行。当未来 tile 变成当前 tile 时，barrier 把二者连接起来。

典型 GEMM loop 会反复使用这种结构。shared memory 的一个 stage 保存当前被 MMA 消费的 tile。
另一个 stage 正在被 TMA 填充。随着 loop 前进，这些角色会轮换。MMA 读取某个 stage 之前，会等待该 stage 的 load barrier。
TMA 覆写某个 stage 之前，kernel 会确保前一个 consumer 已经用完它。

这就是为什么 TMA 和 `mbarrier` 通常一起出现在 Blackwell 和 Hopper 风格的 kernel 中。
TMA 给 kernel 一个异步 copy engine；barrier 给 kernel 一种精确方式，知道复制的字节何时 ready。
