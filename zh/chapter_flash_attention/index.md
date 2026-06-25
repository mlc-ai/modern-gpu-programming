(zh_chap_flash_attention)=
# Flash Attention 4

:::{admonition} 概览
:class: overview

- Attention 会运行两个 MMA，并在它们之间插入 softmax，因此它不能像 GEMM 那样简单地重复同一个 MMA。
- 这个 kernel 把 Part I 中的硬件原语（TMA、`tcgen05`、TMEM、barrier）和 Part III 中的 GEMM 技术组合起来：warp 角色分工、online-softmax rescaling、causal masking 和 GQA。
:::

Attention 是决定 transformer 能不能高效运行的核心 kernel，也是前面构建的所有机制终于汇合的地方。我们为 GEMM 组装过的每个组件都会延续到这里：TMA tile 搬运、`tcgen05` MMA、TMEM、warpgroup register tile，以及显式 barrier。

难点在于，attention 不是简单重复一个 MMA。它是两个 MMA，中间夹着真正的计算：online softmax、causal masking，以及把早期和后续 block 维持在同一尺度下的 rescaling。

新的复杂度正藏在这个中间阶段。普通矩阵乘只需要往 accumulator 里加；attention 则必须在新的 key 和 value 持续流入时，回头重访并重新缩放已经算过的结果。softmax 本身也运行在两个 Tensor Core MMA 之间的 CUDA core 上，因此指数运算和逐行规约直接落在关键路径上。

这就是为什么 attention 优化很大一部分其实是 softmax 优化：重写 `exp`，并把 softmax 与 MMA 重叠起来，而不是让 MMA 停下来等待它。

本章的目标不是从零重新推导 Flash Attention。我们会保留足够的算法视角，让 kernel 能读懂；然后把注意力放在真正新的部分：这个算法如何落到 TIRx 上。

最清晰的入口，是跟随一个 tile 在 kernel 中的流动。`Q`、`K` 和 `V` 作为输入 tile 进入 kernel，从 GMEM 加载到 SMEM。score MMA 将 `Q` 和 `K` 相乘，得到 TMEM 中的 score tile `S`。softmax 把 `S` 变成 numerator tile `P`，value MMA 再组合 `P` 和 `V` 来更新输出 accumulator `O`。

到这里为止，它看起来像是两个矩阵乘粘在一起。但它有一个 GEMM 不必处理的转折：每当运行中的 softmax 最大值发生变化，已经累计到 `O` 里的结果就突然处在了错误的尺度上。它必须先被重新缩放，下一次 value MMA 才能安全地加进去。下面几节会先追踪这条路径，然后再展示 TIRx 如何把每个阶段交给对应的 warpgroup，并把这些阶段串起来。

## 算法形状

在把 tile 放进内存之前，我们需要先看清这些 tile 服务的算法。对于一个 query block，Flash Attention 计算：

$$O = \text{softmax}(QK^{\top} / \sqrt{d})V$$

按字面理解，这个公式会先形成完整的 score 矩阵 `S = QKᵀ`，对它做 softmax，然后再乘以 `V`。这恰恰是我们不能采用的做法，因为完整的 `S` 非常巨大。seq=4096 时，每个 head 大约有 16M 个元素，fp32 下约 64 MB，远远超过 SMEM 或单个 128×512 TMEM 区域的容量。片上根本没有地方放它。Flash Attention 的答案是完全不物化 `S`。它改为按 block 流式读取 `K/V`，并维护三个逐行运行状态，用来概括目前为止看过的全部内容：

- `row_max`：到目前为止看到的最大 score。
- `row_sum`：softmax 分母的运行和。
- `O`：运行中的输出 accumulator。

流式更新负责让这些状态在新 block 到来时仍然正确。微妙之处在于，每处理一个 block，运行中的最大值都可能升高；一旦升高，所有在旧最大值尺度下计算出的内容都落在了错误尺度上。因此在加入新的贡献之前，我们先把旧状态拉回到新的尺度：

```text
S = Q_block @ K_block.T
m_new = max(row_max, rowmax(S))
scale = exp((row_max - m_new) / sqrt(d))
P = exp((S - m_new) / sqrt(d))
row_sum = row_sum * scale + rowsum(P)
O = O * scale + P @ V_block
row_max = m_new
```

单个 `scale` 因子在这里一鱼两吃：它同时重新缩放运行中的分母和运行中的输出，让早先 block 与后续 block 的贡献最终都落在同一个尺度上。

上面的伪代码使用自然 `exp`，并显式写出 `/sqrt(d)`，因为这样最好读；但 kernel 采用了更便宜的路径。它把 `1/sqrt(d)` 和 `log2(e)` 合并成一个常量 `scale_log2 = log2(e)/sqrt(d)`，然后用硬件 `exp2` 在原始 score 上计算所有指数，利用恒等式 `exp(x/sqrt(d)) = exp2(x · scale_log2)`。动机很简单：在这类硬件上，`exp2` 比自然 `exp` 更快。

继续往下之前，有一点值得钉牢：这里的 `P` 不是最终归一化后的 attention 矩阵。它只是当前 K/V block 的 softmax 分子。归一化被刻意推迟，只有最后一个 block 处理完之后，kernel 才写出 `O / row_sum`。

对 TIRx 来说，知道算法算什么只是一半；另一半是 kernel 运行时每个 tile 住在哪里，因为这决定了 layout 和 barrier 代码。`S`、`P` 和 `O` 都是 tile 值，而且各自有自己的家：

- `S` 是 score tile。score MMA 将它写入 TMEM。
- `P` 是 softmax numerator tile。softmax 从 TMEM 把 `S` 读入寄存器，计算 `P = exp((S - m_new) / sqrt(d))`，再把 `P` 写回 TMEM。
- `O` 是输出 accumulator tile。value MMA 从 TMEM 读取 `P`、从 SMEM 读取 `V`，然后累计到 TMEM 中的 `O`。

前面提到的 rescale 也是一个 tile 操作，而不是一段标量记账：当 `row_max` 变化时，旧的 `O` 会从 TMEM 读出，在寄存器中相乘，再写回 TMEM，然后下一次 value MMA 才会继续累加。后续每一节都会沿着同样的结构展开：tile 的位置、硬件路径，以及证明下一个 consumer 可以运行的 barrier。

## Tile-Primitive 图

有了运行状态及其位置之后，我们可以把算法展开成一串具体的 tile 移动。对于一个 K/V block，kernel 从上到下走过这条 tile 路径：

```text
Q, K, V in GMEM
  -> Q, K, V in SMEM        by TMA load
  -> S in TMEM              by score MMA: QK^T
  -> P in TMEM              by softmax numerator: TMEM -> RF -> TMEM
  -> O in TMEM              by value MMA: P V
  -> O in GMEM              by normalization, SMEM staging, and TMA store
```

它和 GEMM 的区别归结为一行。GEMM 是重复一条 MMA 链；FA4 有两个 MMA 阶段，中间坐着 softmax。后面几乎所有复杂度，都是这个额外阶段带来的后果。

如果把上面的短路径展开成显式的 producer-consumer 边，就得到完整图：

| 阶段 | Tile 移动或计算 | TIRx primitive | 硬件路径 |
|-------|------------------|----------------|----------|
| 加载 Q/K/V | GMEM tile -> SMEM tile | `Tx.copy_async(..., dispatch="tma")` | TMA load |
| Score MMA | SMEM 中的 Q 与 K -> TMEM 中的 score tile `S` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` |
| Softmax 读取 | TMEM 中的 `S` -> warpgroup register tile | `Tx.wg.copy_async(reg, tmem)` | `tcgen05.ld` |
| Softmax 写入 | 寄存器中的 numerator tile `P` -> fp16 TMEM view | `Tx.copy_async(tmem_as_f16, reg)` | TMEM store，然后 `tcgen05.wait.st()` |
| Value MMA | TMEM 中的 `P` 与 SMEM 中的 V -> TMEM 中的输出 accumulator `O` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | 带 TMEM operand 的 `tcgen05.mma` |
| Correction | TMEM 中的 `O` -> 寄存器 -> TMEM 中的 `O` | TMEM readback、寄存器乘法、TMEM store | `tcgen05.ld` / TMEM store |
| Epilogue | TMEM 中的最终 `O` -> 寄存器 -> SMEM -> GMEM | TMEM readback、`Tx.copy`、TMA store | `tcgen05.ld` + TMA store |

新增的行是 softmax 和 correction。二者都会增加 TMEM -> register -> TMEM 流量，也都会在 score MMA 与 value MMA 之间制造额外的交接。

**试着让你的 agent 做一遍**：让它只追踪上面的短路径。对每条箭头，指出 producer 阶段、consumer 阶段、源 tile、目标 tile 和硬件路径。然后再问哪些箭头在 GEMM 章节里并不存在。

## Warp 角色与 Scope

数据路径理清之后，自然的下一个问题是：每个阶段到底由谁来跑。这里每个 CTA 有 4 个 warpgroup，总共 512 个线程；它们不是按接触的数据划分，而是按 warpgroup 执行的工作类型划分：

- WG3 驱动硬件引擎：TMA load、MMA 和 TMA store。
- WG0、WG1、WG2 执行这些引擎调用之间的寄存器重计算：softmax、correction 和 epilogue。

精确的角色表如下：

| Owner | 角色 | 做什么 |
|-------|------|--------|
| WG3, warp 1 | TMA load | 从 GMEM 把 Q、K、V tile 加载到 SMEM |
| WG3, warp 0 | MMA | 发出 score MMA 和 value MMA |
| WG3, warp 2 | TMA store | 把最终 O tile 从 SMEM 存回 GMEM |
| WG0 | Q stage 0 的 softmax | 从 TMEM 读取 S，计算 P，把 P 写回 TMEM |
| WG1 | Q stage 1 的 softmax | 对第二个 Q pipeline stage 执行同样工作 |
| WG2 | Correction 与 epilogue | 重新缩放 TMEM 中的 O，做归一化，暂存输出 |

很容易把“两个 Q stage”误读成两个 attention head，但它们不是。它们只是 Q pipeline 中的两个槽位：WG0 拥有一个，WG1 拥有另一个，因此两个 Q tile 可以同时在路上。这就是 softmax 工作出现两份的原因，一份在 WG0，一份在 WG1。

代码用符号坐标选出这些角色：

```python
wg_id = T.warpgroup_id([4])
warp_id = T.warp_id_in_wg([4])
```

读 kernel 时，先找角色分支。它会告诉你分支内部每个 tile primitive 归哪个团队所有。

- WG3 warp 1 启动 TMA load 命令。一个被选出的 lane 发出 copy，TMA 引擎移动 tile。
- WG3 warp 0 发出 `tcgen05.mma` 指令。
- WG0 和 WG1 在完整 warpgroup scope 下运行 softmax。
- WG2 在完整 warpgroup scope 下执行 correction 与 epilogue。

一个不对称性最终塑造了整个 barrier 图：每个 MMA，无论 score 还是 value，都只由 WG3 warp 0 发出。WG0 和 WG1 从不发出 MMA。它们只消费 score tile、运行 softmax，并把 `P` 写回 TMEM。

正是这种分离，让 softmax 周围必须有 barrier。`s_ready` 把 score tile 从 MMA warp 交给 softmax；`p_o_rescale` 则交付 `P`，以及一个对 value MMA 来说安全的 `O` 槽位：要么已经完成 rescale，要么因为不需要 rescale 而被释放。后面几节我们会反复回到这两个名字。

## 阅读代码片段

本章的代码片段摘自 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py)，所以它们不可避免地会引用一些我们没有完整复现的 kernel 内部名称。自解释的名称（`wg_id`、`warp_id`、`BLK_M`/`BLK_N`、`HEAD_DIM`、`kv_stage`、各个 `SMEM_PIPE_DEPTH_*` / `TMEM_PIPE_DEPTH` 深度、`should_accumulate`，以及这里为 1 的 `CTA_GROUP`）会在第一次相关时引入。其余名字先在下表给出一行释义，这样当代码片段突然把一个陌生名字放到你面前时，你有地方可查：

| 名称 | 含义 |
|------|------|
| `q_stage`, `i_q` | Q pipeline stage，取 0 或 1，也就是哪个 Q tile 槽位（`SMEM_PIPE_DEPTH_Q = 2`）。在 WG0/WG1 softmax 内部，warpgroup 自己的 `wg_id`（0 或 1）就是同一个 stage 索引，因此 `S_region[q_stage]`、`P_region[wg_id]` 和 `O_region[i_q]` 都选择同一个 Q stage |
| `MMA_N` | TMEM 列中的 score/output tile 宽度（128） |
| `MMA_K` | `P`/`V` 列方向的 MMA inner-K 步长（16）；`K_SPLIT = 6 * MMA_K = 96` |
| `K_SPLIT` | value-MMA 调度的切分点（见“两段 MMA 阶段”）；第一段 value MMA 覆盖列 `0:K_SPLIT`（`6 * MMA_K = 96`） |
| `should_rescale` | WG2 的逐行标志：旧 `O` 是否需要在下一次 value MMA 前 rescale（通过 `any_sync` 在 warpgroup 内规约） |
| `rescale_threshold` | 跳过小幅 row-max 变化的阈值；当前 kernel 使用 `8.0`，跳过 rescale 时会把 `acc_scale` 精确设为 `1.0` |
| `scale_log2` | log2 单位下的 softmax scale，即 `log2(e)/√d`，因此 `P = exp2((S - m) · scale_log2)` |
| `acc_scale` | softmax 通过 SMEM mailbox 传给 WG2 的逐行 rescale 因子 |
| `chunk_start`/`chunk_end`, `p_start`/`p_end` | 正在读取/写入的 32 宽 softmax chunk 的列范围 |

## 两段 MMA 阶段

对每个流式 K/V tile，Flash Attention 都会运行两个 MMA 阶段，并由 softmax 把它们桥接起来：

```text
Q, K -> score MMA -> S
S    -> softmax   -> P
P, V -> value MMA -> O
```

可以把它看成三个 producer 串成的一条 pipeline。第一个 MMA 产生 attention score `S`，softmax 把 `S` 转成 numerator `P`，第二个 MMA 消费 `P` 来更新输出 accumulator `O`。按 `row_sum` 归一化会一直推迟到 epilogue，等每个 K/V tile 都贡献完之后再做。

下面每个 tile op 都会使用 GEMM 步骤中同样的 **scope / layout / dispatch** 卡片，只额外加一行 **handoff**，用来指出把 tile 交给下一个角色的 barrier。

计算代码从不直接使用裸 TMEM 列号。kernel 会把唯一的 TMEM 分配切成按 stage 索引的视图（`S_region`、`P_region`、`O_region`），然后用 pipeline stage 访问它们（`S_region[q_stage]`、`O_region[i_q]`、`P_region[i_q, 0:K_SPLIT]`）。这些视图由 [TMEM 布局与复用](#tmem-布局与复用) 一节中的 `T.TMEMStages` 定义；现在只要把每个 region 理解为同一块物理 TMEM 的一个具名切片就够了。

### Score MMA

两个阶段中的第一个是 score MMA，也就是打开每个 K/V iteration 的矩阵乘。它计算：

$$S = Q_{\text{block}}K_{\text{block}}^{\top}$$

并把 `128 x 128` score tile 写入 TMEM：

```python
Tx.warp.gemm_async(
    S_region[q_stage],
    Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
    K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
if T.ptx.elect_sync():
    s_ready.arrive(q_stage)
```

我们可以问 GEMM 章节对每个 tile op 问过的同样四个问题：谁运行它、tile 住在哪里、如何 dispatch，以及如何交接：

> **Tile-primitive 解读：Score MMA**
> - Scope：WG3 warp 0 发出它；一个 elected lane 到达 `s_ready`。
> - Layout：Q、K 在 SMEM → TMEM 中的 `S`（`S_region[q_stage]`）。
> - Dispatch：`tcgen05`。
> - Handoff：`s_ready`（→ softmax）。

被选中的单个线程到达 `s_ready`，就是整个交接。它宣告这个 score tile 已经完成，softmax warpgroup 现在可以读取它了。

### 两个 MMA 之间的 Softmax

两个 MMA 之间坐着 softmax，它把 score tile `S` 转成 numerator tile `P`。它的解读卡如下：

> **Tile-primitive 解读：Softmax**
> - Scope：WG0（Q stage 0）/ WG1（Q stage 1），完整 warpgroup。
> - Layout：TMEM 中的 `S` → 寄存器 → fp16 TMEM 中的 `P`（`P_region[wg_id]`）。
> - Dispatch：用 `tcgen05.ld` 读取，用 TMEM store 写入；中间在寄存器中做逐行计算。
> - Handoff：等待 `s_ready`；到达 `p_o_rescale`（前 96 列）和 `p_ready_2`（最后 32 列）。

这个阶段完全没有 GEMM 对应物。WG0/WG1 等待 `s_ready` 上的 score tile 到达，然后每次按寄存器大小的 chunk 从 TMEM 读出：

```python
Tx.copy_async(
    s_chunk[:, chunk_start : chunk_end],
    S_region[wg_id, chunk_start : chunk_end],
)
```

这是 warpgroup scope 下的一次 TMEM-to-register tile 读取。score 进入寄存器后，softmax warpgroup 按顺序做三件事：

1. 计算 row max 和 row sum；
2. 计算 softmax numerator tile `P`；
3. 以 fp16 把 `P` 写回 TMEM。

最后一步形如：

```python
Tx.copy_async(
    P_region[wg_id, p_start : p_end],
    p_chunk[:, p_start : p_end],
)
```

为什么已经在寄存器里算完了，还要把 `P` 写回 TMEM？因为 value MMA 需要把 `P` 当作一个 *tile operand*，而 MMA 不能把分散在每个线程里的标量寄存器直接当成矩阵来读。在这个 kernel 中，MMA 可读的 `P` 形态就是 `P_region`，它是 fp16 TMEM alias `tmem_as_f16` 上的一个视图。所以这次写回不是多余搬运；它是在把 `P` 放进下一个 MMA 唯一能消费的形态。

### Value MMA

第二个阶段，也是每个 K/V iteration 的收尾阶段，是 value MMA。它计算：

$$O = O + P_{\text{block}}V_{\text{block}}$$

这个 MMA 运行时，`O` 已经被放进了当前 K/V block 需要的正确状态：第一个 block 上完成初始化，后续 block 上完成 rescale。因此 MMA 只需要累加。它和 GEMM 的区别在于 operand 的位置：A operand 是 TMEM 中的 `P`，B operand 是 SMEM 中的 `V`，accumulator `O` 也在 TMEM 中：

```python
# 第一段 sub-MMA：列 0:K_SPLIT（P 的前 96 列 / V 的对应行）。
Tx.warp.gemm_async(
    O_region[i_q],
    P_region[i_q, 0:K_SPLIT],
    V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
    transB=True,
    accum=should_accumulate,
    dispatch="tcgen05",
    cta_group=CTA_GROUP,
)
# 第二段 sub-MMA 形式相同，accum=True，由 p_ready_2 gate 控制，
# 覆盖剩余列 K_SPLIT:BLK_N。
```

> **Tile-primitive 解读：Value MMA**
> - Scope：WG3 warp 0。
> - Layout：TMEM 中的 `P` + SMEM 中的 V → TMEM 中的 `O`（`O_region[i_q]`）。
> - Dispatch：带 TMEM operand 的 `tcgen05`。
> - Handoff：等待 `p_o_rescale`、`p_ready_2`、`kv_load.full`；到达 `o_ready`（→ epilogue）。

这种 operand 放置是两个 MMA 在硬件上的差异：

- Score MMA 从 SMEM 读取两个 operand：Q 和 K。
- Value MMA 从 TMEM 读取一个 operand：`P`。
- Value MMA 从 SMEM 读取另一个 operand：V。
- 结果累计到 TMEM 中的 `O`。

`accum=should_accumulate` 标志实现了算法中的“初始化还是相加”选择：query block 的第一个 K/V tile 上为 false，之后每个 tile 上为 true。

你可能还会注意到，value MMA 不是一次性跑完，而是切成 `96 + 32` 的调度：

1. Softmax 以四个 32 列 chunk 写入 `P`。
2. 前三个 chunk 一就绪，value MMA 就开始处理 `P` 的前 96 列和 `V` 的匹配行。
3. 最后 32 列等待 `p_ready_2`。
4. 第二个 MMA 消费最后这个 chunk 并完成 tile。

这样切分是为了让 Tensor Core 保持忙碌。如果 value MMA 作为单条指令运行，整个阶段都要等四个 32 列 `P` chunk 全部完成指数计算并写回后才能开始。先对前三个 chunk 发起 MMA，可以把最后一个 chunk 的 `exp` 和 TMEM 写入，与已经在飞行中的 96 宽 MMA 重叠起来，把本来会空转的时间变成有用工作。

## TMEM 布局与复用

`S`、`P` 和 `O` 都必须共享一个 `128 x 512` TMEM 分配；它们被打包进同一块空间的方式，正是这个 kernel 中 barrier 与 layout 不可分割的原因：

下图直接展示了这种打包：score slot、numerator slot 和 output slot 全部共享同一块 TMEM 分配，因此 barrier 协议负责让这种复用合法。

![TMEM 布局](../img/tmem_layout_v3.png)

可以把图读成一组 tile 槽位：

- Score slot 保存 `S = QK^T`。
- Numerator slot 保存 softmax 指数化后的 `P` tile。
- Output slot 保存 fp32 `O` accumulator。

它们不是彼此独立的 buffer，而是同一块分配中的区域；这种共享不是风格选择，而是容量限制迫出来的。Q pipeline 深度为 2 时，两个 `S` slot（2 × MMA_N = 256 列）和两个 `O` slot（2 × MMA_N = 256 列）已经占满了全部 512 个 fp32 列。没有剩余空间给 `P`，所以 `P` 只能通过更窄的 fp16 view alias 到同一批字节上。安全性的唯一来源，是每个 region 都严格在前一个 consumer 完成之后才复用；这个时序正是 barrier 保证的。因此在 FA4 里，barrier 不只是调度机制；它们本身就是 layout 合法性的条件。

aliasing 技巧通过 `T.TMEMPool` 搭起来。kernel 先拿一个 fp32 view（`tmem`）用于 score 和 output accumulator，然后把 pool base 倒回 0，再在同一批物理字节上拿第二个 fp16 view（`tmem_as_f16`）：

```python
tmem_pool = T.TMEMPool(pool, total_cols=N_COLS_TMEM, cta_group=CTA_GROUP, tmem_addr=tmem_addr)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_pool.move_base_to(0)
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()
```

由于 fp16 元素宽度只有 fp32 的一半，fp16 view 会在同一批字节上暴露两倍数量的可索引列；`P` 正是住在这块空间里，而 fp32 layout 没有余量容纳它。拿到两个 view 后，kernel 使用 `T.TMEMStages` 把 `S`、`P` 和 `O` 槽位切成 staged region，这样计算代码就可以按 pipeline stage 索引，而不必直接操作裸列号：

```python
S_region = T.TMEMStages(tmem,        col_start=0,                       width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
O_region = T.TMEMStages(tmem,        col_start=MMA_N * SMEM_PIPE_DEPTH_Q, width=MMA_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N)
P_region = T.TMEMStages(tmem_as_f16, col_start=MMA_N,                   width=BLK_N, stages=SMEM_PIPE_DEPTH_Q, stride=MMA_N * 2)
```

`P_region` stride 里的 `* 2`，是 aliasing 在代码中显形的一个地方。`S_region` 和 `O_region` 用 fp32 `tmem` 列计数，而 `P_region` 用 fp16 `tmem_as_f16` 列计数；fp16 列只有一半宽，所以 stage 到 stage 的移动需要双倍 stride，才能落在相同的物理字节上。不过 region 一旦定义好，计算代码就保持干净：写 `S_region[q_stage]`，读 `S_region[wg_id, ...]`，写 `P_region[wg_id, ...]`，累计到 `O_region[i_q]`，完全不用碰裸列号。

**试着让你的 agent 做一遍**：让它解释这个 FA4 kernel 里的 fp32（`tmem`）和 fp16（`tmem_as_f16`）两个 view。哪些物理 TMEM 区域保存 `S`、`P` 和 `O`？为什么 `P_region` 的 stride 使用 `MMA_N * 2`？复用问题先留到下一节：看完 barrier 表之后，再检查每个 region 复用前必须等哪些 consumer 完成。

## Barrier 如何连接各个角色

这是整个 kernel 最难的部分，所以值得循序渐进。先从沿主计算路径移动数据的少数 barrier 入手，把其他部分都当作稍后可查的 bookkeeping。数据就绪交接包括：

| Handoff | 含义 |
|---------|------|
| TMA load -> score/value MMA | Q、K 或 V 已经到达 SMEM，可以供 MMA 使用 |
| score MMA -> softmax | `S` 已经在 TMEM 中就绪 |
| softmax/correction -> value MMA | `P` 已经在 TMEM 中就绪，并且 `O` 可以安全累计 |
| value MMA -> epilogue | 最终 `O` 已经在 TMEM 中就绪 |
| epilogue -> TMA store | `O_smem` 已经可以存回 |

不在这张表里的东西都是 pipeline bookkeeping：释放某个 SMEM、TMEM 或 staging buffer，让另一个角色可以复用它。有用的是，每个 barrier 不管携带的是数据还是 bookkeeping，都能用同一种方式阅读：一次 tile handoff。你问谁生产了数据、谁消费它，以及双方完成后哪个 buffer 变得可复用。

下一张图把这些交接压缩成两个 MMA 阶段的精确 readiness gate：score MMA 等什么，value MMA 累计前又必须等什么。

![Flash Attention 4 MMA 输入门控](../img/flash_attention_main_handoff.png)

请把这张图读成一组正确性 gate，而不是调度表。它回答“这个 MMA 发射前必须满足什么”，但不说明具体时序。score MMA 等待 SMEM 中的 Q 和 K，然后产生 `S`。value MMA 同时等待三件事：SMEM 中的 V、softmax 产生的 `P` tile，以及一个由 WG2 释放或 rescale 完成的 `O` 槽位。softmax 到 value 的 gate 会分裂成两段，原因我们已经见过：`P` 的前 96 列就绪后 value MMA 就可以开始，`p_ready_2` 再释放最后 32 列。

有一个 handoff 不符合 tile-readiness 的模板：softmax 到 correction 的边。它不是传递 tile，而是通过一个单槽 SMEM mailbox，把一个标量（K/V loop 中的 `acc_scale`，或 epilogue 中的最终 `row_sum`）传给 WG2。由于这个槽位每次 iteration 都会复用，因此必须由一对 `full`/`empty` barrier 保护：

下图放大了这个 mailbox handshake，因此这对 barrier 应该被读作一个标量 producer-consumer 通道，而不是 tile-ready gate。

![Flash Attention 4 softmax 缩放槽握手](../img/flash_attention_softmax_correction.png)

把 `softmax_corr.full` 和 `softmax_corr.empty` 读成一对 producer-consumer barrier：

1. Softmax 在复用 scale/sum 槽位前等待 `softmax_corr.empty`。
2. Softmax 把 `acc_scale` 或最终 `row_sum` 写入这个槽位。
3. Softmax 到达 `softmax_corr.full`。
4. WG2 等待 `softmax_corr.full`，然后读取这个槽位。
5. WG2 到达 `softmax_corr.empty`。
6. softmax warpgroup 可以在下一阶段复用这个槽位。

要特别小心 `softmax_corr.empty` 表示什么、又不表示什么。它只表示 WG2 已经消费了 scale/sum 槽位。它不说明 `P` 是否就绪，更绝对不是允许 value MMA 开始的 gate。真正的 gate 是 `p_o_rescale`，它在 `P` 的前 96 列写好、且 `O` 槽位可以安全累计时触发。混淆这两者，是产生错误结果的经典来源。

掌握主路径后，完整 barrier 列表就可以作为参考：

| Barrier | Producer -> consumer | 什么变得安全 |
|---------|----------------------|----------------|
| `q_load.full` | TMA load -> score MMA | Q SMEM tile 可以供 MMA 使用 |
| `q_load.empty` | 这个 Q stage 的所有 score MMA -> TMA load | Q SMEM stage 可以复用于下一个任务 |
| `kv_load.full` | TMA load -> score/value MMA | K 或 V SMEM tile 可以供 MMA 使用 |
| `kv_load.empty` | score/value MMA -> TMA load | K/V SMEM stage 可以复用 |
| `s_ready` | score MMA -> softmax | S TMEM tile 可以读取 |
| `p_o_rescale` | softmax + WG2 -> value MMA | P 的前 96 列已经在 TMEM 中，且 O 槽位可以供 value MMA 安全累计 |
| `p_ready_2` | softmax -> value MMA | P 的最后四分之一已经在 TMEM 中 |
| `o_ready` | value MMA -> epilogue | 最终 O accumulator 已经就绪 |
| `softmax_corr.full` | softmax -> WG2 | `acc_scale` 或最终 `row_sum` 已在 SMEM mailbox 中就绪 |
| `softmax_corr.empty` | WG2 -> softmax | WG2 读取后，同一个 SMEM mailbox 槽位可以复用 |
| `corr_epi.full` | epilogue -> TMA store | O_smem 已经可以存储 |
| `corr_epi.empty` | TMA store -> epilogue | O_smem stage 可以复用 |

和 GEMM 一样，你可以从信号生产者预测 barrier 类型：

- TMA load 使用 `TMABar`，因为 TMA 引擎会按字节数统计自己的完成。
- MMA 完成使用 `TCGen05Bar`，因为 `tcgen05.commit` 会 signal completion group。
- 纯线程到线程的 handoff 使用 `MBarrier`，参与线程显式 arrive。

softmax 到 value 的分裂式 handoff 值得仔细看。它使用两个 gate：

- `p_o_rescale` 在 `P` 的前 96 列写好、且 `O` tile 可以安全累计后，允许 value MMA 开始。
- `p_ready_2` 释放 `P` 的最后 32 列，对应上一节中的 `96 + 32` value-MMA 调度。

第一个 K/V block 是简单情况。WG2 会预先 arrive `p_o_rescale`，因为还没有旧的 `O` tile 需要 rescale。

后续 block 必须更谨慎。WG2 只有在跳过一次不必要的 rescale，或完成旧 `O` 的 rescale 之后，才会到达 `p_o_rescale`。跳过测试刻意保守：softmax 计算 log2 缩放后的 delta `(m_old - m_new) * scale_log2`；如果这个值仍然高于 `-rescale_threshold`，说明新的 max 变化还没大到值得 rescale，kernel 就保持旧 max，并把 `acc_scale` 精确设为 1.0。只有更大的 max 跳变才会走 `exp2` 路径，并要求 WG2 rescale `O`。

随后 WG2 用 `any_sync` 在 warpgroup 内规约 `should_rescale`。如果没有任何行需要更新，它就保持 `O` 不动。这个跳过很重要，因为 rescale `O` 是覆盖整个 accumulator 的一次 TMEM -> RF -> TMEM read-modify-write；当阈值逻辑已经把 `acc_scale` 维持为 1.0 时，做这件事就是纯浪费。

注意所有新增 barrier 都聚在同一个地方。`s_ready`、`p_o_rescale`、`p_ready_2`，以及 softmax/correction 这对 barrier，全都围绕 softmax。它们存在只有一个原因：score MMA 和 value MMA 不再相邻。寄存器计算、TMEM 重写和输出 rescale 现在插在两者之间，每一步都需要自己的 handoff。

**试着让你的 agent 做一遍**：让它追踪一个 K/V block 经过 `s_ready`、`p_o_rescale`、`p_ready_2` 和 `o_ready`。对每个 barrier，问谁等待、谁 arrive、哪个 tile 变得可读，以及之后哪个存储可以复用。

## Pipeline 结构

barrier 告诉我们某个角色消费 tile 之前什么必须 *就绪*。但它们没有告诉我们实际上哪些东西会 *并发* 运行；这正是现在要讨论的问题。两者确实不同：一个正确性 gate 可能在 producer 真正运行之前很久或之后很久才满足。

这里不存在单一的 pipeline 深度，因为不同 tile stream 以不同速率移动。因此 kernel 为每条 stream 保留独立的 ring：

- Q pipeline 深度 2：一个 CTA 同时处理两个 Q stage。WG0 处理一个 stage，WG1 处理另一个。
- KV pipeline 深度 3：K 和 V block 在内层循环中流动，同时复用同一批 Q stage。
- TMEM pipeline 深度 2：每个 Q stage 都有自己的 S/P/O TMEM 槽位，并在匹配的 barrier 触发后复用。

下图从正确性 gate 切换到时间线视角，展示这些独立 ring 进入稳定状态后，哪些角色大致可以同时活跃。

![Flash Attention 4 流水线结构](../img/flash_attention_pipeline_v2.png)

请把它读成时间线，而不是 barrier 图。它展示的是同一时刻大致有哪些角色处于活跃状态；前面的 barrier-flow 图才是检查精确 producer-consumer wait 的地方。两张图合起来，回答了本节开头提出的两个不同问题。

每一行对应代码中的一个角色分支：

- WG3 warp 1 发出 TMA load。
- WG3 warp 0 发出 score MMA 和 value MMA。
- WG0 与 WG1 为两个 Q stage 运行 softmax。
- WG2 释放或 rescale `O`，稍后再归一化最终输出。
- WG3 warp 2 发出 TMA store。

沿着图从左到右，可以追踪一个有代表性的 pipeline wave。load warp 先从 `Q0`、`K[n-1]`、`Q1`、`V[n-1]` 开始，然后持续流式读取更低索引的 K/V block。MMA warp 发出最早的 score MMA，产生 `S0` 和 `S1`，WG0/WG1 再把它们转成 `P0` 和 `P1`。

重要的是，MMA warp 不会先跑完所有 score MMA，再跑所有 value MMA。两个 Q stage 预热好之后，它会交错两种 MMA：当前 `V` block 的一次 value MMA，下一次 `K` block 的一次 score MMA，如此继续：

```text
score Q0*K[n-1]
score Q1*K[n-1]
value P0*V[n-1]
score Q0*K[n-2]
value P1*V[n-1]
score Q1*K[n-2]
value P0*V[n-2]
...
```

这种交错正是图中 score、softmax、correction 和 value 各行会重叠，而不是整齐依次执行的原因。

WG2 行标为 `release / rescale`，两半对应我们已经见过的两种情况。第一个 K/V block 上还没有旧 `O`，所以 WG2 只参与允许 value MMA 继续的 handoff；后续 block 上，它可能在 value MMA 累计之前先 rescale 旧的 `O`。归一化和 TMA store 只会发生一次，在 attention task 的最后一个 K/V block 之后。

没有一个 GEMM 风格的单 pipeline 可以描述 FA4，因为 Q、K/V 和 TMEM 槽位都在独立调度上前进。TIRx 把这些调度显式保留下来，用独立的 tile buffer、`PipelineState` cursor 和 barrier phase 表达，而不是把 kernel 藏进一个巨大的 monolithic primitive。代价是移动部件更多；收益是复杂度仍然可见、可检查。

## Rescaling 与 Writeback

rescale 是必须的，不是可以丢掉的优化。online softmax 可能随着每个新 score tile 抬高逐行最大值；一旦发生，早先 block 累计到 `O` 中的内容就是按 *旧* 最大值缩放的。这样早先每一项都会大出一个 `exp(m_new - m_old)` 因子。跳过 correction 会让这些 block 权重过大，最终输出就是错的。修正方式是一次 TMEM → registers → TMEM tile 操作：

$$O_{\text{old}} \leftarrow O_{\text{old}} \cdot e^{(m_{\text{old}} - m_{\text{new}}) / \sqrt{d}}$$

工作分给两个角色完成。softmax 计算逐行 scale，并把它投递到 SMEM mailbox；WG2 等待 `softmax_corr.full`，把当前 `O` 从 TMEM 读出，乘上该 scale，再把 `O` 写回：

```python
RESCALE_TILE = T.meta_var(16)
o_row = T.wg_reg_tile(RESCALE_TILE)
Tx.copy_async(o_row, O_region[i_q, d_start : d_start + RESCALE_TILE])
Tx.mul(o_row, o_row, acc_scale)
Tx.copy_async(O_region[i_q, d_start : d_start + RESCALE_TILE], o_row)
T.ptx.tcgen05.wait.st()
```

值得强调的是，这是覆盖整个 `O` accumulator 的一次完整 TMEM → registers → TMEM tile 操作，不是一点标量记账；它和其他阶段一样，也有自己的解读卡：

> **Tile-primitive 解读：Correction（rescale）**
> - Scope：WG2，完整 warpgroup。
> - Layout：TMEM 中的 `O` → 寄存器 → TMEM 中的 `O`（`O_region[i_q]`）。
> - Dispatch：用 `tcgen05.ld` 读取，用 TMEM store 写入；中间做寄存器乘法。
> - 交接：等待 `softmax_corr.full`；到达 `p_o_rescale`（→ value MMA）和 `softmax_corr.empty`（→ softmax）。

端到端追踪同步过程：

1. Softmax 把 scale 值写入 SMEM。
2. WG2 等待 `softmax_corr.full`。
3. WG2 在 TMEM 中 rescale `O`。
4. WG2 到达 `p_o_rescale`。
5. WG3 的 value MMA 现在可以消费 `P`，并累计到 rescale 后的 `O` tile。

WG2 读取之后，`softmax_corr.empty` 会释放 SMEM 槽位，循环随之闭合，softmax 可以在下一次 iteration 复用 mailbox。

K/V loop 结束后，WG2 从 correction 切换到 epilogue。它等待最终的 `row_sum` 和 `o_ready`，从 TMEM 读取最终 `O`，乘以 `1 / row_sum`（也就是一开始推迟的归一化），转换成 fp16，并写入 `O_smem`。然后 WG3 的 TMA store warp 把 `O_smem` 搬回 GMEM。

如果你打算扩展这个 kernel，有一个限制值得标出。它只计算 forward output，而训练时的 forward pass 通常还要保存 backward pass 需要的 log-sum-exp（LSE）。加入 LSE 时有一个缩放细节要记住：这个 kernel 把 `row_max` 保留为 *未缩放* 的原始 `QK^T` score 最大值，而 `row_sum` 累计的是 `exp((S - row_max) / sqrt(d))`。因此形成自然对数 LSE 时，必须把 `1/\sqrt{d}` 因子重新应用到 `row_max` 上：

$$\mathrm{LSE}_i = \log(\mathrm{row\_sum}_i) + \mathrm{row\_max}_i / \sqrt{d}$$

这个实现只输出 forward 结果，不写 LSE。

## Causal Masking

causal attention 增加了一个约束：一个 query 只能 attend 到自身位置及之前的 key。kernel 用两种互补方式满足它，一种便宜，一种精确。

便宜的方式是直接跳过整块工作。很多 K/V block 完全位于对角线上方，对给定 Q block 没有任何贡献，因此 `get_n_block_max(...)` 会计算该 block 最多可能需要的最后一个 block，循环就根本不加载、不计算剩余部分。

精确的方式处理跨过对角线的 block，也就是一部分列有效、一部分无效的情况。这些 block 仍然运行 score MMA，但 softmax 会在指数化之前把无效列 mask 掉。对每一行，它根据该行 query 位置和 block offset 推导一个列上限，保留不超过该上限的列，并在寄存器中把之后每一列设为 `-inf`，让这些列既不贡献 row max，也不贡献 `exp2` numerator。

实现并不是逐元素分支，而是用 `mask_r2p(...)` 应用这个上限：它把上限转成整个 32 宽 score chunk 上的 bit mask，并一次性 mask 整个 chunk。完全位于对角线下方的 block 保留所有列，不需要 mask。

从 tile-primitive 视角看，causal mode 完全不改写数据路径。它只是缩短 K/V trip count，并把一个 masking 步骤插入寄存器驻留的 softmax 中，位于 score MMA 和 `P` writeback 之间。

## GQA 支持

Grouped Query Attention 允许多个 query head 共享一个 K/V head。这能节省内存带宽，但也提出一个打包问题：如何只保留一个 K/V tile，同时让多个 query head 都使用它？这个 kernel 的答案是：一次处理一整组 query head，让它们共同对应一个调度出来的 `kv_head_idx`：

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

技巧在于重新解释 128 行 Q tile。对于 `GQA_RATIO=4`，它们不再表示 128 个序列位置，而是表示 32 个序列位置 × 4 个 query head；这些 query head 被打包在一起，共乘同一个 K/V tile。行解码如下：

```text
seq_pos = row // GQA_RATIO
q_head  = row % GQA_RATIO
```

Q load 用一个 3D view 表达这种打包。源数据是自然的 `Q[batch, seq, qo_head, dim]` 布局，目标则是 score MMA 随后会当作扁平 `128 x HEAD_DIM` operand 读取的同一个 SMEM tile。view 负责调和这两种形态，而且不需要任何额外 copy：

```python
Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH_Q, SEQ_Q_PER_TILE, GQA_RATIO, HEAD_DIM)
Tx.copy_async(
    Q_smem_3d[i_q, :, :, :],
    Q[batch_idx,
      m_start : m_start + SEQ_Q_PER_TILE,
      kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
      :],
    **tma_copy_q,
)
```

K 和 V 从不在内存中展开，而这正是 GQA 的意义：`kv_head_idx` 对应的单个 K/V tile，会被打包进 Q 行里的全部 `GQA_RATIO` 个 query head 复用。输出侧与输入侧镜像对应，epilogue 之后用匹配的 3D view，把打包行存回 `O[batch, seq, qo_head, dim]`。

结果是，GQA 完全生活在 Q-load 和 O-store 边界上。在内部计算路径中，score MMA 仍然看到一个普通的 `128 x HEAD_DIM` Q tile，其余 tile-primitive 图完全不变。

## Tile 调度

scheduler 的工作是把每个 CTA 映射到一个 `(batch, kv_head, m_block)` attention task；合适的策略取决于 masking 是否让这些 task 代价相同：

- 非 causal mode 使用 `FlashAttentionLinearScheduler`。每个 task 的工作量相同，因此一个固定 CTA 池按 `num_ctas` 前进，就足以把任务均匀摊开。
- Causal mode 使用 `FlashAttentionLPTScheduler`，因为 causal masking 会让工作量极不均匀：靠近开头的 Q block 大约只 attend 一个 K/V block，而靠近结尾的 Q block 会 attend 所有 block。朴素切分会让某些 CTA 远晚于其他 CTA 完成，所以 longest-processing-time scheduler 会优先安排重任务以拉平完成时间，同时仍尽量保持相邻 batch/head task 聚在一起，利于 L2 locality。

尽管两种 scheduler 不同，它们暴露的循环接口完全相同：

```python
while scheduler.valid():
    m_block_idx = scheduler.m_block_idx
    batch_idx = scheduler.batch_idx
    kv_head_idx = scheduler.head_idx
    # 用对应 K/V block 范围处理一个 Q block
    scheduler.next_tile()
```

唯一的行为差异在于 `next_tile()` 做什么：非 causal mode 下，它会让 CTA 前进到另一个 task；causal mode 下，它会在当前 task 后结束循环。无论哪种方式，这都只是调度决策：它选择 CTA 拥有 *哪个* attention tile，而不改变这个 tile 如何计算。循环内部仍然运行同样的本地 primitive：TMA load、score MMA、softmax、value MMA、correction、TMA store。

## 编译与验证

上面的内容都是摘录；要把所有东西合在一起并真正运行 kernel，我们会从 `tirx-kernels` 导入真实实现、编译它，并与 torch reference 对比。完整 kernel 位于 `tirx-kernels` 仓库中的 [`flash_attention4.py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py)，本章讲过的所有部件都组装在这个文件里。它和 GEMM 验证 cell 有两点不同：Flash Attention 的入口更丰富（`get_flash_attention4_kernel`），而且它为内建 profiler 多接收一个 `profiler_buf` 参数。整章只需要运行这一格：

```python
import torch
import torch.nn.functional as F
import tvm
from tirx_kernels.attention.flash_attention4 import (
    get_flash_attention4_kernel, PROFILER_BUFFER_SIZE)

B, S, Hq, Hkv, D = 1, 1024, 32, 8, 128   # GQA：32 个 query head 共享 8 个 KV head
Q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
O = torch.empty(B, S, Hq, D, dtype=torch.float16, device="cuda")
prof = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")

kernel = get_flash_attention4_kernel(B, S, S, Hq, Hkv, D, is_causal=False)
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
ex.mod(Q, K, V, O, prof)   # 和其他章节一样，ex.mod 直接接收 torch tensor
torch.cuda.synchronize()

# torch reference；enable_gqa 允许 32 个 query head 共享 8 个 KV head
qt, kt, vt = (x.transpose(1, 2).float() for x in (Q, K, V))
ref = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True).transpose(1, 2).half()
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)
print(f"FA4: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}, non-causal -> PASS")
```

**预期输出**：`... -> PASS`。kernel 用 fp32 累计 online softmax，但它和高精度 reference 之间仍然存在几类近似：输入和 operand 的 fp16 存储与舍入；基于 `exp2` 的 softmax 重写（把每个指数改写成 `scale_log2 = log2(e)/√d` 形式）；online-softmax 的重排和逐行 rescaling，它按运行尺度分 block 求和，而不是一次性求和；最后还有 writeback 时对 `O` 的 fp16 cast。这里选择的 `rtol`/`atol` 和源 kernel 自己的测试一致，是为了同时覆盖这些因素相对 torch reference 的差异，而不只是单独覆盖 fp16 舍入。因此如果你看到真正失败，而不是边界附近的小偏差，就应该把它当作指向 softmax 路径的路标：漏掉了 `s_ready` / `p_o_rescale` / `p_ready_2` wait，或者 `row_max` / `row_sum` 更新没有被 rescale 步骤正确应用。这些正是本章用 barrier 反复处理的 handoff。

## 与 GEMM 的差异

下表沿发生变化的轴比较 FA4 与 GEMM：

| 方面 | GEMM | Flash Attention 4 |
|------|------|-------------------|
| MMA 阶段 | 一个重复的 MMA | score MMA 和 value MMA |
| MMA 之间的工作 | 除 pipeline handoff 外没有额外工作 | online softmax、masking 和 O rescaling |
| 运行状态 | 只有 accumulator | row max、row sum、O accumulator |
| 主要中间值 | accumulator TMEM tile | S、P 和 O TMEM tile region |
| Warp 角色 | TMA producer、MMA consumer、writeback | TMA load、MMA、softmax、correction、TMA store |
| Barrier | 主要是 load/compute/writeback handoff | 额外的 score/softmax/value/correction handoff |
| 调度单元 | 输出矩阵 tile | attention task：`(batch, kv_head, m_block)` |

这些差异全部可以追溯到本章开头那条结构性变化：第二个 MMA，以及夹在两个 MMA 之间的 softmax。另一方面，底层 TIRx contract 完全没有改变：

- tile primitive 说明哪个 tile 在移动或计算；
- 周围的 scope 说明哪些线程协作；
- layout 说明 tile 住在哪里；
- barrier 说明下一个角色什么时候可以消费它。

所以 FA4 比 GEMM 更难，并不是因为它依赖不同硬件，而是因为 tile 值更多、它们之间的 handoff 也更多。

## 练习

1. 和 GEMM 相比，FA4 在两个 MMA 阶段之间新增了什么 tile handoff？请说出 producer、TMEM tile 和 consumer。
2. 为什么 softmax 要把 numerator tile `P` 写回 TMEM，而不是只把它留在寄存器中供 value MMA 使用？
3. 任选 `p_o_rescale` 或 `p_ready_2`。这个 barrier 精确证明了什么？如果 value MMA 跳过这次等待，可能出什么错？

**试着让你的 agent 做一遍**：任选一个没有注释过的 tile primitive，例如 epilogue 里的 `Tx.copy_async`、fp32 -> fp16 的 `Tx.cast`，或第二段 `gemm_pv` sub-MMA。让它写出 scope / layout / dispatch / handoff 卡片，然后对照源码里的 guard、allocation 和 wait 检查答案。
