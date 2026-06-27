(zh_chap_warp_spec_debug)=
# 调试 Warp-Specialized Kernel

{ref}`zh_chap_gemm_advanced` 中的 GEMM Steps 7-9 会重叠 TMA load、`tcgen05` MMA，以及 TMEM/SMEM writeback。同样的调试方法也适用于 Flash Attention 的 handoff：先识别角色，再识别每个角色拥有的存储，然后用这个模型去核对生成的 CUDA。

不要一上来就重写 kernel。先确认运行环境是有效的，再检查生成的 CUDA。排除环境和编译期问题之后，这类 kernel 的运行时失败通常都能归结为某个 handoff 断了：barrier 未初始化、arrival count 错误、collective 被藏进 role guard、barrier phase 过期，或者 producer 还没让写入可见就复用了存储。

## 调试 kernel 之前

先排除 runtime context 问题：

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

这些 kernel 面向 Blackwell（`sm_100a`）。如果 Python 导入了过期的 TVM checkout，或者 GPU 不是 Blackwell 级别，请先修正这些问题，再改 kernel。随后运行该 kernel 最小的正确性检查，例如 `run_correctness()`，之后再看性能。

## 调试流程

1. 用仍会失败的最小 shape 复现问题。如果失败是 illegal memory access，请在下一次运行前重启 Python。
2. 如果编译失败，先检查已安装 API、target、`dispatch=` 和 buffer scope，再阅读运行时同步代码。
3. 保存 `inspect_source("cuda")` 输出。在重新读 Python 之前，先搜索 role guard、`mbarrier_init`、`tcgen05`、`cp.async.bulk.tensor` 和 `cta_sync()`。
4. 为失败的 kernel 路径写出 roles / storage / handoff / lifetime 表。
5. 用这张表核对生成的 CUDA：barrier init 是否位于 role 分支之前，TMA producer、MMA issuer、writeback group 是否符合预期，以及 CTA-wide collective 是否没有出现在只覆盖 warpgroup 的分支里。
6. 把运行结果分类为 deadlock、crash、wrong result，或 correct-but-slow，然后使用下方对应小节。
7. 一次只改一个 handoff：init count、arrive/wait phase、role guard、fence、TMA store drain、TMEM alloc/dealloc，或 tile-scheduler advance。
8. 测性能前先重新跑正确性。

## 可以迁移的检查表

对于任何异步 kernel，改代码前先写一个小 worksheet：

| 项目 | 需要写下什么 |
|---|---|
| Roles | 发起每个 async operation 的精确线程、warp、warpgroup 或 CTA。 |
| Storage | 每一步中每个 tile 的存活位置：GMEM、SMEM、TMEM 或 registers。 |
| Handoff | producer、consumer、signal object、arrival count、phase，以及让数据可见的 fence 或 drain。 |
| Lifetime | 每个存储槽位最早可以复用、读回或释放的时刻。 |

然后用生成的 CUDA 核对 worksheet：

- Role guard 与 roles 表匹配。
- Barrier init 出现在带 guard 的 role 分支之前。
- Collective operation 没有被 lane、warp 或 warpgroup guard 意外缩窄。
- Arrive/wait phase 与 handoff 表匹配。
- TMA store drain、TMEM dealloc 和 SMEM 复用只在 lifetime 表允许之后发生。

同一张 worksheet 可用于 TMA->MMA->writeback 的 GEMM pipeline，也可用于 Flash Attention 中的 score/softmax/value/correction handoff。

## 如果编译失败

先修编译期失败，再调试运行时同步：

| 症状 | 可能区域 | 第一项检查 |
|---|---|---|
| 未知 TIRx API 或 attribute error | 安装的 wheel 与教程代码不匹配 | 打印 `tvm.__file__` 和 `tvm.__version__`；把 API 名称与 {ref}`zh_chap_language_reference` 对比。 |
| 不支持的 `dispatch=` | 所选 target 或 primitive 不支持该路径 | 检查 `dispatch` 参数和 target capability；本教程中的 `tcgen05` 路径需要 Blackwell。 |
| Buffer scope 不匹配 | buffer 正通过错误的硬件路径使用 | 检查 worksheet 中的 storage 行：TMEM 必须通过 `tcgen05` 访问，TMA operand 必须使用兼容的 GMEM/SMEM layout。 |
| 编译成功但生成的 CUDA 缺少预期路径 | Dispatch 没有按预期 lower | 改算法前，先在生成的 CUDA 中检查 `tcgen05` 和 `cp.async.bulk.tensor`。 |

## 检查生成代码

对任何编译后的 kernel，都先保存 CUDA，方便搜索和 diff：

```python
from pathlib import Path

cuda_source = ex.mod.imports[0].inspect_source("cuda")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/my_kernel.cu").write_text(cuda_source, encoding="utf-8")
print(cuda_source)
```

生成代码会按下表把 TIRx construct 映射到 CUDA：

| TIRx | 生成的 CUDA |
|------|-------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` 内部 guard | `((int)threadIdx.x) < 1`（仅 CTA thread 0） |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

阅读完整 kernel 之前，先搜索这些字符串：

| 生成的 CUDA | 检查点 |
|---|---|
| `if (threadIdx.x < 1)` | 单 CTA-thread guard，通常用于 barrier 初始化 |
| `mbarrier_init` | barrier 初始化存在，并出现在 role 分支之前 |
| `tcgen05` | Tensor Core 路径已经生成 |
| `cp.async.bulk.tensor` | copy 已经 lower 到 TMA |
| `cta_sync();` | CTA-wide barrier；它不能位于 `wg_id` 分支内部 |

## Step 7 参考骨架

正确编译的 Step 7 kernel 顶层形状应如下。下面的 guard 为了可读性使用 role 名称；在生成 CUDA 中，请搜索上表对应表达式。

```c
// (1) Barrier init：顶层，仅 CTA thread 0
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // 由全部 128 个 WG0 线程 arrive
}

// (2) TMEM alloc：WG0 warp 0，发出 warp 的所有 lane 参与
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) Fence + cta_sync，然后初始化 phase：producer=1，consumer=0

// (4) Warp-specialized loop
if (wg_id == 1 && warp_id == 3 && elect_sync) { /* TMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 1 && warp_id == 0 && elect_sync) { /* MMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 0)                                { /* WB   */ while(valid){ ... next_tile(); } }

// (5) Cleanup：发出 warp，无 lane guard
cta_sync();
if (warp_id == 0) { tcgen05_relinquish_alloc_permit(); tcgen05_dealloc(..., 512); }
```

改算法之前先检查这些点：

- Barrier init 位于顶层，而不是 `wg_id` guard 内部。
- `tcgen05_alloc` 和 `tcgen05_dealloc` 有 warp guard，但没有 lane guard；发出 warp 的所有 lane 都参与。
- TMA 和 MMA loop 都迭代 `K_TILES` 次。
- Phase init 是 producer=`1`，consumer=`0`。

## 症状地图

从症状入手，但把它当作线索，而不是最终诊断：

| 线索 | 可能区域 | 第一项检查 |
|---|---|---|
| Kernel hang，随后 runtime 报 unspecified launch failure | Deadlock | Barrier init 位置、arrival count、`cta_sync()` 位置，以及 `next_tile()` 参与情况 |
| Illegal memory access、XID，或后续无关 CUDA 调用也失败 | Crash / poisoned context | 重启 Python，然后检查 pointer range、storage lifetime 和 collective participation |
| 错误行以 128 行或 tile 大小条带出现 | Sync race 或 tile-index mismatch | Producer/consumer phase、scheduler advance，以及哪个 warpgroup 拥有每个行条带 |
| `NaN` 或明显无效值 | Descriptor、operand setup 或未初始化 accumulation | SMEM/TMEM descriptor setup、swizzle/layout，以及 accumulator 初始化 |
| 有限但有规律的错误值 | 过期或仅部分可见的数据 | 缺少 fence、缺少 TMA store drain，或 lifetime 表允许前复用了存储 |
| 输出正确但没有预期加速 | Dispatch 或资源问题 | 生成的 CUDA 路径、pipeline depth、occupancy 和 register spill |

## 什么时候重启 Python

CUDA 错误不一定会把自身清理干净。发生 illegal memory access、XID 或 “CUDA context poisoned” 错误后，后续无关调用（例如 `torch.randn`）也可能继续失败。测试下一个修复前请重启 Python 进程，否则你可能是在调试上一次 crash，而不是当前代码。

## Deadlock

按顺序检查这些点：

- **Arrival count 与 init count 不匹配。** 常见情况：`MBarrier.init(128)`，但 `arrive` 被 `if warp_id == 0: if lane_id == 0:` guard 住，于是只有 1 个线程 arrive，wait 永远不会返回。

  | Barrier | init(count) | 谁 arrive | Arrivals |
  |---|---|---|---|
  | `TMABar` (tma->mma) | 1 | TMA engine 通过 `arrive(stage, bytes)` | 1 |
  | `TCGen05Bar` (mma->tma, mma->ld) | 1 | MMA warp 通过 `tcgen05.commit` | 1 |
  | `MBarrier` (ld->mma) | 128 | 全部 WG0 线程通过 `arrive` | 128 |

- **Barrier init 嵌套在 `wg_id` guard 内。** `.init()` 会 lower 成 `if threadIdx.x < 1:`，也就是 CTA thread 0。CTA thread 0 位于 WG0，所以 `if wg_id == 1:` 会阻止所有线程运行 init。init 必须位于顶层；用 `inspect_source()` 中的 `grep mbarrier_init` 验证。

- **`cta_sync()` 位于 warpgroup 分支内部。** `cta_sync` 是 `__syncthreads()`，要求所有 CTA 线程到达。若放在 `if wg_id == 0:` 内，WG1 永远到不了。单 warpgroup barrier 请使用 `T.cuda.warpgroup_sync(10)`。

- **某些 consumer-warpgroup 线程跳过了 `tile_scheduler.next_tile()`。** scheduler 跟踪 per-thread state；跳过它的线程可能永远循环。

- **TMA 和 MMA 的 K-tile count 不一致。** 如果 MMA 做的是 `K_TILES - 1` 而不是 `K_TILES`，barrier phase 会漂移，第二个 outer tile 可能 deadlock。

- **`PipelineState` 初始 phase 错误。** Producer 从 `phase=1` 开始，因此第一次 wait 会通过；consumer 从 `phase=0` 开始，因此第一次 wait 会阻塞。如果二者从同一 phase 开始，第一个 handoff 就可能立即 deadlock。

## Crash 与 Context Poisoning

常见原因：

- **`pool.alloc` 位于 `pool.commit()` 之后。** Barrier wrapper 内部会调用 `alloc`。正确顺序是：`tmem_addr -> barrier wrappers -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()`。
- **`tcgen05.alloc` 或 `tcgen05.dealloc` 带 lane guard。** 发出 warp 必须所有 lane 参与。`if lane_id == 0:` 只运行一个线程，属于 undefined behavior。
- **`tcgen05.dealloc` 前缺少 `cta_sync()`。** writeback 仍在读取时，TMEM 被释放。
- **GMEM 或 SMEM 越界访问。** 缩小到一个 tile，检查 scheduler 的 `m_idx` / `n_idx`，并检查当前 shape 是否是 kernel tile 或 cluster tile 的倍数。

## 错误结果

猜测前先按模式分类错误输出。整条行带出错通常指向 producer/consumer phase、tile index 或 role ownership 不匹配。`NaN` 输出通常指向 descriptor setup、operand setup 或未初始化 accumulation。有限但有规律的错误值通常意味着 consumer 读到了旧 tile、部分写入的 tile，或 store 尚未 drain 的数据。

- **`tcgen05.commit` 位于 `elect_sync` 外部。** 32 个线程都会创建 commit group；其中 31 个空 group 会立即 signal mbarrier。TMA 可能在 MMA 读取 SMEM 前覆盖它。
- **TMA store 前缺少 `fence.proxy_async("shared::cta")`。** TMA engine 可能看不到线程写入 SMEM 的结果。
- **TMA store 后缺少 `cp_async.bulk.commit_group()` 加 `wait_group(0)`。** 下一个 tile 可能在 store drain 前复用 Dsmem。
- **Persistent kernel 在 1024x1024 这样的小尺寸下间歇失败。** 更大尺寸可能用更长 K-loop 掩盖 race。重新检查 tile 间 phase reset，以及 TMA-store commit/wait。
- **`fence.after_thread_sync()` 通常不是修复。** MMA-completion mbarrier 已经携带 release-acquire 语义。Steps 8 和 9 会在 writeback 边上保守地加入它，位置在 `mma2ld.wait` 之后、第一次 `tcgen05.ld` 之前；不要在 TMA-to-MMA 边上常规加入。

## 正确但很慢

如果输出正确但性能远低于预期，请使用同样的检查循环：

| 线索 | 可能区域 | 第一项检查 |
|---|---|---|
| 生成的 CUDA 没有 `cp.async.bulk.tensor` | Copy 没有 lower 到 TMA | 检查 `dispatch="tma"`、target capability 和 operand layout |
| 生成的 CUDA 没有 `tcgen05` 路径 | MMA 没有 lower 成 Blackwell Tensor Core 指令 | 检查 `dispatch="tcgen05"`、target capability 和 operand layout |
| TMA 和 MMA 没有重叠 | Pipeline 太浅，或 phase 让 producer/consumer 串行化 | 检查生成 CUDA 中 wait/arrive/advance 的顺序 |
| 小 shape 正确性良好但大 shape 很慢 | Register spill、occupancy 或 staging-buffer 压力 | 检查编译器资源报告；减小 tile size、分块 writeback，或降低 pipeline depth |

## 提交好 issue

如果经过上述检查后问题仍然存在，请先最小化复现，再到 [Apache TVM GitHub 仓库](https://github.com/apache/tvm/issues)提交 issue。请包含：

- `tvm.__file__` / `tvm.__version__` 输出和 GPU capability；
- 能复现失败的最小 shape；
- 失败类型：compile-time、deadlock、crash、wrong result，还是 correct-but-slow；
- 最小 kernel 或 notebook cell，以及对应正确性检查；
- 保存的 `inspect_source("cuda")` 输出，或能展示可疑 guard、barrier、dispatch 路径的最小片段。
