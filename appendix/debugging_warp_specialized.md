(chap_warp_spec_debug)=
# Debugging Warp-Specialized Kernels

GEMM Steps 7-9 in {ref}`chap_gemm_advanced` overlap TMA load, `tcgen05` MMA, and TMEM/SMEM writeback. The same debugging method applies to Flash Attention handoffs: identify the roles, identify the storage each role owns, then verify the generated CUDA against that model.

Do not start by rewriting the kernel. First make sure the run is valid, then inspect the generated CUDA. After environment and compile-time issues are ruled out, runtime failures in these kernels usually reduce to a broken handoff: an uninitialized barrier, the wrong arrival count, a collective hidden inside a role guard, a stale barrier phase, or storage reused before the producer has made its writes visible.

## Before Debugging the Kernel

Rule out the runtime context first:

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

These kernels target Blackwell (`sm_100a`). If Python imports a stale TVM checkout, or the GPU is not Blackwell-class, fix that before changing the kernel. Then run the kernel's smallest correctness check, such as `run_correctness()`, before looking at performance.

## Debugging Workflow

1. Reproduce the failure at the smallest shape that still fails. If the failure is an illegal memory access, restart Python before the next run.
2. If compilation fails, check the installed API, target, `dispatch=`, and buffer scopes before reading the runtime synchronization code.
3. Save `inspect_source("cuda")` output. Search it for role guards, `mbarrier_init`, `tcgen05`, `cp.async.bulk.tensor`, and `cta_sync()` before reading the Python again.
4. Write the roles / storage / handoff / lifetime table for the kernel path that failed.
5. Check the generated CUDA against that table: barrier inits before role branches, expected TMA producer, MMA issuer(s), writeback group(s), and no CTA-wide collective inside a warpgroup-only branch.
6. Classify the run as a deadlock, crash, wrong result, or correct-but-slow run, then use the matching section below.
7. Change one handoff at a time: init count, arrive/wait phase, role guard, fence, TMA store drain, TMEM alloc/dealloc, or tile-scheduler advance.
8. Re-run correctness before measuring performance.

## What Transfers

For any asynchronous kernel, make a small worksheet before changing code:

| Item | What to write down |
|---|---|
| Roles | The exact threads, warps, warpgroups, or CTAs that issue each async operation. |
| Storage | The live location of each tile at each step: GMEM, SMEM, TMEM, or registers. |
| Handoff | The producer, the consumer, the signal object, the arrival count, the phase, and the fence or drain that makes the data visible. |
| Lifetime | The earliest point where each storage slot can be reused, read back, or freed. |

Then verify the generated CUDA against the worksheet:

- Role guards match the roles table.
- Barrier inits appear before guarded role branches.
- Collective operations are not accidentally narrowed by lane, warp, or warpgroup guards.
- Arrive/wait phases match the handoff table.
- TMA store drains, TMEM dealloc, and SMEM reuse happen only after the lifetime table says they are legal.

Use the same worksheet for TMA->MMA->writeback GEMM pipelines and for the score/softmax/value/correction handoffs in Flash Attention.

## If Compilation Fails

Fix compile-time failures before debugging runtime synchronization:

| Symptom | Likely area | First check |
|---|---|---|
| Unknown TIRx API or attribute error | Installed wheel does not match the tutorial code | Print `tvm.__file__` and `tvm.__version__`; compare the API name with {ref}`chap_language_reference`. |
| Unsupported `dispatch=` | The selected target or primitive does not support that path | Check the `dispatch` argument and target capability; `tcgen05` paths in this tutorial require Blackwell. |
| Buffer scope mismatch | A buffer is being used through the wrong hardware path | Check the storage row of the worksheet: TMEM must be accessed through `tcgen05`, and TMA operands must use compatible GMEM/SMEM layouts. |
| Compile succeeds but generated CUDA lacks the expected path | Dispatch did not lower the way you expected | Inspect the generated CUDA for `tcgen05` and `cp.async.bulk.tensor` before changing the algorithm. |

## Inspecting Generated Code

For any compiled kernel, save the CUDA so you can search and diff it:

```python
from pathlib import Path

cuda_source = ex.mod.imports[0].inspect_source("cuda")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/my_kernel.cu").write_text(cuda_source, encoding="utf-8")
print(cuda_source)
```

The generated code maps TIRx constructs to CUDA like this:

| TIRx | Generated CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` internal guard | `((int)threadIdx.x) < 1` (CTA thread 0 only) |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

Scan for these strings before reading the full kernel:

| Generated CUDA | Check |
|---|---|
| `if (threadIdx.x < 1)` | Single CTA-thread guard, often barrier initialization |
| `mbarrier_init` | Barrier initialization exists and appears before role branches |
| `tcgen05` | The Tensor Core path was generated |
| `cp.async.bulk.tensor` | The copy lowered to TMA |
| `cta_sync();` | CTA-wide barrier; it must not sit inside a `wg_id` branch |

## Step 7 Reference Skeleton

A correctly compiled Step 7 kernel has this top-level shape. The guards below are written with role names for readability; in generated CUDA, search for the corresponding expressions from the table above.

```c
// (1) Barrier inits: top level, CTA thread 0 only
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // arrived by all 128 WG0 threads
}

// (2) TMEM alloc: WG0 warp 0, all lanes of the issuing warp
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) Fences + cta_sync, then phase init: producer=1, consumer=0

// (4) Warp-specialized loop
if (wg_id == 1 && warp_id == 3 && elect_sync) { /* TMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 1 && warp_id == 0 && elect_sync) { /* MMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 0)                                { /* WB   */ while(valid){ ... next_tile(); } }

// (5) Cleanup: issuing warp, no lane guard
cta_sync();
if (warp_id == 0) { tcgen05_relinquish_alloc_permit(); tcgen05_dealloc(..., 512); }
```

Check these before changing the algorithm:

- Barrier inits sit at top level, not inside a `wg_id` guard.
- `tcgen05_alloc` and `tcgen05_dealloc` have a warp guard but no lane guard; all lanes of the issuing warp participate.
- TMA and MMA loops both iterate `K_TILES` times.
- Phase init is producer=`1`, consumer=`0`.

## Symptom Map

Start from the symptom, but treat it as a clue rather than a final diagnosis:

| Clue | Likely area | First check |
|---|---|---|
| Kernel hangs, then the runtime reports an unspecified launch failure | Deadlock | Barrier init placement, arrival count, `cta_sync()` placement, and `next_tile()` participation |
| Illegal memory access, XID, or later unrelated CUDA calls also fail | Crash / poisoned context | Restart Python, then check pointer ranges, storage lifetime, and collective participation |
| Wrong rows appear in 128-row or tile-sized stripes | Sync race or tile-index mismatch | Producer/consumer phases, scheduler advance, and which warpgroup owns each row stripe |
| `NaN` or obviously invalid values | Descriptor, operand setup, or uninitialized accumulation | SMEM/TMEM descriptor setup, swizzle/layout, and accumulator initialization |
| Finite but patterned wrong values | Stale or partially visible data | Missing fence, missing TMA store drain, or storage reused before the lifetime table allows it |
| Correct output but no expected speedup | Dispatch or resource issue | Generated CUDA path, pipeline depth, occupancy, and register spill |

## When to Restart Python

A CUDA error does not always clean up after itself. After an illegal memory access, XID, or "CUDA context poisoned" error, later unrelated calls such as `torch.randn` may keep failing. Restart the Python process before testing the next fix, otherwise you may be debugging the previous crash instead of the current code.

## Deadlocks

Check these in order:

- **Arrival count does not match init count.** Common case: `MBarrier.init(128)` but `arrive` is guarded by `if warp_id == 0: if lane_id == 0:`, so only 1 thread arrives and the wait never returns.

  | Barrier | init(count) | Who arrives | Arrivals |
  |---|---|---|---|
  | `TMABar` (tma->mma) | 1 | TMA engine via `arrive(stage, bytes)` | 1 |
  | `TCGen05Bar` (mma->tma, mma->ld) | 1 | MMA warp via `tcgen05.commit` | 1 |
  | `MBarrier` (ld->mma) | 128 | All WG0 threads via `arrive` | 128 |

- **Barrier init nested inside a `wg_id` guard.** `.init()` lowers to `if threadIdx.x < 1:`, meaning CTA thread 0. CTA thread 0 lives in WG0, so `if wg_id == 1:` prevents every thread from running the init. Inits must be at top level; `grep mbarrier_init` in `inspect_source()` to verify.

- **`cta_sync()` inside a warpgroup branch.** `cta_sync` is `__syncthreads()`, which requires all CTA threads. Inside `if wg_id == 0:`, WG1 never reaches it. Use `T.cuda.warpgroup_sync(10)` for a single-warpgroup barrier.

- **`tile_scheduler.next_tile()` skipped by some consumer-warpgroup threads.** The scheduler tracks per-thread state; threads that skip it can loop forever.

- **TMA and MMA disagree on K-tile count.** If MMA does `K_TILES - 1` instead of `K_TILES`, barrier phases drift and the second outer tile deadlocks.

- **`PipelineState` initial phase wrong.** Producer starts at `phase=1` so the first wait passes; consumer starts at `phase=0` so the first wait blocks. If both start from the same phase, the first handoff can deadlock immediately.

## Crashes and Context Poisoning

Common causes:

- **`pool.alloc` after `pool.commit()`.** Barrier wrappers call `alloc` internally. Correct order: `tmem_addr -> barrier wrappers -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()`.
- **`tcgen05.alloc` or `tcgen05.dealloc` with a lane guard.** The issuing warp must participate with all lanes. `if lane_id == 0:` runs one thread, which is undefined behavior.
- **Missing `cta_sync()` before `tcgen05.dealloc`.** TMEM is freed while writeback is still reading.
- **Out-of-range GMEM or SMEM access.** Shrink to one tile, check the scheduler's `m_idx` / `n_idx`, and check that the current shape is a multiple of the kernel's tile or cluster tile.

## Wrong Results

Classify wrong output by pattern before guessing. Whole row stripes often point to a producer/consumer phase, tile-index, or role-ownership mismatch. `NaN` output often points to descriptor setup, operand setup, or uninitialized accumulation. Finite but patterned wrong values often mean the consumer read an old tile, a partially written tile, or data whose store had not drained yet.

- **`tcgen05.commit` outside `elect_sync`.** All 32 threads create commit groups; the 31 empty groups signal the mbarrier immediately. TMA can overwrite SMEM before MMA reads it.
- **Missing `fence.proxy_async("shared::cta")` before TMA store.** The TMA engine may not see SMEM writes from threads.
- **Missing `cp_async.bulk.commit_group()` plus `wait_group(0)` after TMA store.** The next tile can reuse Dsmem before the store drains.
- **Persistent kernel fails intermittently at small sizes such as 1024x1024.** Larger sizes can mask the race with longer K-loops. Re-check phase reset between tiles and the TMA-store commit/wait.
- **`fence.after_thread_sync()` is usually not the fix.** The MMA-completion mbarrier already carries release-acquire semantics. Steps 8 and 9 add it conservatively on the writeback edge, after `mma2ld.wait` and before the first `tcgen05.ld`; do not add it routinely on the TMA-to-MMA edge.

## Correct but Slow

If the output is correct but performance is far below expectation, use the same inspection loop:

| Clue | Likely area | First check |
|---|---|---|
| Generated CUDA has no `cp.async.bulk.tensor` | Copy did not lower to TMA | Check `dispatch="tma"`, target capability, and operand layout |
| Generated CUDA has no `tcgen05` path | MMA did not lower to Blackwell Tensor Core instructions | Check `dispatch="tcgen05"`, target capability, and operand layouts |
| TMA and MMA do not overlap | Pipeline too shallow or phases serialize producer/consumer | Inspect the order of wait/arrive/advance in the generated CUDA |
| Good small-shape correctness but poor large-shape speed | Register spill, occupancy, or staging-buffer pressure | Check the compiler resource report; reduce tile size, chunk writeback, or lower pipeline depth |

## Filing a Good Issue

If the failure survives the checks above, reduce it before filing an issue on the [Apache TVM GitHub repository](https://github.com/apache/tvm/issues). Include:

- the `tvm.__file__` / `tvm.__version__` output and GPU capability;
- the smallest shape that reproduces the failure;
- whether the failure is compile-time, deadlock, crash, wrong result, or correct-but-slow;
- the minimal kernel or notebook cell plus its correctness check;
- the saved `inspect_source("cuda")` output, or the smallest excerpt that shows the suspicious guard, barrier, or dispatch path.
