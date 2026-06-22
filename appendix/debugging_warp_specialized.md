(chap_warp_spec_debug)=
# Debugging Warp-Specialized Kernels

GEMM Steps 7-9 in {ref}`chap_gemm_advanced` overlap TMA load, `tcgen05` MMA, and TMEM/SMEM writeback. When one of them hangs, crashes, or returns wrong rows, the cause is usually a broken handoff: an uninitialized barrier, the wrong arrival count, a collective hidden inside a role guard, or a staging buffer reused before the TMA store has drained.

Do not start by rewriting the kernel. First make sure the run is valid, then inspect the generated CUDA. Most bugs show up as a wrong guard, a misplaced barrier init, or a missing store-drain wait.

## Before Debugging the Kernel

Rule out the environment first:

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

These kernels target Blackwell (`sm_100a`). If Python imports a stale TVM checkout, or the GPU is not Blackwell-class, fix that before changing the kernel. Then run the kernel's smallest correctness check, such as `run_correctness()`, before looking at performance.

## Debugging Workflow

1. Reproduce the failure at the smallest shape that still fails. If the failure is an illegal memory access, restart Python before the next run.
2. Save `inspect_source("cuda")` output and search it before reading the Python again.
3. Check the generated skeleton: barrier inits before role branches, the expected TMA producer, MMA issuer(s), and writeback group(s).
4. Classify the run as a deadlock, crash, or wrong result, then use the matching section below.
5. Change one handoff at a time: init count, arrive/wait phase, role guard, fence, TMA store drain, or TMEM alloc/dealloc.
6. Re-run correctness before measuring performance.

## What Transfers

For another asynchronous kernel, keep the same debugging shape and replace the names. Write down four things before changing code:

| Item | Question |
|---|---|
| Roles | Which threads, warps, warpgroups, or CTAs issue each operation? |
| Storage | Where does each in-flight tile live: SMEM, TMEM, registers, or GMEM? |
| Handoff | Which producer signals which consumer, with what barrier count, phase, and fence? |
| Lifetime | When can the storage slot be reused or freed? |

Then check the generated CUDA against that table. A deadlock usually means a missing or mismatched signal. A crash usually means an invalid lifetime or collective participation bug. A structured wrong answer usually means the signal happened before the data was visible or before the store drained. This is the same pattern behind TMA->MMA->writeback GEMM pipelines and the score/softmax/value handoffs in FlashAttention.

## Inspecting Generated Code

For any compiled kernel:

```python
cuda_source = ex.mod.imports[0].inspect_source("cuda")
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
| `mbarrier_init` | Barrier initialization exists and appears before the role branches |
| `tcgen05` | The Tensor Core path was generated |
| `cp.async.bulk.tensor` | The copy lowered to TMA |
| `cta_sync();` | CTA-wide barrier; it must not sit inside a `wg_id` branch |

## Reference: Generated CUDA Skeleton

A correctly compiled Step 7 kernel has this top-level shape:

```c
// (1) Barrier inits: top level, CTA thread 0 only
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // arrived by all 128 WG0 threads
}

// (2) TMEM alloc: WG0 warp 0, all 32 lanes (no lane guard)
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) Fences + cta_sync, then phase init: producer=1, consumer=0

// (4) Warp-specialized loop
if (wg_id == 1 && warp_id == 3 && elect_sync) { /* TMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 1 && warp_id == 0 && elect_sync) { /* MMA  */ while(valid){ ... next_tile(); } }
if (wg_id == 0)                                { /* WB   */ while(valid){ ... next_tile(); } }

// (5) Cleanup: warp 0, no lane guard
cta_sync();
if (warp_id == 0) { tcgen05_relinquish_alloc_permit(); tcgen05_dealloc(..., 512); }
```

Check these before changing the algorithm:

- Barrier inits sit at top level, not inside a `wg_id` guard.
- `tcgen05_alloc` and `tcgen05_dealloc` have a warp guard but no lane guard; all 32 lanes participate.
- TMA and MMA loops both iterate `K_TILES` times.
- Phase init is producer=`1`, consumer=`0`.

## Symptom Map

Start from the symptom:

| Symptom | Failure class | Where to look |
|---|---|---|
| Kernel hangs ~30 s, then "unspecified launch failure" | Deadlock | *Deadlocks* |
| Crash within ms; subsequent `torch.randn` also fails | XID 43 / illegal memory access | *Crashes* |
| Mismatched output rows appear in 128-row stripes (128 / 256 / 384 rows) | Sync race | *Wrong Results* |
| Output `NaN` everywhere | MMA descriptor / TMA descriptor mismatch | *Wrong Results* |

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

- **`PipelineState` initial phase wrong.** Producer must start at `phase=1` so the first wait passes; consumer must start at `phase=0` so the first wait blocks. Same starting phase causes an immediate deadlock.

## Crashes (XID 43 / Illegal Memory Access)

A crash leaves the CUDA context unreliable. If a later unrelated call such as `torch.randn` fails too, restart Python before testing the next fix.

Common causes:

- **`pool.alloc` after `pool.commit()`.** Barrier wrappers call `alloc` internally. Correct order: `tmem_addr -> barrier wrappers -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()`.
- **`tcgen05.alloc` or `tcgen05.dealloc` with a lane guard.** They require all 32 lanes of the warp to participate. `if lane_id == 0:` runs one thread, which is undefined behavior.
- **Missing `cta_sync()` before `tcgen05.dealloc`.** TMEM is freed while writeback is still reading.

## Wrong Results

Classify wrong output by pattern. Exact multiples of 128 wrong rows point to a sync race: a whole warpgroup-sized stripe is wrong because a handoff slipped.

- **`tcgen05.commit` outside `elect_sync`.** All 32 threads create commit groups; the 31 empty groups signal the mbarrier immediately. TMA can overwrite SMEM before MMA reads it.
- **Missing `fence.proxy_async("shared::cta")` before TMA store.** The TMA engine may not see SMEM writes from threads.
- **Missing `cp_async.bulk.commit_group()` plus `wait_group(0)` after TMA store.** The next tile can reuse Dsmem before the store drains.
- **Persistent kernel fails intermittently at small sizes such as 1024x1024.** Larger sizes can mask the race with longer K-loops. Re-check phase reset between tiles and the TMA-store commit/wait.
- **`fence.after_thread_sync()` is usually not the fix.** The MMA-completion mbarrier already carries release-acquire semantics. Steps 8 and 9 add it conservatively on the writeback edge, after `mma2ld.wait` and before the first `tcgen05.ld`; do not add it routinely on the TMA-to-MMA edge.
