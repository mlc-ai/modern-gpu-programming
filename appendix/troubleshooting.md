(chap_troubleshooting)=
# Troubleshooting

Debugging asynchronous GPU kernels requires isolating the failure before reading the code. A crash usually means a bad pointer or an uninitialized barrier; a hang means a mismatched synchronization point; and silent corruption means a missing fence or a phase-tracking bug.

This appendix gathers the common failure modes for the kernels built in this book. It builds on the performance model in {ref}`chap_performance` and the generated-CUDA skeleton in {ref}`chap_gemm_advanced`.

## Isolating the Environment

Before assuming the kernel logic is flawed, verify the runtime context. A kernel compiled for Blackwell will immediately fail with an unspecified launch failure on Hopper, and a stale import path can cause you to debug the wrong code entirely.

1. **Confirm the imported package:** Run `python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"`. This rules out importing a stale checkout.
2. **Confirm the GPU capability:** Run `python -c "import torch; print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"`. The kernels in this book target Blackwell (`sm_100a`); attempting to launch them on older architectures explains most immediate failures.
3. **Run the smallest correctness check:** Always run the kernel's own correctness check (e.g., `run_correctness()`) against a small shape before trusting any performance numbers.

## Symptom, cause, fix

The tables below group the common failures. Each row is *symptom*, *likely cause*,
and *first check / fix*.

### Install and import

| Symptom | Likely cause | First check / fix |
|---|---|---|
| `import tvm.tirx` fails | wrong or missing wheel | `pip install apache-tvm==0.25.0`; re-run check command 1 |
| Imports the wrong code | a second checkout on `PYTHONPATH` | print `tvm.__file__` and confirm the path |
| Launch fails immediately on any kernel | not a Blackwell GPU | check command 2; the kernels require `sm_100a` |

### Compile time

| Symptom | Likely cause | First check / fix |
|---|---|---|
| "unknown TIRx API" / attribute error | API name drift vs. the installed wheel | look the name up in {ref}`chap_language_reference` and the installed `tvm.tirx` |
| unsupported `dispatch=` | a copy/MMA dispatch the target does not support | confirm the `dispatch` argument against {ref}`chap_gemm_async` |
| buffer scope mismatch | a buffer used outside its `scope` (e.g. a `tmem` buffer read like normal memory) | review the scope rules in {ref}`chap_tmem` |

### Crash at run time

| Symptom | Likely cause | First check / fix |
|---|---|---|
| `illegal memory access` | out-of-range index or wrong layout/offset | shrink to one tile; the CUDA context is now poisoned, so see [When to restart Python](#when-to-restart-python) |
| `unspecified launch failure` | a barrier left uninitialized, or a resource over-allocated | check that every `mbarrier` is initialized at top level (see the pitfalls below) |
| Hang / deadlock | a collective barrier that not all expected threads reach | check for `cta_sync()` inside a warpgroup branch (see below) |

### Wrong result (compiles and runs, bad numbers)

These are the dangerous ones: no error, just a wrong answer. Each links to where
the book works through it.

| Symptom | Likely cause | First check / fix |
|---|---|---|
| Wrong after the first K chunk | a reused mbarrier with the phase not flipped (`phase ^= 1` dropped), so a wait returns *before* its MMA finished | {ref}`chap_gemm_basics` walks through the phase-tracking table |
| Hang once warpgroups specialize | `cta_sync()` (`__syncthreads()`) inside a `wg_id` branch, not all CTA threads arrive | use `T.cuda.warpgroup_sync(10)` for a single-warpgroup barrier ({ref}`chap_gemm_advanced`) |
| Garbage / uninitialized accumulator | `mbarrier` `.init()` nested inside a `wg_id` guard, so no thread satisfies it and the barrier is never initialized | `.init()` must be at top level; `mbarrier_init` should appear once near the top of the generated CUDA |
| Corruption near the epilogue / TMEM | missing `cta_sync()` before `tcgen05.dealloc`, so TMEM is freed while writeback still reads it | add the `cta_sync()` before dealloc ({ref}`chap_gemm_advanced`) |
| TMA store writes stale data | missing `T.ptx.fence.proxy_async("shared::cta")` before a TMA store, so the engine does not see thread-written SMEM | fence before the store; see the producer-to-engine handoff in {ref}`chap_async_barriers` |

### Slow (correct but underperforming)

| Symptom | Likely cause | First check / fix |
|---|---|---|
| Generated CUDA has no TMA | a copy that did not lower to `cp.async.bulk.tensor` | grep the generated CUDA for the TMA intrinsic; check the `dispatch="tma"` path |
| Tensor pipe underutilized | too-shallow pipeline / poor overlap | revisit the overlap argument in {ref}`chap_performance` |
| Register spill | tile or unroll too large for the register budget | check the `ptxas` resource report; reduce tile/unroll |

## Inspecting the generated CUDA

The single most useful debugging move is to read what the compiler actually
emitted. Every built module exposes it:

```python
# Default (no argument) prints the lowered source:
print(ex.mod.imports[0].inspect_source())

# Ask for CUDA specifically, and save it so you can search and diff it:
cuda_source = ex.mod.imports[0].inspect_source("cuda")
with open("artifacts/my_kernel.cu", "w") as f:
    f.write(cuda_source)
```

A few patterns are worth recognizing when you scan that output:

| In the generated CUDA | Means |
|---|---|
| `if (threadIdx.x < 1)` | a single-thread guard, e.g. an elected issuer or an `mbarrier` `.init()` |
| `mbarrier_init` | an mbarrier initialization; should be at top level, before any branch |
| `cta_sync();` (`__syncthreads()`) | a CTA-wide barrier and shared-memory ordering point |
| a `tcgen05` MMA call | the Blackwell Tensor Core path was actually generated |
| a `cp.async.bulk.tensor` (TMA) call | the copy lowered to TMA rather than a thread copy |

Use the correct-kernel skeleton in {ref}`chap_gemm_advanced` as the reference
shape, and watch for things that should **not** appear: every lane issuing an MMA,
a `cta_sync()` inside a `wg_id` branch, a missing TMA commit/wait, or a missing
`tcgen05` wait before the accumulator is read.

## When to restart Python

A CUDA error does not always clean up after itself. After an `illegal memory
access` or any "CUDA context poisoned" error, the context is in an undefined state
and **later unrelated calls (even `torch.randn`) may keep failing**. When that
happens, restart the Python process before drawing any conclusions.

## Filing a good issue

This guide covers the most common patterns, but you may encounter issues not listed here. If you are still stuck, please file an issue on the [Apache TVM GitHub repository](https://github.com/apache/tvm/issues). To help others reproduce and debug the problem, please include the environment details (the TVM path and GPU capability commands above) and a minimal reproducible example at a small shape.
