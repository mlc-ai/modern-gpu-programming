# Environment Setup
:label:`chap_setup`

Set up TVM/TIRX first, then verify the install with a tiny GPU kernel.

## Requirements

- **OS**: Linux, Ubuntu 20.04 or newer recommended.
- **GPU**: NVIDIA Blackwell, such as B200 or B100.
- **Driver**: a CUDA 13.0-capable Linux driver (see the [NVIDIA CUDA 13.0 release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/) for the exact minimum version for your platform).
- **Python**: 3.10 or newer.

## Install Packages

> **Installation instructions are being updated.** The TIRX nightly wheel URL and pinned package versions move quickly, so the exact `pip install` commands are pending a refresh and will be added here.

If you are working from source, use the local `tvm` checkout and keep `tirx-kernels` updated to the same development window.

## Minimal Kernel

The smallest useful verification is a one-dimensional GPU kernel. It checks the current TIRX entry-point style:

- `@Tx.prim_func` defines one TIRX primitive function.
- `Tx.device_entry()` marks the start of device code.
- `Tx.cta_id([...])` names the CTA index and launch extent.
- `Tx.thread_id([...])` names the thread index inside a CTA.
- `with Tx.thread():` marks per-thread work.

```{.python .input}
import torch
import tvm
from tvm.script import tirx as Tx

BLOCK = 256
```

```{.python .input}
@Tx.prim_func
def double_it(a_ptr: Tx.handle, b_ptr: Tx.handle):
    n = Tx.int32()
    A = Tx.match_buffer(a_ptr, [n], "float32")
    B = Tx.match_buffer(b_ptr, [n], "float32")

    Tx.device_entry()
    bx = Tx.cta_id([(n + BLOCK - 1) // BLOCK])
    tid = Tx.thread_id([BLOCK])

    with Tx.thread():
        i = bx * BLOCK + tid
        if i < n:
            B[i] = A[i] * 2.0
```

Compile and run:

```{.python .input}
target = tvm.target.Target("cuda")
with target:
    ex = tvm.compile(tvm.IRModule({"main": double_it}), target=target, tir_pipeline="tirx")

n = 1024
a = torch.arange(n, dtype=torch.float32, device="cuda")
b = torch.empty_like(a)
ex.mod(a, b)

torch.testing.assert_close(b, a * 2)
print("PASS")
```

## Inspect Generated CUDA

After compiling, inspect the lowered CUDA source:

```python
cuda_source = ex.mod.imports[0].inspect_source("cuda")
print(cuda_source)
```

For this kernel, search for `blockIdx.x`, `threadIdx.x`, and the final store. For GEMM and Flash Attention kernels, the same inspection method is useful for checking guards, barriers, `cp.async.bulk`, and `tcgen05` instructions.

## Troubleshooting

- **Package URL returns 404**: check that the wheel version is `0.25.dev0`, not the older `0.24.dev0`.
- **CUDA error at launch**: check `nvidia-smi`, the driver version, and that the process is running on a Blackwell GPU.
- **`inspect_source` has no CUDA import**: make sure the target is `cuda` and `tir_pipeline="tirx"` was passed to `tvm.compile`.
- **Import error for TVM FFI**: update `apache-tvm-ffi` and restart the Python process.
