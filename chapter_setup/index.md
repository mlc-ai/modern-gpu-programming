# Setup
:label:`chap_setup`

This chapter gets your environment ready and verifies it works by running a real GPU kernel.


## Install

### Prerequisites

- **OS**: Linux (Ubuntu 20.04+ recommended)

- **GPU**: NVIDIA Blackwell (B200 / B100) with driver >= 570

- **Python**: >= 3.10 with `pip`

### Install Packages

```bash
python -m pip install --pre -U -f https://mlc.ai/wheels "mlc-ai-tirx-cu130==0.0.1b2"
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install numpy
```


## Your First Kernel

The best way to verify your setup is to compile and run a real TIRX kernel. This minimal kernel doubles every element of an array on the GPU:

```{.python .input}
import tvm
from tvm.script import tirx as Tx
import torch

@Tx.prim_func(tirx=True)
def double_it(A_ptr: Tx.handle, B_ptr: Tx.handle):
    n = Tx.int32()
    A = Tx.match_buffer(A_ptr, [n], "float32")
    B = Tx.match_buffer(B_ptr, [n], "float32")
    with Tx.kernel():
        bx = Tx.cta_id([148], parent="kernel")
        tid = Tx.thread_id([256], parent="cta")
        with Tx.thread():
            i = bx * 256 + tid
            if i < n:
                B[i] = A[i] * 2.0

# Compile
target = tvm.target.Target("cuda -arch=sm_100a")
with target:
    lib = tvm.compile(tvm.IRModule({"main": double_it}), target=target, tir_pipeline="tirx")

# Run
device = torch.device('cuda')  # gpu(0)
a = torch.randn(1024, device=device)
b = torch.zeros(1024, device=device)
lib["main"](tvm.runtime.from_dlpack(a), tvm.runtime.from_dlpack(b))

# Verify
match = torch.allclose(b, a * 2)
print(f"Input:  {a[:5].tolist()}")
print(f"Output: {b[:5].tolist()}")
print(f"Match:  {match}")
assert match, "FAIL: output does not match a * 2"
print("\nSetup verified! Your TIRX environment is working correctly.")
```

If you see `Setup verified!`, you're ready to start the tutorial. If not:

- **`ModuleNotFoundError: No module named 'tvm'`**: Re-run the pip install command above

- **`CUDA error`**: Check that `nvidia-smi` shows a Blackwell GPU and driver >= 570

- **Compilation error**: Make sure you installed the `cu130` version of both TVM and PyTorch


## Inspecting Generated Code

After compiling any TIRX kernel, you can see the generated CUDA source:

```python
cuda_source = lib.mod.imports[0].inspect_source()
print(cuda_source)
```

This is useful for debugging — you can search for specific PTX instructions like `tcgen05.mma`, `mbarrier.init`, or `__syncthreads()` to verify your kernel's structure. We'll use this technique in the GEMM chapters.
