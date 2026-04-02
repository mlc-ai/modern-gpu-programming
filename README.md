# TIRX: High-Performance GPU Kernel Programming

A hands-on tutorial for writing GPU kernels on NVIDIA Blackwell GPUs using TIRX.

## Quickstart

**Read online**: Visit the [tutorial website](https://mlc.ai/tirx-tutorial/) (no setup needed to read).

**Run locally** (requires Blackwell GPU):

```bash
pip install --pre -U -f https://mlc.ai/wheels "mlc-ai-tirx-cu130==0.0.1b2"
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install numpy

# Verify: this should print "TIRX OK"
python -c "from tvm.script import tirx as Tx; print('TIRX OK')"

# Run your first kernel
python -c "
import tvm
from tvm.script import tirx as Tx
import torch

@Tx.prim_func(tirx=True)
def hello_gpu(A_ptr: Tx.handle, B_ptr: Tx.handle):
    n = Tx.int32()
    A = Tx.match_buffer(A_ptr, [n], 'float32')
    B = Tx.match_buffer(B_ptr, [n], 'float32')
    with Tx.kernel():
        bx = Tx.cta_id([148], parent='kernel')
        tid = Tx.thread_id([256], parent='cta')
        with Tx.thread():
            i = bx * 256 + tid
            if i < n:
                B[i] = A[i] * 2.0

target = tvm.target.Target('cuda -arch=sm_100a')
with target:
    lib = tvm.compile(tvm.IRModule({'main': hello_gpu}), target=target, tir_pipeline='tirx')
a = torch.randn(1024, device='cuda')
b = torch.zeros(1024, device='cuda')
lib['main'](tvm.runtime.from_dlpack(a), tvm.runtime.from_dlpack(b))
print('Result matches:', torch.allclose(b, a * 2))
"
```

## Building the Tutorial Site

For contributors who want to build the HTML site locally:

```bash
pip install git+https://github.com/d2l-ai/d2l-book
conda install pandoc

# Build (without running GPU kernels)
sed -i 's/eval_notebook = True/eval_notebook = False/' config.ini
d2lbook build html

# Preview
python -m http.server -d _build/html

# Build with GPU kernel execution (requires Blackwell GPU)
sed -i 's/eval_notebook = False/eval_notebook = True/' config.ini
CUDA_VISIBLE_DEVICES=0 d2lbook build html
```
