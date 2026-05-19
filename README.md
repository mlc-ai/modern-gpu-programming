# TIRX: High-Performance GPU Kernel Programming

A hands-on tutorial for writing GPU kernels on NVIDIA Blackwell GPUs using TIRX.

## Quickstart

**Read online**: Tutorial website coming soon.

**Run locally** (requires Blackwell GPU):

```bash
pip install --pre -U "https://github.com/mlc-ai/package/releases/download/v0.9.dev0/mlc_ai_tirx_nightly_cu130-0.24.dev0-py3-none-manylinux_2_28_x86_64.whl"
pip install apache-tvm-ffi==0.1.9rc2
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install numpy

# Verify: this should print "TIRX OK"
python -c "from tvm.script import tirx as Tx; print('TIRX OK')"

# Run your first kernel
python -c "
import tvm
from tvm.script import tirx as Tx
import torch

@Tx.prim_func
def hello_gpu(A_ptr: Tx.handle, B_ptr: Tx.handle):
    n = Tx.int32()
    A = Tx.match_buffer(A_ptr, [n], 'float32')
    B = Tx.match_buffer(B_ptr, [n], 'float32')
    with Tx.kernel():
        bx = Tx.cta_id([148])
        tid = Tx.thread_id([256])
        with Tx.thread():
            i = bx * 256 + tid
            if i < n:
                B[i] = A[i] * 2.0

target = tvm.target.Target('cuda')
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

# Build with GPU kernel execution (requires Blackwell GPU)
sed -i 's/eval_notebook = False/eval_notebook = True/' config.ini
CUDA_VISIBLE_DEVICES=0 d2lbook build html
```

### Previewing the Site

After building, start a local HTTP server:

```bash
python -m http.server -d _build/html
```

**If you are on a remote server** (e.g., SSH into a GPU machine), the server runs
on the remote machine so `localhost:8000` won't work in your local browser.
Use SSH port forwarding — connect to your server with:

```bash
ssh -L 8000:localhost:8000 user@your-server
```

Then open `http://localhost:8000` in your local browser. Alternatively, if you
use VS Code Remote SSH, it will auto-forward the port and show a notification
to open in browser.
