# Practice Kernel: Fused GELU Gate
:label:`chap_fused_gelu`

Fused GELU-Tanh with multiply is used in MLP gate layers of Transformer models. Elementwise kernels are the smallest TIRX programs that still exercise launch geometry, indexing, and coalesced global-memory access.

**Operation**: The input tensor has shape `[batch, 2*d]`, split into two halves along the last dimension. The first half passes through GELU-tanh activation, then multiplies with the second half:

$$\text{output} = \operatorname{GELU\_tanh}(\text{input}[:, :d]) \cdot \text{input}[:, d:]$$

where GELU-tanh is the **tanh approximation** of GELU (not the exact erf version). This is the variant used in most LLMs (GPT, Llama, etc.) because it's faster to compute:

$$\operatorname{GELU\_tanh}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{2/\pi} \cdot (x + 0.044715 \cdot x^3)\right)\right)$$

The operator fuses GELU activation and gate multiply into one kernel. The **gate** is the second half of the input tensor (`input[:, d:]`). In Transformer MLP layers, the architecture splits the linear projection output into two halves, applies GELU to one half, and multiplies it element-wise with the other half. Without fusion, GELU would write a temporary buffer and a second kernel would multiply with the gate, costing an extra launch and an extra global-memory round trip. With fusion, both operations happen in one pass.


**Topics.**

- Flat 1D thread mapping for a 2D tensor
- Fused GELU-tanh and gate multiply
- Correctness checking against a NumPy reference


## Implementation

The strategy:

1. Assign one thread per output element
2. Each thread computes a global ID (`bx * NUM_THREADS + tid`) and converts it to `(row, col)` using division and modulo
3. Read the corresponding values from both halves of the input
4. Compute GELU-tanh, multiply with the gate, write the result

### Indexing and Gate Layout

GPU threads are launched as a flat 1D grid, but the output data is a 2D matrix `[batch, out_dim]`. Each thread converts its global ID to a `(row, col)` coordinate:

![1D Thread Grid to 2D Matrix Mapping](../img/thread_grid_to_matrix.png)

Each thread handles exactly one output element. The flat `gid` fills the matrix row by row: the first `out_dim` threads cover row 0, the next `out_dim` threads cover row 1, and so on. Because consecutive `gid` values map to consecutive `col` values in the same row, neighboring threads read neighboring addresses — the access is coalesced, which is what lets a warp's 32 loads collapse into a few memory transactions.

```python
gid = bx * NUM_THREADS + tid
row = gid // out_dim
col = gid % out_dim
```

The boundary guard handles the final partially filled CTA:

```python
if gid < total:
    ...
```

The input tensor is `[batch, 2 * out_dim]`. For each output element, the kernel reads both halves of the same row:

```python
input1 = input_buf[row, col]              # activation half
input2 = input_buf[row, col + out_dim]    # gate half
```

`input1` goes through the GELU-tanh approximation, then the result is multiplied by `input2`. The constants are cast with `Tx.float16(...)` so the expression stays in fp16 arithmetic. Because the whole GELU path — including the `x^3` term and the `tanh` — runs in fp16 while the numpy reference computes in fp32, a small but bounded rounding error is expected; that is why the verification tolerance is loose (`0.05`) rather than exact.

```{.python .input}
import math
import numpy as np
import tvm
from tvm.script import tirx as Tx
import torch

def ceildiv(a, b):
    return (a + b - 1) // b
```

```{.python .input}
def fused_gelu_kernel(out_dim, batch_size):
    total = batch_size * out_dim
    NUM_THREADS = 256
    NUM_BLOCKS = ceildiv(total, NUM_THREADS)

    @Tx.prim_func
    def fused_gelu_tanh_multiply(input_cat_ptr: Tx.handle, output_ptr: Tx.handle):
        input_buf = Tx.match_buffer(input_cat_ptr, [batch_size, out_dim * 2], "float16")
        output_buf = Tx.match_buffer(output_ptr, [batch_size, out_dim], "float16")

        Tx.device_entry()
        bx = Tx.cta_id([NUM_BLOCKS])
        tid = Tx.thread_id([NUM_THREADS])

        with Tx.thread():
            gid = bx * NUM_THREADS + tid
            row = gid // out_dim
            col = gid % out_dim

            if gid < total:
                # input_buf is [batch, 2*out_dim]: first half is x, second half is gate
                input1 = input_buf[row, col]
                input2 = input_buf[row, col + out_dim]
                x_cubed = input1 * input1 * input1
                inner = Tx.float16(0.7978845608) * (
                    input1 + Tx.float16(0.044715) * x_cubed)
                gelu_out = Tx.float16(0.5) * input1 * (
                    Tx.float16(1.0) + Tx.tanh(inner))
                output_buf[row, col] = gelu_out * input2

    return fused_gelu_tanh_multiply
```

### Compile, Verify, and Benchmark

```{.python .input}
out_dim = 4096
batch_size = 64
device = torch.device('cuda')  # gpu(0)
target = tvm.target.Target("cuda")

# Compile
kernel = fused_gelu_kernel(out_dim, batch_size)
with target:
    ex = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")

# Run
x_cat = torch.randn(batch_size, out_dim * 2, dtype=torch.float16, device=device)
out = torch.zeros(batch_size, out_dim, dtype=torch.float16, device=device)
ex.mod(x_cat, out)

# Verify against numpy reference
x_np = x_cat.float().cpu().numpy()
x1, x2 = x_np[:, :out_dim], x_np[:, out_dim:]
ref = (0.5 * x1 * (1.0 + np.tanh(np.sqrt(2.0/np.pi) * (x1 + 0.044715 * x1**3))) * x2).astype(np.float16)
max_err = float(np.max(np.abs(out.cpu().numpy().astype(np.float32) - ref.astype(np.float32))))
print(f"Fused GELU: batch_size={batch_size}, out_dim={out_dim}")
print(f"Max error vs numpy reference: {max_err:.6f}")
assert max_err < 0.05, f"FAIL: max_err={max_err}"
print("PASS")
```

### Expected Output & Troubleshooting

**Expected output**:

- `PASS` printed. The kernel runs GELU entirely in fp16 while the reference is fp32, so the typical max error is ~0.005–0.01. The cell asserts a loose guard of `< 0.05`; a `max_err` above ~0.5 indicates a real bug (wrong constant or wrong gate offset), not rounding.

**If something goes wrong**:

- `max_err` near the `0.05` guard or above ~0.5: Check that `Tx.float16(0.7978845608)` is correct (this is sqrt(2/pi))

- Kernel hangs: Check `if gid < total` boundary guard is present

- All zeros: Check that `input_buf[row, col + out_dim]` reads from the second half (gate)


## Exercises

1. How many total threads are launched for `batch_size=64, out_dim=4096`? How many CTAs?

2. Why do we use `ceildiv(total, NUM_THREADS)` instead of `total // NUM_THREADS` for the number of blocks?

3. What would happen if we removed the `if gid < total:` guard?

4. This kernel reads each input element exactly once and writes each output element exactly once. Is it compute-bound or memory-bound? Why?

**Try with your agent**: Paste this kernel and ask it to identify the scope, layout, and dispatch of the elementwise store — which threads cooperate, how the flat `gid` maps to `(row, col)`, and which hardware path the `output_buf[row, col] = ...` write lowers to.
