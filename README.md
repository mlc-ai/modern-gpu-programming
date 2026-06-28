# Modern GPU Programming For MLSys

This book teaches modern GPU kernel programming as a progression: **understand the
GPU hardware → learn to program it → write state-of-the-art kernels.** It treats
the Blackwell-class GPU — its memory hierarchy and Tensor Memory, its tensor-core and
asynchronous data-movement engines, warpgroups and clusters — as the real subject. The
vehicle is **TIRx** (Tensor IR next), a Python DSL for writing GPU kernels at the IR level.

📖 **Read it online: <https://mlc.ai/modern-gpu-programming-for-mlsys/>**

**Chinese version: <https://mlc.ai/modern-gpu-programming-for-mlsys/zh/>**

🤝 **Contribute:** Corrections, examples, and improvements are welcome through the
[GitHub repository](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys).

## What's inside

- **Part I — Understanding the GPU.** Execution and memory model, the performance model
  (roofline, overlap), a deep dive into data layout, the memory and compute engines (TMA,
  Tensor Memory, Tensor Cores), asynchronous coordination, and advanced scheduling (CLC).
- **Part II — Programming a GPU with TIRx.** An introduction to TIRx through one runnable
  single-MMA GEMM — scope, layout, and dispatch, and how compilation works — plus the tensor
  layout model (`TileLayout`, named axes, swizzle).
- **Part III — GEMM: Tiled to SOTA.** A tiled GEMM built up through TMA pipelining,
  persistent scheduling, warp specialization, and 2-CTA clusters.
- **Part IV — Flash Attention 4.** A complete attention kernel built from the Part III techniques:
  two MMAs with softmax between them, online-softmax rescaling, causal masking, and GQA.
- **Reference.** TIRx language reference and compiler internals.

## Build the book locally

The book is a [Sphinx](https://www.sphinx-doc.org/) site (Markdown/MyST + reStructuredText):

```bash
pip install -r requirements-docs.txt
sphinx-build -b html . _build/html
```

### Preview

```bash
python -m http.server -d _build/html 8000
```

Open <http://localhost:8000>. On a remote machine the server runs there, so forward the
port — `ssh -L 8000:localhost:8000 user@your-server` — then open the URL locally. (VS Code
Remote SSH auto-forwards it.)

## Running the kernels (requires a Blackwell GPU)

The kernels in this book target Blackwell (`sm_100a`), so running them needs a Blackwell GPU
(such as a B200), the TIRx compiler, and a CUDA build of PyTorch.

**1. Install the TIRx compiler.** It ships as the `tvm.tirx` module of the Apache TVM wheel:

```bash
pip install apache-tvm
```

Verify:

```bash
python -c "import tvm, tvm.tirx; print(tvm.__version__)"
```

**2. Install PyTorch** with a CUDA build matching your GPU (used for the example inputs and the
reference checks) — see <https://pytorch.org>.

**3. (Optional) the reference kernels.** The full GEMM and Flash Attention 4 kernels live in the
companion `tirx-kernels` package (`pip install -e .` from a checkout); run them with, e.g.,
`python -m tirx_kernels.test --kernel fp16_bf16_gemm`.

TIRx parses kernel source via Python source inspection, so examples should live in a file
or notebook cell rather than inside `python -c`.

## Deployment

Every push to `main` is built and published automatically by GitHub Actions
(`.github/workflows/build_deploy.yaml`) to <https://mlc.ai/modern-gpu-programming-for-mlsys/>.
