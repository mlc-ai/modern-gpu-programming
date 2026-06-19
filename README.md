# Modern GPU Programming

This book teaches modern GPU kernel programming as a progression: **understand the
GPU as a machine → learn to program it → write state-of-the-art kernels.** It treats
the Blackwell-class GPU — its memory hierarchy and Tensor Memory, its tensor-core and
asynchronous data-movement engines, warpgroups and clusters — as the real subject. The
vehicle is **TIRx** (Tensor IR neXt), a Python DSL for writing GPU kernels at the IR level.

📖 **Read it online: <https://mlc.ai/modern-gpu-programming/>**

## What's inside

- **Part I — Understanding the GPU.** Execution and memory model, the performance model
  (roofline, overlap), a deep dive into data layout, the memory and compute engines (TMA,
  Tensor Memory, Tensor Cores), asynchronous coordination, and advanced scheduling (CLC).
- **Part II — Programming a GPU with TIRx.** The TIRx native level — writing device kernels
  directly — and the tensor layout model (`TileLayout`, named axes, swizzle).
- **Part III — GEMM: Tiled to SOTA.** A tiled GEMM built up through TMA pipelining,
  persistent scheduling, warp specialization, and 2-CTA clusters.
- **Part IV — Capstone: Flash Attention.** Composing the whole machine into a real kernel.
- **Part V — Workflow & Practice.** Profiling/debugging and writing kernels with agents.
- **Appendix.** API reference and full source listings.

## Build the book locally

The book is a [Sphinx](https://www.sphinx-doc.org/) site (Markdown/MyST + reStructuredText):

```bash
pip install -r requirements-docs.txt
sphinx-build -b html . _build/html
```

The API reference uses autodoc over `tvm`. If `tvm` isn't importable (a docs-only
machine), it is mocked automatically — the book still builds in full and the API pages
degrade gracefully. Force the mock with `DOCS_MOCK_TVM=1`.

### Preview

```bash
python -m http.server -d _build/html 8000
```

Open <http://localhost:8000>. On a remote machine the server runs there, so forward the
port — `ssh -L 8000:localhost:8000 user@your-server` — then open the URL locally. (VS Code
Remote SSH auto-forwards it.)

## Running the kernels (requires a Blackwell GPU)

> **Installation instructions are being updated.** The TIRx nightly wheel URL and pinned
> package versions move quickly, so the exact `pip install` commands are pending a refresh.

Once TIRx is installed, verify the import:

```bash
python -c "from tvm.script import tirx as T; print('TIRx OK')"
```

TIRx parses kernel source via Python source inspection, so examples should live in a file
or notebook cell rather than inside `python -c`.

## Deployment

Every push to `main` is built and published automatically by GitHub Actions
(`.github/workflows/build_deploy.yaml`) to <https://mlc.ai/modern-gpu-programming/>.
