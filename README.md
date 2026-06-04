# TIRX: High-Performance GPU Kernel Programming

A hands-on tutorial for writing GPU kernels on NVIDIA Blackwell GPUs using TIRX.

## Quickstart

**Preview locally**: build the site, then open the local server URL from the
preview instructions below.

**Run locally** (requires Blackwell GPU):

> **Installation instructions are being updated.** The TIRX nightly wheel URL and
> pinned package versions move quickly, so the exact `pip install` commands are
> pending a refresh. See the **Environment Setup** chapter for the current status.

Once TIRX is installed, verify the import:

```bash
# Should print "TIRX OK"
python -c "from tvm.script import tirx as Tx; print('TIRX OK')"
```

For a runnable minimal kernel, see the **Environment Setup** chapter. TIRX parses
kernel source with Python's source inspection, so examples should live in a file
or notebook cell rather than inside `python -c`.

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
