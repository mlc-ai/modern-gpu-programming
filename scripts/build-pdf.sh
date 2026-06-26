#!/usr/bin/env bash
#
# build-pdf.sh — build the book as a PDF via Sphinx -> XeLaTeX.
#
# Run from the repo root:  ./scripts/build-pdf.sh
# Output: _build/latex/moderngpuprogrammingformlsys.pdf
#
# Prerequisites (one-time, need admin/brew — see README "PDF build"):
#   - MacTeX/BasicTeX (xelatex, latexmk) on PATH (/Library/TeX/texbin)
#   - librsvg (rsvg-convert) for the SVG figures:           brew install librsvg
#   - LaTeX packages:  sudo tlmgr install latexmk tex-gyre fncychap wrapfig \
#                        capt-of needspace tabulary varwidth titlesec framed upquote
#   - Python deps:     pip install sphinx myst-parser sphinx-copybutton \
#                        sphinxcontrib-svg2pdfconverter
#
# conf.py already sets:  latex_engine="xelatex"  + tex-gyre OTF fonts (BasicTeX
# lacks Sphinx's default FreeFont), and loads sphinxcontrib.rsvgconverter for SVGs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Make sure the TeX toolchain is reachable even from a fresh shell.
export PATH="/Library/TeX/texbin:$PATH"

PDF="_build/latex/moderngpuprogrammingformlsys.pdf"

echo "==> Checking toolchain"
for bin in sphinx-build xelatex latexmk rsvg-convert; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "ERROR: '$bin' not found on PATH. See README 'PDF build' prerequisites." >&2
    exit 1
  fi
done

echo "==> 1/2  Generating LaTeX from Sphinx sources"
sphinx-build -b latex . _build/latex

echo "==> 2/2  Compiling LaTeX -> PDF (xelatex via latexmk)"
( cd _build/latex && latexmk -C >/dev/null 2>&1 || true; make )

if [[ -f "$PDF" ]]; then
  echo
  echo "==> Done: $REPO_ROOT/$PDF"
  echo "    (Open with:  open \"$REPO_ROOT/$PDF\")"
else
  echo "ERROR: PDF was not produced. Check _build/latex/moderngpuprogrammingformlsys.log" >&2
  exit 1
fi
