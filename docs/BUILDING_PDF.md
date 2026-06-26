# Building the book as a PDF

The book is a Sphinx + MyST-Markdown project that normally builds to **HTML**
(`sphinx-build -b html . _build/html`). This doc explains how to also produce a
**PDF** via the LaTeX builder.

> TL;DR — once prerequisites are installed:
> ```bash
> ./scripts/build-pdf.sh
> open _build/latex/moderngpuprogrammingformlsys.pdf
> ```

## How it works

`Sphinx (LaTeX builder) -> .tex -> xelatex (latexmk) -> PDF`

We use **XeLaTeX**, not the default pdfLaTeX, because the text contains Unicode
symbols (`≈ × ≤ → …`) that pdfLaTeX cannot typeset. SVG figures are converted to
PDF on the fly via `rsvg-convert`.

`conf.py` already contains the needed settings:

```python
latex_engine = "xelatex"
latex_elements = {
    "fontpkg": r"""
\setmainfont{texgyretermes-regular.otf}[ ... ]
\setsansfont{texgyreheros-regular.otf}[ ... ]
\setmonofont{texgyrecursor-regular.otf}[ ... ]
""",
}
# extensions list also includes "sphinxcontrib.rsvgconverter"
```

The tex-gyre fonts are referenced **by OTF filename** because a minimal TeX
install (BasicTeX) doesn't ship Sphinx's default GNU FreeFont, and the font
database doesn't always resolve tex-gyre by family name.

## Prerequisites (one-time)

These need admin rights (`brew`, `sudo tlmgr`) and a fair amount of download.

### 1. Python build deps
```bash
pip install sphinx myst-parser sphinx-copybutton sphinxcontrib-svg2pdfconverter
```

### 2. A TeX distribution (provides xelatex + latexmk)
```bash
brew install --cask basictex     # ~100 MB minimal TeX  (or 'mactex' for the full ~5 GB)
# put TeX on PATH for the current shell:
eval "$(/usr/libexec/path_helper)"
export PATH="/Library/TeX/texbin:$PATH"
```

### 3. LaTeX packages Sphinx's output needs
```bash
sudo tlmgr update --self
sudo tlmgr install latexmk tex-gyre fncychap wrapfig capt-of needspace \
                   tabulary varwidth titlesec framed upquote
```
If a later build complains `LaTeX Error: File 'xxx.sty' not found`, install it:
`sudo tlmgr install xxx`.

### 4. SVG -> PDF converter (for the figures)
```bash
brew install librsvg              # provides rsvg-convert
```

## Build

```bash
./scripts/build-pdf.sh
```

or manually:

```bash
export PATH="/Library/TeX/texbin:$PATH"
sphinx-build -b latex . _build/latex
cd _build/latex && make           # latexmk drives xelatex (multiple passes for TOC/refs)
```

**Output:** `_build/latex/moderngpuprogrammingformlsys.pdf`

`_build/` is gitignored, so the PDF is a local artifact — it is not committed.

## Known limitations of the PDF

1. **Interactive demos are missing.** The book embeds self-contained HTML/JS
   slide demos via `<iframe>` (the `_extra/` content + `html_extra_path`). These
   are live web content and **cannot render in a PDF** — expect blank/absent
   regions where they appear in the HTML.
2. **Some box-drawing / special glyphs may be missing.** The build logs
   `Missing character` warnings for glyphs like `─` (U+2500) used in some
   ASCII-art / code blocks, because the tex-gyre fonts lack them. The PDF still
   builds; those glyphs just render as gaps. To recover them, add a Unicode-rich
   monospace fallback in `conf.py`'s `fontpkg` (e.g. DejaVu Sans Mono / Symbola).
3. **First-pass reference warnings are normal.** "Latex failed to resolve N
   reference(s)" appears mid-build; latexmk re-runs to resolve them. Only a
   non-zero final exit / "no output PDF" indicates a real failure — check
   `_build/latex/moderngpuprogrammingformlsys.log`.

## Troubleshooting

| Symptom in the `.log` | Fix |
| --- | --- |
| `Unicode character ... not set up for use with LaTeX` | Ensure `latex_engine = "xelatex"` (not pdflatex). |
| `fontspec Error: The font "..." cannot be found` | Use the tex-gyre **OTF filenames** (as in `conf.py`), or `sudo tlmgr install tex-gyre`. |
| `File 'xxx.sty' not found` | `sudo tlmgr install xxx`. |
| SVG figure errors / blank figures | `brew install librsvg` and confirm `sphinxcontrib.rsvgconverter` is in `extensions`. |

## Note on conf.py

`conf.py` currently defines `extensions = [...]` **twice** — the second
assignment (which adds `sphinxcontrib.rsvgconverter`) wins. It works, but is
fragile; consider consolidating into a single `extensions` list.
