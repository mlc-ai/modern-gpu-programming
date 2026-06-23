# Sphinx configuration for the Modern GPU Programming For MLSys book.
# Migrated off d2lbook to plain Sphinx + MyST-Parser + sphinx-book-theme.
# Build:  sphinx-build -b html . _build/html

project = "Modern GPU Programming For MLSys"
author = "MLC Community"
copyright = "2026, MLC Community"
release = "0.0.1"

extensions = ["myst_parser", "sphinx_copybutton"]

# Markdown (MyST) is the primary source format.
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"

myst_enable_extensions = [
    "dollarmath",   # $...$ and $$...$$ math
    "amsmath",      # LaTeX environments
    "colon_fence",  # ::: fences
    "deflist",
]
myst_heading_anchors = 3   # auto slug anchors for h1-h3

# Only the toctree-reachable docs are content; keep everything else out so
# Sphinx does not warn about / try to render source, build, and asset files.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "**/README.md",
    "_*.md",
    "**/_*.md",
    "setup.py",
    "tirx_tutorial",
    "references.bib",
    "img/scripts",
    ".git",
    ".github",
]

# --- HTML / theme ---
html_theme = "sphinx_book_theme"
html_title = project
html_logo = "static/mlc-logo-with-text-landscape.svg"
html_favicon = "static/mlc-favicon.ico"
html_static_path = ["static"]
# Interactive slide demos (self-contained HTML+CSS+JS) copied verbatim into the
# site root, then embedded via <iframe>. See chapter_* for the embeds.
html_extra_path = ["_extra"]
html_css_files = ["custom.css", "demo-embed.css"]
html_js_files = ["demo-embed.js"]
html_theme_options = {
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "home_page_in_toc": False,
    "use_download_button": False,
    "use_fullscreen_button": False,
}
