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
templates_path = ["_templates"]
html_static_path = ["static"]
# Interactive slide demos (self-contained HTML+CSS+JS) copied verbatim into the
# site root, then embedded via <iframe>. See chapter_* for the embeds.
html_extra_path = ["_extra", "_extra_zh"]
html_css_files = ["custom.css", "demo-embed.css"]
html_js_files = ["demo-embed.js", "lang-switch.js"]
html_theme_options = {
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "home_page_in_toc": False,
    "use_download_button": False,
    "use_fullscreen_button": False,
}


def _subtree_toctree_html(app, pagename, root_docname, **kwargs):
    """Render sidebar navigation from a non-root toctree."""
    from bs4 import BeautifulSoup
    from sphinx.addnodes import toctree as TocTreeNode
    from sphinx.environment.adapters.toctree import _resolve_toctree
    from pydata_sphinx_theme.toctree import add_collapse_checkboxes

    doctree = app.env.get_doctree(root_docname)
    parts = []
    for node in doctree.findall(TocTreeNode):
        part = _resolve_toctree(
            app.env,
            pagename,
            app.builder,
            node,
            prune=True,
            maxdepth=int(kwargs["maxdepth"]),
            titles_only=kwargs["titles_only"],
            collapse=kwargs["collapse"],
            includehidden=kwargs["includehidden"],
            tags=app.builder.tags,
        )
        if part is not None:
            parts.append(part)

    if not parts:
        return ""

    result = parts[0]
    for part in parts[1:]:
        result.extend(part.children)

    soup = BeautifulSoup(app.builder.render_partial(result)["fragment"], "html.parser")

    for li in soup("li", {"class": "current"}):
        li["class"].append("active")

    for li in soup.select("li"):
        link = li.find("a")
        if link and "#" in link["href"] and link["href"] != "#":
            li.decompose()

    for ul in soup("ul", recursive=False):
        ul.attrs["class"] = [*ul.attrs.get("class", []), "nav", "bd-sidenav"]

    add_collapse_checkboxes(soup)

    show_nav_level = int(kwargs["show_nav_level"])
    for level in range(show_nav_level):
        for details in soup.select(f"li.toctree-l{level} > details"):
            details["open"] = "open"

    return soup


def _add_bilingual_sidebar(app, pagename, templatename, context, doctree):
    def generate_zh_toctree_html(**kwargs):
        return _subtree_toctree_html(app, pagename, "zh/index", **kwargs)

    context["generate_zh_toctree_html"] = generate_zh_toctree_html


def setup(app):
    app.connect("html-page-context", _add_bilingual_sidebar)
