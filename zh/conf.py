# Sphinx configuration for the Chinese edition.
# Build: sphinx-build -b html zh _build/html/zh

project = "面向 MLSys 的现代 GPU 编程"
author = "MLC Community"
copyright = "2026, MLC Community"
release = "0.0.1"
language = "zh_CN"

extensions = ["myst_parser", "sphinx_copybutton"]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "README.md",
    "**/README.md",
    "_*.md",
    "**/_*.md",
    # Release the Chinese edition chapter by chapter. Keep draft sources in
    # zh/, but exclude unreleased pages so they are not published or searchable.
    "appendix/**",
    "chapter_async_barriers/**",
    "chapter_clc/**",
    "chapter_data_layout/**",
    "chapter_flash_attention/**",
    "chapter_gemm_advanced/**",
    "chapter_gemm_async/**",
    "chapter_gemm_basics/**",
    "chapter_intro_tirx/**",
    "chapter_layout_generations/**",
    "chapter_tensor_cores/**",
    "chapter_tirx_layout_api/**",
    "chapter_tma/**",
    "chapter_tmem/**",
    "tirx_guide/**",
]

html_theme = "sphinx_book_theme"
html_title = project
html_logo = "../static/mlc-logo-with-text-landscape.svg"
html_favicon = "../static/mlc-favicon.ico"
html_static_path = ["../static"]
html_extra_path = ["../_extra", "_extra"]
html_css_files = ["custom.css", "demo-embed.css"]
html_js_files = ["demo-embed-zh-20260627.js"]
html_theme_options = {
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "home_page_in_toc": False,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "repository_url": "https://github.com/mlc-ai/modern-gpu-programming-for-mlsys",
    "use_repository_button": True,
}


def add_language_switch_button(app, pagename, templatename, context, doctree):
    """Add a Chinese-to-English switch to the article header."""
    header_buttons = context.get("header_buttons")
    if header_buttons is None:
        return

    en_url = context["pathto"]("../index.html", 1)
    header_buttons.append(
        {
            "type": "javascript",
            "javascript": f"window.location.href='{en_url}'",
            "tooltip": "Switch to English",
            "icon": "fas fa-language",
            "label": "language-switch-button",
            "classes": "pst-navbar-icon",
        }
    )


def setup(app):
    app.connect("html-page-context", add_language_switch_button, priority=502)
