from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as get_version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "idaes-sdoe"
author = "Xiangyu Bi, Dan Gunter"
copyright = (
    "2018-2026, The Regents of the University of California, through "
    "Lawrence Berkeley National Laboratory, National Technology & "
    "Engineering Solutions of Sandia, LLC, Carnegie Mellon University, "
    "West Virginia University Research Corporation, et al."
)

try:
    release = get_version("idaes-sdoe")
except PackageNotFoundError:
    release = "0.1.0"

version = ".".join(release.split(".")[:2])

extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_marimo",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

#html_theme = "pydata_sphinx_theme"
html_theme = "sphinx_book_theme"
html_title = f"{project} {release}"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_sidebars = {
    "**": ["sbt-sidebar-nav.html"]
}
html_show_sourcelink = False
html_theme_options = {
    # Keep the header focused on the project identity and search. The complete
    # documentation tree belongs in the persistent primary sidebar.
    "navbar_center": [],
    "show_prev_next": True,
    "show_nav_level": 2,
    "navigation_with_keys": True,
    "header_links_before_dropdown": 6,
    "search_bar_text": "Search the docs...",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/IDAES/idaes-sdoe",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "IDAES",
    "github_repo": "idaes-sdoe",
    "github_version": "main",
    "doc_path": "docs",
}
# Marimo config
# Optional configuration
marimo_notebook_dir = 'notebooks'  # Directory containing .py Marimo notebooks
marimo_default_height = '600px'
marimo_default_width = '100%'

# Parallel build and caching (default values shown)
marimo_parallel_build = True       # Enable parallel notebook building
marimo_n_jobs = -1                  # Number of parallel jobs (-1 = auto-detect CPU cores)
marimo_cache_notebooks = True       # Enable caching to speed up repeated builds

# Click-to-load configuration (multiple modes available)
marimo_click_to_load = True        # Options: False, True/"overlay", "compact"
marimo_load_button_text = "Load Interactive Notebook"

