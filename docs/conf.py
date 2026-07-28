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
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_theme = "pydata_sphinx_theme"
html_title = f"{project} {release}"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_show_sourcelink = False
html_theme_options = {
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
