import inspect  # noqa: I001
import os
import sys
import tomllib

import cvxpy
from sphinx.ext.napoleon.docstring import GoogleDocstring

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information --
project = "Direct Data-Driven Model Predictive Control (MPC)"
copyright = "2025, Pável A. Campos-Peña"
author = "Pável A. Campos-Peña"

# Read version from `pyproject.toml`
pyproject_path = os.path.join(
    os.path.dirname(__file__), "../../pyproject.toml"
)

with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)
    release = pyproject_data["project"]["version"]

# -- General configuration --
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_sitemap",
]

templates_path = ["_templates"]

toc_object_entries_show_parents = "hide"

autodoc_typehints = "signature"
autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_member_order = "bysource"
autodoc_inherit_docstrings = True
autodoc_default_options = {
    "members": True,
}

autosummary_generate = True
autosummary_generate_overwrite = True
add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "cvxpy": ("https://www.cvxpy.org/", None),
}

intersphinx_disabled_reftypes = ["std:doc"]

napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "np.ndarray": "numpy.ndarray",
    "Vstack": "cvxpy.atoms.affine.vstack.Vstack",
}

# Add CVXPY class type aliases to napoleon_type_aliases
for name, cls in inspect.getmembers(cvxpy, inspect.isclass):
    if cls.__module__.startswith(cvxpy.__name__):
        napoleon_type_aliases[f"cp.{name}"] = f"{cls.__module__}.{name}"


# Patch Napoleon to name class attributes as "Attributes" and not "Variables"
def parse_attributes_section(self: GoogleDocstring, section: str) -> list[str]:
    return self._format_fields("Attributes", self._consume_fields())


GoogleDocstring._parse_attributes_section = parse_attributes_section  #  type: ignore[method-assign]

# -- Options for HTML output --
html_theme = "sphinx_book_theme"
html_title = "Direct Data-Driven MPC"
html_favicon = "_static/favicon.png"
html_last_updated_fmt = ""
html_static_path = ["_static"]
html_css_files = ["styles.css"]
html_baseurl = "https://pavelacamposp.github.io/direct-data-driven-mpc/"
sitemap_url_scheme = "{link}"
html_extra_path = ["googlec911518ad9a1d2fa.html"]

html_context = {
    "display_version": True,
    "version": release,
}

html_theme_options = {
    "repository_url": "https://github.com/pavelacamposp/direct-data-driven-mpc",
    "path_to_docs": "docs/source/",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "home_page_in_toc": True,
    "show_toc_level": 2,
    "use_sidenotes": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pavelacamposp/direct-data-driven-mpc",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/direct-data-driven-mpc",
            "icon": "https://img.shields.io/pypi/v/direct-data-driven-mpc",
            "type": "url",
        },
        {
            "name": "Python",
            "url": "https://docs.python.org/3.10",
            "icon": "https://img.shields.io/badge/Python-3.10%20|%203.12-blue",
            "type": "url",
        },
        {
            "name": "License: MIT",
            "url": "https://opensource.org/license/MIT",
            "icon": "https://img.shields.io/badge/License-MIT-yellow.svg",
            "type": "url",
        },
    ],
    "icon_links_label": "Quick Links",
}
