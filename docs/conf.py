# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime


sys.path.append(os.path.abspath("../"))
# sys.path.append(os.path.abspath("../encortex"))
# sys.path.append(os.path.abspath("../encortex/sources"))
# sys.path.append(os.path.abspath("../encortex/environments"))
# sys.path.append(os.path.abspath("../encortex/sources"))


# -- Project information -----------------------------------------------------

project = "EnCortex"
copyright = f"{datetime.datetime.now().year}, Microsoft"
author = "EnCortex Team, Microsoft Research India"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.autosectionlabel",
    "sphinxemoji.sphinxemoji",
    "breathe",
    "sphinx_markdown_builder",
    "sphinx_copybutton",
    "jupyter_sphinx",
    "myst_parser",
    # "myst_nb",
    "nbsphinx",
    # "sphinx_proof",
    "sphinx_design",
    "sphinxcontrib.video",
    "sphinx_togglebutton",
    "sphinx_tabs.tabs",
]

autodoc_typehints = "description"
autodoc_class_signature = "separated"
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "show-inheritance": False,
}
autodoc_inherit_docstrings = True
myst_enable_extensions = ["colon_fence"]

nb_execution_mode = "off"
nbsphinx_allow_errors = True
nbsphinx_execute = "never"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme_options = {
    "repository_url": "https://github.com/microsoft/EnCortex",
    "use_repository_button": True,
    "use_download_button": True,
}

html_title = "EnCortex Docs"
html_extra_path = ["robots.txt"]
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "furo"
html_theme = "sphinx_book_theme"

# removes the .txt suffix
html_sourcelink_suffix = ""


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "_images"]
