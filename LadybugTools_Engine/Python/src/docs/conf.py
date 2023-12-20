# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os
import re
import sys
from datetime import datetime

from sphinx.application import Sphinx

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "ladybugtools_toolkit"
copyright = f"{datetime.today().year}, BHoM"
author = "BHoM"
release = ""
version = ""

# -- General configuration ---------------------------------------------------

add_module_names = False

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinxcontrib.fulltoc",
]

templates_path = ["_templates"]

source_suffix = ".rst"

master_doc = "index"
language = "en"
exclude_patterns = []
pygments_style = None

# -- Options for HTML output -------------------------------------------------

# html_theme = 'alabaster'
html_theme = "classic"
html_theme_options = {
    # "collapsiblesidebar": True,
}

# Stylesheet path
html_static_path = ["_static"]
html_sidebars = {"**": ["localtoc.html"]}

# -----------------------------------------------------------------------------


def setup(app: Sphinx):
    """Run custom code with access to the Sphinx application object

    Args:
        app (Sphinx):
            The Sphinx application object.
    """

    # Add custom stylesheet
    app.add_css_file("custom.css")
