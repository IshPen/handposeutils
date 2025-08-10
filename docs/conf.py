import os
import sys

# Add the package root to sys.path
sys.path.insert(0, os.path.abspath(".."))

project = "handposeutils"
author = "Ishan Pendyala"
release = "0.1.5"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# Theme
html_theme = "sphinx_rtd_theme"  # You can switch to sphinx_rtd_theme if you prefer
html_static_path = ["_static"]
