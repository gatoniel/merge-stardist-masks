"""Sphinx configuration."""
from datetime import datetime


project = "Merge Stardist Masks"
author = "Niklas Netter"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
]
autodoc_typehints = "description"
html_theme = "furo"
