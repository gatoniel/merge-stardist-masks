"""Sphinx configuration."""

from datetime import datetime


project = "Merge StarDist Masks"
author = "Niklas Netter"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
autodoc_type_aliases = {
    "ArrayLike": "numpy.typing.ArrayLike",
    "PolyToLabelSignature": "merge_stardist_masks.naive_fusion.PolyToLabelSignature",
}
autodoc_typehints = "both"
html_theme = "furo"
