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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import pydata_sphinx_theme
from packaging.version import Version

# -- Project information -----------------------------------------------------

project = "RosettaSciIO"
copyright = "2022, HyperSpy Developers"
author = "HyperSpy Developers"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.towncrier",
]

intersphinx_mapping = {
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "pyusid": ("https://pycroscopy.github.io/pyUSID/", None),
    "zarr": ("https://zarr.readthedocs.io/en/stable", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/hyperspy/hyperspy",
    "icon_links": [
        {
            "name": "Gitter",
            "url": "https://gitter.im/hyperspy/hyperspy",
            "icon": "fab fa-gitter",
        },
    ],
    "logo": {
        "image_light": "_static/logo_rec_oct22.svg",
        "image_dark": "_static/logo_rec_dark_oct22.svg",
    },
    "favicons": [
        {
            "rel": "icon",
            "href": "logo_sq.svg",
        },
    ],
    "header_links_before_dropdown": 6,
}

# Remove when pydata_sphinx_theme minimum requirement is bumped to 0.13
if Version(pydata_sphinx_theme.__version__) < Version("0.13.0.dev0"):
    html_theme_options["logo"]["image_light"] = "logo_rec_oct22.svg"
    html_theme_options["logo"]["image_dark"] = "logo_rec_dark_oct22.svg"

# -- Options for towncrier_draft extension -----------------------------------

# Options: draft/sphinx-version/sphinx-release
towncrier_draft_autoversion_mode = "draft"
towncrier_draft_include_empty = False
towncrier_draft_working_directory = ".."


def setup(app):
    app.add_css_file("custom-styles.css")
