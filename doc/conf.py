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

import numpydoc
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
    # numpydoc is necessary to parse the docstring using sphinx
    # otherwise the nitpicky option will raise many warnings
    "numpydoc",
    "sphinx_favicon",
    "sphinx.ext.autodoc",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinxcontrib.towncrier",
]

intersphinx_mapping = {
    "conda": ("https://conda.io/projects/conda/en/latest", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "exspy": ("https://hyperspy.org/exspy", None),
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "mrcz": ("https://python-mrcz.readthedocs.io", None),
    "numcodecs": ("https://numcodecs.readthedocs.io/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pooch": ("https://www.fatiando.org/pooch/latest", None),
    "python": ("https://docs.python.org/3", None),
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
    "github_url": "https://github.com/hyperspy/rosettasciio",
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
    "header_links_before_dropdown": 6,
    # See https://github.com/pydata/pydata-sphinx-theme/issues/1492
    "navigation_with_keys": False,
}

# -- Options for sphinx_favicon extension -----------------------------------

favicons = {"rel": "icon", "href": "logo_sq.svg", "type": "image/svg+xml"}

# Check links to API when building documentation
nitpicky = True
# Remove when fixed in hyperspy
nitpick_ignore_regex = [(r"py:.*", r"hyperspy.api.*")]

# -- Options for numpydoc extension -----------------------------------

numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"type", "optional", "default", "of"}

if Version(numpydoc.__version__) >= Version("1.6.0rc0"):
    numpydoc_validation_checks = {"all", "ES01", "EX01", "GL02", "GL03", "SA01", "SS06"}

# -- Options for towncrier_draft extension -----------------------------------

# Options: draft/sphinx-version/sphinx-release
towncrier_draft_autoversion_mode = "draft"
towncrier_draft_include_empty = False
towncrier_draft_working_directory = ".."


def setup(app):
    app.add_css_file("custom-styles.css")
