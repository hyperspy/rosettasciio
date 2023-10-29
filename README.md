[![Build Status](https://dev.azure.com/hyperspy/rosettasciio/_apis/build/status/HyperSpy.rosettasciio?branchName=main)](https://dev.azure.com/Hyperspy/rosettasciio/_build/latest?definitionId=3&branchName=main)
[![Tests](https://github.com/hyperspy/rosettasciio/workflows/Tests/badge.svg)](https://github.com/hyperspy/rosettasciio/actions)
[![Codecov Status](https://codecov.io/gh/hyperspy/rosettasciio/branch/main/graph/badge.svg?token=8ZFX8X4Z1I)](https://codecov.io/gh/hyperspy/rosettasciio)
[![Documentation Status](https://readthedocs.org/projects/rosettasciio/badge/?version=latest)](https://rosettasciio.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![Python Version](https://img.shields.io/pypi/pyversions/rosettasciio.svg?style=flat)](https://pypi.python.org/pypi/rosettasciio)
[![PyPi Version](https://img.shields.io/pypi/v/rosettasciio.svg?style=flat)](https://pypi.python.org/pypi/rosettasciio)
[![Anaconda Version](https://anaconda.org/conda-forge/rosettasciio/badges/version.svg)](https://anaconda.org/conda-forge/rosettasciio)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.8011666.svg)](https://doi.org/10.5281/zenodo.8011666)


# RosettaSciIO

<img src="https://github.com/hyperspy/rosettasciio/raw/e6b599a26ed07420730c536be8a4581eaea0e274/docs/_static/logo_rec_dark_oct22.png" width="300" alt="RosettaSciIO">

The **Rosetta Scientific Input Output library** aims at providing easy reading and
writing capabilities in Python for a wide range of
[scientific data formats](https://hyperspy.org/rosettasciio/supported_formats/index.html). Thus
providing an entry point to the wide ecosystem of python packages for scientific data
analysis and computation, as well as an interoperability between different file
formats. Just as the [Rosetta stone](https://en.wikipedia.org/wiki/Rosetta_Stone)
provided a translation between ancient Egyptian hieroglyphs and ancient Greek.
The RosettaSciIO library originates from the [HyperSpy](https://hyperspy.org)
project for multi-dimensional data analysis. As HyperSpy is rooted in the electron
microscopy community, data formats used by this community are still particularly
well represented.

RosettaSciIO provides the dataset, its axes and related metadata contained in a
file in a python dictionary that can be easily handled by other libraries.
Similarly, it takes a dictionary as input for file writers.

See the [documentation](https://hyperspy.org/rosettasciio) for further details.

### Note

RosettaSciIO has recently been split out of the [HyperSpy repository](https://github.com/hyperspy/hyperspy) and the new API is still under development. HyperSpy will use the RosettaSciIO IO-plugins from v2.0. It is already possible to import the readers directly from RosettaSciIO as follows:

```python
from rsciio import msa
msa.file_reader("your_msa_file.msa")
```
