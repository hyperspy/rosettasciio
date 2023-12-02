Installation
============

RosettaSciIO can be installed with `pip <https://pip.pypa.io/>`_,
`conda <https://docs.conda.io/>`_, the
:ref:`hyperspy:hyperspy-bundle`, or from source, and supports Python >= 3.8.
All alternatives are available on Windows, macOS and Linux.

For using HyperSpy, it is not necessary to install RosettaSciIO separetely, as it would
be installed automatically when installing HyperSpy.

.. _install-with-pip:

With pip
--------

RosettaSciIO is availabe from the Python Package Index (PyPI), and can therefore be
installed with `pip <https://pip.pypa.io/en/stable>`__.
To install with all optional dependencies::

    pip install rosettasciio[all]

To install without optional dependencies::

    pip install rosettasciio

To update RosettaSciIO to the latest release::

    pip install --upgrade rosettasciio

To install a specific version of RosettaSciIO (say version 0.1)::

    pip install rosettasciio==0.1

.. _optional-dependencies:

Optional dependencies
*********************

Some functionality is optional and requires extra dependencies which must be installed
manually or by using `extra <https://peps.python.org/pep-0508/#extras>`_:

Install all optional dependencies::

    pip install rosettasciio[all]

The list of *extras*:

+---------------------+-------------------------+------------------------------------------------------------------------------+
| Extra               | Dependencies            | Usage                                                                        |
+=====================+=========================+==============================================================================+
| ``blockfile``       | ``scikit-image``        | Data normalisation                                                           |
+---------------------+-------------------------+------------------------------------------------------------------------------+
| ``eds-steam``       | ``sparse``              | Loading EDS data stream (JEOL ``pts``, Velox ``emd``)                        |
+---------------------+-------------------------+------------------------------------------------------------------------------+
| ``hdf5``            | ``h5py``                | Reading hdf5-based file formats (``hspy``, ``de5``, ``emd``, ``usid``, etc.) |
+---------------------+-------------------------+------------------------------------------------------------------------------+
| ``image``           | ``imageio``             | Reading images, other than tiff format.                                      |
+---------------------+-------------------------+------------------------------------------------------------------------------+
| ``mrcz``            | ``blosc``, ``mrcz``     | Readding ``mrc`` and ``mrcz`` format.                                        |
+---------------------+-------------------------+------------------------------------------------------------------------------+
| ``scalebar_export`` | ``matplotlib-scalebar`` | Exporting image with scalebar.                                               |
+---------------------+-------------------------+------------------------------------------------------------------------------+
| ``speed``           | ``numba``               | Speed up loading some data, for example EDS data.                            |
+---------------------+-------------------------+------------------------------------------------------------------------------+
| ``tiff``            | ``tifffile``            | Read ``tiff`` files.                                                         |
+---------------------+-------------------------+------------------------------------------------------------------------------+
| ``usid``            | ``pyUSID``              | Read ``usid`` files.                                                         |
+---------------------+-------------------------+------------------------------------------------------------------------------+
| ``zspy``            | ``zarr``                | Read ``zspy`` files.                                                         |
+---------------------+-------------------------+------------------------------------------------------------------------------+

And for development, the following *extras* are available (see ``pyproject.toml`` for more information):

- tests
- doc
- dev

.. _install-with-conda:

With conda
----------

To install with conda, we recommend you install it in a
:doc:`conda environment <conda:user-guide/tasks/manage-environments>` with the
`Miniforge distribution <https://github.com/conda-forge/miniforge>`_.
To create an environment and activate it::

    conda create --name rsciio python=3.11
    conda activate rsciio

To install rosettasciio with all dependencies::

    conda install rosettasciio

To install rosettasciio without any dependencies::

    conda install rosettasciio-base

To update RosettaSciIO to the latest release::

    conda update rosettasciio

To install a specific version of RosettaSciIO (say version 0.1)::

    conda install rosettasciio=0.1

.. note::

    Conda used to be slow to install dependencies in large enviroment and mamba could be
    used as a fast drop-in replacement. However, since conda release 23.10, mamba and conda
    use the same "solver" and therefore takes similar time to "solve environment".
    See the `conda blog <https://conda.org/blog/2023-11-06-conda-23-10-0-release>`_ for more information.

.. _install-with-hyperspy-bundle:

With the HyperSpy Bundle
------------------------

The HyperSpy Bundle comes with RosettaSciIO and all its extras pre-installed.
See :ref:`hyperspy:hyperspy-bundle` for instructions.

.. _install-from-source:

From source
-----------

To install RosettaSciIO from source, clone the repository from `GitHub
<https://github.com/hyperspy/rosettasciio>`__, and install with ``pip``::

    git clone https://github.com/hyperspy/rosettasciio.git
    cd rosettasciio
    pip install --editable .

.. note::

    If `setuptools_scm <https://setuptools-scm.readthedocs.io>`_ is
    installed, the version will be determined from the git repository
    at runtime, otherwise, the version will be the one at build time.

To install a development version on CI, it is advised to use
`pip with vcs support <https://pip.pypa.io/en/stable/topics/vcs-support/>`_
in order to get the correct development version, e.g. ``0.3.dev14+g706deac``::

    pip install git+https://github.com/hyperspy/rosettasciio.git
