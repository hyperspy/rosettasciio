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
To install, run the following::

    pip install rosettasciio

To update rosettasciio to the latest release::

    pip install --upgrade rosettasciio

To install a specific version of kikuchipy (say version 0.1)::

    pip install rosettasciio==0.1

.. _optional-dependencies:

Optional dependencies
*********************

Some functionality is optional and requires extra dependencies which must be installed
manually or by using `extra <https://peps.python.org/pep-0508/#extras>`_:

Install all optional dependencies::

    pip install rosettasciio[all]

The list of ``extra``:
- blockfile
- hdf5
- mrcz
- scalebar_export
- tiff
- usid
- zspy

And for development:
- tests
- docs
- dev

.. _install-with-conda:

With conda
----------

To install with conda, we recommend you install it in a
:doc:`conda environment <conda:user-guide/tasks/manage-environments>` with the
`Miniforge distribution <https://github.com/conda-forge/miniforge>`_.
To create an environment and activate it, run the following::

   conda create --name rsciio python=3.11
   conda activate rsciio

To install::

    conda install rosettasciio

To update RosettaSciIO to the latest release::

    conda update rosettasciio

To install a specific version of RosettaSciIO (say version 0.1)::

    conda install rosettasciio=0.1

.. _install-with-hyperspy-bundle:

With the HyperSpy Bundle
------------------------

RosettaSciIO is available in the HyperSpy Bundle. See :ref:`hyperspy:hyperspy-bundle` for
instructions.

.. _install-from-source:

From source
-----------

To install RosettaSciIO from source, clone the repository from `GitHub
<https://github.com/hyperspy/rosettasciio>`__, and install with ``pip``::

    git clone https://github.com/hyperspy/rosettasciio.git
    cd rosettasciio
    pip install --editable .

