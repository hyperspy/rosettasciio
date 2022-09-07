===
API
===

The RosettaSciIO Application Programming Interface allows other python packages
to use its input/output (IO) capabilities.

.. _interfacing-api:

Interfacing the RosettaSciIO plugins
====================================

RosettaSciIO is designed as library offering file reading and writing capabilities
for scientific data formats to other python libraries. The IO plugins have a
common interface through the ``file_reader`` and ``file_writer`` (optionally)
functions. Beyond the ``filename`` and in the case of writers ``object2save``, the
accepted keywords are specific to the different plugins. The object returned in
the case of a reader and passed to a writer is a **python dictionary**.

The **dictionary** contains the following fields:

* ``'data'`` -- multidimensional numpy array
* ``'axes'`` -- list of dictionaries describing the axes containing the fields
  ``'name'``, ``'units'``, ``'index_in_array'``, and
  
  - either ``'size'``, ``'offset'``, and ``'scale'``
  - or a numpy array ``'axis'`` containing the full axes vector

* ``'metadata'`` -- dictionary containing the parsed metadata
* ``'original_metadata'`` -- dictionary containing the full metadata tree from the
  input file

Interfacing the reader from one of the IO plugins:

.. code-block:: python

    from rsciio.hspy.api import file_reader
    fdict = file_reader("norwegianblue.hspy")

Interfacing the writer from one of the IO plugins:

.. code-block:: python

    from rsciio.hspy.api import file_writer
    file_writer("beautifulplumage.hspy", fdict)
   

.. _using-rsciio:

Python packages using RosettaSciIO
----------------------------------

The following python packages available through `PyPI <https://pypi.org/>`_ and/or
`conda-forge <https://anaconda.org/conda-forge/>`_ use the RosettaSciIO plugins
for reading/writing of data files:

* `HyperSpy <https://hyperspy.org>`_: Multidimensional data analysis 

* Any `HyperSpy extension <https://github.com/hyperspy/hyperspy-extensions-list>`_
  that inherits the IO capabilities:

      * `LumiSpy <https://lumispy.org>`_: Luminescence analysis with HyperSpy
      * `Kikuchipy <https://kikuchipy.org>`_: Processing, simulating and analyzing
        electron backscatter diffraction (EBSD) patterns in Python 
      * `PyXem <https://pyxem.readthedocs.io>`_: An open-source Python library for
        multi-dimensional diffraction microscopy. 


.. _defining-plugins:

Defining new RosettaSciIO plugins
=================================

Each read/write plugin resides in a separate directory, the name of which should
be descriptive of the file type/manufacturer. This directory should contain the
following files:

* ``__init__.py`` -- May usually be empty

* ``specifications.yaml`` -- The characteristics of the IO plugin in yaml format:

  .. code-block:: yaml

      format_name: <String>
      description: <String>
      full_support: <Bool>	# Whether all the Hyperspy features are supported
      # Recognised file extension
      file_extensions: <Tuple of string>
      default_extension: <Int>	# Index of the extension that will be used by default
      # Writing capabilities
      writes: <Bool>
      # Support for non-uniform axis
      non_uniform_axis = <Bool>

* ``api.py`` -- Python file that implements the actual reader. The IO functionality
  should be interfaced with the following functions:

  * A function called ``file_reader`` with at least one attribute: ``filename``
  * (optional) A function called ``file_writer`` with at least two attributes: 
    ``filename`` and ``object2save`` (a python dictionary) in that order.

Tests covering the functionality of the plugin should be added to the
``tests`` directory with the naming ``test_spamandeggs.py`` corresponsing to
the plugin residing in the directory ``spamandeggs``. Data files for the tests
should be placed in a corresponding subdirectory [change for pooch].

Documentation should be added both as **docstring**, as well as to the user guide,
for which a corresponding `.rst` file should be created in the directory
`docs/supported_formats/` and the format added to the lists in
`docs/supported_formats/index.rst` and `docs/supported_formats/supported_formats.rst`.

.. Note ::
    It is advisable to clone the files of an existing plugin when initiating a new
    plugin.
