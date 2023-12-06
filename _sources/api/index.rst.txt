===
API
===

The RosettaSciIO Application Programming Interface allows other python packages
to use its input/output (IO) capabilities.

.. toctree::
   :hidden:

   utils


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

    from rsciio.hspy import file_reader
    fdict = file_reader("norwegianblue.hspy")

Interfacing the writer from one of the IO plugins:

.. code-block:: python

    from rsciio.hspy import file_writer
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
  * `exSpy <https://hyperspy.org/exspy/>`_: Analysis of X-ray Energy Dispersive
    Spectroscopy (EDS) and Electron Energy Loss Spectroscopy (EELS).
  * `holospy <https://hyperspy.org/holospy/>`_: Analysis of (off-axis) electron
    holography data.
