===
API
===

The RosettaSciIO Application Programming Interface allows other python packages
to use its input/output (IO) capabilities.

.. toctree::
   :hidden:

   utils
   exceptions

.. automodule:: rsciio
   :members:


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
