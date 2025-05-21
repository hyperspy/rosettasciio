.. _lazy:

============
Lazy loading
============

.. _memory mapping: https://docs.dask.org/en/stable/array-creation.html#memory-mapping
.. _dask distributed: https://distributed.dask.org

Data can be loaded lazily by using ``lazy=True``, however not all formats are :ref:`supported <supported-formats>`.
The data will be loaded as a dask array instead of a numpy array.

.. code:: python

    >>> from rsciio import msa
    >>> d = msa.file_reader("file.mrc", lazy=True)
    >>> d["data"]
    dask.array<array, shape=(10, 20, 30), dtype=int64, chunksize=(10, 20, 30), chunktype=numpy.ndarray>


Chunks
======

Depending on the intended processing after loading the data, it may be necessary to
define the chunking manually to control the memory usage or compute distribution.
The chunking can also be specified as follow using the ``chunks`` parameter:

.. code:: python

    >>> s = hs.load("file.mrc", lazy=True, chunks=(5, 10, 10))

Memory mapping
==============

Binary file formats are loaded lazily using `memory mapping`_ and are compatible with the `dask distributed`_
scheduler. This implementation uses an approach similar to that described in the dask documentation on
`memory mapping`_ - see the :func:`~.utils.distributed.memmap_distributed` function for more information.


Distributed Loading
===================

Not all formats are compatible with the `dask distributed`_ scheduler. See the last columns of the 
:ref:`supported formats <supported-formats>` table to know which reader are supported.

In almost all cases the :func:`~.utils.distributed.memmap_distributed` function can be dropped in-place of the
:class:`numpy.memmap` function. It also now supports the ``positions`` parameter which is different from the equivalent
numpy function.  The ``positions`` parameter is a numpy array of positions which maps some arbitrary scan positions
to a grid.  This is useful for loading arbitrary scan positions from a file.  The ``positions`` parameter does require
that the data is chunked only in the navigation axis.