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


Memory mapping
==============

Binary file formats are loaded lazily using `memory mapping`_.
The common implementation consists in passing the :class:`numpy.memmap` to the :func:`dask.array.from_array`.
However, it has some shortcomings, to name a few: it is not compatible with the `dask distributed`_
scheduler and it has limited control on the memory usage.

For supported file formats, a different implementation can be used to load data lazily in a manner that is
compatible with the `dask distributed`_  scheduler and allow for better control of the memory usage. 
This implementation uses an approach similar to that described in the dask documentation on
`memory mapping`_ and is enabled using the ``distributed`` parameter (not all formats are
:ref:`supported <supported-formats>`):

.. code:: python

    >>> s = hs.load("file.mrc", lazy=True, distributed=True)


Chunks
======

Depending on the intended processing after loading the data, it may be necessary to
define the chunking manually to control the memory usage or compute distribution.
The chunking can also be specified as follow using the ``chunks`` parameter:

.. code:: python

    >>> s = hs.load("file.mrc", lazy=True, distributed=True, chunks=(5, 10, 10))

.. note::

    Some file reader support specifying the ``chunks`` parameter with the ``distributed`` parameter
    being set to ``True`` or ``False``. In both cases the reader will return a dask array with
    specifyed chunks, However, the way the dask array is created differs significantly and if
    there are issues with memory usage or slow loading, it is recommend to try the ``distributed`` implementation.
