.. _zspy-format:

ZSpy - HyperSpy's Zarr Specification
------------------------------------

.. note::
   To read this format, the optional dependency ``zarr`` is required.

Similarly to the :ref:`hspy format <hspy-format>`, the ``.zspy`` format guarantees that no
information will be lost in the writing process and that supports saving data
of arbitrary dimensions. It is based on the `Zarr project <https://zarr.readthedocs.io/en/stable>`_. Which exists as a drop in
replacement for hdf5 with the intention to fix some of the speed and scaling
issues with the hdf5 format and is therefore suitable for saving 
:external+hyperspy:ref:`big data <big_data.saving>`. Example using `HyperSpy
<https://hyperspy.org>`_:


.. code-block:: python

    >>> import hyperspy.api as hs
    >>> s = hs.signals.BaseSignal([0])
    >>> s.save('test.zspy') # will save in nested directory
    >>> hs.load('test.zspy') # loads the directory


When saving to `zspy <https://zarr.readthedocs.io/en/stable>`_, all supported objects in the signal's
:py:attr:`hyperspy.api.signals.BaseSignal.metadata` is stored. This includes lists, tuples and signals.
Please note that in order to increase saving efficiency and speed, if possible,
the inner-most structures are converted to numpy arrays when saved. This
procedure homogenizes any types of the objects inside, most notably casting
numbers as strings if any other strings are present:

By default, a :py:class:`zarr.storage.NestedDirectoryStore` is used, but other
zarr store can be used by providing a :py:mod:`zarr.storage`
instead as argument to the :py:meth:`hyperspy.api.signals.BaseSignal.save` or the
:py:func:`hyperspy.api.load` function. If a ``.zspy`` file has been saved with a different
store, it would need to be loaded by passing a store of the same type:

.. code-block:: python

    >>> import zarr
    >>> filename = 'test.zspy'
    >>> store = zarr.LMDBStore(filename)
    >>> signal.save(store) # saved to LMDB

To load this file again

.. code-block:: python

    >>> import zarr
    >>> filename = 'test.zspy'
    >>> store = zarr.LMDBStore(filename)
    >>> s = hs.load(store) # load from LMDB

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.zspy
   :members:

.. note::

    Lazy operations are often i-o bound. Reading and writing the data creates a bottle neck in processes
    due to the slow read write speed of many hard disks. In these cases, compressing your data is often
    beneficial to the speed of some operations. Compression speeds up the process as there is less to
    read/write with the trade off of slightly more computational work on the CPU.
