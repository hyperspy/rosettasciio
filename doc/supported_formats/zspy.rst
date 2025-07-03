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

Using different Zarr stores
^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, a :py:class:`zarr.storage.NestedDirectoryStore` is used, but other
zarr store can be used when saving to file and this can be done by using one of these
two approaches:

- pass ``"local"`` or ``"zip"`` to the ``store_type`` parameter to use a 
  :class:`zarr.storage.NestedDirectoryStore` or a :class:`zarr.storage.ZipStore`
  zarr store, respectively;
- pass an instance of :py:mod:`zarr.storage` to the ``filename`` parameter.

.. note::
    While the :py:class:`~zarr.storage.ZipStore` is convenient to save as a single file,
    it has limitations:
    
    - The :py:class:`~zarr.storage.ZipStore` currently doesn't support overwriting
      and will raise a ``NotImplementedError`` when writing to an existing file.
    - It doesn't support writing from multiple processes at the same time.

Example of saving a signal using the :py:class:`~zarr.storage.ZipStore`:

.. code-block:: python

    >>> import zarr
    >>> signal.save('test.zspy', store_type="zip") # saved using ZipStore

To load this file again, RosettaSciIO will automatically detect that it
is a zip file and load it using the :py:class:`~zarr.storage.ZipStore`:

.. code-block:: python

    >>> import zarr
    >>> s = hs.load('test.zspy') # load from ZipStore

If other zarr stores are used, it will be necessary to specify the store
type when loading the file:

    >>> import zarr
    >>> store = zarr.storage.ZipStore('test.zspy')  # or the store the data was saved with
    >>> s = hs.load(store) # load using the given store


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.zspy
   :members:

.. note::

    Lazy operations are often i-o bound. Reading and writing the data creates a bottle neck in processes
    due to the slow read write speed of many hard disks. In these cases, compressing your data is often
    beneficial to the speed of some operations. Compression speeds up the process as there is less to
    read/write with the trade off of slightly more computational work on the CPU.
