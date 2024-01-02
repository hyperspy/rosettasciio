.. _hspy-format:

HSpy - HyperSpy's HDF5 Specification
------------------------------------

This is `HyperSpy's <https://hyperspy.org>`_ default format and for data processed
in HyperSpy, it is the only format that guarantees that no
information will be lost in the writing process and that supports saving data
of arbitrary dimensions. It is based on the `HDF5 open standard
<https://www.hdfgroup.org/solutions/hdf5/>`_. The HDF5 file format is supported by `many
applications
<https://support.hdfgroup.org/products/hdf5_tools/SWSummarybyName.htm>`_.
Parts of the specifications are documented in :external+hyperspy:ref:`metadata_structure`.

.. versionadded:: HyperSpy_v1.2
    Enable saving HSpy files with the ``.hspy`` extension. Previously only the
    ``.hdf5`` extension was recognised.

.. versionchanged:: HyperSpy_v1.3
    The default extension for the HyperSpy HDF5 specification is now ``.hspy``.
    The option to change the default is no longer present in ``preferences``.

Only loading of HDF5 files following the HyperSpy specifications is supported by
this plugin. Usually their extension is ``.hspy`` extension, but older versions of
HyperSpy would save them with the ``.hdf5`` extension. Both extensions are
recognised by HyperSpy since version 1.2. However, HyperSpy versions older than 1.2
won't recognise the ``.hspy`` extension. To work around the issue when using old
HyperSpy installations simply change the extension manually to ``.hdf5`` or
directly save the file using this extension by explicitly adding it to the
filename e.g.:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> s = hs.signals.BaseSignal([0])
    >>> s.save('test.hdf5')

.. note::
   To read this format, the optional dependency ``h5py`` is required.


When saving to ``.hspy``, all supported objects in the signal's
:external+hyperspy:attr:`hyperspy.api.signals.BaseSignal.metadata` are stored. This includes lists, tuples
and signals. Please note that in order to increase saving efficiency and speed,
if possible, the inner-most structures are converted to numpy arrays when saved.
This procedure homogenizes any types of the objects inside, most notably casting
numbers as strings if any other strings are present:

.. code-block:: python

    >>> # before saving:
    >>> somelist
    [1, 2.0, 'a name']
    >>> # after saving:
    ['1', '2.0', 'a name']

The change of type is done using numpy "safe" rules, so no information is lost,
as numbers are represented to full machine precision.

This feature is particularly useful when using
:external+exspy:meth:`exspy.signals.EDSSpectrum.get_lines_intensity`:

.. code-block:: python

    >>> s = hs.datasets.example_signals.EDS_SEM_Spectrum()
    >>> s.metadata.Sample.intensities = s.get_lines_intensity()
    >>> s.save('EDS_spectrum.hspy')

    >>> s_new = hs.load('EDS_spectrum.hspy')
    >>> s_new.metadata.Sample.intensities
    [<BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Al_Ka at 1.49 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: C_Ka at 0.28 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Cu_La at 0.93 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Mn_La at 0.63 keV, dimensions: (|)>,
     <BaseSignal, title: X-ray line intensity of EDS SEM Signal1D: Zr_La at 2.04 keV, dimensions: (|)>]

.. _hspy-chunks:

Chunking
^^^^^^^^

.. versionadded:: HyperSpy_v1.3.1
    ``chunks`` keyword argument

The HyperSpy HDF5 format supports chunking the data into smaller pieces to make it possible to load only part
of a dataset at a time. By default, the data is saved in chunks that are optimised to contain at least one
full signal.  It is possible to
customise the chunk shape using the ``chunks`` keyword.
For example, to save the data with ``(20, 20, 256)`` chunks instead of the default ``(7, 7, 2048)`` chunks
for this signal:

.. code-block:: python

    >>> s = hs.signals.Signal1D(np.random.random((100, 100, 2048)))
    >>> s.save("test_chunks", chunks=(20, 20, 256))

Note that currently it is not possible to pass different customised chunk shapes to all signals and
arrays contained in a signal and its metadata. Therefore, the value of ``chunks`` provided on saving
will be applied to all arrays contained in the signal.

By passing ``True`` to ``chunks`` the chunk shape is guessed using the ``guess_chunk`` function of ``h5py``
For large signal spaces, the autochunking usually leads to smaller chunks as ``guess_chunk`` does not impose the
constrain of storing at least one signal per chunk. For example, for the signal in the example above
passing ``chunks=True`` results in chunks of ``(7, 7, 256)``.

Choosing the correct chunk-size can significantly affect the speed of reading,
writing and performance of many HyperSpy algorithms. See the
:external+hyperspy:ref:`HyperSpy chunking section <big_data.chunking>` for more information.

.. Note::

    Also see the :ref:`hdf5-utils` for inspecting HDF5 files.

Format description
^^^^^^^^^^^^^^^^^^
The root of the file must contain a group called ``Experiments``. The ``Experiments``
group can contain any number of subgroups and each subgroup is an experiment or
signal. Each subgroup must contain at least one dataset called ``data``. The
data is an array of arbitrary dimension. In addition, a number equal to the
number of dimensions of the ``data`` dataset of empty groups called ``axis``
followed by a number must exist with the following attributes:

* ``'name'``
* ``'offset'``
* ``'scale'``
* ``'units'``
* ``'size'``
* ``'index_in_array'``

Alternatively to ``'offset'`` and ``'scale'``, the coordinate groups may
contain an ``'axis'`` vector attribute defining the axis points.
The experiment group contains a number of attributes that will be
directly assigned as class attributes of the
:py:class:`hyperspy.api.signals.BaseSignal` instance. In
addition the experiment groups may contain ``'original_metadata'`` and
``'metadata'``-subgroup that will be assigned to the same name attributes
of the :py:class:`hyperspy.api.signals.BaseSignal` instance as a
:py:class:`hyperspy.misc.utils.DictionaryTreeBrowser`.
The ``Experiments`` group can contain attributes that may be common to all
the experiments and that will be accessible as attributes of the
``Experiments`` instance.


Changelog
^^^^^^^^^

v3.3
""""
- Rename ``ragged_shapes`` dataset to ``_ragged_shapes_{key}`` where the ``key``
  is the name of the corresponding ragged ``dataset``.


v3.2
""""
- Deprecated ``record_by`` attribute is removed

v3.1
""""
- add read support for non-uniform DataAxis defined by ``'axis'`` vector
- move ``metadata.Signal.binned`` attribute to ``axes.is_binned`` parameter

v3.0
""""
- add ``Camera`` and ``Stage`` node
- move ``tilt_stage`` to ``Stage.tilt_alpha``

v2.2
""""
- store more metadata as string: ``date``, ``time``, ``notes``, ``authors`` and ``doi``
- store ``quantity`` for intensity axis

v2.1
""""
- Store the ``navigate`` attribute
- ``record_by`` is stored only for backward compatibility but the axes ``navigate``
  attribute takes precendence over ``record_by`` for files with version >= 2.1

v1.3
""""
- Added support for lists, tuples and binary strings


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.hspy
   :members:
