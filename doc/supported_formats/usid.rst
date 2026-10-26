.. _usid-format:

Universal Spectroscopy and Imaging Data (h5USID)
------------------------------------------------

`USID <https://pycroscopy.github.io/USID/about.html>`_ is an open schema for
storing imaging and spectroscopy data of any dimensionality. In an HDF5 file
(h5USID), each dataset is stored as a 2D *Main* dataset of shape
``(positions, spectroscopic values)``, together with ancillary datasets that
describe the position and spectroscopic dimensions. This is the format used by
`pycroscopy <https://pycroscopy.github.io/pycroscopy/about.html>`_ and
`pyUSID <https://pycroscopy.github.io/pyUSID/about.html>`_.

RosettaSciIO reads and writes h5USID files directly with ``h5py``; pyUSID is
not required.

Also see the :ref:`hdf5-utils` for inspecting HDF5 files.

Requirements
^^^^^^^^^^^^

This plugin requires ``h5py``, which can be installed with the ``usid``
optional dependency: ``pip install rosettasciio[usid]``.

In `HyperSpy <https://hyperspy.org>`_, other plugins also use the ``.h5``
and ``.hdf5`` file extensions, so pass ``reader="USID"`` to ``load`` to read
h5USID files.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.usid
   :members:

Reading
^^^^^^^

All USID Main datasets in a file are read, and a list of signals is returned
when there is more than one:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> hs.load("sample.h5", reader="USID")
    [<BaseSignal, title: HAADF, dimensions: (|128, 128)>,
    <BaseSignal, title: EELS, dimensions: (|1024, 64, 64)>]

To read a single dataset, pass its absolute path in the file as
``dataset_path``:

.. code-block:: python

    >>> hs.load("sample.h5", reader="USID", dataset_path="/Measurement_001/Channel_000/Raw_Data")
    <BaseSignal, title: Raw_Data, dimensions: (|1024, 64, 64)>

The title of each signal is the name of its HDF5 dataset. All axes are loaded
as signal axes. Use, for example, :meth:`~hyperspy.signal.BaseSignal.as_signal2D`
or :meth:`~hyperspy.signal.BaseSignal.transpose` to choose the navigation axes.
Dimensions with a single value are dropped.

The ``quantity`` and ``units`` of the dataset, its path in the file and the
attributes of the enclosing ``Channel`` and ``Measurement`` groups are stored
in the ``original_metadata``.

.. note::

    With ``lazy=True``, the file is kept open so that the data can be read
    later. Lazy loading is recommended for large files, or if the content of
    the file is not known in advance.

Compound datasets
"""""""""""""""""

A dataset with a `compound data type
<https://pycroscopy.github.io/USID/usid_model.html#compound-datasets>`_ (with
named fields, for example amplitude and phase) is read as one signal per field.
The name of the field is stored as the ``quantity`` in the ``original_metadata``.

Non-uniform dimensions
""""""""""""""""""""""

Dimensions whose values are not uniformly spaced are not supported. By
default, such a dimension is read with a scale of 1 and an offset of 0, and a
warning is raised:

.. code-block:: python

    >>> hs.load("sample.h5", reader="USID")
    UserWarning: Ignoring non-uniformity of dimension: Bias
    <BaseSignal, title: Raw_Data, dimensions: (|7, 3, 5, 2)>

Set ``ignore_non_uniform_dims=False`` to raise a ``ValueError`` instead.

Writing
^^^^^^^

The navigation axes of the signal are written as position dimensions and the
signal axes as spectroscopic dimensions. Axes without a name or units are
written as ``Pos_Dim_<i>`` or ``Spec_Dim_<i>`` with units ``a. u.``. As signals
do not describe the quantity of the data values, the ``quantity`` and
``units`` of the dataset are set to ``Unknown Quantity`` and ``Unknown Units``.

The metadata and original metadata are flattened (with nested keys joined by
``-``, and original metadata keys prefixed by ``Original``) and written as
attributes of the ``Measurement`` group. Metadata items that cannot be stored
as HDF5 attributes are skipped, with a warning.

If the file already exists, the signal is added to it in a new
``Measurement_<NNN>`` group. Models and other objects linked to the signal are
not written.
