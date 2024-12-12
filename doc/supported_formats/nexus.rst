.. _nexus-format:

NeXus data format
-----------------

.. note::
   To read this format, the optional dependency ``h5py`` is required.

Background
^^^^^^^^^^

`NeXus <https://www.nexusformat.org>`_ is a common data format originally
developed by the neutron and x-ray science x-ray communities. It is still being
developed as an international standard by scientists and programmers
representing major scientific facilities in order to facilitate greater
cooperation in the analysis and visualization of data.
NeXus uses a variety of classes to record data, values,
units and other experimental metadata associated with an experiment.
For specific types of experiments an Application Definition may exist, which
defines an agreed common layout that facilities can adhere to.

NeXus metadata and data are stored in Hierarchical Data Format Files (HDF5) with
a ``.nxs`` extension although standard HDF5 extensions are sometimes used.

.. note::
    In `HyperSpy <https://hyperspy.org>`_, files must use the ``.nxs`` file
    extension in order to default to the NeXus loader. If your file has
    an HDF5 extension, you can also explicitly set the NeXus file reader:

    .. code-block:: python

        # Load a NeXus file with a .h5 extension
        >>> import hyperspy.api as hs
        >>> s = hs.load("filename.h5", reader="nxs")

The loader will follow version 3 of the
`NeXus data rules <https://manual.nexusformat.org/datarules.html#version-3>`_.
The signal type, Signal1D or Signal2D, will be inferred by the ``interpretation``
attribute, if this is set to ``spectrum`` or ``image``, in the ``NXdata``
description. If the `interpretation
<https://manual.nexusformat.org/design.html#design-attributes>`_ attribute is
not set, the loader will return a ``BaseSignal``, which must then be converted
to the appropriate signal type. Following the NeXus data rules, if a ``default``
dataset is not defined, the loader will load NXdata
and HDF datasets according to the keyword options in the reader.
A number of the `NeXus examples <https://github.com/nexusformat/exampledata>`_
from large facilties do not use NXdata or use older versions of the NeXus
implementation. Data can still be loaded from these files but information or
associations may be missing. However, this missing information can be recovered
from within the  ``original_metadata`` which contains the overall structure of
the entry.

As the NeXus format uses the HDF5 format and needs to read both data and
metadata structured in different ways, the loader is written to be quite
flexible and can also be used to inspect any hdf5 based file.


Differences with respect to HSpy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :external+hyperspy:ref:`HyperSpy metadata structure <metadata_structure>`
stores arrays as hdf datasets without attributes
and stores floats, ints and strings as attributes.
The NeXus format uses hdf dataset attributes to store additional
information such as an indication of the units for an axis or the ``NX_class`` which
the dataset structure follows. The metadata, HyperSpy or ``original_metadata``,
therefore needs to be able to indicate the values and attributes of a dataset.
To implement this structure the ``value`` and ``attrs`` of a dataset can also be
defined. The value of a dataset is set using a ``value`` key.
The attributes of a dataset are defined by an ``attrs`` key.

For example, to store an array called ``axis_x``, with a units attribute within
``original_metadata``, the following structure would be used:

::

    ├──original_metadata
    │   ├── axis_x
    │   │   ├── value : array([1.0,2.0,3.0,4.0,5.0])
    │   │   ├── attrs
    │   │   │   ├── units : mm


.. code-block:: python

    >>> original_metadata.set_item(axis_x.value,[1.0,2.0,3.0,4.0,5.0])
    >>> original_metadata.set_item(axis_x.attrs.units,"mm")

To access the axis information:

.. code-block:: python

    >>> original_metadata.axis_x.value
    >>> original_metadata.axis_x.attrs.units

To modify the axis information:

.. code-block:: python

    >>> original_metadata.axis_x.value = [2.0,3.0,4.0,5.0,6.0]
    >>> original_metadata.axis_x.attrs.units = "um"

To store data in a NeXus monochromator format, ``value``
and ``attrs`` keys can define additional attributes:

::

    ├── monochromator
    │   ├── energy
    │   │   ├── value : 12.0
    │   │   ├── attrs
    │   │   │   ├── units : keV
    │   │   │   ├── NXclass : NXmonochromator


The ``attrs`` key can also be used to define NeXus structures for the definition
of structures and relationships between data:

::

    ├── mydata
    │   ├── attrs
    │   │   ├── NX_class : "NXdata"
    │   │   ├── axes : ["x","."]
    │   ├── data
    │   │   ├──value : [[30,23...110]
    │   ├── x
    │   │   ├──value : [1,2.....100]
    │   │   ├── attrs
    │   │   │   ├── unit : "mm"


The use of ``attrs`` or ``value`` to set values within the metadata is optional
and metadata values can also be set, read or modified in the normal way.


.. code-block:: python

    >>> original_metadata.monochromator.energy = 12.5

HyperSpy metadata is stored within the NeXus file and should be automatically
restored when a signal is loaded from a previously saved NeXus file.

.. note::

    Altering the standard metadata structure of a signal
    using ``attrs`` or ``value`` keywords is not recommended.

Also see the :ref:`hdf5-utils` for inspecting HDF5 files.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.nexus
   :members:


Reading examples
^^^^^^^^^^^^^^^^

NeXus files can contain multiple datasets within the same file, but the
ordering of datasets can vary depending on the setup of an experiment or
processing step when the data was collected.
For example, in one experiment Fe, Ca, P, Pb were collected but in the next experiment
Ca, P, K, Fe, Pb were collected. RosettaSciIO supports reading in one or more datasets
and returns a list of signals but in this example case the indexing is different.
To control which data or metadata is loaded and in what order
some additional loading arguments are provided.


.. note::

    Given that HDF5 files can accommodate very large datasets, setting ``lazy=True``
    is strongly recommended if the content of the HDF5 file is not known apriori.
    This prevents issues with regard to loading datasets far larger than memory.

    Also note that setting ``lazy=True`` leaves the file handle to the HDF5 file
    open. In Hyperspy, it can with ``_signals.lazy.LazySignal.close_file``
    or when using ``_signals.lazy.LazySignal.compute`` with ``close_file=True``.


We can load a specific dataset using the ``dataset_path`` keyword argument.
Setting it to the absolute path of the desired dataset will cause
the single dataset to be loaded:

.. code-block:: python

    >>> from rsciio.nexus import file_reader
    >>> # Loading a specific dataset
    >>> file_reader("sample.nxs", dataset_path="/entry/experiment/EDS/data")

We can also choose to load datasets based on a search key using the
``dataset_key`` keyword argument. This can also be used to load NXdata not
outside of the ``default`` version 3 rules. Instead of providing an absolute
path, a string can be provided as well, and datasets with this key will be
returned. The previous example could also be written as:

.. code-block:: python

    >>> # Loading datasets containing the string "EDS"
    >>> file_reader("sample.nxs", dataset_key="EDS")

The difference between ``dataset_path`` and ``dataset_key`` is illustrated
here:

.. code-block:: python

    >>> # Only the dataset /entry/experiment/EDS/data will be loaded
    >>> file_reader("sample.nxs", dataset_path="/entry/experiment/EDS/data")
    >>> # All datasets contain the entire string "/entry/experiment/EDS/data" will be loaded
    >>> file_reader("sample.nxs", dataset_key="/entry/experiment/EDS/data")

Multiple datasets can be loaded by providing a number of keys:

.. code-block:: python

    >>> # Loading a specific dataset
    >>> file_reader("sample.nxs", dataset_key=["EDS", "Fe", "Ca"])

Metadata can also be filtered in the same way using ``metadata_key``:

.. code-block:: python

    >>> # Load data with metadata matching metadata_key
    >>> file_reader("sample.nxs", metadata_key="entry/instrument")

.. note::

    The NeXus loader removes any NXdata blocks from the metadata.

Metadata that are arrays can be skipped by using ``skip_array_metadata``:

.. code-block:: python

    >>> # Load data while skipping metadata that are arrays
    >>> file_reader("sample.nxs", skip_array_metadata=True)

.. note::

    NeXus files also support parameters or dimensions that have been varied
    non-linearly. Since the reading of non-uniform axes is not yet implemented for the 
    NeXus plugin, such non-linear information would be lost in the axes manager and
    replaced with indices.

.. note::

    NeXus and HDF can result in large metadata structures with large datasets within the loaded
    original_metadata. If lazy loading is used this may not be a concern but care must be taken
    when saving the data. To control whether large datasets are loaded or saved,
    use the ``metadata_key`` to load only the most relevant information. Alternatively,
    set ``skip_array_metadata`` to ``True`` to avoid loading those large datasets in original_metadata.


Writing examples
^^^^^^^^^^^^^^^^

Using the ``file_writer`` method will store the NeXus file with the following structure:

::

    ├── entry1
    │   ├── signal_name
    │   │   ├── auxiliary
    │   │   │   ├── original_metadata
    │   │   │   ├── hyperspy_metadata
    │   │   │   ├── learning_results
    │   │   ├── signal_data
    │   │   │   ├── data and axes (NXdata format)



To save multiple signals, the ``file_writer`` can be called directly passing a
list of signals.

.. code-block:: python

    >>> from rsciio.nexus import file_writer
    >>> file_writer("test.nxs",[signal1,signal2])

When saving multiple signals, a default signal can be defined. This can be used when storing
associated data or processing steps along with a final result. All signals can be saved but
a single signal can be marked as the default for easier loading using RosettaSciIO
or plotting with NeXus tools.
The default signal is selected as the first signal in the list:

.. code-block:: python

    >>> from rsciio.nexus import file_writer
    >>> file_writer("test.nxs", [signal1, signal2], use_default = True)

The output will be arranged by signal name:

::

    ├── entry1 (NXentry)
    │   ├── signal_name (NXentry)
    │   │   ├── auxiliary (NXentry)
    │   │   │   ├── original_metadata (NXcollection)
    │   │   │   ├── hyperspy_metadata (NXcollection)
    │   │   │   ├── learning_results  (NXcollection)
    │   │   ├── signal_data (NXdata format)
    │   │   │   ├── data and axes
    ├── entry2 (NXentry)
    │   ├── signal_name (NXentry)
    │   │   ├── auxiliary (NXentry)
    │   │   │   ├── original_metadata (NXcollection)
    │   │   │   ├── hyperspy_metadata (NXcollection)
    │   │   │   ├── learning_results (NXcollection)
    │   │   ├── signal_data (NXdata)
    │   │   │   ├── data and axes


.. note::

    Signals saved as ``.nxs`` by this plugin can be loaded normally in HyperSpy
    and the original_metadata, signal data, axes, metadata and learning_results
    will be restored. Model information is not currently stored.
    NeXus does not store how the data should be displayed.
    To preserve the signal details an additional navigation attribute
    is added to each axis to indicate if it is a navigation axis.
