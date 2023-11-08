.. _emd-format:

Electron Microscopy Dataset (EMD)
---------------------------------

EMD stands for “Electron Microscopy Dataset”. It is a subset of the open source
HDF5 wrapper format. N-dimensional data arrays of any standard type can be
stored in an HDF5 file, as well as tags and other metadata.

.. note::
   To read this format, the optional dependency ``h5py`` is required.

.. _emd_ncem-format:

EMD (NCEM)
^^^^^^^^^^

This `EMD format <https://emdatasets.com>`_ was developed by Colin Ophus at the
National Center for Electron Microscopy (NCEM).
This format is used by the `prismatic software <https://prism-em.com/docs-outputs/>`_
to save the simulation outputs.

Usage examples
""""""""""""""

For files containing several datasets, the `dataset_path` argument can be
used to select a specific one:

.. code-block:: python

    >>> from rsciio.emd import file_reader
    >>> s = file_reader("adatafile.emd", dataset_path="/experimental/science_data_1/data")

Or several by using a list:

.. code-block:: python

    >>> s = file_reader("adatafile.emd",
    ...             dataset_path=[
    ...                 "/experimental/science_data_1/data",
    ...                 "/experimental/science_data_2/data"])


.. _emd_fei-format:

EMD (Velox)
^^^^^^^^^^^

This is a non-compliant variant of the standard EMD format developed by
ThermoFisher (former FEI). RosettaSciIO supports importing images, EDS spectrum and EDS
spectrum streams (spectrum images stored in a sparse format). For spectrum
streams, there are several loading options (described in the docstring below) 
to control the frames and detectors to load and whether to sum them on loading.
The default is to import the sum over all frames and over all detectors in order
to decrease the data size in memory.

.. note::

    Pruned Velox EMD files only contain the spectrum image in a proprietary
    format that RosettaSciIO cannot read. Therefore, don't prune Velox EMD files
    if you intend to read them with RosettaSciIO.

.. note::

    When using `HyperSpy <https://hyperspy.org>`_, FFTs made in Velox are loaded
    in as-is as a HyperSpy ComplexSignal2D object.
    The FFT is not centered and only positive frequencies are stored in the file.
    Making FFTs with HyperSpy from the respective image datasets is recommended.

.. note::

    When using `HyperSpy <https://hyperspy.org>`_, DPC data is loaded in as a HyperSpy ComplexSignal2D object.

.. note::

    Currently, only lazy uncompression rather than lazy loading is implemented.
    This means that it is not currently possible to read EDS SI Velox EMD files
    with size bigger than the available memory.

.. note::
   To load EDS data, the optional dependency ``sparse`` is required.

.. warning::

   This format is still not stable and files generated with the most recent
   version of Velox may not be supported. If you experience issues loading
   a file, please report it  to the RosettaSciIO developers so that they can
   add support for newer versions of the format.

Usage examples
""""""""""""""

.. code-block:: python

    >>> from rsciio.emd import file_reader
    >>> file_reader("sample.emd")
    [<Signal2D, title: HAADF, dimensions: (|179, 161)>,
    <EDSSEMSpectrum, title: EDS, dimensions: (179, 161|4096)>]

.. code-block:: python

    >>> file_reader("sample.emd", sum_EDS_detectors=False)
    [<Signal2D, title: HAADF, dimensions: (|179, 161)>,
    <EDSSEMSpectrum, title: EDS - SuperXG21, dimensions: (179, 161|4096)>,
    <EDSSEMSpectrum, title: EDS - SuperXG22, dimensions: (179, 161|4096)>,
    <EDSSEMSpectrum, title: EDS - SuperXG23, dimensions: (179, 161|4096)>,
    <EDSSEMSpectrum, title: EDS - SuperXG24, dimensions: (179, 161|4096)>]

    >>> file_reader("sample.emd", sum_frames=False, load_SI_image_stack=True, SI_dtype=np.int8, rebin_energy=4)
    [<Signal2D, title: HAADF, dimensions: (50|179, 161)>,
    <EDSSEMSpectrum, title: EDS, dimensions: (50, 179, 161|1024)>]


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.emd
   :members:
