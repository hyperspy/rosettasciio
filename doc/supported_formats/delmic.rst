.. _delmic-format:

Delmic HDF5
-----------

RosettaScIO can read cathodoluminescence ``.h5`` datasets from Delmic containing a multiple streams.
The supported CL data formats currently include: intensity, hyperspectral, angle-resolved, time-resolved
decay trace, time-resolved g\ :sup:`(2)` curves, time-resolved streak camera, and energy-momentum images.
The polarization orientation is not yet implemented in the metadata structure, as well as the
photoluminescence metadata.

.. Note::
    To read the cathodoluminescence .h5 datasets in `HyperSpy <https://hyperspy.org>`_, use the
    ``reader`` argument to define the correct file plugin as the ``.h5``
    extension is not unique to this reader:

    .. code-block:: python

        >>> import hyperspy.api as hs
        >>> hs.load("filename.h5", reader="Delmic")

By default, all datasets found in the acquisition files are opened with RosettaSciIO.
As typically Delmic CL data acquisitions contain three datasets, you will obtain a list
of three datasets. It is possible to load each of them separately.

.. Note::
    To load the various types of datasets in the file, use the ``signal`` argument
    with "cl", "se", "survey", "anchor", to load respectively the cathodoluminescence,
    secondary electron concurrent, secondary electron survey, the drift anchor region.

    .. code-block:: python

        >>> import hyperspy.api as hs
        >>> hs.load("filename.h5", reader="Delmic", signal="cl")

API functions
"""""""""""""

.. automodule:: rsciio.delmic
   :members:
