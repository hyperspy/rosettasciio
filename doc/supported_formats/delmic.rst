.. _delmic-format:

Delmic HDF5
-----------

RosettaScIO can read cathodoluminescence ``.h5`` datasets from Delmic containing one or multiple acquisition streams.
The supported CL data formats currently include:

* intensity
* hyperspectral
* angle-resolved
* time-resolved decay trace
* time-resolved g\ :sup:`(2)` curve
* time-resolved spectrum
* energy-momentum (angle-resolved spectrum)

The photoluminescence is not yet implemented in the metadata structure.

.. Note::
    To read the cathodoluminescence .h5 datasets in `HyperSpy <https://hyperspy.org>`_, use the
    ``reader`` argument to define the correct file plugin as the ``.h5``
    extension is not unique to this reader:

    .. code-block:: python

        >>> import hyperspy.api as hs
        >>> hs.load("filename.h5", reader="Delmic")

Typically, Delmic CL data acquisitions contain three dataset types, but by default, only the
cathodoluminescence datasets found in the acquisition files are opened with RosettaSciIO.
Usually, this is a single dataset, but if the acquisition has multiple streams, then a list
of datasets is returned.

.. Note::
    To load the various types of datasets in the file, use the ``signal`` argument
    with "cl", "se", "survey", or "anchor", to load respectively the cathodoluminescence,
    secondary electron concurrent, secondary electron survey, drift anchor region.
    The special value "all" can be used to load all datasets in the file.

    .. code-block:: python

        >>> import hyperspy.api as hs
        >>> hs.load("filename.h5", reader="Delmic", signal="cl")

API functions
"""""""""""""

.. automodule:: rsciio.delmic
   :members:
