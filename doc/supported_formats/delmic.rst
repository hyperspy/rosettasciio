.. _delmic-format:

Delmic HDF5
-----------

RosettaScIO can only read cathodoluminescence .h5 datasets from Delmic containing a single stream. All the imaging modes are supported as long as only one cathodoluminescence stream is contained in the file. The current supported CL data formats currently include: intensity, hyperspectral, angle-resolved, time-resolved decay trace, time-resolved g(2), time resolved streak camera, and energy-momentum. The file import as a stack of hyperspy datasets will be implemented in the next iteration.

.. Note::
    To read the cathodoluminescence .h5 datasets in `HyperSpy <https://hyperspy.org>`_, use the
    ``reader`` argument to define the correct file plugin as the ``.h5``
    extension is not unique to this reader:

    .. code-block:: python

        >>> import hyperspy.api as hs
        >>> hs.load("filename.h5", reader="Delmic")

For now only the CL dataset is opened with hyperspy, the data extraction as a stack of datasets is the next planned step.


API functions
"""""""""""""

.. automodule:: rsciio.delmic
   :members: