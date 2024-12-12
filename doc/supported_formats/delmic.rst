.. _delmic-format:

Delmic HDF5
-----------

RosettaScIO can only read hyperspectral cathodoluminescence .h5 datasets from Delmic. The file reading will be implemented step by step for the various Delmic data formats.

.. Note::
    To read the cathodoluminescence .h5 datasets in `HyperSpy <https://hyperspy.org>`_, use the
    ``reader`` argument to define the correct file plugin as the ``.h5``
    extension is not unique to this reader:

    .. code-block:: python

        >>> import hyperspy.api as hs
        >>> hs.load("filename.h5", reader="Delmic")

API functions
"""""""""""""

.. automodule:: rsciio.delmic
   :members: