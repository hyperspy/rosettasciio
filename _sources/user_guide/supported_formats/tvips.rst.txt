.. _tvips-format:

TVIPS format
------------

The ``.tvips`` format is the default format for image series collected by pixelated
cameras from the TVIPS company. Typically individual images captured by these
cameras are stored in the :ref:`TIFF format<tiff-format>` which can also be 
loaded by RosettaSciIO. This format instead serves to store image streams from 
in-situ and 4D-STEM experiments. During collection, the maximum file size is
typically capped meaning the dataset is typically split over multiple files
ending in `_xyz.tvips`. The `_000.tvips` will contain the main header and
it is essential for loading the data. If a filename is provided for loading
or saving without a `_000` suffix, this will automatically be added. Loading
will not work if no such file is found.

.. warning::

   While ``.tvips`` files are supported, it is a proprietary format, and future
   versions of the format might therefore not be readable. Complete
   interoperability with the official software can neither be guaranteed.

.. warning::
    
   The ``.tvips`` format currently stores very limited amount of metadata about
   scanning experiments. To reconstruct scan data, e.g. 4D-STEM datasets,
   parameters like the shape and scales of the scan dimensions should be
   manually recorded.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.tvips
   :members:
