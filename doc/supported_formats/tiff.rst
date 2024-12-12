.. _tiff-format:

Tagged image file format (TIFF)
-------------------------------


.. note::
   To read this format, the optional dependency ``tifffile`` is required.

RosettaSciIO can read and write 2D and 3D ``.tiff`` files using using
Christoph Gohlke's `tifffile <https://pypi.org/project/tifffile/>`__ library.
In particular, it supports reading and
writing of TIFF, BigTIFF, OME-TIFF, STK, LSM, NIH, and FluoView files. Most of
these are uncompressed or losslessly compressed 2**(0 to 6) bit integer, 16, 32
and 64-bit float, grayscale and RGB(A) images, which are commonly used in
bio-scientific imaging. See `the library webpage
<https://pypi.org/project/tifffile/>`__ for more details.

.. versionadded: 1.0
   Add support for writing/reading scale and unit to tif files to be read with
   ImageJ or DigitalMicrograph

Currently RosettaSciIO has limited support for reading and saving the TIFF tags.
However, the way that RosettaSciIO reads and saves the scale and the units of ``.tiff``
files is compatible with ImageJ/Fiji and Gatan Digital Micrograph software.
RosettaSciIO can also import the scale and the units from ``.tiff`` files saved using
FEI, Zeiss SEM, Olympus SIS, Jeol SightX and Hamamatsu HPD-TA (streak camera)
software.

Multipage tiff files are read using either series or pages interface built in tifffile,
``series`` interface (default) returns multipage series of images as a single array
with single metadata and original metadata structure of first page.
Using ``multipage_to_list=True`` will use ``pages`` interface and will return a list
of separate arrays and metadata per page.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.tiff
   :members:

.. warning::

    The file will be saved with the same bit depth as the signal. Since
    most processing operations in HyperSpy and numpy will result in 64-bit
    floats, this can result in 64-bit ``.tiff`` files, which are not always
    compatible with other imaging software.

    You can first change the dtype of the signal before saving (example using
    `HyperSpy <https://hyperspy.org>`_):

    .. code-block:: python

        >>> s.data.dtype
        dtype('float64')
        >>> s.change_dtype('float32')
        >>> s.data.dtype
        dtype('float32')
        >>> s.save('file.tif')
