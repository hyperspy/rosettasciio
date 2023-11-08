.. _image-format:

Image formats
-------------

RosettaSciIO can read and write data to `all the image formats
<https://imageio.readthedocs.io/en/stable/formats/index.html>`_ supported by
`imageio <https://imageio.readthedocs.io/en/stable/>`_, which uses the 
`Python Image Library (PIL/pillow) <https://pillow.readthedocs.io/en/stable/>`_.
This includes ``.jpg``, ``.gif``, ``.png``, ``.pdf``, ``.tif``, etc.
It is important to note that these image formats only support 8-bit files, and
therefore have an insufficient dynamic range for most scientific applications.
It is therefore highly discouraged to use any general image format to store data
for analysis purposes (with the exception of the :ref:`tiff-format`, which uses
the separate ``tiffile`` library).

.. note::
   To read this format, the optional dependency ``imageio`` is required.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.image
   :members:

Examples of saving arguments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When saving an image, a scalebar can be added to the image and the formatting,
location, etc. of the scalebar can be set using the ``scalebar_kwds``
arguments:

.. code-block:: python

    >>> from rsciio.image import file_writer
    >>> file_writer('file.jpg', signal, scalebar=True)
    >>> file_writer('file.jpg', signal, scalebar=True, scalebar_kwds={'location':'lower right'})

.. note::
   To add ``scalebar``, the optional dependency ``matplotlib-scalebar`` is
   required.

In the example above, the image is created using
:py:func:`~.matplotlib.pyplot.imshow`, and additional keyword arguments can be
passed to this function using ``imshow_kwds``. For example, this can be used
to save an image displayed using the matplotlib colormap ``viridis``:

.. code-block:: python

    >>> file_writer('file.jpg', signal, imshow_kwds=dict(cmap='viridis'))


The resolution of the exported image can be adjusted:

.. code-block:: python

    >>> file_writer('file.jpg', signal, output_size=512)
