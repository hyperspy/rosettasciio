.. _mrc-format:

MRC format
----------

The ``.mrc`` format is widely used for tomographic data. The implementation of this plugin
is based on `this specification
<https://www2.mrc-lmb.cam.ac.uk/research/locally-developed-software/image-processing-software/>`_
and has partial support for FEI's custom header.

.. Note::
    When reading 4D-STEM data saved by the Velox software, the data are read as a stack
    of diffraction patterns, but the ``navigation_shape`` argument can be used to
    specify the shape of the navigation space.

.. Note::
    For ``.mrc`` files, the :func:`~rsciio.mrc.file_reader` takes the ``mmap_mode``
    keyword argument to load the file using a different mode (default is copy-on-write).
    However, note that lazy loading does not support in-place writing (i.e lazy loading
    and the ``r+`` mode are incompatible).

See also the `format documentation <https://www.ccpem.ac.uk/mrc_format/mrc_format.php>`_
by the Collaborative Computational Project for Electron cryo-Microscopy (CCP-EM).

This plugin does not support writing ``.mrc`` files, which can however be done
using the :ref:`mrcz <mrcz-format>` plugin. No additional feature of the
`mrcz format <https://python-mrcz.readthedocs.io>`_ should be used in order
to write a ``.mrc`` compliant file. In particular, the ``compressor`` argument should
not be passed (Default is ``None``):

.. code-block:: python

    >>> import numpy as np
    >>> from rsciio import mrcz

    >>> data = np.random.randint(100, size=(10, 100, 100)).astype('int16')
    >>> s = hs.signals.Signal2D(data)
    >>> s_dict = s.as_dictionary()

    >>> mrcz.file_writer('test.mrc', s_dict)

Alternatively, use :meth:`hyperspy.api.signals.BaseSignal.save`, which will pick the
``mrcz`` plugin automatically:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> import numpy as np

    >>> data = np.random.randint(100, size=(10, 100, 100)).astype('int16')
    >>> s = hs.signals.Signal2D(data)
    >>> s.save("data.mrc")

MRC Format (Direct Electron)
----------------------------
Loading from Direct Electron's ``.mrc`` as well as reading the metadata from the .txt file
saved by the software is supported by passing the ``metadata_file`` argument to the
``file_reader`` function. The ``metadata_file`` argument can be a string or a file-like
object. Additionally, the ``metadata_file`` argument can be automatically inferred.  This requires
that the file name is of the form ``uniqueid_suffix_movie.mrc`` and that the metadata file is
named ``uniqueid_suffix_info.txt``.

This will automatically set the navigation shape based on the ``Scan - Size X`` and                                                   = 256
``Scan - Size Y`` parameters in the metadata file. The navigation shape can be overridden
by passing the ``navigation_shape`` argument to the :func:`~rsciio.mrc.file_reader` function.

Additionally virtual_images/ external detectors can be loaded by passing a list of file names to the
``external_images`` or the ``virtual_images`` parameter.  This will also automatically be inferred
if the file names are of the form ``uniqueid_suffix_ext1_extName.mrc`` and
``uniqueid_suffix_1_virtualName.mrc``. The first virtual image will be used as the navigation image
for fast plotting.

.. code-block:: python

    >>> import hyperspy.api as hs

    # Automatically load metadata_file="20220101_0001_info.txt" and
    # any external/virtual images with the same naming convention
    
    >>> hs.load("20220101_0001_movie.mrc")
    <Signal2D, title: 20220101_0001_movie, dimensions: (32, 32|256, 256)>
    
    # Load metadata from data_info.txt
    
    >>> hs.load("data.mrc", metadata_file="data_info.txt") 

    # Load external image 1

    >>> s = hs.load(
    ...     "20220101_0001_movie.mrc",
    ...     external_images=["20220101_0001_ext1_Ext #1.mrc"]
    ... )
    >>> s.metadata["General"]["external_detectors"][0]
    <Signal2D, title:, dimensions: (|32,32)>

    # Will load virtual image 1

    >>> s = hs.load(
    ...     "20220101_0001_movie.mrc",
    ...     virtual_images=["20220101_0001_1_Virtual #1.mrc"]
    ... )
    >>> s.metadata["General"]["virtual_images"][0]
    <Signal2D, title:, dimensions: (|32,32)>


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.mrc
   :members:
