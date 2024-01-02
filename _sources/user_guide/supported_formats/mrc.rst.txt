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
    For ``.mrc`` files, the ``file_reader`` takes the ``mmap_mode`` keyword argument
    to load the file using a different mode (default is copy-on-write) . However,
    note that lazy loading does not support in-place writing (i.e lazy loading and
    the ``r+`` mode are incompatible).

See also the `format documentation <https://www.ccpem.ac.uk/mrc_format/mrc_format.php>`_
by the Collaborative Computational Project for Electron cryo-Microscopy (CCP-EM).

This plugin does not support writing ``.mrc`` files, which can however be done
using the :ref:`mrcz <mrcz-format>` plugin. No additional feature of the
`mrcz format <https://python-mrcz.readthedocs.io>`_ should be used in order
to write a ``.mrc`` compliant file. In particular, the ``compressor`` argument should
not be passed (Default is ``None``):

.. code-block:: python

    import numpy as np
    from rsciio import mrcz

    data = np.random.randint(100, size=(10, 100, 100)).astype('int16')
    s = hs.signals.Signal2D(data)
    s_dict = s.as_dictionary()

    mrcz.file_writer('test.mrc', s_dict)

Alternatively, use :py:meth:`hyperspy.api.signals.BaseSignal.save`, which will pick the
``mrcz`` plugin automatically:

.. code-block:: python

    import hyperspy.api as hs
    import numpy as np

    data = np.random.randint(100, size=(10, 100, 100)).astype('int16')
    s = hs.signals.Signal2D(data)
    s.save("data.mrc")

MRC Format (Direct Electron)
----------------------------
Loading from Direct Electron's ``.mrc`` as well as reading the metadata from the .txt file
saved by the software is supported by passing the ``metadata_file`` argument to the
``file_reader`` function. The ``metadata_file`` argument can be a string or a file-like
object.

This will automatically set the navigation shape based on the ``Scan - Size X`` and                                                   = 256
``Scan - Size Y`` as well as the ``Scan - Repeats`` and ``Scan - Point Repeats``
parameters in the metadata file. The navigation shape can be overridden
by passing the ``navigation_shape`` argument to the ``file_reader`` function.


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.mrc
   :members:
