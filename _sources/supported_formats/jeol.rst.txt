.. _jeol-format:

JEOL Analyst Station (ASW, ...)
-------------------------------

This is the file format used by the `JEOL Analysist Station software` for which
RosettaSciIO can read the ``.asw``, ``.pts``, ``.map`` and ``.eds`` format. To read the
calibration, it is required to load the ``.asw`` file, which will load all others
files automatically.

.. note::
   To load EDS data, the optional dependency ``sparse`` is required.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.jeol
   :members:


Reading examples
^^^^^^^^^^^^^^^^

`HyperSpy <https://hyperspy.org>`_ example of loading data downsampled and with
cropped energy range, where the original navigation dimension is 512 x 512 and
the EDS range 40 keV over 4096 channels:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> hs.load("sample40kv.asw", downsample=8, cutoff_at_kV=10)
    [<Signal2D, title: IMG1, dimensions: (|512, 512)>,
     <Signal2D, title: C K, dimensions: (|512, 512)>,
     <Signal2D, title: O K, dimensions: (|512, 512)>,
     <EDSTEMSpectrum, title: EDX, dimensions: (64, 64|1096)>]

Load the same file without extra arguments:

.. code-block:: python

    >>> hs.load("sample40kv.asw")
    [<Signal2D, title: IMG1, dimensions: (|512, 512)>,
     <Signal2D, title: C K, dimensions: (|512, 512)>,
     <Signal2D, title: O K, dimensions: (|512, 512)>,
     <EDSTEMSpectrum, title: EDX, dimensions: (512, 512|4096)>]
