.. _bruker-format:

Bruker
------

RosettaSciIO can read ``.spx`` single spectrum and ``.bcf`` "hypermaps" file
formats saved with Bruker's Esprit v1.x or v2.x in ``.bcf``
hybrid (virtual file system/container with xml and binary data, optionally
compressed) format. Most ``.bcf`` import functionality is implemented. Both
high-resolution 16-bit SEM images and hyperspectral EDX data can be retrieved
simultaneously.

BCF can look as all inclusive format, however it does not save some key EDS
parameters: any of dead/live/real times, full width at half maximum (FWHM)
at :math:`Mn_{K\alpha}` line. However, real time for whole map is calculated from metadata 
(pixelAverage, lineAverage, pixelTime, lineCounter and map height).

Note that Bruker Esprit uses a similar format for EBSD data, but it is not
currently supported by RosettaSciIO.

The format contains an extensive list of details and parameters of EDS analyses
which in `HyperSpy <https://hyperspy.org>`_ are mapped to the ``metadata`` and
``original_metadata`` dictionaries.


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.bruker
   :members:
