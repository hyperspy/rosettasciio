.. _hamamatsu-format:

Hamamatsu
---------

Reader for spectroscopy data saved in ``.img`` (ITEX) files from the HPD-TA
(High Performance Digital Temporal Analyzer) or HiPic (High Performance image control)
softwares from Hamamatsu, e.g. for images from streak cameras or high performance
CCD cameras.

If `LumiSpy <https://lumispy.org>`_ is installed, ``TransientSpectrum`` will be
used as the ``signal_type``, which is intended for streak camera images with
both wavelength and time axes. Note that alternatively, Hamamatsu streak images
saved in the :ref:`tiff-format` format can be read in.

.. Note::

   Currently, reading files containing multiple channels or multiple images per
   channel is not implemented.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.hamamatsu
   :members:
