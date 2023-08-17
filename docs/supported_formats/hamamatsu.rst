.. _hamamatsu-format:

Hamamatsu
---------

Reader for spectroscopy data saved in ``.img`` (ITEX) files from the HPD-TA
(High Performance Digital Temporal Analyzer) or HiPic (High Performance image control)
softwares from Hamamatsu, e.g. for streak cameras or high performance CCD cameras.

If `LumiSpy <https://lumispy.org>`_ is installed, ``Luminescence`` will be
used as the ``signal_type``.

.. Note::

   Reading files containing multiple channels or multiple images per channel
   is not implemented.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.hamamatsu
   :members: