.. _trivista-format:

TriVista
--------

Reader for spectroscopy data saved using the VistaControl software
for TriVista spectrometers from Teledyne Princeton Instruments.
Currently, RosettaSciIO can only read the ``.tvf`` format from TriVista.
However, this format supports spectral maps and contains all relevant metadata.

If `LumiSpy <https://lumispy.org>`_ is installed, ``LumiSpectrum`` will be
used as the ``signal_type``.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.trivista
   :members: