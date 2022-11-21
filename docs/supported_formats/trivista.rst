.. _trivista-format:

TriVista
--------

Reader for spectroscopy data saved using the VistaControl software
for TriVista spectrometers from Teledyne Princeton Instruments.
Currently, RosettaSciIO can only read the XML-based ``.tvf`` format from TriVista
(the binary ``.tvb`` format is not supported).
However, this format supports spectral maps and contains all relevant metadata.

If `LumiSpy <https://lumispy.org>`_ is installed, ``Luminescence`` will be
used as the ``signal_type``.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.trivista
   :members: