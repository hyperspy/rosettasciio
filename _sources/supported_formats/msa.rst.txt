.. _msa-format:

EMSA/MSA format
---------------

The ``.msa`` format is an `open standard format
<https://microscopy.org/Scientific-Data/Standards>`__
widely used to exchange single spectrum data, but does not support
multidimensional data. It can for example be used to exchange single spectra
with Gatan's Digital Micrograph or other software packages. A wide range of
programs supports exporting to and reading from the ``.msa`` format.

For ``DATATYPE: XY`` files, the signal axis comes from the X column of the
data, not from the ``OFFSET`` and ``XPERCHAN`` keywords. When the X values are
not evenly spaced, the signal has a non-uniform axis. A signal with a
non-uniform axis is always saved as XY data.

.. Note::
    If several spectra are loaded and stacked in `HyperSpy <https://hyperspy.org>`_
    (``hs.load('pattern', stack_signals=True)``)
    the calibration is read from the first spectrum and applied to all other spectra.

Reference
^^^^^^^^^

For specifications of the format, see the `documentation by the Microscopy Society
of America <https://microscopy.org/Scientific-Data/Standards/>`__ and the
`ISO 22029:2022 <https://www.iso.org/standard/78268.html>`_ standard.


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.msa
   :members:
