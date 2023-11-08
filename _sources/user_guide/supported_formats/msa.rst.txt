.. _msa-format:

EMSA/MSA format
---------------

The ``.msa`` format is an `open standard format
<https://microscopy.org/Scientific-Data/Standards>`__
widely used to exchange single spectrum data, but does not support
multidimensional data. It can for example be used to exchange single spectra
with Gatan's Digital Micrograph or other software packages. A wide range of
programs supports exporting to and reading from the ``.msa`` format.

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
