.. _msa-format:

EMSA/MSA format
---------------

The ``.msa`` format is an `open standard format
<https://www.microscopy.org/resources/scientific_data/index.cfm>`_
widely used to exchange single spectrum data, but does not support
multidimensional data. It can for example be used to exchange single spectra
with Gatan's Digital Micrograph or other software packages. A wide range of
programs supports exporting to and reading from the ``.msa`` format.

.. WARNING::
    If several spectra are loaded and stacked in `HyperSpy <https://hyperspy.org>`_
    (``hs.load('pattern', stack_signals=True)``)
    the calibration is read from the first spectrum and applied to all other spectra.

Reference
^^^^^^^^^

For specifications of the format, see the `documentation by the Microscopy Society
of America <https://www.microscopy.org/resources/scientific_data/>`_.


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.msa
   :members:
