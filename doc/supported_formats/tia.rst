.. _tia-format:

FEI TIA (SER & EMI)
-------------------

RosettaSciIO can read ``.ser`` and ``.emi`` files files created by the FEI (now
ThermoFisher) software TEM Imaging & Analysis (TIA), but the reading features are
not complete (and probably they will be unless FEI/ThermoFisher releases the specifications
of the format). That said we know that this is an important feature and if loading
a particular ``.ser`` or ``.emi`` file fails for you, please report it as an issue in the
`issues tracker <https://github.com/hyperspy/rosettasciio/issues>`__ to make us
aware of the problem.

RosettaSciIO (unlike TIA) can read data directly from the ``.ser`` files. However,
by doing so, the information that is stored in the ``.emi`` file is lost.
Therefore, it is strongly recommended to load using the ``.emi`` file instead.

If several ``.ser`` files are associated to the ``.emi`` file being read,
all of them will be read and returned as a list.


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.tia
   :members:

