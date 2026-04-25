.. _quadstar-format:

Quadstar (SAC, SBC)
-------------------

RosettaSciIO can read Balzers/Pfeiffer Quadstar binary files from
Quadstar software (version 4.x and later):

- ``.sac`` (Scan Analog): one or more analog traces over time.
- ``.sbc`` (Scan Bargraph): peak/bargraph cycles with mass labels and
  intensities.

For ``.sac`` files, each trace is returned as a separate signal dictionary.
For ``.sbc`` files, a single signal dictionary is returned.

In both cases, the signal axis corresponds to mass-to-charge ratio (``m/z``),
and the navigation axis corresponds to measurement cycles/time when multiple
cycles are present.

.. note::
   Lazy loading is currently not supported for Quadstar files.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.quadstar
   :members:
