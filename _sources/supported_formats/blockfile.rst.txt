.. _blockfile-format:

Blockfile
---------

RosettaSciIO can read and write the blockfile format from NanoMegas ASTAR software.
It is used to store a series of diffraction patterns from scanning precession
electron diffraction (SPED) measurements, with a limited set of metadata. The
header of the blockfile contains information about centering and distortions
of the diffraction patterns, but is not applied to the signal during reading.
Blockfiles only support data values of type
`np.uint8 <https://numpy.org/doc/stable/user/basics.types.html>`_ (integers
in range 0-255).

.. warning::

   While Blockfiles are supported, it is a proprietary format, and future
   versions of the format might therefore not be readable. Complete
   interoperability with the official software can neither be guaranteed.

Blockfiles are by default loaded in a "copy-on-write" manner using
`numpy.memmap
<https://numpy.org/doc/stable/reference/generated/numpy.memmap.html>`_ .
For blockfiles ``load`` takes the ``mmap_mode`` keyword argument enabling
loading the file using a different mode. However, note that lazy loading
does not support in-place writing (i.e lazy loading and the "r+" mode
are incompatible).

.. note::
   To use the ``intensity_scaling`` functionality, the optional dependency
   ``scikit-image`` is required.


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.blockfile
   :members:
