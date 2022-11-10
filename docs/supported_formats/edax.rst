.. _edax-format:

EDAX TEAM/Genesis (SPC, SPD)
----------------------------

RosettaSciIO can read both ``.spd`` (spectrum image) and ``.spc`` (single spectra)
files from the EDAX TEAM software and its predecessor EDAX Genesis.
If reading an ``.spd`` file, the calibration of the
spectrum image is loaded from the corresponding ``.ipr`` and ``.spc`` files
stored in the same directory, or from specific files indicated by the user.
If these calibration files are not available, the data from the ``.spd``
file will still be loaded, but with no spatial or energy calibration.
When using `HyperSpy <https://hyperspy.org>`_, if elemental information has been
defined in the spectrum image, those
elements will automatically be added to the signal loaded by HyperSpy.

In HyperSpy, if an ``.spd`` file is loaded, the result will be a Hyperspy
``EDSSEMSpectrum`` map, and the calibration will be loaded from the appropriate
``.spc`` and ``.ipr`` files (if available). If an ``.spc`` file is loaded, the
result will be a single ``EDSSEMSpectrum`` with no other files needed for
calibration.

.. Note ::

    Currently, HyperSpy will load an ``EDSSEMSpectrum`` signal. If support for
    TEM EDS data is needed, please open an issue in the `issues tracker
    <https://github.com/hyperspy/rosettasciio/issues>`_ to alert the developers
    of the need.

For reference, :ref:`file specifications <edax-file_specification>` for the EDAX
file formats have been publicly available from EDAX.

Extra loading arguments for SPD files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``spc_fname``: {None, str}, name of file from which to read the spectral
  calibration. If data was exported fully from EDAX TEAM software, an ``.spc``
  file with the same name as the .spd should be present. If `None`, the default
  filename will be searched for. Otherwise, the name of the ``.spc`` file to use
  for calibration can be explicitly given as a string.
- ``ipr_fname``: {None, str}, name of file from which to read the spatial
  calibration. If data was exported fully from EDAX TEAM software, an ``.ipr``
  file with the same name as the ``.spd`` (plus a "_Img" suffix) should be present.
  If `None`, the default filename will be searched for. Otherwise, the name of the
  ``.ipr`` file to use for spatial calibration can be explicitly given as a string.
- ``**kwargs``: remaining arguments are passed to the Numpy ``memmap`` function.

Extra loading arguments for SPD and SPC files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``load_all_spc`` : bool, switch to control if all of the ``.spc`` header is
  read, or just the important parts for import into HyperSpy.
