.. _mla_xpoints-format:

MLA xpoints
-----------

Reader for the raw X-ray (EDS) spectra stored in ``.xpoints`` files by the FEI/Thermo
Fisher Mineral Liberation Analyser (MLA) software. An MLA measurement acquires one
spectrum for every point of a grid, or for every grain, of a mosaic of scanning
electron microscope frames; a single measurement can contain more than a million
spectra.

The reader returns one signal with a navigation axis that runs over the spectra, in
acquisition order, and a signal axis with the 2000 channels of the spectra. The signal
type is ``EDS_SEM``, so `exspy <https://hyperspy.org/exspy>`_ provides EDS-specific
methods when it is installed. The ``.xpoints`` extension is not used by other readers,
therefore :func:`hyperspy.api.load` selects this reader automatically:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> s = hs.load("3_GXMAP_spx.xpoints", lazy=True)
    >>> s  # with exspy installed, otherwise a LazySignal1D
    <LazyEDSSEMSpectrum, title: 3_GXMAP_spx, dimensions: (1216798|2000)>

Reading scans the headers of all records, because the position of a record depends on
the sizes of all records before it. This reads the whole file, which takes a few seconds
for a file with about 1.2 million spectra (2.5 GB). The counts are always returned as
``uint16``, so loading such a file completely needs about 5 GB. Use ``lazy=True`` to
process these files in chunks of complete spectra, for example to compute a sum spectrum
or the total counts of every spectrum.

Energy axis
^^^^^^^^^^^

The files do not store a calibration. The signal axis uses the calibration fitted to nine
K lines (Mg Kα to Fe Kβ) of the summed spectrum of one frame of one sample of the dataset
that the format was decoded from: an offset of -10.04 eV and a gain of 10.0163 eV per
channel, in keV. The residuals of the fit are within 3.3 eV. If the detector settings of
your measurement differ, set the ``offset`` and ``scale`` of the signal axis after
loading. A file with a number of channels other than 2000 gets an uncalibrated channel
axis.

Metadata
^^^^^^^^

The fields of the record header that are decoded are available for each spectrum as
arrays in ``original_metadata["spectrum_headers"]``:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Field
      - Meaning
    * - ``x``, ``y``
      - Pixel column and row of the spectrum in its frame (500 × 500 pixels).
    * - ``width``, ``height``
      - Side, in pixels, of the area that the spectrum stands for: the step size of the
        grid (10 pixels) or a single spot (1 pixel). In rare cases it is 0.
    * - ``bse_mode``
      - Most frequent backscattered-electron grey level of the grain that the spectrum
        belongs to.
    * - ``bse_mode_fraction``
      - Fraction of the pixels of that grain with the grey level ``bse_mode``,
        truncated to three decimals.

The ``.xpoints`` file does not say which frame a spectrum belongs to. This is stored in
the ``.jkf`` file of the X-Ray processing stage with the same name, for example
``<name>_spx.jkf`` for ``<name>_spx.xpoints``. If this file is found next to the
``.xpoints`` file, ``original_metadata["jkf"]`` adds, for each spectrum, the arrays
``frame``, ``particle`` and ``grain`` (numbers of the frame, the particle and the grain
label at the position of the spectrum) and ``stage_x_um`` and ``stage_y_um`` (stage
position in µm). The index in the ``.jkf`` file can skip spectra whose frames were
dropped during processing: these spectra are matched by their position in the
``.xpoints`` file and have -1 for the numbers and NaN for the positions. A ``.jkf`` file
that does not belong to the ``.xpoints`` file is ignored with a warning. Use the ``jkf``
argument to select another file or to turn this off.

The stage position follows from the stage position of the centre of the frame and a
pixel size of 3 µm, with the image x axis along the stage X axis and the image rows
along the negative stage Y axis. The pixel size and the half-pixel origin of the pixel
positions are assumed, they are not stored in the file.

File layout
^^^^^^^^^^^

A ``.xpoints`` file has no header and no footer. It is a sequence of records, each of
which consists of an 80-byte little-endian header followed by the counts of one
spectrum: ``n_channels`` values as ``uint8`` when the size of the payload (stored at byte
0x4C of the header) equals ``n_channels``, or as little-endian ``uint16`` when it equals
twice ``n_channels``. In the sample of the dataset that was checked record by record, a
spectrum is stored as ``uint16`` exactly when one of its channels exceeds 255, and across
the dataset between 0.01 % and 2 % of the spectra are stored that way. The size of the
payload is the only difference between the two kinds of record.

Source and limitations
^^^^^^^^^^^^^^^^^^^^^^

The format is not documented by the vendor. It was decoded from the files of the public
dataset *SEM-based automated mineralogy analysis of various granodiorite samples from
the Kindisch quarry* (`RODARE 4896 <https://rodare.hzdr.de/record/4896>`_, published by the
Helmholtz-Zentrum Dresden-Rossendorf under the license
`CC BY 4.0 <https://creativecommons.org/licenses/by/4.0/>`_), which was measured with
an FEI Quanta 650F and two Bruker XFlash 5030 detectors in the GXMAP mode. In one
sample, all 1.2 million records parse back to back to the last byte and every header equals
the corresponding entry of the ``.jkf`` file. For all 20 full-size files of the dataset, the
number of records and of ``uint16`` records follows from the file size and the logged number
of spectra, it agrees with the ``.jkf`` index (which omits some frames in one sample), and
the first 16,000 headers of each file equal that index. The files were written by **MLA Suite 3.1.4.686 only**; other versions of the software, other measurement
modes and other detectors may write a different layout.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.mla_xpoints
   :members:
