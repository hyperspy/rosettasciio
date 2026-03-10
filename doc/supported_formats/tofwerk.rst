.. _tofwerk-format:

Tofwerk (fibTOF FIB-SIMS)
--------------------------

Reads HDF5 files (`.h5`) written by TofDAQ acquisition software for Tofwerk
time-of-flight mass spectrometry instruments, including the fibTOF FIB-SIMS
system (Tescan plasma-FIB with integrated Tofwerk ToF-SIMS detector).

Two file states are supported:

* **Opened files** — files that have been processed by the Tofwerk software
  (``PeakData/PeakData`` dataset present).  Returns a single 4-D signal with
  shape ``(depth, y, x, m/z)`` where the mass axis contains the peak-integrated
  intensities from the peak table.

* **Raw files** — unprocessed TofDAQ output (``FullSpectra/EventList`` only).
  Returns two signals: a 1-D sum spectrum and a 2-D total ion count (TIC) map.
  Full 4-D reconstruction from the event list is not yet supported.

.. Note::
    The ``.h5`` extension is shared with other formats (EMD, Arina, etc.).
    The plugin identifies Tofwerk files by checking for TofDAQ-specific HDF5
    groups (``FullSpectra``, ``TimingData``, ``AcquisitionLog``) and the
    ``TofDAQ Version`` root attribute.  Specify ``reader="tofwerk"`` explicitly
    if automatic detection fails.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.tofwerk
   :members:
