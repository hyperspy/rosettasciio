.. _tofwerk-format:

Tofwerk (fibTOF FIB-SIMS)
--------------------------

Reads HDF5 files (``.h5``) written by TofDAQ acquisition software for Tofwerk
time-of-flight mass spectrometry instruments, including the fibTOF FIB-SIMS
system (Tescan plasma-FIB with integrated Tofwerk ToF-SIMS detector).

Data dimensions
^^^^^^^^^^^^^^^

A fibTOF acquisition produces a 4-D dataset:

* **depth** — one entry per FIB milling + SIMS acquisition cycle (``NbrWrites``
  root attribute).  Each write mills a thin layer and then rasters the FIB
  beam across the exposed face to collect SIMS spectra.

* **y** — rows of the 2-D SIMS raster scan (``NbrSegments``).

* **x** — columns of the 2-D SIMS raster scan.

* **m/z** — one bin per peak defined in ``PeakData/PeakTable``.

.. Note::
    The SIMS ion data may be acquired at lower spatial resolution than the
    simultaneously recorded FIB secondary-electron (SE) images.  For example,
    a "256×256 2×2" acquisition has 256×256 FIB SE images but 128×128 SIMS
    pixels.  The pixel size is derived from ``FIBParams.ViewField`` (in mm)
    divided by the SIMS pixel count, not the FIB SE image resolution.

File states
^^^^^^^^^^^

Two file states are supported:

* **Opened files** (``PeakData/PeakData`` present) — Tofwerk software has
  already integrated the raw events into per-peak counts.  The
  ``"sum_spectrum"`` and ``"peak_data"`` signals are available.  The raw
  ``EventList`` is not present in opened files.

* **Raw files** (``FullSpectra/EventList`` only) — unprocessed TofDAQ output.
  The ``"sum_spectrum"`` and ``"event_list"`` signals are always available.
  The ``"peak_data"`` signal can be reconstructed from the EventList using the
  integration windows in ``PeakData/PeakTable``, but this is computationally
  intensive (see below).

Selecting signals
^^^^^^^^^^^^^^^^^

The ``signal`` parameter (passed as a keyword argument to
:func:`hs.load <hyperspy.io.load>` or directly to :func:`rsciio.tofwerk.file_reader`)
controls which signals are returned.  The default is ``"sum_spectrum"``, which
is always fast.

.. code-block:: python

    import hyperspy.api as hs

    # Default: 1-D cumulative spectrum (fast, always available)
    s = hs.load("acquisition.h5", reader="tofwerk")

    # 4-D peak array (depth × y × x × m/z)
    s = hs.load("acquisition.h5", reader="tofwerk", signal="peak_data")

    # Raw TDC timestamps as a ragged signal (raw files only)
    s = hs.load("acquisition.h5", reader="tofwerk", signal="event_list")

    # Multiple signals at once
    signals = hs.load(
        "acquisition.h5",
        reader="tofwerk",
        signal=["sum_spectrum", "peak_data"],
    )

    # All signals available for this file
    signals = hs.load("acquisition.h5", reader="tofwerk", signal="all")

Valid values for ``signal``:

``"sum_spectrum"`` (default)
    1-D cumulative spectrum from ``FullSpectra/SumSpectrum``.  Always
    available in both file states.  Supports lazy loading.

``"peak_data"``
    4-D array ``(depth, y, x, m/z)``.

    * *Opened files*: reads ``PeakData/PeakData`` directly.  Supports lazy
      loading (recommended for large files).
    * *Raw files*: reconstructs the 4-D array by walking the variable-length
      ``FullSpectra/EventList`` and integrating events within each peak window.
      Always eager.  On large files this can take several minutes; install
      ``tqdm`` to see a progress bar, and ``numba`` for a ~19× speed-up.

``"event_list"``
    Ragged object array ``(depth, y, x)`` of raw uint16 TDC timestamps, one
    variable-length array per pixel.  Available only in raw files.  Always
    loaded eagerly (``lazy=True`` is ignored for this signal).  HyperSpy
    represents it as a ragged signal and does not support plotting it
    directly.

``"all"``
    All signals available for the file: ``["sum_spectrum", "peak_data"]`` for
    opened files; ``["sum_spectrum", "peak_data", "event_list"]`` for raw
    files.

Lazy loading
^^^^^^^^^^^^

Pass ``lazy=True`` to defer reading large arrays until ``.compute()`` is called:

.. code-block:: python

    s = hs.load("large_acquisition.h5", reader="tofwerk", lazy=True, signal="peak_data")
    # s.data is a dask array; chunks are one depth-slice each
    s.compute()

Lazy loading is supported for ``"sum_spectrum"`` and ``"peak_data"`` (opened
files only).  The ``"event_list"`` signal is always loaded eagerly regardless
of the ``lazy`` flag.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

* ``tqdm`` — progress bars during EventList reconstruction (``signal="peak_data"``
  on a raw file) and during eager ``"event_list"`` loading.
* ``numba`` — JIT-accelerates EventList reconstruction (~19× faster than the
  NumPy fallback on large datasets).

Metadata
^^^^^^^^

The ``metadata`` dict follows the RosettaSciIO/HyperSpy standard.  FIB-SIMS
specific fields are stored under ``Acquisition_instrument.FIB_SIMS``:

.. code-block:: python

    meta = s.metadata.Acquisition_instrument.FIB_SIMS

    meta.file_type          # "opened" or "raw"
    meta.FIB.hardware       # FIB hardware identifier (e.g. "Tescan")
    meta.FIB.voltage_kV     # Ion beam accelerating voltage in kV
    meta.FIB.current_A      # Ion beam current in A
    meta.FIB.view_field_mm  # Field of view in mm
    meta.FIB.pixel_size_um  # SIMS pixel size in µm (derived from ViewField)
    meta.ToF.ion_mode       # "positive" or "negative"
    meta.ToF.tofdaq_version # TofDAQ software version string

.. Note::
    The ``.h5`` extension is shared with other formats (EMD, Arina, etc.).
    The plugin identifies Tofwerk files by checking for TofDAQ-specific HDF5
    groups (``FullSpectra``, ``TimingData``, ``AcquisitionLog``) and the
    ``TofDAQ Version`` root attribute.  Specify ``reader="tofwerk"``
    explicitly if automatic detection fails.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.tofwerk
   :members:
