.. _tofwerk-format:

Tofwerk (fibTOF FIB-SIMS)
--------------------------

.. warning::

   This plugin and the file format documentation below were developed by
   reverse-engineering real acquisition files without access to official Tofwerk
   format specifications.  Descriptions of HDF5 groups, attributes, and data
   layouts may be incomplete or incorrect.  If you notice anything wrong, please
   `open an issue <https://github.com/hyperspy/rosettasciio/issues>`_ on the
   RosettaSciIO issue tracker!

Reads HDF5 files (``.h5``) written by TofDAQ acquisition software for Tofwerk
time-of-flight mass spectrometry instruments, including the fibTOF FIB-SIMS
system (Tescan plasma-FIB with integrated Tofwerk ToF-SIMS detector).

Acquired signals
^^^^^^^^^^^^^^^^

A fibTOF acquisition produces two raw signal types and one derived signal:

**Raw signals**

* **EventList** — the primary raw detector output.  For each pixel in the
  3-D raster ``(depth, y, x)``, a variable-length array of TDC timestamps
  records every individual ion detection event.  Each write (depth slice)
  mills a thin layer from the sample surface and then rasters the FIB beam
  across the exposed face to collect SIMS spectra.  The spatial axes are:

  - **depth** — one entry per FIB milling + SIMS acquisition cycle (``NbrWrites``).
  - **y** — rows of the 2-D SIMS raster scan (``NbrSegments``).
  - **x** — columns of the 2-D SIMS raster scan.

* **FIB SE images** — secondary-electron images acquired simultaneously at
  the full FIB scan resolution (e.g. 256×256), one image per depth slice.
  The SE image resolution is typically higher than the SIMS raster (e.g.
  128×128 for a "256×256 2×2" acquisition), so SE images and SIMS data have
  independent pixel sizes — both derived from ``FIBParams.ViewField`` (in mm)
  divided by the respective pixel count.

**Derived signal**

* **Peak data** — the signal most users work with.  The EventList is
  integrated over user-defined mass windows (``PeakData/PeakTable``) to
  produce a 4-D array ``(depth, y, x, m/z)`` of per-peak ion counts.  In
  pre-processed files this integration has already been performed by the
  Tofwerk software and the result is stored as ``PeakData/PeakData``.  For
  raw files it can be reconstructed on load (see ``signal="peak_data"``
  below).

File states
^^^^^^^^^^^

Two file states are supported:

* **Pre-processed files** (``PeakData/PeakData`` present) -- A file where the
  Tofwerk software has already integrated the raw events into per-peak counts.

* **Raw files** (no ``PeakData/PeakData``) -- unprocessed TofDAQ output as originally
  saved by the acquisition software.  The ``"peak_data"`` signal can be reconstructed
  from the ``EventList`` data using the integration windows in ``PeakData/PeakTable``,
  but this can be computationally intensive (see below).

HDF5 file structure
^^^^^^^^^^^^^^^^^^^

All TofDAQ ``.h5`` files share the following top-level layout.  Groups marked
``[raw only]`` are absent from pre-processed files; groups marked
``[pre-processed only]`` are absent from raw files (though the Tofwerk
software may preserve ``EventList`` when pre-processing).

.. code-block:: text

    /                               root — acquisition-wide attributes and metadata
    ├── AcquisitionLog/
    │   └── Log                     compound dataset; one row per log entry.
    │                                 Log[0]['timestring'] is the authoritative
    │                                 ISO-8601 acquisition timestamp. This node
    │                                 also contains the acquisition "finish" time.
    ├── FIBImages/
    │   ├── Image0000/
    │   │   └── Data                float64 (H, W) — SE image at full FIB resolution
    │   ├── Image0001/ ...
    │   └── Image000N/              one subgroup per image; sorted lexicographically
    ├── FIBParams/                  group attributes — FIB column settings
    ├── FibParams/
    │   └── FibPressure/
    │       └── TwData              float64 (NbrWrites, 1) — chamber pressure in Pa
    │                                 for every depth slice in the dataset
    ├── FullSpectra/
    │   ├── MassAxis                float32 (NbrSamples,) — calibrated m/z in Da
    │   ├── SumSpectrum             float64 (NbrSamples,) — cumulative ion counts
    │   ├── SaturationWarning       uint8   (NbrWrites, NbrSegments)
    │   │                              Per-buffer saturation flag: 0 = no saturation,
    │   │                              1 = ADC or TDC saturation detected. High values
    │   │                              indicate that the detector was overloaded during that
    │   │                              (write, segment) combination and the corresponding
    │   │                              spectral data should be treated with caution.
    │   └── EventList               vlen uint16 (NbrWrites, NbrSegments, NbrX)
    │                                 each pixel contains one variable-length TDC
    │                                 timestamp array
    ├── PeakData/
    │   ├── PeakTable               compound (NbrPeaks,) — peak definitions:
    │   │                             label, mass (Da), lower/upper integration limits.
    │   │                             Will contain "nominal" peaks created by the TofDAQ
    │   │                             software, together with any user-defined peak integration
    │   │                             windows created before the file was saved
    │   └── PeakData                float32 (NbrWrites, NbrSegments, NbrX, NbrPeaks)
    │                                 [pre-processed only] — peak-integrated ion counts
    ├── TimingData/
    │   ├── BufTimes                float64 (NbrWrites, NbrSegments) — wall-clock
    │   │                             timestamps per buffer, in seconds
    │   └── (group attributes)      TofPeriod — ADC samples per ToF pulse
    └── TPS2/                       instrument telemetry — Sampled once per scan line.
        ├── TwData                  float64 (NbrWrites, NbrSegments, 75) — target
        │                             and monitored voltages/flags for 75 named
        │                             channels.  Mostly low-level instrument debug
        │                             data not relevant to signal processing.
        │                             Channels of potential diagnostic interest:
        │                             MCP bias monitor, filament emission monitor,
        │                             and ToF pulser voltages.
        └── TwInfo                  |S256 (75,) — channel names and units

Key root attributes:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``TofDAQ Version``
     - float32
     - Primary format-detection marker (e.g. ``1.99``).
   * - ``NbrWrites``
     - int32
     - Number of depth slices (milling steps).
   * - ``NbrSegments``
     - int32
     - Number of Y scan lines per write.
   * - ``NbrSamples``
     - int32
     - ADC samples per spectrum (length of ``MassAxis``).
   * - ``NbrPeaks``
     - int32
     - Number of peaks in the peak table.
   * - ``NbrWaveforms``
     - int32
     - Hardware waveform averaging count (normally 1).
   * - ``IonMode``
     - bytes
     - ``b"positive"`` or ``b"negative"``.

Key ``FIBParams`` group attributes:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Attribute
     - Type
     - Description
   * - ``FibHardware``
     - bytes
     - FIB platform identifier (e.g. ``b"Tescan"``).
   * - ``Voltage``
     - float64
     - Primary ion beam voltage in V (divide by 1000 for kV).
   * - ``Current``
     - float64
     - Ion beam current in A (zero if not measured).
   * - ``ViewField``
     - float64
     - Field of view in mm.  Divided by the pixel count to obtain pixel
       size in µm — separately for the SIMS raster and the FIB SE images.

Inspecting available signals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :func:`rsciio.tofwerk.available_signals` to check which signals a file
contains before loading:

.. code-block:: python

    from rsciio.tofwerk import available_signals

    available_signals("acquisition.h5")
    # ['sum_spectrum', 'peak_data', 'fib_images']          # pre-processed file
    # ['sum_spectrum', 'peak_data', 'event_list', 'fib_images']  # raw file

Selecting signals
^^^^^^^^^^^^^^^^^

The ``signal`` parameter controls which signals are returned.  The default is
``"sum_spectrum"``, which is always fast.

.. code-block:: python

    import hyperspy.api as hs

    # Default: 1-D cumulative spectrum (fast, always available)
    s = hs.load("acquisition.h5", file_format="Tofwerk")

    # 4-D peak array (depth × y × x × m/z)
    s = hs.load("acquisition.h5", file_format="Tofwerk", signal="peak_data")

    # FIB SE image stack (depth × y × x), full FIB scan resolution
    s = hs.load("acquisition.h5", file_format="Tofwerk", signal="fib_images")

    # Raw TDC timestamps as a ragged signal
    s = hs.load("acquisition.h5", file_format="Tofwerk", signal="event_list")

    # Multiple signals at once
    signals = hs.load(
        "acquisition.h5",
        file_format="Tofwerk",
        signal=["sum_spectrum", "peak_data"],
    )

    # All signals available for this file
    signals = hs.load("acquisition.h5", file_format="Tofwerk", signal="all")

Valid values for ``signal``:

``"sum_spectrum"`` (default)
    1-D cumulative spectrum from ``FullSpectra/SumSpectrum``.  Always
    available.  Supports lazy loading.

``"peak_data"``
    4-D array ``(depth, y, x, m/z)``.

    * *Pre-processed files*: reads ``PeakData/PeakData`` directly.  Supports lazy
      loading (recommended for large files).
    * *Raw files*: reconstructs the 4-D array by walking the variable-length
      ``FullSpectra/EventList`` and integrating events within each peak window.
      Always eager.  On large files this can take several minutes; install
      ``tqdm`` for a progress bar and ``numba`` for a ~19× speed-up.

``"event_list"``
    Ragged object array ``(depth, y, x)`` of raw uint16 TDC timestamps, one
    variable-length array per pixel.  Present in all raw files; also available
    in pre-processed files if the Tofwerk software did not remove it.  Supports lazy
    loading.  HyperSpy represents it as a ragged signal and does not support
    plotting it directly.

``"fib_images"``
    3-D stack ``(depth, y, x)`` of secondary-electron images at full FIB scan
    resolution.  Available in FIB-SIMS files that contain a ``FIBImages``
    group.  Supports lazy loading.  If any images have a non-dominant shape
    (e.g. a truncated final frame), they are skipped with a warning.

``"all"``
    All signals available for the file.  Use
    :func:`~rsciio.tofwerk.available_signals` to see the list in advance.

Lazy loading
^^^^^^^^^^^^

Pass ``lazy=True`` to defer reading large arrays until ``.compute()`` is called:

.. code-block:: python

    s = hs.load(
        "large_acquisition.h5",
        file_format="Tofwerk",
        lazy=True,
        signal="peak_data",
    )
    # s.data is a dask array; inspect size before computing:
    print(s.data.nbytes / 1e9, "GB")
    s.compute()

Lazy loading is supported for ``"sum_spectrum"``, ``"peak_data"``
(pre-processed files), ``"event_list"``, and ``"fib_images"``.  Reconstructing
``"peak_data"`` from a raw file's EventList is always eager.

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

* ``tqdm`` — progress bars during EventList reconstruction (``signal="peak_data"``
  on a raw file) and during eager ``"event_list"`` loading.
* ``numba`` — JIT-accelerates EventList reconstruction (~19× faster than the
  NumPy fallback on large datasets).

Metadata
^^^^^^^^

FIB-SIMS specific fields are stored under ``Acquisition_instrument.FIB_SIMS``:

.. code-block:: python

    meta = s.metadata.Acquisition_instrument.FIB_SIMS

    meta.file_type              # "pre-processed" or "raw"
    meta.FIB.hardware           # FIB hardware identifier (e.g. "Tescan")
    meta.FIB.voltage_kV         # Ion beam accelerating voltage in kV
    meta.FIB.current_A          # Ion beam current in A
    meta.FIB.view_field_mm      # Field of view in mm
    meta.FIB.pixel_size_um      # SIMS pixel size in µm
    meta.ToF.ion_mode           # "positive" or "negative"
    meta.DAQ.tofdaq_version     # TofDAQ software version string

.. Note::
    The ``.h5`` extension is shared with other formats (EMD, Arina, etc.).
    The plugin identifies Tofwerk files by checking for TofDAQ-specific HDF5
    groups (``FullSpectra``, ``TimingData``, ``AcquisitionLog``) and the
    ``TofDAQ Version`` root attribute.  Specify ``file_format="Tofwerk"``
    explicitly if automatic detection fails.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.tofwerk
   :members:
