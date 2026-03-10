"""
Generate synthetic Tofwerk TofDAQ HDF5 fixture files for testing.

Run once from this directory:

    cd rsciio/tests/data/tofwerk
    python generate_fixtures.py

Then update the test registry:

    python -c "from rsciio.tests.registry_utils import update_registry; update_registry()"


==============================================================================
Fixture File Contents
==============================================================================

This file writes two test files, the structure of which have been deduced from
comparison to live-collected Tofwerk FIB-SIMS files collected from v1.99 of the
TofDAQ software, but optimized to be small file size.

Both fixtures share the following acquisition parameters (see module-level
constants NWRITES, NSEGS, NX, NPEAKS, NSAMPLES):

  NbrWrites   = 5      depth slices (milling steps)
  NbrSegments = 16     Y pixels (scan lines per write)
  NbrX        = 16     X pixels (pixels per scan line)
  NbrPeaks    = 10     peaks in the peak table
  NbrSamples  = 512    ADC samples per spectrum

Instrument metadata (same in both files):
  FIB hardware:       Tescan
  Accelerating voltage: 30 kV  (30000 V stored in FIBParams.Voltage)
  Ion beam current:   0 A  (not measured)
  Ion mode:           positive
  DAQ hardware:       Cronologic xTDC4
  TofDAQ version:     1.99
  Field of view:      10 µm  (FIBParams.ViewField = 1e-5 m)
  Pixel size:         0.625 µm/pixel  (10 µm / 16 pixels)
  Chamber pressure:   1.7e-4 Pa  (constant across all writes)
  Acquisition time:   2025-01-01T12:00:00-05:00  (UTC-5, Eastern Standard Time)

Peak table (10 synthetic peaks, uniform 1 Da spacing):
  Peak 0: label=b"nominal_0", mass=1.0 Da, window=[0.5, 1.5] Da
  Peak 1: label=b"nominal_1", mass=2.0 Da, window=[1.5, 2.5] Da
  ...
  Peak 9: label=b"nominal_9", mass=10.0 Da, window=[9.5, 10.5] Da

Mass axis: np.linspace(0.0, 200.0, 512) as float32 → 0.0 to 200.0 Da in 512
  uniformly-spaced steps (note: real MassAxis is non-uniform due to ToF
  physics; this fixture uses a uniform approximation for simplicity).

FIB images: 3 images (Image0000, Image0001, Image0002), each 128×128 float64,
  filled with uniform random values (seed 0).

BufTimes: uniform random float64 values, shape (5, 16), seed 2.

SaturationWarning: all zeros — no saturation in any write/segment.

Mass calibration (mode 0, two-parameter):
  p1 = 812.2415
  p2 = 222.0153
  SampleInterval = 8.333e-10 s  (~1.2 GHz ADC)
  Single Ion Signal = 1.0


fib_sims_raw.h5
---------------
Represents a file as written by TofDAQ before any post-processing.

Key distinction: contains FullSpectra/EventList, does NOT contain
PeakData/PeakData.

EventList: vlen uint16, shape (5, 16, 16).  Each cell holds a randomly-
  sized array of TDC sample indices simulating ion events.  Event counts per
  pixel follow a Poisson distribution with mean=20 (seed 42), values in
  [0, 9999].  Total events per file ≈ 5 × 16 × 16 × 20 = 25,600.

SumSpectrum: 512 float64 values drawn from Exponential(scale=10) (seed 1).

Expected reader output (rsciio.tofwerk.file_reader):
  Returns 2 signals:
    [0] Sum spectrum — shape (512,), axes: [m/z in Da (non-uniform, 512 pts)]
        metadata.General.title = "<stem> (sum spectrum)"
    [1] TIC map      — shape (16, 16), axes: [y in µm, x in µm]
        metadata.General.title = "<stem> (TIC map)"
        dtype: int32, values ≈ 20 per pixel (Poisson mean)
  metadata.Acquisition_instrument.FIB_SIMS.file_type = "raw"


fib_sims_opened.h5
------------------
Represents a file after it has been opened and saved by Tofwerk software.

Key distinction: contains PeakData/PeakData, does NOT contain
FullSpectra/EventList.

PeakData/PeakData: float32, shape (5, 16, 16, 10), gzip-compressed.  Values
  drawn from Exponential(scale=50) (seed 7), simulating peak-integrated ion
  counts.  All axes use the same fixture parameters as above.

Expected reader output (rsciio.tofwerk.file_reader):
  Returns 1 signal:
    [0] 4D peak data — shape (5, 16, 16, 10)
        axes:
          axis 0 — depth: size=5,  units="slice", scale=1,     navigate=True
          axis 1 — y:     size=16, units="µm",    scale=0.625, navigate=True
          axis 2 — x:     size=16, units="µm",    scale=0.625, navigate=True
          axis 3 — m/z:   values=[1.0, 2.0, …, 10.0] Da (non-uniform),
                          navigate=False
        dtype: float32
  metadata.Signal.signal_type = "FIB-SIMS"
  metadata.Signal.binned = True
  metadata.Acquisition_instrument.FIB_SIMS.file_type = "opened"
  metadata.Acquisition_instrument.FIB_SIMS.FIB.voltage_kV = 30.0
  metadata.Acquisition_instrument.FIB_SIMS.FIB.hardware = "Tescan"
  metadata.Acquisition_instrument.FIB_SIMS.FIB.pixel_size_um = 0.625
  metadata.Acquisition_instrument.FIB_SIMS.ToF.ion_mode = "positive"
  metadata.General.date = "2025-01-01"
  metadata.General.time = "12:00:00"
  metadata.General.time_zone = "-05:00"


==============================================================================
Tofwerk TofDAQ HDF5 Format Reference (reverse-engineered, so may have inaccuracies)
==============================================================================

Overview
--------
TofDAQ is the acquisition software for Tofwerk time-of-flight mass spectrometry
instruments, including the fibTOF FIB-SIMS system.  Every acquisition produces a
single ``.h5`` file.  The file passes through two states during a typical
workflow:

  Raw file     — written during acquisition; contains full-resolution TDC
                 timestamps in the EventList and a running SumSpectrum.
                 PeakData/PeakData is absent.

  Opened file  — produced when the raw file is opened and saved by Tofwerk's
                 desktop software (Tofwerk Acquire / Fiblys); the software
                 integrates the EventList over user-defined mass windows and
                 writes the 4-D peak-integrated array PeakData/PeakData.
                 All other groups are carried over byte-for-byte from the raw
                 file.

The distinction is detected by checking for ``"PeakData/PeakData" in file``.


HDF5 File Structure
-------------------
The top-level layout (all groups present in both file states unless noted):

  /                           (root — carries acquisition-wide attributes)
  ├── AcquisitionLog/
  │   └── Log                 compound dataset; one row per log entry
  ├── FIBImages/
  │   ├── Image0000/
  │   │   └── Data            float64 (H, W) secondary-electron survey image
  │   ├── Image0001/ ...
  │   └── Image000N/          images are acquired before/after each write
  ├── FIBParams/              group; attributes describe FIB column settings
  ├── FibParams/
  │   └── FibPressure/
  │       └── TwData          float64 (NbrWrites, 1) chamber pressure in Pa
  ├── FullSpectra/
  │   ├── MassAxis            float32 (NbrSamples,) mass-to-charge in Da
  │   ├── SumSpectrum         float64 (NbrSamples,) cumulative ion counts
  │   ├── SaturationWarning   uint8   (NbrWrites, NbrSegments) saturation flags
  │   └── EventList           vlen uint16 (NbrWrites, NbrSegments, NbrX)
  │                           [raw files only — absent in opened files]
  ├── PeakData/
  │   ├── PeakTable           compound dataset (NbrPeaks,) peak definitions
  │   └── PeakData            float32 (NbrWrites, NbrSegments, NbrX, NbrPeaks)
  │                           [opened files only — absent in raw files]
  └── TimingData/
      └── BufTimes            float64 (NbrWrites, NbrSegments) wall-clock time
                              of each buffer acquisition, in seconds


Root-Level Attributes
---------------------
These are stored directly on the root HDF5 group (``file.attrs``):

  TofDAQ Version          float32   Software version, e.g. 1.99.
                                    Presence of this attribute is the primary
                                    TofDAQ format-detection marker.
  FiblysGUIVersion        bytes     Version of the Fiblys GUI, e.g. b"1.12.2.0"
  DAQ Hardware            bytes     DAQ board identifier, e.g. b"Cronologic xTDC4"
  IonMode                 bytes     b"positive" or b"negative"
  NbrWrites               int32     Number of depth slices (milling steps).
  NbrSegments             int32     Number of scan lines (Y pixels) per write.
  NbrBufs                 int32     Number of hardware buffers; equals NbrSegments.
  NbrPeaks                int32     Number of peaks in the peak table.
  NbrSamples              int32     Number of time samples in each spectrum
                                    (length of MassAxis / SumSpectrum).
  NbrWaveforms            int32     Hardware waveform averaging count; normally 1.
  NbrBlocks               int32     Number of hardware acquisition blocks; normally 1.
  HDF5 File Creation Time bytes     Local datetime string: b"DD.MM.YYYY HH:MM:SS".
                                    Note: no timezone.  Use AcquisitionLog for
                                    an ISO-8601 timestamp with timezone offset.
  Computer ID             bytes     Hostname of the acquisition PC.
  Configuration File Contents
                          bytes     INI-format text blob of the full TofDAQ
                                    configuration.  Key field used by this
                                    plugin:
                                      [TOFParameter]
                                      Ch1FullScale = <FOV in µm>
                                    This is the field of view (physical scan
                                    width / height in µm) and is divided by
                                    NbrSegments to obtain the pixel size.

Note on spatial dimensions:
  NbrSegments is used for both the Y (scan line) and X (pixel) dimension in
  these fixtures because the fibTOF scans a square field.  In real files the
  X pixel count is the same as NbrSegments.  The ``NbrX`` dimension is
  implicit — derived from EventList.shape[2] or PeakData.shape[2].


AcquisitionLog/Log
------------------
Compound dataset.  Each row is one log entry.  Fields:

  timestamp   uint64    Monotonic hardware counter at time of log event.
  timestring  S26       ISO-8601 datetime with UTC offset, e.g.
                        b"2025-01-01T12:00:00-05:00".  The first entry is
                        written at acquisition start and is the authoritative
                        source for the file creation timestamp.
  logtext     S256      Human-readable description of the log event.


FIBImages/Image000N/Data
------------------------
float64 array, shape (H, W).  Secondary-electron images acquired by the FIB
column at various points during the experiment (typically before the first
write, after the last write, and optionally at intermediate steps).  Images
are sorted lexicographically by group name — Image0000 is earliest,
Image000N is latest.  Real files often have 7+ images; this fixture uses 3.

The pixel values are raw detector counts; no physical unit is stored.  The
image resolution (128×128 in these fixtures) is set by the FIB scan
parameters and is independent of the ToF raster size (NbrSegments × NbrX).


FIBParams (group attributes)
-----------------------------
All attributes are stored on the FIBParams group object, not as datasets.

  FibHardware           bytes     FIB platform identifier, e.g. b"Tescan".
  FibInterfaceVersion   bytes     Version of the FIB interface firmware.
  Voltage               float64   Primary ion beam accelerating voltage in V
                                  (e.g. 30000.0 → 30 kV).  Divide by 1000 for kV.
  Current               float64   Ion beam current in A at time of acquisition.
                                  Zero if not measured by the hardware.
  ViewField             float64   Field of view in m (e.g. 1e-5 = 10 µm).
                                  Used as fallback for pixel size when
                                  Configuration File Contents is unavailable.
  ScanSpeed             float64   FIB scan speed parameter (instrument-specific).


FibParams/FibPressure/TwData
-----------------------------
float64 array, shape (NbrWrites, 1).  Chamber pressure in Pa recorded once
per milling step.  Stored under ``FibParams`` (lowercase) rather than
``FIBParams`` (uppercase).  Mean value is reported in metadata as
``chamber_pressure_Pa``.


FullSpectra (group attributes)
--------------------------------
  MassCalibMode         int32   Calibration polynomial mode:
                                  0 — two-parameter: m = (p1 / (sample - p2))^2
                                  1 — three-parameter: adds offset p3
  MassCalibration p1    float64  Calibration coefficient p1.
  MassCalibration p2    float64  Calibration coefficient p2.
  MassCalibration p3    float64  Calibration coefficient p3 (mode 1 only).
  SampleInterval        float64  ADC sampling interval in seconds
                                 (e.g. 8.333e-10 s ≈ 1.2 GHz).
  Single Ion Signal     float64  Threshold (in ADC counts) for a single ion
                                 event.  Used to convert raw TDC values to
                                 ion counts.


FullSpectra/MassAxis
---------------------
float32 (NbrSamples,).  Pre-computed mass-to-charge ratio in Da for each
ADC sample index, derived from the MassCalibration coefficients.  Index 0
corresponds to sample 0 (t=0) and typically has mass ≈ 0 Da.  The axis is
non-uniform because m/z scales as t^2 (time-of-flight physics).


FullSpectra/SumSpectrum
------------------------
float64 (NbrSamples,).  Running sum of ion counts per mass sample across all
writes, segments, and pixels acquired so far.  Updated incrementally during
acquisition.  Useful for real-time monitoring and for raw-file analysis when
the EventList has not been processed.


FullSpectra/SaturationWarning
------------------------------
uint8 (NbrWrites, NbrSegments).  Per-buffer saturation flag: 0 = no
saturation, 1 = ADC or TDC saturation detected.  High values indicate that
the detector was overloaded during that (write, segment) combination and the
corresponding spectral data should be treated with caution.


FullSpectra/EventList  [raw files only]
----------------------------------------
Variable-length uint16 array, shape (NbrWrites, NbrSegments, NbrX).  Each
cell contains a 1-D array of raw TDC sample indices — one entry per detected
ion event.  To convert a sample index ``s`` to mass:

    mass_Da = (MassCalibration_p1 / (s - MassCalibration_p2)) ** 2

The number of events per cell follows a Poisson distribution whose mean is
proportional to ion beam current × dwell time.  These fixtures use a Poisson
mean of 20 events per pixel.

EventList is absent from opened files — the Tofwerk software consumes it to
produce PeakData/PeakData and does not preserve it.


PeakData/PeakTable
-------------------
Compound dataset, shape (NbrPeaks,).  Defines the mass windows used for peak
integration.  Fields:

  label                   S64       Human-readable ion label, e.g. b"28Si+"
  mass                    float32   Nominal peak center mass in Da.
  lower integration limit float32   Lower bound of integration window in Da.
  upper integration limit float32   Upper bound of integration window in Da.

The ``mass`` field is used as the signal axis values for the m/z dimension.
Real peak tables may have non-uniform spacing (user-defined peak lists), so
the axis must be stored as explicit values rather than offset+scale.

In these fixtures, peaks are synthetic: mass[i] = i+1 Da (1, 2, …, NbrPeaks)
with integration windows [i+0.5, i+1.5] Da (1 Da wide, centered on each peak).


PeakData/PeakData  [opened files only]
----------------------------------------
float32 array, shape (NbrWrites, NbrSegments, NbrX, NbrPeaks).  Peak-
integrated ion counts for each spatial pixel and depth slice.  Dimension
order maps to (depth, y, x, m/z) in HyperSpy terminology.

  axis 0 — depth  (NbrWrites):    milling steps / depth slices
  axis 1 — y      (NbrSegments):  scan lines (navigate=True)
  axis 2 — x      (NbrX):         pixels per line (navigate=True)
  axis 3 — m/z    (NbrPeaks):     peak-integrated mass channels (navigate=False)

Values are produced by integrating the EventList over each peak's
[lower, upper] mass window; units are ion counts.  Stored with gzip
compression in real files (replicated in these fixtures).


TimingData (group attributes)
------------------------------
  TofPeriod   int32   Number of ADC samples per ToF extraction pulse period
                      (e.g. 9500 samples at 1.2 GHz ≈ 7.9 µs).  Determines
                      the maximum observable mass.


TimingData/BufTimes
---------------------
float64 (NbrWrites, NbrSegments).  Wall-clock timestamp in seconds for each
buffer (one per scan line per milling step), measured from acquisition start.
Useful for computing depth profile time axes and diagnosing timing anomalies.


Pixel Size Derivation
----------------------
The physical pixel size (µm/pixel) is derived in priority order:

  1. Parse the INI blob in ``Configuration File Contents`` root attribute:
       [TOFParameter]
       Ch1FullScale = <FOV_um>    # total field of view in µm
     pixel_size_um = Ch1FullScale / NbrX

  2. Fall back to FIBParams.ViewField (field of view in m):
       pixel_size_um = (ViewField * 1e6) / NbrX

  In these fixtures, Ch1FullScale = 10.0 µm and NbrX = 16, giving
  pixel_size_um = 0.625 µm/pixel.


Creation Time Parsing
----------------------
Two sources are available in priority order:

  1. AcquisitionLog/Log[0]['timestring'] — ISO-8601 with UTC offset, e.g.
     "2025-01-01T12:00:00-05:00".  Parse with datetime.fromisoformat().
     This is the preferred source because it includes timezone information.

  2. Root attribute HDF5 File Creation Time — local datetime string in the
     format "DD.MM.YYYY HH:MM:SS" (European date order, no timezone).
     Parse with datetime.strptime(s, "%d.%m.%Y %H:%M:%S").

"""

import h5py
import numpy as np

NWRITES = 5
NSEGS = 16
NX = 16
NPEAKS = 10
NSAMPLES = 512


def _write_common(f, nwrites, nsegs, nx, npeaks, nsamples):
    """Write HDF5 groups common to both raw and opened fixture files."""
    # Root attributes
    f.attrs["TofDAQ Version"] = np.float32(1.99)
    f.attrs["FiblysGUIVersion"] = b"1.12.2.0"
    f.attrs["DAQ Hardware"] = b"Cronologic xTDC4"
    f.attrs["IonMode"] = b"positive"
    f.attrs["NbrWrites"] = np.int32(nwrites)
    f.attrs["NbrSegments"] = np.int32(nsegs)
    f.attrs["NbrBufs"] = np.int32(nsegs)
    f.attrs["NbrPeaks"] = np.int32(npeaks)
    f.attrs["NbrSamples"] = np.int32(nsamples)
    f.attrs["NbrWaveforms"] = np.int32(1)
    f.attrs["NbrBlocks"] = np.int32(1)
    f.attrs["HDF5 File Creation Time"] = b"01.01.2025 12:00:00"
    f.attrs["Computer ID"] = b"test-fixture"
    f.attrs["Configuration File Contents"] = (
        b"[TOFParameter]\n"
        b"Ch1FullScale=0.5\nCh2FullScale=0.5\nCh3FullScale=0.5\nCh4FullScale=0.5\n"
        b"Ch1Offset=-0.5\nCh2Offset=0\nCh3Offset=0\nCh4Offset=0\n"
        b"Ch1PreampGain=11\nCh2PreampGain=1\nCh3PreampGain=11\nCh4PreampGain=1\n"
        b"Ch1Record=1\nCh2Record=0\nCh3Record=0\nCh4Record=0\n"
    )

    # AcquisitionLog
    log_dtype = np.dtype(
        [
            ("timestamp", np.uint64),
            ("timestring", "S26"),
            ("logtext", "S256"),
        ]
    )
    log = np.array(
        [
            (0, b"2025-01-01T12:00:00-05:00", b"Acquisition started"),
            (1, b"2025-01-01T12:00:03-05:00", b"End of acquisition"),
        ],
        dtype=log_dtype,
    )
    f.create_group("AcquisitionLog").create_dataset("Log", data=log)

    # FIBImages
    fibimages = f.create_group("FIBImages")
    rng = np.random.default_rng(0)
    for i in range(3):
        img = rng.random((128, 128)).astype(np.float64)
        fibimages.create_group(f"Image{i:04d}").create_dataset("Data", data=img)

    # FIBParams
    fibparams = f.create_group("FIBParams")
    fibparams.attrs["FibHardware"] = b"Tescan"
    fibparams.attrs["FibInterfaceVersion"] = b"3.2.24"
    fibparams.attrs["Voltage"] = np.float64(30000.0)
    fibparams.attrs["Current"] = np.float64(0.0)
    fibparams.attrs["ViewField"] = np.float64(1e-5)
    fibparams.attrs["ScanSpeed"] = np.float64(10.0)

    # FullSpectra
    rng2 = np.random.default_rng(1)
    mass_axis = np.linspace(0.0, 200.0, nsamples, dtype=np.float32)
    sum_spec = rng2.exponential(10, nsamples).astype(np.float64)
    sat_warn = np.zeros((nwrites, nsegs), dtype=np.uint8)
    fullspectra = f.create_group("FullSpectra")
    fullspectra.attrs["MassCalibMode"] = np.int32(0)
    fullspectra.attrs["MassCalibration p1"] = np.float64(812.2415)
    fullspectra.attrs["MassCalibration p2"] = np.float64(222.0153)
    fullspectra.attrs["SampleInterval"] = np.float64(8.333e-10)
    # ClockPeriod = SampleInterval → clock_ratio = 1 (events are direct ADC indices)
    fullspectra.attrs["ClockPeriod"] = np.float64(8.333e-10)
    fullspectra.attrs["Single Ion Signal"] = np.float64(1.0)
    fullspectra.create_dataset("MassAxis", data=mass_axis)
    fullspectra.create_dataset("SumSpectrum", data=sum_spec)
    fullspectra.create_dataset("SaturationWarning", data=sat_warn)

    # PeakData/PeakTable (peak definitions — PeakData array added in opened fixture)
    peak_dtype = np.dtype(
        [
            ("label", "S64"),
            ("mass", np.float32),
            ("lower integration limit", np.float32),
            ("upper integration limit", np.float32),
        ]
    )
    peaks = np.array(
        [
            (
                f"nominal_{i}".encode(),
                float(i + 1),
                float(i) + 0.5,
                float(i) + 1.5,
            )
            for i in range(npeaks)
        ],
        dtype=peak_dtype,
    )
    f.create_group("PeakData").create_dataset("PeakTable", data=peaks)

    # TimingData
    timingdata = f.create_group("TimingData")
    timingdata.attrs["TofPeriod"] = np.int32(9500)
    rng3 = np.random.default_rng(2)
    buf_times = rng3.random((nwrites, nsegs)).astype(np.float64)
    timingdata.create_dataset("BufTimes", data=buf_times)

    # FibParams/FibPressure
    fibpressure = f.create_group("FibParams/FibPressure")
    fibpressure.create_dataset(
        "TwData", data=np.full((nwrites, 1), 1.7e-4, dtype=np.float64)
    )


def make_raw_fixture(path):
    """Create fib_sims_raw.h5 — no PeakData/PeakData, has EventList."""
    with h5py.File(path, "w") as f:
        _write_common(f, NWRITES, NSEGS, NX, NPEAKS, NSAMPLES)
        vlen = h5py.vlen_dtype(np.uint16)
        el = f["FullSpectra"].create_dataset(
            "EventList", shape=(NWRITES, NSEGS, NX), dtype=vlen
        )
        rng = np.random.default_rng(42)
        for w in range(NWRITES):
            for s in range(NSEGS):
                for x in range(NX):
                    n_events = int(rng.poisson(20))
                    # Values in [0, NSAMPLES) so they are valid ADC sample indices
                    # (ClockPeriod = SampleInterval → clock_ratio = 1 in the fixture)
                    el[w, s, x] = rng.integers(0, NSAMPLES, n_events, dtype=np.uint16)


def make_opened_fixture(path):
    """Create fib_sims_opened.h5 — has PeakData/PeakData."""
    with h5py.File(path, "w") as f:
        _write_common(f, NWRITES, NSEGS, NX, NPEAKS, NSAMPLES)
        rng = np.random.default_rng(7)
        peak_data = rng.exponential(50, (NWRITES, NSEGS, NX, NPEAKS)).astype(np.float32)
        f["PeakData"].create_dataset("PeakData", data=peak_data, compression="gzip")


if __name__ == "__main__":
    make_raw_fixture("fib_sims_raw.h5")
    make_opened_fixture("fib_sims_opened.h5")
