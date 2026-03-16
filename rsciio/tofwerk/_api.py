# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# RosettaSciIO is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RosettaSciIO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

"""
Reader for Tofwerk HDF5 files (.h5) written by TofDAQ acquisition software.

Supports fibTOF FIB-SIMS instruments (Tescan plasma-FIB + Tofwerk ToF-SIMS
detector).

Data dimensions
---------------
A fibTOF acquisition produces a 4-D dataset with the following axes:

* **depth** (``nwrites``, ``NbrWrites`` in HDF5 root attributes) — one entry
  per FIB milling + SIMS acquisition cycle.  Each write mills a thin layer
  from the sample surface and then rasters the FIB beam across the exposed
  face to collect SIMS spectra.  This is the depth-profiling axis.

* **y** (``nsegs``, ``NbrSegments``) — rows of the 2-D SIMS raster scan.

* **x** (``nx``, third dimension of ``PeakData/PeakData`` or
  ``FullSpectra/EventList``) — columns of the 2-D SIMS raster scan.

* **m/z** (``npeaks``) — mass-to-charge axis, one bin per peak defined in
  ``PeakData/PeakTable``.

Note that the SIMS ion data is acquired at a lower spatial resolution than
the simultaneously recorded FIB secondary-electron (SE) images.  For
example, a dataset described as "256×256 2×2" has FIB SE images of
256×256 pixels, while the SIMS ion data is 128×128 pixels — the "2×2"
denotes 2×2 SIMS pixel binning relative to the FIB scan grid.  The
``FIBParams.ViewField`` attribute (in mm) is divided by the SIMS pixel
count (128 in this example) to obtain the SIMS pixel size, not by the
FIB SE image resolution.

Example dimensions for a typical acquisition
("256×256 2×2, 1301 frames, 217 peaks")::

    PeakData/PeakData  shape: (1301, 128, 128, 217)
                              depth × y × x × m/z
    FullSpectra/EventList     shape: ( 1302,  128, 128)   [ragged]
                                     (depth,    y,   x)
                              each element:
                                  variable/ragged-length uint32 array of
                                  raw time-to-digital converter (TDC) timestamps, one
                                  per ion detection event.
                              len(EventList[w, s, x]) == ion count for that pixel.
    FIBImages/<slice>         shape: (256, 256)          [SE image, full FIB resolution]

File states
-----------
Two file states are handled:

* **Pre-processed** files (``PeakData/PeakData`` present) — the Tofwerk software
  has already integrated the raw events into per-peak counts.  The
  ``"sum_spectrum"`` signal (default) returns the 1-D cumulative spectrum.
  The ``"peak_data"`` signal returns the 4-D ``(depth, y, x, m/z)``
  peak-integrated array.  The Tofwerk software typically removes the
  EventList when opening a file, but if it is still present
  ``signal="event_list"`` will return it for reprocessing.

* **Raw** files (``FullSpectra/EventList`` only) — the raw TDC timestamp
  data has not yet been peak-integrated.  The ``"sum_spectrum"`` signal
  (default) returns the 1-D cumulative spectrum.  Pass
  ``signal="peak_data"`` to reconstruct the 4-D peak array from the
  EventList using the integration windows in ``PeakData/PeakTable``.
  Pass ``signal="event_list"`` to retrieve the raw ragged TDC timestamps.

The ``signal`` parameter controls which signals are returned.  Pass a
string or list of strings from ``{"sum_spectrum", "peak_data",
"event_list", "fib_images", "all"}``.  The default
``signal="sum_spectrum"`` is fast and always available; ``"peak_data"``
on a raw file triggers full EventList reconstruction; ``"fib_images"``
returns the stack of secondary-electron images at full FIB resolution.
"""

import enum
import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from rsciio._docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC
from rsciio.utils._decorator import jit_ifnumba

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal parameter enum
# ---------------------------------------------------------------------------


class TofwerkSignal(str, enum.Enum):
    """Valid values for the ``signal`` parameter of :func:`file_reader`."""

    SUM_SPECTRUM = "sum_spectrum"
    PEAK_DATA = "peak_data"
    EVENT_LIST = "event_list"
    FIB_IMAGES = "fib_images"
    ALL = "all"


# ---------------------------------------------------------------------------
# Format detection helpers
# ---------------------------------------------------------------------------


def _is_tofwerk_file(file):
    """Return True if the HDF5 file was written by TofDAQ."""
    return (
        "FullSpectra" in file
        and "TimingData" in file
        and "AcquisitionLog" in file
        and "TofDAQ Version" in file.attrs
    )


def _is_fib_sims(file):
    """Return True if this is a fibTOF FIB-SIMS file."""
    return _is_tofwerk_file(file) and "FIBImages" in file and "FIBParams" in file


def _has_peak_data(file):
    """Return True if Tofwerk software has pre-processed the file."""
    return "PeakData/PeakData" in file


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_creation_time(file):
    """
    Return (date_str, time_str, tz_str) from the acquisition log.

    Primary source: ``AcquisitionLog/Log[0]['timestring']`` — ISO-8601 with
    timezone offset (e.g. ``"2025-01-01T12:00:00-05:00"``).

    Fallback: root ``HDF5 File Creation Time`` attribute — local datetime
    string in ``"DD.MM.YYYY HH:MM:SS"`` format.
    """
    date_str = time_str = tz_str = ""
    try:
        timestring = file["AcquisitionLog/Log"][0]["timestring"]
        if isinstance(timestring, (bytes, np.bytes_)):
            timestring = timestring.decode()
        dt = datetime.fromisoformat(timestring)
        date_str = dt.date().isoformat()
        time_str = dt.strftime("%H:%M:%S")
        if dt.tzinfo is not None:
            total_seconds = int(dt.utcoffset().total_seconds())
            sign = "+" if total_seconds >= 0 else "-"
            h, m = divmod(abs(total_seconds) // 60, 60)
            tz_str = f"{sign}{h:02d}:{m:02d}"
    except Exception:
        try:
            raw = file.attrs.get("HDF5 File Creation Time", b"")
            if isinstance(raw, (bytes, np.bytes_)):
                raw = raw.decode()
            dt = datetime.strptime(raw, "%d.%m.%Y %H:%M:%S")
            date_str = dt.date().isoformat()
            time_str = dt.strftime("%H:%M:%S")
        except Exception:
            # Neither datetime source is available; return empty strings
            _logger.warning(
                "Could not parse acquisition time; date/time fields omitted."
            )
    return date_str, time_str, tz_str


def _parse_pixel_size(file, nx):
    """
    Return pixel size in µm.

    Source: ``FIBParams.ViewField`` (in mm, converted to µm) divided by *nx*.

    Note: the ``[TOFParameter]`` section of ``Configuration File Contents``
    contains ``Ch1FullScale``–``Ch4FullScale`` values which are ADC input
    voltage ranges (V), not spatial dimensions, and must not be used for pixel
    size calculation.
    """
    try:
        view_field_mm = float(np.asarray(file["FIBParams"].attrs["ViewField"]).flat[0])
        view_field_um = view_field_mm * 1e3
        return view_field_um / nx
    except Exception:
        _logger.warning(
            "Could not read FIBParams.ViewField metadata parameter; spatial axes will be uncalibrated."
        )
        return None


# ---------------------------------------------------------------------------
# Axes builders
# ---------------------------------------------------------------------------


def _build_axes_preprocessed(nwrites, nsegs, nx, peak_masses, pixel_size_um):
    """Return the list of 4 axis dicts for a pre-processed (PeakData) file."""
    return [
        {
            "name": "depth",
            "size": nwrites,
            "offset": 0,
            "scale": 1,
            "units": "slice",
            "navigate": True,
        },
        {
            "name": "y",
            "size": nsegs,
            **(
                {"offset": 0, "scale": pixel_size_um, "units": "µm"}
                if pixel_size_um is not None
                else {}
            ),
            "navigate": True,
        },
        {
            "name": "x",
            "size": nx,
            **(
                {"offset": 0, "scale": pixel_size_um, "units": "µm"}
                if pixel_size_um is not None
                else {}
            ),
            "navigate": True,
        },
        {
            "name": "m/z",
            "axis": peak_masses.astype(float),
            "units": "Da",
            "navigate": False,
            "is_binned": True,
        },
    ]


def _build_axes_raw_sum(mass_axis):
    """Return a 1-D axis dict for the raw sum spectrum."""
    return [
        {
            "name": "m/z",
            "axis": mass_axis.astype(float),
            "units": "Da",
            "navigate": False,
            "is_binned": True,
        }
    ]


def _build_axes_event_list(nwrites, nsegs, nx, pixel_size_um):
    """Return 3 navigation axis dicts for the ragged EventList signal."""
    return [
        {
            "name": "depth",
            "size": nwrites,
            "offset": 0,
            "scale": 1,
            "units": "slice",
            "navigate": True,
        },
        {
            "name": "y",
            "size": nsegs,
            **(
                {"offset": 0, "scale": pixel_size_um, "units": "µm"}
                if pixel_size_um is not None
                else {}
            ),
            "navigate": True,
        },
        {
            "name": "x",
            "size": nx,
            **(
                {"offset": 0, "scale": pixel_size_um, "units": "µm"}
                if pixel_size_um is not None
                else {}
            ),
            "navigate": True,
        },
    ]


def _build_axes_fib_images(n_images, fib_ny, fib_nx, fib_pixel_size_um):
    """Return 3 axis dicts for a FIB SE image stack (depth, y, x)."""
    return [
        {
            "name": "depth",
            "size": n_images,
            "offset": 0,
            "scale": 1,
            "units": "slice",
            "navigate": True,
        },
        {
            "name": "y",
            "size": fib_ny,
            **(
                {"offset": 0, "scale": fib_pixel_size_um, "units": "µm"}
                if fib_pixel_size_um is not None
                else {}
            ),
            "navigate": False,
        },
        {
            "name": "x",
            "size": fib_nx,
            **(
                {"offset": 0, "scale": fib_pixel_size_um, "units": "µm"}
                if fib_pixel_size_um is not None
                else {}
            ),
            "navigate": False,
        },
    ]


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


def _build_metadata(file, filename, has_peak_data):
    """Construct the HyperSpy metadata dict from TofDAQ HDF5 attributes."""
    date_str, time_str, tz_str = _parse_creation_time(file)
    stem = Path(filename).stem

    # Root-level attributes
    root_attrs = file.attrs
    fullspectra_attrs = file["FullSpectra"].attrs
    timingdata_attrs = file["TimingData"].attrs

    # FIBParams (may not be present for non-FIB-SIMS files)
    fib_meta = {}
    try:
        fibparams_attrs = file["FIBParams"].attrs
        nx = int(np.asarray(root_attrs.get("NbrSegments", 1)).flat[0])
        pixel_size_um = _parse_pixel_size(file, nx)

        fib_meta["FIB"] = {
            "hardware": _decode(fibparams_attrs.get("FibHardware", b"")),
            "voltage_kV": float(np.asarray(fibparams_attrs["Voltage"]).flat[0]) / 1000,
            "current_A": float(np.asarray(fibparams_attrs["Current"]).flat[0]),
            "view_field_mm": float(np.asarray(fibparams_attrs["ViewField"]).flat[0]),
            "pixel_size_um": pixel_size_um,
        }
    except Exception:
        # FIBParams group absent in non-FIB-SIMS TofDAQ files; FIB metadata omitted
        _logger.warning("FIBParams not found; FIB metadata will not be populated.")

    # Mass range: use peak table for pre-processed files, raw spectrum axis for raw files
    try:
        if has_peak_data:
            peak_masses = np.asarray(file["PeakData/PeakTable"]["mass"]).astype(float)
            mass_range = [float(peak_masses.min()), float(peak_masses.max())]
        else:
            mass_axis = np.asarray(file["FullSpectra/MassAxis"])
            valid = np.isfinite(mass_axis) & (mass_axis > 0)
            mass_range = [float(mass_axis[valid].min()), float(mass_axis[valid].max())]
    except Exception:
        mass_range = [0.0, 0.0]

    # Chamber pressure
    try:
        pressure_data = np.asarray(file["FibParams/FibPressure/TwData"])
        chamber_pressure = float(np.mean(pressure_data))
    except Exception:
        chamber_pressure = None

    tof_meta = {
        "ion_mode": _decode(root_attrs.get("IonMode", b"")),
        "n_peaks": int(np.asarray(root_attrs.get("NbrPeaks", 0)).flat[0]),
        "mass_calib_mode": int(
            np.asarray(fullspectra_attrs.get("MassCalibMode", 0)).flat[0]
        ),
        "mass_calib_p1": float(
            np.asarray(fullspectra_attrs.get("MassCalibration p1", 0.0)).flat[0]
        ),
        "mass_calib_p2": float(
            np.asarray(fullspectra_attrs.get("MassCalibration p2", 0.0)).flat[0]
        ),
        "sample_interval_s": float(
            np.asarray(fullspectra_attrs.get("SampleInterval", 0.0)).flat[0]
        ),
        "single_ion_signal": float(
            np.asarray(fullspectra_attrs.get("Single Ion Signal", 1.0)).flat[0]
        ),
        "tof_period_samples": int(
            np.asarray(timingdata_attrs.get("TofPeriod", 0)).flat[0]
        ),
        "mass_range_Da": mass_range,
    }
    # Optional p3 calibration coefficient
    if "MassCalibration p3" in fullspectra_attrs:
        tof_meta["mass_calib_p3"] = float(
            np.asarray(fullspectra_attrs["MassCalibration p3"]).flat[0]
        )

    daq_meta = {
        "hardware": _decode(root_attrs.get("DAQ Hardware", b"")),
        "tofdaq_version": str(np.asarray(root_attrs.get("TofDAQ Version", "")).flat[0]),
        "gui_version": _decode(root_attrs.get("FiblysGUIVersion", b"")),
        "computer_id": _decode(root_attrs.get("Computer ID", b"")),
    }

    fib_sims_meta = {
        **fib_meta,
        "ToF": tof_meta,
        "DAQ": daq_meta,
        "n_depth_slices": int(np.asarray(root_attrs.get("NbrWrites", 0)).flat[0]),
        "n_segments": int(np.asarray(root_attrs.get("NbrSegments", 0)).flat[0]),
        "file_type": "pre-processed" if has_peak_data else "raw",
    }
    if chamber_pressure is not None:
        fib_sims_meta["chamber_pressure_Pa"] = chamber_pressure

    # PeakTable: stored in Signal so users can modify integration windows and reintegrate
    peak_table_list = None
    if "PeakData/PeakTable" in file:
        peak_table_list = _peak_table_to_list(np.asarray(file["PeakData/PeakTable"]))

    signal_meta = {"signal_type": "FIB-SIMS"}
    if peak_table_list is not None:
        signal_meta["peak_table"] = peak_table_list

    metadata = {
        "General": {
            "title": stem,
            "original_filename": str(filename),
        },
        "Signal": signal_meta,
        "Acquisition_instrument": {
            "FIB_SIMS": fib_sims_meta,
        },
    }
    if date_str:
        metadata["General"]["date"] = date_str
    if time_str:
        metadata["General"]["time"] = time_str
    if tz_str:
        metadata["General"]["time_zone"] = tz_str

    return metadata


def _build_original_metadata(file):
    """
    Return a dict of raw HDF5 root attributes and key group attributes,
    decoded for JSON serialisability.
    """
    orig = {}
    for k, v in file.attrs.items():
        orig[k] = _decode_attr(v)

    # PeakTable: store as a list of dicts (raw record of what was in the file)
    if "PeakData/PeakTable" in file:
        orig["PeakTable"] = _peak_table_to_list(np.asarray(file["PeakData/PeakTable"]))

    # Capture FIBImages as a list of image names (not data — avoid embedding large arrays)
    if "FIBImages" in file:
        orig["FIBImages"] = list(file["FIBImages"].keys())

    # SaturationWarning is a per-pixel flag array — store only shape/dtype, not data
    if "FullSpectra/SaturationWarning" in file:
        ds = file["FullSpectra/SaturationWarning"]
        orig["SaturationWarning"] = {"shape": list(ds.shape), "dtype": str(ds.dtype)}

    # MassAxis and FullSpectra timing attributes — needed to reintegrate EventList
    # data without re-opening the file (see FIBSIMSSpectrum.reintegrate_peaks).
    if "FullSpectra/MassAxis" in file:
        orig["MassAxis"] = np.asarray(file["FullSpectra/MassAxis"]).tolist()
    if "FullSpectra" in file:
        fs_attrs = {}
        for key in ("ClockPeriod", "SampleInterval"):
            if key in file["FullSpectra"].attrs:
                fs_attrs[key] = float(
                    np.asarray(file["FullSpectra"].attrs[key]).flat[0]
                )
        if fs_attrs:
            orig["FullSpectra"] = fs_attrs

    return orig


# ---------------------------------------------------------------------------
# PeakTable helpers
# ---------------------------------------------------------------------------


def _peak_table_to_list(pt):
    """
    Convert a PeakTable structured array to a JSON-serialisable list of dicts.

    Parameters
    ----------
    pt : numpy.ndarray
        Structured array with fields ``label``, ``mass``,
        ``lower integration limit``, ``upper integration limit``.

    Returns
    -------
    list of dict
        Each dict has keys ``label`` (str), ``mass`` (float),
        ``lower_integration_limit`` (float), ``upper_integration_limit`` (float).
    """
    return [
        {
            "label": _decode(row["label"]),
            "mass": float(row["mass"]),
            "lower_integration_limit": float(row["lower integration limit"]),
            "upper_integration_limit": float(row["upper integration limit"]),
        }
        for row in pt
    ]


def _peak_table_from_list(peak_table_list):
    """
    Convert a peak table list of dicts back to arrays suitable for integration.

    Parameters
    ----------
    peak_table_list : list of dict
        As returned by :func:`_peak_table_to_list`.

    Returns
    -------
    masses : numpy.ndarray, float64
    peak_low : numpy.ndarray, float64
    peak_high : numpy.ndarray, float64
    """
    masses = np.array([p["mass"] for p in peak_table_list], dtype=float)
    peak_low = np.array(
        [p["lower_integration_limit"] for p in peak_table_list], dtype=float
    )
    peak_high = np.array(
        [p["upper_integration_limit"] for p in peak_table_list], dtype=float
    )
    return masses, peak_low, peak_high


# ---------------------------------------------------------------------------
# EventList / peak data reconstruction helpers
# ---------------------------------------------------------------------------


def _count_active_channels(ini):
    """
    Return the number of active TDC recording channels from the INI config string.

    TofDAQ records events from multiple ADC/TDC channels simultaneously
    (e.g. Ch1 and Ch3 in a typical fibTOF setup).  Each active channel
    contributes an independent copy of every ion event to the EventList,
    so the raw event count per pixel is multiplied by this factor.

    Parameters
    ----------
    ini : str
        Contents of the ``Configuration File Contents`` root attribute.
    """
    matches = re.findall(r"Ch\dRecord=(\d)", ini)
    return max(sum(int(m) for m in matches), 1)


def _flatten_event_row(row):
    """
    Pack a row of variable-length event arrays into a flat contiguous buffer.

    Parameters
    ----------
    row : array-like of arrays
        Object array of shape ``(nx,)`` as returned by ``el[w, s, :]``.

    Returns
    -------
    flat : numpy.ndarray, dtype int64
        All events concatenated in pixel order.
    offsets : numpy.ndarray, shape (nx+1,), dtype int64
        ``flat[offsets[x]:offsets[x+1]]`` are the events for pixel *x*.
    """
    nx = len(row)
    lengths = np.array([len(r) for r in row], dtype=np.int64)
    offsets = np.empty(nx + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    flat = (
        np.concatenate(row).astype(np.int64)
        if offsets[-1] > 0
        else np.empty(0, dtype=np.int64)
    )
    return flat, offsets


@jit_ifnumba(cache=True)
def _accumulate_peak_counts(
    flat,
    offsets,
    nx,
    clock_ratio,
    nbr_samples,
    mass_axis,
    sorted_low,
    sorted_high,
    npeaks,
    result_sx,
):  # pragma: no cover
    """
    Accumulate per-pixel peak counts into *result_sx* (sorted-peak order).

    Parameters
    ----------
    flat : int64 array
        Flat event buffer from :func:`_flatten_event_row`.
    offsets : int64 array, shape (nx+1,)
        Pixel start/end indices into *flat*.
    nx : int
        Number of pixels in this row.
    clock_ratio : int
        TDC-to-ADC sample index divisor.
    nbr_samples : int
        Length of MassAxis; events outside [0, nbr_samples) are discarded.
    mass_axis : float64 array
        Calibrated mass (Da) for each ADC sample index.
    sorted_low, sorted_high : float64 arrays, shape (npeaks,)
        Integration window bounds in ascending mass order.
    npeaks : int
        Number of peaks.
    result_sx : float32 array, shape (nx, npeaks)
        Output accumulator, pre-zeroed by caller.  Indexed by sorted peak order.
    """
    for x in range(nx):
        start = offsets[x]
        end = offsets[x + 1]
        for i in range(start, end):
            sample = flat[i] // clock_ratio
            if sample < 0 or sample >= nbr_samples:
                continue
            mass = mass_axis[sample]
            bin_idx = np.searchsorted(sorted_low, mass, side="right") - 1
            if bin_idx >= 0 and mass <= sorted_high[bin_idx]:
                result_sx[x, bin_idx] += 1


def compute_peak_data_from_eventlist(
    event_list, mass_axis, nbr_samples, clock_ratio, normalization, peak_table
):
    """
    Reconstruct the 4-D peak-integrated array from raw EventList data.

    This replicates the processing performed by the Tofwerk proprietary
    software when opening a raw file:

    1. For each pixel's variable-length EventList (raw TDC timestamps),
       convert timestamps to ADC sample indices by integer division with the
       TDC-to-ADC clock ratio (``SampleInterval / ClockPeriod``, typically 64).
    2. Look up the calibrated mass for each sample index via ``MassAxis``.
    3. Count events that fall within each peak's integration window.
    4. Divide by ``NbrWaveforms × NActiveChannels`` — the number of times each
       physical ion is recorded in the EventList (once per ToF cycle per active
       recording channel).

    If ``numba`` is installed, events are processed via a JIT-compiled loop
    over a flat event buffer (no intermediate arrays per pixel).  Otherwise
    falls back to a vectorised NumPy path that allocates per-pixel arrays but
    is still significantly faster than a pure-Python triple loop.

    Parameters
    ----------
    event_list : array-like, shape (nwrites, nsegs, nx)
        Ragged object array of uint16 TDC timestamps, one variable-length
        array per pixel.  Accepts h5py variable-length datasets or
        numpy object arrays (as loaded by ``signal="event_list"``).
    mass_axis : numpy.ndarray, shape (nbr_samples,)
        Calibrated mass (Da) for each ADC sample index, from
        ``FullSpectra/MassAxis``.
    nbr_samples : int
        Length of ``mass_axis``; events outside ``[0, nbr_samples)`` are
        discarded.
    clock_ratio : int
        TDC-to-ADC sample index divisor (``SampleInterval / ClockPeriod``).
        Pass 1 if ``ClockPeriod`` is not available.
    normalization : int
        ``NbrWaveforms × NActiveChannels`` — the divisor used to convert raw
        event counts to per-waveform ion counts.
    peak_table : list of dict
        Integration windows.  Each dict must have keys
        ``lower_integration_limit`` and ``upper_integration_limit`` (Da).

    Returns
    -------
    numpy.ndarray
        Shape ``(nwrites, nsegs, nx, npeaks)``, dtype float32.
        Identical in structure to ``PeakData/PeakData`` in a pre-processed file.
    """
    el = event_list
    mass_axis = np.asarray(mass_axis, dtype=np.float64)
    _, peak_low, peak_high = _peak_table_from_list(peak_table)
    npeaks = len(peak_table)
    nwrites, nsegs, nx = el.shape

    # Sort peak windows for O(log n) assignment via searchsorted
    sort_idx = np.argsort(peak_low)
    sorted_low = peak_low[sort_idx]
    sorted_high = peak_high[sort_idx]
    # Inverse permutation: maps sorted-peak index back to original peak order.
    # Used to avoid numpy's advanced-indexing-on-LHS shape quirk where
    # result[w, s, :, sort_idx] = arr yields shape (npeaks, nx) not (nx, npeaks).
    inv_sort_idx = np.argsort(sort_idx)

    result = np.zeros((nwrites, nsegs, nx, npeaks), dtype=np.float32)

    _logger.warning(
        f"Reconstructing peak data from EventList "
        f"({nwrites} depth slices × {nsegs} × {nx} pixels × {npeaks} peaks). "
        f"This may take considerable time depending on the size of the file. "
        f"Consider opening the file with Tofwerk software first for instant lazy loading."
    )

    # Prefer the numba JIT path when numba is installed: events are packed into a
    # flat contiguous buffer per row and processed in a single compiled loop,
    # avoiding per-pixel Python object overhead entirely.  Falls back to a
    # vectorised NumPy path (one array allocation per pixel) when numba is absent.
    try:
        import numba  # noqa: F401

        _use_numba = True
    except ImportError:
        _use_numba = False

    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        _tqdm = None

    def _make_pbar(n, desc="Reconstructing peak data"):
        if _tqdm is not None:
            return _tqdm(total=n, desc=desc, unit=" slices")
        return None

    if _use_numba:
        # numba path: flatten each (write, segment) row into a contiguous int64
        # buffer and dispatch to the JIT-compiled _accumulate_peak_counts kernel.
        # result_sx is reused across rows to avoid repeated allocation.
        # Avoids ~6 intermediate per-pixel arrays; benchmarked as ~19× faster than a
        # per-pixel HDF5 read loop on a 2.2GB dataset.
        result_sx = np.zeros((nx, npeaks), dtype=np.float32)
        pbar = _make_pbar(nwrites)
        for w in range(nwrites):
            for s in range(nsegs):
                # event_list (el) shape is (depth, y, x); w=depth slice, s=y row, :=all x
                # pixels. Each pixel holds variable-length ToF clock ticks, so we
                # flatten into a contiguous int64 buffer (flat) + per-pixel
                # slice boundaries (offsets) for the Numba kernel.
                flat, offsets = _flatten_event_row(el[w, s, :])
                result_sx[:] = 0
                _accumulate_peak_counts(
                    flat,
                    offsets,
                    nx,
                    clock_ratio,
                    nbr_samples,
                    mass_axis,
                    sorted_low,
                    sorted_high,
                    npeaks,
                    result_sx,
                )
                result[w, s] = result_sx[:, inv_sort_idx]
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
    else:
        # NumPy fallback path: iterate over pixels explicitly, converting each
        # pixel's variable-length event array to masses and binning with
        # searchsorted + bincount.  Slower than the numba path but avoids any
        # compiled dependency.  Reading per row rather than per pixel is still
        # Benchmarked as ~3× faster than a per-pixel HDF5 read loop on a 2.2GB dataset.
        pbar = _make_pbar(nwrites)
        for w in range(nwrites):
            for s in range(nsegs):
                row = el[w, s, :]
                for x in range(nx):
                    events = row[x]
                    if len(events) == 0:
                        continue
                    processed = events // clock_ratio
                    valid = (processed >= 0) & (processed < nbr_samples)
                    if not valid.any():
                        continue
                    event_masses = mass_axis[processed[valid]]
                    bin_idx = (
                        np.searchsorted(sorted_low, event_masses, side="right") - 1
                    )
                    in_peak = (
                        (bin_idx >= 0)
                        & (bin_idx < npeaks)
                        & (event_masses <= sorted_high[bin_idx])
                    )
                    counts = np.bincount(bin_idx[in_peak], minlength=npeaks)
                    result[w, s, x, sort_idx] = counts
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

    pbar = _make_pbar(nwrites, desc="Normalising")
    for w in range(nwrites):
        result[w] /= normalization
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    return result


# ---------------------------------------------------------------------------
# Private file-level EventList reconstruction helper
# ---------------------------------------------------------------------------


def _compute_peak_data_from_file(file, peak_table=None):
    """
    Extract all required arrays from an open HDF5 file and call
    :func:`compute_peak_data_from_eventlist`.

    This is the internal entry point used by :func:`file_reader`.  External
    callers should use the public API via a loaded EventList signal and
    :func:`compute_peak_data_from_eventlist` directly.

    Parameters
    ----------
    file : h5py.File
        Open Tofwerk TofDAQ HDF5 file (must contain ``FullSpectra/EventList``).
    peak_table : list of dict or None
        Integration windows.  If None, read from ``PeakData/PeakTable`` in
        the file.

    Returns
    -------
    numpy.ndarray
        Shape ``(nwrites, nsegs, nx, npeaks)``, dtype float32.
    """
    if peak_table is None:
        peak_table = _peak_table_to_list(np.asarray(file["PeakData/PeakTable"]))

    mass_axis = np.asarray(file["FullSpectra/MassAxis"], dtype=np.float64)
    nbr_samples = len(mass_axis)

    fs_attrs = file["FullSpectra"].attrs
    if "ClockPeriod" in fs_attrs:
        sample_interval = float(np.asarray(fs_attrs.get("SampleInterval", 1.0)).flat[0])
        clock_period = float(np.asarray(fs_attrs["ClockPeriod"]).flat[0])
        clock_ratio = int(round(sample_interval / clock_period)) if clock_period else 1
    else:
        clock_ratio = 1

    nbr_waveforms = int(np.asarray(file.attrs.get("NbrWaveforms", 1)).flat[0])
    ini = _decode(file.attrs.get("Configuration File Contents", b""))
    n_active = _count_active_channels(ini)
    normalization = nbr_waveforms * n_active

    event_list = file["FullSpectra/EventList"]
    return compute_peak_data_from_eventlist(
        event_list, mass_axis, nbr_samples, clock_ratio, normalization, peak_table
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _decode(value):
    """Decode bytes to str; pass through strings unchanged."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_attr(value):
    """Make an HDF5 attribute JSON-serialisable."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "U", "O"):
            return [_decode(v) for v in value.flat]
        return value.tolist()
    return value


# ---------------------------------------------------------------------------
# EventList post-processing
# ---------------------------------------------------------------------------


def _mark_event_list_ragged(signal):
    """Mark EventList signals as ragged so HyperSpy handles them correctly.

    Setting ``signal.ragged = True`` registers the variable-length TDC
    timestamp arrays as HyperSpy's native ragged signal type.  The signal
    repr becomes ``(depth, y, x | ragged)`` and ``plot()`` raises HyperSpy's
    standard ``RuntimeError("Plotting ragged signal is not supported.")``.
    """
    signal.ragged = True
    return signal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_signals(filename):
    """
    Return the list of signal names available in a Tofwerk TofDAQ HDF5 file.

    This is a fast, read-only inspection call — no data arrays are loaded.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the ``.h5`` file.

    Returns
    -------
    list of str
        Subset of ``["sum_spectrum", "peak_data", "event_list", "fib_images"]``
        depending on what is present in the file.

    Raises
    ------
    IOError
        If the file is not a Tofwerk TofDAQ HDF5 file.

    Examples
    --------
    >>> from rsciio.tofwerk import available_signals
    >>> available_signals("my_acquisition.h5")
    ['sum_spectrum', 'peak_data', 'fib_images']
    """
    import h5py

    with h5py.File(str(filename), "r") as f:
        if not _is_tofwerk_file(f):
            raise IOError(
                f"{filename!r} is not a Tofwerk TofDAQ HDF5 file. "
                "Missing required groups or 'TofDAQ Version' root attribute."
            )
        signals = ["sum_spectrum"]
        has_peak_data = _has_peak_data(f)
        has_event_list = "FullSpectra/EventList" in f
        has_peak_table = "PeakData/PeakTable" in f
        if has_peak_data or (has_event_list and has_peak_table):
            signals.append("peak_data")
        if has_event_list:
            signals.append("event_list")
        if "FIBImages" in f and len(f["FIBImages"]) > 0:
            signals.append("fib_images")
    return signals


# our custom chunks API requires a different docstring than the one in
# rosettasciio/rsciio/_docstrings.py:
_CHUNKS_READ_DOC = """chunks : tuple, int, dict, str or None, default=None
        The chunks used when reading the data lazily. This argument is passed
        to :func:`dask.array.from_array`. Only has an effect when ``lazy=True``.
        If ``None``, a signal-appropriate default is used: one chunk per depth
        slice for ``"peak_data"`` and ``"event_list"``, and a single chunk for
        ``"sum_spectrum"`` and ``"fib_images"``.
    """


def file_reader(filename, lazy=False, signal="sum_spectrum", chunks=None):
    """
    Read a Tofwerk TofDAQ HDF5 file.

    Parameters
    ----------
    %s
    %s
    signal : str or list of str, optional
        Which signal(s) to return.  Valid values:

        ``"sum_spectrum"`` (default)
            1-D cumulative spectrum from ``FullSpectra/SumSpectrum``.
            Always available.
        ``"peak_data"``
            4-D array ``(depth, y, x, m/z)``.  For pre-processed files, reads
            ``PeakData/PeakData`` directly.  For raw files, reconstructs
            from ``FullSpectra/EventList`` using the windows in
            ``PeakData/PeakTable``.
        ``"event_list"``
            Ragged object array ``(depth, y, x)`` of raw uint16 TDC
            timestamps.  Present in all raw files; also available in
            pre-processed files if the Tofwerk software did not remove it.
        ``"fib_images"``
            3-D array ``(depth, y, x)`` of secondary-electron images at
            full FIB scan resolution (e.g. 256×256).  Available only in
            FIB-SIMS files that contain a ``FIBImages`` group.
        ``"all"``
            All signals available for the file.

        Pass a list to request multiple specific signals, e.g.
        ``signal=["sum_spectrum", "peak_data"]``.  The returned list has
        one entry per requested signal (or all available for ``"all"``).
    %s

    Raises
    ------
    IOError
        If the file is not a Tofwerk TofDAQ HDF5 file.
    ValueError
        If ``signal`` contains an unrecognised value.

    %s
    """
    import h5py

    filename = str(filename)
    f = h5py.File(filename, "r")

    if not _is_tofwerk_file(f):
        f.close()
        raise IOError(
            f"{filename!r} is not a Tofwerk TofDAQ HDF5 file. "
            "Missing required groups or 'TofDAQ Version' root attribute."
        )

    # ------------------------------------------------------------------
    # Parse and validate the signal parameter
    # ------------------------------------------------------------------
    if isinstance(signal, str):
        signal_list = [signal]
    else:
        signal_list = list(signal)

    try:
        signal_list = [TofwerkSignal(s) for s in signal_list]
    except ValueError:
        f.close()
        valid = [s.value for s in TofwerkSignal]
        raise ValueError(
            f"Invalid signal value(s) in {signal!r}. Must be one or more of {valid}."
        )

    want_all = TofwerkSignal.ALL in signal_list
    want_sum = want_all or TofwerkSignal.SUM_SPECTRUM in signal_list
    want_peak = want_all or TofwerkSignal.PEAK_DATA in signal_list
    want_evl = want_all or TofwerkSignal.EVENT_LIST in signal_list
    want_fib = want_all or TofwerkSignal.FIB_IMAGES in signal_list

    # ------------------------------------------------------------------
    # File state and metadata
    # ------------------------------------------------------------------
    has_peak_data = _has_peak_data(f)
    metadata = _build_metadata(f, filename, has_peak_data)
    original_metadata = _build_original_metadata(f)
    stem = metadata["General"]["title"]

    # ------------------------------------------------------------------
    # Determine spatial dimensions
    # ------------------------------------------------------------------
    nwrites = int(np.asarray(f.attrs.get("NbrWrites", 1)).flat[0])
    nsegs = int(np.asarray(f.attrs.get("NbrSegments", 1)).flat[0])
    has_event_list = "FullSpectra/EventList" in f
    if has_peak_data:
        nx = f["PeakData/PeakData"].shape[2]
    elif has_event_list:
        nx = f["FullSpectra/EventList"].shape[2]
    else:
        nx = nsegs
    pixel_size_um = _parse_pixel_size(f, nx)

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------
    has_peak_table = "PeakData/PeakTable" in f
    can_peak = has_peak_data or (has_event_list and has_peak_table)
    can_evl = has_event_list
    can_fib = "FIBImages" in f and len(f["FIBImages"]) > 0

    if want_peak and not can_peak:
        _logger.warning(
            "signal='peak_data' requested but not available "
            "(no PeakData/PeakData and no EventList+PeakTable). Skipping."
        )
    if want_evl and not can_evl:
        _logger.warning(
            "signal='event_list' requested but FullSpectra/EventList is not present "
            "(the Tofwerk software may have removed it when opening the file). Skipping."
        )
    if want_fib and not can_fib:
        _logger.warning(
            "signal='fib_images' requested but FIBImages group is absent or empty. Skipping."
        )

    produce_sum = want_sum
    produce_peak = want_peak and can_peak
    produce_evl = want_evl and can_evl
    produce_fib = want_fib and can_fib

    # Log available-but-not-requested signals
    if not want_all:
        available = ["sum_spectrum"]
        if can_peak:
            available.append("peak_data")
        if can_evl:
            available.append("event_list")
        if can_fib:
            available.append("fib_images")
        requested = {s.value for s in signal_list}
        not_requested = [s for s in available if s not in requested]
        if not_requested:
            _logger.info(
                f"Other signals available: {not_requested}. "
                f"Pass signal={available!r} to load them."
            )

    signals = []

    # ------------------------------------------------------------------
    # Sum spectrum (present in both raw and pre-processed files)
    # ------------------------------------------------------------------
    if produce_sum:
        sum_ds = f["FullSpectra/SumSpectrum"]
        mass_axis_ds = np.asarray(f["FullSpectra/MassAxis"])

        # TofDAQ mass calibrations may have an invalid prefix at low flight times
        # (negative values or a non-monotonically increasing region).
        # HyperSpy requires non-uniform axes to be strictly increasing, so advance
        # past any leading samples where the calibration has not yet become valid.
        first_valid = int(np.argmax(mass_axis_ds >= 0))
        while (
            first_valid < len(mass_axis_ds) - 1
            and mass_axis_ds[first_valid] >= mass_axis_ds[first_valid + 1]
        ):
            first_valid += 1
        mass_axis_ds = mass_axis_ds[first_valid:]

        sum_axes = _build_axes_raw_sum(mass_axis_ds)
        sum_meta = dict(metadata)
        sum_meta["General"] = dict(metadata["General"])
        sum_meta["General"]["title"] = stem + " (sum spectrum)"

        if lazy:
            import dask.array as da

            sum_data = da.from_array(
                sum_ds, chunks=chunks if chunks is not None else -1
            )[first_valid:]
        else:
            sum_data = np.asarray(sum_ds)[first_valid:]

        signals.append(
            {
                "data": sum_data,
                "axes": sum_axes,
                "metadata": sum_meta,
                "original_metadata": original_metadata,
            }
        )

    # ------------------------------------------------------------------
    # 4D peak-integrated array
    #   Pre-processed files: read PeakData/PeakData directly
    #   Raw files: reconstruct from EventList
    # ------------------------------------------------------------------
    if produce_peak:
        if has_peak_data:
            peak_ds = f["PeakData/PeakData"]
            peak_table = np.asarray(f["PeakData/PeakTable"])
        else:
            peak_table = np.asarray(f["PeakData/PeakTable"])

        peak_masses = peak_table["mass"].astype(float)
        sort_idx = np.argsort(peak_masses)
        already_sorted = np.array_equal(sort_idx, np.arange(len(sort_idx)))
        peak_masses = peak_masses[sort_idx]
        peak_axes = _build_axes_preprocessed(
            nwrites, nsegs, nx, peak_masses, pixel_size_um
        )

        peak_meta = dict(metadata)
        peak_meta["General"] = dict(metadata["General"])
        peak_meta["General"]["title"] = stem + " (peak data)"
        peak_meta["Signal"] = {**metadata.get("Signal", {}), "signal_type": "FIB-SIMS"}

        if has_peak_data:
            if lazy:
                import dask.array as da

                # One chunk per depth slice by default: avoids tiny native HDF5
                # chunks creating huge dask task graphs, and matches the natural
                # access pattern.
                peak_data = da.from_array(
                    peak_ds,
                    chunks=chunks if chunks is not None else (1, nsegs, nx, -1),
                )
                if not already_sorted:
                    peak_data = peak_data[:, :, :, sort_idx]
            else:
                nbytes = peak_ds.size * peak_ds.dtype.itemsize
                if nbytes > 1e9:
                    _logger.warning(
                        f"Loading {nbytes / 1e9:.1f} GB of PeakData into memory "
                        f"(lazy=True is recommended for large files)."
                    )
                if already_sorted:
                    peak_data = np.asarray(peak_ds)
                else:
                    # Apply sort_idx in write-batches so each batch fits in
                    # cache.  Avoids cache-thrashing column permutation across
                    # the full array (which is O(n_cycle × dataset_size)).
                    _BATCH = 64
                    nw_ds = peak_ds.shape[0]
                    peak_data = np.empty(peak_ds.shape, dtype=peak_ds.dtype)
                    for _w0 in range(0, nw_ds, _BATCH):
                        _w1 = min(_w0 + _BATCH, nw_ds)
                        peak_data[_w0:_w1] = np.asarray(peak_ds[_w0:_w1])[
                            :, :, :, sort_idx
                        ]
        else:
            # Raw file: always eager (EventList is variable-length).
            # Pass the active peak_table from metadata so users can reintegrate
            # with modified windows by reloading with signal="peak_data".
            active_peak_table = peak_meta.get("Signal", {}).get("peak_table")
            peak_data = _compute_peak_data_from_file(f, peak_table=active_peak_table)
            if not already_sorted:
                try:
                    from tqdm.auto import tqdm as _tqdm_sort
                except ImportError:
                    _tqdm_sort = None
                _BATCH = 64
                nw_ev = peak_data.shape[0]
                sorted_array = np.empty(peak_data.shape, dtype=peak_data.dtype)
                pbar = (
                    _tqdm_sort(total=nw_ev, desc="Sorting m/z axis", unit=" slices")
                    if _tqdm_sort is not None
                    else None
                )
                for _w0 in range(0, nw_ev, _BATCH):
                    _w1 = min(_w0 + _BATCH, nw_ev)
                    sorted_array[_w0:_w1] = peak_data[_w0:_w1][:, :, :, sort_idx]
                    if pbar is not None:
                        pbar.update(_w1 - _w0)
                if pbar is not None:
                    pbar.close()
                peak_data = sorted_array

        signals.append(
            {
                "data": peak_data,
                "axes": peak_axes,
                "metadata": peak_meta,
                "original_metadata": original_metadata,
            }
        )

    # ------------------------------------------------------------------
    # EventList: ragged TDC timestamps (raw files only)
    # ------------------------------------------------------------------
    if produce_evl:
        el_ds = f["FullSpectra/EventList"]
        if lazy:
            # Lazy path: wrap the open h5py VL dataset directly in dask.
            # Each chunk is one depth slice; elements are fetched from the
            # file on .compute().  The file must remain open (see
            # has_lazy_array below).
            import dask.array as da

            el_data = da.from_array(
                el_ds, chunks=chunks if chunks is not None else (1, nsegs, nx)
            )
        else:
            # Eager path: load all depth slices into an object array now.
            _logger.warning(
                f"Loading EventList ({nwrites} depth slices × {nsegs} × {nx} pixels). "
                f"Pass lazy=True to defer reading to .compute() calls."
            )
            try:
                from tqdm.auto import tqdm as _tqdm_el
            except ImportError:
                _tqdm_el = None
            el_data = np.empty((nwrites, nsegs, nx), dtype=object)
            pbar = (
                _tqdm_el(total=nwrites, desc="Loading EventList", unit=" slices")
                if _tqdm_el is not None
                else None
            )
            for w in range(nwrites):
                el_data[w] = el_ds[w]
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
        el_axes = _build_axes_event_list(nwrites, nsegs, nx, pixel_size_um)
        el_meta = dict(metadata)
        el_meta["General"] = dict(metadata["General"])
        el_meta["General"]["title"] = stem + " (event list)"
        el_meta["Signal"] = {"signal_type": ""}
        sig_dict = {
            "data": el_data,
            "axes": el_axes,
            "metadata": el_meta,
            "original_metadata": original_metadata,
            # Mark as ragged so HyperSpy represents it as (depth, y, x | ragged)
            # and raises its standard RuntimeError when .plot() is called.
            "post_process": [_mark_event_list_ragged],
        }
        if not lazy:
            # HyperSpy's lazy data wrapping hangs on object-dtype numpy arrays
            # (dask cannot auto-chunk object dtype).  Since the data is already
            # in memory, force a non-lazy signal regardless of the caller's flag.
            sig_dict["attributes"] = {"_lazy": False}
        signals.append(sig_dict)

    # ------------------------------------------------------------------
    # FIB SE image stack  (FIBImages/<slice_name>, shape: fib_ny × fib_nx)
    # ------------------------------------------------------------------
    if produce_fib:
        fib_grp = f["FIBImages"]
        all_names = sorted(fib_grp.keys())
        # Each entry is a subgroup with a "Data" dataset: FIBImages/Image000N/Data
        # Images are usually the same size, but truncated acquisitions can produce
        # one or more images with a different shape (typically the last image).
        # Keep only images that match the most common shape.
        shape_counts: dict = {}
        for name in all_names:
            s = fib_grp[name]["Data"].shape
            shape_counts[s] = shape_counts.get(s, 0) + 1
        dominant_shape = max(shape_counts, key=lambda s: (shape_counts[s], s))
        slice_names = [
            n for n in all_names if fib_grp[n]["Data"].shape == dominant_shape
        ]
        skipped = [
            (n, fib_grp[n]["Data"].shape)
            for n in all_names
            if fib_grp[n]["Data"].shape != dominant_shape
        ]
        if skipped:
            details = ", ".join(f"{name} {shape}" for name, shape in skipped)
            _logger.warning(
                f"FIBImages: {len(skipped)} image(s) skipped — shape differs from "
                f"dominant {dominant_shape}: {details}"
            )
        n_fib = len(slice_names)
        fib_ny, fib_nx = dominant_shape
        fib_pixel_size_um = _parse_pixel_size(f, fib_nx)
        fib_axes = _build_axes_fib_images(n_fib, fib_ny, fib_nx, fib_pixel_size_um)
        fib_meta = dict(metadata)
        fib_meta["General"] = dict(metadata["General"])
        fib_meta["General"]["title"] = stem + " (FIB SE images)"
        fib_meta["Signal"] = {"signal_type": ""}

        if lazy:
            import dask.array as da

            fib_data = da.stack(
                [
                    da.from_array(
                        fib_grp[name]["Data"],
                        chunks=chunks if chunks is not None else -1,
                    )
                    for name in slice_names
                ]
            )
        else:
            fib_data = np.stack(
                [np.asarray(fib_grp[name]["Data"]) for name in slice_names]
            )

        signals.append(
            {
                "data": fib_data,
                "axes": fib_axes,
                "metadata": fib_meta,
                "original_metadata": original_metadata,
            }
        )

    # Keep the file open if any lazy dask arrays still reference it
    has_lazy_array = lazy and (
        produce_sum or (produce_peak and has_peak_data) or produce_evl or produce_fib
    )
    if not has_lazy_array:
        f.close()

    return signals


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, _CHUNKS_READ_DOC, RETURNS_DOC)
