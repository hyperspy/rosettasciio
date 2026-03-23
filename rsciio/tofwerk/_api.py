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
                                  variable/ragged-length uint16 array of
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
  peak-integrated array.  If the EventList is still present,
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

from __future__ import annotations

import enum
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

if TYPE_CHECKING:
    import h5py
    from hyperspy.signals import BaseSignal  # type: ignore[import-unresolved]

from rsciio._docstrings import (
    CHUNKS_READ_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    RETURNS_DOC,
    SHOW_PROGRESSBAR_DOC,
)
from rsciio.tofwerk._reconstruction import (
    _count_active_channels,
    compute_peak_data_from_eventlist,
)

_logger = logging.getLogger(__name__)


_SignalT = TypeVar("_SignalT", bound="BaseSignal")

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


def _is_tofwerk_file(file: h5py.File) -> bool:
    """Return True if the HDF5 file was written by TofDAQ."""
    return (
        "FullSpectra" in file
        and "TimingData" in file
        and "AcquisitionLog" in file
        and "TofDAQ Version" in file.attrs
    )


def _is_fib_sims(file: h5py.File) -> bool:
    """Return True if this is a fibTOF FIB-SIMS file."""
    return _is_tofwerk_file(file) and "FIBImages" in file and "FIBParams" in file


def _has_peak_data(file: h5py.File) -> bool:
    """Return True if Tofwerk software has pre-processed the file."""
    return "PeakData/PeakData" in file


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_creation_time(file: h5py.File) -> tuple[str, str, str]:
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
        utcoffset = dt.utcoffset() if dt.tzinfo is not None else None
        if utcoffset is not None:
            total_seconds = int(utcoffset.total_seconds())
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


def _parse_pixel_size(file: h5py.File, nx: int) -> float | None:
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


def _build_axes_preprocessed(
    nwrites: int,
    nsegs: int,
    nx: int,
    peak_masses: np.ndarray,
    pixel_size_um: float | None,
) -> list[dict[str, Any]]:
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


def _build_axes_raw_sum(mass_axis: np.ndarray) -> list[dict[str, Any]]:
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


def _build_axes_event_list(
    nwrites: int,
    nsegs: int,
    nx: int,
    pixel_size_um: float | None,
) -> list[dict[str, Any]]:
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


def _build_axes_fib_images(
    n_images: int,
    fib_ny: int,
    fib_nx: int,
    fib_pixel_size_um: float | None,
) -> list[dict[str, Any]]:
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


def _build_metadata(
    file: h5py.File, filename: str, has_peak_data: bool
) -> dict[str, Any]:
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

    signal_meta: dict[str, Any] = {"signal_type": "FIB-SIMS"}
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


def _build_original_metadata(file: h5py.File) -> dict[str, Any]:
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


def _peak_table_to_list(pt: np.ndarray) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Private file-level EventList reconstruction helper
# ---------------------------------------------------------------------------


def _compute_peak_data_from_file(
    file: h5py.File,
    peak_table: list[dict[str, Any]] | None = None,
    depth_start: int = 0,
    depth_stop: int | None = None,
    show_progressbar: bool = True,
) -> np.ndarray:
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
    depth_start : int, optional
        First depth slice index (inclusive).  Default 0.
    depth_stop : int or None, optional
        Last depth slice index (exclusive).  Default None (all slices).

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

    el_ds = file["FullSpectra/EventList"]
    el_total = el_ds.shape[0]
    if depth_stop is None:
        depth_stop = el_total
    if depth_start == 0 and depth_stop == el_total:
        event_list = el_ds
    else:
        # Slice the variable-length (VL) HDF5 dataset into a numpy object array
        # for the reconstruction. This reads only the requested depth slices from disk.
        event_list = el_ds[depth_start:depth_stop]
    return compute_peak_data_from_eventlist(
        event_list,
        mass_axis,
        nbr_samples,
        clock_ratio,
        normalization,
        peak_table,
        show_progressbar=show_progressbar,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _decode(value: object) -> str:
    """Decode bytes to str; pass through strings unchanged."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_attr(value: object) -> str | int | float | list | object:
    """Make an HDF5 attribute JSON-serialisable."""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "U", "O"):
            return [_decode(v) for v in value.flat]
        return value.tolist()  # type: ignore[no-matching-overload]
    return value


# ---------------------------------------------------------------------------
# EventList post-processing
# ---------------------------------------------------------------------------


def _mark_event_list_ragged(signal: _SignalT) -> _SignalT:
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


def available_signals(filename: str | Path) -> list[str]:
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


def file_reader(
    filename: str | Path,
    lazy: bool = False,
    signal: str | list[str] = "sum_spectrum",
    chunks: tuple | int | dict | str = "auto",
    mz_range: tuple[float, float] | None = None,
    depth_range: tuple[int, int] | None = None,
    dtype: np.dtype | None = None,
    show_progressbar: bool = True,
    peak_data_batch_size: int = 1,
) -> list[dict[str, Any]]:
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
            timestamps.  Present in raw files; may also be present in
            pre-processed files.
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
    mz_range : tuple, optional
        Restrict the m/z axis of ``"peak_data"`` to peaks whose nominal
        mass falls within ``[mz_range[0], mz_range[1]]`` (inclusive, Da).
        For pre-processed files only the selected columns are retained after
        reading; for raw files only the selected peaks are reconstructed from
        the EventList, so the reconstruction cost scales with the number of
        selected peaks.  If ``None`` (default), all peaks are returned.
    depth_range : tuple, optional
        Restrict the depth axis to slices ``[depth_range[0], depth_range[1])``
        (0-indexed, exclusive upper bound).  Applies to ``"peak_data"``,
        ``"event_list"``, and ``"fib_images"``.  For pre-processed files the
        HDF5 dataset is sliced before loading, so only the requested depth
        slices are read from disk.  If ``None`` (default), all depth slices
        are returned.
    dtype : numpy.dtype, optional
        Cast the ``"peak_data"`` array to this dtype after loading.  Useful to
        reduce memory usage, e.g. ``dtype=np.float16`` or ``dtype=np.uint16``
        for low-count data.  If ``None`` (default), the on-disk dtype
        (``float32``) is preserved.
    peak_data_batch_size : int, optional
        Number of depth slices to read and permute per iteration when loading
        ``"peak_data"`` from a pre-processed file whose peaks are not already
        in ascending mass order.  Smaller batches keep the sort permutation
        working on arrays that fit in CPU cache, which is faster for large
        datasets.  Default is 1 (one slice at a time).  Has no effect when
        peaks are already sorted or when ``lazy=True``.
    %s

    %s

    Raises
    ------
    IOError
        If the file is not a Tofwerk TofDAQ HDF5 file.
    ValueError
        If ``signal`` contains an unrecognised value, or if ``mz_range`` or
        ``depth_range`` are out of bounds, or if an explicitly requested signal
        is not available in the file.
    NotImplementedError
        If ``lazy=True`` and ``signal="peak_data"`` on a raw file that requires
        EventList reconstruction.
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
    # depth_range / mz_range / dtype validation
    # ------------------------------------------------------------------
    if depth_range is not None:
        depth_start = int(depth_range[0])
        depth_stop = int(depth_range[1])
        if depth_start < 0 or depth_stop > nwrites or depth_start >= depth_stop:
            f.close()
            raise ValueError(
                f"depth_range={depth_range!r} is out of bounds for a file with "
                f"{nwrites} depth slices.  Valid range: [0, {nwrites})."
            )
    else:
        depth_start, depth_stop = 0, nwrites
    nwrites_loaded = depth_stop - depth_start

    if mz_range is not None:
        if len(mz_range) != 2 or mz_range[0] >= mz_range[1]:
            f.close()
            raise ValueError(
                f"mz_range={mz_range!r} must be a (min, max) tuple with min < max."
            )

    if dtype is not None:
        dtype = np.dtype(dtype)

    # ------------------------------------------------------------------
    # Availability checks
    # ------------------------------------------------------------------
    has_peak_table = "PeakData/PeakTable" in f
    can_peak = has_peak_data or (has_event_list and has_peak_table)
    can_evl = has_event_list
    can_fib = "FIBImages" in f and len(f["FIBImages"]) > 0

    if want_peak and not can_peak and not want_all:
        f.close()
        raise ValueError(
            "signal='peak_data' requested but not available "
            "(no PeakData/PeakData and no EventList+PeakTable)."
        )
    if want_evl and not can_evl and not want_all:
        f.close()
        raise ValueError(
            "signal='event_list' requested but FullSpectra/EventList is not present "
            "(FullSpectra/EventList group not found in this file)."
        )
    if want_fib and not can_fib and not want_all:
        f.close()
        raise ValueError(
            "signal='fib_images' requested but FIBImages group is absent or empty."
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

            sum_data = da.from_array(sum_ds, chunks=chunks)[first_valid:]
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
        # Sort peaks by mass so the m/z signal axis is monotonically increasing
        # (required by HyperSpy non-uniform axes).
        sort_idx = np.argsort(peak_masses)
        # Skip the reorder step on the data array when peaks are already sorted.
        already_sorted = np.array_equal(sort_idx, np.arange(len(sort_idx)))
        peak_masses = peak_masses[sort_idx]

        # mz_range: compute boolean mask on the sorted peak_masses array.
        # mz_sel holds the indices (into sorted order) of the selected peaks.
        if mz_range is not None:
            mz_mask = (peak_masses >= mz_range[0]) & (peak_masses <= mz_range[1])
            mz_sel = np.where(mz_mask)[0]
            if len(mz_sel) == 0:
                _logger.warning(
                    f"mz_range={mz_range!r} excludes all {len(peak_masses)} peaks "
                    "in PeakTable; skipping peak_data signal."
                )
                produce_peak = False
            else:
                peak_masses = peak_masses[mz_mask]
        else:
            mz_sel = None

    if produce_peak:
        peak_axes = _build_axes_preprocessed(
            nwrites_loaded, nsegs, nx, peak_masses, pixel_size_um
        )

        peak_meta = dict(metadata)
        peak_meta["General"] = dict(metadata["General"])
        peak_meta["General"]["title"] = stem + " (peak data)"
        peak_meta["Signal"] = {**metadata.get("Signal", {}), "signal_type": "FIB-SIMS"}

        if has_peak_data:
            if lazy:
                import dask.array as da

                peak_data = da.from_array(peak_ds, chunks=chunks)[
                    depth_start:depth_stop
                ]
                if not already_sorted:
                    peak_data = peak_data[:, :, :, sort_idx]
                if mz_sel is not None:
                    peak_data = peak_data[:, :, :, mz_sel]
                if dtype is not None:
                    peak_data = peak_data.astype(dtype)
            else:
                if mz_sel is not None:
                    # When mz_range is active, read only the selected columns
                    # directly from HDF5 so we never materialise the full mass
                    # axis in memory.  final_idx maps each selected sorted-mass
                    # position back to its original column index in the file.
                    # h5py requires fancy indices to be in strictly ascending
                    # order, so we sort them, read, then undo the reorder.
                    final_idx = sort_idx[mz_sel]
                    _asc_order = np.argsort(final_idx)
                    _undo = np.argsort(_asc_order)
                    peak_data = np.asarray(
                        peak_ds[
                            depth_start:depth_stop,
                            :,
                            :,
                            final_idx[_asc_order].tolist(),
                        ]
                    )[:, :, :, _undo]
                elif already_sorted:
                    peak_data = np.asarray(peak_ds[depth_start:depth_stop])
                else:
                    # Apply sort_idx in batches so each batch fits in CPU cache.
                    # Avoids cache-thrashing column permutation across the full
                    # array.  Batch size is tunable via peak_data_batch_size.
                    peak_data = np.empty(
                        (nwrites_loaded, nsegs, nx, peak_ds.shape[3]),
                        dtype=peak_ds.dtype,
                    )
                    for _w0 in range(0, nwrites_loaded, peak_data_batch_size):
                        _w1 = min(_w0 + peak_data_batch_size, nwrites_loaded)
                        peak_data[_w0:_w1] = np.asarray(
                            peak_ds[depth_start + _w0 : depth_start + _w1]
                        )[:, :, :, sort_idx]
                if dtype is not None:
                    peak_data = peak_data.astype(dtype)
        else:
            # Raw file: reconstruction from EventList is inherently eager because
            # compute_peak_data_from_eventlist materialises the full array.
            # Silently falling through to eager loading when the caller passed
            # lazy=True is misleading (the data lands in RAM before the signal
            # is returned), so we raise explicitly instead.
            if lazy:
                f.close()
                raise NotImplementedError(
                    "lazy=True is not supported when requesting signal='peak_data', "
                    "since it requires eagerly integrating the raw EventList data. "
                    "To compute the peak_data signal, either load the dataset eagerly "
                    "(lazy=False) or open the file with the Tofwerk software first "
                    "to produce PeakData/PeakData, which will enable lazy loading in "
                    "rsciio."
                )
            # Pass the active peak_table from metadata so users can reintegrate
            # with modified windows by reloading with signal="peak_data".
            active_peak_table = cast(
                list[dict[str, Any]], peak_meta["Signal"]["peak_table"]
            )

            # Pre-sort peak_table into ascending mass order (full or filtered).
            # Reconstruction emits one column per peak_table entry in input
            # order, so pre-sorting here avoids a separate post-hoc sort pass.
            sel = sort_idx[mz_sel] if mz_sel is not None else sort_idx
            active_peak_table = [active_peak_table[i] for i in sel]
            peak_data = _compute_peak_data_from_file(
                f,
                peak_table=active_peak_table,
                depth_start=depth_start,
                depth_stop=depth_stop,
                show_progressbar=show_progressbar,
            )

            if dtype is not None:
                peak_data = peak_data.astype(dtype)

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
            # Note: dask cannot auto-chunk object-dtype arrays (raises
            # NotImplementedError), so one depth slice per chunk is used
            # as the default for the EventList.
            import dask.array as da

            el_data = da.from_array(
                el_ds, chunks=(1, nsegs, nx) if chunks == "auto" else chunks
            )[depth_start:depth_stop]
        else:
            # Eager path: load requested depth slices into an object array now.
            _logger.warning(
                f"Loading EventList ({nwrites_loaded} depth slices × {nsegs} × {nx} pixels). "
                f"Pass lazy=True to defer reading to .compute() calls."
            )
            try:
                from tqdm.auto import tqdm as _tqdm_el
            except ImportError:
                _tqdm_el = None  # type: ignore[assignment]
            el_data = np.empty((nwrites_loaded, nsegs, nx), dtype=object)
            pbar = (
                _tqdm_el(total=nwrites_loaded, desc="Loading EventList", unit=" slices")
                if _tqdm_el is not None and show_progressbar
                else None
            )
            for i, w in enumerate(range(depth_start, depth_stop)):
                el_data[i] = el_ds[w]
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
        el_axes = _build_axes_event_list(nwrites_loaded, nsegs, nx, pixel_size_um)
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
        ][depth_start:depth_stop]
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
                    da.from_array(fib_grp[name]["Data"], chunks=chunks)
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


file_reader.__doc__ %= (  # type: ignore[operator]
    FILENAME_DOC,
    LAZY_DOC,
    CHUNKS_READ_DOC,
    SHOW_PROGRESSBAR_DOC,
    RETURNS_DOC,
)
