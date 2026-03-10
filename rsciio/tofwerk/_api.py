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
detector).  Two file states are handled:

* **Opened** files (``PeakData/PeakData`` present) — returns a 4-D
  ``(depth, y, x, m/z)`` peak-integrated signal.
* **Raw** files (``FullSpectra/EventList`` only) — returns the sum spectrum
  and a 2-D TIC map.
"""

import logging
import re
from datetime import datetime
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)


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
    """Return True if Tofwerk software has processed the file (opened mode)."""
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

    Source: ``FIBParams.ViewField`` (in m, converted to µm) divided by *nx*.

    Note: ``ViewField`` in TofDAQ reflects the FIB deflector range set by the
    operator at the time of acquisition.  It may represent the full-range
    maximum rather than the actual scan area, in which case the returned value
    will be an overestimate.  No higher-accuracy spatial calibration is stored
    in the raw HDF5 file.

    Note: the ``[TOFParameter]`` section of ``Configuration File Contents``
    contains ``Ch1FullScale``–``Ch4FullScale`` values which are ADC input
    voltage ranges (V), not spatial dimensions, and must not be used for pixel
    size calculation.
    """
    try:
        view_field_m = float(np.asarray(file["FIBParams"].attrs["ViewField"]).flat[0])
        return (view_field_m * 1e6) / nx
    except Exception:
        _logger.warning("Could not determine pixel size; defaulting to 1 µm/pixel.")
        return 1.0


# ---------------------------------------------------------------------------
# Axes builders
# ---------------------------------------------------------------------------


def _build_axes_opened(nwrites, nsegs, nx, peak_masses, pixel_size_um):
    """Return the list of 4 axis dicts for an opened (PeakData) file."""
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
            "offset": 0,
            "scale": pixel_size_um,
            "units": "µm",
            "navigate": True,
        },
        {
            "name": "x",
            "size": nx,
            "offset": 0,
            "scale": pixel_size_um,
            "units": "µm",
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


def _build_axes_raw_tic(nsegs, nx, pixel_size_um):
    """Return 2 axis dicts for the raw TIC map."""
    return [
        {
            "name": "y",
            "size": nsegs,
            "offset": 0,
            "scale": pixel_size_um,
            "units": "µm",
            "navigate": False,
        },
        {
            "name": "x",
            "size": nx,
            "offset": 0,
            "scale": pixel_size_um,
            "units": "µm",
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
            "view_field_m": float(np.asarray(fibparams_attrs["ViewField"]).flat[0]),
            "pixel_size_um": pixel_size_um,
        }
    except Exception:
        # FIBParams group absent in non-FIB-SIMS TofDAQ files; FIB metadata omitted
        _logger.warning("FIBParams not found; FIB metadata will not be populated.")

    # Mass range: use peak table for opened files, raw spectrum axis for raw files
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
        "file_type": "opened" if has_peak_data else "raw",
    }
    if chamber_pressure is not None:
        fib_sims_meta["chamber_pressure_Pa"] = chamber_pressure

    metadata = {
        "General": {
            "title": stem,
            "original_filename": str(filename),
        },
        "Signal": {
            "signal_type": "FIB-SIMS",
            "binned": True,
        },
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

    # Capture FIBImages as a list of image names (not data — avoid embedding large arrays)
    if "FIBImages" in file:
        orig["FIBImages"] = list(file["FIBImages"].keys())

    # Arrays too large for metadata
    try:
        orig["SaturationWarning"] = np.asarray(file["FullSpectra/SaturationWarning"])
    except Exception:
        # SaturationWarning dataset is optional; omit if absent
        _logger.warning(
            "FullSpectra/SaturationWarning not found; omitting from metadata."
        )

    return orig


# ---------------------------------------------------------------------------
# TIC map computation for raw files
# ---------------------------------------------------------------------------


def _compute_tic_map(file, nwrites, nsegs, nx):
    """
    Compute the TIC (total ion count) map from the variable-length EventList.

    Returns a (nsegs, nx) int32 array.
    """
    el = file["FullSpectra/EventList"]
    tic_map = np.zeros((nsegs, nx), dtype=np.int32)
    for w in range(nwrites):
        for s in range(nsegs):
            for x in range(nx):
                tic_map[s, x] += len(el[w, s, x])
    return tic_map


def _count_active_channels(file):
    """
    Return the number of active TDC recording channels from the INI config.

    TofDAQ records events from multiple ADC/TDC channels simultaneously
    (e.g. Ch1 and Ch3 in a typical fibTOF setup).  Each active channel
    contributes an independent copy of every ion event to the EventList,
    so the raw event count per pixel is multiplied by this factor.
    """
    ini = _decode(file.attrs.get("Configuration File Contents", b""))
    matches = re.findall(r"Ch\dRecord=(\d)", ini)
    return max(sum(int(m) for m in matches), 1)


def _compute_peak_data_from_eventlist(file):
    """
    Reconstruct the 4-D peak-integrated array from raw EventList data.

    This replicates the processing performed by the Tofwerk proprietary
    software when opening a raw file:

    1. For each pixel's variable-length EventList (raw TDC timestamps),
       convert timestamps to ADC sample indices by integer division with the
       TDC-to-ADC clock ratio (``SampleInterval / ClockPeriod``, typically 64).
    2. Look up the calibrated mass for each sample index via ``MassAxis``.
    3. Count events that fall within each peak's integration window as defined
       in ``PeakData/PeakTable``.
    4. Divide by ``NbrWaveforms × NActiveChannels`` — the number of times each
       physical ion is recorded in the EventList (once per ToF cycle per active
       recording channel).

    Parameters
    ----------
    file : h5py.File
        Open HDF5 file handle (raw, not yet opened by Tofwerk software).

    Returns
    -------
    numpy.ndarray
        Shape ``(nwrites, nsegs, nx, npeaks)``, dtype float32.
        Identical in structure to ``PeakData/PeakData`` in an opened file.
    """
    el = file["FullSpectra/EventList"]
    mass_axis = np.asarray(file["FullSpectra/MassAxis"])
    pt = np.asarray(file["PeakData/PeakTable"])

    nwrites, nsegs, nx = el.shape
    npeaks = len(pt)
    nbr_samples = int(np.asarray(file.attrs["NbrSamples"]).flat[0])

    # TDC-to-ADC sample index conversion factor.  If ClockPeriod is absent
    # (uncommon), assume events are already in ADC sample units (ratio = 1).
    try:
        clock_period = float(
            np.asarray(file["FullSpectra"].attrs["ClockPeriod"]).flat[0]
        )
        sample_interval = float(
            np.asarray(file["FullSpectra"].attrs["SampleInterval"]).flat[0]
        )
        clock_ratio = round(sample_interval / clock_period)
    except KeyError:
        clock_ratio = 1

    # Normalisation: each ion appears once per waveform per active channel
    nbr_waveforms = int(np.asarray(file.attrs.get("NbrWaveforms", 1)).flat[0])
    n_channels = _count_active_channels(file)
    normalization = nbr_waveforms * n_channels

    # Sort peak windows for O(log n) assignment via searchsorted
    peak_low = pt["lower integration limit"].astype(float)
    peak_high = pt["upper integration limit"].astype(float)
    sort_idx = np.argsort(peak_low)
    sorted_low = peak_low[sort_idx]
    sorted_high = peak_high[sort_idx]

    result = np.zeros((nwrites, nsegs, nx, npeaks), dtype=np.float32)

    for w in range(nwrites):
        for s in range(nsegs):
            for x in range(nx):
                events = el[w, s, x]
                if len(events) == 0:
                    continue
                processed = events // clock_ratio
                valid = (processed >= 0) & (processed < nbr_samples)
                if not valid.any():
                    continue
                event_masses = mass_axis[processed[valid]]
                bin_idx = np.searchsorted(sorted_low, event_masses, side="right") - 1
                in_peak = (
                    (bin_idx >= 0)
                    & (bin_idx < npeaks)
                    & (event_masses <= sorted_high[bin_idx])
                )
                counts = np.bincount(bin_idx[in_peak], minlength=npeaks)
                result[w, s, x, sort_idx] = counts

    result /= normalization
    return result


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
# Public API
# ---------------------------------------------------------------------------


def file_reader(filename, lazy=False, compute_peak_data=False):
    """
    Read a Tofwerk TofDAQ HDF5 file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the ``.h5`` file.
    lazy : bool, optional
        If True, data arrays are returned as :class:`dask.array.Array` objects
        and the file handle is kept open.  Default is False.
    compute_peak_data : bool, optional
        If True and the file is a raw (unopened) file that contains a
        ``PeakData/PeakTable`` dataset, reconstruct the 4-D peak-integrated
        array from ``FullSpectra/EventList`` by applying the mass calibration
        and integrating within each peak's defined window.  The result is
        identical in structure to the ``PeakData/PeakData`` dataset written by
        the Tofwerk proprietary software.  This option has no effect on opened
        files (which already contain ``PeakData/PeakData``).  Default is False.

    Returns
    -------
    list of dict
        Each element is a signal dictionary with keys ``data``, ``axes``,
        ``metadata``, and ``original_metadata``.

        * **Opened files** (``PeakData/PeakData`` present): three signals —
          a 1-D sum spectrum, a 2-D TIC map, and a 4-D peak array
          ``(depth, y, x, m/z)``.
        * **Raw files**: two signals — a 1-D sum spectrum and a 2-D TIC map.
          If ``compute_peak_data=True`` a third 4-D signal is also returned.

    Raises
    ------
    IOError
        If the file is not a Tofwerk TofDAQ HDF5 file.
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

    has_peak_data = _has_peak_data(f)
    metadata = _build_metadata(f, filename, has_peak_data)
    original_metadata = _build_original_metadata(f)

    stem = metadata["General"]["title"]
    signals = []

    # ------------------------------------------------------------------
    # Determine spatial dimensions
    # ------------------------------------------------------------------
    nwrites = int(np.asarray(f.attrs.get("NbrWrites", 1)).flat[0])
    nsegs = int(np.asarray(f.attrs.get("NbrSegments", 1)).flat[0])
    if has_peak_data:
        nx = f["PeakData/PeakData"].shape[2]
    else:
        try:
            nx = f["FullSpectra/EventList"].shape[2]
        except Exception:
            nx = nsegs
    pixel_size_um = _parse_pixel_size(f, nx)

    # ------------------------------------------------------------------
    # Signal 0: sum spectrum (present in both raw and opened files)
    # ------------------------------------------------------------------
    sum_ds = f["FullSpectra/SumSpectrum"]
    mass_axis_ds = np.asarray(f["FullSpectra/MassAxis"])

    # Real TofDAQ files may have a negative-mass prefix at the start of the
    # mass axis (low ToF flight times where the calibration is not valid).
    # HyperSpy requires non-uniform axes to be strictly increasing, so trim
    # any leading samples where the mass is negative.
    first_valid = int(np.argmax(mass_axis_ds >= 0))
    mass_axis_ds = mass_axis_ds[first_valid:]

    sum_axes = _build_axes_raw_sum(mass_axis_ds)

    sum_meta = dict(metadata)
    sum_meta["General"] = dict(metadata["General"])
    sum_meta["General"]["title"] = stem + " (sum spectrum)"

    if lazy:
        import dask.array as da

        sum_data = da.from_array(sum_ds, chunks=-1)[first_valid:]
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
    # Signal 1: TIC map
    #   Raw files:    computed from EventList (always eager)
    #   Opened files: sum PeakData over depth and mass axes
    # ------------------------------------------------------------------
    tic_axes = _build_axes_raw_tic(nsegs, nx, pixel_size_um)
    tic_meta = dict(metadata)
    tic_meta["General"] = dict(metadata["General"])
    tic_meta["General"]["title"] = stem + " (TIC map)"

    if has_peak_data:
        if lazy:
            import dask.array as da

            peak_ds_lazy = da.from_array(f["PeakData/PeakData"], chunks="auto")
            tic_map = peak_ds_lazy.sum(axis=(0, 3)).compute()
        else:
            tic_map = np.asarray(f["PeakData/PeakData"]).sum(axis=(0, 3))
        tic_map = tic_map.astype(np.float32)
    else:
        tic_map = _compute_tic_map(f, nwrites, nsegs, nx)

    signals.append(
        {
            "data": tic_map,
            "axes": tic_axes,
            "metadata": tic_meta,
            "original_metadata": original_metadata,
        }
    )

    # ------------------------------------------------------------------
    # Signal 2: 4D peak-integrated array
    #   Opened files: read PeakData/PeakData directly
    #   Raw files with compute_peak_data=True: reconstruct from EventList
    # ------------------------------------------------------------------
    can_compute = (
        not has_peak_data
        and compute_peak_data
        and "PeakData/PeakTable" in f
        and "FullSpectra/EventList" in f
    )

    if has_peak_data or can_compute:
        if has_peak_data:
            peak_ds = f["PeakData/PeakData"]
            nwrites, nsegs, nx, npeaks = peak_ds.shape
            peak_table = np.asarray(f["PeakData/PeakTable"])
        else:
            peak_table = np.asarray(f["PeakData/PeakTable"])

        peak_masses = peak_table["mass"].astype(float)
        peak_axes = _build_axes_opened(nwrites, nsegs, nx, peak_masses, pixel_size_um)

        peak_meta = dict(metadata)
        peak_meta["General"] = dict(metadata["General"])
        peak_meta["General"]["title"] = stem + " (peak data)"

        if has_peak_data:
            if lazy:
                import dask.array as da

                peak_data = da.from_array(peak_ds, chunks="auto")
            else:
                peak_data = np.asarray(peak_ds)
        else:
            # compute_peak_data=True: always eager (EventList is variable-length)
            peak_data = _compute_peak_data_from_eventlist(f)

        signals.append(
            {
                "data": peak_data,
                "axes": peak_axes,
                "metadata": peak_meta,
                "original_metadata": original_metadata,
            }
        )

    if not lazy:
        f.close()

    return signals
