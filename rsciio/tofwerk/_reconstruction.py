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
EventList reconstruction for Tofwerk TofDAQ HDF5 files.

This module contains the peak-integration reconstruction functions that convert
raw TDC EventList data into a 4-D ``(depth, y, x, m/z)`` peak-counts array,
replicating the processing performed by the Tofwerk proprietary software.

Public API
----------
compute_peak_data_from_eventlist
    Reconstruct the 4-D peak-integrated array from a raw EventList.
"""

import logging
import re
from typing import Any

import numpy as np

from rsciio.utils._decorator import jit_ifnumba

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PeakTable helpers
# ---------------------------------------------------------------------------


def _peak_table_from_list(
    peak_table_list: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _count_active_channels(ini: str) -> int:
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


def _flatten_event_row(row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack all events from one row of pixels into a flat contiguous buffer.

    Parameters
    ----------
    row : numpy.ndarray
        1-D object array of shape ``(nx,)`` for one (depth, y) row, as
        returned by ``el[w][s]`` or ``el[w, s, :]``.

    Returns
    -------
    flat : numpy.ndarray, dtype int64
        All events concatenated in pixel order (x varies).
    offsets : numpy.ndarray, shape (nx + 1,), dtype int64
        ``flat[offsets[x]:offsets[x+1]]`` are the events for pixel ``x``.
    """
    _nx = len(row)
    lengths = np.array([len(row[x]) for x in range(_nx)], dtype=np.int64)
    offsets = np.empty(_nx + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    flat = (
        np.concatenate(list(row)).astype(np.int64)
        if offsets[-1] > 0
        else np.empty(0, dtype=np.int64)
    )
    return flat, offsets


@jit_ifnumba(cache=True)
def _accumulate_peak_counts(
    flat: np.ndarray,
    offsets: np.ndarray,
    nx: int,
    clock_ratio: int,
    nbr_samples: int,
    mass_axis: np.ndarray,
    sorted_low: np.ndarray,
    sorted_high: np.ndarray,
    npeaks: int,
    result_sx: np.ndarray,
) -> None:  # pragma: no cover
    """
    Accumulate per-pixel peak counts for one row into *result_sx*.

    Parameters
    ----------
    flat : int64 array
        Flat event buffer from :func:`_flatten_event_row`.
    offsets : int64 array, shape (nx + 1,)
        Pixel start/end indices into *flat*.
    nx : int
        Number of x pixels in the row.
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
    event_list: np.ndarray,
    mass_axis: np.ndarray,
    nbr_samples: int,
    clock_ratio: int,
    normalization: int,
    peak_table: list[dict[str, Any]],
    show_progressbar: bool = True,
) -> np.ndarray:
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

    .. note::
        If ``numba`` is installed, each depth slice is read in a single h5py
        call (instead of one call per row), then each row is packed into a flat
        int64 buffer and processed by a JIT-compiled kernel.  Otherwise falls
        back to a vectorised NumPy path that allocates per-pixel arrays but is
        still significantly faster than a pure-Python triple loop.

    Parameters
    ----------
    event_list : array-like
        Ragged object array of shape ``(nwrites, nsegs, nx)`` of uint16 TDC
        timestamps, one variable-length array per pixel.  Accepts h5py
        variable-length datasets or numpy object arrays (as loaded by
        ``signal="event_list"``).
    mass_axis : numpy.ndarray
        1-D array of shape ``(nbr_samples,)`` with calibrated mass (Da) for
        each ADC sample index, from ``FullSpectra/MassAxis``.
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
    show_progressbar : bool, default=True
        Whether to show tqdm progress bars during reconstruction.

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

    _logger.info(
        f"Reconstructing peak data from EventList "
        f"({nwrites} depth slices × {nsegs} × {nx} pixels × {npeaks} peaks)."
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
        _tqdm = None  # type: ignore[assignment]

    def _make_pbar(n, desc="Reconstructing peak data"):
        if _tqdm is not None and show_progressbar:
            return _tqdm(total=n, desc=desc, unit=" slices")
        return None

    if _use_numba:
        # numba path: read one depth slice per h5py call, then process one row
        # at a time -- flatten the row into a contiguous int64 buffer and
        # dispatch the JIT-compiled kernel per row.
        result_sx = np.zeros((nx, npeaks), dtype=np.float32)
        pbar = _make_pbar(nwrites)
        for w in range(nwrites):
            slice_data = el[w]  # one h5py read covers all nsegs rows
            for s in range(nsegs):
                flat, offsets = _flatten_event_row(slice_data[s])
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
        # NumPy fallback path: read one depth slice at a time, then iterate
        # over rows and pixels explicitly, converting each pixel's
        # variable-length event array to masses and binning with searchsorted
        # + bincount.  Slower than the numba path but avoids any compiled
        # dependency.
        pbar = _make_pbar(nwrites)
        for w in range(nwrites):
            slice_data = el[w]  # one read covers all nsegs rows for this depth slice
            for s in range(nsegs):
                row = slice_data[s]
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

    result /= normalization
    return result
