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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

import logging
from pathlib import Path

import dask
import dask.array as da
import numpy as np

from rsciio._docstrings import FILENAME_DOC, RETURNS_DOC

from ._jkf import read_point_index
from ._records import HEADER_SIZE, read_spectra, scan_records

_logger = logging.getLogger(__name__)

# Channel count of the files the energy axis was calibrated on (RODARE 4896).
CALIBRATED_N_CHANNELS = 2000
# E(eV) = -10.04 + 10.0163 * channel, a least-squares fit of nine K lines.
ENERGY_OFFSET_KEV = -0.01004
ENERGY_SCALE_KEV = 0.0100163

HEADER_FIELDS = ("x", "y", "width", "height", "bse_mode", "bse_mode_fraction")
# Spectra copied per step when not loading lazily (65 MB of temporary data).
_EAGER_BLOCK = 1 << 14


def _row_chunks(chunks, n_spectra, n_channels):
    """Spectra per chunk. Chunks always hold complete spectra."""
    if isinstance(chunks, (tuple, list)):
        chunks = chunks[0] if chunks else "auto"
    return da.core.normalize_chunks(
        (chunks, -1), shape=(n_spectra, n_channels), dtype=np.uint16
    )[0]


def _lazy_data(path, offsets, payload_bytes, n_channels, chunks):
    blocks, start = [], 0
    for rows in _row_chunks(chunks, len(offsets), n_channels):
        stop = start + rows
        block = dask.delayed(read_spectra)(
            str(path), offsets[start:stop], payload_bytes[start:stop], n_channels
        )
        blocks.append(da.from_delayed(block, shape=(rows, n_channels), dtype=np.uint16))
        start = stop
    return da.concatenate(blocks, axis=0)


def _eager_data(path, offsets, payload_bytes, n_channels):
    data = np.empty((len(offsets), n_channels), np.uint16)
    for start in range(0, len(offsets), _EAGER_BLOCK):
        stop = start + _EAGER_BLOCK
        data[start:stop] = read_spectra(
            path, offsets[start:stop], payload_bytes[start:stop], n_channels
        )
    return data


def _find_jkf(path):
    """Sibling X-Ray stage index: ``name.xpoints`` -> ``name.jkf``."""
    for suffix in (".jkf", ".JKF"):
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _jkf_metadata(jkf_path, offsets, payload_bytes, headers):
    """Frame, particle, grain and stage position of every ``.xpoints`` record.

    The ``.jkf`` index is matched to the records by byte offset, because it may
    skip records. Records without an index entry get -1 (NaN for the positions).
    """
    index = read_point_index(jkf_path)
    if not len(index):
        raise ValueError(f"No entry of {jkf_path.name} matches a record of the file.")
    wanted = offsets.astype(np.uint64)
    at = np.minimum(np.searchsorted(index["offset"], wanted), len(index) - 1)
    found = index["offset"][at] == wanted
    if not found.any():
        raise ValueError(f"No entry of {jkf_path.name} matches a record of the file.")
    matched, rows = np.flatnonzero(found), index[at[found]]
    expected_bytes = HEADER_SIZE + payload_bytes[matched].astype(np.int64)
    consistent = (rows["record_bytes"].astype(np.int64) == expected_bytes) & np.all(
        [rows[f] == headers[f][matched] for f in ("x", "y", "width", "height")], axis=0
    )
    if not consistent.all():
        raise ValueError(
            f"{jkf_path.name} does not belong to this file: "
            f"{int((~consistent).sum())} of {len(rows)} matching entries differ "
            "from the record headers."
        )
    if not found.all():
        _logger.warning(
            f"{int((~found).sum())} of {len(offsets)} spectra have no entry in "
            f"{jkf_path.name}. Their frame, particle and grain are -1 and "
            "their stage position is NaN."
        )
    out = {
        "frame": np.full(len(offsets), -1, np.int32),
        "particle": np.full(len(offsets), -1, np.int32),
        "grain": np.full(len(offsets), -1, np.int16),
        "stage_x_um": np.full(len(offsets), np.nan),
        "stage_y_um": np.full(len(offsets), np.nan),
    }
    for name, values in out.items():
        values[matched] = rows[name]
    return {"filename": jkf_path.name, **out}


def _optional_jkf_metadata(jkf, path, offsets, payload_bytes, headers):
    if jkf is False or jkf is None:
        return None
    if jkf is not True:
        return _jkf_metadata(Path(jkf), offsets, payload_bytes, headers)
    jkf_path = _find_jkf(path)
    if jkf_path is None:
        _logger.info(f"No {path.with_suffix('.jkf').name} next to the file.")
        return None
    try:
        return _jkf_metadata(jkf_path, offsets, payload_bytes, headers)
    except ValueError as err:
        _logger.warning(f"Ignoring {jkf_path.name}: {err}")
        return None


def _axes(n_spectra, n_channels):
    if n_channels == CALIBRATED_N_CHANNELS:
        signal_axis = {
            "name": "Energy",
            "size": n_channels,
            "index_in_array": 1,
            "scale": ENERGY_SCALE_KEV,
            "offset": ENERGY_OFFSET_KEV,
            "units": "keV",
            "navigate": False,
        }
    else:
        _logger.warning(
            f"The energy calibration is only known for {CALIBRATED_N_CHANNELS} "
            f"channels, the file has {n_channels}. The channel axis is not calibrated."
        )
        signal_axis = {
            "name": "Channel",
            "size": n_channels,
            "index_in_array": 1,
            "scale": 1.0,
            "offset": 0.0,
            "units": "",
            "navigate": False,
        }
    navigation_axis = {
        "name": "Spectrum",
        "size": n_spectra,
        "index_in_array": 0,
        "scale": 1.0,
        "offset": 0.0,
        "units": "",
        "navigate": True,
    }
    return [navigation_axis, signal_axis]


def file_reader(filename, lazy=False, chunks="auto", jkf=True):
    """
    Read the raw X-ray spectra of a FEI/Thermo MLA ``.xpoints`` file.

    The signal has one navigation axis (the spectrum index, in acquisition order) and
    one signal axis with the 2000 channels of the spectra. The values are always
    ``uint16``, even for the records that MLA stored as ``uint8``.

    Reading scans the headers of all records, because the position of a record can
    only be known from the sizes of the records before it. This touches the whole
    file, which takes a few seconds for a full-size file (about 1.2 million spectra,
    2.5 GB). Use ``lazy=True`` for those, loading them completely needs about 5 GB.

    Parameters
    ----------
    %s
    lazy : bool, default=False
        Whether to load the spectra lazily with ``dask``. The headers of the records
        are always read when the file is opened, the counts when the data is computed.
        No file stays open.
    chunks : int, str or tuple of int, default="auto"
        Number of spectra per chunk when reading lazily. ``"auto"`` lets
        :func:`dask.array.core.normalize_chunks` choose. When a tuple is given
        its first element is used, because a chunk always holds complete spectra.
    jkf : bool, str or pathlib.Path, default=True
        Whether to add the frame, particle, grain and stage position of each spectrum
        from the X-Ray stage ``.jkf`` file to ``original_metadata``. If ``True``, the
        file next to the ``.xpoints`` file with the same name and the extension
        ``.jkf`` is used when it exists, and ignored with a warning if it does not
        belong to the ``.xpoints`` file. A path selects the file to use and raises
        an error when it cannot be used. ``False`` does not read a ``.jkf`` file.

    %s

    Notes
    -----
    The per-spectrum fields of the record headers are in
    ``original_metadata["spectrum_headers"]`` as arrays with one value per
    spectrum: ``x`` and ``y`` (pixel position in the frame), ``width`` and
    ``height`` (side in pixels of the area the spectrum stands for),
    ``bse_mode`` and ``bse_mode_fraction`` (most frequent backscattered-electron
    grey level of the spectrum's grain and the fraction of the grain at that
    level).

    With a ``.jkf`` file, ``original_metadata["jkf"]`` holds the arrays ``frame``,
    ``particle`` and ``grain`` (-1 where unknown) and ``stage_x_um``, ``stage_y_um``
    (stage position in um, NaN where unknown).

    The format was decoded from files written by MLA Suite 3.1.4.686 and the energy
    axis (offset -10.04 eV, gain 10.0163 eV per channel) was fitted on one frame of
    one sample; other versions or detector settings may differ.

    Examples
    --------
    >>> from rsciio.mla_xpoints import file_reader
    >>> s = file_reader("3_GXMAP_spx.xpoints", lazy=True)
    """
    path = Path(filename)
    offsets, payload_bytes, n_channels, headers = scan_records(path)
    n_spectra = len(offsets)

    if lazy:
        data = _lazy_data(path, offsets, payload_bytes, n_channels, chunks)
    else:
        data = _eager_data(path, offsets, payload_bytes, n_channels)

    original_metadata = {
        "spectrum_headers": {
            name: np.ascontiguousarray(headers[name]) for name in HEADER_FIELDS
        }
    }
    jkf_metadata = _optional_jkf_metadata(jkf, path, offsets, payload_bytes, headers)
    if jkf_metadata is not None:
        original_metadata["jkf"] = jkf_metadata

    metadata = {
        "General": {"original_filename": path.name, "title": path.stem},
        "Signal": {"signal_type": "EDS_SEM", "quantity": "X-rays (Counts)"},
    }
    return [
        {
            "data": data,
            "axes": _axes(n_spectra, n_channels),
            "metadata": metadata,
            "original_metadata": original_metadata,
        }
    ]


file_reader.__doc__ %= (FILENAME_DOC, RETURNS_DOC)
