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

"""Record layer of the MLA ``.xpoints`` format.

A ``.xpoints`` file has no file header. It is a plain concatenation of records::

    record  = 80-byte header + payload
    payload = n_channels counts
              uint8          if payload_bytes == n_channels
              uint16 (LE)    if payload_bytes == 2 * n_channels

``payload_bytes`` (at byte 0x4C of the header) is the only switch between the two
payload types, so the offset of a record is only known once all earlier headers have
been read. Bytes of the header that are not listed in :data:`HEADER_DTYPE` are either
zero or uninitialised memory (struct padding) and must be ignored.
"""

import array
import struct

import numpy as np

HEADER_SIZE = 80
RECORD_TAG = 0x5A

# n_channels (u16 at 0x46), 4 bytes that are constant or padding, payload_bytes (u32 at 0x4C)
_SIZES = struct.Struct("<H4xI")
_SIZES_OFFSET = 0x46

HEADER_DTYPE = np.dtype(
    {
        "names": [
            "tag",
            "version_a",
            "version_b",
            "x",
            "y",
            "width",
            "height",
            "unknown_44",
            "bse_mode",
            "bse_mode_fraction",
            "n_channels",
            "payload_bytes",
        ],
        "formats": [
            "u1",
            "<u2",
            "<u2",
            "<u2",
            "<u2",
            "<u2",
            "<u2",
            "<u2",
            "<f4",
            "<f4",
            "<u2",
            "<u4",
        ],
        "offsets": [0, 2, 4, 12, 14, 40, 42, 44, 48, 52, 70, 76],
        "itemsize": HEADER_SIZE,
    }
)

# Bytes read at a time while walking the file.
_BLOCK = 1 << 26


def scan_records(path):
    """Walk the record chain of a ``.xpoints`` file and validate every header.

    The file is read in blocks, so the memory used does not depend on its size apart
    from the tables that are returned.

    Parameters
    ----------
    path : str or pathlib.Path
        Path of the ``.xpoints`` file.

    Returns
    -------
    offsets : numpy.ndarray
        ``int64`` byte offset of each record.
    payload_bytes : numpy.ndarray
        ``uint32`` payload size of each record in bytes.
    n_channels : int
        Number of channels, which is the same in all records.
    headers : numpy.ndarray
        Structured array (dtype :data:`HEADER_DTYPE`), one row per record.

    Raises
    ------
    ValueError
        If the file is empty or a record does not follow the layout above.
    """
    offsets = array.array("q")
    payload_bytes = array.array("I")
    raw_headers = bytearray()
    n_channels = None
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        if size == 0:
            raise ValueError(f"{path} is empty, it is not a MLA .xpoints file.")
        buf, base, off = b"", 0, 0
        while off < size:
            index = len(offsets)
            if off + HEADER_SIZE > size:
                raise ValueError(
                    f"Record {index} at byte {off}: truncated header "
                    f"({size - off} bytes left)."
                )
            if off - base + HEADER_SIZE > len(buf):
                # The header is not (completely) in the block: read the next one.
                fh.seek(off)
                buf, base = fh.read(_BLOCK), off
                if len(buf) < HEADER_SIZE:
                    raise ValueError(f"{path} changed while it was read.")
            rel = off - base
            nch, plen = _check_header(buf, rel, off, index, size, n_channels)
            n_channels = nch
            raw_headers += buf[rel : rel + HEADER_SIZE]
            offsets.append(off)
            payload_bytes.append(plen)
            off += HEADER_SIZE + plen
    return (
        np.frombuffer(offsets, np.int64),
        np.frombuffer(payload_bytes, np.uint32),
        n_channels,
        np.frombuffer(raw_headers, HEADER_DTYPE),
    )


def _check_header(buf, rel, off, index, size, n_channels):
    """Validate the header at ``buf[rel:]``, which is at byte ``off`` of the file."""
    where = f"Record {index} at byte {off}"
    if buf[rel] != RECORD_TAG:
        raise ValueError(
            f"{where}: tag {buf[rel]:#04x}, expected {RECORD_TAG:#04x}. "
            "This is not a MLA .xpoints file."
        )
    nch, plen = _SIZES.unpack_from(buf, rel + _SIZES_OFFSET)
    if nch == 0 or plen not in (nch, 2 * nch):
        raise ValueError(
            f"{where}: payload_bytes={plen} does not match n_channels={nch}."
        )
    if n_channels is not None and nch != n_channels:
        raise ValueError(f"{where}: n_channels={nch}, the file uses {n_channels}.")
    if off + HEADER_SIZE + plen > size:
        raise ValueError(f"{where}: payload runs past the end of the file.")
    return nch, plen


def read_spectra(path, offsets, payload_bytes, n_channels):
    """Read the counts of consecutive records.

    Parameters
    ----------
    path : str or pathlib.Path
        Path of the ``.xpoints`` file.
    offsets : numpy.ndarray
        Byte offset of the records to read, which must be consecutive records of
        the file (increasing, without gaps).
    payload_bytes : numpy.ndarray
        Payload size in bytes of these records.
    n_channels : int
        Number of channels of a record.

    Returns
    -------
    numpy.ndarray
        ``uint16`` array of shape ``(len(offsets), n_channels)``.
    """
    n_rows = len(offsets)
    out = np.empty((n_rows, n_channels), np.uint16)
    if n_rows == 0:
        return out
    first = int(offsets[0])
    last = int(offsets[-1]) + HEADER_SIZE + int(payload_bytes[-1])
    with open(path, "rb") as fh:
        fh.seek(first)
        raw = np.frombuffer(fh.read(last - first), np.uint8)
    if len(raw) != last - first:
        raise ValueError(f"{path} is shorter than its record table says.")
    starts = (offsets - first + HEADER_SIZE).tolist()
    for row, (start, plen) in enumerate(zip(starts, payload_bytes.tolist())):
        if plen == n_channels:
            out[row] = raw[start : start + plen]
        else:
            out[row] = raw[start : start + plen].view("<u2")
    return out
