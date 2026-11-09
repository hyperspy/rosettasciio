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

"""Builders of synthetic MLA ``.xpoints`` and X-Ray stage ``.jkf`` files.

The files follow the layout documented in ``rsciio/mla_xpoints/_records.py`` and
``rsciio/mla_xpoints/_jkf.py``. They hold random spectra instead of data of an
instrument and are used by ``test_mla_xpoints.py``.
"""

import struct

import numpy as np

N_CH = 2000
RECORD_BYTES = 80 + N_CH  # a record with uint8 counts


def xpoints_record(
    counts, x=1, y=2, size=10, bse=88.0, frac=0.113, wide=None, garbage=None
):
    """One record. ``garbage`` fills the padding bytes, as MLA does with stale memory.

    ``size`` is the width and height of the area of the spectrum, or a tuple of both.
    """
    width, height = size if isinstance(size, tuple) else (size, size)
    counts = np.asarray(counts)
    wide = bool(counts.max() > 255) if wide is None else wide
    payload = counts.astype("<u2" if wide else np.uint8).tobytes()
    header = bytearray(garbage[:80] if garbage else bytes(80))
    for off in (8, 16, 24, 28, 32, 36, 56, 60, 64):  # always zero in real files
        struct.pack_into("<I", header, off, 0)
    header[0] = 0x5A
    struct.pack_into("<HH", header, 2, 3, 2)
    struct.pack_into("<HH", header, 12, x, y)
    struct.pack_into("<HHH", header, 40, width, height, 1)
    struct.pack_into("<ff", header, 48, bse, frac)
    header[68] = 4
    struct.pack_into("<H", header, 70, len(counts))
    header[72] = 4
    struct.pack_into("<I", header, 76, len(payload))
    return bytes(header) + payload


def random_spectra(n, seed=0, n_channels=N_CH, peak=200):
    return np.random.default_rng(seed).integers(0, peak, size=(n, n_channels))


def packbits(data):
    """PackBits encoder (runs of 3 or more become repeat packets)."""
    out, i, n = bytearray(), 0, len(data)
    while i < n:
        run = 1
        while i + run < n and run < 128 and data[i + run] == data[i]:
            run += 1
        if run >= 3:
            out += bytes((257 - run, data[i]))
            i += run
            continue
        j = i
        while (
            j < n
            and j - i < 128
            and not (j + 2 < n and data[j] == data[j + 1] == data[j + 2])
        ):
            j += 1
        out += bytes((j - i - 1,)) + data[i:j]
        i = j
    return bytes(out)


def jkf_point(offset, record_bytes, x, y, size, bse, frac):
    """64-byte index record of one spectrum."""
    width, height = size if isinstance(size, tuple) else (size, size)
    rec = bytearray(64)
    struct.pack_into("<H", rec, 0, 2)
    rec[2:8] = b"\x72\x71\x00\x00\x1d\x05"  # stale memory, as in real files
    struct.pack_into("<QIII", rec, 8, offset, record_bytes, x, y)
    struct.pack_into("<II", rec, 40, width, height)
    struct.pack_into("<iI", rec, 48, -1, 0)
    struct.pack_into("<ff", rec, 56, bse, frac)
    return bytes(rec)


def jkf_particle(pid, bbox, image, points):
    xmin, ymin, xmax, ymax = bbox
    height, width = image.shape
    stride = (width + 3) // 4 * 4
    padded = np.zeros((height, stride), np.uint8)
    padded[:, :width] = image
    data = packbits(padded.tobytes())
    img = (
        struct.pack(
            "<IIQIIIQI", 1, 0xA28, 40 + len(data), width, height, stride, len(data), 0
        )
        + data
    )
    pts = b""
    if points:
        pts = struct.pack("<IIII", 2, 0xA64, 16 + 64 * len(points), len(points))
        pts += b"".join(points)
    body = struct.pack("<IIII", 0, pid - 1, 0, 0)
    body += struct.pack("<I4III", 800, xmin, ymin, xmax, ymax, 0, 2)
    body += struct.pack("<I", 0xCB)
    length = 16 + len(body) + len(img) + len(pts)
    return struct.pack("<IIQ", pid, 0x64A, length) + body + img + pts


def jkf_frame(number, center, particles, size=(500, 500)):
    hdr = struct.pack("<III", number, number, len(particles)) + struct.pack("<QI", 0, 0)
    hdr += struct.pack("<II", *size) + struct.pack("<I", 1) + bytes(0x48 - 0x34)
    hdr += struct.pack("<fffff", 158.667, center[0], center[1], 0.0, 0.0)
    hdr += struct.pack("<I", 0xCB)
    assert len(hdr) == 0x60 - 16
    body = hdr + b"".join(particles)
    return struct.pack("<IIQ", number + 3, 0x5DC, 16 + len(body)) + body


def jkf_file(frames):
    head = bytearray(0x530)
    head[0:4] = b"JK\x04\x00"
    struct.pack_into("<III", head, 0x120, 2, 0xC1C, 0x410)  # palette block
    content = bytes(head) + b"".join(frames) + b"\x00\x00"
    struct.pack_into("<Q", head, 0x10, len(content))
    return bytes(head) + b"".join(frames) + b"\x00\x00"


# Positions of the first frame, particle and image block of ``jkf_file`` output
FRAME0 = 0x530
PARTICLE0 = FRAME0 + 0x60
IMAGE0 = PARTICLE0 + 64

# (x, y, (width, height), bse_mode, bse_mode_fraction) of the spectra of ``dataset``
SPECTRA = [
    (11, 21, (10, 4), 70.0, 0.25),
    (15, 24, (10, 5), 71.0, 0.5),
    (101, 101, (1, 1), 90.0, 1.0),
    (3, 3, (8, 3), 120.0, 0.125),
    (5, 1, (0, 0), 121.0, 0.375),
]
FRAME_CENTERS = {10: (1000.0, -2000.0), 11: (-1500.0, 0.0)}
# frames are not square, to tell width from height
FRAME_SIZES = {10: (500, 400), 11: (450, 500)}
# frame, particle, grain of each spectrum; the last one lies outside the particle image
LOCATIONS = [(10, 1, 1), (10, 1, 2), (10, 2, 7), (11, 1, 3), (11, 1, -1)]


def jkf_for(offsets, sizes, rows=range(len(SPECTRA)), shift=None, reverse=False):
    """``.jkf`` bytes indexing ``rows`` of ``dataset``: two frames, four particles.

    ``shift`` adds a number to fields of the index records (``"x"``, ``"y"``,
    ``"width"``, ``"height"``), so that they differ from the record headers.
    ``reverse`` lists the frames and the spectra of a particle in reverse order.
    """
    shift = shift or {}

    def points(*which):
        out = []
        for i in which[::-1] if reverse else which:
            if i not in rows:
                continue
            x, y, (width, height), bse, frac = SPECTRA[i]
            size = (width + shift.get("width", 0), height + shift.get("height", 0))
            out.append(
                jkf_point(
                    offsets[i],
                    sizes[i],
                    x + shift.get("x", 0),
                    y + shift.get("y", 0),
                    size,
                    bse,
                    frac,
                )
            )
        return out

    labels = np.zeros((6, 8), np.uint8)
    labels[1:3, 1:3] = 1
    labels[3:5, 4:7] = 2
    frame10 = jkf_frame(
        10,
        FRAME_CENTERS[10],
        [
            jkf_particle(1, (10, 20, 18, 26), labels, points(0, 1)),
            jkf_particle(2, (100, 100, 104, 102), np.full((2, 4), 7), points(2)),
            jkf_particle(3, (200, 200, 203, 203), np.full((3, 3), 1), []),
        ],
        FRAME_SIZES[10],
    )
    frame11 = jkf_frame(
        11,
        FRAME_CENTERS[11],
        [jkf_particle(1, (0, 0, 4, 4), np.full((4, 4), 3), points(3, 4))],
        FRAME_SIZES[11],
    )
    return jkf_file([frame11, frame10] if reverse else [frame10, frame11])
