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

"""Reader for the index of the MLA X-Ray processing stage ``.jkf`` file.

Only the parts needed to place the spectra of a ``.xpoints`` file are decoded. The
file is a sequence of blocks, every block starting with ``u32 id, u32 type,
u64 length`` (the length is counted from the start of the block)::

    file header (0x120 bytes: "JK", version 4, file size ...)
    palette block   (type 0xC1C, at 0x120)
    frame block     (type 0x5DC)   one per frame, in frame order
      particle block  (type 0x64A)  one per particle of the frame
        image block     (type 0xA28)  PackBits u8 image of the particle bounding box
                                      (grain labels, 0 = outside, in the X-Ray stage)
        points block    (type 0xA64)  u32 length, u32 count, then 64-byte index
                                      records, one per spectrum of the particle

Every length and count is checked against the enclosing block before it is used, so
a corrupt file raises :class:`ValueError` and never reads out of bounds.
"""

import struct

import numpy as np

FRAME_TYPE = 0x5DC
PARTICLE_TYPE = 0x64A
IMAGE_TYPE = 0xA28
POINTS_TYPE = 0xA64
PALETTE_TYPE = 0xC1C
PALETTE_BLOCK_OFFSET = 0x120
FRAME_HEADER_SIZE = 0x60
PARTICLE_HEADER_SIZE = 64
IMAGE_HEADER_SIZE = 40
POINT_RECORD_SIZE = 64
# Real files end 2 bytes after the last frame block.
MAX_TAIL_BYTES = 16
# Frame and particle numbers are stored as u32 and returned as int32.
INT32_MAX = 2**31 - 1
# Limits of an image block. A particle image is at most a frame, 500 x 500 pixels in
# all files seen, and PackBits cannot expand data by more than 64 times, so larger
# declared sizes cannot be real and would only make the decoder allocate memory.
MAX_IMAGE_BYTES = 1 << 26
MAX_PACKBITS_EXPANSION = 64
# Where to look for a frame block if there is no palette block.
SCAN_WINDOW = 1 << 16
# Pixel size of the processing stage in um. It is a setting of the MLA measurement
# and is not stored in the ``.jkf``, so the value of the dataset it was decoded
# from is used.
PIXEL_SIZE_UM = 3.0

POINT_DTYPE = np.dtype(
    {
        "names": [
            "offset",
            "record_bytes",
            "x",
            "y",
            "width",
            "height",
            "bse_mode",
            "bse_mode_fraction",
        ],
        "formats": ["<u8", "<u4", "<u4", "<u4", "<u4", "<u4", "<f4", "<f4"],
        "offsets": [8, 16, 20, 24, 40, 44, 56, 60],
        "itemsize": POINT_RECORD_SIZE,
    }
)

INDEX_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("record_bytes", "<u4"),
        ("x", "<u4"),
        ("y", "<u4"),
        ("width", "<u4"),
        ("height", "<u4"),
        ("bse_mode", "<f4"),
        ("bse_mode_fraction", "<f4"),
        ("frame", "<i4"),
        ("particle", "<i4"),
        ("grain", "<i2"),
        ("stage_x_um", "<f8"),
        ("stage_y_um", "<f8"),
    ]
)
"""One row per spectrum of the index; see :func:`read_point_index`."""


def _unpack(fmt, buf, pos, limit):
    """``struct.unpack_from`` that refuses to read past ``limit``."""
    if pos < 0 or pos + struct.calcsize(fmt) > limit:
        raise ValueError(
            f"byte {pos}: {fmt} field runs past its block (limit {limit})."
        )
    return struct.unpack_from(fmt, buf, pos)


def _block(buf, pos, limit):
    return _unpack("<IIQ", buf, pos, limit)


def unpack_packbits(src, start, end, expected):
    """Decode the PackBits data in ``src[start:end]`` into ``expected`` bytes.

    Control byte ``c``: 0..127 copies the next ``c + 1`` bytes, 129..255 repeats the
    next byte ``257 - c`` times, 128 is a no-op.
    """
    if start < 0 or end > len(src) or start > end:
        raise ValueError(f"PackBits span {start}..{end} is outside the file.")
    out = bytearray()
    i = start
    while i < end:
        c = src[i]
        i += 1
        if c < 128:
            if i + c + 1 > end:
                raise ValueError(
                    f"PackBits literal at byte {i - 1} overruns its block."
                )
            out += src[i : i + c + 1]
            i += c + 1
        elif c > 128:
            if i >= end:
                raise ValueError(f"PackBits repeat at byte {i - 1} overruns its block.")
            out += bytes((src[i],)) * (257 - c)
            i += 1
        if len(out) > expected:
            raise ValueError(f"PackBits output exceeds the expected {expected} bytes.")
    if len(out) != expected:
        raise ValueError(f"PackBits decoded {len(out)} bytes, expected {expected}.")
    return bytes(out)


def _parse_image(buf, img, end, bbox):
    """Locate the image sub-block of a particle; returns its data span and the block end."""
    _, ityp, ilen = _block(buf, img, end)
    if ityp != IMAGE_TYPE:
        raise ValueError(f"byte {img}: expected an image block, got type {ityp:#x}.")
    width, height, stride, dlen = _unpack("<IIIQ", buf, img + 16, end)
    d0 = img + IMAGE_HEADER_SIZE
    if d0 + dlen != img + ilen or img + ilen > end:
        raise ValueError(f"byte {img}: image block length {ilen} is inconsistent.")
    if width != bbox[2] - bbox[0] or height != bbox[3] - bbox[1] or stride < width:
        raise ValueError(
            f"byte {img}: image {width}x{height} (stride {stride}) does not "
            f"match its bounding box {bbox}."
        )
    if (
        height * stride > MAX_IMAGE_BYTES
        or height * stride > MAX_PACKBITS_EXPANSION * dlen
    ):
        raise ValueError(
            f"byte {img}: image of {height} x {stride} bytes is too large to be "
            f"unpacked from {dlen} bytes."
        )
    return (d0, d0 + dlen, width, height, stride), d0 + dlen


def _parse_points(buf, pos, end):
    _, ptyp, plen, count = _unpack("<IIII", buf, pos, end)
    if (
        ptyp != POINTS_TYPE
        or plen != 16 + POINT_RECORD_SIZE * count
        or pos + plen != end
    ):
        raise ValueError(f"byte {pos}: malformed points block.")
    return np.frombuffer(buf, POINT_DTYPE, count=count, offset=pos + 16)


def _grains(buf, span, bbox, points):
    """Grain label under each spectrum of a particle, -1 where it lies outside the image."""
    d0, d1, width, height, stride = span
    raw = unpack_packbits(buf, d0, d1, expected=height * stride)
    labels = np.frombuffer(raw, np.uint8).reshape(height, stride)[:, :width]
    dx = points["x"].astype(np.int64) - bbox[0]
    dy = points["y"].astype(np.int64) - bbox[1]
    inside = (dx >= 0) & (dx < width) & (dy >= 0) & (dy < height)
    out = np.full(len(points), -1, np.int16)
    out[inside] = labels[dy[inside], dx[inside]]
    return out


def _parse_particle(buf, pos, limit, frame):
    """Rows of the index for one particle block; ``frame`` = (number, width, height, cx, cy)."""
    pid, typ, length = _block(buf, pos, limit)
    end = pos + length
    if pid > INT32_MAX:
        raise ValueError(f"byte {pos}: particle number {pid} is out of range.")
    if typ != PARTICLE_TYPE:
        raise ValueError(f"byte {pos}: expected a particle block, got type {typ:#x}.")
    if length < PARTICLE_HEADER_SIZE + IMAGE_HEADER_SIZE or end > limit:
        raise ValueError(
            f"byte {pos}: particle length {length} does not fit its frame."
        )
    number, f_width, f_height, cx, cy = frame
    bbox = _unpack("<4I", buf, pos + 36, end)
    if not (bbox[0] < bbox[2] <= f_width and bbox[1] < bbox[3] <= f_height):
        raise ValueError(
            f"byte {pos}: particle bounding box {bbox} is outside the "
            f"{f_width}x{f_height} frame."
        )
    span, after_image = _parse_image(buf, pos + PARTICLE_HEADER_SIZE, end, bbox)
    if after_image == end:
        return end, None
    points = _parse_points(buf, after_image, end)
    rows = np.zeros(len(points), INDEX_DTYPE)
    for name in POINT_DTYPE.names:
        rows[name] = points[name]
    rows["frame"] = number
    rows["particle"] = pid
    rows["grain"] = _grains(buf, span, bbox, points)
    px = points["x"].astype(np.float64)
    py = points["y"].astype(np.float64)
    # Image x runs along stage +X, image rows run along stage -Y.
    rows["stage_x_um"] = cx + (px + 0.5 - f_width / 2) * PIXEL_SIZE_UM
    rows["stage_y_um"] = cy - (py + 0.5 - f_height / 2) * PIXEL_SIZE_UM
    return end, rows


def _parse_frame(buf, pos, size):
    block_id, typ, length = _block(buf, pos, size)
    end = pos + length
    if typ != FRAME_TYPE or length < FRAME_HEADER_SIZE or end > size:
        raise ValueError(
            f"byte {pos}: bad frame block (type {typ:#x}, length {length})."
        )
    number, _, n_particles = _unpack("<III", buf, pos + 16, end)
    if number > INT32_MAX:
        raise ValueError(f"byte {pos}: frame number {number} is out of range.")
    f_width, f_height = _unpack("<II", buf, pos + 40, end)
    _, cx, cy = _unpack("<fff", buf, pos + 0x48, end)
    max_particles = (length - FRAME_HEADER_SIZE) // (
        PARTICLE_HEADER_SIZE + IMAGE_HEADER_SIZE
    )
    if n_particles > max_particles:
        raise ValueError(
            f"frame {number}: {n_particles} particles cannot fit in {length} bytes."
        )
    frame = (number, f_width, f_height, cx, cy)
    parts, p = [], pos + FRAME_HEADER_SIZE
    for _ in range(n_particles):
        p, rows = _parse_particle(buf, p, end, frame)
        if rows is not None:
            parts.append(rows)
    if p != end:
        raise ValueError(f"frame {number}: particle chain ends at {p}, block at {end}.")
    return end, parts


def _first_frame(buf, size):
    """Offset of the first frame block, which follows the palette block in all files seen.

    Without a palette block at its usual place, the first plausible frame header in
    the first 64 KiB is used.
    """
    try:
        _, typ, length = _block(buf, PALETTE_BLOCK_OFFSET, size)
    except ValueError:
        typ = None
    if typ == PALETTE_TYPE:
        start = PALETTE_BLOCK_OFFSET + length
        if size - start <= MAX_TAIL_BYTES:
            return start  # no frame at all
        if _block(buf, start, size)[1] != FRAME_TYPE:
            raise ValueError(
                f"byte {start}: the palette block is not followed by a frame block."
            )
        return start
    tag = struct.pack("<I", FRAME_TYPE)
    pos = 3
    while True:
        pos = buf.find(tag, pos + 1, min(size, SCAN_WINDOW))
        if pos < 0:
            raise ValueError("no frame block found.")
        try:
            _parse_frame(buf, pos - 4, size)
            return pos - 4
        except ValueError:
            continue


def read_point_index(path):
    """Read the spectrum index of an X-Ray stage ``.jkf`` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path of the ``.jkf`` file.

    Returns
    -------
    numpy.ndarray
        Structured array (dtype :data:`INDEX_DTYPE`) with one row per spectrum, sorted
        by ``offset``, the byte offset of the spectrum's record in the ``.xpoints``
        file. ``frame`` and ``particle`` are the numbers of the frame and the particle
        that contain the spectrum, ``grain`` is the grain label under it (-1 if it lies
        outside the particle image) and ``stage_x_um``, ``stage_y_um`` its stage
        position in um.

    Raises
    ------
    ValueError
        If the file is not a ``.jkf`` file or does not follow the decoded layout.
    """
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf[:2] != b"JK":
        raise ValueError(f"{path} has no 'JK' magic, it is not a .jkf file.")
    size = len(buf)
    if size < PALETTE_BLOCK_OFFSET:
        raise ValueError(f"{path} is too short for a .jkf file ({size} bytes).")
    parts, pos = [], _first_frame(buf, size)
    while size - pos > MAX_TAIL_BYTES:
        pos, rows = _parse_frame(buf, pos, size)
        parts.extend(rows)
    index = np.concatenate(parts) if parts else np.zeros(0, INDEX_DTYPE)
    return index[np.argsort(index["offset"], kind="stable")]
