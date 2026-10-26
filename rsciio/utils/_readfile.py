# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

"""Helpers to read single scalar values of a given byte order from a binary file."""

import struct

from rsciio.exceptions import ByteOrderError

_BYTE_ORDER_PREFIX = {"big": ">", "little": "<"}

# 1-byte character structs, used directly by the DigitalMicrograph reader
B_char = struct.Struct(">c")
L_char = struct.Struct("<c")


def _make_reader(fmt, description):
    structs = {
        endian: struct.Struct(prefix + fmt)
        for endian, prefix in _BYTE_ORDER_PREFIX.items()
    }

    def reader(f, endian):
        try:
            s = structs[endian]
        except KeyError:
            raise ByteOrderError(endian) from None
        return s.unpack(f.read(s.size))[0]

    reader.__doc__ = (
        f"Read {description} from the file object ``f``.\n\n"
        "``endian`` is the byte order and must be either 'big' or 'little'."
    )
    return reader


read_short = _make_reader("h", "a signed 2-byte integer")
read_ushort = _make_reader("H", "an unsigned 2-byte integer")
read_long = _make_reader("l", "a signed 4-byte integer")
read_ulong = _make_reader("L", "an unsigned 4-byte integer")
read_long_long = _make_reader("q", "a signed 8-byte integer")
read_ulong_long = _make_reader("Q", "an unsigned 8-byte integer")
read_float = _make_reader("f", "a 4-byte float")
read_double = _make_reader("d", "an 8-byte float")
read_boolean = _make_reader("B", "a 1-byte boolean (as an unsigned integer)")
read_byte = _make_reader("b", "a signed 1-byte integer")
read_char = _make_reader("c", "a 1-byte character")
