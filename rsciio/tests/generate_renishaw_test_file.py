# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

"""Creates files replicating Renishaw's .wdf files metadata structure (PSET Blocks)."""

import numpy as np

from rsciio.renishaw._api import (
    MetadataFlags,
    MetadataTypeMulti,
    MetadataTypeSingle,
    TypeNames,
    WDFReader,
)

# logging.basicConfig(level=10)


def _write_numeric(file, data, type):
    if type not in TypeNames.keys():
        raise ValueError(
            f"Trying to write number with unknown dataformat.\n"
            f"Input: {type}\n"
            f"Supported types: {list(TypeNames.keys())}"
        )
    np.array(data).astype(TypeNames[type]).tofile(file)


def _write_string(file, input_str, encoding="utf-8"):
    file.write(input_str.encode(encoding))


def _write_flag(file, type):
    _write_numeric(file, MetadataFlags[type].value, "uint8")


def _write_type(file, type):
    if type in MetadataTypeSingle._member_names_:
        _write_string(file, MetadataTypeSingle[type].value)
    elif type in MetadataTypeMulti._member_names_:
        _write_string(file, MetadataTypeMulti[type].value)
    else:
        raise ValueError(f"Invalid type {type}")


def _write_key(file, number):
    _write_numeric(file, number, "uint16")


def _write_header(file, id, uid, size):
    _write_string(file, id)
    _write_numeric(file, uid, "uint32")
    _write_numeric(file, size, "uint64")
    _write_numeric(file, int(0x54455350), "uint32")  # STREAM_IS_PSET
    _write_numeric(file, size - 24, "uint32")  # PSET_SIZE


def _write_pset_single(file, type, key, val):
    _write_type(file, type)
    _write_flag(file, "normal")
    _write_key(file, key)
    _write_numeric(file, val, type)  # value


def _write_pset_str(file, key, val):
    _write_type(file, "string")
    _write_flag(file, "normal")
    _write_key(file, key)
    _write_numeric(file, len(val), "uint32")
    _write_string(file, val)


def _write_pset_key(file, key, val):
    _write_type(file, "key")
    _write_flag(file, "normal")
    _write_key(file, key)
    _write_numeric(file, len(val), "uint32")
    _write_string(file, val)


def _write_pset_bin(file, key, val):
    _write_type(file, "binary")
    _write_flag(file, "normal")
    _write_key(file, key)
    _write_numeric(file, len(val), "uint32")
    _write_string(file, val, encoding="ascii")


def _write_pset_nested(file, key, length):
    _write_type(file, "nested")
    _write_flag(file, "normal")
    _write_key(file, key)
    _write_numeric(file, length, "uint32")


def _write_pset_array(file, type, key, data):
    _write_type(file, type)
    _write_flag(file, "array")
    _write_key(file, key)
    _write_numeric(file, len(data), "uint32")
    for entry in data:
        _write_numeric(file, entry, type)


def _write_pset_compressed(file, type, key, data):
    _write_type(file, type)
    _write_flag(file, "compressed")
    _write_key(file, key)
    _write_numeric(file, len(data), "uint32")
    _write_string(file, data)


class WDFFileGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate_flat_array_compressed(f):
        _write_header(f, "TEST", 0, 97)

        _write_pset_key(f, 1, "array")
        _write_pset_key(f, 2, "compressed")
        _write_pset_compressed(f, "string", 2, "abcdef")
        _write_pset_array(f, "int16", 1, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    @staticmethod
    def generate_nested_normal_testfile(f):
        _write_header(f, "TEST", 0, 367)

        _write_pset_key(f, 47612, "pair_before_nested")
        _write_pset_single(f, "int16", 47612, -10)
        _write_pset_key(f, 234, "key_before_nested")
        _write_pset_single(f, "int32", 463, 17)
        ## START nested (key before)
        _write_pset_key(f, 9874, "nested1")
        _write_pset_nested(f, 9874, 76)
        _write_pset_key(f, 1, "pair_in_nested")
        _write_pset_key(f, 47612, "pair_in_nested_doubled_key")
        _write_pset_single(f, "int32", 1, 40891)
        _write_pset_single(f, "int64", 47612, -8279)
        ## END nested 1
        _write_pset_single(f, "uint8", 234, 87)
        _write_pset_key(f, 463, "val_before_nested")
        ## START nested2 (key after)
        _write_pset_nested(f, 99, 126)
        _write_pset_key(f, 1, "pair_in_nested_doubled_key")
        _write_pset_single(f, "int8", 1, 43)
        ## START nested3
        _write_pset_key(f, 1234, "nested3")
        _write_pset_nested(f, 1234, 41)
        _write_pset_key(f, 1, "pair_in_double_nested")
        _write_pset_single(f, "int64", 1, -123)
        ## END nested 3
        ## START nested 4
        _write_pset_nested(f, 456, 0)
        _write_pset_key(f, 456, "nested4")
        ## END nested 4
        ## END nested 2
        _write_pset_key(f, 99, "nested2")

    @staticmethod
    def generate_flat_normal_testfile(f):
        _write_header(f, "TEST", 0, 311)

        _write_pset_single(f, "int8", 123, -12)
        _write_pset_key(f, 123, "single->key")
        _write_pset_key(f, 456, "key->single")
        _write_pset_single(f, "uint8", 456, 46)
        _write_pset_single(f, "int16", 1, 72)
        _write_pset_single(f, "int32", 2, 379)
        _write_pset_single(f, "int64", 3, 347)
        _write_pset_key(f, 1, "order1")
        _write_pset_key(f, 3, "order3")
        _write_pset_key(f, 2, "order2")
        _write_pset_key(f, 4, "order4")
        _write_pset_key(f, 5, "order5")
        _write_pset_key(f, 6, "order6")
        _write_pset_single(f, "float", 4, 74.7843)
        _write_pset_single(f, "double", 6, -378.36)
        _write_pset_single(f, "windows_filetime", 5, 8043148)
        _write_pset_str(f, 10, "test string 123")
        _write_pset_key(f, 10, "key for test string")
        _write_pset_key(f, 11, "key for binary")
        _write_pset_bin(f, 11, "binary string 123")


class WDFFileHandler:
    def __init__(self, filepath):
        self.filepath = filepath

    def write_file(self, func):
        if self.filepath.exists():
            self.filepath.unlink()
        self.filepath.parent.mkdir(exist_ok=True)
        with open(self.filepath, "wb+") as file:
            func(file)

    def read_file(self):
        with open(self.filepath, "rb") as file:
            wdf = WDFReader(
                file,
                self.filepath.name,
                use_uniform_signal_axis=True,
                load_unmatched_metadata=False,
            )
            filesize = self.filepath.stat().st_size
            wdf._block_info = wdf.locate_all_blocks(filesize)
            wdf._parse_metadata("TEST_0")
            return wdf.original_metadata
