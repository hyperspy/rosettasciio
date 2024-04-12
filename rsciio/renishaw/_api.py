# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

# This file incorporates work covered by the following copyright and
# permission notice:
#
#   MIT License
#
#   Copyright (c) 2022 T.Tian
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

# As mentioned above, this code is based on the py-wdf-reader
# (https://github.com/alchem0x2A/py-wdf-reader),
# which is inspired by Henderson, Alex
# (https://doi.org/10.5281/zenodo.495477).
# Moreover, this code is inspired by gwyddion (http://gwyddion.net).

# known limitations/problems:
#   - cannot parse BKXL-Block
#   - many blocks exist according to gwyddion that are not covered by testfiles
#       -> not parsed
#   - unmatched values for pset metadata blocks
#   - MapType is not used -> snake pattern read incorrectly for example
#   - quantity name is always set to Intensity (not extracted from file)
#   - many DataTypes are not considered for axes, only the following are used
#       - Spectral for signal
#       - X, Y, Z, Time, FocusTrackZ for navigation
#   - unclear what MAP contains

import datetime
import importlib.util
import logging
import os
from copy import deepcopy
from enum import Enum, EnumMeta, IntEnum
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyfit

from rsciio._docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC
from rsciio.utils import rgb_tools

_logger = logging.getLogger(__name__)


try:
    from PIL import Image
except ImportError:
    PIL_installed = False
    _logger.warning("Pillow not installed. Cannot load whitelight image.")
else:
    PIL_installed = True


def _find_key(data, target):
    """Finds key in nested dictionary. Returns generator object."""
    for key, value in data.items():
        if isinstance(value, dict):
            yield from _find_key(value, target)
        elif key == target:
            yield value


def _get_key(data, target):
    """Wrapper for _find_key(). Handles situation where no key is found."""
    gen_obj = _find_key(data, target)
    key = None
    try:
        key = next(gen_obj)
    except StopIteration:
        key = None
    return key


def _remove_none_from_dict(dict_in):
    """Recursive removal of None-values from a dictionary."""
    for key, value in list(dict_in.items()):
        if isinstance(value, dict):
            _remove_none_from_dict(value)
        elif value is None:
            del dict_in[key]


def convert_windowstime_to_datetime(wt):
    base = datetime.datetime(1601, 1, 1, 0, 0, 0, 0)
    delta = datetime.timedelta(microseconds=wt / 10)
    return (base + delta).isoformat("#", "seconds")


class DefaultEnum(IntEnum):
    Unknown = 0


class DefaultEnumMeta(EnumMeta):
    def __call__(cls, value, *args, **kwargs):
        if value not in cls._value2member_map_:
            return DefaultEnum(0)
        else:
            return super().__call__(value, *args, **kwargs)


class MetadataTypeSingle(Enum, metaclass=DefaultEnumMeta):
    int8 = "c"  # maybe char, but as far as I can tell only used as a number
    uint8 = "?"
    int16 = "s"
    int32 = "i"
    int64 = "w"
    float = "r"
    double = "q"
    windows_filetime = "t"


## no enum, because double entries
MetadataLengthSingle = {
    "len_int8": 1,
    "len_uint8": 1,
    "len_int16": 2,
    "len_int32": 4,
    "len_int64": 8,
    "len_float": 4,
    "len_double": 8,
    "len_windows_filetime": 8,
}


class MetadataTypeMulti(Enum, metaclass=DefaultEnumMeta):
    string = "u"
    nested = "p"
    key = "k"
    binary = "b"


class MetadataFlags(IntEnum, metaclass=DefaultEnumMeta):
    normal = 0
    compressed = 64
    array = 128


## < specifies little endian
## no enum, because double entries
TypeNames = {
    "int8": "<i1",  # byte int
    "int16": "<i2",  # short int
    "int32": "<i4",  # int
    "int64": "<i8",  # long int
    "uint8": "<u1",  # unsigned byte int
    "uint16": "<u2",  # unsigned short int
    "uint32": "<u4",  # unsigned int
    "uint64": "<u8",  # unsigned long int
    "float": "<f4",  # float (32)
    "double": "<f8",  # double (64)
    "windows_filetime": "<u8",  # same as uint64, converted in __read_numeric
}


class MeasurementType(IntEnum, metaclass=DefaultEnumMeta):
    Unspecified = 0
    Single = 1
    Series = 2
    Mapping = 3


## parser does not depend on this parameter (only put into original_metadata)
## testfiles: 2, 3, 4, 5, 8 missing
class ScanType(IntEnum, metaclass=DefaultEnumMeta):
    Unspecified = 0
    Static = 1
    Continuous = 2
    StepRepeat = 3
    FilterScan = 4
    FilterImage = 5
    StreamLine = 6
    StreamLineHR = 7
    PointDetector = 8


## TODO: in testfiles maps have value 0. What does this mean? unspecified?
## tested/covered cases: 0?, 2, 128
## cases: 1, 4, 8, 64 not implemented
## cases that should work, but are not tested: 16, 32 (negative scale)
class MapType(IntEnum, metaclass=DefaultEnumMeta):
    randompoints = 1  # rectangle
    column_major = 2  # x then y
    alternating = 4  # snake pattern, not implemented
    linefocus_mapping = 8  # linefocus -> laserspot is a line
    inverted_rows = 16  # rows collected right to left (negative scale)
    inverted_columns = 32  # columns collected bottom to top (negative scale)
    surface_profile = 64  # Z data is non-regular (gwyddion)?
    xyline = 128  # linescan -> use absolute distance starting from zero
    # offset + scale saved in metadata


# TODO: wavelength is named wavenumber in py-wdf-reader, gwyddion
class UnitType(IntEnum, metaclass=DefaultEnumMeta):
    arbitrary = 0
    raman_shift = 1
    wavelength = 2
    nanometer = 3
    electron_volt = 4
    micrometer = 5
    counts = 6
    electrons = 7
    millimeter = 8
    meter = 9
    kelvin = 10
    pascal = 11
    seconds = 12
    milliseconds = 13
    hours = 14
    days = 15
    pixels = 16
    intensity = 17
    relative_intensity = 18
    degrees = 19
    radians = 20
    celsius = 21
    fahrenheit = 22
    kelvin_per_minute = 23
    windows_file_time = 24
    microseconds = (
        25  # different from gwyddion, see PR#39 streamhr rapide mode (py-wdf-reader)
    )

    def __str__(self):
        unit_str = dict(
            arbitrary="",
            raman_shift="1/cm",
            wavelength="nm",
            nanometer="nm",
            electron_volt="eV",
            micrometer="µm",
            counts="counts",
            electrons="electrons",
            millimeter="mm",
            meter="m",
            kelvin="K",
            pascal="Pa",
            seconds="s",
            milliseconds="ms",
            hours="h",
            days="d",
            pixels="px",
            intensity="",
            relative_intensity="",
            degrees="°",
            radians="rad",
            celsius="°C",
            fahrenheit="°F",
            kelvin_per_minute="K/min",
            windows_file_time="ns",
            microseconds="µs",
        )
        return unit_str[self.name]


# used in ORGN/XLST
# TODO: figure out difference between frequency and spectral
# 1 (spectral) DEPRECATED according to gwyddion -> use frequency instead
# in py-wdf-reader the naming of spectral and frequency is switched
# in testfiles 1 (spectral) occurs exclusively for signal (XLST)
# also there are no frequency units in UnitType
class DataType(IntEnum, metaclass=DefaultEnumMeta):
    Arbitrary = 0
    Spectral = 1
    Intensity = 2
    X = 3
    Y = 4
    Z = 5
    Spatial_R = 6
    Spatial_Theta = 7
    Spatial_Phi = 8
    Temperature = 9
    Pressure = 10
    Time = 11
    Derivative = 12
    Polarization = 13
    FocusTrack_Z = 14
    TempRampRate = 15
    SpectrumDataChecksum = 16
    BitFlags = 17
    ElapsedTimeIntervals = 18
    Frequency = 19
    Mp_Well_Spatial_X = 22
    Mp_Well_Spatial_Y = 23
    Mp_LocationIndex = 24
    Mp_WellReference = 25
    EndMarker = 26
    ExposureTime = (
        27  # different from gwyddion, see PR#39 streamhr rapide mode (py-wdf-reader)
    )


class WDFReader(object):
    """Reader for Renishaw(TM) WiRE Raman spectroscopy files (.wdf format)

    The wdf file format is separated into several DataBlocks, with starting 4-char
    strings:

    fully parsed blocks:
    `WDF1`: header, contains general information
    `DATA`: Spectra data (intensity)
    `XLST`: contains signal axis, usually the Raman shift or wavelength
    `YLST`: ?
    `WMAP`: Information for mapping
    `MAP `: Mapping information(?), can exist multiple times
    `ORGN`: Data for stage origin + other axes, important for series
    `WXDA`: wiredata metadata, extra 1025 bytes at the end (null)
    `WXDM`: measurement metadata (contains VBScript for data acquisition, can be compressed)
    `WXCS`: calibration metadata
    `WXIS`: instrument metadata
    `WARP`: processing metadata(?), can exist multiple times
    `ZLDC`: zero level and dark current metadata
    `WHTL`: White light image
    `TEXT`: Annotation text etc

    not parsed blocks, but present in testfiles:
    `BKXL`: ?

    not parsed because not present in testfiles (from gwyddion):
    `WXDB`: datasetdata?
    `NAIL`: Thumbnail?
    `CFAR`: curvefit?
    `DLCS`: component?
    `PCAR`: pca?
    `MCRE`: em?
    `RCAL`: responsecall?
    `CAP `: cap?
    `WARA`: analysis?
    `WLBL`: spectrumlabels?
    `WCHK`: checksum?
    `RXCD`: rxcaldata?
    `RXCF`: rxcalfit?
    `XCAL`: xcal?
    `SRCH`: specsearch?
    `TEMP`: tempprofile?
    `UNCV`: unitconvert?
    `ARPR`: arplate?
    `ELEC`: elecsign?
    `AUX`: auxilarydata?
    `CHLG`: changelog?
    `SURF`: surface?

    Following the block name, there are two indicators:
    Block uid: uint32 (set, when blocks exist multiple times)
    Block size: uint64

    Blocks with the same name can occur multiple times (MAP, WARP in testfiles),
    these differ by UID.

    This parser first skips through the file to extract all Datablocks
    (__locate_all_blocks), the respective size and position is saved in _block_info.
    After that all blocks are parsed individually, however the order matters in some
    cases.

    Metadata is read from PSET blocks, whose structure is similar.
    However, there are lots of unmatched keys/values for unknown reasons.

    Signal axis is extracted from XLST-Block, data from DATA-Block,
    Navigation axes from ORGN/WMAP-Blocks.
    """

    _known_blocks = [
        "WDF1",
        "DATA",
        "XLST",
        "YLST",
        "WMAP",
        "MAP ",
        "ORGN",
        "WXDA",
        "WXDM",
        "WXCS",
        "WXIS",
        "WARP",
        "ZLDC",
        "WHTL",
        "TEXT",
        "BKXL",
    ]

    def __init__(self, f, filename, use_uniform_signal_axis, load_unmatched_metadata):
        self._file_obj = f
        self._filename = filename
        self._use_uniform_signal_axis = use_uniform_signal_axis
        self._load_unmatched_metadata = load_unmatched_metadata

        self.original_metadata = {}
        self._unmatched_metadata = {}

        self._unmatched_WXDM_keys = None
        self._reverse_signal = None
        self._block_info = None
        self._points_per_spectrum = None
        self._num_spectra = None
        self._measurement_type = None
        self.data = None
        self.axes = None
        self.metadata = None

    def read_file(self, filesize):
        ## first parse through to determine file structure (blocks)
        self._block_info = self.locate_all_blocks(filesize)

        ## parse header, this needs to be done first as it contains sizes for ORGN, DATA, XLST, YLST
        header_data = self._parse_WDF1()
        self._points_per_spectrum = header_data["points_per_spectrum"]
        self._num_spectra = header_data["num_spectra"]
        self._measurement_type = header_data["measurement_type"]

        ## parse metadata blocks
        self._parse_YLST(header_data["YLST_length"])
        self._parse_metadata("WXIS_0")
        self._parse_metadata("WXCS_0")
        ## WXDA has extra 1025 bytes at the end (newline: n\x00\x00...)
        self._parse_metadata("WXDA_0")
        self._parse_metadata("ZLDC_0")
        self._parse_metadata("WARP_0")
        self._parse_metadata("WARP_1")
        self._parse_metadata("WXDM_0")
        self._map_WXDM()
        self._parse_MAP("MAP_0")
        self._parse_MAP("MAP_1")
        self._parse_TEXT()

        ## parse blocks with axes information
        signal_dict = self._parse_XLST()
        nav_orgn = self._parse_ORGN(header_data["num_ORGN"])
        nav_wmap = self._parse_WMAP()

        ## set axes
        nav_dict = self._set_nav_axes(nav_orgn, nav_wmap)
        self.axes = self._set_axes(signal_dict, nav_dict)

        ## extract data + reshape
        self.data = self._parse_DATA()
        self._reshape_data()

        ## map metadata
        self.metadata = self.map_metadata()

        ## debug unmatched metadata
        if self._load_unmatched_metadata:
            _remove_none_from_dict(self._unmatched_metadata)
            self.original_metadata.update({"UNMATCHED": self._unmatched_metadata})

    def __read_numeric(self, type, size=1, ret_array=False, convert=True):
        if type not in TypeNames.keys():
            raise ValueError(
                f"Trying to read number with unknown dataformat.\n"
                f"Input: {type}\n"
                f"Supported types: {list(TypeNames.keys())}"
            )
        data = np.fromfile(self._file_obj, dtype=TypeNames[type], count=size)
        ## convert unsigned ints to ints
        ## because int + uint = float -> problems with indexing
        if type in ["uint8", "uint16", "uint32", "uint64"] and convert:
            dt = "<i8"
            data = data.astype(np.dtype(dt))
        elif type == "windows_filetime" and convert:
            data = list(map(convert_windowstime_to_datetime, data))
        if size == 1 and not ret_array:
            return data[0]
        else:
            return data

    def __read_utf8(self, size):
        return self._file_obj.read(size).decode("utf8").replace("\x00", "")

    def __locate_single_block(self, pos):
        self._file_obj.seek(pos)
        block_name = self.__read_utf8(0x4)
        if block_name not in self._known_blocks:
            _logger.debug(f"Unknown Block {block_name} encountered.")
        block_uid = self.__read_numeric("uint32")
        block_size = self.__read_numeric("uint64")
        return block_name, block_uid, block_size

    def locate_all_blocks(self, filesize):
        _logger.debug("ID     UID CURPOS   SIZE")
        _logger.debug("--------------------------------------")
        block_info = {}
        block_header_size = 16
        curpos = 0
        block_size = 0
        while curpos < filesize:
            try:
                block_name, block_uid, block_size = self.__locate_single_block(curpos)
            except EOFError:
                _logger.warning("Missing characters at the end of the file.")
                break
            except UnicodeDecodeError:
                _logger.warning(
                    f"{block_name} size ({block_size}) invalid or extra characters at EOF."
                )
                break
            else:
                _logger.debug(f"{block_name}   {block_uid}   {curpos:<{9}}{block_size}")
                block_info[block_name.replace(" ", "") + "_" + str(block_uid)] = (
                    curpos + block_header_size,
                    block_size,
                )
                curpos += block_size

        if curpos > filesize:
            _logger.warning("Missing characters at the end of the file.")
        _logger.debug("--------------------------------------")
        _logger.debug(f"filesize:    {filesize}")
        _logger.debug(f"parsed size: {curpos}")
        return block_info

    def _check_block_exists(self, block_name):
        if block_name in self._block_info.keys():
            return True
        else:
            _logger.debug(f"Block {block_name} not present in file.")
            return False

    @staticmethod
    def _check_block_size(name, error_area, expected_size, actual_size):
        if expected_size < actual_size:
            _logger.warning(
                f"Unexpected size of {name} Block."
                f"{error_area} may be read incorrectly."
            )
        elif expected_size > actual_size:
            if name == "WXDA_0":
                _logger.debug(
                    f"{name} is not read completely ({expected_size - actual_size} bytes missing)\n"
                    f"WXDA has extra 1025 bytes at the end (newline + NULL: n\\x00\\x00\\x00...)"
                )
            else:
                _logger.debug(
                    f"{name} is not read completely ({expected_size - actual_size} bytes missing)"
                )

    def _parse_metadata(self, id):
        _logger.debug(f"Parsing {id}")
        pset_size = self._get_psetsize(id)
        if pset_size == 0:
            return
        metadata = self._pset_read_metadata(pset_size, id)
        if metadata is None:
            _logger.warning(f"Invalid key in {id}")
        self.original_metadata.update({id: metadata})

    def _get_psetsize(self, id, warning=True):
        if not self._check_block_exists(id):
            return 0
        stream_is_pset = int(0x54455350)
        pos, block_size = self._block_info[id]
        self._file_obj.seek(pos)
        is_pset = self.__read_numeric("uint32")
        pset_size = self.__read_numeric("uint32")
        if is_pset != stream_is_pset:
            _logger.debug(f"No PSET found in {id} -> cannot extract metadata.")
            return 0
        if warning:
            self._check_block_size(id, "Metadata", block_size - 16, pset_size + 8)
        return pset_size

    def _pset_read_metadata(self, length, id):
        key_dict = {}
        value_dict = {}
        remaining = length
        ## remaining slightly above 0 may lead to problems
        ## as type, flag, key and entry are still read (for example remaining=2)
        ## however, this does not happen in testfiles (always ends exactly on 0)
        while remaining > 0:
            type = self.__read_utf8(0x1)
            flag = MetadataFlags(self.__read_numeric("uint8"))
            if flag == DefaultEnum.Unknown.name:
                _logger.error(
                    f"Invalid metadata flag ({flag}) encountered while parsing {id}."
                    f"Cannot read metadata from this Block."
                )
                return None
            key = str(self.__read_numeric("uint16"))
            remaining -= 4
            entry, num_bytes = self._pset_switch_read_on_flag(
                flag, type, f"{id}_{remaining}"
            )
            if type == "k":
                key_dict[key] = entry
            else:
                value_dict[key] = entry
            remaining -= num_bytes
        return self._pset_match_keys_and_values(id, key_dict, value_dict)

    def _pset_match_keys_and_values(self, id, key_dict, value_dict):
        result = {}
        ## TODO: debug setup, remove once it is understood why there are unmatched keys/values
        ## print(f"{id}(keys, {len(list(key_dict))}): {list(key_dict)}")
        ## print(f"{id}(values, {len(list(value_dict))}): {list(value_dict)}")
        ## print()
        for key in list(key_dict.keys()):
            ## keep mismatched keys for debugging
            try:
                val = value_dict.pop(key)
            except KeyError:
                pass  ## only occurs for WXDM in testfiles
            else:
                result[key_dict.pop(key)] = val
        ## TODO: Why are there unmatched keys/values
        if len(key_dict) != 0 or len(value_dict) != 0:
            self._unmatched_metadata.update(
                {id: {"keys": key_dict, "values": value_dict}}
            )
        return result

    def _pset_switch_read_on_flag(self, flag, type, id):
        return {
            MetadataFlags.normal: self._pset_read_normal_flag,
            MetadataFlags.array: self._pset_read_array_flag,
            MetadataFlags.compressed: self._pset_read_compressed_flag,
        }.get(flag)(type=type, id=id, flag=flag)

    def _pset_read_normal_flag(self, type, id, **kwargs):
        result, num_bytes = self._pset_read_entry(type, 1, id)
        return result, num_bytes

    def _pset_read_array_flag(self, type, id, **kwargs):
        if type not in MetadataTypeSingle._value2member_map_:
            _logger.debug(f"array flag, but not single dtype: {type}")
        size = self.__read_numeric("uint32")
        result, num_bytes = self._pset_read_entry(type, size, id)
        return result, num_bytes + 4  # +4, because of size read

    def _pset_read_compressed_flag(self, type, **kwargs):
        if type != "u":
            _logger.debug(f"compressed flag, but no string ({type})")
        size = self.__read_numeric("uint32")
        self._file_obj.seek(size, 1)  # move fp from current position
        result = None
        num_bytes = size + 4  # +4, because of size read
        return result, num_bytes

    def _pset_read_entry(self, type, size, id):
        if type in MetadataTypeSingle._value2member_map_:
            type_str = MetadataTypeSingle(type).name
            type_len = MetadataLengthSingle[f"len_{type_str}"]
            result = self.__read_numeric(type_str, size=size)
            return result, type_len * size
        elif type in MetadataTypeMulti._value2member_map_:
            length = self.__read_numeric("uint32")
            result = self._pset_read_multitype_entry_helper(type, length, id)
            return result, length + 4  # reading the length itself equals 4 bytes
        else:
            _logger.error(
                f"Invalid type ({type}) encountered while parsing metadata from {id}-Block."
            )
            raise RuntimeError()

    def _pset_read_multitype_entry_helper(self, type, length, id):
        if type == "b":
            result = "binary: " + self._file_obj.read(length).hex().upper()
        elif type in ["u", "k"]:
            result = self.__read_utf8(length)
        elif type == "p":
            result = self._pset_read_metadata(length, id)
        return result

    def _match_WXDM_values(self, level, keys_in, unmatched_keys):
        result = {}
        keys_lvl = [i for i in keys_in if i.count("_") == level]
        for num, k in enumerate(keys_lvl):
            keys_in.remove(k)
            subkeys = [i for i in keys_in if i.startswith(k)]
            unmatched_values_k = self._unmatched_metadata[k]["values"]
            result[f"sub{num}"] = {
                "subdicts": self._match_WXDM_values(level + 1, subkeys, unmatched_keys)
            }
            if len(result[f"sub{num}"]["subdicts"]) == 0:
                del result[f"sub{num}"]["subdicts"]
            matches = {}
            for k2, v2 in unmatched_values_k.items():
                try:
                    matches[unmatched_keys[k2]] = v2
                except KeyError:
                    pass
                else:
                    self._unmatched_WXDM_keys.pop(k2, None)
            result[f"sub{num}"].update(matches)
        return result

    def _map_WXDM(self):
        if "WXDM_0" not in list(self.original_metadata):
            return
        WXDM_entries = [
            i for i in list(self._unmatched_metadata) if i.startswith("WXDM")
        ]
        ## only top level contains unmatched keys
        ## these keys are used multiple times for different values in different subdicts
        self._unmatched_WXDM_keys = self._unmatched_metadata["WXDM_0"].get("keys")
        if self._unmatched_WXDM_keys is None:
            return
        matched_WXDM = self._match_WXDM_values(
            1, WXDM_entries, deepcopy(self._unmatched_WXDM_keys)
        )
        self.original_metadata["WXDM_0"].update(
            {"extra_matches": matched_WXDM["sub0"]["subdicts"]}
        )
        ## remove matched values/keys from unmatched_metadata
        for k in WXDM_entries:
            for k2 in deepcopy(list(self._unmatched_metadata[k])):
                if k2 not in self._unmatched_WXDM_keys:
                    self._unmatched_metadata[k].pop(k2)

    ## `MAP ` contains more information than just PSET
    ## TODO: what is this extra data? max peak? (MAP slightly off, MAP1 different)
    def _parse_MAP(self, id):
        if not self._check_block_exists(id):
            return
        pset_size = self._get_psetsize(id, warning=False)
        _, block_size = self._block_info[id]
        metadata = self._pset_read_metadata(pset_size, id)
        npoints = self.__read_numeric("uint64")
        ## 8 bytes from npoints + 4*npoints data
        self._check_block_size(
            id, "Metadata", block_size - 16, pset_size + 8 + 8 + 4 * npoints
        )
        metadata["npoints"] = npoints
        metadata["data"] = self.__read_numeric("float", npoints)
        self.original_metadata.update({id.replace(" ", ""): metadata})

    def _parse_WDF1(self):
        header = {}
        result = {}
        self._file_obj.seek(0)
        pos, size_block = self._block_info["WDF1_1"]
        if size_block != 512:
            _logger.warning("Unexpected header size. File might be invalid.")
        if pos != 16:
            _logger.warning("Unexpected start of file. File might be invalid.")

        ## TODO: what is ntracks?
        ## TODO: warning when file_status_error_code nonzero?
        self._file_obj.seek(pos)
        header["flags"] = self.__read_numeric("uint64")
        header["uuid"] = f"{self.__read_numeric('uint32', convert=False)}"
        for _ in range(3):
            header["uuid"] += f"-{self.__read_numeric('uint32', convert=False)}"
        _ = self.__read_numeric("uint32", size=3)
        header["ntracks"] = self.__read_numeric("uint32")
        header["file_status_error_code"] = self.__read_numeric("uint32")
        result["points_per_spectrum"] = self.__read_numeric("uint32")
        header["capacity"] = self.__read_numeric("uint64")
        result["num_spectra"] = self.__read_numeric("uint64")
        ## if cosmic ray removal (crr) is enabled,
        ## then 2 extra spectra are collected
        ## each frame is then averaged with these 2 extra spectra
        ## accumulations in the WDF1 section includes crr
        ## whereas accumulations in WXDM does not
        ## the WXDM accumulations is used for the metadata attribute
        header["accumulations_per_spectrum"] = self.__read_numeric("uint32")
        result["YLST_length"] = self.__read_numeric("uint32")
        header["XLST_length"] = self.__read_numeric("uint32")
        result["num_ORGN"] = self.__read_numeric("uint32")
        header["app_name"] = self.__read_utf8(24)  # must be "WiRE"
        header["app_version"] = f"{self.__read_numeric('uint16', convert=False)}"
        for _ in range(3):
            header["app_version"] += f"-{self.__read_numeric('uint16', convert=False)}"
        header["scan_type"] = ScanType(self.__read_numeric("uint32")).name
        result["measurement_type"] = MeasurementType(self.__read_numeric("uint32")).name
        time_start_wt = self.__read_numeric("uint64")
        time_end_wt = self.__read_numeric("uint64")
        header["time_start"] = convert_windowstime_to_datetime(time_start_wt)
        header["time_end"] = convert_windowstime_to_datetime(time_end_wt)
        header["quantity_unit"] = UnitType(self.__read_numeric("uint32")).name
        header["laser_wavenumber"] = self.__read_numeric("float")
        _ = self.__read_numeric("uint64", size=6)
        header["username"] = self.__read_utf8(32)
        header["title"] = self.__read_utf8(160)

        header.update(result)
        self.original_metadata.update({"WDF1_1": header})
        if header["num_spectra"] != header["capacity"]:
            _logger.warning(
                f"Unfinished measurement."
                f"The number of spectra ({header['num_spectra']}) written to the file is different"
                f"from the set number of spectra ({header['capacity']})."
                "Trying to still use the data of the measured data."
            )
        if header["points_per_spectrum"] != header["XLST_length"]:
            raise RuntimeError("File contains ambiguous signal axis sizes.")
        return result

    def _parse_XLST_or_YLST(self, name, size):
        pos, block_size = self._block_info[name.upper() + "LST_0"]
        if name.upper() == "X":
            self._check_block_size("XLST", "Signal axis", block_size - 16, 4 * size + 8)
        else:
            self._check_block_size("YLST", "Metadata", block_size - 16, 4 * size + 8)

        self._file_obj.seek(pos)
        type = DataType(self.__read_numeric("uint32")).name
        if type != "Spectral" and name.upper() == "X":
            _logger.warning(
                "Signal axis not classified as spectral. File may be invalid"
            )
        unit_type = UnitType(self.__read_numeric("uint32"))
        unit = str(unit_type)
        axis_name = self._convert_signal_axis_name(unit_type)
        data = self.__read_numeric("float", size=size)
        return axis_name, unit, data

    @staticmethod
    def _convert_signal_axis_name(unit_type):
        if unit_type == UnitType.raman_shift:
            name = "Raman Shift"
        elif unit_type == UnitType.wavelength:
            name = "Wavenumber"
        elif unit_type in [UnitType.nanometer, UnitType.micrometer]:
            name = "Wavelength"
        else:
            name = "Unknown"
        return name

    def _parse_XLST(self):
        if not self._check_block_exists("XLST_0"):
            raise RuntimeError(
                "File contains no information on signal axis (XLST-Block missing)."
            )
        name, unit, data = self._parse_XLST_or_YLST("X", self._points_per_spectrum)
        if name == "Unknown":
            _logger.warning("Cannot identify signal axis name.")
        signal_dict = {}
        signal_dict["size"] = self._points_per_spectrum
        signal_dict["navigate"] = False
        signal_dict["name"] = name
        signal_dict["units"] = unit
        if data[0] > data[1]:
            data = data[::-1]
            self._reverse_signal = True
        else:
            self._reverse_signal = False
        if self._use_uniform_signal_axis:
            signal_dict["offset"], signal_dict["scale"] = self._fit_axis(data)
        else:
            signal_dict["axis"] = data
        return signal_dict

    @staticmethod
    def _fit_axis(data, threshold=1):
        offset, scale = polyfit(np.arange(data.size), data, deg=1)
        scale_compare = 100 * np.max(np.abs(np.diff(data) - scale) / scale)
        if scale_compare > threshold:
            _logger.warning(
                f"The relative variation of the signal-axis-scale ({scale_compare:.2f}%) exceeds 1%.\n"
                "                              "
                "Using a non-uniform-axis is recommended."
            )
        return offset, scale

    def _parse_YLST(self, size):
        if not self._check_block_exists("YLST_0"):
            return
        name, unit, data = self._parse_XLST_or_YLST("Y", size)
        self.original_metadata.update(
            {
                "YLST_0": {
                    "name": name,
                    "units": unit,
                    "size": size,
                    "data": data,
                }
            }
        )

    def _parse_ORGN(self, header_orgn_count):
        if not self._check_block_exists("ORGN_0"):
            return {}
        orgn_nav = {}
        orgn_metadata = {}
        pos, block_size = self._block_info["ORGN_0"]
        self._file_obj.seek(pos)
        origin_count = self.__read_numeric("uint32")
        if origin_count != header_orgn_count:
            _logger.warning(
                "Ambiguous number of entrys for ORGN block."
                "This may lead to incorrect metadata and axes."
            )
        self._check_block_size(
            "ORGN",
            "Navigation axes",
            block_size - 16,
            4 + origin_count * (24 + 8 * self._num_spectra),
        )
        for _ in range(origin_count):
            ax_tmp_dict = {}
            ## ignore first bit of dtype read (sometimes 0, sometimes 1 in testfiles)
            dtype = DataType(
                self.__read_numeric("uint32", convert=False) & ~(0b1 << 31)
            ).name
            ax_tmp_dict["units"] = str(UnitType(self.__read_numeric("uint32")))
            ax_tmp_dict["annotation"] = self.__read_utf8(0x10)
            ax_tmp_dict["data"] = self._set_data_for_ORGN(dtype)

            if dtype not in [
                DataType.SpectrumDataChecksum.name,
                DataType.BitFlags.name,
            ]:
                self._warning_msg_untested_dtype_ORGN(dtype)
                orgn_nav[dtype] = ax_tmp_dict
            else:
                orgn_metadata[dtype] = ax_tmp_dict
        if self._measurement_type != MeasurementType.Series.name or len(orgn_nav) > 1:
            orgn_metadata["Time"] = orgn_nav.pop("Time")
        self.original_metadata.update({"ORGN_0": orgn_metadata})
        return orgn_nav

    def _warning_msg_untested_dtype_ORGN(self, dtype):
        if dtype not in [
            DataType.X.name,
            DataType.Y.name,
            DataType.Z.name,
            DataType.Time.name,
            DataType.FocusTrack_Z.name,
        ]:
            _logger.warning(
                f"Loading {dtype}-axis from ORGN-Block, which is not supported by tests."
            )

    def _set_data_for_ORGN(self, dtype):
        if dtype == DataType.Time.name:
            result = self.__read_numeric(
                "uint64", size=self._num_spectra, ret_array=True
            )
            result -= result[0]
        else:
            result = self.__read_numeric(
                "double", size=self._num_spectra, ret_array=True
            )
        return result

    def _parse_WMAP(self):
        if not self._check_block_exists("WMAP_0"):
            return {}
        pos, block_size = self._block_info["WMAP_0"]
        self._file_obj.seek(pos)
        self._check_block_size(
            "WMAP", "Navigation axes", block_size - 16, 4 * (3 + 3 * 3)
        )

        flag = MapType(self.__read_numeric("uint32")).name
        _ = self.__read_numeric("uint32")
        offset_xyz = [self.__read_numeric("float") for _ in range(3)]
        scale_xyz = [self.__read_numeric("float") for _ in range(3)]
        size_xyz = [self.__read_numeric("uint32") for _ in range(3)]
        linefocus_size = self.__read_numeric("uint32")

        result = {
            "linefocus_size": linefocus_size,
            "flag": flag,
            "offset_xyz": offset_xyz,
            "scale_xyz": scale_xyz,
            "size_xyz": size_xyz,
        }
        self.original_metadata.update({"WMAP_0": result})
        return result

    def _set_nav_via_WMAP(self, wmap_dict, units):
        result = {}
        for idx, ax_name in enumerate(["X", "Y"]):
            axis_tmp = {
                "name": ax_name,
                "offset": wmap_dict["offset_xyz"][idx],
                "scale": wmap_dict["scale_xyz"][idx],
                "size": wmap_dict["size_xyz"][idx],
                "navigate": True,
                "units": units,
            }
            result[ax_name] = axis_tmp

        # TODO: differentiate between more map_modes/flags
        flag = wmap_dict["flag"]
        if flag == MapType.xyline.name:
            result = self._set_wmap_nav_linexy(result["X"], result["Y"])
        elif flag == DefaultEnum.Unknown.name:
            _logger.info(f"Unknown flag ({wmap_dict['flag']}) for WMAP mapping.")
        return result

    def _set_wmap_nav_linexy(self, x_axis, y_axis):
        # TODO: save original axis scales and offset in metadata?
        # currently only in original_metadata.WMAP_0
        result = deepcopy(x_axis)
        scale_abs = np.sqrt(x_axis["scale"] ** 2 + y_axis["scale"] ** 2)
        result["scale"] = scale_abs
        result["offset"] = 0
        result["name"] = "Abs. Distance"
        return {"Distance": result}

    def _set_nav_axes(self, orgn_data, wmap_data):
        if not self._compare_measurement_type_to_ORGN_WMAP(orgn_data, wmap_data):
            _logger.warning(
                "Inconsistent MeasurementType and ORGN/WMAP Blocks."
                "Navigation axes may be set incorrectly."
            )
        if self._measurement_type == "Mapping":
            ## access units from arbitrary ORGN axis (should be the same for all)
            units = orgn_data[next(iter(orgn_data))]["units"]
            nav_dict = self._set_nav_via_WMAP(wmap_data, units)
        elif self._measurement_type == "Series":
            nav_dict = self._set_nav_via_ORGN(orgn_data)
        else:
            nav_dict = {}
        return nav_dict

    def _set_nav_via_ORGN(self, orgn_data):
        nav_dict = deepcopy(orgn_data)
        if len(nav_dict) != 1:
            _logger.warning(
                f"Series, but number of navigation axes ({len(nav_dict)}) exist is not 1."
            )
        for axis in orgn_data.keys():
            del nav_dict[axis]["annotation"]
            data = nav_dict[axis].pop("data")
            nav_dict[axis]["navigate"] = True
            nav_dict[axis]["size"] = data.size
            nav_dict[axis]["name"] = axis
            scale_mean = np.mean(np.diff(data))
            if axis == "FocusTrack_Z" or scale_mean == 0:
                # FocusTrack_Z is not uniform and not necessarily ordered
                # Fix me when hyperspy supports non-ordered non-uniform axis
                # For now, remove units and fall back on default axis
                # nav_dict[axis]["axis"] = data
                if scale_mean == 0:
                    # case "scale_mean == 0" is for series where the axis is invariant.
                    # In principle, this should happen but the WiRE software allows it
                    reason = f"Axis {axis} is invariant"
                else:
                    reason = "Non-ordered axis is not supported"
                _logger.warning(
                    f"{reason}, a default axis with scale 1 "
                    "and offset 0 will be used."
                )
                del nav_dict[axis]["units"]
            else:
                # time axis in test data is not perfectly uniform, but X,Y,Z are
                nav_dict[axis]["offset"] = data[0]
                nav_dict[axis]["scale"] = scale_mean

        return nav_dict

    def _compare_measurement_type_to_ORGN_WMAP(self, orgn_data, wmap_data):
        no_wmap = len(wmap_data) == 0
        no_orgn = len(orgn_data) == 0

        if self._measurement_type not in MeasurementType._member_names_:
            raise ValueError("Invalid measurement type.")
        elif self._measurement_type != "Mapping" and (not no_wmap):
            _logger.warning("No Mapping expected, but WMAP Block exists.")
            return False
        elif self._measurement_type == "Mapping" and no_wmap:
            _logger.warning("Mapping expected, but no WMAP Block.")
            return False
        elif self._measurement_type == "Series" and no_orgn:
            _logger.warning("Series expected, but no (X, Y, Z, time) data.")
            return False
        elif self._measurement_type == "Single" and (not no_orgn or not no_wmap):
            _logger.warning("Spectrum expected, but extra axis present.")
            return False
        elif self._measurement_type == "Unspecified":
            _logger.warning(
                "Unspecified measurement type. May lead to incorrect results."
            )
        return True

    def _set_axes(self, signal_dict, nav_dict):
        signal_dict["index_in_array"] = len(nav_dict)
        if len(nav_dict) == 2:
            nav_dict["Y"]["index_in_array"] = 0
            nav_dict["X"]["index_in_array"] = 1
        elif len(nav_dict) == 1:
            axis = next(iter(nav_dict))
            nav_dict[axis]["index_in_array"] = 0

        axes = deepcopy(nav_dict)
        axes["signal_dict"] = deepcopy(signal_dict)
        return sorted(axes.values(), key=lambda item: item["index_in_array"])

    def _parse_DATA(self):
        """Get information from DATA block"""
        if not self._check_block_exists("DATA_0"):
            raise RuntimeError("File does not contain data (DATA-Block missing).")
        pos, block_size = self._block_info["DATA_0"]
        size = self._points_per_spectrum * self._num_spectra
        self._check_block_size("DATA", "Data", block_size - 16, 4 * size)
        self._file_obj.seek(pos)
        return self.__read_numeric("float", size=size)

    def _reshape_data(self):
        if self._use_uniform_signal_axis:
            signal_size = self.axes[-1]["size"]
        else:
            signal_size = self.axes[-1]["axis"].size

        axes_sizes = []
        for i in range(len(self.axes) - 1):
            axes_sizes.append(self.axes[i]["size"])

        ## np.prod of an empty array is 1
        if self._num_spectra != np.array(axes_sizes).prod():
            _logger.warning(
                "Axes sizes do not match data size.\n"
                "Data is averaged over multiple collected spectra."
            )
            self.data = np.mean(self.data.reshape(self._num_spectra, -1), axis=0)

        axes_sizes.append(signal_size)
        self.data = np.reshape(self.data, axes_sizes)
        if self._reverse_signal:
            self.data = np.flip(self.data, len(axes_sizes) - 1)

    def _map_general_md(self):
        general = {}
        general["title"] = self.original_metadata.get("WDF1_1", {}).get("title")
        general["original_filename"] = os.path.split(self._filename)[1]
        try:
            date, time = self.original_metadata["WDF1_1"]["time_start"].split("#")
        except KeyError:
            pass
        else:
            general["date"] = date
            general["time"] = time
        return general

    def _map_signal_md(self):
        signal = {}
        if importlib.util.find_spec("lumispy") is None:
            _logger.warning(
                "Cannot find package lumispy, using generic signal class BaseSignal."
            )
            signal["signal_type"] = ""
        else:
            signal["signal_type"] = "Luminescence"  # pragma: no cover

        try:
            quantity_unit = self.original_metadata.get("WDF1_1", {}).get(
                "quantity_unit"
            )
        except KeyError:
            signal["quantity"] = "Intensity"
        else:
            signal["quantity"] = f"Intensity ({quantity_unit.capitalize()})"
        return signal

    def _map_laser_md(self):
        laser = {}
        laser_original_md = self.original_metadata.get("WXCS_0", {}).get("Lasers")
        if laser_original_md is None:
            return None
        laser["wavelength"] = 1e7 / _get_key(laser_original_md, "Wavenumber")
        return laser

    def _map_spectrometer_md(self):
        spectrometer = {"Grating": {}}
        gratings_original_md = self.original_metadata.get("WXCS_0", {}).get("Gratings")
        if gratings_original_md is None:
            return None
        spectrometer["Grating"]["groove_density"] = _get_key(
            gratings_original_md, "Groove Density (lines/mm)"
        )
        return spectrometer

    def _map_detector_md(self):
        detector = {}
        detector["detector_type"] = "CCD"
        ccd_original_metadata = self.original_metadata.get("WXCS_0", {}).get("CCD")
        if ccd_original_metadata is None:
            return None
        detector["model"] = ccd_original_metadata.get("CCD")
        detector["temperature"] = ccd_original_metadata.get("Target temperature")
        wxdm_metadata = self.original_metadata.get("WXDM_0", {}).get("extra_matches")
        if wxdm_metadata is None:
            return detector
        processing = {}
        detector["frames"] = _get_key(wxdm_metadata, "Accumulations")
        exposure_per_frame = _get_key(wxdm_metadata, "Exposure Time")  # ms
        try:
            detector["integration_time"] = (
                exposure_per_frame * detector["frames"] / 1000
            )  # s
        except TypeError:
            pass
        processing["cosmic_ray_removal"] = _get_key(
            wxdm_metadata, "Use Cosmic Ray Remove"
        )
        detector["processing"] = processing
        return detector

    def map_metadata(self):
        general = self._map_general_md()
        signal = self._map_signal_md()
        detector = self._map_detector_md()
        laser = self._map_laser_md()
        spectrometer = self._map_spectrometer_md()

        # TODO: find laser power?

        metadata = {
            "General": general,
            "Signal": signal,
            "Acquisition_instrument": {
                "Laser": laser,
                "Detector": detector,
                "Spectrometer": spectrometer,
            },
        }
        _remove_none_from_dict(metadata)
        return metadata

    def _parse_TEXT(self):
        if not self._check_block_exists("TEXT_0"):
            return
        pos, block_size = self._block_info["TEXT_0"]
        self._file_obj.seek(pos)
        text = self.__read_utf8(block_size - 16)
        self.original_metadata.update({"TEXT_0": text})

    def _get_WHTL(self):
        if not self._check_block_exists("WHTL_0"):
            return None
        pos, size = self._block_info["WHTL_0"]
        jpeg_header = 0x10
        self._file_obj.seek(pos)
        img_bytes = self._file_obj.read(size - jpeg_header)
        img = BytesIO(img_bytes)

        ## extract and parse EXIF tags
        if PIL_installed:
            from rsciio.utils.image import _parse_axes_from_metadata, _parse_exif_tags

            pil_img = Image.open(img)
            original_metadata = {}
            data = rgb_tools.regular_array2rgbx(np.array(pil_img))
            original_metadata["exif_tags"] = _parse_exif_tags(pil_img)
            axes = _parse_axes_from_metadata(original_metadata["exif_tags"], data.shape)
            metadata = {
                "General": {"original_filename": os.path.split(self._filename)[1]},
                "Signal": {"signal_type": ""},
            }

            map_md = self.original_metadata.get("WMAP_0")
            if map_md is not None:
                width = map_md["scale_xyz"][0] * map_md["size_xyz"][0]
                length = map_md["scale_xyz"][1] * map_md["size_xyz"][1]
                offset = (
                    np.array(map_md["offset_xyz"][:2]) + np.array([width, length]) / 2
                )

                marker_dict = {
                    "class": "Rectangles",
                    "name": "Map",
                    "plot_on_signal": True,
                    "kwargs": {
                        "offsets": offset,
                        "widths": width,
                        "heights": length,
                        "color": ("red",),
                        "facecolor": "none",
                    },
                }

                metadata["Markers"] = {"Map": marker_dict}

            return {
                "axes": axes,
                "data": data,
                "metadata": metadata,
                "original_metadata": original_metadata,
            }
        else:  # pragma: no cover
            # Explicit return for readibility
            return None


def file_reader(
    filename,
    lazy=False,
    use_uniform_signal_axis=False,
    load_unmatched_metadata=False,
):
    """
    Read Renishaw's ``.wdf`` file. In case of mapping data, the image area will
    be returned with a marker showing the mapped area.

    Parameters
    ----------
    %s
    %s
    use_uniform_signal_axis : bool, default=False
        Can be specified to choose between non-uniform or uniform signal axes.
        If `True`, the ``scale`` attribute is calculated from the average delta
        along the signal axis and a warning is raised in case the delta varies
        by more than 1%%.
    load_unmatched_metadata : bool, default=False
        Some of the original_metadata cannot be matched (no key, just value).
        Part of this is a VisualBasic-Script used for data acquisition (~230kB),
        which blows up the size of ``original_metadata``. If this option is set to
        `True`, this metadata will be included and can be accessed by
        ``s.original_metadata.UNMATCHED``,
        otherwise the ``UNMATCHED`` tag will not exist.

    %s
    """
    if lazy is not False:
        raise NotImplementedError("Lazy loading is not supported.")

    filesize = Path(filename).stat().st_size
    original_filename = Path(filename).name
    dictionary = {}
    with open(str(filename), "rb") as f:
        wdf = WDFReader(
            f,
            filename=original_filename,
            use_uniform_signal_axis=use_uniform_signal_axis,
            load_unmatched_metadata=load_unmatched_metadata,
        )
        wdf.read_file(filesize)

        dictionary["data"] = wdf.data
        dictionary["axes"] = wdf.axes
        dictionary["metadata"] = deepcopy(wdf.metadata)
        dictionary["original_metadata"] = deepcopy(wdf.original_metadata)

        image_dict = wdf._get_WHTL()

    dict_list = [dictionary]
    if image_dict is not None:
        dict_list.append(image_dict)

    return dict_list


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)
