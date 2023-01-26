# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

# TODO: manage licensing and acknowledge previous contributions
# upon this reader is based on

# Renishaw wdf Raman spectroscopy file reader
# Code inspired by Henderson, Alex DOI:10.5281/zenodo.495477

# TODO: general:
#   - check axes order (snake pattern?)
#   - check time axis
#   - remove debug flag once unmatched keys values are understood
#   - how to handle WHTL cropping
#       - convert IFDRational in hyperspy?
#       - show cropped image?
#   - how to incorporate start points for linescan outside of original_metadata?
#       - currently only saved in WMAP
#   - remove UID from blocknames for more consistent names?

# known limitations/problems:
#   - cannot parse BKXL-Block
#   - many blocks exist according to gwyddion that are not covered by testfiles
#       -> not parsed
#   - unmatched keys/values for pset metadata blocks
#   - MapType is not used -> snake pattern read incorrectly for example
#   - quantity name is always set to Intensity (not extracted from file)
#   - many DataTypes are not considered for axes, only the following are used
#       - XLST for signal
#       - X,Y,Z for nav
#       - especially no time series
#   - ScanType not used for anything
#   - unclear what MAP contains
#   - metadata mapping extendable (especially integration time)

import logging
import datetime
import importlib.util
from pathlib import Path
from enum import IntEnum, Enum
from copy import deepcopy
from io import BytesIO

import numpy as np
from numpy.polynomial.polynomial import polyfit

from rsciio.docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC

_logger = logging.getLogger(__name__)
_logger.setLevel(10)

try:
    from PIL import Image
except ImportError:
    PIL_installed = False
    _logger.warning("Pillow not installed. Cannot load whitelight image into metadata")
else:
    PIL_installed = True


def _convert_float(input):
    """Handle None-values when converting strings to float."""
    if input is None:
        return None  # pragma: no cover
    else:
        return float(input)


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


class MetadataTypeSingle(Enum):
    char = "c"
    uint8 = "?"
    int16 = "s"
    int32 = "i"
    int64 = "w"
    float = "r"
    double = "q"
    windows_filetime = "t"


MetadataLengthSingle = {
    "len_char": 1,
    "len_uint8": 1,
    "len_int16": 2,
    "len_int32": 4,
    "len_int64": 8,
    "len_float": 4,
    "len_double": 8,
    "len_windows_filetime": 8,
}


class MetadataTypeMulti(Enum):
    string = "u"
    nested = "p"
    key = "k"
    binary = "b"


class MetadataFlags(IntEnum):
    normal = 0
    compressed = 64
    array = 128


## < specifies little endian
## no enum, because double entries
TypeNames = {
    "char": "<i1",  # same as int8, converted in __read_numeric
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


class MeasurementType(IntEnum):
    Unspecified = 0
    Single = 1
    Series = 2
    Mapping = 3


## testfiles: 2, 3, 4, 5, 8 missing
class ScanType(IntEnum):
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
## linescan is identified correctly (128),
## but the flag is not necessary to identify a linescan
## TODO: what is linefocus?
## cases column_major vs alternating (vs row_major?) need to be covered
class MapType(IntEnum):
    randompoints = 1  # rectangle
    column_major = 2  # x then y
    alternating = 4  # snake pattern
    linefocus_mapping = 8  # ?
    inverted_rows = 16  # rows collected right to left (negative scale)
    inverted_columns = 32  # columns collected bottom to top (negative scale)
    surface_profile = 64  # Z data is non-regular (gwyddion)?
    xyline = 128  # linescan -> x always contains data, y is 0


# TODO: wavelength is named wavenumber in py-wdf-reader, gwyddion
class UnitType(IntEnum):
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
class DataType(IntEnum):
    Arbitrary = 0
    Frequency = 1  # TODO: DEPRECATED according to gwyddion (switched with spectral)
    Intensity = 2
    Spatial_X = 3
    Spatial_Y = 4
    Spatial_Z = 5
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
    Spectral = 19
    Mp_Well_Spatial_X = 22
    Mp_Well_Spatial_Y = 23
    Mp_LocationIndex = 24
    Mp_WellReference = 25
    EndMarker = 26
    ExposureTime = (
        27  # different from gwyddion, see PR#39 streamhr rapide mode (py-wdf-reader)
    )


# for wthl image
class ExifTags(IntEnum):
    # Standard EXIF TAGS
    ImageDescription = 0x10E  # 270
    Make = 0x10F  # 271
    ExifOffset = 0x8769  # 34665
    FocalPlaneXResolution = 0xA20E  # 41486
    FocalPlaneYResolution = 0xA20F  # 41487
    FocalPlaneResolutionUnit = 0xA210  # 41488
    # Customized EXIF TAGS from Renishaw
    FocalPlaneXYOrigins = 0xFEA0  # 65184
    FieldOfViewXY = 0xFEA1  # 65185
    Unknown = 0xFEA2  # 65186


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

    def __init__(self, f, filesize, filename, use_uniform_signal_axis, debug):
        self._file_obj = f
        self._filename = filename
        self._use_uniform_signal_axis = use_uniform_signal_axis
        self._debug = debug
        self.original_metadata = {}
        self._unmatched_metadata = {}
        self._reverse_signal = False

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
        self._parse_metadata("WXDM_0")
        ## WXDA has extra 1025 bytes at the end (newline: n\x00\x00...)
        self._parse_metadata("WXDA_0")
        self._parse_metadata("ZLDC_0")
        self._parse_metadata("WARP_0")
        self._parse_metadata("WARP_1")
        self._parse_MAP("MAP_0")
        self._parse_MAP("MAP_1")
        self._parse_TEXT()
        self._parse_WHTL()

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
        if self._debug:
            _remove_none_from_dict(self._unmatched_metadata)
            self.original_metadata.update({"UNMATCHED": self._unmatched_metadata})

    def __read_numeric(self, type, size=1, ret_array=False, convert=True):
        if type not in TypeNames.keys():
            raise ValueError(
                f"Trying to read number with unknown dataformat.\n"
                f"Input: {type}\n"
                f"Supported formats: {list(TypeNames.keys())}"
            )
        data = np.fromfile(self._file_obj, dtype=TypeNames[type], count=size)
        ## convert unsigned ints to ints
        ## because int + uint = float -> problems with indexing
        if type in ["uint8", "uint16", "uint32", "uint64"] and convert:
            dt = "<i8"
            data = data.astype(np.dtype(dt))
        elif type == "windows_filetime" and convert:
            data = list(map(convert_windowstime_to_datetime, data))
        elif type == "char" and convert:
            data = list(map(chr, data))
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
        ## >4 instead of >0, because key without result meaningless
        ## this case happens for MAP (see _parse_MAP)
        while remaining > 4:
            type = self.__read_utf8(0x1)
            flag = self.__read_numeric("uint8")
            key = str(self.__read_numeric("uint16"))
            remaining -= 4
            try:
                entry, num_bytes = self._pset_switch_read_on_flag(
                    flag, type, f"{id}_{remaining}"
                )
            except RuntimeError:
                return None
            else:
                if type == "k":
                    key_dict[key] = entry
                else:
                    value_dict[key] = entry
                remaining -= num_bytes
        return self._pset_match_keys_and_values(id, key_dict, value_dict)

    def _pset_match_keys_and_values(self, id, key_dict, value_dict):
        result = {}
        if self._debug:
            print(f"{id}(keys, {len(list(key_dict))}): {list(key_dict)}")
            print(f"{id}(values, {len(list(value_dict))}): {list(value_dict)}")
            print()
        for key in list(key_dict.keys()):
            ## keep mismatched keys for debugging
            try:
                val = value_dict.pop(key)
            except KeyError:
                pass  ## only occurs for WXDM in testfiles
            else:
                result[key_dict.pop(key)] = val
        ## TODO: Why are there unmatched keys/values
        if self._debug and (len(key_dict) != 0 or len(value_dict) != 0):
            self._unmatched_metadata.update(
                {id: {"keys": key_dict, "values": value_dict}}
            )
        return result

    def _pset_switch_read_on_flag(self, flag, type, id):
        return {
            MetadataFlags.normal: self._pset_read_normal_flag,
            MetadataFlags.array: self._pset_read_array_flag,
            MetadataFlags.compressed: self._pset_read_compressed_flag,
        }.get(flag, self._pset_error_msg_invalid_flag)(type=type, id=id, flag=flag)

    def _pset_error_msg_invalid_flag(self, id, flag, **kwargs):
        _logger.error(
            f"Invalid metadata flag ({flag}) encountered while parsing {id}."
            f"Cannot read metadata from this Block."
        )
        raise RuntimeError

    def _pset_read_normal_flag(self, type, id, **kwargs):
        result, num_bytes = self._pset_read_entry(type, 1, id)
        return result, num_bytes

    def _pset_read_array_flag(self, type, id, **kwargs):
        if type not in MetadataTypeSingle._value2member_map_:
            _logger.debug(f"array flag, but not single dtype: {type}")
        size = self.__read_numeric("uint32")
        result, num_bytes = self._pset_read_entry(type, size, id)
        return result, num_bytes + 4  ## +4, because of size read

    def _pset_read_compressed_flag(self, type, **kwargs):
        if type != "u":
            _logger.debug(f"compressed flag, but no string ({type})")
        size = self.__read_numeric("uint32")
        self._file_obj.seek(size, 1)  # move fp from current position
        result = None
        num_bytes = size + 4  ## +4, because of size read
        return result, num_bytes

    def _pset_read_entry(self, type, size, id):
        if type in MetadataTypeSingle._value2member_map_:
            type_str = MetadataTypeSingle(type).name
            type_len = MetadataLengthSingle[f"len_{type_str}"]
            result = self.__read_numeric(type_str, size=size)
            return result, type_len
        elif type in MetadataTypeMulti._value2member_map_:
            length = self.__read_numeric("uint32")
            result = self._pset_read_multitype_entry_helper(type, length, id)
            return result, length + 4  ## reading the length itself equals 4 bytes
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

    ## `MAP ` contains more information than just PSET
    ## TODO: what is this extra data? max peak? (MAP slightly off, MAP1 different)
    def _parse_MAP(self, id):
        if not self._check_block_exists(id):
            return
        pset_size = self._get_psetsize(id, warning=False)
        _, block_size = self._block_info[id]
        metadata = self._pset_read_metadata(pset_size - 4, id)
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
        ## warning when file_status_error_code nonzero?
        ## mulitple accumulations -> average or sum?
        self._file_obj.seek(pos)
        header["flags"] = self.__read_numeric("uint64")
        header["uuid"] = f"{self.__read_numeric('uint32', convert=False)}"
        for _ in range(3):
            header["uuid"] += f"-{self.__read_numeric('uint32', convert=False)}"
        unused1 = self.__read_numeric("uint32", size=3)
        header["ntracks"] = self.__read_numeric("uint32")
        header["file_status_error_code"] = self.__read_numeric("uint32")
        result["points_per_spectrum"] = self.__read_numeric("uint32")
        header["capacity"] = self.__read_numeric("uint64")
        result["num_spectra"] = self.__read_numeric("uint64")
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
        unused2 = self.__read_numeric("uint64", size=6)
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
        unit = str(UnitType(self.__read_numeric("uint32")))
        axis_name = self._convert_signal_axis_name(type, unit)
        data = self.__read_numeric("float", size=size)
        return axis_name, unit, data

    @staticmethod
    def _convert_signal_axis_name(type, unit):
        if type == "Frequency":
            if unit == "1/cm":
                type = "Wavenumber"
            elif unit in ["nm", "µm", "m", "mm"]:
                type = "Wavelength"
        return type

    def _parse_XLST(self):
        if not self._check_block_exists("XLST_0"):
            raise RuntimeError(
                "File contains no information on signal axis (XLST-Block missing)."
            )
        name, unit, data = self._parse_XLST_or_YLST("X", self._points_per_spectrum)
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

            # TODO: maybe need to add more types here (R, Theta, Phi, Time)
            if dtype in [
                DataType.Spatial_X.name,
                DataType.Spatial_Y.name,
                DataType.Spatial_Z.name,
            ]:
                orgn_nav[f"{dtype[-1]}-axis"] = ax_tmp_dict
            else:
                orgn_metadata[dtype] = ax_tmp_dict
        self.original_metadata.update({"ORGN_0": orgn_metadata})
        return orgn_nav

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

        flag = self.__read_numeric("uint32")
        unused = self.__read_numeric("uint32")
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
        for idx, letter in enumerate(["X", "Y"]):
            key_name = letter + "-axis"
            axis_tmp = {
                "name": letter,
                "offset": wmap_dict["offset_xyz"][idx],
                "scale": wmap_dict["scale_xyz"][idx],
                "size": wmap_dict["size_xyz"][idx],
                "navigate": True,
                "units": units,
            }
            result[key_name] = axis_tmp

        # TODO: differentiate between more map_modes/flags
        if wmap_dict["flag"] == MapType.xyline:
            result = self._set_wmap_nav_linexy(result["X-axis"], result["Y-axis"])
        return result

    def _set_wmap_nav_linexy(self, x_axis, y_axis):
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
        for axis in orgn_data.keys():
            del nav_dict[axis]["annotation"]
            nav_dict[axis]["navigate"] = True
            data = np.unique(nav_dict[axis].pop("data"))
            nav_dict[axis]["size"] = data.size
            nav_dict[axis]["offset"] = data[0]
            nav_dict[axis]["scale"] = data[1] - data[0]
            nav_dict[axis]["name"] = axis[0]
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
            _logger.warning("Series expected, but no (X, Y, Z) data.")
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
            nav_dict["Y-axis"]["index_in_array"] = 0
            nav_dict["X-axis"]["index_in_array"] = 1
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
        general["original_filename"] = self._filename
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
                "Cannot find package lumispy, using BaseSignal as signal_type."
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
        if len(laser_original_md) != 1:
            return laser
        laser_wavenumber = next(iter(laser_original_md.values())).get("Wavenumber")
        laser["wavelength"] = 1e7 / _convert_float(laser_wavenumber)
        return laser

    def _map_spectrometer_md(self):
        spectrometer = {"Grating": {}}
        gratings_original_md = self.original_metadata.get("WXCS_0", {}).get("Gratings")
        if gratings_original_md is None:
            return None
        if len(gratings_original_md) != 1:
            return spectrometer
        skip_lvl_dict = next(iter(gratings_original_md.values()))
        for v in skip_lvl_dict.values():
            if isinstance(v, dict):
                groove_density = v.get("Groove Density (lines/mm)")
        spectrometer["Grating"]["groove_density"] = _convert_float(groove_density)
        return spectrometer

    def _map_detector_md(self):
        detector = {}
        detector["detector_type"] = "CCD"
        ccd_original_metadata = self.original_metadata.get("WXCS_0", {}).get("CCD")
        if ccd_original_metadata is None:
            return None
        detector["model"] = ccd_original_metadata.get("CCD")
        detector["temperature"] = _convert_float(
            ccd_original_metadata.get("Target temperature")
        )
        detector["frames"] = self.original_metadata.get("WDF1_1", {}).get(
            "accumulations_per_spectrum"
        )
        return detector

    def map_metadata(self):
        general = self._map_general_md()
        signal = self._map_signal_md()
        detector = self._map_detector_md()
        laser = self._map_laser_md()
        spectrometer = self._map_spectrometer_md()

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

    def _parse_WHTL(self):
        if not self._check_block_exists("WHTL_0"):
            return
        pos, size = self._block_info["WHTL_0"]
        jpeg_header = 0x10
        self._file_obj.seek(pos)
        img_bytes = self._file_obj.read(size - jpeg_header)
        img = BytesIO(img_bytes)
        whtl_metadata = {"image": img}

        ## extract EXIF tags and store them in metadata
        if PIL_installed:
            pil_img = Image.open(img)
            # missing header keys when Pillow >= 8.2.0 -> does not flatten IFD anymore
            # see https://pillow.readthedocs.io/en/stable/releasenotes/8.2.0.html#image-getexif-exif-and-gps-ifd
            # Use fall-back _getexif method instead
            exif_header = dict(pil_img._getexif())
            try:
                w_px = exif_header[ExifTags.FocalPlaneXResolution]
                h_px = exif_header[ExifTags.FocalPlaneYResolution]
                x0_img_micro, y0_img_micro = exif_header[ExifTags.FocalPlaneXYOrigins]
                img_dimension_unit = str(
                    UnitType(exif_header[ExifTags.FocalPlaneResolutionUnit])
                )
                img_description = exif_header[ExifTags.ImageDescription]
                make = exif_header[ExifTags.Make]
                unknown = exif_header[ExifTags.Unknown]
                fov_xy = exif_header[ExifTags.FieldOfViewXY]
            except KeyError:
                _logger.debug("Some keys in white light image header cannot be read!")
            else:
                whtl_metadata["FocalPlaneResolutionUnit"] = img_dimension_unit
                whtl_metadata["FocalPlaneXResolution"] = w_px
                whtl_metadata["FocalPlaneYResolution"] = h_px
                whtl_metadata["FocalPlaneXOrigin"] = x0_img_micro
                whtl_metadata["FocalPlaneYOrigin"] = y0_img_micro
                whtl_metadata["ImageDescription"] = img_description
                whtl_metadata["Make"] = make
                whtl_metadata["Unknown"] = unknown
                whtl_metadata["FieldOfViewXY"] = fov_xy

        self.original_metadata.update({"WHTL_0": whtl_metadata})


def file_reader(
    filename, lazy=False, use_uniform_signal_axis=True, debug=False, **kwds
):
    """Reads Renishaw's ``.wdf`` file.

    Parameters
    ----------
    %s
    %s
    use_uniform_signal_axis: bool, default=False
        Can be specified to choose between non-uniform or uniform signal axes.
        If `True`, the ``scale`` attribute is calculated from the average delta
        along the signal axis and a warning is raised in case the delta varies
        by more than 1%%.

    %s
    """
    filesize = Path(filename).stat().st_size
    original_filename = Path(filename).name
    dictionary = {}
    with open(str(filename), "rb") as f:
        wdf = WDFReader(
            f,
            filesize=filesize,
            filename=original_filename,
            use_uniform_signal_axis=use_uniform_signal_axis,
            debug=debug,
        )

        dictionary["data"] = wdf.data
        dictionary["axes"] = wdf.axes
        dictionary["metadata"] = deepcopy(wdf.metadata)
        dictionary["original_metadata"] = deepcopy(wdf.original_metadata)

    return [
        dictionary,
    ]


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)
