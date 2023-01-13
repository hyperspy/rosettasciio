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
#   - check axes order (negative scale?, snake pattern?)
#   - check time axis
#   - revert signal axis + spectra when scale is negative?
#   - metadata mapping

import logging
import datetime
from pathlib import Path
from enum import IntEnum, Enum
from copy import deepcopy

import numpy as np
from numpy.polynomial.polynomial import polyfit

_logger = logging.getLogger(__name__)

try:
    import PIL
except ImportError:
    PIL = None
    _logger.warning("Pillow not installed. Cannot load whitelight image into metadata")
else:
    from PIL import Image, ImageDraw
    from PIL.TiffImagePlugin import IFDRational
    from io import BytesIO

    def rational2float(v):
        """Pillow<7.2.0 returns tuple, Pillow>=7.2.0 returns IFDRational"""
        if not isinstance(v, IFDRational):
            return v[0] / v[1]
        return float(v)


def convert_windowstime_to_datetime(wt):
    base = datetime.datetime(1601, 1, 1, 0, 0, 0, 0)
    delta = datetime.timedelta(microseconds=wt / 10)
    return (base + delta).isoformat()


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


class UnitType(IntEnum):
    arbitrary = 0
    raman_shift = 1  # cm^-1 by default
    wavenumber = 2  # nm TODO: why call it wavenumber then?
    nanometer = 3
    electron_volt = 4
    micrometer = 5  # same for EXIF units TODO: what does that mean?
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
        """Rewrite the unit name output"""
        unit_str = dict(
            arbitrary="",
            raman_shift="1/cm",  # cm^-1 by default
            wavenumber="nm",  # nm
            nanometer="nm",
            electron_volt="eV",
            micrometer="µm",  # same for EXIF units
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


# TODO: spectral/frequency switched in gwyddion
# used in ORGN/XLST
class DataType(IntEnum):
    Arbitrary = 0
    Frequency = 1  # DEPRECATED according to gwyddion
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

    Attributes:
    count (int) : Numbers of experiments (same type), can be smaller than capacity
    points_per_spectrum (int): Should be identical to xlist_length
    data_origin_count (int) : Number of rows in data origin list
    capacity (int) : Max number of spectra
    accumulation_count (int) : Single or multiple measurements
    block_info (dict) : contains information about all located blocks
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

    def __init__(self, f, filesize, use_uniform_signal_axis):
        self.file_obj = f
        self.original_metadata = {}
        self.metadata = {}

        self.locate_all_blocks(filesize)
        header_data = self._parse_WDF1()  # this needs to be parsed first,
        ## because it contains sizes for ORGN, DATA (reshape), XLST, YLST

        ## parse metadata blocks
        self._parse_YLST(header_data["YLST_length"])
        self._parse_metadata("WXIS")
        self._parse_metadata("WXCS")
        self._parse_metadata("WXDM")
        # WXDA has extra 1025 bytes at the end (newline: n\x00\x00...)
        self._parse_metadata("WXDA")
        self._parse_metadata("ZLDC")
        self._parse_metadata("WARP")
        self._parse_metadata("WARP1")
        self._parse_MAP("MAP ")
        self._parse_MAP("MAP 1")
        self._parse_TEXT()
        self._parse_WHTL()

        ## parse blocks with axes/data information
        signal_dict = self._parse_XLST(
            header_data["XLST_length"], use_uniform_signal_axis
        )
        nav_orgn = self._parse_ORGN(
            header_orgn_count=header_data["origin_count"],
            ncollected_spectra=header_data["count"],
        )
        nav_wmap = self._parse_WMAP()

        nav_dict = self._set_nav_axes(
            nav_orgn, nav_wmap, header_data["measurement_type"]
        )
        self.axes = self._set_axes(signal_dict, nav_dict)
        self.data = self._parse_DATA(
            size=header_data["count"] * header_data["points_per_spectrum"]
        )
        self._reshape_data(header_data["count"], use_uniform_signal_axis)

    def __read_numeric(self, type, size=1, ret_array=False, convert=True):
        """Reads the file_object at the current position as the specified type.
        Supported types: see TypeNames
        """
        if type not in TypeNames.keys():
            raise ValueError(
                f"Trying to read number with unknown dataformat.\n"
                f"Input: {type}\n"
                f"Supported formats: {list(TypeNames.keys())}"
            )
        data = np.fromfile(self.file_obj, dtype=TypeNames[type], count=size)
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
        """Reads the file_object at the current position as utf8 of length size."""
        ## TODO check if removing .replace("\x00", "") makes any difference
        return self.file_obj.read(size).decode("utf8")

    def __locate_single_block(self, pos):
        """Get block information starting at pos."""
        self.file_obj.seek(pos)
        block_name = self.__read_utf8(0x4)
        if block_name not in self._known_blocks:
            _logger.debug(f"Unknown Block {block_name} encountered.")
        block_uid = self.__read_numeric("uint32")
        block_size = self.__read_numeric("uint64")
        return block_name, block_uid, block_size

    def locate_all_blocks(self, filesize):
        """Get information for all data blocks and store them inside self.block_info"""
        self.block_info = {}
        block_header_size = 16
        curpos = 0
        _logger.debug("ID     UID CURPOS   SIZE")
        _logger.debug("--------------------------------------")
        block_size = 0
        while True:
            curpos += block_size
            if curpos == filesize:
                break
            elif curpos > filesize:
                _logger.warning("Missing characters at the file end.")
                break

            try:
                block_name, block_uid, block_size = self.__locate_single_block(curpos)
            except EOFError:
                _logger.warning("Unexpected extra characters at the file end.")
                break
            else:
                _logger.debug(f"{block_name}   {block_uid}   {curpos:<{9}}{block_size}")
                if block_name in self.block_info.keys():
                    # both blocks should only differ in UID
                    if self._check_block_equality(
                        block_name, block_size, curpos + block_header_size
                    ):
                        continue
                    else:
                        block_name += str(block_uid)

                self.block_info[block_name] = (
                    curpos + block_header_size,
                    block_size,
                )
        _logger.debug("--------------------------------------")
        _logger.debug(f"filesize:    {filesize}")
        _logger.debug(f"parsed size: {curpos}")

    def _check_block_equality(self, name, size, pos):
        _logger.debug(f"Multiple {name} entries in the file.")
        pos1, size1 = self.block_info[name]
        self.file_obj.seek(pos1)
        read1 = self.file_obj.read(size1 - 16)
        self.file_obj.seek(pos)
        read2 = self.file_obj.read(size - 16)
        if size1 != size:
            _logger.debug(
                f"{name} exists multiple times in the file with different sizes."
            )
            return False
        elif read1 != read2:
            _logger.debug(
                f"{name} exists multiple times in the file with different content."
            )
            return False
        else:
            return True

    def _check_block_exists(self, block_name):
        if block_name in self.block_info.keys():
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
            _logger.debug(
                f"{name} is not read completely ({expected_size - actual_size} bytes missing)"
            )

    ## `MAP ` contains more information than just PSET,
    ## TODO: what is this extra data? max peak?
    def _parse_MAP(self, id):
        if not self._check_block_exists(id):
            return
        pset_size = self._get_psetsize(id, warning=False)
        _, block_size = self.block_info[id]
        metadata = self.__read_pset_metadata(pset_size - 4)
        npoints = self.__read_numeric("uint64")
        ## 8 bytes from npoints + 4*npoints data
        self._check_block_size(
            id, "Metadata", block_size - 16, pset_size + 8 + 8 + 4 * npoints
        )
        metadata["npoints"] = npoints
        metadata["data"] = self.__read_numeric("float", npoints)
        self.original_metadata.update({id.replace(" ", ""): metadata})

    def _parse_metadata(self, id):
        _logger.debug(f"Parsing {id}")
        pset_size = self._get_psetsize(id)
        metadata = self.__read_pset_metadata(pset_size)
        if metadata is None:
            _logger.warning(f"Invalid key in {id}")
        self.original_metadata.update({id: metadata})

    def _get_psetsize(self, id, warning=True):
        if not self._check_block_exists(id):
            return 0
        stream_is_pset = int(0x54455350)
        pos, block_size = self.block_info[id]
        self.file_obj.seek(pos)
        is_pset = self.__read_numeric("uint32")
        pset_size = self.__read_numeric("uint32")
        if is_pset != stream_is_pset:
            _logger.debug(f"No PSET found in {id} -> cannot extract metadata.")
            return 0
        if warning:
            self._check_block_size(id, "Metadata", block_size - 16, pset_size + 8)
        return pset_size

    def __read_pset_entry(self, type, size=1):
        if type in MetadataTypeSingle._value2member_map_:
            type_str = MetadataTypeSingle(type).name
            type_len = MetadataLengthSingle[f"len_{type_str}"]
            result = self.__read_numeric(type_str, size=size)
            # _logger.debug(f"{type} = {result}")
            return result, type_len
        elif type in MetadataTypeMulti._value2member_map_:
            length = self.__read_numeric("uint32")
            # _logger.debug(f"type length: {length}")
            ## TODO: read binary? does not occur in test files
            if type == "b":
                # _logger.debug("BINARY ENCOUNTERED")
                result = None
                self.file_obj.read(length)
            elif type in ["u", "k"]:
                result = self.__read_utf8(length)
                # _logger.debug(result)
            elif type == "p":
                result = self.__read_pset_metadata(length)
            return result, length + 4  ## reading the length itself equals 4 bytes
        else:
            raise ValueError(f"Unknown type: {type}")

    def __read_pset_metadata(self, length):
        key_dict = {}
        value_dict = {}
        remaining = length
        ## >4 instead of >0, because key without result meaningless
        ## this case happens for MAP (see _parse_MAP)
        while remaining > 4:
            type = self.__read_utf8(0x1)
            flag = self.__read_numeric("uint8")
            key = self.__read_numeric("uint16")
            remaining -= 4
            # _logger.debug(f"type: {type}, flag: {flag}, key: {key}")
            if flag == MetadataFlags.normal:
                size = 1
                result, num_bytes = self.__read_pset_entry(type, size)
            elif flag == MetadataFlags.array:
                if type not in MetadataTypeSingle._value2member_map_:
                    _logger.debug(f"array flag, but not single dtype: {type}")
                size = self.__read_numeric("uint32")
                remaining -= 4
                result, num_bytes = self.__read_pset_entry(type, size)
            elif flag == MetadataFlags.compressed:
                if type != "u":
                    _logger.debug(f"compressed flag, but no string ({type})")
                size = self.__read_numeric("uint32")
                remaining -= 4
                self.file_obj.seek(size, 1)  # move fp from current position
                result = None
                num_bytes = size
            else:
                _logger.warning(
                    f"Invalid flag {flag} encountered when reading metadata."
                )
                return None
            if type == "k":
                key_dict[key] = result
            else:
                value_dict[key] = result
            remaining -= num_bytes
        retDict = {}
        for key in list(key_dict):
            try:
                retDict[key_dict.pop(key)] = value_dict.pop(key)
            except KeyError:
                pass
        ## TODO: Why are there unmatched keys/values?
        # if len(key_dict) != 0:
        # _logger.debug(f"Unmatched keys: {key_dict}")
        # if len(value_dict) != 0:
        # _logger.debug(f"Unmatched values: {value_dict}")
        return retDict

    def _parse_WDF1(self):
        header_metadata = {}
        return_metadata = {}
        self.file_obj.seek(0)
        pos, size_block = self.block_info["WDF1"]
        if size_block != 512:
            _logger.warning("Unexpected header size. File might be invalid.")
        if pos != 16:
            _logger.warning("Unexpected start of file. File might be invalid.")
        self.file_obj.seek(pos)
        header_metadata["flags"] = self.__read_numeric("uint64")
        header_metadata["uuid"] = f"{self.__read_numeric('uint32', convert=False)}"
        for _ in range(3):
            header_metadata[
                "uuid"
            ] += f"-{self.__read_numeric('uint32', convert=False)}"
        unused1 = self.__read_numeric("uint32", size=3)
        ## TODO: what is ntracks, status, accumulation_count
        header_metadata["ntracks"] = self.__read_numeric("uint32")
        header_metadata["status"] = self.__read_numeric("uint32")
        return_metadata["points_per_spectrum"] = self.__read_numeric("uint32")
        header_metadata["capacity"] = self.__read_numeric("uint64")
        return_metadata["count"] = self.__read_numeric("uint64")
        return_metadata["accumulation_count"] = self.__read_numeric("uint32")
        return_metadata["YLST_length"] = self.__read_numeric("uint32")
        return_metadata["XLST_length"] = self.__read_numeric("uint32")
        return_metadata["origin_count"] = self.__read_numeric("uint32")
        header_metadata["app_name"] = self.__read_utf8(24)  # must be "WiRE"
        header_metadata[
            "app_version"
        ] = f"{self.__read_numeric('uint16', convert=False)}"
        for _ in range(3):
            header_metadata[
                "app_version"
            ] += f"-{self.__read_numeric('uint16', convert=False)}"
        header_metadata["scan_type"] = ScanType(self.__read_numeric("uint32")).name
        return_metadata["measurement_type"] = MeasurementType(
            self.__read_numeric("uint32")
        ).name
        time_start_wt = self.__read_numeric("uint64")
        header_metadata["time_start"] = convert_windowstime_to_datetime(time_start_wt)
        time_end_wt = self.__read_numeric("uint64")
        header_metadata["time_end"] = convert_windowstime_to_datetime(time_end_wt)
        header_metadata["spectral_unit"] = UnitType(self.__read_numeric("uint32")).name
        header_metadata["laser_wavenumber"] = self.__read_numeric("float")
        unused2 = self.__read_numeric("uint64", size=6)
        header_metadata["username"] = self.__read_utf8(32)
        header_metadata["title"] = self.__read_utf8(160)

        header_metadata.update(return_metadata)
        self.original_metadata.update({"WDF1": header_metadata})
        if return_metadata["count"] != header_metadata["capacity"]:
            _logger.warning(
                f"Unfinished measurement."
                f"The number of measured spectra ({header_metadata['count']}) is different"
                f"from the set number of spectra ({header_metadata['capacity']})."
                "It is tried to still use the data of the measured data."
            )
        if return_metadata["points_per_spectrum"] != return_metadata["XLST_length"]:
            raise IOError("File contains ambiguous signal axis sizes.")
        return return_metadata

    def _parse_TEXT(self):
        if not self._check_block_exists("TEXT"):
            return
        pos, block_size = self.block_info["TEXT"]
        self.file_obj.seek(pos)
        text = self.__read_utf8(block_size - 16)
        self.original_metadata.update({"TEXT": text})

    def _parse_DATA(self, size):
        """Get information from DATA block"""
        if not self._check_block_exists("DATA"):
            raise IOError("File does not contain data (DATA-Block missing).")
        pos, block_size = self.block_info["DATA"]
        self._check_block_size("DATA", "Data", block_size - 16, 4 * size)
        self.file_obj.seek(pos)
        return self.__read_numeric("float", size=size)

    ## TODO: signal seems to be in reverse order, maybe revert this?
    def _parse_XLST(self, size, use_uniform_signal_axis=True):
        if not self._check_block_exists("XLST"):
            raise IOError(
                "File contains no information on signal axis (XLST-Block missing)."
            )
        type, unit, data = self._parse_XLST_or_YLST("X", size)
        signal_dict = {}
        signal_dict["size"] = size
        signal_dict["navigate"] = False
        signal_dict["name"] = type
        signal_dict["units"] = unit
        if use_uniform_signal_axis:
            offset, scale = polyfit(np.arange(size), data, deg=1)
            scale_compare = 100 * np.max(np.abs(np.diff(data) - scale) / scale)
            if scale_compare > 1:
                _logger.warning(
                    f"The relative variation of the signal-axis-scale ({scale_compare:.2f}%) exceeds 1%.\n"
                    "                              "
                    "Using a non-uniform-axis is recommended."
                )

            signal_dict["offset"] = offset
            signal_dict["scale"] = scale
        else:
            signal_dict["axis"] = data
        return signal_dict

    def _parse_YLST(self, size):
        if not self._check_block_exists("YLST"):
            return
        type, unit, data = self._parse_XLST_or_YLST("Y", size)
        self.original_metadata["YLST"] = {
            "name": type,
            "units": unit,
            "size": size,
            "data": data,
        }

    def _parse_XLST_or_YLST(self, name, size):
        pos, block_size = self.block_info[name.upper() + "LST"]
        if name.upper() == "X":
            self._check_block_size("XLST", "Signal axis", block_size - 16, 4 * size + 8)
        else:
            self._check_block_size("YLST", "Metadata", block_size - 16, 4 * size + 8)

        self.file_obj.seek(pos)
        type = DataType(self.__read_numeric("uint32")).name
        unit = str(UnitType(self.__read_numeric("uint32")))
        if type == "Frequency":
            if unit == "1/cm":
                type = "Wavenumber"
            elif unit in ["nm", "µm", "m", "mm"]:
                type = "Wavelength"
        data = self.__read_numeric("float", size=size)
        return type, unit, data

    def _parse_ORGN(self, header_orgn_count, ncollected_spectra):
        if not self._check_block_exists("ORGN"):
            return {}
        orgn_nav = {}
        orgn_metadata = {}
        pos, block_size = self.block_info["ORGN"]
        self.file_obj.seek(pos)
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
            4 + origin_count * (24 + 8 * ncollected_spectra),
        )
        for _ in range(origin_count):
            ax_tmp_dict = {}
            ## ignore first bit of dtype read (sometimes 0, sometimes 1 in testfiles)
            dtype = DataType(
                self.__read_numeric("uint32", convert=False) & ~(0b1 << 31)
            ).name
            ax_tmp_dict["units"] = str(UnitType(self.__read_numeric("uint32")))
            ax_tmp_dict["annotation"] = self.__read_utf8(0x10)

            if dtype == DataType.Time.name:
                array = self.__read_numeric(
                    "uint64", size=ncollected_spectra, ret_array=True
                )
                ax_tmp_dict["data"] = array - array[0]
                # [convert_windowstime_to_datetime(array[i]) for i in range(array.size)]
            else:
                ax_tmp_dict["data"] = self.__read_numeric(
                    "double", size=ncollected_spectra, ret_array=True
                )
            # TODO: maybe need to add more types here (R, Theta, Phi, Time)
            if dtype in [
                DataType.Spatial_X.name,
                DataType.Spatial_Y.name,
                DataType.Spatial_Z.name,
            ]:
                orgn_nav[f"{dtype[-1]}-axis"] = ax_tmp_dict
            else:
                orgn_metadata[dtype] = ax_tmp_dict
        self.original_metadata.update({"ORGN": orgn_metadata})
        return orgn_nav

    def _parse_WMAP(self):
        if not self._check_block_exists("WMAP"):
            return {}
        pos, block_size = self.block_info["WMAP"]
        self.file_obj.seek(pos)
        self._check_block_size(
            "WMAP", "Navigation axes", block_size - 16, 4 * (3 + 3 * 3)
        )
        if (block_size - 16) != 4 * (3 + 3 * 3):
            _logger.warning(
                "Unexpected size of WMAP block."
                "Navigation axes may be extracted incorrectly."
            )

        # TODO: check flags/map_mode
        flags = self.__read_numeric("uint32")
        unused = self.__read_numeric("uint32")
        offset_xyz = [self.__read_numeric("float") for _ in range(3)]
        scale_xyz = [self.__read_numeric("float") for _ in range(3)]
        size_xyz = [self.__read_numeric("uint32") for _ in range(3)]
        linefocus_size = self.__read_numeric("uint32")

        metadata = {"linefocus_size": linefocus_size, "flags": flags}
        result = {}
        for idx, letter in enumerate(["X", "Y", "Z"]):
            key_name = letter + "-axis"
            axis_tmp = {
                "name": letter,
                "offset": offset_xyz[idx],
                "scale": scale_xyz[idx],
                "size": size_xyz[idx],
            }
            if size_xyz[idx] <= 1:
                metadata[key_name] = axis_tmp
            else:
                result[key_name] = axis_tmp
        self.original_metadata["WMAP"] = metadata
        return result

    def _set_nav_axes(self, orgn_data, wmap_data, measurement_type):
        if not self._compare_measurement_type_to_ORGN_WMAP(
            orgn_data, wmap_data, measurement_type
        ):
            _logger.warning(
                "Inconsistent MeasurementType and ORGN/WMAP Blocks cannot"
                "Navigation axes may be set incorrectly."
            )
        nav_orgn = self._set_nav_via_ORGN(orgn_data)
        nav_wmap = self._set_nav_via_WMAP(orgn_data, wmap_data)
        if not self._compare_WMAP_ORGN(nav_wmap, nav_orgn):
            _logger.warning(
                "Inconsistent ORGN and WMAP Blocks."
                "Navigation axes may be set incorrectly."
            )
        ## WMAP only exists for mapping
        ## for linescan nav_orgn contains doubled axis -> unclear if X or Y
        if measurement_type == "Mapping":
            return nav_wmap
        else:
            return nav_orgn

    def _set_nav_via_ORGN(self, orgn_data):
        nav_dict = deepcopy(orgn_data)
        for axis in orgn_data.keys():
            del nav_dict[axis]["annotation"]
            nav_dict[axis]["navigate"] = True
            data = nav_dict[axis].pop("data")
            data_unique = data[np.unique(data, return_index=True)[1]]
            if data_unique.size <= 1:
                del nav_dict[axis]
                continue
            nav_dict[axis]["size"] = data_unique.size
            offset_data, scale_data = polyfit(
                np.arange(data_unique.size), data_unique, deg=1
            )
            nav_dict[axis]["offset"] = offset_data
            nav_dict[axis]["scale"] = scale_data
            nav_dict[axis]["name"] = axis[0]
        return nav_dict

    def _set_nav_via_WMAP(self, orgn_data, wmap_data):
        nav_dict = deepcopy(wmap_data)
        for axis in wmap_data.keys():
            if axis not in orgn_data.keys():
                _logger.warning(
                    f"Axis {wmap_data[axis]['name']} exists in WMAP, but not in ORGN."
                    f"Invalid file format. Navigation axes may be loaded incorrectly."
                )
                continue
            nav_dict[axis]["navigate"] = True
            nav_dict[axis]["units"] = orgn_data[axis]["units"]
        return nav_dict

    def _compare_WMAP_ORGN(self, wmap_nav, orgn_nav):
        ## ORGN may contain extra axes (i.e. X and Y doubled for linescan)
        ## -> comparison only in 1 direction
        is_valid = True
        for axis in wmap_nav:
            wmap_ax = wmap_nav[axis]
            if axis not in orgn_nav.keys():
                ## warning message in _set_nav_via_WMAP
                continue
            orgn_ax = orgn_nav[axis]
            close_offset = np.isclose(wmap_ax["offset"], orgn_ax["offset"])
            close_scale = np.isclose(wmap_ax["scale"], orgn_ax["scale"])
            same_size = wmap_ax["size"] == orgn_ax["size"]
            is_valid = is_valid and close_offset and close_scale and same_size
        return is_valid

    def _compare_measurement_type_to_ORGN_WMAP(
        self, orgn_data, wmap_data, measurement_type
    ):
        no_wmap = len(wmap_data) == 0
        no_orgn = len(orgn_data) == 0

        if (not no_wmap) and measurement_type != "Mapping":
            _logger.warning("No Mapping expected, but WMAP Block exists.")
            return False

        if measurement_type == "Mapping":
            if no_wmap:
                _logger.warning("Mapping expected, but no WMAP Block.")
                return False
        elif measurement_type == "Series":
            if no_orgn:
                _logger.warning("Series expected, but no (X, Y, Z) data.")
                return False
        elif measurement_type == "Single":
            if not no_orgn:
                _logger.warning("Spectrum expected, but extra axis present.")
                return False
        elif measurement_type == "Unspecified":
            _logger.warning(
                "Unspecified measurement type. May lead to incorrect results."
            )
        else:
            raise ValueError("Invalid measurement type.")
        return True

    def _set_axes(self, signal_dict, nav_dict):
        if "Y-axis" in nav_dict.keys():
            nav_dict["Y-axis"]["index_in_array"] = 0
            if "X-axis" in nav_dict.keys():
                nav_dict["X-axis"]["index_in_array"] = 1
        else:
            if "X-axis" in nav_dict.keys():
                nav_dict["X-axis"]["index_in_array"] = 0

        ## only appears for Z-scan
        if "Z-axis" in nav_dict.keys():
            nav_dict["Z-axis"]["index_in_array"] = 0

        signal_dict["index_in_array"] = len(nav_dict)

        axes = deepcopy(nav_dict)
        axes["signal_dict"] = deepcopy(signal_dict)
        return sorted(axes.values(), key=lambda item: item["index_in_array"])

    def _parse_WHTL(self):
        if not self._check_block_exists("WHTL"):
            return
        pos, size = self.block_info["WHTL"]
        jpeg_header = 0x10
        self.file_obj.seek(pos)
        img_bytes = self.file_obj.read(size - jpeg_header)
        img = BytesIO(img_bytes)
        whtl_metadata = {}
        if PIL is not None:
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
                # ## TODO: check boundaries, seem small in image
                # w_px = rational2float(w_px)
                # h_px = rational2float(h_px)
                # x0_img_micro = rational2float(x0_img_micro)
                # y0_img_micro = rational2float(y0_img_micro)

                # w_img_px = pil_img.width
                # h_img_px = pil_img.height
                # w_img_micro = rational2float(fov_xy[0])
                # h_img_micro = rational2float(fov_xy[1])
                # self.map_boundaries[dtype] = (min, max) from ORGN block
                # left_micro, right_micro = self.map_boundaries[DataType.Spatial_X.name]
                # bottom_micro, top_micro = self.map_boundaries[DataType.Spatial_Y.name]

                # micro_in_px_x = w_img_px / w_img_micro
                # micro_in_px_y = h_img_px / h_img_micro

                # left_border_px = int(micro_in_px_x * (left_micro - x0_img_micro))
                # right_border_px = int(micro_in_px_x * (right_micro - x0_img_micro))
                # bottom_border_px = int(micro_in_px_y * (bottom_micro - y0_img_micro))
                # top_border_px = int(micro_in_px_y * (top_micro - y0_img_micro))

                # draw = ImageDraw.Draw(pil_img)
                # draw.rectangle(
                #     (
                #         (left_border_px, bottom_border_px),
                #         (right_border_px, top_border_px),
                #     ),
                #     width=2,
                # )
                # pil_img.show()

                whtl_metadata["image"] = pil_img
                whtl_metadata["FocalPlaneResolutionUnit"] = img_dimension_unit
                whtl_metadata["FocalPlaneXResolution"] = w_px
                whtl_metadata["FocalPlaneYResolution"] = h_px
                whtl_metadata["FocalPlaneXOrigin"] = x0_img_micro
                whtl_metadata["FocalPlaneYOrigin"] = y0_img_micro
                whtl_metadata["ImageDescription"] = img_description
                whtl_metadata["Make"] = make
                whtl_metadata["Unknown"] = unknown
                whtl_metadata["FieldOfViewXY"] = fov_xy

        self.original_metadata.update({"WHTL": whtl_metadata})

    def _reshape_data(self, ncollected_spectra, use_uniform_signal_axis):
        if use_uniform_signal_axis:
            signal_size = self.axes[-1]["size"]
        else:
            signal_size = self.axes[-1]["axis"].size

        axes_sizes = []
        for i in range(len(self.axes) - 1):
            axes_sizes.append(self.axes[i]["size"])

        ## np.prod of an empty array is 1
        if ncollected_spectra != np.array(axes_sizes).prod():
            # TODO: maybe load as a stack instead, check order
            _logger.warning(
                "Axes sizes do not match data size.\n"
                "Data is averaged over multiple collected spectra."
            )
            self.data = np.mean(self.data.reshape(-1, ncollected_spectra), axis=1)

        axes_sizes.append(signal_size)
        self.data = np.reshape(self.data, axes_sizes)


def file_reader(filename, lazy=False, use_uniform_signal_axis=True, **kwds):
    filesize = Path(filename).stat().st_size
    dictionary = {}
    with open(str(filename), "rb") as f:
        wdf = WDFReader(
            f,
            filesize=filesize,
            use_uniform_signal_axis=use_uniform_signal_axis,
        )

        dictionary["data"] = wdf.data
        dictionary["axes"] = wdf.axes
        dictionary["metadata"] = deepcopy(wdf.metadata)
        dictionary["original_metadata"] = deepcopy(wdf.original_metadata)

    return [
        dictionary,
    ]
