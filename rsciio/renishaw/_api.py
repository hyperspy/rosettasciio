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

import logging
import datetime
from pathlib import Path
from io import BytesIO
from enum import IntEnum, Enum
from copy import deepcopy

import numpy as np
from numpy.polynomial.polynomial import polyfit

_logger = logging.getLogger(__name__)

try:
    import PIL
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational
except ImportError:
    PIL = None


full_blockname_gwyddion = {
    "WDF1": "FILE",
    "DATA": "DATA",
    "YLST": "YLIST",
    "XLST": "XLIST",
    "ORGN": "ORIGIN",
    "TEXT": "COMMENT",
    "WXDA": "WIREDATA",
    "WXDB": "DATASETDATA",
    "WXDM": "MEASUREMENT",
    "WXCS": "CALIBRATION",
    "WXIS": "INSTRUMENT",
    "WMAP": "MAPAREA",
    "WHTL": "WHITELIGHT",
    "NAIL": "THUMBNAIL",
    "MAP ": "MAP",
    "CFAR": "CURVEFIT",
    "DCLS": "COMPONENT",
    "PCAR": "PCAR",
    "MCRE": "EM",
    "ZLDC": "ZELDAC",
    "RCAL": "RESPONSECAL",
    "CAP ": "CAP",
    "WARP": "PROCESSING",
    "WARA": "ANALYSIS",
    "WLBL": "SPECTRUMLABELS",
    "WCHK": "CHECKSUM",
    "RXCD": "RXCALDATA",
    "RXCF": "RXCALFIT",
    "XCAL": "XCAL",
    "SRCH": "SPECSEARCH",
    "TEMP": "TEMPPROFILE",
    "UNCV": "UNITCONVERT",
    "ARPR": "ARPLATE",
    "ELEC": "ELECSIGN",
    "BKXL": "BKXLIST",
    "AUX ": "AUXILARYDATA",
    "CHLG": "CHANGELOG",
    "SURF": "SURFACE",
    "PSET": "STREAM_IS_PSET",
}


def convert_windowstime_to_datetime(wt):
    base = datetime.datetime(1601, 1, 1, 0, 0, 0, 0)
    delta = datetime.timedelta(microseconds=wt / 10)
    return (base + delta).isoformat()


# TODO: use lumispy for this?
def convert_wl(wn):
    """Convert wavenumber (cm^-1) to nm"""
    try:
        wl = 1 / (wn * 1e2) / 1e-9
    except ZeroDivisionError:
        wl = np.nan
    return wl


## time is handled differently
## because it cannot be read with __read_numeric
class MetadataTypeSingle(Enum):
    char = "c"
    uint8 = "?"
    int16 = "s"
    int32 = "i"
    int64 = "w"
    float = "r"
    double = "q"
    len_char = 1
    len_uint8 = 1
    len_int16 = 2
    len_int32 = 4
    len_int64 = 8
    len_float = 4
    len_double = 8


class MetadataTypeMulti(Enum):
    string = "u"
    nested = "p"
    key = "k"
    binary = "b"


class MetadataFlags(IntEnum):
    normal = 0
    array = 128
    compressed = 64


## < specifies little endian
class TypeNames(Enum):
    char = "<c"
    int8 = "<b"  # byte int
    int16 = "<h"  # short int
    int32 = "<i"  # int
    int64 = "<l"  # long int
    int128 = "<q"  # long long int
    uint8 = "<B"  # unsigned byte int
    uint16 = "<H"  # unsigned short int
    uint32 = "<I"  # unsigned int
    uint64 = "<L"  # unsigned long int
    uint128 = "<Q"  # unsigned long long int
    float = "<f"  # float (32)
    double = "<d"  # double (64)


class MeasurementType(IntEnum):
    Unspecified = 0
    Single = 1
    Series = 2
    Mapping = 3


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


# TODO: check units/compare to gwyddion
# possibly explicit mapping different to __str__
## TODO: why is the wavenumber unit nm?
class UnitType(IntEnum):
    Arbitrary = 0
    RamanShift = 1  # cm^-1 by default
    Wavenumber = 2  # nm
    Nanometre = 3
    ElectronVolt = 4
    Micron = 5  # same for EXIF units
    Counts = 6
    Electrons = 7
    Millimetres = 8
    Metres = 9
    Kelvin = 10
    Pascal = 11
    Seconds = 12
    Milliseconds = 13
    Hours = 14
    Days = 15
    Pixels = 16
    Intensity = 17
    RelativeIntensity = 18
    Degrees = 19
    Radians = 20
    Celsius = 21
    Fahrenheit = 22
    KelvinPerMinute = 23
    FileTime = 24
    Microseconds = 25

    def __str__(self):
        """Rewrite the unit name output"""
        unit_str = dict(
            Arbitrary="",
            RamanShift="1/cm",  # cm^-1 by default
            Wavenumber="nm",  # nm
            Nanometre="nm",
            ElectronVolt="eV",
            Micron="µm",  # same for EXIF units
            Counts="counts",
            Electrons="electrons",
            Millimetres="mm",
            Metres="m",
            Kelvin="K",
            Pascal="Pa",
            Seconds="s",
            Milliseconds="ms",
            Hours="h",
            Days="d",
            Pixels="px",
            Intensity="",
            RelativeIntensity="",
            Degrees="°",
            Radians="rad",
            Celsius="°C",
            Fahrenheit="°F",
            KelvinPerMinute="K/min",
            FileTime="s",  # FileTime use stamps and in relative second
        )
        return unit_str[self.name]


class DataType(IntEnum):
    Arbitrary = 0
    Frequency = 1
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
    Derived = 12
    Polarization = 13
    FocusTrack = 14
    RampRate = 15
    Checksum = 16
    Flags = 17
    ElapsedTime = 18
    Spectral = 19
    Mp_Well_Spatial_X = 22
    Mp_Well_Spatial_Y = 23
    Mp_LocationIndex = 24
    Mp_WellReference = 25
    EndMarker = 26
    ExposureTime = 27


class ExifTags(IntEnum):
    """Customized EXIF TAGS"""

    # Standard EXIF TAGS
    FocalPlaneXResolution = 0xA20E
    FocalPlaneYResolution = 0xA20F
    FocalPlaneResolutionUnit = 0xA210
    # Customized EXIF TAGS from Renishaw
    FocalPlaneXYOrigins = 0xFEA0
    FieldOfViewXY = 0xFEA1


class WDFReader(object):
    """Reader for Renishaw(TM) WiRE Raman spectroscopy files (.wdf format)

    The wdf file format is separated into several DataBlocks, with starting 4-char
    strings such as (incomplete list):
    `WDF1`: File header for information
    `DATA`: Spectra data
    `XLST`: Data for X-axis of data, usually the Raman shift or wavelength
    `YLST`: Data for Y-axis of data, possibly not important
    `WMAP`: Information for mapping, e.g. StreamLine or StreamLineHR mapping
    `MAP `: Mapping information(?)
    `ORGN`: Data for stage origin
    `TEXT`: Annotation text etc
    `WXDA`: wiredata metadata
    `WXDM`: measurement metadata (contains code???)
    `ZLDC`: zero level and dark current metadata
    `BKXL`: ?
    `WXCS`: calibration metadata
    `WXIS`: instrument metadata
    `WHTL`: White light image

    Following the block name, there are two indicators:
    Block uid: uint32 (not used for anything)
    Block size: uint64

    metadata is read from PSET blocks,
    only the blocks where pset is present

    Args:
    f : File object (opened in binary read mode) of a wdf-file

    Attributes:
    count (int) : Numbers of experiments (same type), can be smaller than capacity
    points_per_spectrum (int): Should be identical to xlist_length
    data_origin_count (int) : Number of rows in data origin list
    capacity (int) : Max number of spectra
    accumulation_count (int) : Single or multiple measurements
    block_info (dict) : contains information about all located blocks
    """

    def __init__(self, f):
        self.file_obj = f

    def __read_numeric(self, type, size=1, ret_array=False):
        """Reads the file_object at the current position as the specified type.
        Supported types: see TypeNames
        """
        if type not in TypeNames._member_names_:
            raise ValueError(
                f"Trying to read number with unknown dataformat."
                f"Input: {type}"
                f"Supported formats: {TypeNames._member_names_}"
            )
        data = np.fromfile(self.file_obj, dtype=TypeNames[type].value, count=size)
        ## convert unsigned ints to ints
        ## because int + uint -> float -> problems with indexing
        if type in ["uint8", "uint16", "uint32", "uint64", "uint128"]:
            data = data.astype(int)
        if size == 1 and not ret_array:
            return data[0]
        else:
            return data

    def __read_string(self, size, repr="utf8"):
        """Reads the file_object at the current position as utf8 or ascii of length size."""
        if repr not in ["utf8", "ascii"]:
            raise ValueError(
                f"Trying to read/decode string with invalid format."
                f"Input: {repr}"
                f"Supported formats: utf8, ascii"
            )
        return self.file_obj.read(size).decode(repr).replace("\x00", "")

    def __read_hex(self, size):
        """Reads the file_object at the current position as hex and converts the result to int."""
        return int(self.file_obj.read(size).hex(), base=16)

    def __locate_single_block(self, pos):
        """Get block information starting at pos."""
        self.file_obj.seek(pos)
        block_name = self.__read_string(0x4)
        if len(block_name) < 4:
            raise EOFError
        block_uid = self.__read_numeric("uint32")
        block_size = self.__read_numeric("uint64")
        return block_name, block_uid, block_size

    ## TODO: maybe remove uid from blockinfo (just used in _debug_block_names)
    ## TODO: already add Offsets.block_header to curpos in block_info?
    def locate_all_blocks(self):
        """Get information for all data blocks and store them inside self.block_info"""
        self.block_info = {}
        block_header_size = 16
        curpos = 0
        finished = False
        while not finished:
            try:
                block_name, block_uid, block_size = self.__locate_single_block(curpos)
                self.block_info[block_name] = (block_uid, curpos + block_header_size, block_size)
                curpos += block_size
            except EOFError:
                finished = True
        _logger.debug(self._debug_block_names())

    def _debug_block_names(self):
        retStr = "ID     GWYDDION_ID   UID CURPOS   SIZE\n"
        retStr += "--------------------------------------\n"
        total_size = 0
        for key, val in self.block_info.items():
            retStr += f"{key}   {full_blockname_gwyddion[key]:<{14}}{val[0]}   {val[1]:<{9}}{val[2]}\n"
            total_size += val[2]
        retStr += f"total size: {total_size}\n"
        retStr += "--------------------------------------\n"
        return retStr

    def _check_block_exists(self, block_name):
        if block_name in self.block_info.keys():
            return True
        else:
            _logger.debug(f"Block {block_name} not present in file.")
            return False

    def _parse_metadata(self, id):
        """Parse blocks with pset metadata."""
        if not self._check_block_exists(id):
            return
        wdf_stream_is_pset = int(0x54455350)
        _, pos, _ = self.block_info[id]
        self.file_obj.seek(pos)
        is_pset = self.__read_numeric("uint32")
        pset_size = self.__read_numeric("uint32")
        if is_pset != wdf_stream_is_pset:
            _logger.debug("No PSET found in this Block -> cannot extract metadata.")
            return
        metadata = self.__read_pset_metadata(pset_size)
        self.original_metadata.update({id: metadata})

    def __read_pset_entry(self, type, size=1):
        if type in MetadataTypeSingle._value2member_map_:
            type_str = MetadataTypeSingle(type).name
            type_len = MetadataTypeSingle[f"len_{type_str}"].value
            result = self.__read_numeric(type_str, size=size)
            # _logger.debug(f"{type} = {result}")
            return result, type_len
        elif type in MetadataTypeMulti._value2member_map_:
            length = self.__read_numeric("uint32")
            # _logger.debug(f"type length: {length}")
            if type == "b":
                # _logger.debug("BINARY ENCOUNTERED")
                result = None
                self.file_obj.read(length)
            elif type in ["u", "k"]:
                result = self.__read_string(length)
                # _logger.debug(result)
            elif type == "p":
                result = self.__read_pset_metadata(length)
            ## reading the length itself equals 4 bytes
            return result, length + 4
        elif type == "t":
            length = 8
            result = None
            self.file_obj.read(length)
            return result, length
        else:
            raise ValueError(f"Unknown type: {type}")

    def __read_pset_metadata(self, length):
        key_dict = {}
        value_dict = {}
        remaining = length
        while remaining > 0:
            type = self.__read_string(1)
            flag = self.__read_hex(1)
            key = self.__read_numeric("uint16")
            remaining -= 4
            # _logger.debug(f"type: {type}, flag: {flag}, key: {key}")
            if flag != 0:
                size = self.__read_numeric("uint32")
                remaining -= 4
            else:
                size = 1
            result, num_bytes = self.__read_pset_entry(type, size)
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
        ## TODO: why there are unmatched keys/values?
        # if len(key_dict) != 0:
        # _logger.debug(f"Unmatched keys: {key_dict}")
        # if len(value_dict) != 0:
        # _logger.debug(f"Unmatched values: {value_dict}")
        return retDict

    def _parse_WDF1(self):
        header_metadata = {}
        return_metadata = {}
        self.file_obj.seek(0)
        _, pos, _ = self.block_info["WDF1"]
        if pos != 16:
            _logger.warning("Unexpected start of file. File might be invalid.")
        self.file_obj.seek(pos)
        header_metadata["flags"] = self.__read_numeric("uint64")
        header_metadata["uuid"] = f"{self.__read_numeric('uint32')}"
        for _ in range(3):
            header_metadata["uuid"] += f"-{self.__read_numeric('uint32')}"
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
        header_metadata["app_name"] = self.__read_string(24)  # must be "WiRE"
        header_metadata["app_version"] = f"{self.__read_numeric('uint16')}"
        for _ in range(3):
            header_metadata["app_version"] += f"-{self.__read_numeric('uint16')}"
        header_metadata["scan_type"] = ScanType(self.__read_numeric("uint32")).name
        header_metadata["measurement_type"] = MeasurementType(
            self.__read_numeric("uint32")
        ).name
        time_start_wt = self.__read_numeric("uint64")
        header_metadata["time_start"] = convert_windowstime_to_datetime(time_start_wt)
        time_end_wt = self.__read_numeric("uint64")
        header_metadata["time_end"] = convert_windowstime_to_datetime(time_end_wt)
        header_metadata["spectral_unit"] = UnitType(self.__read_numeric("uint32")).name
        header_metadata["laser_wavenumber"] = self.__read_numeric("float")
        unused2 = self.__read_numeric("uint64", size=6)
        header_metadata["username"] = self.__read_string(32)
        header_metadata["title"] = self.__read_string(160)

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

        print(return_metadata["accumulation_count"])
        print(return_metadata["count"])
        print(return_metadata["points_per_spectrum"])
        return return_metadata

    ## TODO: signal seems to be in reverse order, maybe revert this first
    def _parse_XLST(self, size, use_uniform_signal_axis=True):
        """Parse XLST Block and extract signal axis information."""
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

    def _set_axes(self, signal_dict, nav_orgn, nav_wmap):
        nav_dict = deepcopy(nav_wmap)
        for axis in nav_dict.keys():
            nav_dict[axis]["navigate"] = True
            nav_dict[axis]["units"] = nav_orgn[axis]["unit"]
        signal_dict["index_in_array"] = len(nav_dict)
        if "Y-axis" in nav_dict.keys():
            nav_dict["Y-axis"]["index_in_array"] = 0
            if "X-axis" in nav_dict.keys():
                nav_dict["X-axis"]["index_in_array"] = 1
        else:
            if "X-axis" in nav_dict.keys():
                nav_dict["X-axis"]["index_in_array"] = 0

        self.axes = nav_dict
        self.axes["signal_dict"] = signal_dict
        self.axes = sorted(self.axes.values(), key=lambda item: item["index_in_array"])

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
        """Get information from XLST or YLST blocks"""
        _, pos, size_block = self.block_info[name.upper() + "LST"]
        ## check if the block size from the WDF1 and of YLST/XLST are consistent
        if 4 * size != (size_block - 16 - 8) and name.upper() == "X":
            if name.upper() == "X":
                _logger.warning(
                    "Inconsistent size of XLST-Block."
                    "This may lead to incorrect signal axes."
                )
        self.file_obj.seek(pos)
        type = DataType(self.__read_numeric("uint32")).name
        unit = str(UnitType(self.__read_numeric("uint32")))
        ## TODO: why wavenumber unit is nm?
        if type == "Frequency":
            if unit == "1/cm":
                type = "Wavenumber"
            elif unit in ["nm", "µm", "m", "mm"]:
                type = "Wavelength"
        data = self.__read_numeric("float", size=size)
        return type, unit, data

    def _parse_DATA(self, size):
        """Get information from DATA block"""
        if not self._check_block_exists("DATA"):
            raise IOError("File does not contain data (DATA-Block missing).")
        _, pos, size_block = self.block_info["DATA"]
        if 4 * size != (size_block - 16):
            _logger.warning(
                "Size of data block is inconsistent with header metadata."
                "This may lead to incorrect data results."
            )
        self.file_obj.seek(pos)
        self.data = self.__read_numeric("float", size=size)

    def _parse_ORGN(self, header_orgn_count, ncollected_spectra):
        """Get information from OriginList"""
        if not self._check_block_exists("ORGN"):
            return {}
        nav_dict = {}
        orgn_metadata = {}
        _, pos, _ = self.block_info["ORGN"]
        self.file_obj.seek(pos)
        origin_count = self.__read_numeric("uint32")
        if origin_count != header_orgn_count:
            _logger.warning(
                "Ambiguous number of entrys for ORGN block."
                "This may lead to incorrect metadata and axes."
            )
        for _ in range(origin_count):
            ax_tmp_dict = {}
            ## ignore first bit of dtype read (sometimes 0, sometimes 1 in testfiles)
            dtype = DataType(self.__read_numeric("uint32") & ~(0b1 << 31)).name
            ax_tmp_dict["unit"] = str(UnitType(self.__read_numeric("uint32")))
            ax_tmp_dict["annotation"] = self.__read_string(0x10)

            if dtype == DataType.Time.name:
                array = self.__read_numeric(
                    "uint64", size=ncollected_spectra, ret_array=True
                )
                ax_tmp_dict["data"] = [
                    convert_windowstime_to_datetime(array[i]) for i in range(array.size)
                ]
            else:
                ax_tmp_dict["data"] = self.__read_numeric(
                    "double", size=ncollected_spectra, ret_array=True
                )
            # TODO: maybe need to add more types here (Z, R, Theta, Phi, Time)
            if dtype in [DataType.Spatial_X.name, DataType.Spatial_Y.name]:
                nav_dict[f"{dtype[-1]}-axis"] = ax_tmp_dict
            else:
                orgn_metadata[dtype] = ax_tmp_dict
        self.original_metadata.update({"ORGN": orgn_metadata})
        return nav_dict

    def _check_consistency_wmap_origin(self, orgn_data, wmap_data):
        # TODO: What is done here? Can this be removed?
        # Determine if the xy-grid spacing is same as in x_pad and y_pad
        # if (len(self.xpos) > 1) and (len(self.ypos) > 1):
        #     xdist = np.abs(self.xpos - self.xpos[0])
        #     ydist = np.abs(self.ypos - self.ypos[0])
        #     xdist = xdist[np.nonzero(xdist)]
        #     ydist = ydist[np.nonzero(ydist)]
        #     # Get minimal non-zero padding in the grid
        #     try:
        #         x_pad_grid = np.min(xdist)
        #     except ValueError:
        #         x_pad_grid = 0

        #     try:
        #         y_pad_grid = np.min(ydist)
        #     except ValueError:
        #         y_pad_grid = 0
        pass

    def _parse_WMAP(self):
        """Get information about mapping in StreamLine and StreamLineHR"""
        if not self._check_block_exists("WMAP"):
            return {}
        _, pos, _ = self.block_info["WMAP"]
        self.file_obj.seek(pos)

        # TODO: check flags -> maybe need to compare with ORGN
        # flags encode how the mapping is performed
        flags = self.__read_numeric("uint32")
        unused = self.__read_numeric("uint32")
        offset_xyz = [self.__read_numeric("float") for _ in range(3)]
        scale_xyz = [self.__read_numeric("float") for _ in range(3)]
        size_xyz = [self.__read_numeric("uint32") for _ in range(3)]
        linefocus_size = self.__read_numeric("uint32")

        metadata = {"linefocus_size": linefocus_size, "flags": flags}
        result = {}
        for idx, letter in enumerate(["x", "y", "z"]):
            key_name = letter.upper() + "-axis"
            axis_tmp = {
                "name": letter.upper(),
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

    ## TODO: is this necessary? how to load in hyperspy?
    ## maybe extra flag/argument load_image?
    def _parse_img(self, *args):
        """Extract the white-light JPEG image
        The size of while-light image is coded in its EXIF
        Use PIL to parse the EXIF information
        """
        try:
            _, pos, size = self.block_info["WHTL"]
        except KeyError:
            _logger.debug("The wdf file does not contain an image")
            return

        # Read the bytes. `self.img` is a wrapped IO object mimicking a file
        ## TODO: need to adapt offset here, because of change of pos = curpos + block_header_size
        jpeg_header = 0x10
        self.file_obj.seek(pos + jpeg_header)
        img_bytes = self.file_obj.read(size - jpeg_header)
        self.img = BytesIO(img_bytes)
        # Handle image dimension if PIL is present
        if PIL is not None:
            pil_img = Image.open(self.img)
            # Weird missing header keys when Pillow >= 8.2.0.
            # see https://pillow.readthedocs.io/en/stable/releasenotes/8.2.0.html#image-getexif-exif-and-gps-ifd
            # Use fall-back _getexif method instead
            exif_header = dict(pil_img._getexif())
            try:
                # Get the width and height of image
                w_ = exif_header[ExifTags.FocalPlaneXResolution]
                h_ = exif_header[ExifTags.FocalPlaneYResolution]
                x_org_, y_org_ = exif_header[ExifTags.FocalPlaneXYOrigins]

                def rational2float(v):
                    """Pillow<7.2.0 returns tuple, Pillow>=7.2.0 returns IFDRational"""
                    if not isinstance(v, IFDRational):
                        return v[0] / v[1]
                    return float(v)

                w_, h_ = rational2float(w_), rational2float(h_)
                x_org_, y_org_ = rational2float(x_org_), rational2float(y_org_)

                # The dimensions (width, height)
                # with unit `img_dimension_unit`
                self.img_dimensions = np.array([w_, h_])
                # Origin of image is at upper right corner
                self.img_origins = np.array([x_org_, y_org_])
                # Default is microns (5)
                self.img_dimension_unit = UnitType(
                    exif_header[ExifTags.FocalPlaneResolutionUnit]
                )
                # Give the box for cropping
                # Following the PIL manual
                # (left, upper, right, lower)
                self.img_cropbox = self.__calc_crop_box()

            except KeyError:
                _logger.debug(
                    "Some keys in white light image header" " cannot be read!"
                )
        return

    def __calc_crop_box(self):
        """Helper function to calculate crop box"""

        def _proportion(x, minmax, pixels):
            """Get proportional pixels"""
            min, max = minmax
            return int(pixels * (x - min) / (max - min))

        pil_img = PIL.Image.open(self.img)
        w_, h_ = self.img_dimensions
        x0_, y0_ = self.img_origins
        pw = pil_img.width
        ph = pil_img.height
        map_xl = self.xpos.min()
        map_xr = self.xpos.max()
        map_yt = self.ypos.min()
        map_yb = self.ypos.max()
        left = _proportion(map_xl, (x0_, x0_ + w_), pw)
        right = _proportion(map_xr, (x0_, x0_ + w_), pw)
        top = _proportion(map_yt, (y0_, y0_ + h_), ph)
        bottom = _proportion(map_yb, (y0_, y0_ + h_), ph)
        return (left, top, right, bottom)

    def _reshape_data(self, ncollected_spectra, use_uniform_signal_axis):
        if use_uniform_signal_axis:
            signal_size = self.axes[-1]["size"]
        else:
            signal_size = self.axes[-1]["axis"].size

        axes_sizes = []
        for i in range(len(self.axes) - 1):
            axes_sizes.append(self.axes[i]["size"])

        ## np.prod of an empty array is 1
        ## TODO: check timeseries (this will lead to an error)
        if ncollected_spectra != np.array(axes_sizes).prod():
            raise NotImplementedError("Reading timeseries is not yet implemented.")

        axes_sizes.append(signal_size)
        self.data = np.reshape(self.data, axes_sizes)

    def read_file(self, use_uniform_signal_axis=True):
        self.original_metadata = {}
        self.metadata = {}
        self.axes = {}

        self.locate_all_blocks()
        header_data = self._parse_WDF1()  # this needs to be parsed first,
        ## because it contains sizes for ORGN, DATA (reshape), XLST, YLST

        ## parse metadata blocks
        self._parse_YLST(header_data["YLST_length"])
        self._parse_metadata("WXIS")
        self._parse_metadata("WXCS")
        self._parse_metadata("WXDM")
        self._parse_metadata("WXDA")
        self._parse_metadata("WARP")
        self._parse_metadata("ZLDC")

        ## parse blocks with axes/data information
        signal_dict = self._parse_XLST(
            header_data["XLST_length"], use_uniform_signal_axis
        )
        nav_orgn = self._parse_ORGN(
            header_orgn_count=header_data["origin_count"],
            ncollected_spectra=header_data["count"],
        )
        nav_wmap = self._parse_WMAP()
        self._parse_DATA(size=header_data["count"] * header_data["points_per_spectrum"])

        self._check_consistency_wmap_origin(nav_orgn, nav_wmap)
        self._set_axes(signal_dict, nav_orgn, nav_wmap)
        self._reshape_data(header_data["count"], use_uniform_signal_axis)
        # self._map_metadata()


def file_reader(filename, lazy=False, use_uniform_signal_axis=True, **kwds):
    _logger.debug(f"filesize in bytes: {Path(filename).stat().st_size}")
    dictionary = {}
    with open(str(filename), "rb") as f:
        wdf = WDFReader(f)
        wdf.read_file(use_uniform_signal_axis=use_uniform_signal_axis)

        dictionary["data"] = wdf.data
        dictionary["axes"] = wdf.axes
        dictionary["metadata"] = deepcopy(wdf.metadata)
        dictionary["original_metadata"] = deepcopy(wdf.original_metadata)

    return [
        dictionary,
    ]
