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

# The details of the format were taken from
# https://www.biochem.mpg.de/doc_tom/TOM_Release_2008/IOfun/tom_mrcread.html
# and https://ami.scripps.edu/software/mrctools/mrc_specification.php

import logging
from pathlib import Path
from copy import deepcopy
from enum import IntEnum

import numpy as np

from rsciio.docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC

_logger = logging.getLogger(__name__)
_logger.setLevel(10)

## < specifies little endian
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
}


class FileType(IntEnum):
    bit8_itex = 0
    compressed = 1
    bit16 = 2
    bit32 = 3
    bit24_rgb = 11
    bit48_rgb = 12
    bit96_rgb = 13


class ApplicationType(IntEnum):
    application_hipic = 1
    application_ta = 2
    application_em = 3


## type is 37 in testfile -> not used
class CameraType(IntEnum):
    no_camera = 0
    C4880 = 1
    C4742 = 2
    analog0 = 3
    analog1 = 4
    analog2 = 5
    analog3 = 6
    C474295 = 7
    C488080 = 8
    C474298 = 9
    C7300 = 19
    C800020 = 20
    C800010 = 21
    FlatPanel = 22
    DCam = 23
    OrcaHR = 24
    C8484 = 25
    C9100 = 26
    C8800 = 27


## not in testfile
class CameraSubType(IntEnum):
    C4880_00 = 1
    C4880_60 = 2
    C4880_80 = 3
    C4880_91 = 4
    C4880_92 = 5
    C4880_93 = 6
    C4742_95 = 7
    C4880_60OU = 8
    C4880_1K2K = 9
    C4742_98 = 11
    C7300_10 = 12
    C4880_10 = 13
    C4880_20 = 14
    C4880_21 = 15
    C4880_30 = 16
    C4880_40 = 17
    C7190_10 = 18
    C8000_20 = 19
    C4742_95ER = 20
    C4880_31 = 21
    C4880_50 = 22
    C8000_10 = 23
    FlatPanel_C7942 = 24
    FlatPanel_C7943 = 25
    C4742_95HR = 26
    C8484_01 = 27
    FlatPanel_C7921 = 28
    C4742_98ER = 29
    C4742_98BT_K = 30
    C4742_98BT_L = 31
    C7300_10_NRK = 32
    FlatPanel_C7930DP = 33
    C9100_01 = 34
    C9100_02 = 35
    C9100_11 = 36
    C9100_12 = 37
    C8800_01 = 38
    C8800_21 = 39


class AcqMode(IntEnum):
    live = 1
    acquire = 2
    photon_counting = 3
    analog_integration = 4


class DatType(IntEnum):
    dat8 = 1
    dat10 = 2
    dat12 = 3
    dat16 = 4
    dat812 = 5
    dat14 = 6
    dat16u = 7
    dat32 = 8


class FrameGrabber(IntEnum):
    grbNone = 0
    grbAFG = 1
    grbICP = 2
    grbPC = 3
    grbNI = 4
    grbDCam = 5


class AcquisitionMode(IntEnum):
    amdig = 1
    amvs = 2
    cam_link = 3


class LUTSize(IntEnum):
    lut_size_8 = 1
    lut_size_10 = 2
    lut_size_12 = 3
    lut_size_16 = 4
    lut_size_812 = 5
    lut_size_14 = 6
    lut_size_16x = 9


class LUTColor(IntEnum):
    lut_color_bw = 1
    lut_color_rainbow = 2
    lut_color_bw_without_color = 3


## no numbers in user manual, 0 in testfile -> linear?
class LUTType(IntEnum):
    lut_type_linear = 1
    lut_type_gamma = 2
    lut_type_sigmoid = 3


class Scaling_Type(IntEnum):
    scaling_linear = 1
    scaling_table = 2


class IMGReader:
    def __init__(self, file, filesize, filename):
        self._file_obj = file
        self._filesize = filesize
        self._original_filename = filename

        self.metadata = {}
        self.original_metadata = {}

        self.data, comment = self.parse_file()
        self._map_comment(comment)
        self.axes = self._read_calibration()
        self._reshape_data()

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
        elif type == "char" and convert:
            data = list(map(chr, data))
        if size == 1 and not ret_array:
            return data[0]
        else:
            return data

    def __read_utf8(self, size):
        return self._file_obj.read(size).decode("utf8").replace("\x00", "")

    def parse_file(self):
        self._file_obj.seek(0)
        header = {}
        header["character_im"] = self.__read_utf8(2)
        com_len = int(self.__read_numeric("int16"))
        ## IMPORTANT to convert int16 to int
        ## as int16 leads to problems when defining sizes of numpy arrays
        ## here data is read incorrectly
        self.w_px = int(self.__read_numeric("int16"))
        self.h_lines = int(self.__read_numeric("int16"))
        header["offset_x"] = int(self.__read_numeric("int16"))
        header["offset_y"] = int(self.__read_numeric("int16"))
        file_type = self.__read_numeric("int16")
        header["file_type"] = FileType(file_type).name
        header["num_img"] = int.from_bytes(self._file_obj.read(3), "little")
        header["num_channels"] = int(self.__read_numeric("int16"))
        header["channel_number"] = int(self.__read_numeric("int16"))
        header["timestamp"] = self.__read_numeric("double")
        header["marker"] = self.__read_utf8(3)
        header["additional_info"] = self.__read_utf8(29)
        comment = self.__read_utf8(com_len)
        ## TODO: check dtype for different filetypes
        data = self.__read_numeric("uint32", size=self.w_px * self.h_lines)
        header["image_width_px"] = self.w_px
        header["image_height_lines"] = self.h_lines
        header["com_len"] = com_len
        _logger.debug(f"file read until: {self._file_obj.tell()}")
        _logger.debug(f"total file_size: {self._filesize}")
        ## missing bytes, because calibration data at the end
        self.original_metadata.update(header)
        return data, comment

    @staticmethod
    def _get_scaling_entry(scaling_dict, attr_name):
        x_val = scaling_dict.get("ScalingX" + attr_name)
        y_val = scaling_dict.get("ScalingY" + attr_name)
        return x_val, y_val

    def _extract_calibration_data(self, cal):
        if cal[0] == "#":
            pos, size = map(int, cal[1:].split(","))
            self._file_obj.seek(pos)
            return self.__read_numeric("float", size=size)
        else:
            return None

    def _set_axis(self, name, scale_type, unit, cal_addr, scale_val):
        axis = {"units": unit, "name": name, "navigate": False}
        if scale_type == 1:
            ## in this mode (focus mode) the y-axis does not correspond to time
            ## photoelectrons are not deflected here -> natural spread
            ## TODO: scale, name, unit for this?
            axis["scale"] = 20 / self.h_lines
            axis["offset"] = 0
            axis["size"] = self.h_lines
        elif scale_type == 2:
            data = self._extract_calibration_data(cal_addr)
            axis["axis"] = data
            axis["_type"] = "DataAxis"
        else:
            raise ValueError
        return axis

    ## TODO: refactor into get_axes
    def _read_calibration(self):
        scaling_md = self.original_metadata.get("Comment", {}).get("Scaling", {})
        x_cal_address, y_cal_address = self._get_scaling_entry(
            scaling_md, "ScalingFile"
        )
        x_unit, y_unit = self._get_scaling_entry(scaling_md, "Unit")
        x_type, y_type = map(int, self._get_scaling_entry(scaling_md, "Type"))
        x_scale, y_scale = map(float, self._get_scaling_entry(scaling_md, "Scale"))
        x_axis = self._set_axis("X", x_type, x_unit, x_cal_address, x_scale)
        y_axis = self._set_axis("Y", y_type, y_unit, y_cal_address, y_scale)
        _logger.debug(f"file read until: {self._file_obj.tell()}")
        y_axis["index_in_array"] = 0
        x_axis["index_in_array"] = 1
        wavelength_axis = x_axis["axis"]
        if wavelength_axis[0] > wavelength_axis[1]:
            self._reverse_signal = True
            x_axis["axis"] = wavelength_axis[::-1]
        else:
            self._reverse_signal = False
        axes_list = sorted([x_axis, y_axis], key=lambda item: item["index_in_array"])
        return axes_list

    def _reshape_data(self):
        axes_sizes = []
        for ax in self.axes:
            try:
                axes_sizes.append(ax["axis"].size)
            except KeyError:
                axes_sizes.append(ax["size"])

        self.data = np.reshape(self.data, axes_sizes)
        if self._reverse_signal:
            self.data = np.flip(self.data, 1)

    @staticmethod
    def _split_header_from_comment(input):
        initial_split = input.split("[")[1:]
        result = {}
        for entry in initial_split:
            sep_idx = entry.index("]")
            header = entry[:sep_idx]
            body = entry[sep_idx + 2 :].rstrip()
            result[header] = body
        return result

    @staticmethod
    def _get_range_for_val(v, sep, count, num_entries, str_len):
        if v[sep + 1] == '"':
            end_val = v.index('"', sep + 2)
            start_val = sep + 2
            total_end = end_val + 2
        else:
            if count == num_entries:
                end_val = str_len
            else:
                end_val = v.index(",", sep)
            start_val = sep + 1
            total_end = end_val + 1
        return start_val, end_val, total_end

    def _extract_entries_from_comment(self, v):
        result = {}
        str_len = len(v)
        cur_pos = 0
        count = 0
        num_entries = v.count("=")
        if num_entries == 0:
            return v
        while cur_pos < str_len:
            count += 1
            sep = v.index("=", cur_pos)
            key = v[cur_pos:sep]
            start_val, end_val, cur_pos = self._get_range_for_val(
                v, sep, count, num_entries, str_len
            )
            val = v[start_val:end_val]
            result[key] = val
        return result

    def _map_comment(self, comment):
        initial_split = self._split_header_from_comment(comment)
        result = {}
        for k, v in initial_split.items():
            result[k] = self._extract_entries_from_comment(v)
        self.original_metadata.update({"Comment": result})


def file_reader(filename, lazy=False, **kwds):
    """Reads Hamamatsu's ``.img`` file.

    Parameters
    ----------
    %s
    %s

    %s
    """
    filesize = Path(filename).stat().st_size
    original_filename = Path(filename).name
    result = {}
    with open(str(filename), "rb") as f:
        wdf = IMGReader(
            f,
            filesize=filesize,
            filename=original_filename,
        )

        result["data"] = wdf.data
        result["axes"] = wdf.axes
        result["metadata"] = deepcopy(wdf.metadata)
        result["original_metadata"] = deepcopy(wdf.original_metadata)

    return [
        result,
    ]


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)
