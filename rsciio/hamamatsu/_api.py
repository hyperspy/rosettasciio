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

import importlib.util
import logging
from copy import deepcopy
from enum import EnumMeta, IntEnum
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyfit

from rsciio._docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC

_logger = logging.getLogger(__name__)


def _str2numeric(input, type):
    """Handle None-values when converting strings to float."""
    try:
        if type == "float":
            return float(input)
        elif type == "int":
            return int(input)
        else:
            return None
    except (ValueError, TypeError):
        return None


def _str2bool(input):
    if input == "-1":
        return True
    elif input == "0":
        return False
    else:
        return None


def _remove_none_from_dict(dict_in):
    """Recursive removal of None-values from a dictionary."""
    for key, value in list(dict_in.items()):
        if isinstance(value, dict):
            _remove_none_from_dict(value)
        elif value is None:
            del dict_in[key]


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


class DefaultEnum(IntEnum):
    Unknown = 0


class DefaultEnumMeta(EnumMeta):
    def __call__(cls, value, *args, **kwargs):
        if value not in cls._value2member_map_:
            return DefaultEnum(0)
        else:
            return super().__call__(value, *args, **kwargs)


## general itex formats may have different conventions
## here the convention from the HPD-TA manual is used
class FileType(IntEnum, metaclass=DefaultEnumMeta):
    bit8 = 0
    compressed = 1  # not used by HPD-TA
    bit16 = 2
    bit32 = 3


class AcqMode(IntEnum, metaclass=DefaultEnumMeta):
    live = 1
    acquire = 2
    photon_counting = 3
    analog_integration = 4


class Scaling_Type(IntEnum, metaclass=DefaultEnumMeta):
    scaling_linear = 1
    scaling_table = 2


class IMGReader:
    def __init__(self, file, filesize, filename, use_uniform_signal_axes):
        self._file_obj = file
        self._filesize = filesize
        self._original_filename = filename
        self._use_uniform_signal_axes = use_uniform_signal_axes

        self.original_metadata = {}
        self._h_lines = None
        self._reverse_signal = False

        self.data, comment = self.parse_file()

        processed_comment = self._process_comment(comment)
        self.original_metadata.update({"Comment": processed_comment})

        self.axes = self._get_axes()
        self._reshape_data()
        self.metadata = self.map_metadata()

    def __read_numeric(self, type, size=1, ret_array=False, convert=True):
        if type not in TypeNames.keys():
            raise ValueError(
                f"Trying to read number with unknown dataformat.\n"
                f"Input: {type}\n"
                f"Supported formats: {list(TypeNames.keys())}"
            )
        data = np.fromfile(self._file_obj, dtype=TypeNames[type], count=size)
        # convert unsigned ints to ints
        # because int + uint = float -> problems with indexing
        # this leads to problems with uin64, because there is no int128 in numpy
        if type in ["uint8", "uint16", "uint32", "uint64"] and convert:
            data = data.astype(np.dtype("<i8"))
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
        header["comment_length"] = com_len
        ## IMPORTANT to convert int16 to int
        ## as int16 leads to problems when defining sizes of numpy arrays
        ## -> data is read incorrectly
        w_px = int(self.__read_numeric("int16"))
        header["image_width_px"] = w_px
        self._h_lines = int(self.__read_numeric("int16"))
        header["image_height_lines"] = self._h_lines
        header["offset_x"] = int(self.__read_numeric("int16"))
        header["offset_y"] = int(self.__read_numeric("int16"))
        file_type = FileType(int(self.__read_numeric("int16"))).name
        header["file_type"] = file_type
        header["num_images_in_channel"] = int(self.__read_numeric("int32"))
        header["num_additional_channels"] = int(self.__read_numeric("int16"))
        header["channel_number"] = int(self.__read_numeric("int16"))
        header["timestamp"] = self.__read_numeric("double")
        header["marker"] = self.__read_utf8(4)
        ## according to the manual, additional_info is one byte shorter
        ## however, there is also an unexplained 1 byte gap between marker and additional info
        ## so this extra byte is absorbed in additional_info
        ## in the testfiles both marker and additional_info contain only zeros
        header["additional_info"] = self.__read_utf8(30)
        comment = self.__read_utf8(com_len)
        if file_type == "bit8":
            dtype = "uint8"
        elif file_type == "bit16":
            dtype = "uint16"
        elif file_type == "bit32":
            dtype = "uint32"
        else:
            raise RuntimeError(f"reading type: {file_type} not implemented")
        data = self.__read_numeric(dtype, size=w_px * self._h_lines)
        self.original_metadata.update(header)
        return data, comment

    @staticmethod
    def _get_scaling_entry(scaling_dict, attr_name):
        x_val = scaling_dict.get("ScalingX" + attr_name)
        y_val = scaling_dict.get("ScalingY" + attr_name)
        if y_val == "us":
            y_val = "µs"
        return x_val, y_val

    def _extract_calibration_data(self, cal):
        if cal[0] == "#":
            pos, size = map(int, cal[1:].split(","))
            self._file_obj.seek(pos)
            return self.__read_numeric("float", size=size)
        else:
            raise RuntimeError(
                f"Cannot read axis data (invalid start for address {cal})"
            )

    def _set_axis(self, name, scale_type, unit, cal_addr):
        axis = {"units": unit, "name": name, "navigate": False}
        if scale_type == 1:
            ## in this mode (focus mode) the y-axis does not correspond to time
            ## photoelectrons are not deflected here -> natural spread
            axis["units"] = "px"
            axis["scale"] = 1
            axis["offset"] = 0
            axis["size"] = self._h_lines
            axis["name"] = "Vertical CCD Position"
        elif scale_type == 2:
            data = self._extract_calibration_data(cal_addr)
            # in testfile wavelength is exactly uniform
            # time is close
            if name == "Wavelength":
                if data[0] > data[1]:
                    self._reverse_signal = True
                    data = np.ascontiguousarray(data[::-1])
                else:
                    self._reverse_signal = False
            if self._use_uniform_signal_axes:
                offset, scale = polyfit(np.arange(data.size), data, deg=1)
                axis["offset"] = offset
                axis["scale"] = scale
                axis["size"] = data.size
                scale_compare = 100 * np.max(np.abs(np.diff(data) - scale) / scale)
                if scale_compare > 1:
                    _logger.warning(
                        f"The relative variation of the signal-axis-scale ({scale_compare:.2f}%) exceeds 1%.\n"
                        "                            "
                        "Using a non-uniform-axis is recommended."
                    )
            else:
                axis["axis"] = data
        else:
            raise ValueError(
                f"Cannot extract {name}-axis information (invalid scale-type)."
            )
        return axis

    def _get_axes(self):
        scaling_md = self.original_metadata.get("Comment", {}).get("Scaling", {})
        x_cal_address, y_cal_address = self._get_scaling_entry(
            scaling_md, "ScalingFile"
        )
        x_unit, y_unit = self._get_scaling_entry(scaling_md, "Unit")
        x_type, y_type = map(int, self._get_scaling_entry(scaling_md, "Type"))
        x_axis = self._set_axis("Wavelength", x_type, x_unit, x_cal_address)
        y_axis = self._set_axis("Time", y_type, y_unit, y_cal_address)
        y_axis["index_in_array"] = 0
        x_axis["index_in_array"] = 1
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
            self.data = np.ascontiguousarray(self.data[:, ::-1])

    @staticmethod
    def _split_sections_from_comment(input):
        initial_split = input[1:].split("[")  # ignore opening bracket at start
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

    def _extract_entries_from_section(self, entries_str):
        result = {}
        str_len = len(entries_str)
        cur_pos = 0
        counter = 0
        num_entries = entries_str.count("=")
        if num_entries == 0:
            return entries_str
        while cur_pos < str_len:
            counter += 1
            sep_idx = entries_str.index("=", cur_pos)
            key = entries_str[cur_pos:sep_idx]
            start_val, end_val, cur_pos = self._get_range_for_val(
                entries_str, sep_idx, counter, num_entries, str_len
            )
            val = entries_str[start_val:end_val]
            result[key] = val
        return result

    def _process_comment(self, comment):
        section_split = self._split_sections_from_comment(comment)
        result = {}
        for k, v in section_split.items():
            result[k] = self._extract_entries_from_section(v)
        return result

    def _map_general_md(self):
        general = {}
        general["title"] = self._original_filename.split(".")[0]
        general["original_filename"] = self._original_filename
        try:
            date = self.original_metadata["Comment"]["Application"]["Date"]
            time = self.original_metadata["Comment"]["Application"]["Time"]
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover
        else:
            delimiters = ["/", "."]
            for d in delimiters:
                date_split = date.split(d)
                if len(date_split) == 3:
                    general["date"] = (
                        date_split[2] + "-" + date_split[1] + "-" + date_split[0]
                    )
                    break
            else:
                _logger.warning("Unknown date format, cannot transfrom to ISO.")
                general["date"] = date
            general["time"] = time.split(".")[0]
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
            quantity = self.original_metadata["Comment"]["Acquisition"]["ZAxisLabel"]
            quantity_unit = self.original_metadata["Comment"]["Acquisition"][
                "ZAxisUnit"
            ]
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover
        else:
            if quantity_unit == "Count":
                quantity_unit = "Counts"
            signal["quantity"] = f"{quantity} ({quantity_unit})"
        return signal

    def _map_detector_md(self):
        detector = {}
        acq_dict = self.original_metadata.get("Comment", {}).get("Acquisition", {})
        streak_dict = self.original_metadata.get("Comment", {}).get("Streak camera", {})

        detector["frames"] = _str2numeric(acq_dict.get("NrExposure"), "int")
        try:
            exp_time_str = acq_dict["ExposureTime"]
        except KeyError:
            pass
        else:
            exp_time_split = exp_time_str.split(" ")
            if len(exp_time_split) == 2:
                exp_time, exp_time_units = exp_time_split
                exp_time = _str2numeric(exp_time, "float")
                if exp_time_units == "s":
                    pass
                elif exp_time_units == "ms":
                    exp_time /= 1000
                else:
                    _logger.warning(
                        f"integration_time is given in {exp_time_units} instead of seconds."
                    )
                detector["integration_time"] = exp_time * detector["frames"]
            else:
                _logger.warning("integration_time could not be extracted")

        try:
            binning_str = acq_dict["pntBinning"]
        except KeyError:
            pass
        else:
            if len(binning_str.split(",")) == 2:
                detector["binning"] = tuple(map(int, binning_str.split(",")))

        detector["processing"] = {
            "shading_correction": _str2bool(acq_dict.get("ShadingCorr")),
            "background_correction": _str2bool(acq_dict.get("BacksubCorr")),
            "curvature_correction": _str2bool(acq_dict.get("CurveCorr")),
            "defect_correction": _str2bool(acq_dict.get("DefectCorrection")),
        }
        detector["detector_type"] = "StreakCamera"
        detector["model"] = streak_dict.get("DeviceName")

        detector["mcp_gain"] = _str2numeric(streak_dict.get("MCP Gain"), "float")
        try:
            time_range_str = streak_dict["Time Range"]
        except KeyError:
            pass
        else:
            time_range_split = time_range_str.split(" ")
            if len(time_range_split) == 2:
                time_range, time_range_units = time_range_split
                time_range = _str2numeric(time_range, "float")
                if time_range_units == "us":
                    time_range_units = "µs"
                detector["time_range"] = time_range
                detector["time_range_units"] = time_range_units
            else:
                # TODO: add warning? only occurs for shading file
                time_range = _str2numeric(time_range_str, "float")
                detector["time_range"] = time_range
        detector["acquisition_mode"] = AcqMode(int(acq_dict.get("AcqMode"))).name
        return detector

    def _map_spectrometer_md(self):
        spectrometer = {}
        spectro_dict = self.original_metadata.get("Comment", {}).get("Spectrograph", {})
        try:
            groove_density_str = spectro_dict["Grating"]
        except KeyError:
            groove_density = None
        else:
            groove_density_split = groove_density_str.split(" ")
            if len(groove_density_split) == 2:
                groove_density, groove_density_units = groove_density_str.split(" ")
                groove_density = _str2numeric(groove_density, "int")
                if groove_density_units != "g/mm":
                    _logger.warning(
                        f"groove_density is given in {groove_density_units}"
                    )
            else:
                groove_density = groove_density_str
        ## Remove grating when no unit (->1 or 2, but not lines per mm)
        ## Same for blaze
        ## warning for these cases?
        if spectro_dict.get("Ruling") != "0" and spectro_dict.get("Blaze") != 0:
            spectrometer["Grating"] = {
                "blazing_wavelength": _str2numeric(spectro_dict.get("Blaze"), "float"),
                "groove_density": _str2numeric(groove_density, "float"),
            }
        spectrometer["model"] = spectro_dict.get("DeviceName")
        spectrometer["entrance_slit_width"] = _str2numeric(
            spectro_dict.get("Side Ent. Slitw."), "float"
        )  ## TODO: units?, side entry iris?
        spectrometer["central_wavelength"] = _str2numeric(
            spectro_dict.get("Wavelength"), "float"
        )
        return spectrometer

    def map_metadata(self):
        """Maps original_metadata to metadata."""
        general = self._map_general_md()
        signal = self._map_signal_md()
        detector = self._map_detector_md()
        spectrometer = self._map_spectrometer_md()

        acquisition_instrument = {
            "Detector": detector,
            "Spectrometer": spectrometer,
        }

        metadata = {
            "Acquisition_instrument": acquisition_instrument,
            "General": general,
            "Signal": signal,
        }
        _remove_none_from_dict(metadata)
        return metadata


def file_reader(filename, lazy=False, use_uniform_signal_axes=False, **kwds):
    """
    Read Hamamatsu's ``.img`` file.

    Parameters
    ----------
    %s
    %s
    use_uniform_signal_axes : bool, default=False
        Can be specified to choose between non-uniform or uniform signal axis.
        If ``True``, the ``scale`` attribute is calculated from the average delta
        along the signal axis and a warning is raised in case the delta varies
        by more than 1 percent.
    **kwds : dict, optional
        Extra keyword argument will be ignored.

    %s
    """
    filesize = Path(filename).stat().st_size
    original_filename = Path(filename).name
    result = {}
    with open(str(filename), "rb") as f:
        img = IMGReader(
            f,
            filesize=filesize,
            filename=original_filename,
            use_uniform_signal_axes=use_uniform_signal_axes,
        )

        result["data"] = img.data
        result["axes"] = img.axes
        result["metadata"] = deepcopy(img.metadata)
        result["original_metadata"] = deepcopy(img.original_metadata)

    return [
        result,
    ]


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)
