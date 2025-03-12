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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

# Plugin to read the mountainsmap surface format (sur)
# Current state can bring support to the surface format if the file is an
# attolight hyperspectral map, but cannot bring write nor support for other
# mountainsmap files (.pro etc.). I need to write some tests, check whether the
# comments can be systematically parsed into metadata and write a support for
# original_metadata or other

import ast
import datetime
import logging
import os
import re
import struct
import warnings
import zlib
from copy import deepcopy

# Commented for now because I don't know what purpose it serves
# import traits.api as t
# Dateutil allows to parse date but I don't think it's useful here
# import dateutil.parser
import numpy as np

# Maybe later we can implement reading the class with the io utils tools instead
# of re-defining read functions in the class
# import rsciio.utils.readfile as iou
# This module will prove useful when we write the export function
# import rsciio.utils.tools
# DictionaryTreeBrowser class handles the fancy metadata dictionnaries
# from hyperspy.misc.utils import DictionaryTreeBrowser
from rsciio._docstrings import (
    FILENAME_DOC,
    LAZY_UNSUPPORTED_DOC,
    RETURNS_DOC,
    SIGNAL_DOC,
)
from rsciio.utils.date_time_tools import get_date_time_from_metadata
from rsciio.utils.exceptions import MountainsMapFileError
from rsciio.utils.rgb_tools import is_rgb, is_rgba

_logger = logging.getLogger(__name__)


def parse_metadata(cmt: str, prefix: str = "$", delimiter: str = "=") -> dict:
    """
    Parse metadata from the comment field of a digitalsurf file, or any other
    str in similar formatting. Return it as a hyperspy-compatible nested dict.

    Parameters
    ----------
    cmt : str
        Str containing contents of a digitalsurf file "comment" field.
    prefix : str
        Prefix character, must be present at the start of each line,
        otherwise the line is ignored. ``"$"`` for digitalsurf files,
        typically an empty string (``""``) when parsing from text files.
        Default is ``"$"``.
    delimiter : str
        Character that delimit key-value pairs in digitalsurf comment.
        Default is ``"="``.

    Returns
    -------
    dict
        Nested dictionnary of the metadata.
    """
    # dict_ms is created as an empty dictionnary
    dict_md = {}
    # Title lines start with an underscore
    titlestart = "{:s}_".format(prefix)

    key_main = None

    for line in cmt.splitlines():
        # Here we ignore any empty line or line starting with @@
        ignore = False
        if not line.strip() or line.startswith("@@"):
            ignore = True
        # If the line must not be ignored
        if not ignore:
            if line.startswith(titlestart):
                # We strip keys from whitespace at the end and beginning
                key_main = line[len(titlestart) :].strip()
                dict_md[key_main] = {}
            elif line.startswith(prefix):
                if key_main is None:
                    key_main = "UNTITLED"
                    dict_md[key_main] = {}
                key, *li_value = line.split(delimiter)
                # Key is also stripped from beginning or end whitespace
                key = key[len(prefix) :].strip()
                str_value = li_value[0] if len(li_value) > 0 else ""
                # remove whitespace at the beginning of value
                str_value = str_value.strip()
                li_value = str_value.split(" ")
                try:
                    if key == "Grating":
                        dict_md[key_main][key] = li_value[
                            0
                        ]  # we don't want to eval this one
                    else:
                        dict_md[key_main][key] = ast.literal_eval(li_value[0])
                except Exception:
                    dict_md[key_main][key] = li_value[0]
                if len(li_value) > 1:
                    dict_md[key_main][key + "_units"] = li_value[1]
    return dict_md


class DigitalSurfHandler(object):
    """Class to read Digital Surf MountainsMap files.

    Attributes
    ----------
    filename, signal_dict, _work_dict, _list_sur_file_content, _Object_type,
    _N_data_object, _N_data_channels,

    Methods
    -------
    parse_file, parse_header, get_image_dictionaries

    Class Variables
    ---------------
    _object_type : dict key: int containing the mountainsmap object types

    """

    # Object types
    _mountains_object_types = {
        -1: "_ERROR",
        0: "_UNKNOWN",
        1: "_PROFILE",
        2: "_SURFACE",
        3: "_BINARYIMAGE",
        4: "_PROFILESERIE",
        5: "_SURFACESERIE",
        6: "_MERIDIANDISC",
        7: "_MULTILAYERPROFILE",
        8: "_MULTILAYERSURFACE",
        9: "_PARALLELDISC",  # not implemented
        10: "_INTENSITYIMAGE",
        11: "_INTENSITYSURFACE",
        12: "_RGBIMAGE",
        13: "_RGBSURFACE",  # Deprecated
        14: "_FORCECURVE",  # Deprecated
        15: "_SERIEOFFORCECURVE",  # Deprecated
        16: "_RGBINTENSITYSURFACE",  # Surface + Image
        17: "_CONTOURPROFILE",
        18: "_SERIESOFRGBIMAGES",
        20: "_SPECTRUM",
        21: "_HYPCARD",
    }

    def __init__(self, filename: str):
        # We do not need to check for file existence here because
        # io module implements it in the load function
        self.filename = filename

        # The signal_dict dictionnary has to be returned by the
        # file_reader function. By default, we return the minimal
        # mandatory fields
        self.signal_dict = {
            "data": np.empty((0, 0, 0)),
            "axes": [],
            "metadata": {},
            "original_metadata": {},
        }

        # Dictionary to store, read and write fields in the binary file
        # defined in the MountainsMap SDK. Structure is
        # _work_dict['Field']['value'] : access field value
        # _work_dict['Field']['b_unpack_fn'](f) : unpack value from file f
        # _work_dict['Field']['b_pack_fn'](f,v): pack value v in file f
        self._work_dict = {
            "_01_Signature": {
                "value": "DSCOMPRESSED",  # Uncompressed key is DIGITAL SURF
                "b_unpack_fn": lambda f: self._get_str(f, 12),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 12),
            },
            "_02_Format": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_03_Number_of_Objects": {
                "value": 1,
                "b_unpack_fn": self._get_uint16,
                "b_pack_fn": self._set_uint16,
            },
            "_04_Version": {
                "value": 1,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_05_Object_Type": {
                "value": 2,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_06_Object_Name": {
                "value": "",
                "b_unpack_fn": lambda f: self._get_str(
                    f,
                    30,
                ),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 30),
            },
            "_07_Operator_Name": {
                "value": "ROSETTA",
                "b_unpack_fn": lambda f: self._get_str(
                    f,
                    30,
                ),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 30),
            },
            "_08_P_Size": {
                "value": 1,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_09_Acquisition_Type": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_10_Range_Type": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_11_Special_Points": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_12_Absolute": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_13_Gauge_Resolution": {
                "value": 0.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_14_W_Size": {
                "value": 0,
                "b_unpack_fn": self._get_uint32,
                "b_pack_fn": self._set_uint32,
            },
            "_15_Size_of_Points": {
                "value": 16,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_16_Zmin": {
                "value": 0,
                "b_unpack_fn": self._get_int32,
                "b_pack_fn": self._set_int32,
            },
            "_17_Zmax": {
                "value": 0,
                "b_unpack_fn": self._get_int32,
                "b_pack_fn": self._set_int32,
            },
            "_18_Number_of_Points": {
                "value": 1,
                "b_unpack_fn": self._get_int32,
                "b_pack_fn": self._set_int32,
            },
            "_19_Number_of_Lines": {
                "value": 1,
                "b_unpack_fn": self._get_int32,
                "b_pack_fn": self._set_int32,
            },
            "_20_Total_Nb_of_Pts": {
                "value": 1,
                "b_unpack_fn": self._get_int32,
                "b_pack_fn": self._set_int32,
            },
            "_21_X_Spacing": {
                "value": 1.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_22_Y_Spacing": {
                "value": 1.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_23_Z_Spacing": {
                "value": 1.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_24_Name_of_X_Axis": {
                "value": "X",
                "b_unpack_fn": lambda f: self._get_str(f, 16),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 16),
            },
            "_25_Name_of_Y_Axis": {
                "value": "Y",
                "b_unpack_fn": lambda f: self._get_str(f, 16),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 16),
            },
            "_26_Name_of_Z_Axis": {
                "value": "Z",
                "b_unpack_fn": lambda f: self._get_str(f, 16),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 16),
            },
            "_27_X_Step_Unit": {
                "value": "um",
                "b_unpack_fn": lambda f: self._get_str(f, 16),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 16),
            },
            "_28_Y_Step_Unit": {
                "value": "um",
                "b_unpack_fn": lambda f: self._get_str(f, 16),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 16),
            },
            "_29_Z_Step_Unit": {
                "value": "um",
                "b_unpack_fn": lambda f: self._get_str(f, 16),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 16),
            },
            "_30_X_Length_Unit": {
                "value": "um",
                "b_unpack_fn": lambda f: self._get_str(f, 16),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 16),
            },
            "_31_Y_Length_Unit": {
                "value": "um",
                "b_unpack_fn": lambda f: self._get_str(f, 16),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 16),
            },
            "_32_Z_Length_Unit": {
                "value": "um",
                "b_unpack_fn": lambda f: self._get_str(f, 16),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 16),
            },
            "_33_X_Unit_Ratio": {
                "value": 1.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_34_Y_Unit_Ratio": {
                "value": 1.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_35_Z_Unit_Ratio": {
                "value": 1.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_36_Imprint": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_37_Inverted": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_38_Levelled": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_39_Obsolete": {
                "value": b"",
                "b_unpack_fn": lambda f: self._get_bytes(f, 12),
                "b_pack_fn": lambda f, v: self._set_bytes(f, v, 12),
            },
            "_40_Seconds": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_41_Minutes": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_42_Hours": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_43_Day": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_44_Month": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_45_Year": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_46_Day_of_week": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_47_Measurement_duration": {
                "value": 0.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_48_Compressed_data_size": {
                "value": 0,
                "b_unpack_fn": self._get_uint32,
                "b_pack_fn": self._set_uint32,
            },
            "_49_Obsolete": {
                "value": b"",
                "b_unpack_fn": lambda f: self._get_bytes(f, 6),
                "b_pack_fn": lambda f, v: self._set_bytes(f, v, 6),
            },
            "_50_Comment_size": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_51_Private_size": {
                "value": 0,
                "b_unpack_fn": self._get_int16,
                "b_pack_fn": self._set_int16,
            },
            "_52_Client_zone": {
                "value": b"",
                "b_unpack_fn": lambda f: self._get_bytes(f, 128),
                "b_pack_fn": lambda f, v: self._set_bytes(f, v, 128),
            },
            "_53_X_Offset": {
                "value": 0.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_54_Y_Offset": {
                "value": 0.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_55_Z_Offset": {
                "value": 0.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_56_T_Spacing": {
                "value": 0.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_57_T_Offset": {
                "value": 0.0,
                "b_unpack_fn": self._get_float,
                "b_pack_fn": self._set_float,
            },
            "_58_T_Axis_Name": {
                "value": "T",
                "b_unpack_fn": lambda f: self._get_str(f, 13),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 13),
            },
            "_59_T_Step_Unit": {
                "value": "um",
                "b_unpack_fn": lambda f: self._get_str(f, 13),
                "b_pack_fn": lambda f, v: self._set_str(f, v, 13),
            },
            "_60_Comment": {
                "value": 0,
                "b_unpack_fn": self._unpack_comment,
                "b_pack_fn": self._pack_comment,
            },
            "_61_Private_zone": {
                "value": b"",
                "b_unpack_fn": self._unpack_private,
                "b_pack_fn": self._pack_private,
            },
            "_62_points": {
                "value": 0,
                "b_unpack_fn": self._unpack_data,
                "b_pack_fn": self._pack_data,
            },
        }

        # List of all measurement
        self._list_sur_file_content = []

        # The surface files convention is that when saving multiple data
        # objects at once, they are all packed in the same binary file.
        # Every single object contains a full header with all the sections,
        # but only the first one contains the relevant infos about
        # object type, the number of objects in the file and other.
        # Hence they will be made attributes.
        # Object type
        self._Object_type = "_UNKNOWN"

        # Number of data objects in the file.
        self._N_data_objects = 1
        self._N_data_channels = 1

        # Attributes useful for save and export

        # Number of nav / sig axes
        self._n_ax_nav: int = 0
        self._n_ax_sig: int = 0

        # All as a rsciio-convention axis dict or empty
        self.Xaxis: dict = {}
        self.Yaxis: dict = {}
        self.Zaxis: dict = {}
        self.Taxis: dict = {}

        # These must be set in the split functions
        self.data_split = []
        self.objtype_split = []

    # File Writer Inner methods

    def _write_sur_file(self):
        """Write self._list_sur_file_content to a file. This method is
        start-and-forget. The brainwork is performed in the construction
        of sur_file_content list of dictionaries."""

        with open(self.filename, "wb") as f:
            for dic in self._list_sur_file_content:
                # Extremely important! self._work_dict must access
                # other fields to properly encode and decode data,
                # comments etc. etc.
                self._move_values_to_workdict(dic)
                # Then inner consistency is trivial
                for key in self._work_dict:
                    self._work_dict[key]["b_pack_fn"](f, self._work_dict[key]["value"])

    def _build_sur_file_contents(
        self,
        set_comments: str = "auto",
        is_special: bool = False,
        compressed: bool = True,
        comments: dict = {},
        object_name: str = "",
        operator_name: str = "",
        absolute: int = 0,
        private_zone: bytes = b"",
        client_zone: bytes = b"",
    ):
        """Build the _sur_file_content list necessary to write a signal dictionary to
        a ``.sur`` or ``.pro`` file. The signal dictionary's inner consistency is the
        responsibility of hyperspy, and the this function's responsibility is to make
        a consistent list of _sur_file_content."""

        self._list_sur_file_content = []

        # Compute number of navigation / signal axes
        self._n_ax_nav, self._n_ax_sig = DigitalSurfHandler._get_n_axes(
            self.signal_dict
        )

        # Choose object type based on number of navigation and signal axes
        # Populate self._Object_type
        # Populate self.Xaxis, self.Yaxis, self.Taxis (if not empty)
        # Populate self.data_split and self.objtype_split (always)
        self._split_signal_dict()

        # Raise error if wrong extension
        # self._validate_filename()

        # Get a dictionary to be saved in the comment fielt of exported file
        comment_dict = self._get_comment_dict(
            self.signal_dict["original_metadata"], method=set_comments, custom=comments
        )
        # Convert the dictionary to a string of suitable format.
        comment_str = self._stringify_dict(comment_dict)

        # A _work_dict is created for each of the data arrays and object
        # that have splitted from the main object. In most cases, only a
        # single object is present in the split.
        for data, objtype in zip(self.data_split, self.objtype_split):
            self._build_workdict(
                data,
                objtype,
                self.signal_dict["metadata"],
                comment=comment_str,
                is_special=is_special,
                compressed=compressed,
                object_name=object_name,
                operator_name=operator_name,
                absolute=absolute,
                private_zone=private_zone,
                client_zone=client_zone,
            )
            # if the objects are multiple, comment is erased after the first
            # object. This is not mandatory, but makes marginally smaller files.
            if comment_str:
                comment_str = ""

            # Finally we push it all to the content list.
            self._append_work_dict_to_content()

    # Signal dictionary analysis methods
    @staticmethod
    def _get_n_axes(sig_dict: dict):
        """Return number of navigation and signal axes in the signal dict (in that order).
        Could be moved away from the .sur api as other functions probably use this as well

        Args:
            sig_dict (dict): signal dict, has to contain keys: 'data', 'axes', 'metadata'

        Returns:
            Tuple[int,int]: nax_nav,nax_sig. Number of navigation and signal axes
        """
        nax_nav = 0
        nax_sig = 0
        for ax in sig_dict["axes"]:
            if ax["navigate"]:
                nax_nav += 1
            else:
                nax_sig += 1
        return nax_nav, nax_sig

    def _is_spectrum(self) -> bool:
        """Determine if a signal is a spectrum type based on axes naming
        for export of sur_files. Could be cross-checked with other criteria
        such as hyperspy subclass etc... For now we keep it simple. If it has
        an ax named like a spectral axis, then probably its a spectrum."""

        spectrumlike_axnames = ["Wavelength", "Energy", "Energy Loss", "E"]
        is_spec = False

        for ax in self.signal_dict["axes"]:
            if ax["name"] in spectrumlike_axnames:
                is_spec = True

        return is_spec

    def _is_binary(self) -> bool:
        return self.signal_dict["data"].dtype == bool

    # Splitting /subclassing methods
    def _split_signal_dict(self):
        """Select the suitable _mountains_object_types"""

        n_nav = self._n_ax_nav
        n_sig = self._n_ax_sig

        # Here, I manually unfold the nested conditions for legibility.
        # Since there are a fixed number of dimensions supported by
        # digitalsurf .sur/.pro files, I think this is the best way to
        # proceed.
        if (n_nav, n_sig) == (0, 1):
            if self._is_spectrum():
                self._split_spectrum()
            else:
                self._split_profile()
        elif (n_nav, n_sig) == (0, 2):
            if self._is_binary():
                self._split_binary_img()
            elif is_rgb(self.signal_dict["data"]):  # "_RGBIMAGE"
                self._split_rgb()
            elif is_rgba(self.signal_dict["data"]):
                warnings.warn(
                    "A channel discarded upon saving \
                              RGBA signal in .sur format"
                )
                self._split_rgb()
            else:  # _INTENSITYSURFACE
                self._split_surface()
        elif (n_nav, n_sig) == (1, 0):
            warnings.warn(
                f"Exporting surface signal dimension {n_sig} and navigation dimension \
                          {n_nav} falls back on profile type but is not good practice. Consider \
                          transposing before saving to avoid unexpected behaviour."
            )
            self._split_profile()
        elif (n_nav, n_sig) == (1, 1):
            if self._is_spectrum():
                self._split_spectrum()
            else:
                self._split_profileserie()
        elif (n_nav, n_sig) == (1, 2):
            if is_rgb(self.signal_dict["data"]):
                self._split_rgbserie()
            elif is_rgba(self.signal_dict["data"]):
                warnings.warn(
                    "Alpha channel discarded upon saving RGBA signal in .sur format"
                )
                self._split_rgbserie()
            else:
                self._split_surfaceserie()
        elif (n_nav, n_sig) == (2, 0):
            warnings.warn(
                f"Signal dimension {n_sig} and navigation dimension {n_nav} exported "
                "as surface type. Consider transposing signal object before exporting "
                "if this is intentional."
            )
            if self._is_binary():
                self._split_binary_img()
            elif is_rgb(self.signal_dict["data"]):  # "_RGBIMAGE"
                self._split_rgb()
            elif is_rgba(self.signal_dict["data"]):
                warnings.warn(
                    "A channel discarded upon saving \
                            RGBA signal in .sur format"
                )
                self._split_rgb()
            else:
                self._split_surface()
        elif (n_nav, n_sig) == (2, 1):
            self._split_hyperspectral()
        else:
            raise MountainsMapFileError(
                msg=f"Object with signal dimension {n_sig} and navigation dimension {n_nav} not supported for .sur export"
            )

    def _split_spectrum(
        self,
    ):
        """Set _Object_type, axes except Z, data_split, objtype_split _N_data_objects, _N_data_channels"""
        # When splitting spectrum, no series axis (T/W),
        # X axis is the spectral dimension and Y the series dimension (if series).
        obj_type = 20
        self._Object_type = self._mountains_object_types[obj_type]

        nax_nav = self._n_ax_nav
        nax_sig = self._n_ax_sig

        # _split_signal_dict ensures that the correct dims are sent here.
        if (nax_nav, nax_sig) == (0, 1) or (nax_nav, nax_sig) == (1, 0):
            self.Xaxis = self.signal_dict["axes"][0]
        elif (nax_nav, nax_sig) == (1, 1):
            self.Xaxis = next(
                ax for ax in self.signal_dict["axes"] if not ax["navigate"]
            )
            self.Yaxis = next(ax for ax in self.signal_dict["axes"] if ax["navigate"])

        self.data_split = [self.signal_dict["data"]]
        self.objtype_split = [obj_type]
        self._N_data_objects = 1
        self._N_data_channels = 1

    def _split_profile(
        self,
    ):
        """Set _Object_type, axes except Z, data_split, objtype_split _N_data_objects, _N_data_channels"""

        obj_type = 1
        self._Object_type = self._mountains_object_types[obj_type]
        self.Xaxis = self.signal_dict["axes"][0]
        self.data_split = [self.signal_dict["data"]]
        self.objtype_split = [obj_type]
        self._N_data_objects = 1
        self._N_data_channels = 1

    def _split_profileserie(
        self,
    ):
        """Set _Object_type, axes except Z, data_split, objtype_split _N_data_objects, _N_data_channels"""
        obj_type = 4  # '_PROFILESERIE'
        self._Object_type = self._mountains_object_types[obj_type]

        self.Xaxis = next(ax for ax in self.signal_dict["axes"] if not ax["navigate"])
        self.Taxis = next(ax for ax in self.signal_dict["axes"] if ax["navigate"])

        self.data_split = self._split_data_alongaxis(self.Taxis)
        self.objtype_split = [obj_type] + [1] * (len(self.data_split) - 1)
        self._N_data_objects = len(self.objtype_split)
        self._N_data_channels = 1

    def _split_binary_img(
        self,
    ):
        """Set _Object_type, axes except Z, data_split, objtype_split _N_data_objects, _N_data_channels"""
        obj_type = 3
        self._Object_type = self._mountains_object_types[obj_type]

        self.Xaxis = self.signal_dict["axes"][1]
        self.Yaxis = self.signal_dict["axes"][0]

        self.data_split = [self.signal_dict["data"]]
        self.objtype_split = [obj_type]
        self._N_data_objects = 1
        self._N_data_channels = 1

    def _split_rgb(
        self,
    ):
        """Set _Object_type, axes except Z, data_split, objtype_split _N_data_objects, _N_data_channels"""
        obj_type = 12
        self._Object_type = self._mountains_object_types[obj_type]
        self.Xaxis = self.signal_dict["axes"][1]
        self.Yaxis = self.signal_dict["axes"][0]
        self.data_split = [
            np.int32(self.signal_dict["data"]["R"]),
            np.int32(self.signal_dict["data"]["G"]),
            np.int32(self.signal_dict["data"]["B"]),
        ]
        self.objtype_split = [obj_type] + [10, 10]
        self._N_data_objects = 1
        self._N_data_channels = 3

    def _split_surface(
        self,
    ):
        """Set _Object_type, axes except Z, data_split, objtype_split _N_data_objects, _N_data_channels"""
        obj_type = 2
        self._Object_type = self._mountains_object_types[obj_type]
        self.Xaxis = self.signal_dict["axes"][1]
        self.Yaxis = self.signal_dict["axes"][0]
        self.data_split = [self.signal_dict["data"]]
        self.objtype_split = [obj_type]
        self._N_data_objects = 1
        self._N_data_channels = 1

    def _split_rgbserie(self):
        """Set _Object_type, axes except Z, data_split, objtype_split _N_data_objects, _N_data_channels"""
        obj_type = 18  # "_SERIESOFRGBIMAGE"
        self._Object_type = self._mountains_object_types[obj_type]

        sigaxes_iter = iter(ax for ax in self.signal_dict["axes"] if not ax["navigate"])
        self.Yaxis = next(sigaxes_iter)
        self.Xaxis = next(sigaxes_iter)
        self.Taxis = next(ax for ax in self.signal_dict["axes"] if ax["navigate"])
        tmp_data_split = self._split_data_alongaxis(self.Taxis)

        # self.data_split = []
        self.objtype_split = []
        for d in tmp_data_split:
            self.data_split += [
                d["R"].astype(np.int16).copy(),
                d["G"].astype(np.int16).copy(),
                d["B"].astype(np.int16).copy(),
            ]
            # self.objtype_split += [12,10,10]
        self.objtype_split = [12, 10, 10] * self.Taxis["size"]
        self.objtype_split[0] = obj_type
        # self.data_split = rgbx2regular_array(self.signal_dict['data'])

        self._N_data_objects = self.Taxis["size"]
        self._N_data_channels = 3

    def _split_surfaceserie(self):
        """Set _Object_type, axes except Z, data_split, objtype_split _N_data_objects, _N_data_channels"""
        obj_type = 5
        self._Object_type = self._mountains_object_types[obj_type]
        sigaxes_iter = iter(ax for ax in self.signal_dict["axes"] if not ax["navigate"])
        self.Yaxis = next(sigaxes_iter)
        self.Xaxis = next(sigaxes_iter)
        self.Taxis = next(ax for ax in self.signal_dict["axes"] if ax["navigate"])
        self.data_split = self._split_data_alongaxis(self.Taxis)
        self.objtype_split = [2] * len(self.data_split)
        self.objtype_split[0] = obj_type
        self._N_data_objects = len(self.data_split)
        self._N_data_channels = 1

    def _split_hyperspectral(self):
        """Set _Object_type, axes except Z, data_split, objtype_split _N_data_objects, _N_data_channels"""
        obj_type = 21
        self._Object_type = self._mountains_object_types[obj_type]
        sigaxes_iter = iter(ax for ax in self.signal_dict["axes"] if ax["navigate"])
        self.Yaxis = next(sigaxes_iter)
        self.Xaxis = next(sigaxes_iter)
        self.Taxis = next(ax for ax in self.signal_dict["axes"] if not ax["navigate"])
        self.data_split = [self.signal_dict["data"]]
        self.objtype_split = [obj_type]
        self._N_data_objects = 1
        self._N_data_channels = 1

    def _split_data_alongaxis(self, axis: dict):
        """Split the data in a series of lower-dim datasets that can be exported to
        a surface / profile file"""
        idx = self.signal_dict["axes"].index(axis)
        # return idx
        datasplit = []
        for dslice in np.rollaxis(self.signal_dict["data"], idx):
            datasplit.append(dslice)
        return datasplit

    def _norm_data(self, data: np.ndarray, is_special: bool):
        """Normalize input data to 16-bits or 32-bits ints and initialize an axis on which the data is normalized.

        Args:
            data (np.ndarray): dataset
            is_special (bool): whether NaNs get sent to N.M points in the sur format and apply saturation

        Raises:
            MountainsMapFileError: raised if input is of complex type
            MountainsMapFileError: raised if input is of unsigned int type
            MountainsMapFileError: raised if input is of int > 32 bits type

        Returns:
            tuple[int,int,int,float,float,np.ndarray[int]]: pointsize, Zmin, Zmax, Zscale, Zoffset, data_int
        """
        data_type = data.dtype

        if np.issubdtype(data_type, np.complexfloating):
            raise MountainsMapFileError(
                "digitalsurf file formats do not support export of complex data. Convert data to real-value representations before before export"
            )
        elif np.issubdtype(data_type, bool):
            pointsize = 16
            Zmin = 0
            Zmax = 1
            Zscale = 1
            Zoffset = 0
            data_int = data.astype(np.int16)
        elif data_type == np.uint8:
            warnings.warn("np.uint8 datatype exported as np.int16.")
            pointsize = 16
            Zmin, Zmax, Zscale, Zoffset = self._norm_signed_int(data, is_special)
            data_int = data.astype(np.int16)
        elif data_type == np.uint16:
            warnings.warn("np.uint16 datatype exported as np.int32")
            pointsize = 32  # Pointsize has to be 16 or 32 in surf format
            Zmin, Zmax, Zscale, Zoffset = self._norm_signed_int(data, is_special)
            data_int = data.astype(np.int32)
        elif np.issubdtype(data_type, np.unsignedinteger):
            raise MountainsMapFileError(
                "digitalsurf file formats do not support unsigned int >16bits. Convert data to signed integers before export."
            )
        elif data_type == np.int8:
            pointsize = 16  # Pointsize has to be 16 or 32 in surf format
            Zmin, Zmax, Zscale, Zoffset = self._norm_signed_int(data, is_special)
            data_int = data.astype(np.int16)
        elif data_type == np.int16:
            pointsize = 16
            Zmin, Zmax, Zscale, Zoffset = self._norm_signed_int(data, is_special)
            data_int = data
        elif data_type == np.int32:
            pointsize = 32
            data_int = data
            Zmin, Zmax, Zscale, Zoffset = self._norm_signed_int(data, is_special)
        elif np.issubdtype(data_type, np.integer):
            raise MountainsMapFileError(
                "digitalsurf file formats do not support export integers larger than 32 bits. Convert data to 32-bit representation before exporting"
            )
        elif np.issubdtype(data_type, np.floating):
            pointsize = 32
            Zmin, Zmax, Zscale, Zoffset, data_int = self._norm_float(data, is_special)

        return pointsize, Zmin, Zmax, Zscale, Zoffset, data_int

    def _norm_signed_int(self, data: np.ndarray, is_special: bool = False):
        """Normalized data of integer type. No normalization per se, but the Zmin and Zmax
        threshold are set if saturation flagging is asked."""
        # There are no NaN values for integers. Special points means saturation of integer scale.
        data_int_min = np.iinfo(data.dtype).min
        data_int_max = np.iinfo(data.dtype).max

        is_satlo = (data == data_int_min).sum() >= 1 and is_special
        is_sathi = (data == data_int_max).sum() >= 1 and is_special

        Zmin = data_int_min + 1 if is_satlo else data.min()
        Zmax = data_int_max - 1 if is_sathi else data.max()
        Zscale = 1.0
        Zoffset = Zmin

        return Zmin, Zmax, Zscale, Zoffset

    def _norm_float(
        self,
        data: np.ndarray,
        is_special: bool = False,
    ):
        """Normalize float data on a 32 bits int scale. Inherently lossy
        but that's how things are with mountainsmap files."""

        Zoffset_f = np.nanmin(data)
        Zmax_f = np.nanmax(data)
        is_nan = np.any(np.isnan(data))

        if is_special and is_nan:
            Zmin = -(2 ** (32 - 1)) + 2
            Zmax = 2**32 + Zmin - 3
        else:
            Zmin = -(2 ** (32 - 1))
            Zmax = 2**32 + Zmin - 1

        Zscale = (Zmax_f - Zoffset_f) / (Zmax - Zmin)
        data_int = (data - Zoffset_f) / Zscale + Zmin

        if is_special and is_nan:
            data_int[np.isnan(data)] = Zmin - 2

        data_int = data_int.astype(np.int32)

        return Zmin, Zmax, Zscale, Zoffset_f, data_int

    def _get_Zname_Zunit(self, metadata: dict):
        """Attempt reading Z-axis name and Unit from metadata.Signal.Quantity field.
        Return empty str if do not exist.

        Returns:
            tuple[str,str]: Zname,Zunit
        """
        quantitystr: str = metadata.get("Signal", {}).get("quantity", "")
        quantitystr = quantitystr.strip()
        quantity = quantitystr.split(" ")
        if len(quantity) > 1:
            Zunit = quantity.pop()
            Zunit = Zunit.strip("()")
            Zname = " ".join(quantity)
        elif len(quantity) == 1:
            Zname = quantity.pop()
            Zunit = ""

        return Zname, Zunit

    def _build_workdict(
        self,
        data: np.ndarray,
        obj_type: int,
        metadata: dict = {},
        comment: str = "",
        is_special: bool = True,
        compressed: bool = True,
        object_name: str = "",
        operator_name: str = "",
        absolute: int = 0,
        private_zone: bytes = b"",
        client_zone: bytes = b"",
    ):
        """Populate _work_dict with the"""

        if not compressed:
            self._work_dict["_01_Signature"]["value"] = (
                "DIGITAL SURF"  # DSCOMPRESSED by default
            )
        else:
            self._work_dict["_01_Signature"]["value"] = (
                "DSCOMPRESSED"  # DSCOMPRESSED by default
            )

        # self._work_dict['_02_Format']['value'] = 0 # Dft. other possible value is 257 for MacintoshII computers with Motorola CPUs. Obv not supported...
        self._work_dict["_03_Number_of_Objects"]["value"] = self._N_data_objects
        # self._work_dict['_04_Version']['value'] = 1 # Version number. Always default.
        self._work_dict["_05_Object_Type"]["value"] = obj_type
        self._work_dict["_06_Object_Name"]["value"] = (
            object_name  # Obsolete, DOS-version only (Not supported)
        )
        self._work_dict["_07_Operator_Name"]["value"] = (
            operator_name  # Should be settable from kwargs
        )
        self._work_dict["_08_P_Size"]["value"] = self._N_data_channels

        self._work_dict["_09_Acquisition_Type"]["value"] = (
            0  # AFM data only, could be inferred
        )
        self._work_dict["_10_Range_Type"]["value"] = (
            0  # Only 1 for high-range (z-stage scanning), AFM data only, could be inferred
        )

        self._work_dict["_11_Special_Points"]["value"] = int(is_special)

        self._work_dict["_12_Absolute"]["value"] = (
            absolute  # Probably irrelevant in most cases. Absolute vs rel heights (for profilometers), can be inferred
        )
        self._work_dict["_13_Gauge_Resolution"]["value"] = (
            0.0  # Probably irrelevant. Only for profilometers (maybe AFM), can be inferred
        )

        # T-axis acts as W-axis for spectrum / hyperspectrum surfaces.
        if obj_type in [21]:
            ws = self.Taxis.get("size", 0)
        else:
            ws = 0
        self._work_dict["_14_W_Size"]["value"] = ws

        bsize, Zmin, Zmax, Zscale, Zoffset, data_int = self._norm_data(data, is_special)
        Zname, Zunit = self._get_Zname_Zunit(metadata)

        # Axes element set regardless of object size
        self._work_dict["_15_Size_of_Points"]["value"] = bsize
        self._work_dict["_16_Zmin"]["value"] = Zmin
        self._work_dict["_17_Zmax"]["value"] = Zmax
        self._work_dict["_18_Number_of_Points"]["value"] = self.Xaxis.get("size", 1)
        self._work_dict["_19_Number_of_Lines"]["value"] = self.Yaxis.get("size", 1)
        # This needs to be this way due to the way we export our hyp maps
        self._work_dict["_20_Total_Nb_of_Pts"]["value"] = self.Xaxis.get(
            "size", 1
        ) * self.Yaxis.get("size", 1)

        self._work_dict["_21_X_Spacing"]["value"] = self.Xaxis.get("scale", 0.0)
        self._work_dict["_22_Y_Spacing"]["value"] = self.Yaxis.get("scale", 0.0)
        self._work_dict["_23_Z_Spacing"]["value"] = Zscale
        self._work_dict["_24_Name_of_X_Axis"]["value"] = self.Xaxis.get("name", "")
        self._work_dict["_25_Name_of_Y_Axis"]["value"] = self.Yaxis.get("name", "")
        self._work_dict["_26_Name_of_Z_Axis"]["value"] = Zname
        self._work_dict["_27_X_Step_Unit"]["value"] = self.Xaxis.get("units", "")
        self._work_dict["_28_Y_Step_Unit"]["value"] = self.Yaxis.get("units", "")
        self._work_dict["_29_Z_Step_Unit"]["value"] = Zunit
        self._work_dict["_30_X_Length_Unit"]["value"] = self.Xaxis.get("units", "")
        self._work_dict["_31_Y_Length_Unit"]["value"] = self.Yaxis.get("units", "")
        self._work_dict["_32_Z_Length_Unit"]["value"] = Zunit
        self._work_dict["_33_X_Unit_Ratio"]["value"] = 1
        self._work_dict["_34_Y_Unit_Ratio"]["value"] = 1
        self._work_dict["_35_Z_Unit_Ratio"]["value"] = 1

        # _36_Imprint  -> Obsolete
        # _37_Inverted -> Always No
        # _38_Levelled -> Always No
        # _39_Obsolete -> Obsolete

        dt: datetime.datetime = get_date_time_from_metadata(
            metadata, formatting="datetime"
        )
        if dt is not None:
            self._work_dict["_40_Seconds"]["value"] = dt.second
            self._work_dict["_41_Minutes"]["value"] = dt.minute
            self._work_dict["_42_Hours"]["value"] = dt.hour
            self._work_dict["_43_Day"]["value"] = dt.day
            self._work_dict["_44_Month"]["value"] = dt.month
            self._work_dict["_45_Year"]["value"] = dt.year
            self._work_dict["_46_Day_of_week"]["value"] = dt.weekday()

        # _47_Measurement_duration -> Nonsaved and non-metadata, but float in seconds

        if compressed:
            data_bin = self._compress_data(
                data_int, nstreams=1
            )  # nstreams hard-set to 1. Could be unlocked in the future
            compressed_size = len(data_bin)
        else:
            fmt = (
                "<h" if self._work_dict["_15_Size_of_Points"]["value"] == 16 else "<i"
            )  # select between short and long integers
            data_bin = data_int.ravel().astype(fmt).tobytes()
            compressed_size = 0

        self._work_dict["_48_Compressed_data_size"]["value"] = (
            compressed_size  # Obsolete in case of non-compressed
        )

        # _49_Obsolete

        comment_len = len(f"{comment}".encode("latin-1"))
        if comment_len >= 2**15:
            warnings.warn("Comment exceeding max length of 32.0 kB and will be cropped")
            comment_len = np.int16(2**15 - 1)

        self._work_dict["_50_Comment_size"]["value"] = comment_len

        privatesize = len(private_zone)
        if privatesize >= 2**15:
            warnings.warn(
                "Private size exceeding max length of 32.0 kB and will be cropped"
            )
            privatesize = np.uint16(2**15 - 1)

        self._work_dict["_51_Private_size"]["value"] = privatesize

        self._work_dict["_52_Client_zone"]["value"] = client_zone

        self._work_dict["_53_X_Offset"]["value"] = self.Xaxis.get("offset", 0.0)
        self._work_dict["_54_Y_Offset"]["value"] = self.Yaxis.get("offset", 0.0)
        self._work_dict["_55_Z_Offset"]["value"] = Zoffset
        self._work_dict["_56_T_Spacing"]["value"] = self.Taxis.get("scale", 0.0)
        self._work_dict["_57_T_Offset"]["value"] = self.Taxis.get("offset", 0.0)
        self._work_dict["_58_T_Axis_Name"]["value"] = self.Taxis.get("name", "")
        self._work_dict["_59_T_Step_Unit"]["value"] = self.Taxis.get("units", "")

        self._work_dict["_60_Comment"]["value"] = comment

        self._work_dict["_61_Private_zone"]["value"] = private_zone
        self._work_dict["_62_points"]["value"] = data_bin

    # Read methods
    def _read_sur_file(self):
        """Read the binary, possibly compressed, content of the surface
        file. Surface files can be encoded as single or a succession
        of objects. The file is thus read iteratively and from metadata of the
        first file"""

        with open(self.filename, "rb") as f:
            # We read the first object
            self._read_single_sur_object(f)
            # We append the first object to the content list
            self._append_work_dict_to_content()
            # Lookup how many objects are stored in the file and save
            self._N_data_objects = self._get_work_dict_key_value(
                "_03_Number_of_Objects"
            )
            self._N_data_channels = self._get_work_dict_key_value("_08_P_Size")

            # Determine how many objects we need to read, at least 1 object and 1 channel
            # even if metadata is set to 0 (happens sometimes)
            n_objects_to_read = max(self._N_data_channels, 1) * max(
                self._N_data_objects, 1
            )

            # Lookup what object type we are dealing with and save
            self._Object_type = DigitalSurfHandler._mountains_object_types[
                self._get_work_dict_key_value("_05_Object_Type")
            ]

            # if more than 1
            if n_objects_to_read > 1:
                # continue reading until everything is done
                for i in range(1, n_objects_to_read):
                    # We read an object
                    self._read_single_sur_object(f)
                    # We append it to content list
                    self._append_work_dict_to_content()

    def _read_single_sur_object(self, file):
        for key, val in self._work_dict.items():
            self._work_dict[key]["value"] = val["b_unpack_fn"](file)

    def _append_work_dict_to_content(self):
        """Save the values stored in the work dict in the surface file list"""
        datadict = deepcopy({key: val["value"] for key, val in self._work_dict.items()})
        self._list_sur_file_content.append(datadict)

    def _move_values_to_workdict(self, dic: dict):
        for key in self._work_dict:
            self._work_dict[key]["value"] = deepcopy(dic[key])

    def _get_work_dict_key_value(self, key):
        return self._work_dict[key]["value"]

    # Signal dictionary methods
    def _build_sur_dict(self):
        """Create a signal dict with an unpacked object"""

        # If the signal is of the type spectrum or hypercard
        if self._Object_type in ["_HYPCARD"]:
            self._build_hyperspectral_map()
        elif self._Object_type in ["_SPECTRUM"]:
            self._build_spectrum()
        elif self._Object_type in ["_PROFILE"]:
            self._build_general_1D_data()
        elif self._Object_type in ["_PROFILESERIE"]:
            self._build_1D_series()
        elif self._Object_type in ["_BINARYIMAGE"]:
            self._build_surface()
            self.signal_dict.update({"post_process": [self.post_process_binary]})
        elif self._Object_type in ["_SURFACE", "_INTENSITYIMAGE"]:
            self._build_surface()
        elif self._Object_type in ["_SURFACESERIE"]:
            self._build_surface_series()
        elif self._Object_type in ["_MULTILAYERSURFACE"]:
            self._build_surface_series()
        elif self._Object_type in ["_RGBSURFACE"]:
            self._build_RGB_surface()
        elif self._Object_type in ["_RGBIMAGE"]:
            self._build_RGB_image()
        elif self._Object_type in ["_RGBINTENSITYSURFACE"]:
            self._build_RGB_surface()
        elif self._Object_type in ["_SERIESOFRGBIMAGES"]:
            self._build_RGB_image_series()
        else:
            raise MountainsMapFileError(
                f"{self._Object_type} is not a supported mountain object."
            )

        return self.signal_dict

    @staticmethod
    def _build_Xax(unpacked_dict, ind=0, nav=False, binned=False):
        """Return X axis dictionary from an unpacked dict. index int and navigate
        boolean can be optionally passed. Default 0 and False respectively."""
        xax = {
            "name": unpacked_dict["_24_Name_of_X_Axis"],
            "size": unpacked_dict["_18_Number_of_Points"],
            "index_in_array": ind,
            "scale": unpacked_dict["_21_X_Spacing"],
            "offset": unpacked_dict["_53_X_Offset"],
            "units": unpacked_dict["_27_X_Step_Unit"],
            "navigate": nav,
            "is_binned": binned,
        }
        return xax

    @staticmethod
    def _build_Yax(unpacked_dict, ind=1, nav=False, binned=False):
        """Return X axis dictionary from an unpacked dict. index int and navigate
        boolean can be optionally passed. Default 1 and False respectively."""
        yax = {
            "name": unpacked_dict["_25_Name_of_Y_Axis"],
            "size": unpacked_dict["_19_Number_of_Lines"],
            "index_in_array": ind,
            "scale": unpacked_dict["_22_Y_Spacing"],
            "offset": unpacked_dict["_54_Y_Offset"],
            "units": unpacked_dict["_28_Y_Step_Unit"],
            "navigate": nav,
            "is_binned": binned,
        }
        return yax

    @staticmethod
    def _build_Tax(unpacked_dict, size_key, ind=0, nav=True, binned=False):
        """Return T axis dictionary from an unpacked surface object dict.
        Unlike x and y axes, the size key can be determined from various keys:
        _14_W_Size, _15_Size_of_Points or _03_Number_of_Objects. index int
        and navigate boolean can be optionally passed. Default 0 and
        True respectively."""

        # The T axis is somewhat special because it is only defined on series
        # and is thus only navigation. It is only defined on the first object
        # in a serie.
        # Here it needs to be checked that the T axis scale is not 0 Otherwise
        # it raises hyperspy errors
        scale = unpacked_dict["_56_T_Spacing"]
        if scale == 0:
            scale = 1

        tax = {
            "name": unpacked_dict["_58_T_Axis_Name"],
            "size": unpacked_dict[size_key],
            "index_in_array": ind,
            "scale": scale,
            "offset": unpacked_dict["_57_T_Offset"],
            "units": unpacked_dict["_59_T_Step_Unit"],
            "navigate": nav,
            "is_binned": binned,
        }
        return tax

    # Build methods for individual surface objects
    def _build_hyperspectral_map(
        self,
    ):
        """Build a hyperspectral map. Hyperspectral maps are single-object
        files with datapoints of _14_W_Size length"""

        # Check that the object contained only one object.
        # Probably overkill at this point but better safe than sorry
        if len(self._list_sur_file_content) != 1:
            raise MountainsMapFileError(
                "Input {:s} File is not of Hyperspectral type".format(self._Object_type)
            )

        # We get the dictionary with all the data
        hypdic = self._list_sur_file_content[0]

        # Add all the axes to the signal dict
        self.signal_dict["axes"].append(self._build_Yax(hypdic, ind=0, nav=True))
        self.signal_dict["axes"].append(self._build_Xax(hypdic, ind=1, nav=True))
        # Wavelength axis in hyperspectral surface files are stored as T Axis
        # with length set as _14_W_Size
        self.signal_dict["axes"].append(
            self._build_Tax(hypdic, "_14_W_Size", ind=2, nav=False)
        )

        # We reshape the data in the correct format
        self.signal_dict["data"] = hypdic["_62_points"].reshape(
            hypdic["_19_Number_of_Lines"],
            hypdic["_18_Number_of_Points"],
            hypdic["_14_W_Size"],
        )

        self._set_metadata_and_original_metadata(hypdic)

    def _build_general_1D_data(
        self,
    ):
        """Build general 1D Data objects. Currently work with spectra"""

        # Check that the object contained only one object.
        # Probably overkill at this point but better safe than sorry
        if len(self._list_sur_file_content) != 1:
            raise MountainsMapFileError("Corrupt file")

        # We get the dictionary with all the data
        hypdic = self._list_sur_file_content[0]

        # Add the axe to the signal dict
        self.signal_dict["axes"].append(self._build_Xax(hypdic, ind=0, nav=False))

        # We reshape the data in the correct format
        self.signal_dict["data"] = hypdic["_62_points"]

        # Build the metadata
        self._set_metadata_and_original_metadata(hypdic)

    def _build_spectrum(
        self,
    ):
        """Build spectra objects. Spectra and 1D series of spectra are
        saved in the same object."""

        # We get the dictionary with all the data
        hypdic = self._list_sur_file_content[0]

        # If there is more than 1 spectrum also add the navigation axis
        if hypdic["_19_Number_of_Lines"] != 1:
            self.signal_dict["axes"].append(self._build_Yax(hypdic, ind=0, nav=True))

        # Add the signal axis_src to the signal dict
        self.signal_dict["axes"].append(self._build_Xax(hypdic, ind=1, nav=False))

        # We reshape the data in the correct format.
        # Edit: the data is now squeezed for unneeded dimensions
        data_shape = (hypdic["_19_Number_of_Lines"], hypdic["_18_Number_of_Points"])
        data_array = np.squeeze(hypdic["_62_points"].reshape(data_shape, order="C"))

        self.signal_dict["data"] = data_array

        self._set_metadata_and_original_metadata(hypdic)

    def _build_1D_series(
        self,
    ):
        """Build a series of 1D objects. The T axis is navigation and set from
        the first object"""

        # First object dictionary
        hypdic = self._list_sur_file_content[0]

        # Metadata are set from first dictionary
        self._set_metadata_and_original_metadata(hypdic)

        # Add the series-axis to the signal dict
        self.signal_dict["axes"].append(
            self._build_Tax(hypdic, "_03_Number_of_Objects", ind=0, nav=True)
        )

        # All objects must share the same signal axis
        self.signal_dict["axes"].append(self._build_Xax(hypdic, ind=1, nav=False))

        # We put all the data together
        data = []
        for obj in self._list_sur_file_content:
            data.append(obj["_62_points"])

        self.signal_dict["data"] = np.stack(data)

    def _build_surface(
        self,
    ):
        """Build a surface"""

        # Check that the object contained only one object.
        # Probably overkill at this point but better safe than sorry
        if len(self._list_sur_file_content) != 1:
            raise MountainsMapFileError("CORRUPT {:s} FILE".format(self._Object_type))

        # We get the dictionary with all the data
        hypdic = self._list_sur_file_content[0]

        # Add all the axes to the signal dict
        self.signal_dict["axes"].append(self._build_Yax(hypdic, ind=0, nav=False))
        self.signal_dict["axes"].append(self._build_Xax(hypdic, ind=1, nav=False))

        # We reshape the data in the correct format
        shape = (hypdic["_19_Number_of_Lines"], hypdic["_18_Number_of_Points"])
        self.signal_dict["data"] = hypdic["_62_points"].reshape(shape)

        self._set_metadata_and_original_metadata(hypdic)

    def _build_surface_series(
        self,
    ):
        """Build a series of surfaces. The T axis is navigation and set from
        the first object"""

        # First object dictionary
        hypdic = self._list_sur_file_content[0]

        # Metadata are set from first dictionary
        self._set_metadata_and_original_metadata(hypdic)

        # Add the series-axis to the signal dict
        self.signal_dict["axes"].append(
            self._build_Tax(hypdic, "_03_Number_of_Objects", ind=0, nav=True)
        )

        # All objects must share the same signal axes
        self.signal_dict["axes"].append(self._build_Yax(hypdic, ind=1, nav=False))
        self.signal_dict["axes"].append(self._build_Xax(hypdic, ind=2, nav=False))

        # shape of the surfaces in the series
        shape = (hypdic["_19_Number_of_Lines"], hypdic["_18_Number_of_Points"])
        # We put all the data together
        data = []
        for obj in self._list_sur_file_content:
            data.append(obj["_62_points"].reshape(shape))

        self.signal_dict["data"] = np.stack(data)

    def _build_RGB_surface(
        self,
    ):
        """Build a series of surfaces. The T axis is navigation and set from
        P Size"""

        # First object dictionary
        hypdic = self._list_sur_file_content[0]

        # Metadata are set from first dictionary
        self._set_metadata_and_original_metadata(hypdic)

        # Add the series-axis to the signal dict
        self.signal_dict["axes"].append(
            self._build_Tax(hypdic, "_08_P_Size", ind=0, nav=True)
        )

        # All objects must share the same signal axes
        self.signal_dict["axes"].append(self._build_Yax(hypdic, ind=1, nav=False))
        self.signal_dict["axes"].append(self._build_Xax(hypdic, ind=2, nav=False))

        # shape of the surfaces in the series
        shape = (hypdic["_19_Number_of_Lines"], hypdic["_18_Number_of_Points"])
        # We put all the data together
        data = []
        for obj in self._list_sur_file_content:
            data.append(obj["_62_points"].reshape(shape))

        # Pushing data into the dictionary
        self.signal_dict["data"] = np.stack(data)

    def _build_RGB_image(
        self,
    ):
        """Build an RGB image. The T axis is navigation and set from
        P Size"""

        # First object dictionary
        hypdic = self._list_sur_file_content[0]

        # Metadata are set from first dictionary
        self._set_metadata_and_original_metadata(hypdic)

        # Add the series-axis to the signal dict
        self.signal_dict["axes"].append(
            self._build_Tax(hypdic, "_08_P_Size", ind=0, nav=True)
        )

        # All objects must share the same signal axes
        self.signal_dict["axes"].append(self._build_Yax(hypdic, ind=1, nav=False))
        self.signal_dict["axes"].append(self._build_Xax(hypdic, ind=2, nav=False))

        # shape of the surfaces in the series
        shape = (hypdic["_19_Number_of_Lines"], hypdic["_18_Number_of_Points"])
        # We put all the data together
        data = []
        for obj in self._list_sur_file_content:
            data.append(obj["_62_points"].reshape(shape))

        # Pushing data into the dictionary
        self.signal_dict["data"] = np.stack(data)

        self.signal_dict.update({"post_process": [self.post_process_RGB]})

    def _build_RGB_image_series(
        self,
    ):
        # First object dictionary
        hypdic = self._list_sur_file_content[0]

        # Metadata are set from first dictionary
        self._set_metadata_and_original_metadata(hypdic)

        # We build the series-axis
        self.signal_dict["axes"].append(
            self._build_Tax(hypdic, "_03_Number_of_Objects", ind=0, nav=False)
        )

        # All objects must share the same signal axes
        self.signal_dict["axes"].append(self._build_Yax(hypdic, ind=1, nav=False))
        self.signal_dict["axes"].append(self._build_Xax(hypdic, ind=2, nav=False))

        # shape of the surfaces in the series
        shape = (hypdic["_19_Number_of_Lines"], hypdic["_18_Number_of_Points"])
        nimg = hypdic["_03_Number_of_Objects"]
        nchan = hypdic["_08_P_Size"]
        # We put all the data together
        data = np.empty(shape=(nimg, *shape, nchan))
        i = 0
        for imgidx in range(nimg):
            for chanidx in range(nchan):
                obj = self._list_sur_file_content[i]
                data[imgidx, ..., chanidx] = obj["_62_points"].reshape(shape)
                i += 1

        # for obj in self._list_sur_file_content:
        #     data.append(obj["_62_points"].reshape(shape))

        # data = np.stack(data)

        # data = data.reshape(nimg,nchan,*shape)
        # data = np.rollaxis(data,)

        # Pushing data into the dictionary
        self.signal_dict["data"] = data

        # Add the color-axis to the signal dict so it can be consumed
        self.signal_dict["axes"].append(
            self._build_Tax(hypdic, "_08_P_Size", ind=3, nav=True)
        )

        self.signal_dict.update({"post_process": [self.post_process_RGB]})

    # Metadata utility methods

    @staticmethod
    def _choose_signal_type(unpacked_dict: dict) -> str:
        """Choose the correct signal type based on the header content"""
        if unpacked_dict.get("_26_Name_of_Z_Axis") in ["CL Intensity"]:
            return "CL"
        else:
            return ""

    def _build_generic_metadata(self, unpacked_dict):
        """Return a minimalistic metadata dictionary according to hyperspy
        format. Accept a dictionary as an input because dictionary with the
        headers of a mountians object.

        Parameters
        ----------
        unpacked_dict: dictionary from the header of a surface file

        Returns
        -------
        metadict: dictionnary in the hyperspy metadata format

        """

        # Formatting for complicated strings. We add parentheses to units
        qty_unit = unpacked_dict["_29_Z_Step_Unit"]
        # We strip unit from any character that might pre-format it
        qty_unit = qty_unit.strip(" \t\n()[]")
        # If unit string is still truthy after strip we add parentheses
        if qty_unit:
            qty_unit = "({:s})".format(qty_unit)

        quantity_str = " ".join([unpacked_dict["_26_Name_of_Z_Axis"], qty_unit]).strip()

        # Date and time are set in metadata only if all values are not set to 0

        date = [
            unpacked_dict["_45_Year"],
            unpacked_dict["_44_Month"],
            unpacked_dict["_43_Day"],
        ]
        if not all(v == 0 for v in date):
            date_str = "{:4d}-{:02d}-{:02d}".format(date[0], date[1], date[2])
        else:
            date_str = ""

        time = [
            unpacked_dict["_42_Hours"],
            unpacked_dict["_41_Minutes"],
            unpacked_dict["_40_Seconds"],
        ]

        if not all(v == 0 for v in time):
            time_str = "{:02d}:{:02d}:{:02d}".format(time[0], time[1], time[2])
        else:
            time_str = ""

        signal_type = self._choose_signal_type(unpacked_dict)

        # Metadata dictionary initialization
        metadict = {
            "General": {
                "authors": unpacked_dict["_07_Operator_Name"],
                "date": date_str,
                "original_filename": os.path.split(self.filename)[1],
                "time": time_str,
            },
            "Signal": {
                "quantity": quantity_str,
                "signal_type": signal_type,
            },
        }

        return metadict

    def _build_original_metadata(
        self,
    ):
        """Builds a metadata dictionary from the header"""
        original_metadata_dict = {}

        # Iteration over Number of data objects
        for i in range(self._N_data_objects):
            # Iteration over the Number of Data channels
            for j in range(max(self._N_data_channels, 1)):
                # Creating a dictionary key for each object
                k = (i + 1) * (j + 1)
                key = "Object_{:d}_Channel_{:d}".format(i, j)
                original_metadata_dict.update({key: {}})

                # We load one full object header
                a = self._list_sur_file_content[k - 1]

                # Save it as original metadata dictionary
                headerdict = {
                    "H" + k.lstrip("_"): a[k]
                    for k in a
                    if k not in ("_62_points", "_61_Private_zone")
                }

                original_metadata_dict[key].update({"Header": headerdict})

                # The second dictionary might contain custom mountainsmap params
                # Check if it is the case and append it to original metadata if yes
                valid_comment = self._check_comments(a["_60_Comment"], "$", "=")
                if valid_comment:
                    parsedict = parse_metadata(a["_60_Comment"], "$", "=")
                    parsedict = {k.lstrip("_"): m for k, m in parsedict.items()}
                    original_metadata_dict[key].update({"Parsed": parsedict})

        return original_metadata_dict

    def _build_signal_specific_metadata(
        self,
    ) -> dict:
        """Build additional metadata specific to signal type.
        return a dictionary for update in the metadata."""
        if self.signal_dict["metadata"]["Signal"]["signal_type"] == "CL":
            return self._map_CL_metadata()
        else:
            return {}

    def _map_SEM_metadata(self) -> dict:
        """Return SEM metadata according to hyperspy specifications"""
        atto_omd = self.signal_dict["original_metadata"]
        # get nested dictionaries in an error-handling way
        atto_omd = atto_omd.get("Object_0_Channel_0", {})
        atto_omd = atto_omd.get("Parsed", {})
        if not atto_omd:
            return {}
        else:
            sem = atto_omd.get("SEM", {})
            stage_image = atto_omd.get("SITE IMAGE", {})

        sem_metadata = {
            # "beam_current": None,
            "beam_energy": sem.get("Beam Energy"),
            "beam_energy_units": sem.get("Beam Energy_units"),
            # "probe_area" : None,
            # "convergence_angle": None,
            "magnification": sem.get("Real Magnification"),
            "microscope": "Attolight Allalin",
            "Stage": {
                "rotation": stage_image.get("stage_rotation_z"),
                "rotation_units": "deg",
                "tilt_alpha": stage_image.get("stage_rotation_x"),
                "tilt_alpha_units": "deg",
                "tilt_beta": stage_image.get("stage_rotation_y"),
                "tilt_beta_units": "deg",
                "x": stage_image.get("stage_position_x"),
                "x_units": "mm",
                "y": stage_image.get("stage_position_y"),
                "y_units": "mm",
                "z": stage_image.get("stage_position_z"),
                "z_units": "mm",
            },
        }

        return sem_metadata

    def _map_Spectrometer_metadata(self) -> dict:
        """return Spectrometer metadata according to lumispy specifications"""
        atto_omd = self.signal_dict["original_metadata"]
        # get nested dictionaries in an error-handling way
        atto_omd = atto_omd.get("Object_0_Channel_0", {})
        atto_omd = atto_omd.get("Parsed", {})
        if not atto_omd:
            return {}
        else:
            spectrometer = atto_omd.get("SPECTROMETER", {})

        spectrometer_metadata = {
            # "model":
            # "acquisition_mode": ,
            "entrance_slit_width": spectrometer.get("Entrance slit width"),
            "entrance_slit_width_units": spectrometer.get("Entrance slit width_units"),
            "exit_slit_width": spectrometer.get("Exit slit width"),
            "exit_slit_width_units": spectrometer.get("Exit slit width_units"),
            "central_wavelength": spectrometer.get("Central wavelength"),
            "central_wavelength_units": spectrometer.get("Central wavelength_units"),
            # "start_wavelength(nm)":
            # "step_size(nm)"
            "Grating": spectrometer.get("Grating"),
            "groove_density": spectrometer.get("Grating - Groove Density"),
            "groove_density_units": spectrometer.get("Grating - Groove Density_units"),
            "blazing_wavelength": spectrometer.get("Grating - Blaze Angle"),
            "blazing_wavelength_units": spectrometer.get("Central wavelength_units"),
            "Filter": {"filter_type": spectrometer.get("Filter")},
        }

        return spectrometer_metadata

    def _map_spectral_detector_metadata(self) -> dict:
        """return Spectrometer metadata according to lumispy specifications"""

        atto_omd = self.signal_dict["original_metadata"]
        # get nested dictionaries in an error-handling way
        atto_omd = atto_omd.get("Object_0_Channel_0", {})
        atto_omd = atto_omd.get("Parsed", {})
        if not atto_omd:
            return {}
        else:
            ccd = atto_omd.get("CCD", {})

        spectral_detector_metadata = {
            "detector_type": "CCD",
            "model": ccd.get("Camera Model"),
            # "frames": ,
            "integration_time": ccd.get("Exposure Time"),
            "integration_time_units": ccd.get("Exposure Time"),
            # "saturation_fraction": CCD.get(''),
            "binning": (ccd.get("ReadMode"), ccd.get("Horizontal Binning")),
            # "processing": ,
            # "sensor_roi": ,
            "pixel_size": ccd.get("Pixel Width"),
            "pixel_size_units": ccd.get("Pixel Width_units"),
        }

        return spectral_detector_metadata

    def _map_CL_metadata(self) -> dict:
        """Build CL-signal-specific metadata. Currently maps from the hyperspy metadata format"""

        cl_metadata_dict = {
            "Acquisition_instrument": {
                "SEM": self._map_SEM_metadata(),
                "Spectrometer": self._map_Spectrometer_metadata(),
                "Detector": self._map_spectral_detector_metadata(),
            }
        }

        return cl_metadata_dict

    def _set_metadata_and_original_metadata(self, unpacked_dict):
        """Run successively _build_metadata and _build_original_metadata
        and set signal dictionary with results"""

        self.signal_dict["metadata"] = self._build_generic_metadata(unpacked_dict)
        self.signal_dict["original_metadata"] = self._build_original_metadata()
        self.signal_dict["metadata"].update(self._build_signal_specific_metadata())

    @staticmethod
    def _check_comments(commentsstr, prefix, delimiter):
        """Check if comment string is parsable into metadata dictionary.
        Some specific lines (empty or starting with @@) will be ignored,
        but any non-ignored line must conform to being a title line (beginning
        with the titlestart indicator) or being parsable (starting with Prefix
        and containing the key data delimiter). At the end, the comment is
        considered parsable if it contains minimum 1 parsable line and no
        non-ignorable non-parsable non-title line.

        Parameters
        ----------
        commentsstr: string containing comments
        prefix: string (or char) character assumed to start each line.
        '$' if a .sur file.
        delimiter: string that delimits the keyword from value. always '='

        Returns
        -------
        valid: boolean
        """

        # Titlestart markers start with Prefix ($) followed by underscore
        titlestart = "{:s}_".format(prefix)

        # We start by assuming that the comment string is valid
        # but contains 0 valid (= parsable) lines
        valid = True
        n_valid_lines = 0

        for line in commentsstr.splitlines():
            # Here we ignore any empty line or line starting with @@
            ignore = False
            if not line.strip() or line.startswith("@@"):
                ignore = True
            # If the line must not be ignored
            if not ignore:
                # If line starts with a titlestart marker we it counts as valid
                if line.startswith(titlestart):
                    n_valid_lines += 1
                # if it does not we check that it has the delimiter and
                # starts with prefix
                else:
                    # We check that line contains delimiter and prefix
                    # if it does the count of valid line is increased
                    if delimiter in line and line.startswith(prefix):
                        n_valid_lines += 1
                    # Otherwise the whole comment string is thrown out
                    else:
                        valid = False

        # finally, it total number of valid line is 0 we throw out this comments
        if n_valid_lines == 0:
            valid = False

        # return falsiness of the string.
        return valid

    @staticmethod
    def _get_comment_dict(
        original_metadata: dict, method: str = "auto", custom: dict = {}
    ) -> dict:
        """Return the dictionary used to set the dataset comments (akA custom parameters) while exporting a file.

        By default (method='auto'), tries to identify if the object was originally imported by rosettasciio
        from a digitalsurf .sur/.pro file with a comment field parsed as original_metadata (i.e.
        Object_0_Channel_0.Parsed). In that case, digitalsurf ignores non-parsed original metadata
        (ie .sur/.pro file headers). If the original metadata contains multiple objects with
        non-empty parsed content (Object_0_Channel_0.Parsed, Object_0_Channel_1.Parsed etc...), only
        the first non-empty X.Parsed sub-dictionary is returned. This falls back on returning the
        raw 'original_metadata'

        Optionally the raw 'original_metadata' dictionary can be exported (method='raw'),
        a custom dictionary provided by the user (method='custom'), or no comment at all (method='off')

        Args:
            method (str, optional): method to export. Defaults to 'auto'.
            custom (dict, optional): custom dictionary. Ignored unless method is set to 'custom', Defaults to {}.

        Raises:
            MountainsMapFileError: if an invalid key is entered

        Returns:
            dict: dictionary to be exported as a .sur object
        """
        if method == "raw":
            return original_metadata
        elif method == "custom":
            return custom
        elif method == "off":
            return {}
        elif method == "auto":
            pattern = re.compile(r"Object_\d*_Channel_\d*")
            omd = original_metadata
            # filter original metadata content of dict type and matching pattern.
            validfields = [
                omd[key]
                for key in omd
                if pattern.match(key) and isinstance(omd[key], dict)
            ]
            # In case none match, give up filtering and return raw
            if not validfields:
                return omd
            # In case some match, return first non-empty "Parsed" sub-dict
            for field in validfields:
                # Return none for non-existing "Parsed" key
                candidate = field.get("Parsed")
                # For non-none, non-empty dict-type candidate
                if candidate and isinstance(candidate, dict):
                    return candidate
                # dict casting for non-none but non-dict candidate
                elif candidate is not None:
                    return {"Parsed": candidate}
                # else none candidate, or empty dict -> do nothing
            # Finally, if valid fields are present but no candidate
            # did a non-empty return, it is safe to return empty
            return {}
        else:
            raise MountainsMapFileError(
                "Non-valid method for setting mountainsmap file comment. Choose one of: 'auto','raw','custom','off' "
            )

    @staticmethod
    def _stringify_dict(omd: dict):
        """Pack nested dictionary metadata into a string. Pack dictionary-type elements
        into digitalsurf "Section title" metadata type ('$_ preceding section title). Pack
        other elements into equal-sign separated key-value pairs.

        Supports the key-units logic {'key': value, 'key_units': 'un'} used in hyperspy.
        """

        # Separate dict into list of keys and list of values to authorize index-based pop/insert
        keys_queue = list(omd.keys())
        vals_queue = list(omd.values())
        # commentstring to be returned
        cmtstr: str = ""
        # Loop until queues are empty
        while keys_queue:
            # pop first object
            k = keys_queue.pop(0)
            v = vals_queue.pop(0)
            # if object is header
            if isinstance(v, dict):
                cmtstr += f"$_{k}\n"
                keys_queue = list(v.keys()) + keys_queue
                vals_queue = list(v.values()) + vals_queue
            else:
                try:
                    ku_idx = keys_queue.index(k + "_units")
                    has_units = True
                except ValueError:
                    ku_idx = None
                    has_units = False

                if has_units:
                    _ = keys_queue.pop(ku_idx)
                    vu = vals_queue.pop(ku_idx)
                    cmtstr += f"${k} = {v.__str__()} {vu}\n"
                else:
                    cmtstr += f"${k} = {v.__str__()}\n"

        return cmtstr

    # Post processing
    @staticmethod
    def post_process_RGB(signal):
        signal = signal.transpose()
        max_data = np.max(signal.data)
        if max_data <= 255:
            signal.change_dtype("uint8")
            signal.change_dtype("rgb8")
        elif max_data <= 65536:
            signal.change_dtype("uint16")
            signal.change_dtype("rgb16")
        else:
            warnings.warn(
                """RGB-announced data could not be converted to
            uint8 or uint16 datatype"""
            )

        return signal

    @staticmethod
    def post_process_binary(signal):
        signal.change_dtype("bool")
        return signal

    # pack/unpack binary quantities

    @staticmethod
    def _get_uint16(file):
        """Read a 16-bits int with a user-definable default value if
        no file is given"""
        b = file.read(2)
        return struct.unpack("<H", b)[0]

    @staticmethod
    def _set_uint16(file, val):
        file.write(struct.pack("<H", val))

    @staticmethod
    def _get_int16(
        file,
    ):
        """Read a 16-bits int with a user-definable default value if
        no file is given"""
        b = file.read(2)
        return struct.unpack("<h", b)[0]

    @staticmethod
    def _set_int16(file, val):
        file.write(struct.pack("<h", val))

    @staticmethod
    def _get_str(file, size, encoding="latin-1"):
        """Read a str of defined size in bytes with a user-definable default
        value if no file is given"""
        read_str = file.read(size).decode(encoding)
        return read_str.strip(" \t\n")

    @staticmethod
    def _set_str(file, val, size, encoding="latin-1"):
        """Write a str of defined size in bytes to a file. struct.pack
        will automatically trim the string if it is too long"""
        file.write(
            struct.pack(
                "<{:d}s".format(size),
                f"{val}".ljust(size).encode(encoding),
            )
        )

    @staticmethod
    def _get_int32(file):
        """Read a 32-bits int with a user-definable default value if no
        file is given"""
        b = file.read(4)
        return struct.unpack("<i", b)[0]

    @staticmethod
    def _set_int32(file, val):
        """Write a 32-bits int in a file f"""
        file.write(struct.pack("<i", val))

    @staticmethod
    def _get_float(
        file,
    ):
        """Read a 4-bytes (single precision) float from a binary file f with a
        default value if no file is given"""
        return struct.unpack("<f", file.read(4))[0]

    @staticmethod
    def _set_float(file, val):
        """write a 4-bytes (single precision) float in a file"""
        file.write(struct.pack("<f", val))

    @staticmethod
    def _get_uint32(
        file,
    ):
        b = file.read(4)
        return struct.unpack("<I", b)[0]

    @staticmethod
    def _set_uint32(file, val):
        file.write(struct.pack("<I", val))

    @staticmethod
    def _get_bytes(file, size):
        return file.read(size)

    @staticmethod
    def _set_bytes(file, val, size):
        file.write(struct.pack("<{:d}s".format(size), val))

    def _unpack_comment(self, file, encoding="latin-1"):
        commentsize = self._get_work_dict_key_value("_50_Comment_size")
        return self._get_str(file, commentsize, encoding)

    def _pack_comment(self, file, val, encoding="latin-1"):
        commentsize = self._get_work_dict_key_value("_50_Comment_size")
        self._set_str(file, val, commentsize)

    def _unpack_private(self, file, encoding="latin-1"):
        privatesize = self._get_work_dict_key_value("_51_Private_size")
        return self._get_str(file, privatesize, encoding)

    def _pack_private(self, file, val, encoding="latin-1"):
        privatesize = self._get_work_dict_key_value("_51_Private_size")
        self._set_str(file, val, privatesize)

    def _is_data_int(
        self,
    ):
        """Determine wether data consists of unscaled int values.
        This is not the case for all objects. Surface and surface series can admit
        this logic. In theory, hyperspectral studiables as well but it is more convenient
        to use them as floats due to typical data treatment in hyperspy (scaling etc)"""
        objtype = self._mountains_object_types[
            self._get_work_dict_key_value("_05_Object_Type")
        ]
        if objtype in ["_SURFACESERIE", "_SURFACE"]:
            scale = self._get_work_dict_key_value(
                "_23_Z_Spacing"
            ) / self._get_work_dict_key_value("_35_Z_Unit_Ratio")
            offset = self._get_work_dict_key_value("_55_Z_Offset")
            if float(scale).is_integer() and float(offset).is_integer():
                return True
            else:
                return False
        else:
            return False

    def _is_data_scaleint(
        self,
    ):
        """Digitalsurf image formats are not stored as their raw int values, but instead are
        scaled and a scale / offset is set so that the data scales down to uint. Why this is
        done this way is not clear to me."""
        objtype = self._mountains_object_types[
            self._get_work_dict_key_value("_05_Object_Type")
        ]
        if objtype in [
            "_RGBIMAGE",
            "_SERIESOFRGBIMAGES",
            "_INTENSITYIMAGE",
        ]:
            return True
        else:
            return False

    def _is_data_bin(self):
        """Digitalsurf image formats can be binary sometimes"""
        objtype = self._mountains_object_types[
            self._get_work_dict_key_value("_05_Object_Type")
        ]
        if objtype in [
            "_BINARYIMAGE",
        ]:
            return True
        else:
            return False

    def _get_uncompressed_datasize(self) -> int:
        """Return size of uncompressed data in bytes"""
        psize = int(self._get_work_dict_key_value("_15_Size_of_Points") / 8)
        # Datapoints in X and Y dimensions
        Npts_tot = self._get_work_dict_key_value("_20_Total_Nb_of_Pts")
        # Datasize in WL. max between value and 1 as often W_Size saved as 0
        Wsize = max(self._get_work_dict_key_value("_14_W_Size"), 1)
        # Wsize = 1

        datasize = Npts_tot * Wsize * psize

        return datasize

    def _unpack_data(self, file, encoding="latin-1"):
        # Size of datapoints in bytes. Always int16 (==2) or 32 (==4)
        psize = int(self._get_work_dict_key_value("_15_Size_of_Points") / 8)
        dtype = np.int16 if psize == 2 else np.int32

        if self._get_work_dict_key_value("_01_Signature") != "DSCOMPRESSED":
            # If the points are not compressed we need to read the exact
            # size occupied by datapoints

            # Datapoints in X and Y dimensions
            Npts_tot = self._get_work_dict_key_value("_20_Total_Nb_of_Pts")
            # Datasize in WL
            Wsize = max(self._get_work_dict_key_value("_14_W_Size"), 1)

            # We need to take into account the fact that Wsize is often
            # set to 0 instead of 1 in non-spectral data to compute the
            # space occupied by data in the file
            readsize = Npts_tot * psize * Wsize

            buf = file.read(readsize)
            # Read the exact size of the data
            _points = np.frombuffer(buf, dtype=dtype)

        else:
            # If the points are compressed do the uncompress magic. There
            # the space occupied by datapoints is self-taken care of.
            # Number of streams
            _directoryCount = self._get_uint32(file)

            # empty lists to store the read sizes
            rawLengthData = []
            zipLengthData = []
            for i in range(_directoryCount):
                # Size of raw and compressed data sizes in each stream
                rawLengthData.append(self._get_uint32(file))
                zipLengthData.append(self._get_uint32(file))

            # We now initialize an empty binary string to store the results
            rawData = b""
            for i in range(_directoryCount):
                # And for each stream we uncompress using zip lib
                # and add it to raw string
                rawData += zlib.decompress(file.read(zipLengthData[i]))

            # Finally numpy converts it to a numeric object
            _points = np.frombuffer(rawData, dtype=dtype)

        # rescale data
        # We set non measured points to nan according to .sur ways
        nm = []
        if self._get_work_dict_key_value("_11_Special_Points") == 1:
            # has non-measured points
            nm = _points == self._get_work_dict_key_value("_16_Zmin") - 2

        Zmin = self._get_work_dict_key_value("_16_Zmin")
        scale = self._get_work_dict_key_value(
            "_23_Z_Spacing"
        ) / self._get_work_dict_key_value("_35_Z_Unit_Ratio")
        offset = self._get_work_dict_key_value("_55_Z_Offset")

        # Packing data into ints or float, with or without scaling.
        if self._is_data_int():
            pass  # Case left here for future modification
        elif self._is_data_scaleint():
            _points = (_points.astype(float) - Zmin) * scale + offset
            _points = np.round(_points).astype(int)
        elif self._is_data_bin():
            pass
        else:
            _points = (_points.astype(float) - Zmin) * scale + offset
            _points[nm] = np.nan  # Ints have no nans

        # Return the points, rescaled
        return _points

    def _pack_data(self, file, val, encoding="latin-1"):
        """This needs to be special because it writes until the end of file."""
        # Also valid for uncompressed
        if self._get_work_dict_key_value("_01_Signature") != "DSCOMPRESSED":
            datasize = self._get_uncompressed_datasize()
        else:
            datasize = self._get_work_dict_key_value("_48_Compressed_data_size")
        self._set_bytes(file, val, datasize)

    @staticmethod
    def _compress_data(data_int, nstreams: int = 1) -> bytes:
        """Pack the input data using the digitalsurf zip approach and return the result as a
        binary string ready to be written onto a file."""

        if nstreams <= 0 or nstreams > 8:
            raise MountainsMapFileError(
                "Number of compression streams must be >= 1, <= 8"
            )

        bstr = b""
        bstr += struct.pack("<I", nstreams)

        data_1d = data_int.ravel()
        tot_size = len(data_1d)

        if tot_size % nstreams != 0:
            streamlen = len(data_1d) // nstreams + 1
        else:
            streamlen = len(data_1d) // nstreams

        zipdat = []
        for i in range(nstreams):
            # Extract sub-array and its raw size
            data_comp = data_1d[i * streamlen : (i + 1) * streamlen]
            rdl = len(data_comp) * data_comp.itemsize
            # rdl = len(data_comp.tobytes())

            # Compress and extract compressed size
            data_zip = zlib.compress(data_comp)
            cdl = len(data_zip)

            # Export bytes
            bstr += struct.pack("<I", rdl)
            bstr += struct.pack("<I", cdl)
            zipdat.append(data_zip)

        for zd in zipdat:
            bstr += zd

        return bstr


def file_reader(filename, lazy=False):
    """
    Read a mountainsmap ``.sur`` or ``.pro`` file.

    Parameters
    ----------
    %s
    %s

    %s
    """
    if lazy is not False:
        raise NotImplementedError("Lazy loading is not supported.")
    ds = DigitalSurfHandler(filename)

    ds._read_sur_file()

    surdict = ds._build_sur_dict()

    return [
        surdict,
    ]


def file_writer(
    filename,
    signal: dict,
    set_comments: str = "auto",
    is_special: bool = False,
    compressed: bool = True,
    comments: dict = {},
    object_name: str = "",
    operator_name: str = "",
    absolute: int = 0,
    private_zone: bytes = b"",
    client_zone: bytes = b"",
):
    """
    Write a mountainsmap ``.sur`` or ``.pro`` file.

    Parameters
    ----------
    %s
    %s
    set_comments : str , default = 'auto'
        Whether comments should be a simplified version original_metadata ('auto'),
        the raw original_metadata dictionary ('raw'), skipped ('off'), or supplied
        by the user as an additional kwarg ('custom').
    is_special : bool , default = False
        If True, NaN values in the dataset or integers reaching the boundary of the
        signed int-representation are flagged as non-measured or saturating,
        respectively. If False, those values are not flagged (converted to valid points).
    compressed : bool, default =True
        If True, compress the data in the export file using zlib. Can help dramatically
        reduce the file size.
    comments : dict, default = {}
        Set a custom dictionnary in the comments field of the exported file.
        Ignored if set_comments is not set to 'custom'.
    object_name : str, default = ''
        Set the object name field in the output file.
    operator_name : str, default = ''
        Set the operator name field in the exported file.
    absolute : int, default = 0,
        Unsigned int capable of flagging whether surface heights are relative (0) or
        absolute (1). Higher unsigned int values can be used to distinguish several
        data series sharing internal reference.
    private_zone : bytes, default = b'',
        Set arbitrary byte-content in the private_zone field of exported file metadata.
        Maximum size is 32.0 kB and content will be cropped if this size is exceeded.
    client_zone : bytes, default = b''
        Set arbitrary byte-content in the client_zone field of exported file metadata.
        Maximum size is 128 B and and content will be cropped if this size is exceeded.
    """
    ds = DigitalSurfHandler(filename=filename)
    ds.signal_dict = signal

    ds._build_sur_file_contents(
        set_comments,
        is_special,
        compressed,
        comments,
        object_name,
        operator_name,
        absolute,
        private_zone,
        client_zone,
    )
    ds._write_sur_file()


file_reader.__doc__ %= (FILENAME_DOC, LAZY_UNSUPPORTED_DOC, RETURNS_DOC)
file_writer.__doc__ %= (FILENAME_DOC, SIGNAL_DOC)
