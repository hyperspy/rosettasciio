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


import numpy as np
import os
import logging
import struct


import dask.array as da

from rsciio.utils.tools import read_binary_metadata, convert_xml_to_dict

_logger = logging.getLogger(__name__)
data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}  # Stream Pix data types


def file_reader(filename, navigation_size=(), celeritas=False,
                **kwargs):
    if celeritas and ("top" not in kwargs and "bottom" not in kwargs):
        if "Top" in filename:
            top = filename
            bottom = filename.replace("Top", "Bottom", 1)
            filename = filename[:filename.index("_Top")+".seq"]

        elif "Bottom" in filename:
            bottom = filename
            top = filename.replace("Bottom", "Top",1)
            filename = filename[:filename.index("_Bottom") + ".seq"]
        else:
            _logger.error(msg="For the Celeritas Camera Top and Bottom "
                              "frames must be explicitly given by passing the"
                              "top and bottom kwargs or the file name must have"
                              "'Top' or 'bottom' in the file name")
    elif celeritas and "top" in kwargs and "bottom" in kwargs:
        top = kwargs["top"]
        bottom = kwargs["top"]

    file_extensions = {"metadata": ".metadata",
                       "dark": ".dark.mrc",
                       "gain": ".gain.mrc",
                       "xml": ".Config.Metadata.xml"}
    for ext in file_extensions:
        if not ext in kwargs and kwargs[ext] is not None:
            kwargs[ext] = filename + file_extensions[ext]

    if celeritas:
        read_celeritas(top, bottom, dark, gain, xml, metadata)


def read_celeritas(top, bottom, dark, gain, xml, metadata):
    header = parse_header(top)
    metadata = parse_metadata(metadata)

def parse_header(file):
    metadata_dict = {"ImageWidth":["<u4", 548],
                      "ImageHeight":["<u4", 552],
                      "ImageBitDepth":["<u4",556],
                      "ImageBitDepthReal":["<u4",560],
                      "NumFrames":["<i", 572],
                      "ImgBytes": ["<i", 580],
                      "FPS": ['<d', 584],
                    }
    return read_binary_metadata(file, metadata_dict)


def parse_metadata(file):
    metadata_dict = {"SensorGain": [np.float64, 320],
                     "Magnification": [np.float64, 328],
                     "PixelSize": [np.float64, 336],
                     "CameraLength": [np.float64, 344],
                     "DiffPixelSize": [np.float64, 352],
                     }
    return read_binary_metadata(file, metadata_dict)


def parse_xml(file):
    try:
        with open(file) as f:
            xml_dict = convert_xml_to_dict(f)
    except FileNotFoundError:
        _logger.warning(msg="File " + file + " not found. Please"
                                             "move it to the same directory to read"
                                             " the metadata ")
    return xml_dict


def read_binary_reshape(file,
                        dtypes,
                        offset,
                        navigation_shape=None):
    keys = [d[0] for d in dtypes]
    mapped = np.memmap(file,
                       offset=offset,
                       dtype=dtypes,
                       shape=navigation_shape)
    bin_data = {k: mapped[k] for k in keys}
    return bin_data


def read_stitch_binary(top, bottom, dtypes,offset, navigation_shape):
    keys = [d[0] for d in dtypes]
    top_mapped = np.memmap(top,
                           offset=offset,
                           dtype=dtypes,
                           shape=navigation_shape)
    bottom_mapped = np.memmap(bottom,
                              offset=offset,
                              dtype=dtypes,
                              shape=navigation_shape)
    bin_data = {k: da.concatenate([top_mapped[k], bottom_mapped[k]], -1)
                for k in keys}
    return bin_data









