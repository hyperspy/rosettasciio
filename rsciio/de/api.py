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
import xml.etree.ElementTree as ET

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
                              "'Top' or 'Bottom' in the file name")
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
                      "TrueImageSize": ["<i", 580],
                      "FPS": ['<d', 584],
                    }
    return read_binary_metadata(file, metadata_dict)


def parse_metadata(file):
    metadata_header_dict = {"Version": ["<u4", 0],
                     "HeaderSizeAlways": ["<u4", 4],
                     "IndexCountNumber": ["<u4", 8],
                     "MetadataSize": ["<u4", 12],
                     "MetadataInfoSize": ["<u4", 16],
                     "MetadataLeadSize": ["<u4", 20],
                     "SensorGain": [np.float64, 320],
                     "Magnification": [np.float64, 328],
                     "PixelSize": [np.float64, 336],
                     "CameraLength": [np.float64, 344],
                     "DiffPixelSize": [np.float64, 352],
                     }
    metadata_header_dict = read_binary_metadata(file, metadata_header_dict)
    print(metadata_header_dict)

    start_read = int(metadata_header_dict["HeaderSizeAlways"][()])
    metadata_dtype = [("UnitSize", "<u4"),
                      ("MetadataCount", "<u4"),
                      ("MetadataUID", "<u4"),
                      ("Datasize", "<u4"),
                      ("ID", "<u8"),
                      ("Data", np.int_),]
    data = []
    for i in range(metadata_header_dict["IndexCountNumber"]):
        metdata_info = np.fromfile(file, dtype=metadata_dtype,
                                    count=1, offset=start_read)
        data.append(metdata_info)

        start_read += metadata_header_dict["MetadataSize"]
    return


def parse_xml(file):
    try:
        tree = ET.parse(file)
        xml_dict = {}
        for i in tree.iter():
            xml_dict[i.tag]=i.attrib
        #clean_xml
        for k1 in xml_dict:
            for k2 in xml_dict[k1]:
                if k2 =="Value":
                    try:
                        xml_dict[k1] = float(xml_dict[k1][k2])
                    except ValueError:
                        xml_dict[k1] = xml_dict[k1][k2]
    except FileNotFoundError:
        _logger.warning(msg="File " + file + " not found. Please"
                                             "move it to the same directory to read"
                                             " the metadata ")
    return xml_dict


def read_full_seq(file,
                  ImageWidth,
                  ImageHeight,
                  ImageBitDepthReal,
                  TrueImageSize,
                  navigation_shape=None):
    empty = TrueImageSize-((ImageWidth*ImageHeight*2)+8)
    dtype = [("Array", np.int16, (ImageWidth, ImageHeight)),
             ("sec", "<u4"),
             ("ms", "<u2"),
             ("mis", "<u2"),
             ("empty", bytes, empty)]
    data = read_binary_reshape(file,dtypes=dtype, offset=8192)
    return data

def read_split_seq(top,
                  ImageWidth,
                  ImageHeight,
                  ImageBitDepthReal,
                  TrueImageSize,
                  navigation_shape=None):
    empty = TrueImageSize-((ImageWidth*ImageHeight*2)+8)
    dtype = [("Array", np.int16, (ImageWidth, ImageHeight)),
             ("sec", "<u4"),
             ("ms", "<u2"),
             ("mis", "<u2"),
             ("empty", bytes, empty)]
    data = read_binary_reshape(file,dtypes=dtype, offset=8192)
    return data
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

