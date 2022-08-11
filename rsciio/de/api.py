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
import glob

import dask.array as da

from rsciio.utils.tools import read_binary_metadata, parse_xml

_logger = logging.getLogger(__name__)
data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}  # Stream Pix data types



def file_reader(filename, navigation_shape=(), celeritas=False,
                **kwargs):
    if celeritas:
        if "top" not in kwargs and "bottom" not in kwargs:
            if "Top" in filename:
                top = filename
                leading_str = filename.rsplit('_Top', 1)[0]
                bottom = glob.glob(leading_str+"_Bottom*.seq")[0]
                filename = leading_str+".seq"

            elif "Bottom" in filename:
                bottom = filename
                leading_str = filename.rsplit('_Bottom', 1)[0]
                top = glob.glob(leading_str + "_Top*.seq")[0]
                filename = leading_str+".seq"
            if "metadata" not in kwargs:
                kwargs["metadata"]=bottom+".metadata"
            else:
                _logger.error(msg="For the Celeritas Camera Top and Bottom "
                                  "frames must be explicitly given by passing the"
                                  "top and bottom kwargs or the file name must have"
                                  "'Top' or 'Bottom' in the file name")
        elif celeritas and "top" in kwargs and "bottom" in kwargs:
            top = kwargs["top"]
            bottom = kwargs["top"]
        else:
            _logger.error(msg="For the Celeritas Camera Top and Bottom "
                              "frames must be explicitly given by passing the"
                              "top and bottom kwargs or the file name must have"
                              "'Top' or 'Bottom' in the file name")

    file_extensions = {"metadata": ".metadata",
                       "dark": ".dark.mrc",
                       "gain": ".gain.mrc",
                       "xml": ".Config.Metadata.xml"}
    for ext in file_extensions:
        if not ext in kwargs:
            kwargs[ext] = filename + file_extensions[ext]

    if celeritas:
        reader = CeleritasReader(file=filename,
                                 top=top,
                                 bottom=bottom,
                                 **kwargs)
    else:
        reader = SeqReader(file=filename, **kwargs)
    return reader.read_data(navigation_shape=navigation_shape)


class SeqReader:
    def __init__(self, file,
                 dark=None,
                 gain=None,
                 metadata=None,
                 xml=None):
        """
        Initializes a general reader for Binary signals.

        Parameters
        ----------
        file: str
            A file to be read.
        """
        self.file = file
        self.metadata_file=metadata
        self.dark = dark
        self.gain = gain
        self.xml=xml
        # Output
        self.original_metadata = {"InputFiles":{"file":file,
                                                "metadata":self.metadata_file,
                                                "dark":dark,
                                                "gain":gain,
                                                "xml":xml}}
        self.metadata = {'General': {"filename":file, },
                         'Signal': {'signal_type': 'Signal2D'}}
        self.data = None
        self.axes = []
        self.buffer = None

    def _read_metadata(self):
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
        metadata_dict = read_binary_metadata(self.metadata_file, metadata_header_dict)
        self.original_metadata["Metadata"] = metadata_dict
        self.metadata["acquisition_instrument"] = {"TEM":
                                                       {'camera_length': metadata_dict["CameraLength"],
                                                        'magnification': metadata_dict["Magnification"]}}
        return metadata_dict

    def _read_file_header(self):
        file_header_dict = {"ImageWidth":["<u4", 548],
                            "ImageHeight":["<u4", 552],
                            "ImageBitDepth":["<u4",556],
                            "ImageBitDepthReal":["<u4",560],
                            "NumFrames":["<i", 572],
                            "TrueImageSize": ["<i", 580],
                            "FPS": ['<d', 584],
                            }
        metadata_dict = read_binary_metadata(self.file, file_header_dict)
        self.original_metadata["FileHeader"] = metadata_dict
        self.metadata["ImageHeader"] = metadata_dict
        return metadata_dict

    def _read_xml(self):
        xml_dict = parse_xml(self.xml)

        self.original_metadata["xml"] = xml_dict
        return xml_dict

    def _read_dark_gain(self):
        gain_img = read_ref(self.gain)
        dark_img = read_ref(self.dark)
        self.metadata["Reference"] = {"dark": dark_img,
                                      "gain": gain_img}
        return dark_img, gain_img

    def _create_axes(self, header, nav_shape=None):
        if nav_shape is None or len(nav_shape)==1:
            self.axes.append({'name': 'time',
                              'offset': 0,
                              'unit': "sec",
                              'scale': 1 / header["FPS"],
                              'size': header["NumFrames"],
                              'navigate': True,
                              'index_in_array': 0})
        else:
            for s in nav_shape:
                self.axes.append({'offset': 0,
                                  'scale': 1,
                                  'size': s,
                                  'navigate': True,
                                  'index_in_array': 0}
                                 )
        self.axes.append({'name': 'y',
                          'offset': 0,
                          'scale': 1,
                          'size': header["ImageHeight"],
                          'navigate': False,
                          'index_in_array': 1})
        self.axes.append({'name': 'x',
                          'offset': 0,
                          'scale': 1,
                          'size': header["ImageWidth"],
                          'navigate': False,
                          'index_in_array': 2})
        if (self.original_metadata["Metadata"] != {} and
            self.original_metadata["Metadata"]["PixelSize"] > 1e-30):
            # need to still determine a way to properly set units and scale
            self.axes[-2]['scale'] = self.original_metadata["Metadata"]["PixelSize"]
            self.axes[-1]['scale'] = self.original_metadata["Metadata"]["PixelSize"]
        return

    def read_data(self, navigation_shape=None):
        header = self._read_file_header()
        dark_img, gain_img = self._read_dark_gain()
        self._read_xml()
        self._read_metadata()
        if navigation_shape is None or navigation_shape== ():
            navigation_shape = (header["NumFrames"],)
        data, time = read_full_seq(self.file,
                                   ImageWidth=header["ImageWidth"],
                                   ImageHeight=header["ImageHeight"],
                                   ImageBitDepth=header["ImageBitDepth"],
                                   TrueImageSize=header["TrueImageSize"],
                                   navigation_shape=navigation_shape)
        if dark_img is not None:
            data = np.subtract(data, dark_img)
        if gain_img is not None:
            data = np.multiply(data, gain_img)
        self.original_metadata["Timestamps"] = time
        self.metadata["Timestamps"] = time
        self._create_axes(header=header, nav_shape=navigation_shape)
        return {"data": data,
                "metadata": self.metadata,
                "axes": self.axes,
                "original_metadata": self.original_metadata}


class CeleritasReader(SeqReader):
    def __init__(self, top, bottom,  **kwargs):
        super().__init__(**kwargs)
        self.top = top
        self.bottom = bottom

    def _read_file_header(self):
        file_header_dict = {"ImageWidth":["<u4", 548],
                            "ImageHeight":["<u4", 552],
                            "ImageBitDepth":["<u4",556],
                            "ImageBitDepthReal":["<u4",560],
                            "NumFrames":["<i", 572],
                            "TrueImageSize": ["<i", 580],
                            "FPS": ['<d', 584],
                            }
        metadata_dict = read_binary_metadata(self.top, file_header_dict)
        self.original_metadata["FileHeader"] = metadata_dict
        self.metadata["ImageHeader"] = metadata_dict
        return metadata_dict

    def _read_xml(self):
        xml_dict = parse_xml(self.xml)

        self.original_metadata["xml"] = xml_dict
        print(xml_dict)
        if xml_dict is not None:
            if "SegmentPreBuffer" in xml_dict:
                self.buffer = xml_dict["SegmentPreBuffer"]
        return xml_dict

    def read_data(self, navigation_shape=None):
        header = self._read_file_header()
        dark_img, gain_img = self._read_dark_gain()
        self._read_xml()
        self._read_metadata()
        data, time = read_split_seq(self.top,
                                    self.bottom,
                                    ImageWidth=header["ImageWidth"],
                                    ImageHeight=header["ImageHeight"],
                                    ImageBitDepth=header["ImageBitDepth"],
                                    TrueImageSize=header["TrueImageSize"],
                                    SegmentPreBuffer=self.buffer,
                                    total_frames=header["NumFrames"],
                                    navigation_shape=navigation_shape)
        if dark_img is not None:
            data = np.subtract(data, dark_img[np.newaxis])
        if gain_img is not None:
            data = np.multiply(data, gain_img[np.newaxis])
        self.original_metadata["Timestamps"] = time
        self.metadata["Timestamps"] = time
        self._create_axes(header=header, nav_shape=navigation_shape)
        return {"data": data,
                "metadata": self.metadata,
                "axes": self.axes,
                "original_metadata": self.original_metadata}


"""
Functions for reading the different binary files used.
"""

def read_full_seq(file,
                  ImageWidth,
                  ImageHeight,
                  ImageBitDepth,
                  TrueImageSize,
                  navigation_shape=None):
    data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}
    empty = TrueImageSize-((ImageWidth*ImageHeight*2)+8)
    dtype = [("Array", data_types[int(ImageBitDepth)], (ImageWidth, ImageHeight)),
             ("sec", "<u4"),
             ("ms", "<u2"),
             ("mis", "<u2"),
             ("empty", bytes, empty)]
    data = np.memmap(file,
                     offset=8192,
                     dtype=dtype,
                     shape=navigation_shape)
    return data["Array"], {"sec": data["sec"],
                           "ms": data["ms"],
                           "mis": data["mis"]}


def read_split_seq(top,
                   bottom,
                   ImageWidth,
                   ImageHeight,
                   ImageBitDepth,
                   TrueImageSize,
                   SegmentPreBuffer=None,
                   total_frames=None,
                   navigation_shape=None,):
    data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}
    empty = TrueImageSize-((ImageWidth*ImageHeight*2)+8)
    if SegmentPreBuffer is None:
        _logger.warning(msg="No XML File given. Guessing Segment PreBuffer "
                            "This is may not be correct...")
        if ImageWidth == 512:
            SegmentPreBuffer = 16
        elif ImageWidth == 256:
            SegmentPreBuffer= 64
        else:
            SegmentPreBuffer = 4
    dtype = [("Array", data_types[int(ImageBitDepth)], (int(SegmentPreBuffer),
                                                        int(ImageHeight/SegmentPreBuffer),
                                                        int(ImageWidth))),
             ("sec", "<u4"),
             ("ms", "<u2"),
             ("mis", "<u2"),
             ("empty", bytes, empty)]
    if navigation_shape is not None:
        # need to read out extra buffered frames
        total_buffer_frames = int(np.ceil(np.divide(np.product(navigation_shape),
                               SegmentPreBuffer)))
    else:
        total_buffer_frames = np.ceil(total_frames/SegmentPreBuffer)

    data, time = read_stitch_binary(top, bottom, dtypes=dtype,
                                    total_buffer_frames=int(total_buffer_frames),
                                    offset=8192,
                                    navigation_shape=navigation_shape)
    return data, time


def read_stitch_binary(top, bottom, dtypes, offset,
                       total_buffer_frames=None, navigation_shape=None):
    keys = [d[0] for d in dtypes]
    top_mapped = np.memmap(top,
                           offset=offset,
                           dtype=dtypes,
                           shape=total_buffer_frames)
    bottom_mapped = np.memmap(bottom,
                              offset=offset,
                              dtype=dtypes,
                              shape=total_buffer_frames)

    array = da.concatenate([da.flip(top_mapped["Array"].reshape(-1, *top_mapped["Array"].shape[2:]), axis=1),
                            bottom_mapped["Array"].reshape(-1, *bottom_mapped["Array"].shape[2:])],
                            1)
    if navigation_shape is not None and navigation_shape != ():
        cut = np.product(navigation_shape)
        array = array[:cut]
        new_shape = tuple(navigation_shape) + array.shape[1:]
        array = array.reshape(new_shape)

    time = {k: bottom_mapped[k] for k in keys if k not in ["Array", "empty"]}
    return array, time


def read_ref(file_name):
    """Reads a reference image from the file using the file name as well as the width and height of the image. """
    if file_name is None:
        return
    try:
        shape = np.array(np.fromfile(file_name, dtype=np.int32, count=2), dtype=int)
        shape = tuple(shape[::-1])
        ref = np.memmap(file_name, dtype=np.float32, shape=shape, offset=1024)
        ref.shape
        return ref
    except FileNotFoundError:
        _logger.warning("No Dark Reference image found.  The Dark reference should be in the same directory "
                        "as the image and have the form xxx.seq.dark.mrc")
        return None



