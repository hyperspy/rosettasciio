# -*- coding: utf-8 -*-
# Copyright 2023-2023 The HyperSpy developers
# Copyright 2021-2023 Matus Krajnak
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
#
# Adapted from https://github.com/matkraj/read_mib under GPL-3.0 license

import logging
import os
from pathlib import Path

import numpy as np

from rsciio._docstrings import (
    FILENAME_DOC,
    LAZY_DOC,
    MMAP_DOC,
    NAVIGATION_SHAPE,
    RETURNS_DOC,
)


_logger = logging.getLogger(__name__)


_PATH_DOCSTRING = """path : str or buffer
            The path to the ``mib`` file, otherwise the memory buffer
            of the ``mib`` file.
        """


class MIBProperties:
    """Class covering Merlin MIB file properties."""

    def __init__(self):
        """
        Initialisation of default MIB properties. Single detector, 1 frame, 12 bit"""
        self.path = ""
        self.buffer = True
        self.merlin_size = (256, 256)
        self.single = True
        self.quad = False
        self.raw = False
        self.dynamic_range = "12-bit"
        self.packed = False
        self.pixel_type = "uint16"
        self.head_size = 384
        self.offset = 0
        self.navigation_shape = tuple()
        self.xy = 1
        self.number_of_frames_in_file = 1
        self.gap = 0
        self.quad_scale = 1
        self.detector_geometry = "1x1"
        self.frame_double = 1
        self.roi_rows = 256
        self.file_size = None

    def __repr__(self):
        """
        Show current properties of the Merlin file.
        Use parse_mib_properties(path/buffer) to populate
        """
        if not self.buffer:
            print("\nPath:", self.path)
        else:
            print("\nData is from a buffer")
        if self.single:
            print("\tData is single")
        if self.quad:
            print("\tData is quad")
            print("\tDetector geometry", self.detector_geometry)
        print("\tData pixel size", self.merlin_size)
        if self.raw:
            print("\tData is RAW")
        else:
            print("\tData is processed")
        print("\tPixel type:", np.dtype(self.pixel_type))
        print("\tDynamic range:", self.dynamic_range)
        print("\tHeader size:", self.head_size, "bytes")
        print("\tNumber of frames in the file/buffer:", self.number_of_frames_in_file)
        print("\tNumber of frames to be read:", self.xy)

    def parse_file(self, path):
        """
        Parse header of a MIB data and return object containing frame parameters

        Parameters
        ----------
        %s
        """

        # read header from the start of the file or buffer
        if isinstance(path, str):
            try:
                with open(path, "rb") as f:
                    head = f.read(384).decode().split(",")
                    f.seek(0, os.SEEK_END)
                    self.file_size = f.tell()
            except:
                raise RuntimeError("File does not contain MIB header.")
        elif isinstance(path, bytes):
            try:
                head = path[:384].decode().split(",")
                self.file_size = len(path)
            except:
                raise RuntimeError("Buffer does not contain MIB header.")
        else:
            raise ValueError("`path` must be a str or a buffer.")

        # parse header info
        self.path = path
        # read detector size
        self.merlin_size = (int(head[4]), int(head[5]))

        # test if RAW
        if head[6] == "R64":
            self.raw = True

        if head[7].endswith("2x2"):
            self.detector_geometry = "2x2"
        if head[7].endswith("Nx1"):
            self.detector_geometry = "Nx1"

        # test if single
        if head[2] == "00384":
            self.single = True
        # test if quad and read full quad header
        if head[2] == "00768":
            # read quad data
            with open(self.path, "rb") as f:
                head = f.read(768).decode().split(",")
            self.head_size = 768
            self.quad = True
            self.single = False

        # set bit-depths for processed data (binary is U08 as well)
        if not self.raw:
            if head[6] == "U08":
                self.pixel_type = np.dtype("uint8").name
                self.dynamic_range = "1 or 6-bit"
            if head[6] == "U16":
                self.pixel_type = np.dtype(">u2").name
                self.dynamic_range = "12-bit"
            if head[6] == "U32":
                self.pixel_type = np.dtype(">u4").name
                self.dynamic_range = "24-bit"

    parse_file.__doc__ %= _PATH_DOCSTRING


def load_mib_data(
    path,
    lazy=False,
    mmap_mode=None,
    navigation_shape=None,
    mib_prop=None,
):
    """
    Load Quantum Detectors MIB file from a path or a memory buffer.

    Parameters
    ----------
    %s
    %s
    %s
    %s

    Returns
    -------
    data : numpy.memmap
        The data from the mib reshape according to the ``navigation_shape``
        argument.

    """
    if mmap_mode is None:
        mmap_mode = "r" if lazy else "c"

    if mib_prop is None:
        mib_prop = MIBProperties()
        mib_prop.parse_file(path)

    # find the size of the data
    merlin_frame_dtype = np.dtype(
        [
            ("header", np.string_, mib_prop.head_size),
            ("data", mib_prop.pixel_type, mib_prop.merlin_size),
        ]
    )
    mib_prop.number_of_frames_in_file = (
        mib_prop.file_size // merlin_frame_dtype.itemsize
    )

    if navigation_shape is None:
        # Use number_of_frames_in_file
        mib_prop.navigation_shape = (mib_prop.number_of_frames_in_file,)
    elif isinstance(navigation_shape, tuple):
        mib_prop.navigation_shape = navigation_shape
    else:
        raise ValueError("`navigation_shape` must be `None` or a tuple.")

    mib_prop.xy = np.prod(mib_prop.navigation_shape)

    # correct for buffer/file logic
    if isinstance(path, str):
        mib_prop.buffer = False

    if mib_prop.xy > mib_prop.number_of_frames_in_file:
        # TODO: check if this related to interrupted acquision, if so,
        # add support for reading these by appending suitable number of frames
        raise RuntimeError(
            f"Requested number of frames: {mib_prop.xy} is smaller than the "
            f"number of available frames {mib_prop.number_of_frames_in_file}."
        )

    if mib_prop.raw:
        raise NotImplementedError("RAW MIB data not supported.")

    # map the file to memory, if a numpy or memmap array is given, work with
    # it as with a buffer
    # buffer needs to have the exact structure of MIB file,
    # if it is read from TCPIP interface it needs to drop first 15 bytes which
    # describe the stream size. Also watch for the coma in front of the stream.
    if isinstance(mib_prop.path, str):
        data = np.memmap(
            mib_prop.path,
            dtype=merlin_frame_dtype,
            offset=mib_prop.offset,
            shape=mib_prop.navigation_shape,
        )

    elif isinstance(mib_prop.path, bytes):
        data = np.frombuffer(
            mib_prop.path,
            dtype=merlin_frame_dtype,
            count=mib_prop.xy,
            offset=mib_prop.offset,
        )
        data = data.reshape(mib_prop.navigation_shape + mib_prop.merlin_size)
    else:
        raise ValueError("`path` must be a str or a buffer.")

    # remove header data and return
    return data["data"]


load_mib_data.__doc__ %= (_PATH_DOCSTRING, LAZY_DOC, MMAP_DOC, NAVIGATION_SHAPE)


def parse_hdr_file(path):
    result = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("HDR") or line.startswith("End\t"):
                continue
            k, v = line.split("\t", 1)
            k = k.rstrip(":")
            v = v.rstrip("\n")
            result[k] = v

    return result


def file_reader(filename, lazy=False, mmap_mode=None, navigation_shape=None):
    """
    Read a Quantum Detectors ``mib`` file

    Parameters
    ----------
    %s
    %s
    %s
    %s

    %s
    """
    mib_prop = MIBProperties()
    mib_prop.parse_file(filename)
    hdr_filename = str(filename).replace(".mib", ".hdr")
    hdr = None

    original_metadata = {"mib_properties": vars(mib_prop)}

    if Path(hdr_filename).exists():
        hdr = parse_hdr_file(hdr_filename)
        original_metadata["hdr_file"] = hdr
    else:
        _logger.info("`hdr` file couldn't be found.")

    if navigation_shape is None and hdr is not None:
        # (x, y)
        navigation_shape = (
            int(hdr["Frames per Trigger (Number)"]),
            int(hdr["Frames in Acquisition (Number)"])
            // int(hdr["Frames per Trigger (Number)"]),
        )

    data = load_mib_data(
        filename,
        mmap_mode=mmap_mode,
        navigation_shape=navigation_shape,
        mib_prop=mib_prop,
    ).squeeze()
    data = np.flip(data, axis=-2)

    # data has 3 dimension but we need to to take account the dimension of the
    # navigation_shape after reshape
    dim = len(data.shape)
    navigates = [True] * (dim - 2) + [False, False]
    axes = [
        {
            "size": data.shape[i],
            "index_in_array": i,
            "name": "",
            "scale": 1.0,
            "offset": 0.0,
            "units": "",
            "navigate": nav,
        }
        for i, nav in enumerate(navigates)
    ]

    metadata = {
        "General": {"original_filename": os.path.split(filename)[1]},
        "Signal": {"signal_type": ""},
    }

    dictionary = {
        "data": data,
        "axes": axes,
        "metadata": metadata,
        "original_metadata": original_metadata,
        "mapping": {},
    }

    return [
        dictionary,
    ]


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, MMAP_DOC, NAVIGATION_SHAPE, RETURNS_DOC)
