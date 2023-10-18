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


_PATH_DOCSTRING = """path : str or bytes
            The path to the ``mib`` file, otherwise the memory buffer
            of the ``mib`` file. Lazy loading is not supported with memory
            buffer
        """


class MIBProperties:
    """Class covering Merlin MIB file properties."""

    def __init__(self):
        """
        Initialisation of default MIB properties. Single detector, 1 frame, 12 bit
        """
        self.path = None
        self.buffer = None
        self.merlin_size = ()
        self.assembly_size = None
        self.raw = False
        self.dynamic_range = ""
        self.packed = None
        self.pixel_type = ""
        self.head_size = None
        self.offset = 0
        self.navigation_shape = list()
        self.xy = None
        self.number_of_frames_in_file = None
        self.gap = None
        self.quad_scale = None
        self.detector_geometry = ""
        self.frame_double = None
        self.roi_rows = None
        self.exposure = None
        self.timestamp = ""
        self.file_size = None

    def __repr__(self):
        """
        Show current properties of the Merlin file.
        Use parse_mib_properties(path/buffer) to populate
        """
        str_ = ""
        if not self.buffer:
            str_ += "\nPath: {}".format(self.path)
        else:  # pragma: no cover
            str_ += "\nData is from a buffer"

        str_ += "\nChip configuration is {}".format(self.assembly_size)
        if self.assembly_size == "quad":
            str_ += "\nDetector geometry: {}".format(self.detector_geometry)
        str_ += "\nData pixel size {}".format(self.merlin_size)
        if self.raw:  # pragma: no cover
            str_ += "\n\tData is RAW"
        else:
            str_ += "\nData is processed"
        str_ += "\nPixel type: {}".format(np.dtype(self.pixel_type))
        str_ += "\nDynamic range: {}".format(self.dynamic_range)
        str_ += "\nHeader size: {} bytes".format(self.head_size)
        str_ += "\nNumber of frames in the file/buffer: {}".format(
            self.number_of_frames_in_file
        )
        str_ += "\nNumber of frames to be read: {}".format(self.xy)
        str_ += "\nexposure: {}".format(self.exposure)
        str_ += "\ntimestamp: {}".format(self.timestamp)

        return str_

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
                    self.buffer = False
                    self.path = path
            except:  # pragma: no cover
                raise RuntimeError("File does not contain MIB header.")
        elif isinstance(path, bytes):
            try:
                head = path[:384].decode().split(",")
                self.file_size = len(path)
                self.buffer = True
            except:  # pragma: no cover
                raise RuntimeError("Buffer does not contain MIB header.")
        else:  # pragma: no cover
            raise ValueError("`path` must be a str or a buffer.")

        # read detector size
        self.merlin_size = (int(head[4]), int(head[5]))

        # test if RAW
        if head[6] == "R64":  # pragma: no cover
            self.raw = True

        if head[7].endswith("2x2"):
            self.detector_geometry = "2x2"
        if head[7].endswith("Nx1"):  # pragma: no cover
            self.detector_geometry = "Nx1"

        # test if single
        if head[2] == "00384":
            self.head_size = 384
            self.assembly_size = "single"
        # test if quad and read full quad header
        if head[2] == "00768":
            # read quad data
            with open(self.path, "rb") as f:
                head = f.read(768).decode().split(",")
            self.head_size = 768
            self.assembly_size = "quad"

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

        self.exposure = _parse_exposure_to_ms(head[-3])
        self.timestamp = head[-4]

    parse_file.__doc__ %= _PATH_DOCSTRING


def load_mib_data(
    path,
    lazy=False,
    mmap_mode=None,
    navigation_shape=None,
    mib_prop=None,
    return_header=False,
    print_info=False,
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
        mib_prop.navigation_shape = [
            mib_prop.number_of_frames_in_file,
        ]
    elif isinstance(navigation_shape, (list, tuple)):
        mib_prop.navigation_shape = navigation_shape = list(navigation_shape)
    else:  # pragma: no cover
        raise ValueError("`navigation_shape` must be `None` or a tuple.")

    mib_prop.xy = np.prod(mib_prop.navigation_shape)

    if print_info:
        print(mib_prop)

    if mib_prop.raw:  # pragma: no cover
        raise NotImplementedError("RAW MIB data not supported.")

    # map the file to memory, if a numpy or memmap array is given, work with
    # it as with a buffer
    # buffer needs to have the exact structure of MIB file,
    # if it is read from TCPIP interface it needs to drop first 15 bytes which
    # describe the stream size. Also watch for the coma in front of the stream.
    if isinstance(mib_prop.path, str):
        if mib_prop.xy > mib_prop.number_of_frames_in_file:
            # Case of interrupted acquisition, add move data
            # Set the corrected number of lines
            # To keep the implementation simple only read completed line
            navigation_shape[0] = (
                mib_prop.number_of_frames_in_file // mib_prop.navigation_shape[1]
            )

        data = np.memmap(
            mib_prop.path,
            dtype=merlin_frame_dtype,
            offset=mib_prop.offset,
            # need to use np.prod(navigation_shape) to crop number line
            shape=np.prod(navigation_shape),
            mode=mmap_mode,
        )

    elif isinstance(path, bytes):
        data = np.frombuffer(
            path,
            dtype=merlin_frame_dtype,
            count=mib_prop.xy,
            offset=mib_prop.offset,
        )
    else:  # pragma: no cover
        raise ValueError("`path` must be a str or a buffer.")

    if navigation_shape is not None and len(navigation_shape) > 1:
        # Only reshape when necessary
        # reverse navigation_shape to match hyperspy ordering
        data = data.reshape(navigation_shape[::-1])

    if return_header:
        return data["data"], data["header"]
    else:
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


def _parse_exposure_to_ms(str_):
    # exposure is in "ns", remove unit, convert to float and to ms
    return float(str_[:-2]) / 1e6


_HEADER_DOCSTRING = """header : bytes str or iterable of bytes str
        The header as a bytes string.
    """


_MAX_INDEX_DOCSTRING = """max_index : int
        Define the maximum index of the frame to be considered to avoid
        reading the header of all frames. If -1 (default), all frames will
        be read.
    """


def parse_exposures(header, max_index=10000):
    """
    Parse the exposure time from the header of each frames.

    Parameters
    ----------
    %s
    %s

    Returns
    -------
    exposures : list
        The exposure in ms of each frame.

    Examples
    --------
    >>> from rsciio.quantumdetector import load_mib_data, parse_exposures
    >>> data, header = load_mib_data(path, return_header=True)
    >>> exposures = parse_exposures(header)

    """
    if isinstance(header, bytes):
        header = [header]

    if max_index > 1:
        max_index = min(max_index, len(header))

    # exposure time are in ns
    return [
        _parse_exposure_to_ms(header_.decode().split(",")[-3])
        for header_ in header[:max_index]
    ]


parse_exposures.__doc__ %= (_HEADER_DOCSTRING, _MAX_INDEX_DOCSTRING)


def parse_timestamps(header, max_index=10000):
    """
    Parse the timestamp time from the header of each frames.

    Parameters
    ----------
    %s
    %s

    Returns
    -------
    timestamps : list
        The timestamp of each frame.

    Examples
    --------
    >>> from rsciio.quantumdetector import load_mib_data, parse_exposures
    >>> data, header = load_mib_data(path, return_header=True)
    >>> timestamps = parse_timestamps(header)
    >>> len(timestamps)
    10000

    By default, the timestamp of the first 10 000 frames are parsed. All frames
    can be parsed by using ``max_index=-1``:

    >>> data, header = load_mib_data(path, return_header=True)
    >>> timestamps = parse_timestamps(header, max_index=-1)
    >>> len(timestamps)
    65536

    """
    if isinstance(header, bytes):
        header = [header]

    if max_index > 1:
        max_index = min(max_index, len(header))

    return [header_.decode().split(",")[-4] for header_ in header[:max_index]]


parse_timestamps.__doc__ %= (_HEADER_DOCSTRING, _MAX_INDEX_DOCSTRING)


def file_reader(
    filename, lazy=False, mmap_mode=None, navigation_shape=None, print_info=False
):
    """
    Read a Quantum Detectors ``mib`` file

    Parameters
    ----------
    %s
    %s
    %s
    %s

    %s

    Note
    ----
    In case of interrupted acquisition, only the completed line are read and
    the incomplete line are discarded.

    """
    mib_prop = MIBProperties()
    mib_prop.parse_file(filename)
    hdr_filename = str(filename).replace(".mib", ".hdr")
    hdr = None

    original_metadata = {"mib_properties": vars(mib_prop)}

    if Path(hdr_filename).exists():
        hdr = parse_hdr_file(hdr_filename)
        original_metadata["hdr_file"] = hdr
    else:  # pragma: no cover
        _logger.warning("`hdr` file couldn't be found.")

    if navigation_shape is None and hdr is not None:
        # (x, y)
        navigation_shape = [
            int(hdr["Frames per Trigger (Number)"]),
            int(hdr["Frames in Acquisition (Number)"])
            // int(hdr["Frames per Trigger (Number)"]),
        ]

    data = load_mib_data(
        filename,
        mmap_mode=mmap_mode,
        navigation_shape=navigation_shape,
        mib_prop=mib_prop,
        print_info=print_info,
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

    date, time = mib_prop.timestamp.split("T")
    if "Z" in time:
        time = time.strip("Z")
        time_zone = "UTC"
    else:  # pragma: no cover
        time_zone = None

    metadata = {
        "General": {
            "original_filename": os.path.split(filename)[1],
            "date": date,
            "time": time,
        },
        "Signal": {"signal_type": ""},
        "Acquisition_instrument": {
            "dwell_time": mib_prop.exposure * 1e-6,
        },
    }
    if time_zone:
        metadata["General"]["time_zone"] = time_zone

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
