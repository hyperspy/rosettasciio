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
import warnings
from pathlib import Path

import dask.array as da
import numpy as np

from rsciio._docstrings import (
    CHUNKS_READ_DOC,
    DISTRIBUTED_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    MMAP_DOC,
    NAVIGATION_SHAPE,
    RETURNS_DOC,
)
from rsciio.utils.distributed import memmap_distributed

_logger = logging.getLogger(__name__)


_PATH_DOCSTRING = """path : str or bytes
            The path to the ``mib`` file, otherwise the memory buffer
            of the ``mib`` file. Lazy loading is not supported with memory
            buffer.
        """

_FIRST_LAST_FRAME = """first_frame, last_frame : int or None, default=None
            The first/last frame to load. It follows python indexing syntax,
            i.e. negative integer means reverse indexing. If ``None``, it uses
            first/last index.
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
        self.dtype = None
        self.head_size = None
        self.offset = 0
        self.navigation_shape = ()
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
        str_ += "\nData size {}".format(self.merlin_size)
        if self.raw:  # pragma: no cover
            str_ += "\n\tData is RAW"
        else:
            str_ += "\nData is processed"
        str_ += "\nData type: {}".format(self.dtype)
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
        Parse headers of a MIB data and return object containing frame parameters

        Parameters
        ----------
        %s
        """

        # read the first header from the start of the file or buffer
        if isinstance(path, str):
            try:
                with open(path, "rb") as f:
                    head = f.read(384).decode().split(",")
                    f.seek(0, os.SEEK_END)
                    self.file_size = f.tell()
                    self.buffer = False
                    self.path = path
            except BaseException:  # pragma: no cover
                raise RuntimeError("File does not contain MIB header.")
        elif isinstance(path, bytes):
            try:
                head = path[:384].decode().split(",")
                self.file_size = len(path)
                self.buffer = True
            except BaseException:  # pragma: no cover
                raise RuntimeError("Buffer does not contain MIB header.")
        else:  # pragma: no cover
            raise TypeError("`path` must be a str or a buffer.")

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
            # We use the dtype name to be able to save it
            # as metadata
            if head[6] == "U08":
                self.dtype = np.dtype(">u1").name
                self.dynamic_range = "1 or 6-bit"
            if head[6] == "U16":
                self.dtype = np.dtype(">u2").name
                self.dynamic_range = "12-bit"
            if head[6] == "U32":
                self.dtype = np.dtype(">u4").name
                self.dynamic_range = "24-bit"

        self.exposure = _parse_exposure_to_ms(head[-3])
        self.timestamp = head[-4]

    parse_file.__doc__ %= _PATH_DOCSTRING


def load_mib_data(
    path,
    lazy=False,
    chunks="auto",
    mmap_mode=None,
    navigation_shape=None,
    first_frame=None,
    last_frame=None,
    distributed=False,
    mib_prop=None,
    return_headers=False,
    print_info=False,
    return_mmap=True,
):
    """
    Load Quantum Detectors MIB file from a path or a memory buffer.

    Parameters
    ----------
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    mib_prop : ``MIBProperties``, default=None
        The ``MIBProperties`` instance of the file. If None, it will be
        parsed from the file.
    return_headers : bool, default=False
        If True, also return headers.
    print_info : bool, default=False
        If True, display information when loading the file.
    return_mmap : bool
        If True, return the :class:`numpy.memmap` object. Default is True.

    Returns
    -------
    numpy.ndarray or dask.array.Array or numpy.memmap
        The data from the mib reshaped according to the ``navigation_shape``
        argument.

    """
    if mmap_mode is None:
        mmap_mode = "r" if lazy else "c"

    if mib_prop is None:
        mib_prop = MIBProperties()
        mib_prop.parse_file(path)

    if lazy and isinstance(path, bytes):
        raise ValueError("Loading memory buffer lazily is not supported.")

    # As we save the dtype name, we don't have the endianess and we
    # need to specify it here
    data_dtype = np.dtype(mib_prop.dtype).newbyteorder(">")
    merlin_frame_dtype = np.dtype(
        [
            ("header", np.bytes_, mib_prop.head_size),
            ("data", data_dtype, mib_prop.merlin_size),
        ]
    )
    # find the number of frames in the file
    frame_number_in_file = mib_prop.file_size // merlin_frame_dtype.itemsize

    # Get the frame slice to load, taking into `None` and negative indexing
    first_frame, last_frame, _ = slice(first_frame, last_frame).indices(
        frame_number_in_file
    )
    number_of_frames_to_load = int(last_frame - first_frame)

    if navigation_shape is None:
        # Use number_of_frames_to_load to support slicing range of frames
        navigation_shape = (number_of_frames_to_load,)
    elif isinstance(navigation_shape, tuple):
        frame_number = np.prod(navigation_shape)
        if frame_number > frame_number_in_file:
            # Case of interrupted acquisition
            # Set the corrected number of lines
            # To keep the implementation simple only load completed line
            # Reshape only when the slice from zeros
            if first_frame == 0 and len(navigation_shape) > 1:
                navigation_shape = (
                    navigation_shape[0],
                    frame_number_in_file // navigation_shape[0],
                )[::-1]
            else:
                navigation_shape = (number_of_frames_to_load,)
        elif number_of_frames_to_load < frame_number:
            # in case the given navigation is not None and the total number of frame
            # to load is too small, we can't reshape and we fall back to stack of images
            _logger.warning(
                "The `navigation_shape` doesn't match the number of frames to load. "
                "The `navigation_shape` is set to the number of read to read: "
                f"({number_of_frames_to_load},)."
            )
            navigation_shape = (number_of_frames_to_load,)
        else:
            navigation_shape = navigation_shape[::-1]
    else:
        raise TypeError("`navigation_shape` must be `None` or of tuple type.")

    mib_prop.navigation_shape = navigation_shape
    mib_prop.xy = np.prod(mib_prop.navigation_shape)
    mib_prop.frame_number_in_file = frame_number_in_file

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
        memmap_kwargs = dict(
            filename=mib_prop.path,
            # take into account first_frame
            offset=mib_prop.offset + merlin_frame_dtype.itemsize * first_frame,
            # need to use np.prod(navigation_shape) to crop number line
            shape=np.prod(navigation_shape),
            dtype=merlin_frame_dtype,
        )
        if distributed:
            data = memmap_distributed(chunks=chunks, key="data", **memmap_kwargs)
            if not lazy:
                data = data.compute()
                # get_file_handle(data).close()
        else:
            data = np.memmap(mode=mmap_mode, **memmap_kwargs)
    elif isinstance(path, bytes):
        data = np.frombuffer(
            path,
            dtype=merlin_frame_dtype,
            count=mib_prop.xy,
            offset=mib_prop.offset,
        )

    else:  # pragma: no cover
        raise TypeError("`path` must be a str or a buffer.")

    if not distributed:
        headers = data["header"]
        data = data["data"]
    if not return_mmap:
        if not distributed and lazy:
            if isinstance(chunks, tuple) and len(chunks) > 2:
                # Since the data is reshaped later on, we set only the
                # signal dimension chunks here
                _chunks = ("auto",) + chunks[-2:]
            else:
                _chunks = chunks
            data = da.from_array(data, chunks=_chunks)
        else:
            data = np.array(data)

    # remove navigation_dimension with value 1 before reshaping
    navigation_shape = tuple(i for i in navigation_shape if i > 1)
    data = data.reshape(navigation_shape + mib_prop.merlin_size)
    if lazy and isinstance(chunks, tuple) and len(chunks) > 2:
        # rechunk navigation space when chunking is specified as a tuple
        data = data.rechunk(chunks)

    if return_headers:
        if distributed:
            raise ValueError(
                "Retuning headers is not supported with `distributed=True`."
            )
        return data, headers
    else:
        return data


load_mib_data.__doc__ %= (
    _PATH_DOCSTRING,
    LAZY_DOC,
    CHUNKS_READ_DOC,
    MMAP_DOC,
    NAVIGATION_SHAPE,
    _FIRST_LAST_FRAME,
    DISTRIBUTED_DOC,
)


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


_HEADERS_DOCSTRING = """headers : bytes str or iterable of bytes str
        The headers as a bytes string.
    """


_MAX_INDEX_DOCSTRING = """max_index : int
        Define the maximum index of the frame to be considered to avoid
        reading the header of all frames. If -1 (default), all frames will
        be read.
    """


def parse_exposures(headers, max_index=10000):
    """
    Parse the exposure time from the header of each frames.

    Parameters
    ----------
    %s
    %s

    Returns
    -------
    list
        The exposure in ms of each frame.

    Examples
    --------
    Use ``load_mib_data`` function to the headers and parse the exposures
    from the headers. By default, reads only the first 10 000 frames.

    >>> from rsciio.quantumdetector import load_mib_data, parse_exposures
    >>> data, headers = load_mib_data(path, return_headers=True, return_mmap=True)
    >>> exposures = parse_exposures(headers)

    All frames can be parsed by using ``max_index=-1``:

    >>> data, headers = load_mib_data(path, return_headers=True)
    >>> timestamps = parse_exposures(headers, max_index=-1)
    >>> len(timestamps)
    65536
    """
    if isinstance(headers, bytes):
        headers = [headers]

    if max_index > 1:
        max_index = min(max_index, len(headers))

    # exposure time are in ns
    return [
        _parse_exposure_to_ms(header.decode().split(",")[-3])
        for header in headers[:max_index]
    ]


parse_exposures.__doc__ %= (_HEADERS_DOCSTRING, _MAX_INDEX_DOCSTRING)


def parse_timestamps(headers, max_index=10000):
    """
    Parse the timestamp time from the header of each frames.

    Parameters
    ----------
    %s
    %s

    Returns
    -------
    list
        The timestamp of each frame.

    Examples
    --------
    Use ``load_mib_data`` function to get the headers and parse the timestamps
    from the headers. By default, reads only the first 10 000 frames.

    >>> from rsciio.quantumdetector import load_mib_data, parse_exposures
    >>> data, header = load_mib_data(path, return_headers=True)
    >>> timestamps = parse_timestamps(headers)
    >>> len(timestamps)
    10000

    All frames can be parsed by using ``max_index=-1``:

    >>> data, headers = load_mib_data(path, return_headers=True)
    >>> timestamps = parse_timestamps(headers, max_index=-1)
    >>> len(timestamps)
    65536

    """
    if isinstance(headers, bytes):
        headers = [headers]

    if max_index > 1:
        max_index = min(max_index, len(headers))

    return [header.decode().split(",")[-4] for header in headers[:max_index]]


parse_timestamps.__doc__ %= (_HEADERS_DOCSTRING, _MAX_INDEX_DOCSTRING)


def file_reader(
    filename,
    lazy=False,
    chunks="auto",
    mmap_mode=None,
    navigation_shape=None,
    first_frame=None,
    last_frame=None,
    distributed=False,
    print_info=False,
):
    """
    Read a Quantum Detectors ``mib`` file.

    If a ``hdr`` file with the same file name was saved along the ``mib`` file,
    it will be used to read the metadata.

    Parameters
    ----------
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    print_info : bool
        Display information about the mib file.

    %s

    Notes
    -----
    In case of interrupted acquisition, only the completed lines are read and
    the incomplete line are discarded.

    When the scanning shape (i. e. navigation shape) is not available from the
    metadata (for example with acquisition using pixel trigger), the timestamps
    will be used to guess the navigation shape.

    Examples
    --------
    In case, the navigation shape can't read from the data itself (for example,
    type of acquisition unsupported), the ``navigation_shape`` can be specified:

    .. code-block:: python

        >>> from rsciio.quantumdetector import file_reader
        >>> s_dict = file_reader("file.mib", navigation_shape=(256, 256))

    """
    mib_prop = MIBProperties()
    mib_prop.parse_file(filename)
    hdr_filename = str(filename).replace(".mib", ".hdr")

    original_metadata = {"mib_properties": vars(mib_prop)}

    if Path(hdr_filename).exists():
        hdr = parse_hdr_file(hdr_filename)
        original_metadata["hdr_file"] = hdr
    else:
        hdr = None
        _logger.warning("`hdr` file couldn't be found.")

    frame_per_trigger = 1
    headers = None
    if navigation_shape is None:
        if hdr is not None:
            # Use the hdr file to find the number of frames
            frame_per_trigger = int(hdr["Frames per Trigger (Number)"])
            frames_number = int(hdr["Frames in Acquisition (Number)"])
        else:
            _, headers = load_mib_data(filename, return_headers=True)
            frames_number = len(headers)

        if frame_per_trigger == 1:
            if headers is None:
                _, headers = load_mib_data(filename, return_headers=True)
            # Use parse_timestamps to find the number of frame per line
            # we will get a difference of timestamps at the beginning of each line
            with warnings.catch_warnings():
                # Filter warning for converting timezone aware datetime
                # The time zone is dropped
                # Changed from `DeprecationWarning` to `UserWarning` in numpy 2.0
                warnings.simplefilter("ignore")
                times = np.array(parse_timestamps(headers)).astype(dtype="datetime64")

            times_diff = np.diff(times).astype(float)
            if len(times_diff) > 0:
                # Substract the mean and take the first position above 0
                indices = np.argwhere(times_diff - np.mean(times_diff) > 0)
                if len(indices) > 0 and len(indices[0]) > 0:
                    frame_per_trigger = indices[0][0] + 1

        if frames_number == 0:
            # Some hdf files have the "Frames per Trigger (Number)": 0
            # in this case, we don't reshape
            # Possibly for "continuous and indefinite" acquisition
            navigation_shape = None
        else:
            navigation_shape = (frame_per_trigger, frames_number // frame_per_trigger)

    data = load_mib_data(
        filename,
        lazy=lazy,
        chunks=chunks,
        mmap_mode=mmap_mode,
        navigation_shape=navigation_shape,
        first_frame=first_frame,
        last_frame=last_frame,
        distributed=distributed,
        mib_prop=mib_prop,
        print_info=print_info,
        return_mmap=False,
    )
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
        "Signal": {"signal_type": "electron_diffraction"},
        "Acquisition_instrument": {
            "dwell_time": mib_prop.exposure * 1e-3,  # ms to s
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


file_reader.__doc__ %= (
    FILENAME_DOC,
    LAZY_DOC,
    CHUNKS_READ_DOC,
    MMAP_DOC,
    NAVIGATION_SHAPE,
    _FIRST_LAST_FRAME,
    DISTRIBUTED_DOC,
    RETURNS_DOC,
)
