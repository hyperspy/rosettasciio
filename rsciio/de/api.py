# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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


import numpy as np
import logging
import glob

import dask.array as da

from rsciio.utils.tools import read_binary_metadata, parse_xml, get_chunk_index, memmap_distributed

_logger = logging.getLogger(__name__)
data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}  # Stream Pix data types


def file_reader(filename,
                navigation_shape=None,
                lazy=False,
                celeritas=False,
                chunks="auto",
                distributed=False,
                **kwargs):
    """ Reads the .seq file format from the DE 16 and DE Celeritas cameras.  This file
    format is generic and used by the 3rd party software StreamPix.  While this
    file loader may load data saved from other cameras it is not guaranteed  to

    Parameters
    ----------
    filename: str
        The file name to be loaded. This should have a `.seq` ending to the file.
        All additional information (dark, gain, metadata) should be in the same
        file with the same naming scheme.

        If celeritas ==True either the bottom or top frame can be passed and the other frame
        will automatically be found if it is in the same folder with the same naming
        scheme.
    navigation_shape: tuple
        The shape of the navigation axis. This will coerce the data into the shape
        given. Adding extra dimensions as necessary
    lazy: bool
        If the data should be loaded lazily using dask
    celeritas: bool
        If the data came from the celeritas camera.  Important for loading
        data saved with a prebuffer and split between top and bottom frames.
    distributed: bool
        If the data should be loaded in a way that is supported by the distributed
        backend. Slightly slower for smaller datasets but could potentially see
        gains on larger datasets or more extensive hardware.
    kwargs:
        Any additional parameters such as:
            top: str
                The name of the top filename if passed directly
            bottom: str
                The name of the bottom file if passed directly
            gain: str
                The name of the gain image if passed directly
            metadata: str
                The name of the metadata file if passed directly
            xml: str
                The name of the xml file if passed directly

    Returns
    -------

    """
    if celeritas:
        if "top" not in kwargs and "bottom" not in kwargs:
            if "Top" in filename:
                top = filename
                leading_str = filename.rsplit("_Top", 1)[0]
                bottom = glob.glob(leading_str + "_Bottom*.seq")[0]
                filename = leading_str + ".seq"

            elif "Bottom" in filename:
                bottom = filename
                leading_str = filename.rsplit("_Bottom", 1)[0]
                top = glob.glob(leading_str + "_Top*.seq")[0]
                filename = leading_str + ".seq"
            if "metadata" not in kwargs:
                kwargs["metadata"] = bottom + ".metadata"
            else:
                _logger.error(
                    msg="For the Celeritas Camera Top and Bottom "
                    "frames must be explicitly given by passing the"
                    "top and bottom kwargs or the file name must have"
                    "'Top' or 'Bottom' in the file name"
                )
        elif celeritas and "top" in kwargs and "bottom" in kwargs:
            top = kwargs["top"]
            bottom = kwargs["top"]
        else:
            _logger.error(
                msg="For the Celeritas Camera Top and Bottom "
                "frames must be explicitly given by passing the"
                "top and bottom kwargs or the file name must have"
                "'Top' or 'Bottom' in the file name"
            )

    file_extensions = {
        "metadata": ".metadata",
        "dark": ".dark.mrc",
        "gain": ".gain.mrc",
        "xml": ".Config.Metadata.xml",
    }
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
    return reader.read_data(navigation_shape=navigation_shape,
                            lazy=lazy,
                            distributed=distributed,
                            chunks=chunks)


class SeqReader:
    def __init__(self, file, dark=None, gain=None, metadata=None, xml=None):
        """
        Initializes a general reader for the .seq file format. Some functions are
        overwritten with the CeleritasSEQ reader

        Parameters
        ----------
        file: str
            The signal to be read.
        dark: str
            The file path for the dark reference to be read.
        gain: str
            The file path for the gain reference to be read.
        metadata: str
            The file path for the metadata to be read
        xml: str
            The file path for the xml file to be read.
        """
        self.file = file
        self.metadata_file = metadata
        self.dark = dark
        self.gain = gain
        self.xml = xml
        # Output
        self.original_metadata = {
            "InputFiles": {
                "file": file,
                "metadata": self.metadata_file,
                "dark": dark,
                "gain": gain,
                "xml": xml,
            }
        }
        self.metadata = {
            "General": {"filename": file,},
            "Signal": {"signal_type": "Signal2D"},
        }
        self.data = None
        self.axes = []
        self.buffer = None

    def _read_metadata(self):
        metadata_header_dict = {
            "Version": ["<u4", 0],
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
        self.metadata["acquisition_instrument"] = {
            "TEM": {
                "camera_length": metadata_dict["CameraLength"],
                "magnification": metadata_dict["Magnification"],
            }
        }
        return metadata_dict

    def _read_file_header(self):
        file_header_dict = {
            "ImageWidth": ["<u4", 548],
            "ImageHeight": ["<u4", 552],
            "ImageBitDepth": ["<u4", 556],
            "ImageBitDepthReal": ["<u4", 560],
            "NumFrames": ["<i", 572],
            "TrueImageSize": ["<i", 580],
            "FPS": ["<d", 584],
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
        self.metadata["Reference"] = {"dark": dark_img, "gain": gain_img}
        return dark_img, gain_img

    def _create_axes(self, header, nav_shape=None, prebuffer=None):
        if nav_shape is None or nav_shape == ():
            self.axes.append(
                {
                    "name": "time",
                    "offset": 0,
                    "unit": "sec",
                    "scale": 1 / header["FPS"],
                    "size": header["NumFrames"],
                    "navigate": True,
                    "index_in_array": 0,
                }
            )
        else:
            for s in nav_shape:
                self.axes.append(
                    {
                        "offset": 0,
                        "scale": 1,
                        "size": s,
                        "navigate": True,
                        "index_in_array": 0,
                    }
                )
        if prebuffer is not None:
            self.axes.append(
                {
                    "name": "y",
                    "offset": 0,
                    "scale": 1,
                    "size": int(header["ImageHeight"] * 2 / prebuffer),
                    "navigate": False,
                    "index_in_array": 1,
                }
            )
        else:
            self.axes.append(
                {
                    "name": "y",
                    "offset": 0,
                    "scale": 1,
                    "size": header["ImageHeight"],
                    "navigate": False,
                    "index_in_array": 1,
                }
            )
        self.axes.append(
            {
                "name": "x",
                "offset": 0,
                "scale": 1,
                "size": header["ImageWidth"],
                "navigate": False,
                "index_in_array": 2,
            }
        )
        if (
            self.original_metadata["Metadata"] != {}
            and self.original_metadata["Metadata"]["PixelSize"] > 1e-30
        ):
            # need to still determine a way to properly set units and scale
            self.axes[-2]["scale"] = self.original_metadata["Metadata"]["PixelSize"]
            self.axes[-1]["scale"] = self.original_metadata["Metadata"]["PixelSize"]
        return

    def read_data(self, navigation_shape=None, chunks="auto", distributed=False, lazy=False):
        """Reads the data from self.file given a navigation shape.
        Parameters
        ----------
        navigation_shape: None or tuple
            Reads the data and then coerces it into the navigation shape
            given
        """
        header = self._read_file_header()
        dark_img, gain_img = self._read_dark_gain()
        self._read_xml()
        self._read_metadata()
        signal_shape = (header["ImageWidth"], header["ImageHeight"])
        num_frames = header["NumFrames"]
        if navigation_shape is None or navigation_shape == ():
            navigation_shape = (header["NumFrames"],)

        data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}
        empty = header["TrueImageSize"] - ((header["ImageWidth"] * header["ImageHeight"] * 2) + 8)
        dtype = [
            ("Array", data_types[int(header["ImageBitDepth"])],
             (header["ImageWidth"],
              header["ImageHeight"])),
            ("sec", "<u4"),
            ("ms", "<u2"),
            ("mis", "<u2"),
            ("empty", bytes, empty),
        ]

        # This is useful if the camera drops frames at the end of an acquisition.
        if navigation_shape is not None and np.product(navigation_shape) > num_frames:
            _logger.warning(
                "The number of frames and the navigation shape are not "
                "equal. Adding frames to the end of the dataset. ")
            t = np.memmap(self.file, offset=8192,
                          dtype=dtype, shape=num_frames,
                          mode="r+")
            t.flush()
            num_frames = np.product(navigation_shape)

        if navigation_shape is not None:
            shape = navigation_shape + signal_shape
        else:
            shape = (num_frames,) + signal_shape


        if distributed:
            data = memmap_distributed(self.file, dtype, offset=8192,
                                      shape=navigation_shape, key="Array",
                                      chunks=chunks)
        else:
            data = np.memmap(self.file, dtype=dtype, offset=8192, shape=navigation_shape)
            time = {"sec": data["sec"], "ms": data["ms"], "mis": data["mis"]}
            data = data["Array"]
            if lazy:
                data = da.from_array(data, chunks=chunks)
            self.original_metadata["Timestamps"] = time
            self.metadata["Timestamps"] = time
        if dark_img is not None:
            data = data-dark_img
        if gain_img is not None:
            data = data*gain_img
        self._create_axes(header=header, nav_shape=navigation_shape)
        return {
            "data": data,
            "metadata": self.metadata,
            "axes": self.axes,
            "original_metadata": self.original_metadata,
        }


class CeleritasReader(SeqReader):
    def __init__(self, top, bottom, **kwargs):
        """
        Initializes a reader for the .seq file format written from the celeritas camera

        Parameters
        ----------
        file: str
            The signal to be read.
        dark: str
            The file path for the dark reference to be read.
        gain: str
            The file path for the gain reference to be read.
        metadata: str
            The file path for the metadata to be read
        xml: str
            The file path for the xml file to be read.
        top: str
            The signal from the top half of the camera
        bottom: str
            The signal from the bottom half of the camera.
        """
        super().__init__(**kwargs)
        self.top = top
        self.bottom = bottom
        self.bad_pixels = None

    def _read_file_header(self):
        file_header_dict = {
            "ImageWidth": ["<u4", 548],
            "ImageHeight": ["<u4", 552],
            "ImageBitDepth": ["<u4", 556],
            "ImageBitDepthReal": ["<u4", 560],
            "NumFrames": ["<i", 572],
            "TrueImageSize": ["<i", 580],
            "FPS": ["<d", 584],
        }
        metadata_dict = read_binary_metadata(self.top, file_header_dict)
        self.original_metadata["FileHeader"] = metadata_dict
        self.metadata["ImageHeader"] = metadata_dict
        return metadata_dict

    def _read_xml(self):
        xml_dict = parse_xml(self.xml)

        self.original_metadata["xml"] = xml_dict
        if xml_dict is not None:
            # The segment prebuffer is used to save data more
            # quickly as multiple images are saved in one buffer
            if "SegmentPreBuffer" in xml_dict["FileInfo"]:
                self.buffer = int(xml_dict["FileInfo"]["SegmentPreBuffer"]["Value"])
            try:  # try handling bad pixels
                #rows = int(xml_dict["BadPixels"]["BadPixelMap"]["Rows"])
                #columns = int(xml_dict["BadPixels"]["BadPixelMap"]["Columns"])
                defects = xml_dict["BadPixels"]["BadPixelMap"]["Defect"]
                if not isinstance(defects, list):
                    defects = [defects,]
                x_offset = xml_dict["FileInfo"]["ImageOffsetX"]["Value"]
                y_offset = xml_dict["FileInfo"]["ImageOffsetY"]["Value"]
                x_size = int(xml_dict["FileInfo"]["ImageSizeX"]["Value"])
                y_size = int(xml_dict["FileInfo"]["ImageSizeY"]["Value"])
                bad_pixels = np.zeros(shape=(y_size, x_size), dtype=bool)
                slices = []
                for defect in defects:
                    if "Column" in defect:
                        start = defect["Column"]
                        start = int(start)-int(x_offset)
                        col_slic = (start,) if 0 < start < x_size else ()
                    elif "Columns" in defect:
                        start, stop = defect["Columns"].split("-")
                        start = int(start) - int(x_offset)
                        start = start if 0 < start < x_size else None
                        stop = int(stop) - int(x_offset)
                        stop = stop if 0 < stop < x_size else None
                        col_slic = slice(start, stop) if start is not None and stop is not None else ()
                    else:
                        col_slic = slice(None)

                    if "Row" in defect:
                        start = defect["Row"]
                        start = int(start) - int(y_offset)
                        row_slic = (start,) if 0 < start < y_size else ()
                    elif "Rows" in defect:
                        start, stop = defect["Rows"].split("-")
                        start = int(start) - int(y_offset)
                        start = start if 0 < start < y_size else None
                        stop = int(stop) - int(y_offset)
                        stop = stop if 0 < stop < y_size else None
                        row_slic = slice(start, stop) if start is not None and stop is not None else ()
                    else:
                        row_slic = slice(None)
                    slices.append((row_slic, col_slic))
                for s in slices:
                    bad_pixels[s] = True
                self.original_metadata["BadPixels"] = bad_pixels
                self.metadata["Signal"]["BadPixels"] = bad_pixels
            except IndexError:
                _logger.info("Bad Pixel correction not preformed")
        return xml_dict

    def _read_metadata(self):
        # metadata specification might not always work as intended.
        metadata_header_dict = {
            "Version": ["<u4", 0],
            "MetadataVersion": [np.uintc, 276],
            "UnbinnedFrameWidth": [np.uintc, 280],
            "UnbinnedFrameHeight": ["<u4", 284],
            "HardwareBinning": ["<u4", 296],
            "CameraBitRate": ["<u4", 300],
            "FrameRate": [np.float64, 304],
            "ImageRotation": ["<u4", 312],
            "ImageFlipping": ["<u4", 316],
            "OKRA": [bool, 320],
            "GlobalShutter": [bool, 321],
            "OffChipCDS": [bool, 322],
            "AKRA": [bool, 323],
            "SensorSequence": ["<u4", 324],
            "CameraType": ["<u4", 328],
            "SensorGainMode": ["<u4", 332],
            "SensorGain": [np.float64, 336],
            "Magnification": [np.float64, 344],
            "PixelSize": [np.float64, 352],
            "CameraLength": [np.float64, 360],
            "DiffPixelSize": [np.float64, 368],
        }
        metadata_dict = read_binary_metadata(self.metadata_file, metadata_header_dict)
        self.original_metadata["Metadata"] = metadata_dict
        self.metadata["acquisition_instrument"] = {
            "TEM": {
                "camera_length": metadata_dict["CameraLength"],
                "magnification": metadata_dict["Magnification"],
            }
        }
        return metadata_dict

    def read_data(self, navigation_shape=None, chunks="auto",
                  lazy=False, distributed=False):
        header = self._read_file_header()
        dark_img, gain_img = self._read_dark_gain()
        self._read_xml()
        self._read_metadata()
        if self.buffer is None:  # could be improved based on the image bit size
            _logger.warning(
                msg="No XML File given. Guessing Segment PreBuffer "
                "This is may not be correct..."
            )
            self.buffer = int(header["ImageHeight"] / header["ImageWidth"])
        num_frames = header["NumFrames"] * self.buffer

        data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}
        empty = header["TrueImageSize"] - (
            (header["ImageWidth"] * header["ImageHeight"] * 2) + 8
        )
        dtype = [
            (
                "Array",
                data_types[int(header["ImageBitDepth"])],
                (
                    int(self.buffer),
                    int(header["ImageHeight"] / self.buffer),
                    int(header["ImageWidth"]),
                ),
            ),
            ("sec", "<u4"),
            ("ms", "<u2"),
            ("mis", "<u2"),
            ("empty", bytes, empty),
        ]

        signal_shape = (
            int(header["ImageHeight"] * 2 / self.buffer),
            int(header["ImageWidth"]),
        )

        if navigation_shape is not None and np.product(navigation_shape) > num_frames:
            _logger.warning(
                "The number of frames and the navigation shape are not "
                "equal. Adding frames to the end of the dataset. "
            )
            buffer_frames = int(
                np.ceil(np.divide(np.product(navigation_shape), self.buffer))
            )
            t = np.memmap(
                self.top, offset=8192, dtype=dtype, shape=buffer_frames, mode="r+"
            )
            t.flush()
            b = np.memmap(
                self.bottom, offset=8192, dtype=dtype, shape=buffer_frames, mode="r+"
            )
            b.flush()
        else:
            buffer_frames = header["NumFrames"]
        if navigation_shape is not None:
            shape = navigation_shape + signal_shape

        else:
            buffer_frames = header["NumFrames"]
            shape = (buffer_frames * self.buffer,) + signal_shape

        if distributed:
            data = read_stitch_binary_distributed(
                top=self.top,
                bottom=self.bottom,
                buffer_size=self.buffer,
                shape=shape,
                offset=8192,
                dtypes=dtype,
                total_buffer_frames=buffer_frames,
                chunks=chunks,
                dark=dark_img,
                gain=gain_img,
            )
        else:
            data, time = read_stitch_binary(top=self.top,
                                            bottom=self.bottom,
                                            dtypes=dtype,
                                            offset=8192,
                                            total_buffer_frames=buffer_frames,
                                            navigation_shape=navigation_shape,
                                            lazy=lazy)
            if dark_img is not None:
                data = data-dark_img
            if gain_img is not None:
                data = data*gain_img

        self._create_axes(
            header=header, nav_shape=navigation_shape, prebuffer=self.buffer
        )
        return {
            "data": data,
            "metadata": self.metadata,
            "axes": self.axes,
            "original_metadata": self.original_metadata,
        }


"""
Functions for reading split binary files. Generic binary distributed readers are
located in rosettasciio.utils.tools
"""


def read_split_seq(
    top,
    bottom,
    ImageWidth,
    ImageHeight,
    ImageBitDepth,
    TrueImageSize,
    SegmentPreBuffer=None,
    total_frames=None,
    navigation_shape=None,
    lazy=False,
):
    """Read a split top/bottom seq file.
    Parameters
    ----------
    top: str
        The filename of the top of the frame
    bottom: str
        The filename of the bottom of the frame
    ImageWidth:int
        The width of one of the frames
    ImageHeight: int
        The height of one of the frame buffers
    ImageBitDepth: int
        The bit depth of the image. This should be 16 in most cases
    TrueImageSize: int
        The size of each frame buffersin bytes.  This includes the time stamp and
        empty bytes after the frame
    SegmentPreBuffer: int
        The size of the segment pre buffer.  This should be an int based on the
         size of each of the frames and the FPS.
    navigation_shape: tuple
        The navigation shape of the data.
    """
    data_types = {8: np.uint8, 16: np.uint16, 32: np.uint32}
    empty = TrueImageSize - ((ImageWidth * ImageHeight * 2) + 8)
    if SegmentPreBuffer is None:
        _logger.warning(
            msg="No XML File given. Guessing Segment PreBuffer "
            "This is may not be correct..."
        )
        # This might be better to guess based on the FPS. Much safer just to use the XML.
        if ImageWidth == 512:
            SegmentPreBuffer = 16
        elif ImageWidth == 256:
            SegmentPreBuffer = 64
        else:
            SegmentPreBuffer = 4
    dtype = [
        (
            "Array",
            data_types[int(ImageBitDepth)],
            (
                int(SegmentPreBuffer),
                int(ImageHeight / SegmentPreBuffer),
                int(ImageWidth),
            ),
        ),
        ("sec", "<u4"),
        ("ms", "<u2"),
        ("mis", "<u2"),
        ("empty", bytes, empty),
    ]
    if navigation_shape is not None and navigation_shape != ():
        # need to read out extra buffered frames
        total_buffer_frames = int(
            np.ceil(np.divide(np.product(navigation_shape), SegmentPreBuffer))
        )
        # add in extra for adding frames.
    else:
        total_buffer_frames = total_frames

    data, time = read_stitch_binary(
        top,
        bottom,
        dtypes=dtype,
        total_buffer_frames=int(total_buffer_frames),
        offset=8192,
        navigation_shape=navigation_shape,
        lazy=lazy,
    )
    return data, time


def read_stitch_binary_distributed(
    top,
    bottom,
    shape,
    dtypes,
    offset,
    total_buffer_frames=None,
    buffer_size=None,
    chunks=None,
    dark=None,
    gain=None,
):

    indexes = get_chunk_index(
        shape=shape,
        signal_axes=(-1, -2),
        chunks=chunks,
        block_size_limit=None,
        dtype=np.float32,
    )
    data = da.map_blocks(
        slic_stitch_binary,
        indexes,
        top=top,
        bottom=bottom,
        dtypes=dtypes,
        offset=offset,
        total_buffer_frames=total_buffer_frames,
        buffer_size=buffer_size,
        gain=gain,
        dark=dark,
        dtype=np.float32,
        new_axis=(len(indexes.shape), len(indexes.shape)+1),
        chunks=indexes.chunks + (shape[-2], shape[-1]),
    )
    return data


def slic_stitch_binary(
    indexes,
    top,
    bottom,
    dtypes,
    offset=8192,
    total_buffer_frames=None,
    buffer_size=None,
    gain=None,
    dark=None,
):
    top_mapped = np.memmap(
        top, offset=offset, dtype=dtypes, shape=total_buffer_frames, mode="r"
    )["Array"]
    bottom_mapped = np.memmap(
        bottom, offset=offset, dtype=dtypes, shape=total_buffer_frames, mode="r",
    )["Array"]
    if buffer_size != None:
        indexes = np.divmod(indexes, buffer_size)

    bottom = bottom_mapped[indexes]
    top = np.flip(top_mapped[indexes], axis=-2)
    chunk = np.concatenate([top, bottom], axis=-2)
    if dark is not None:
        chunk = chunk - dark
    if gain is not None:
        chunk = chunk * gain
    return chunk

def slic_binary(
    indexes,
    file,
    navigation_shape,
    dtypes,
    offset=8192,
    gain=None,
    dark=None,
):
    mapped = np.memmap(file,
                       offset=offset,
                       dtype=dtypes,
                       shape=navigation_shape,
                       mode="r")["Array"]
    mapped = mapped[indexes]
    if dark is not None:
        mapped = mapped - dark
    if gain is not None:
        mapped = mapped * gain
    return mapped


def read_stitch_binary(
    top,
    bottom,
    dtypes,
    offset,
    total_buffer_frames=None,
    navigation_shape=None,
    lazy=False,
):
    """Read and stitch the top and bottom files
    Parameters
    ----------
    top: str
        The filename of the top of the frame
    bottom: str
        The filename of the bottom of the frame
    total_buffer_frames: int
        The size of the segment pre buffer.  This should be an int based on the
         size of each of the frames and the FPS.
    navigation_shape: tuple
        The navigation shape of the data.
    lazy: bool
        If the data should be cast to a dask array.
    """
    keys = [d[0] for d in dtypes]
    top_mapped = np.memmap(
        top, offset=offset, dtype=dtypes, shape=total_buffer_frames, mode="r"
    )
    bottom_mapped = np.memmap(
        bottom, offset=offset, dtype=dtypes, shape=total_buffer_frames, mode="r",
    )

    if lazy:
        top_mapped = da.from_array(top_mapped)
        bottom_mapped = da.from_array(bottom_mapped)

    array = np.concatenate(
        [
            np.flip(
                top_mapped["Array"].reshape(-1, *top_mapped["Array"].shape[2:]), axis=1
            ),
            bottom_mapped["Array"].reshape(-1, *bottom_mapped["Array"].shape[2:]),
        ],
        1,
    )
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
        _logger.warning(
            "No Reference image: "
            + file_name
            + " found.  The dark/gain references should be in the same directory "
            "as the image and have the form xxx.seq.dark.mrc or xxx.seq.gain.mrc"
        )
        return None
