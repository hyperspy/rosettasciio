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

import os
import re
import logging
import warnings
from dateutil.parser import parse as dtparse
from datetime import datetime, timezone

import numpy as np
import dask.array as da
import dask
from dask.diagnostics import ProgressBar
import pint
from numba import njit

from rsciio.utils.tools import DTBox, sarray2dict
from rsciio.utils.tools import dummy_context_manager
from rsciio.utils.tools import _UREG


_logger = logging.getLogger(__name__)


TVIPS_RECORDER_GENERAL_HEADER = [
    ("size", "u4"),  # likely the size of generalheader in bytes
    ("version", "u4"),  # 1 or 2
    ("dimx", "u4"),  # image size width
    ("dimy", "u4"),  # image size height
    ("bitsperpixel", "u4"),  # 8 or 16
    ("offsetx", "u4"),  # generally 0
    ("offsety", "u4"),
    ("binx", "u4"),  # camera binning
    ("biny", "u4"),
    ("pixelsize", "u4"),  # physical pixel size in nm
    ("ht", "u4"),  # high tension, voltage
    ("magtotal", "u4"),  # magnification/camera length
    ("frameheaderbytes", "u4"),  # number of bytes per frame header
    ("dummy", "S204"),  # placeholder contains TVIPS TVIPS TVIPS...
]


TVIPS_RECORDER_FRAME_HEADER = [
    ("num", "u4"),  # tends to cycle
    ("timestamp", "u4"),  # seconds since 1.1.1970
    ("ms", "u4"),  # additional milliseconds to the timestamp
    ("LUTidx", "u4"),  # related to color, useless
    ("fcurrent", "f4"),  # usually 0 for all frames
    ("mag", "u4"),  # same for all frames, can be different from magtotal
    ("mode", "u4"),  # 1 -> image, 2 -> diffraction
    ("stagex", "f4"),
    ("stagey", "f4"),
    ("stagez", "f4"),
    ("stagea", "f4"),
    ("stageb", "f4"),
    ("rotidx", "u4"),  # encodes information about the scan
    ("temperature", "f4"),  # cycles between 0.0 and 9.0 with step 1.0
    ("objective", "f4"),  # kind of randomly between 0.0 and 1.0
    # TODO: sometimes scan positions may be present in header, may require more reverse engineering
]


def _guess_image_mode(signal):
    """
    Guess whether the dataset contains images (1) or diffraction patterns (2).
    If no decent guess can be made, None is returned.
    """
    # original pixel scale
    scale = signal["axes"][-2]["scale"]
    unit = signal["axes"][-2]["units"]
    mode = None
    try:
        pixel_size = scale * _UREG(unit)
    except (AttributeError, pint.UndefinedUnitError):
        pass
    else:
        if pixel_size.is_compatible_with("m"):
            mode = 1
        elif pixel_size.is_compatible_with("1/m"):
            mode = 2
        else:
            pass
    return mode


def _get_main_header_from_signal(signal, version=2, frame_header_extra_bytes=0):
    dt = np.dtype(TVIPS_RECORDER_GENERAL_HEADER)
    header = np.zeros((1,), dtype=dt)
    header["size"] = dt.itemsize
    header["version"] = version
    # original pixel scale
    mode = _guess_image_mode(signal)
    axes = signal["axes"]
    scale = axes[0]["scale"]
    offsetx = axes[-1]["offset"]
    offsety = axes[-2]["offset"]
    unit = axes[-2]["units"]
    if mode == 1:
        to_unit = "nm"
    elif mode == 2:
        to_unit = "1/nm"
    else:
        to_unit = ""
    if to_unit:
        scale = round((scale * _UREG(unit)).to(to_unit).magnitude)
        offsetx = round((offsetx * _UREG(unit)).to(to_unit).magnitude)
        offsety = round((offsety * _UREG(unit)).to(to_unit).magnitude)
    else:
        warnings.warn(
            "Image scale units could not be converted, " "saving axes scales as is.",
            UserWarning,
        )
    metadata = DTBox(signal["metadata"], box_dots=True)
    header["dimx"] = axes[-1]["size"]
    header["dimy"] = axes[-2]["size"]
    header["offsetx"] = offsetx
    header["offsety"] = offsety
    header["pixelsize"] = scale
    header["bitsperpixel"] = signal["data"].dtype.itemsize * 8
    header["binx"] = 1
    header["biny"] = 1
    dtf = np.dtype(TVIPS_RECORDER_FRAME_HEADER)
    header["frameheaderbytes"] = dtf.itemsize + frame_header_extra_bytes
    header["dummy"] = "HYPERSPY " * 22 + "HYPERS"
    header["ht"] = metadata.get("Acquisition_instrument.TEM.beam_energy", 0)
    cl = metadata.get("Acquisition_instrument.TEM.camera_length", 0)
    mag = metadata.get("Acquisition_instrument.TEM.magnification", 0)
    if cl != 0 and mag != 0:
        header["magtotal"] = 0
    elif cl != 0 and mode == 2:
        header["magtotal"] = cl
    elif mag != 0 and mode == 1:
        header["magtotal"] = mag
    else:
        header["magtotal"] = 0
    return header


def _get_frame_record_dtype_from_signal(signal, extra_bytes=0):
    fhdtype = TVIPS_RECORDER_FRAME_HEADER.copy()
    if extra_bytes > 0:
        fhdtype.append(("extra", bytes, extra_bytes))
    dimx = signal["axes"][-1]["size"]
    dimy = signal["axes"][-2]["size"]
    fhdtype.append(("data", signal["data"].dtype, (dimy, dimx)))
    dt = np.dtype(fhdtype)
    return dt


def _is_valid_first_tvips_file(filename):
    """Check if the provided first tvips file path is valid"""
    filpattern = re.compile(r".+\_([0-9]{3})\.(.*)")
    match = re.match(filpattern, filename)
    if match is not None:
        num, ext = match.groups()
        if ext.lower() != "tvips":
            raise ValueError(f"Invalid tvips file: extension {ext}, must be tvips")
        if int(num) != 0:
            raise ValueError("Can only read video sequences starting with part 000")
        return True
    else:
        raise ValueError("Could not recognize as a valid tvips file")


def _find_auto_scan_start_stop(rotidxs):
    """Find the start and stop index in a rotator index array"""
    diff = rotidxs[1:] - rotidxs[:-1]
    indx = np.where(diff > 0)[0]
    if indx.size == 0:
        return None, None
    else:
        startx = indx[0]
        if rotidxs[startx] == 0:
            startx += 1
        return startx, indx[-1] + 1


@njit
def _guess_scan_index_grid(rotidx, start, stop):
    indxs = np.zeros(rotidx[stop], dtype=np.int64)
    rotidx = rotidx[start : stop + 1]
    inv = 0  # index of the value we fill in
    for i in range(rotidx.shape[0]):
        if rotidx[inv] != rotidx[i]:
            # when we encounter a new value, we fill in indices
            pos_start = rotidx[inv] - 1
            pos_end = rotidx[i] - 1
            stack = np.arange(inv, i)[: pos_end - pos_start]
            indxs[pos_start:pos_end] = stack[-1]
            indxs[pos_start : pos_start + stack.shape[0]] = stack
            inv = i
    # the last value we fill in at the end
    indxs[rotidx[inv] - 1 :] = inv
    return indxs + start


def file_reader(
    filename,
    lazy=True,
    scan_shape=None,
    scan_start_frame=0,
    winding_scan_axis=None,
    hysteresis=0,
    rechunking="auto",
    **kwds,
):
    """
    TVIPS stream file reader for in-situ and 4D STEM data

    Parameters
    ----------
    lazy : bool, default=True
        Whether to open the file lazily or not.
    scan_shape : str or tuple of int, optional
        By default the data is loaded as an image stack (1 navigation axis).
        A tuple of integers can be provided to indicate the shape of the
        navigation axes. For example, `(3, 4)` will have 3 scan points in the y
        direction and 4 in the x direction.
        If it concerns a 4D-STEM dataset, the (..., ``scan_y``, ``scan_x)``
        dimension can be provided. ``"auto"`` can also be selected, in which case
        the ``rotidx`` information in the frame headers will be used to try to
        reconstruct the scan. Additional navigation axes must be prepended. Since
        this only works for square scan grids and is prone to failure, this option
        is not recommended.
    scan_start_frame : int, optional
        Index of the first frame of the dataset to consider. Mainly relevant for
        4D-STEM datasets.If ``scan_shape = "auto"`` this is ignored.
    winding_scan_axis : str, optional
        If the acquisition software collected data without beam flyback but with
        a winding "snake" scan, then every second scan row or column needs to be
        reversed to make sense of the data. This can be indicated with values
        ``"x"`` or ``"y"``, depending on whether winding happened along the
        primary or secondary axis. By default, flyback scan without winding
        is assumed with ``x`` the fast scan and ``y`` the slow scan direction.
    hysteresis: int, optional
        Only applicable if ``winding_scan_axis`` is not ``None``, as it is likely
        there is an overshoot of a few pixels (2-5) every second scan row. This
        parameter allows shifts every second row by the indicated number of scan
        points to align even and odd scan rows. Default is 0, no hysteresis.
    rechunking: bool, str, or Dict, Default="auto"
        Only relevant when using lazy loading. If set to False each tvips file is
        a single chunk. For a better experience, with the default setting of 
        ``"auto"`` rechunking is performed such that the navigation axes
        are optimally chunked and the signal axes are not chunked.
        If set to anything else, e.g. a dictionary, the value will be passed to
        the chunks argument in dask.array.rechunk.

    Returns
    -------
    dict
        Dictionary containing the elements ``data``, ``axes``, ``metadata``
        and ``original_metadata``. See `API guide
        <https://hyperspy.org/rosettasciio/api.html>`_.
    """
    # check whether we start at the first tvips file
    _is_valid_first_tvips_file(filename)

    # get all other filenames in case they exist
    other_files = []
    basename = filename[:-9]  # last bit: 000.tvips
    file_index = 1
    _, ext = os.path.splitext(filename)
    while True:
        fn = basename + "{:03d}{}".format(file_index, ext)
        if not os.path.exists(fn):
            break
        other_files.append(fn)
        file_index += 1

    # parse the header from the first file
    with open(filename, "rb") as f:
        f.seek(0)
        # read the main header in file 0
        header = np.fromfile(f, dtype=TVIPS_RECORDER_GENERAL_HEADER, count=1)
        dtype = np.dtype(f"u{header['bitsperpixel'][0]//8}")
        dimx = header["dimx"][0]
        dimy = header["dimy"][0]
        # the size of the frame header varies with version
        if header["version"][0] == 1:
            increment = 12  # pragma: no cover
        elif header["version"][0] == 2:
            increment = header["frameheaderbytes"][0]
        else:
            raise NotImplementedError(
                f"This version {header.version} is not yet supported"
                " in HyperSpy. Please report this as an issue at "
                "https://github.com/hyperspy/hyperspy/issues."
            )  # pragma: no cover
        frame_header_dt = np.dtype(TVIPS_RECORDER_FRAME_HEADER)
        # the record must consume less bytes than reported in the main header
        if increment < frame_header_dt.itemsize:
            raise ValueError(
                "The frame header record consumes more bytes than stated in the main header"
            )  # pragma: no cover
        # save metadata
        original_metadata = {"tvips_header": sarray2dict(header)}
        # create custom dtype for memmap padding the frame_header as required
        extra_bytes = increment - frame_header_dt.itemsize
        record_dtype = TVIPS_RECORDER_FRAME_HEADER.copy()
        if extra_bytes > 0:
            record_dtype.append(("extra", bytes, extra_bytes))
        record_dtype.append(("data", dtype, (dimy, dimx)))

    # memmap the data
    records_000 = np.memmap(
        filename, mode="r", dtype=record_dtype, offset=header["size"][0]
    )
    # the array data
    all_array_data = [records_000["data"]]
    # in case we also want the frame header metadata later
    metadata_keys = np.array(TVIPS_RECORDER_FRAME_HEADER)[:, 0]
    metadata_000 = records_000[metadata_keys]
    all_metadata = [metadata_000]
    # also load data from other files
    for i in other_files:
        # no offset on the other files
        records = np.memmap(i, mode="r", dtype=record_dtype)
        all_metadata.append(records[metadata_keys])
        all_array_data.append(records["data"])
    if lazy:
        data_stack = da.concatenate(all_array_data, axis=0)
    else:
        data_stack = np.concatenate(all_array_data, axis=0)

    # extracting some units/scales/offsets of the DP's or images
    mode = all_metadata[0]["mode"][0]
    DPU = "1/nm" if mode == 2 else "nm"
    SDP = header["pixelsize"][0]
    offsetx = header["offsetx"][0]
    offsety = header["offsety"][0]
    # modify the data if there is scan information
    # we construct a 2D array of indices to slice the data_stack
    if scan_shape is not None:
        # try to deduce start and stop of the scan based on rotator index
        if scan_shape == "auto":
            record_idxs = np.concatenate([i["rotidx"] for i in all_metadata])
            scan_start_frame, scan_stop_frame = _find_auto_scan_start_stop(record_idxs)
            if scan_start_frame is None or scan_stop_frame is None:
                raise ValueError(
                    "Scan start and stop information could not be automatically "
                    "determined. Please supply a scan_shape and scan_start_frame."
                )  # pragma: no cover
            total_scan_frames = record_idxs[scan_stop_frame]  # last rotator
            scan_dim = int(np.sqrt(total_scan_frames))
            if not np.allclose(scan_dim, np.sqrt(total_scan_frames)):
                raise ValueError(
                    "Scan was not square, please supply a scan_shape and start_frame."
                )
            scan_shape = (scan_dim, scan_dim)
            # there may be discontinuities which must be filled up
            indices = _guess_scan_index_grid(
                record_idxs, scan_start_frame, scan_stop_frame
            ).reshape(scan_shape)
        # scan shape and start are provided
        else:
            total_scan_frames = np.prod(scan_shape)
            max_frame_index = np.prod(data_stack.shape[:-2])
            final_frame = scan_start_frame + total_scan_frames
            if final_frame > max_frame_index:
                raise ValueError(
                    f"Shape {scan_shape} requires image index {final_frame-1} "
                    f"which is out of bounds. Final frame index: {max_frame_index-1}."
                )
            indices = np.arange(scan_start_frame, final_frame).reshape(scan_shape)

        # with winding scan, every second column or row must be inverted
        # due to hysteresis there is also a predictable offset
        if winding_scan_axis is not None:
            if winding_scan_axis in ["x", 0]:
                indices[..., ::2, :] = indices[..., ::2, :][..., :, ::-1]
                indices[..., ::2, :] = np.roll(
                    indices[..., ::2, :], hysteresis, axis=-1
                )
            elif winding_scan_axis in ["y", 1]:
                indices[..., :, ::2] = indices[..., :, ::2][..., ::-1, :]
                indices[..., :, ::2] = np.roll(
                    indices[..., :, ::2], hysteresis, axis=-2
                )
            else:
                raise ValueError("Invalid winding scan axis")

        with dask.config.set(**{"array.slicing.split_large_chunks": True}):
            data_stack = data_stack[indices.ravel()]
        data_stack = data_stack.reshape(*indices.shape, dimy, dimx)
        units = (indices.ndim - 2) * [""] + ["nm", "nm", DPU, DPU]
        names = (indices.ndim - 2) * [""] + ["y", "x", "dy", "dx"]
        # no scale information stored in the scan!
        scales = (indices.ndim - 2) * [1] + [1, 1, SDP, SDP]
        offsets = (indices.ndim - 2) * [0] + [0, 0, offsety, offsetx]
        # Create the axis objects for each axis
        dim = data_stack.ndim
        axes = [
            {
                "size": data_stack.shape[i],
                "index_in_array": i,
                "name": names[i],
                "scale": scales[i],
                "offset": offsets[i],
                "units": units[i],
                "navigate": True if i < len(scan_shape) else False,
            }
            for i in range(dim)
        ]
    else:
        # we load as a regular image stack
        units = ["s", DPU, DPU]
        names = ["time", "dy", "dx"]
        times = np.concatenate([i["timestamp"] + i["ms"] / 1000 for i in all_metadata])
        timescale = 1 if times.shape[0] <= 0 else times[1] - times[0]
        scales = [timescale, SDP, SDP]
        offsets = [times[0], offsety, offsetx]
        # Create the axis objects for each axis
        dim = data_stack.ndim
        axes = [
            {
                "size": data_stack.shape[i],
                "index_in_array": i,
                "name": names[i],
                "scale": scales[i],
                "offset": offsets[i],
                "units": units[i],
                "navigate": True if i == 0 else False,
            }
            for i in range(dim)
        ]
    dtobj = datetime.fromtimestamp(all_metadata[0]["timestamp"][0])
    date = dtobj.date().isoformat()
    time = dtobj.time().isoformat()
    current = all_metadata[0]["fcurrent"][0]
    stagex = all_metadata[0]["stagex"][0]
    stagey = all_metadata[0]["stagey"][0]
    stagez = all_metadata[0]["stagez"][0]
    stagealpha = all_metadata[0]["stagea"][0]
    stagebeta = all_metadata[0]["stageb"][0]
    # mag = all_metadata[0]["mag"][0]  # TODO it is unclear what this value is
    focus = all_metadata[0]["objective"][0]
    metadata = {
        "General": {
            "original_filename": os.path.split(filename)[1],
            "date": date,
            "time": time,
            "time_zone": "UTC",
        },
        "Acquisition_instrument": {
            "TEM": {
                "magnification": header["magtotal"][0],
                "beam_energy": header["ht"][0],
                "beam_current": current,
                "defocus": focus,
                "Stage": {
                    "tilt_alpha": stagealpha,
                    "tilt_beta": stagebeta,
                    "x": stagex,
                    "y": stagey,
                    "z": stagez,
                },
            },
        },
    }

    if lazy:
        if rechunking:
            if rechunking == "auto":
                navdims = data_stack.ndim - 2
                chunks = {ax_index: "auto" for ax_index in range(navdims)}
                chunks[navdims] = None
                chunks[navdims + 1] = None
            else:
                chunks = rechunking
            data_stack = data_stack.rechunk(chunks)

    if mode == 2:
        metadata["Signal"] = {"signal_type": "diffraction"}
    # TODO at the moment hyperspy doesn't have a signal type for mode==1, imaging

    dictionary = {
        "data": data_stack,
        "axes": axes,
        "metadata": metadata,
        "original_metadata": original_metadata,
        "mapping": {},
    }

    return [
        dictionary,
    ]


def file_writer(filename, signal, **kwds):
    """
    Write signal to TVIPS file.

    Parameters
    ----------
    filename: str
        Filename of the file to write to. If not supplied, a _000 suffix will
        automatically be appended before the extension.
    signal: dict
        Dictionary containing the signal object as defined in the `API guide
        <https://hyperspy.org/rosettasciio/api.html>`_.
    max_file_size: int, optional
        Approximate maximum size of individual files in bytes. In this way a
        dataset can be split into multiple files. A file needs to be at least the
        size of the main header in the first file plus one frame and its frame
        header. Sequential files are denoted by a suffix _xxx starting from _000.
        By default there is no maximum and the entire dataset is stored in a
        single file.
    version: int, optional
        TVIPS file format version (only version ``1`` or ``2`` supported),
        defaults to version ``2``.
    frame_header_extra_bytes: int, optional
        Number of bytes to pad the frame headers with, defaults to 0
    mode: int, optional
        ``1`` for imaging, ``2`` for diffraction. By default, the mode is
        guessed from signal type and signal units.
    """
    # only signal2d is allowed
    axes = signal["axes"]
    metadata = DTBox(signal["metadata"], box_dots=True)
    signal_dim = len([axis for axis in axes if not axis["navigate"]])
    nav_shape = [axis["size"] for axis in axes if axis["navigate"]]
    num_frames = np.prod(nav_shape) if nav_shape else 0
    if signal_dim != 2:
        raise ValueError("Only Signal2D supported for writing to TVIPS file.")
    fnb, ext = os.path.splitext(filename)
    if fnb.endswith("_000"):
        fnb = fnb[:-4]
    version = kwds.pop("version", 2)
    fheb = kwds.pop("frame_header_extra_bytes", 0)
    main_header = _get_main_header_from_signal(signal, version, fheb)
    # frame header + frame dtype
    record_dtype = _get_frame_record_dtype_from_signal(signal, fheb)
    total_file_size = main_header.itemsize + num_frames * record_dtype.itemsize
    max_file_size = kwds.pop("max_file_size", None)
    if max_file_size is None:
        max_file_size = total_file_size
    minimum_file_size = main_header.itemsize + record_dtype.itemsize
    if max_file_size < minimum_file_size:
        warnings.warn(
            f"The minimum file size for this dataset is {minimum_file_size} bytes"
        )
        max_file_size = minimum_file_size
    # frame metadata
    start_date_str = metadata.get("General.date", "1970-01-01")
    start_time_str = metadata.get("General.time", "00:00:00")
    tz = metadata.get("General.time_zone", "UTC")
    datetime_str = f"{start_date_str} {start_time_str} {tz}"
    time_dt = dtparse(datetime_str)
    time_dt_utc = time_dt.astimezone(timezone.utc)
    # workaround for timestamp not working on Windows, see https://bugs.python.org/issue37527
    BEGIN = datetime(1970, 1, 1, 0).replace(tzinfo=timezone.utc)
    timestamp = (time_dt_utc - BEGIN).total_seconds()
    if num_frames:
        nav_units = signal["axes"][-3]["units"]
        nav_increment = signal["axes"][-3]["scale"]
        try:
            time_increment = (nav_increment * _UREG(nav_units)).to("ms").magnitude
        except (AttributeError, pint.UndefinedUnitError, pint.DimensionalityError):
            time_increment = 1
    # imaging or diffraction
    mode = kwds.pop("mode", None)
    if mode is None:
        mode = _guess_image_mode(signal)
    mode = 2 if mode is None else mode
    stagex = metadata.get("Acquisition_instrument.TEM.Stage.x", 0)
    stagey = metadata.get("Acquisition_instrument.TEM.Stage.y", 0)
    stagez = metadata.get("Acquisition_instrument.TEM.Stage.z", 0)
    stagea = metadata.get("Acquisition_instrument.TEM.tilt_alpha", 0)
    stageb = metadata.get("Acquisition_instrument.TEM.tilt_beta", 0)
    # TODO: is fcurrent actually beam current??
    fcurrent = metadata.get("Acquisition_instrument.TEM.beam_current", 0)
    frames_to_save = num_frames if num_frames else 1
    current_frame = 0
    file_index = 0
    data = signal["data"]
    if num_frames:
        fdata = data.reshape((num_frames, axes[-2]["size"], axes[-1]["size"]))
    while frames_to_save != 0:
        suffix = "_" + (f"{file_index}".zfill(3))
        filename = fnb + suffix + ext
        if file_index == 0:
            with open(filename, "wb") as f:
                main_header.tofile(f)
                file_location = f.tell()
                open_mode = "r+"
        else:
            file_location = 0
            open_mode = "w+"
        frames_saved = (max_file_size - file_location) // record_dtype.itemsize
        # last file can contain fewer images
        if frames_to_save < frames_saved:
            frames_saved = frames_to_save
        file_memmap = np.memmap(
            filename,
            dtype=record_dtype,
            mode=open_mode,
            offset=file_location,
            shape=frames_saved,
        )
        # fill in the metadata
        file_memmap["mode"] = mode
        file_memmap["stagex"] = stagex
        file_memmap["stagey"] = stagey
        file_memmap["stagez"] = stagez
        file_memmap["stagea"] = stagea
        file_memmap["stageb"] = stageb
        file_memmap["fcurrent"] = fcurrent
        rotator = np.arange(current_frame, current_frame + frames_saved)
        milliseconds = rotator * time_increment
        timestamps = (timestamp + milliseconds / 1000).astype(int)
        milliseconds = milliseconds % 1000
        file_memmap["timestamp"] = timestamps
        file_memmap["ms"] = milliseconds
        file_memmap["rotidx"] = rotator + 1
        data = fdata[current_frame : current_frame + frames_saved]
        if signal["attributes"]["_lazy"]:
            try:
                from hyperspy.defaults_parser import preferences

                hs_show_progressbar = preferences.General.show_progressbar
            except:  # pragma: no cover
                hs_show_progressbar = None
            show_progressbar = kwds.get("show_progressbar", hs_show_progressbar)
            cm = ProgressBar if show_progressbar else dummy_context_manager
            with cm():
                data.store(file_memmap["data"])
        else:
            file_memmap["data"] = data
        file_memmap.flush()
        file_index += 1
        frames_to_save -= frames_saved
        current_frame += frames_saved
