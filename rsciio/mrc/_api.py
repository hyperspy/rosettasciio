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

# The details of the format were taken from
# https://www.biochem.mpg.de/doc_tom/TOM_Release_2008/IOfun/tom_mrcread.html
# and https://ami.scripps.edu/software/mrctools/mrc_specification.php

import os
import logging

import numpy as np

from rsciio._docstrings import (
    ENDIANESS_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    MMAP_DOC,
    NAVIGATION_SHAPE,
    RETURNS_DOC,
)
from rsciio.utils.tools import sarray2dict


_logger = logging.getLogger(__name__)


def get_std_dtype_list(endianess="<"):
    end = endianess
    dtype_list = [
        ("NX", end + "u4"),
        ("NY", end + "u4"),
        ("NZ", end + "u4"),
        ("MODE", end + "u4"),
        ("NXSTART", end + "u4"),
        ("NYSTART", end + "u4"),
        ("NZSTART", end + "u4"),
        ("MX", end + "u4"),
        ("MY", end + "u4"),
        ("MZ", end + "u4"),
        ("Xlen", end + "f4"),
        ("Ylen", end + "f4"),
        ("Zlen", end + "f4"),
        ("ALPHA", end + "f4"),
        ("BETA", end + "f4"),
        ("GAMMA", end + "f4"),
        ("MAPC", end + "u4"),
        ("MAPR", end + "u4"),
        ("MAPS", end + "u4"),
        ("AMIN", end + "f4"),
        ("AMAX", end + "f4"),
        ("AMEAN", end + "f4"),
        ("ISPG", end + "u2"),
        ("NSYMBT", end + "u2"),
        ("NEXT", end + "u4"),
        ("CREATID", end + "u2"),
        ("EXTRA", (np.void, 30)),
        ("NINT", end + "u2"),
        ("NREAL", end + "u2"),
        ("EXTRA2", (np.void, 28)),
        ("IDTYPE", end + "u2"),
        ("LENS", end + "u2"),
        ("ND1", end + "u2"),
        ("ND2", end + "u2"),
        ("VD1", end + "u2"),
        ("VD2", end + "u2"),
        ("TILTANGLES", (np.float32, 6)),
        ("XORIGIN", end + "f4"),
        ("YORIGIN", end + "f4"),
        ("ZORIGIN", end + "f4"),
        ("CMAP", (bytes, 4)),
        ("STAMP", (bytes, 4)),
        ("RMS", end + "f4"),
        ("NLABL", end + "u4"),
        ("LABELS", (bytes, 800)),
    ]

    return dtype_list


def get_fei_dtype_list(endianess="<"):
    end = endianess
    dtype_list = [
        ("a_tilt", end + "f4"),  # Alpha tilt (deg)
        ("b_tilt", end + "f4"),  # Beta tilt (deg)
        # Stage x position (Unit=m. But if value>1, unit=???m)
        ("x_stage", end + "f4"),
        # Stage y position (Unit=m. But if value>1, unit=???m)
        ("y_stage", end + "f4"),
        # Stage z position (Unit=m. But if value>1, unit=???m)
        ("z_stage", end + "f4"),
        # Signal2D shift x (Unit=m. But if value>1, unit=???m)
        ("x_shift", end + "f4"),
        # Signal2D shift y (Unit=m. But if value>1, unit=???m)
        ("y_shift", end + "f4"),
        ("defocus", end + "f4"),  # Defocus Unit=m. But if value>1, unit=???m)
        ("exp_time", end + "f4"),  # Exposure time (s)
        ("mean_int", end + "f4"),  # Mean value of image
        ("tilt_axis", end + "f4"),  # Tilt axis (deg)
        ("pixel_size", end + "f4"),  # Pixel size of image (m)
        ("magnification", end + "f4"),  # Magnification used
        # Not used (filling up to 128 bytes)
        ("empty", (np.void, 128 - 13 * 4)),
    ]
    return dtype_list


def get_data_type(mode):
    mode_to_dtype = {
        0: np.int8,
        1: np.int16,
        2: np.float32,
        4: np.complex64,
        6: np.uint16,
        12: np.float16,
    }

    mode = int(mode)
    if mode in mode_to_dtype:
        return np.dtype(mode_to_dtype[mode])
    else:
        raise ValueError(f"Unrecognised mode '{mode}'.")


def file_reader(
    filename, lazy=False, mmap_mode=None, endianess="<", navigation_shape=None
):
    """
    File reader for the MRC format for tomographic data.

    Parameters
    ----------
    %s
    %s
    %s
    %s
    %s

    %s
    """

    metadata = {}
    f = open(filename, "rb")
    std_header = np.fromfile(f, dtype=get_std_dtype_list(endianess), count=1)
    fei_header = None
    if std_header["NEXT"] / 1024 == 128:
        _logger.info(f"{filename} seems to contain an extended FEI header")
        fei_header = np.fromfile(f, dtype=get_fei_dtype_list(endianess), count=1024)
    if f.tell() == 1024 + std_header["NEXT"]:
        _logger.debug("The FEI header was correctly loaded")
    else:
        f.seek(1024 + std_header["NEXT"][0])
        fei_header = None
    NX, NY, NZ = std_header["NX"], std_header["NY"], std_header["NZ"]
    if mmap_mode is None:
        mmap_mode = "r" if lazy else "c"
    shape = (NX[0], NY[0], NZ[0])
    if navigation_shape is not None:
        shape = shape[:2] + navigation_shape
    data = (
        np.memmap(
            f,
            mode=mmap_mode,
            offset=f.tell(),
            dtype=get_data_type(std_header["MODE"]),
        )
        .reshape(shape, order="F")
        .squeeze()
        .T
    )

    original_metadata = {"std_header": sarray2dict(std_header)}
    # Convert bytes to unicode
    for key in ["CMAP", "STAMP", "LABELS"]:
        original_metadata["std_header"][key] = original_metadata["std_header"][
            key
        ].decode()
    if fei_header is not None:
        fei_dict = sarray2dict(
            fei_header,
        )
        del fei_dict["empty"]
        original_metadata["fei_header"] = fei_dict

    if fei_header is None:
        # The scale is in Angstroms, we convert it to nm
        scales = [
            float(std_header["Zlen"] / std_header["MZ"]) / 10
            if float(std_header["Zlen"]) != 0 and float(std_header["MZ"]) != 0
            else 1,
            float(std_header["Ylen"] / std_header["MY"]) / 10
            if float(std_header["MY"]) != 0
            else 1,
            float(std_header["Xlen"] / std_header["MX"]) / 10
            if float(std_header["MX"]) != 0
            else 1,
        ]
        offsets = [
            float(std_header["ZORIGIN"]) / 10,
            float(std_header["YORIGIN"]) / 10,
            float(std_header["XORIGIN"]) / 10,
        ]

    else:
        # FEI does not use the standard header to store the scale
        # It does store the spatial scale in pixel_size, one per angle in
        # meters
        scales = [
            1,
        ] + [
            fei_header["pixel_size"][0] * 10**9,
        ] * 2
        offsets = [
            0,
        ] * 3

    units = [None, "nm", "nm"]
    names = ["z", "y", "x"]
    navigate = [True, False, False]
    nav_axis_to_add = 0
    if navigation_shape is not None:
        nav_axis_to_add = len(navigation_shape) - 1
        for i in range(nav_axis_to_add):
            print(i)
            units.insert(0, None)
            names.insert(0, "")
            navigate.insert(0, True)
            scales.insert(0, 1)
            offsets.insert(0, 0)

    metadata = {
        "General": {"original_filename": os.path.split(filename)[1]},
        "Signal": {"signal_type": ""},
    }
    # create the axis objects for each axis
    dim = len(data.shape)
    axes = [
        {
            "size": data.shape[i],
            "index_in_array": i,
            "name": names[i + nav_axis_to_add + 3 - dim],
            "scale": scales[i + nav_axis_to_add + 3 - dim],
            "offset": offsets[i + nav_axis_to_add + 3 - dim],
            "units": units[i + nav_axis_to_add + 3 - dim],
            "navigate": navigate[i + nav_axis_to_add + 3 - dim],
        }
        for i in range(dim)
    ]

    dictionary = {
        "data": data,
        "axes": axes,
        "metadata": metadata,
        "original_metadata": original_metadata,
        "mapping": mapping,
    }

    return [
        dictionary,
    ]


mapping = {
    "fei_header.a_tilt": ("Acquisition_instrument.TEM.Stage.tilt_alpha", None),
    "fei_header.b_tilt": ("Acquisition_instrument.TEM.Stage.tilt_beta", None),
    "fei_header.x_stage": ("Acquisition_instrument.TEM.Stage.x", None),
    "fei_header.y_stage": ("Acquisition_instrument.TEM.Stage.y", None),
    "fei_header.z_stage": ("Acquisition_instrument.TEM.Stage.z", None),
    "fei_header.exp_time": (
        "Acquisition_instrument.TEM.Detector.Camera.exposure",
        None,
    ),
    "fei_header.magnification": ("Acquisition_instrument.TEM.magnification", None),
}


file_reader.__doc__ %= (
    FILENAME_DOC,
    LAZY_DOC,
    MMAP_DOC,
    ENDIANESS_DOC,
    NAVIGATION_SHAPE,
    RETURNS_DOC,
)
