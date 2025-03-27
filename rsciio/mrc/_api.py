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

import glob
import logging
import os

import dask.array as da
import numpy as np

from rsciio._docstrings import (
    CHUNKS_READ_DOC,
    DISTRIBUTED_DOC,
    ENDIANESS_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    MMAP_DOC,
    NAVIGATION_SHAPE,
    RETURNS_DOC,
)
from rsciio.utils.distributed import memmap_distributed
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

    mode = int(mode[0])
    if mode in mode_to_dtype:
        return np.dtype(mode_to_dtype[mode])
    else:
        raise ValueError(f"Unrecognised mode '{mode}'.")


def read_de_metadata_file(filename, nav_shape=None):
    """This reads the metadata ".txt" file that is saved alongside a DE .mrc file.

    There are 3 ways in which the files are saved:


    Parameters
    ----------
    filename : str
        The filename of the metadata file.
    nav_shape : tuple
        The shape of the navigation axes.

    Returns
    -------
    all_lines : dict
        A dictionary containing all the metadata.
    shape : tuple
        The shape of the data in real space.
    """
    original_metadata = {}
    with open(filename) as metadata:
        for line in metadata.readlines():
            key, value = line.split("=")
            key = key.strip()
            value = value.strip()
            original_metadata[key] = value

    # -1 -> Not read from TEM Channel 0 -> TEM 1 -> STEM
    in_stem_mode = int(original_metadata.get("Instrument Project TEMorSTEM Mode", -1))
    scanning = original_metadata.get("Scan - Enable", "Disable") == "Enable"
    raster = original_metadata.get("Scan - Type", "Raster") == "Raster"
    if not raster:  # pragma: no cover
        _logger.warning(
            "Non-raster scans are not fully supported yet. Please raise an issue on GitHub"
            " if you need this feature."
        )
    if in_stem_mode == -1:
        in_stem_mode = scanning
    elif in_stem_mode == 0:  # 0 -> TEM Mode
        in_stem_mode = False
    else:
        in_stem_mode = True

    has_camera_length = float(
        original_metadata.get("Instrument Project Camera Length (centimeters)", -1)
    )
    diffracting = (
        has_camera_length != -1 or in_stem_mode
    )  # Force diffracting if in STEM mode

    if in_stem_mode and scanning and raster or nav_shape is not None:
        axes_scales = np.array(
            [
                float(
                    original_metadata.get("Scan - Time (seconds)", 1)
                    + original_metadata.get("Scan - Repeat Delay (seconds)", 1)
                ),
                float(original_metadata.get("Specimen Pixel Size Y (nanometers)", 1)),
                float(original_metadata.get("Specimen Pixel Size X (nanometers)", 1)),
                1,  # Signal Axes below
                float(original_metadata.get("Diffraction Pixel Size Y", 1)),
                float(original_metadata.get("Diffraction Pixel Size X", 1)),
            ]
        )
        axes_scales[axes_scales != -1] = 1
        axes_units = ["sec", "nm", "nm", "times", "nm^-1", "nm^-1"]
        axes_names = ["time", "y", "x", "repeats", "ky", "kx"]
        sizex = int(original_metadata.get("Scan - Size X", 1))
        sizey = int(original_metadata.get("Scan - Size Y", 1))
        if nav_shape is not None and len(nav_shape) == 3:
            time = nav_shape[0]
            sizey = nav_shape[1]
            sizex = nav_shape[2]
        elif nav_shape is not None and len(nav_shape) == 2:
            sizex = nav_shape[1]
            sizey = nav_shape[0]
            time = 1
        elif nav_shape is not None and len(nav_shape) == 1:
            sizex = nav_shape[0]
            sizey = 1
            time = 1
        else:
            time = 1
        sizekx = int(original_metadata.get("Image Size X (pixels)", -1))
        sizeky = int(original_metadata.get("Image Size Y (pixels)", -1))
        navigate = [True, True, True, True, False, False]
        axes_shapes = [time, sizey, sizex, 1, sizeky, sizekx]
        nav_shape = tuple([shape for shape in [time, sizex, sizey, 1] if shape != 1])
    else:
        navigate = [True, False, False]

        nav_shape = None  # read from the .mrc file
        frame_sum = float(original_metadata.get("Autosave Movie Sum Count", 1))
        frame_time = float(original_metadata.get("Frames Per Second", 1))
        sec_per_frame = 1 / (frame_time * frame_sum)
        axes_shapes = [-1, -1, -1]  # get from the .mrc file
        if diffracting:
            axes_scales = np.array(
                [
                    sec_per_frame,
                    original_metadata.get("Diffraction Pixel Size Y", 1),
                    original_metadata.get("Diffraction Pixel Size X", 1),
                ]
            )
            axes_units = ["sec", "nm^-1", "nm^-1"]
            axes_names = ["time", "ky", "kx"]
        else:
            axes_scales = np.array(
                [
                    sec_per_frame,
                    original_metadata.get("Specimen Pixel Size Y (nanometers)", 1),
                    original_metadata.get("Specimen Pixel Size X (nanometers)", 1),
                ]
            )
            axes_units = ["sec", "nm", "nm"]
            axes_names = ["time", "y", "x"]
    axes = []
    ind = 0
    for i, s in enumerate(axes_shapes):
        if s != 1:
            axes.append(
                {
                    "name": axes_names[i],
                    "size": s,  # if -1, get from the .mrc file
                    "units": axes_units[i],
                    "scale": axes_scales[i],
                    "navigate": navigate[i],
                    "index_in_array": ind,
                }
            )
            ind += 1
    electron_gain = float(original_metadata.get("ADUs Per Electron Bin1x", 1))
    magnification = original_metadata.get("Instrument Project Magnification", None)
    camera_model = original_metadata.get("Camera Model", None)
    timestamp = original_metadata.get("Timestamp (seconds since Epoch)", None)
    fps = original_metadata.get("Frames Per Second", None)

    metadata = {
        "Acquisition_instrument": {
            "TEM": {
                "magnification": magnification,
                "detector": camera_model,
                "frames_per_second": fps,
            }
        },
        "Signal": {
            "Noise_properties": {"gain_factor": 1 / electron_gain},
            "quantity": "$e^-$",
        },
        "General": {"timestamp": timestamp},
    }
    return original_metadata, metadata, axes, nav_shape


def file_reader(
    filename,
    lazy=False,
    mmap_mode=None,
    endianess="<",
    navigation_shape=None,
    distributed=False,
    chunks="auto",
    metadata_file="auto",
    virtual_images=None,
    external_images=None,
):
    """
    File reader for the MRC format for tomographic and 4D-STEM data.

    Parameters
    ----------
    %s
    %s
    %s
    %s
    %s
    %s
    %s
    metadata_file : str
        The filename of the metadata file, if "auto" it will try to find the
        metadata file automatically. For DE movies of 4D STEM datasets this
        defines the shape and metadata.
    virtual_images : list
        A list of filenames of virtual images. For DE movies these are automatically loaded.
    external_images : list
        A list of filenames of external images (e.g. external detectors) to be loaded
        alongside the main data. For DE movies these are automatically loaded.

    %s
    """
    if metadata_file == "auto":
        if "movie" in filename:
            try:  # DE movie
                dir_name = os.path.dirname(filename)
                base_name = os.path.basename(filename)
                split = base_name.split("_")
                unique_id = "_".join(split[:2])
                if len(split) > 3:  # File Suffix
                    suffix = "_".join(split[2:-1])
                else:
                    suffix = ""
                metadata = glob.glob(dir_name + "/" + unique_id + suffix + "_info.txt")
                virtual_images = glob.glob(
                    dir_name + "/" + unique_id + suffix + "_[0-4]_*.mrc"
                )
                external_images = glob.glob(
                    dir_name + "/" + unique_id + suffix + "_ext[1-4]_*.mrc"
                )
            except (
                IndexError
            ):  # Not a DE movie or File Naming Convention is not followed
                _logger.warning("Could not find metadata file for DE movie.")
                metadata = []
        else:
            metadata = []
        if len(metadata) == 1:
            metadata_file = metadata[0]
        else:
            metadata_file = None

    if metadata_file is not None:
        # Check if the metadata file exists
        (
            de_metadata,
            metadata,
            metadata_axes,
            _navigation_shape,
        ) = read_de_metadata_file(metadata_file, nav_shape=navigation_shape)
        if navigation_shape is None:
            navigation_shape = _navigation_shape
        original_metadata = {"de_metadata": de_metadata}
    else:
        original_metadata = {}
        metadata = {"General": {}, "Signal": {}}
        metadata_axes = None
    metadata["General"]["original_filename"] = os.path.split(filename)[1]
    metadata["Signal"]["signal_type"] = ""

    if virtual_images is not None and len(virtual_images) > 0:
        imgs = []
        for v in virtual_images:
            imgs.append(file_reader(v)[0]["data"])
        metadata["General"]["virtual_images"] = imgs
        # checking to make sure the navigator is valid
        if navigation_shape is not None and navigation_shape[::-1] == imgs[0].shape:
            metadata["_HyperSpy"] = {}
            metadata["_HyperSpy"]["navigator"] = imgs[0]

    if external_images is not None and len(external_images) > 0:
        imgs = []
        for e in external_images:
            imgs.append(file_reader(e)[0]["data"])
        metadata["General"]["external_detectors"] = imgs

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
    if distributed:
        data = memmap_distributed(
            filename,
            offset=f.tell(),
            shape=shape[::-1],
            dtype=get_data_type(std_header["MODE"]),
            chunks=chunks,
        )
        if not lazy:
            data = data.compute()
    else:
        data = np.memmap(
            f,
            mode=mmap_mode,
            shape=shape[::-1],
            offset=f.tell(),
            dtype=get_data_type(std_header["MODE"]),
        ).squeeze()
        if lazy:
            data = da.from_array(data, chunks=chunks)

    original_metadata["std_header"] = sarray2dict(std_header)

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
            (
                float((std_header["Zlen"] / std_header["MZ"])[0]) / 10
                if float(std_header["Zlen"][0]) != 0 and float(std_header["MZ"][0]) != 0
                else 1
            ),
            (
                float((std_header["Ylen"] / std_header["MY"])[0]) / 10
                if float(std_header["MY"][0]) != 0
                else 1
            ),
            (
                float((std_header["Xlen"] / std_header["MX"])[0]) / 10
                if float(std_header["MX"][0]) != 0
                else 1
            ),
        ]
        offsets = [
            float(std_header["ZORIGIN"][0]) / 10,
            float(std_header["YORIGIN"][0]) / 10,
            float(std_header["XORIGIN"][0]) / 10,
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

    if metadata_axes is None:
        units = [None, "nm", "nm"]
        names = ["z", "y", "x"]
        navigate = [True, False, False]
        nav_axis_to_add = 0
        if navigation_shape is not None:
            nav_axis_to_add = len(navigation_shape) - 1
            for i in range(nav_axis_to_add):
                units.insert(0, None)
                names.insert(0, "")
                navigate.insert(0, True)
                scales.insert(0, 1)
                offsets.insert(0, 0)

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
    else:
        axes = metadata_axes
        for ax, s in zip(axes, shape[::-1]):
            ax["size"] = s  # Update the size of the axes

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
    DISTRIBUTED_DOC,
    CHUNKS_READ_DOC,
    RETURNS_DOC,
)
