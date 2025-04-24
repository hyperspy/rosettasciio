# -*- coding: utf-8 -*-
#
# Copyright 2024 The HyperSpy developers
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with any project and source this library is coupled.
# If not, see <https://www.gnu.org/licenses/#GPL>.

import logging
import os
from pathlib import Path

import h5py
import hdf5plugin
import numpy as np

from rsciio._docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC
from rsciio.utils.tools import get_file_handle

_logger = logging.getLogger(__name__)


def file_reader(
    filename,
    lazy=False,
    scan_width=None,
    binfactor=1,
    dtype=None,
    flatfield=None,
):
    """Read arina 4D-STEM datasets.

    Parameters
    ----------
    scan_width : int, optional
        x dimension of scan. If None, it will assume a square acquisition.
    binfactor : int, default=1
        Diffraction space binning factor for bin-on-load.
    dtype : float, optional
        Specify datatype for load.
    flatfield : numpy.ndarray, optional
        Flatfield for correction factors, converts data to float.
    """
    if lazy:
        raise NotImplementedError("Lazy loading is not supported for arina files")

    filename = Path(filename)
    if not filename.exists():
        raise FileNotFoundError(f"File {filename} not found")

    # Get the base filename without the _master.h5 suffix
    base_filename = str(filename).replace("_master.h5", "")

    # Open the master file
    with h5py.File(filename, "r") as f:
        nimages = 0
        # Count the number of images in all datasets
        for dset in f["entry"]["data"]:
            nimages = nimages + f["entry"]["data"][dset].shape[0]
            height = f["entry"]["data"][dset].shape[1]
            width = f["entry"]["data"][dset].shape[2]
            dtype = f["entry"]["data"][dset].dtype

    width = width // binfactor
    height = height // binfactor

    if scan_width is None:
        scan_width = int(np.sqrt(nimages))

    if nimages % scan_width > 1e-6:
        raise ValueError("scan_width must be integer multiple of x*y size")

    if dtype.type is np.uint32 and flatfield is None:
        _logger.info("Dataset is uint32 but will be converted to uint16")
        dtype = np.dtype(np.uint16)

    if flatfield is not None:
        array_3D = np.empty((nimages, width, height), dtype=np.float32)
        _logger.info("Dataset will be converted to float32 due to flatfield correction")
    elif dtype:
        array_3D = np.empty((nimages, width, height), dtype=dtype)
    else:
        array_3D = np.empty((nimages, width, height), dtype=dtype)

    image_index = 0

    if flatfield is None:
        correction_factors = 1
    else:
        correction_factors = np.median(flatfield) / flatfield
        # Avoid div by 0 errors -> pixel with value 0 will be set to median
        correction_factors[flatfield == 0] = 1

    # Process each dataset
    with h5py.File(filename, "r") as f:
        for dset in f["entry"]["data"]:
            image_index = _process_dataset(
                f["entry"]["data"][dset],
                image_index,
                array_3D,
                binfactor,
                correction_factors,
            )

    scan_height = int(nimages / scan_width)

    data = array_3D.reshape(
        scan_width,
        scan_height,
        array_3D.shape[1],
        array_3D.shape[2],
    )

    # Create axes information
    axes = [
        {
            "name": "x",
            "scale": 1.0,
            "offset": 0.0,
            "units": "1",
            "size": data.shape[0],
        },
        {
            "name": "y",
            "scale": 1.0,
            "offset": 0.0,
            "units": "1",
            "size": data.shape[1],
        },
        {
            "name": "qx",
            "scale": 1.0,
            "offset": 0.0,
            "units": "1",
            "size": data.shape[2],
        },
        {
            "name": "qy",
            "scale": 1.0,
            "offset": 0.0,
            "units": "1",
            "size": data.shape[3],
        },
    ]

    # Try to read pixel size from metadata file
    try:
        with h5py.File(f"{base_filename}.h5", "r") as f:
            pixel_size = f["STEM Metadata"].attrs["Pixel Size"][0]
            # Update the scale for real space axes
            axes[0]["scale"] = pixel_size * 10  # Convert to Angstroms
            axes[1]["scale"] = pixel_size * 10
            axes[0]["units"] = "A"
            axes[1]["units"] = "A"
    except Exception as e:
        _logger.warning(f"Could not read metadata: {e}")

    # Return a list containing the dictionary
    return [
        {
            "data": data,
            "axes": axes,
            "metadata": {},
            "original_metadata": {},
            "post_process": [],
            "mapping": {},
        }
    ]


def _process_dataset(
    dset,
    start_index,
    array_3D,
    binfactor,
    correction_factors,
):
    """Process a single dataset from the arina file.

    Parameters
    ----------
    dset : h5py.Dataset
        The dataset to process.
    start_index : int
        The starting index in the output array.
    array_3D : numpy.ndarray
        The output array to fill.
    binfactor : int
        The binning factor to apply.
    correction_factors : numpy.ndarray
        The correction factors to apply.

    Returns
    -------
    int
        The next index in the output array.
    """
    image_index = start_index
    nimages_dset = dset.shape[0]

    for i in range(nimages_dset):
        if binfactor == 1:
            array_3D[image_index] = np.multiply(
                dset[i].astype(array_3D.dtype), correction_factors
            )
        else:
            array_3D[image_index] = bin2D(
                np.multiply(dset[i].astype(array_3D.dtype), correction_factors),
                binfactor,
            )

        image_index = image_index + 1
    return image_index


def bin2D(array, binfactor):
    """Bin a 2D array by a factor.

    Parameters
    ----------
    array : numpy.ndarray
        The array to bin.
    binfactor : int
        The binning factor.

    Returns
    -------
    numpy.ndarray
        The binned array.
    """
    if binfactor == 1:
        return array
    else:
        return array.reshape(
            array.shape[0] // binfactor,
            binfactor,
            array.shape[1] // binfactor,
            binfactor,
        ).mean(axis=(1, 3))
