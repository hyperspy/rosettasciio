# -*- coding: utf-8 -*-
#
# Copyright 2025 The HyperSpy developers
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
from pathlib import Path

import h5py

# import needed to be able to hdf5 filter
import hdf5plugin  # noqa F401
import numpy as np

from rsciio._docstrings import (
    FILENAME_DOC,
    LAZY_UNSUPPORTED_DOC,
    RETURNS_DOC,
)

_logger = logging.getLogger(__name__)


def file_reader(
    filename,
    lazy=False,
    navigation_shape=None,
    rebin_diffraction=1,
    dtype=None,
    flatfield=None,
):
    """
    Read arina 4D-STEM datasets.

    Parameters
    ----------
    %s
    %s
    navigation_shape : tuple or int or None, default = None
        Specify the shape of the navigation space. If None, assumes square acquisition.
        A tuple can be specified as (x_scan_dimension, y_scan_dimension), (x_scan_dimension, "auto"),
        or ("auto", y_scan_dimension). With "auto" the length is inferred from the number of
        diffraction patterns.  If only an integer is passed, it assumed to be the x_scan_dimension.
    rebin_diffraction : int, default=1
        Diffraction space binning factor for bin-on-load.
    dtype : float, optional
        Datatype for dataset.
    flatfield : numpy.ndarray, optional
        Flatfield for correction factors, converts data to float.

    %s

    Notes
    -----
    The hdf5plugin library is needed in addition to h5py due to compression filters used by the detector
    when writing data.
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
            dtype_scan = f["entry"]["data"][dset].dtype

    width = width // rebin_diffraction
    height = height // rebin_diffraction

    if navigation_shape is None:
        navigation_shape = (int(np.sqrt(nimages)), int(np.sqrt(nimages)))
    elif np.isscalar(navigation_shape):
        navigation_shape = (navigation_shape, "auto")

    if navigation_shape[0] == "auto":
        navigation_shape = (nimages // navigation_shape[1], navigation_shape[1])
    elif navigation_shape[1] == "auto":
        navigation_shape = (navigation_shape[0], nimages // navigation_shape[0])

    if nimages != navigation_shape[0] * navigation_shape[1]:
        raise ValueError("navigation_shape must be integer multiple of x*y size")

    if dtype_scan.type is np.uint32 and flatfield is None:
        _logger.info("Dataset is uint32 but will be converted to uint16")
        dtype_scan = np.dtype(np.uint16)

    if flatfield is not None:
        array_3D = np.empty((nimages, width, height), dtype=np.float32)
        _logger.info("Dataset will be converted to float32 due to flatfield correction")
    elif dtype:
        array_3D = np.empty((nimages, width, height), dtype=dtype)
    else:
        array_3D = np.empty((nimages, width, height), dtype=dtype_scan)

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
                rebin_diffraction,
                correction_factors,
            )

    data = array_3D.reshape(
        navigation_shape[0],
        navigation_shape[1],
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


file_reader.__doc__ %= (
    FILENAME_DOC,
    LAZY_UNSUPPORTED_DOC,
    RETURNS_DOC,
)


def _process_dataset(
    dset,
    start_index,
    array_3D,
    rebin_diffraction,
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
    rebin_diffraction : int
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
        if rebin_diffraction == 1:
            array_3D[image_index] = np.multiply(
                dset[i].astype(array_3D.dtype), correction_factors
            )
        else:
            array_3D[image_index] = bin2D(
                np.multiply(dset[i].astype(array_3D.dtype), correction_factors),
                rebin_diffraction,
            )

        image_index = image_index + 1
    return image_index


def bin2D(array, rebin_diffraction):
    """Bin a 2D array by a factor.

    Parameters
    ----------
    array : numpy.ndarray
        The array to bin.
    rebin_diffraction : int
        The binning factor.

    Returns
    -------
    numpy.ndarray
        The binned array.
    """
    if rebin_diffraction == 1:
        return array
    else:
        return array.reshape(
            array.shape[0] // rebin_diffraction,
            rebin_diffraction,
            array.shape[1] // rebin_diffraction,
            rebin_diffraction,
        ).mean(axis=(1, 3))
