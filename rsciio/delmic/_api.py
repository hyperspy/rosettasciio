# -*- coding: utf-8 -*-
# Copyright 2024-2025 Delmic
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

# The Delmic format is based on the SVI format, as defined here:
# http://www.svi.nl/HDF5
# A file follows this structure:
# + /
#   + Preview (contain thumbnails)
#     + RGB image (*) (HDF5 Image with Dimension Scales)
#     + DimensionScale*
#     + *Offset (position on the axis)
#   + AcquisitionName (one per set of emitter/detector)
#     + ImageData
#       + Image (HDF5 Image with Dimension Scales CTZYX or CAZYX)
#       + DimensionScale*
#       + *Offset (position on the axis)
#     + PhysicalData
#     + SVIData


import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List

import h5py
import numpy as np

from rsciio._docstrings import FILENAME_DOC, LAZY_UNSUPPORTED_DOC, RETURNS_DOC

# LumiSpy provides extra Signal types for Cathodoluminescence, which can be nice for metadata,
# but everything should work too without this extension being available.
try:
    import lumispy
except ImportError:
    lumispy = None


_logger = logging.getLogger(__name__)

# HDF5 SVI: State of the metadata -> how trustable is the value
ST_INVALID = 111
ST_DEFAULT = 112
ST_ESTIMATED = 113
ST_REPORTED = 114
ST_VERIFIED = 115

# Typical names used in Odemis streams, for different types of acquisitions
TITLE_SURVEY = "Secondary electrons survey"
TITLE_SE = "Secondary electrons concurrent"
TITLE_PMT = "CL intensity"
TITLE_AR = "Angle-resolved"
TITLE_EK1 = "AR Spectrum"
TITLE_TEMPORAL = "Time Correlator"
TITLE_TEMP_SPEC = "Temporal Spectrum"
TITLE_SPEC = "Spectrum"  # ....
TITLE_SPEC_LA = "Large Area Spectrum"  # ...
TITLE_DRIFT = "Anchor region"


class AcquisitionType(Enum):
    """
    For internal use only: the type of acquisition data.
    The order matters: it will be adhered to when returning data of multiple type
    """

    PMT = "pmt"
    AR = "ar"
    EK1 = "ek1"
    TempSpec = "temporal spectrum"
    Temporal = "temporal"
    Spectrum = "spectrum"
    SE = "se"
    Anchor = "anchor"
    Unknown = "__unknown__"
    Survey = "survey"  # SEM survey always last, to make it easy to access with [-1]


# All the type of data that is considered "CL" when opening a file
CL_ACQ_TYPES = (
    AcquisitionType.PMT,
    AcquisitionType.AR,
    AcquisitionType.EK1,
    AcquisitionType.TempSpec,
    AcquisitionType.Temporal,
    AcquisitionType.Spectrum,
)


def _h5svi_get_state(dataset: h5py.Dataset, default=None):
    """
    Read the "State" of a dataset: the confidence that can be put in the value
    dataset (Dataset): the dataset
    default: to be returned if no state is present
    return state (int or list of int): the state value (ST_*) which will be duplicated
     as many times as the shape of the dataset. If it's a list, it will be directly
     used, as is. If no state available, default is returned.
    """
    try:
        state = dataset.attrs["State"]
    except IndexError:
        return default

    return state.tolist()


def read_image_title(acq: h5py.Group) -> str:
    """
    Retrieve the image text description (aka title) from the acquisition group.

    Access the 'PhysicalData' group within an acquisition object to retrieve
    the associated image type.
    :return: the description, or "__unnamed__" if the metadata is not found or is invalid
    """
    try:
        title = acq["PhysicalData"]["Title"]
        title_str = title[()].decode("utf-8")
    except (AttributeError, TypeError, UnicodeDecodeError) as ex:  # pragma: no cover
        title_str = "__unnamed__"
        _logger.warning("Failed to read acquisition title: %s", ex)

    return title_str


def load_image(acq: h5py.Group) -> np.ndarray:
    """
    Load the raw data of an Odemis acquisition.
    Returns
    -------
    5D dataset (CTZYX or CAZYX)
    """
    image = acq["ImageData"]["Image"]
    # Check if Image is a h5py dataset
    if not isinstance(image, h5py.Dataset):
        raise TypeError("The input 'image' must be a h5 dataset.")

    # Check if Image has 5 dimensions
    if image.ndim != 5:
        raise ValueError("The input 'image' must be a 5D h5 dataset.")

    return image[:]  # Forces it to become a numpy.array


def get_unit_prefix(number: float) -> str:
    """Return the SI prefix for the given number based on its magnitude."""
    if 1e-15 <= np.abs(number) < 1e-12:
        prefix = "f"
    elif 1e-12 <= np.abs(number) < 1e-9:
        prefix = "p"
    elif 1e-9 <= np.abs(number) < 1e-6:
        prefix = "n"
    elif 1e-6 <= np.abs(number) < 1e-3:
        prefix = "µ"
    elif 1e-3 <= np.abs(number) < 1:
        prefix = "m"
    else:
        prefix = ""
    return prefix


def unit_factor(prefix: str) -> float:
    """Return the multiplication factor for a SI unit prefix."""
    prefix_to_factor = {"f": 1e15, "p": 1e12, "n": 1e9, "µ": 1e6, "m": 1e3}

    return prefix_to_factor.get(prefix, 1)


def make_axes(acq: h5py.Group) -> List[Dict[str, Any]]:
    """
    Create a list of the axes of a dataset.
    """
    img_data = acq["ImageData"]
    image = img_data["Image"]

    # List of possible axes: dimension label (in HDF5), axis name (for HyperSpy), scale, offset
    # The standard order in Odemis is CTZYX (or CAZYX), but this shouldn't matter, as we'll read
    # the information for each dimension, in the order stored.
    AXIS_INFO = {
        "C": ("Wavelength", "DimensionScaleC", None),
        # T and A are never present simultaneously
        "T": ("Time", "DimensionScaleT", None),
        "A": ("Angle", "DimensionScaleA", None),
        # Typically Z not used in CL data, so will be always be "squeezed" later
        "Z": ("Z", "DimensionScaleZ", "ZOffset"),
        "Y": ("Y", "DimensionScaleY", "YOffset"),
        "X": ("X", "DimensionScaleX", "XOffset"),
    }

    # Iterate over the axes, adding them all to the axes list. In the order of the image dimensions.
    axes = []
    for i, dim in enumerate(image.dims):
        axis_dict = {}
        axes.append(axis_dict)

        try:
            axis_name, scale_key, offset_key = AXIS_INFO[dim.label]
        except KeyError as ex:  # pragma: no cover
            _logger.warning("Data %s has unknown axis %s", acq.name, ex)
            axis_dict["name"] = dim.label
            continue

        axis_dict["name"] = axis_name
        if scale_key not in img_data:
            # No info available about this axis (probably because this dimension is empty)
            if image.shape[i] > 1:
                _logger.warning(
                    "Axis %s has length %s but no scale information",
                    axis_name,
                    image.shape[i],
                )
            continue

        try:
            scale = np.array(img_data[scale_key])

            # The DimensionScale array for C & T axes is explicit: for each index -> the position
            if axis_name == "Wavelength":
                # TODO: in theory, can also be unknown, in which case it's in pixel indices
                scale_value = np.mean(scale)
                prefix = get_unit_prefix(scale_value)
                axis_dict.update(
                    {
                        "axis": scale * unit_factor(prefix),  # Full axis loaded
                        "units": prefix + "m",  # Adjusted unit prefix
                        "navigate": False,  # No navigation for the C axis
                    }
                )
            elif axis_name == "Time":
                scale_value = np.mean(scale)
                prefix = get_unit_prefix(scale_value)
                axis_dict.update(
                    {
                        "axis": scale * unit_factor(prefix),  # Full axis loaded
                        "units": prefix + "s",  # Adjusted unit prefix
                        "navigate": False,  # No navigation for the T axis
                    }
                )
            elif axis_name == "Angle":
                # The dimension scale contains the axis in radians, at most between -90° -> 90°, but
                # can also contain NaN for data acquired outside from the range (typically, irrelevant
                # data that can be discarded). For now, keep it simple, and just pass the pixel number.
                scale_a = np.arange(0, image.shape[i])
                axis_dict.update(
                    {
                        "axis": scale_a,
                        "units": "",
                        "navigate": False,  # No navigation for the A axis
                    }
                )
            else:  # XYZ
                # The pixel size is defined in the DimensionScale, as a single value, and the *center* of
                # the image is defined by the Offset
                try:
                    scale_value = float(scale)
                    prefix = get_unit_prefix(scale_value)
                except (TypeError, ValueError) as e:  # pragma: no cover
                    raise TypeError(
                        f"Expected a numeric value for '{scale_key}', got {type(scale)} instead."
                    ) from e

                # Y goes up, so top (first) pixel has the largest value => invert scale
                if axis_name == "Y":
                    scale_value = -scale_value

                center = float(img_data[offset_key][()])
                # In HyperSpy, offset is the (center of) the first pixel
                offset = center - ((image.shape[i] - 1) / 2) * scale_value

                axis_dict.update(
                    {
                        "size": image.shape[i],
                        "offset": offset * unit_factor(prefix),
                        "scale": scale_value * unit_factor(prefix),
                        "units": prefix + "m",
                        "navigate": True,
                    }
                )
        except Exception as ex:  # pragma: no cover
            _logger.warning("Failed to parse axis %s: %s", axis_name, ex)

    return axes


def make_metadata(
    acq: h5py.Group, acq_type: AcquisitionType, original_md: Dict[str, Any]
) -> Dict[str, dict]:
    """
    Create a metadata dictionary for a given acquisition.
    See https://hyperspy.org/hyperspy-doc/current/reference/metadata.html
    In practice, it's used by HyperSpy to instantiate a Signal.
    """
    img_title = read_image_title(acq)
    metadata = {
        "General": {
            "title": img_title,
        },
        "Signal": {
            "quantity": "Counts",
            "signal_type": "",
        },
    }

    # If LumiSpy is available => more fancy Signal classes available
    if lumispy:
        ACQ_TO_SIGNAL_TYPE = {
            AcquisitionType.TempSpec: "LumiTransientSpectrum",
            AcquisitionType.Temporal: "LumiTransient",
            AcquisitionType.Spectrum: "CL_SEM",
        }
        signal_type = ACQ_TO_SIGNAL_TYPE.get(acq_type, "")
        metadata["Signal"]["signal_type"] = signal_type

    # Store polarization information as a "detector filter", if present
    pol = original_md.get("Polarization")
    if pol is not None:
        metadata["Acquisition_instrument"] = {
            "Spectrometer": {
                "Filter": {
                    "filter_type": "polarization analyzer",
                    "position": pol,
                },
            },
        }

    try:
        acq_date = original_md["AcquisitionDate"]
        dt = datetime.fromtimestamp(acq_date, timezone.utc)
        date_str = dt.date().isoformat()  # 'YYYY-MM-DD'
        time_str = dt.time().isoformat()  # 'HH:MM:SS.microseconds'
        metadata["General"].update(
            {
                "date": date_str,
                "time": time_str,
            }
        )
    except KeyError:
        pass  # No acquisition date info, then we can skip it
    except Exception:  # pragma: no cover
        _logger.warning("Failed to parse acquisition date")

    return metadata


def make_original_metadata(acq: h5py.Group) -> Dict[str, Any]:
    """
    Create a dictionary for the original metadata of a dataset."""
    original_metadata = {}

    # Copy every value found in PhysicalData (almost) as-is.
    # The exceptions are:
    # * Title is always the same as "ChannelDescription", so discard it
    # * if the state is "Invalid", then discard it.
    # * ExtraSettings is special, as it's a dict encoded in JSON, so decode it
    for name, item in acq["PhysicalData"].items():
        # Only care about datasets (typically, there are only datasets anyway)
        if not isinstance(item, h5py.Dataset):
            continue
        if name == "Title":  # Special value, copy of ChannelDescription[0]
            continue

        # Each Dataset is an array of 1 value, with an attribute containing its state
        state = _h5svi_get_state(item)
        if state is not None and state[0] == ST_INVALID:
            continue

        try:
            # It's either a numpy scalar, or an array of 1 dimension, with 1 element
            if item.ndim == 0:  # scalar
                value = item[()]
            else:  # ndim == 1
                value = item[0]

            # Convert to a simple Python type
            if name == "ExtraSettings":
                value = json.loads(value)
            elif isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, (np.ndarray, np.generic)):
                value = value.tolist()
            original_metadata[name] = value
        except Exception as ex:  # pragma: no cover
            _logger.warning("Failed to parse original metadata %s: %s", name, ex)

    # Add metadata from the ImageData
    t_offset = acq["ImageData"]["TOffset"]
    acq_date = float(t_offset[()])  # unix timestamp
    original_metadata["AcquisitionDate"] = acq_date

    # Add HDF5 Group name, typically "/Acquisition.."
    original_metadata["HDF5Group"] = acq.name

    # Add from SVIData: all data are just plain strings
    svi_data = acq["SVIData"]
    original_metadata["SVIData"] = {}
    for k in ("Company", "WriterVersion"):
        try:
            value = svi_data[k][()].decode("utf-8")
            original_metadata["SVIData"][k] = value
        except Exception as ex:  # pragma: no cover
            _logger.warning("Failed to parse original metadata %s: %s", k, ex)

    return original_metadata


def guess_signal_type(data, axes, original_metadata) -> AcquisitionType:
    # Primarily distinguish based on the dimensions, and use metadata and title as extra clue
    title = original_metadata["ChannelDescription"]

    if title == TITLE_DRIFT:  # Quick short-cut for drift correction (SEM over time)
        return AcquisitionType.Anchor

    def get_axis_length(name):
        for idx, axis in enumerate(axes):
            if axis.get("name") == name:
                return data.shape[idx]
        return 1  # not present => it's just a single position

    C = get_axis_length("Wavelength")
    A = get_axis_length("Angle")
    T = get_axis_length("Time")

    if A > 1:  # EK (because AR doesn't explicitly have angular axis)
        if C > 1:
            return AcquisitionType.EK1
        else:
            return AcquisitionType.Unknown

    # A == 1
    if T > 1:  # Temporal or Temporal Spectrum
        if C == 1:
            return AcquisitionType.Temporal
        else:
            return AcquisitionType.TempSpec

    # A == 1, and T == 1
    if C > 1:
        return AcquisitionType.Spectrum

    # Just 2D data (assuming Z == 1, which is typically the case)
    if isinstance(
        original_metadata.get("PolePosition"), list
    ):  # Typical metadata for AR
        return AcquisitionType.AR
    elif title == TITLE_SE:
        return AcquisitionType.SE
    elif title == TITLE_SURVEY:
        return AcquisitionType.Survey
    elif title == TITLE_PMT:
        return AcquisitionType.PMT

    _logger.info("Found acquisition data of unknown type")
    return AcquisitionType.Unknown


def read_acquisition(acq: h5py.Group) -> Dict[str, Any]:
    """
    Reads one Odemis acquisition data, with its metadata
    """
    data = load_image(acq)
    axes = make_axes(acq)
    original_metadata = make_original_metadata(acq)
    acq_type = guess_signal_type(data, axes, original_metadata)
    metadata = make_metadata(acq, acq_type, original_metadata)

    acq_dict = {
        "data": data,
        "axes": axes,
        "metadata": metadata,
        "original_metadata": original_metadata,
        "_acq_type": acq_type,
    }
    return acq_dict


def merge_ar_data(
    ar_data: List[Dict[str, Any]], se_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Search for multiple Angular-Resolved acquisition (of a single beam position, so 2D) and merge them
    together along the X & Y dimensions.
    Parameters
    ----------
    ar_data:  All the AR acquisition of shape 111BA
    se_data: the SEM concurrent data

    Returns
    -------
    ar_merged: all the AR data replaced by a single one, 4D.
    """
    # Each AR data has been acquired at one (different) beam position. Each beam position corresponds
    # to one pixel of the SEM concurrent data. So we use the same XY axes of the SEM concurrent data
    # to add the spatial information of the AR data.
    # Note that 2D AR data does have the position of the ebeam stored as XY axes (single coordinate).
    # So, in theory, it should be possible to recreate it without SEM concurrent... but it's more
    # complicated.
    Y, X = se_data["data"].shape[-2:]
    if X * Y != len(ar_data):  # pragma: no cover
        raise ValueError(f"Expected {X}x{Y} AR data, but got {len(ar_data)}")

    # As we know that the scan order is always the same order, the simplest is to use acquisition
    # timestamp to place the data in the right position. An alternative would be to use the X, Y
    # (offset) position, but that's more complicated, especially if there is rotation.
    sorted_ar = sorted(ar_data, key=lambda d: d["original_metadata"]["AcquisitionDate"])

    ar0 = sorted_ar[0]
    ar0_shape = sorted_ar[0]["data"].shape
    # Get the 2 angle dimensions from first AR data (assuming all the AR data is the same size, and
    # the other dimensions are 1)
    if ar0_shape[:3] != (1, 1, 1):  # pragma: no cover
        logging.warning(
            "Unexpected extra dimension on AR data, with shape: %s", ar0_shape
        )
    B, A = ar0_shape[-2:]
    # Create an array with all the data, introducing an extra dimension on the "slow" (left) side
    merged_data = np.array([d["data"] for d in sorted_ar])  # (Y * X, B, A)
    merged_data.shape = (Y, X, B, A)  # Assumes it's been scanned X fast, Y slow

    # The X & Y axes in the original 2D AR image (camera) don't directly correspond to angles.
    # They would need to be "projected" according to the parabolic mirror shape (described in the metadata).
    # Just number them in terms of "pixels" to avoid confusion.
    merged_axes = [
        se_data["axes"][-2],  # Y
        se_data["axes"][-1],  # X
        {
            "name": "Angle B",
            "axis": np.arange(0, B),
            "units": "",
            "navigate": False,
        },
        {
            "name": "Angle A",
            "axis": np.arange(0, A),
            "units": "",
            "navigate": False,
        },
    ]

    merged_ar = {
        "data": merged_data,
        "axes": merged_axes,
        "metadata": ar0["metadata"],
        "original_metadata": ar0["original_metadata"],
        "_acq_type": ar0["_acq_type"],
    }

    return merged_ar


def squeeze_dimensions(acq_data: Dict[str, Any]) -> None:
    """
    Removes empty dimensions from a dataset. It removes them from the numpy array shape, and from
    the axes information.
    Parameters
    ----------
    acq_data: the dataset with "data" and "axes" keys at least. Its content will be updated.
    """
    data = acq_data["data"]
    axes = acq_data["axes"]
    # Scan the dimensions in reverse order to not be affected when the dimensions are deleted
    for i in range(data.ndim - 1, -1, -1):
        if data.shape[i] == 1:
            # Remove the current dimension
            data.shape = data.shape[:i] + data.shape[i + 1 :]
            del axes[i]


HS_AXES_ORDER = "Z", "Y", "X", "Time", "Angle", "Wavelength"


def reorder_dimensions(acq_data: Dict[str, Any]) -> None:
    """
    Reorder the axes, in order to make the data most straightforwards to visualize and manipulate
    in HyperSpy.
    Parameters
    ----------
    acq_data: the dataset, with "data" and "axes" keys at least. Its content will be updated. Both
    data and axes are reordered. Unknown axes are placed at the end, in their original order.
    """
    data = acq_data["data"]
    axes = acq_data["axes"]

    def index_in_axes(n):
        try:
            return HS_AXES_ORDER.index(n)
        except ValueError:  # n is not known => put last
            return len(HS_AXES_ORDER) + 1

    orig_names = [a["name"] for a in axes]
    sorted_names = sorted(orig_names, key=index_in_axes)
    if sorted_names != orig_names:
        # Build the new order of axes
        order = [orig_names.index(name) for name in sorted_names]
        # Move axes to the new order
        data = np.moveaxis(data, order, range(len(order)))
        acq_data["data"] = data
        # Reorder axes list accordingly
        acq_data["axes"] = [axes[i] for i in order]


def file_reader(
    filename: str, signal: str = "cl", lazy: bool = False
) -> List[Dict[str, Any]]:
    """
    Read a Delmic HDF5 hyperspectral image.

    Parameters
    ----------
    %s
    signal : str, default="cl"
      Specifies the type of data to load. Can be "cl" (cathodoluminescence signal), "se"
      (concurrent SEM signal), "survey" (SEM image of the whole field of view), "anchor"
      (drift correction region over time), or "all" to obtain all the data in the file.
      Convenient, as typically in Odemis, every CL acquisition automatically stores also the survey
      and concurrent SEM data.
    %s

    %s
    """
    if lazy is not False:
        raise NotImplementedError("Lazy loading is not supported.")

    if signal == "all":
        # TODO: do not include "Unknown" and "Anchor" types?
        filtered_acq_types = tuple(AcquisitionType)  # all of them
    elif signal == "cl":
        filtered_acq_types = CL_ACQ_TYPES
    elif signal in ("survey", "se", "anchor"):
        filtered_acq_types = (AcquisitionType(signal),)
    else:
        raise ValueError(f"Unexpected signal '{signal}'")

    # Load all the data, which matches the signal type
    with h5py.File(filename, "r") as hdf:
        acq_data = []  # Return value: list of dicts containing each acquisition

        # For merging Angular-Resolved data into a single 4D array
        ar_data: Dict[str, list] = {}  # polarization -> list of AR acquisitions
        sem_concurrent_ar_data = None

        # Go through each "Group", and check whether it could be an Odemis acquisition
        for oname, obj in hdf.items():
            # Only interested in "Acquisition..." groups
            if not isinstance(obj, h5py.Group) or not oname.startswith("Acquisition"):
                continue
            # Needs to have SVIData, ImageData/Image, and PhysicalData
            if (
                "SVIData" not in obj
                or "PhysicalData" not in obj
                or "ImageData" not in obj
                or "Image" not in obj["ImageData"]
            ):
                continue  # not conforming => try next object

            # Load data
            try:
                data = read_acquisition(obj)
            except Exception as ex:  # pragma: no cover
                _logger.warning("Failed to read data %s, skipping it: %s", oname, ex)
                continue

            # Skip the data which is not in one of the acquisition types the user is interested in
            if AcquisitionType.AR in filtered_acq_types:
                # Keep the AR data apart, for merging it later
                if data["_acq_type"] == AcquisitionType.AR:
                    # Standard AR acquisitions have no polarization, and "polarized" ones have typically
                    # 6 polarizations positions.
                    pol = data["original_metadata"].get("Polarization", "")
                    if pol not in ar_data:
                        ar_data[pol] = []
                    ar_data[pol].append(data)
                elif data["_acq_type"] == AcquisitionType.SE:
                    sem_concurrent_ar_data = data

            if (
                data["_acq_type"] in filtered_acq_types
                and data["_acq_type"] != AcquisitionType.AR
            ):
                acq_data.append(data)

        # Merge AR data, if there is AR data
        if ar_data:
            if sem_concurrent_ar_data is None:  # pragma: no cover
                _logger.warning(
                    "AR data present but no SEM concurrent data found to reconstruct it"
                )
                acq_data.extend(ar_data)  # Still pass it on, as-is
            else:
                for pol, ar_polarized in ar_data.items():
                    try:
                        ar_merged = merge_ar_data(ar_polarized, sem_concurrent_ar_data)
                        acq_data.append(ar_merged)
                    except Exception as ex:  # pragma: no cover
                        _logger.warning(
                            "Failed to merge AR data, will pass it as separate data: %s",
                            ex,
                        )
                        acq_data.extend(ar_polarized)

        # No data found: that's the sign it's not a Delmic HDF5
        if not acq_data:
            raise IOError("The file is not a valid Delmic HDF5 file")

        # squeeze dimensions: remove axes which are empty
        for data in acq_data:
            squeeze_dimensions(data)
            reorder_dimensions(data)

        # Reorder the datasets to always be in the same (user-friendly) order:
        # the order in which the AcquisitionType enum is defined.
        acq_data.sort(key=lambda d: tuple(AcquisitionType).index(d["_acq_type"]))

        return acq_data


file_reader.__doc__ %= (FILENAME_DOC, LAZY_UNSUPPORTED_DOC, RETURNS_DOC)
