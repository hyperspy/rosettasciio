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

import h5py
import numpy as np
import importlib.util

from rsciio._docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC


def count_acquisitions(hdf: h5py.File) -> int: 
    """Count the number of first-level groups in an HDF5 file."""

    # Count the number of first-level groups
    group_count = sum(
        1 for item in hdf.values() if isinstance(item, h5py.Group)
        )
    return group_count



def read_image_type(Acq: h5py.Group) -> str:
    """
    Retrieve the image type from the 'PhysicalData' group.

    Access the 'PhysicalData' group within an acquisition object to retrieve
    the associated image type.
    """
    if Acq is None:
        raise TypeError(
            "The input 'Acq' is None. A valid acquisition object is required."
        )

    if "PhysicalData" in Acq:
        PhysData = Acq.get("PhysicalData")
        if PhysData is not None:
            Title = PhysData.get("Title")
            if Title is not None:
                try:
                    Title_str = Title[()].decode("utf-8")
                except (AttributeError, TypeError, UnicodeDecodeError) as e:
                    Title_str = f"Decoding error: {e}"
            else:
                Title_str = "Title dataset missing"
        else:
            Title_str = "PhysicalData group missing"
    else:
        Title_str = "PhysicalData group missing"

    return Title_str


def is_AR(filename: str) -> bool:
    """
    Check if the HDF5 file has 'Angle-resolved' dataset type.

    This function verifies whether any acquisition in the provided HDF5
    file has the "Angle-resolved" type.
    """
    try:
        with h5py.File(filename, "r") as hdf:
            num_acq = count_acquisitions(hdf)

            for i in range(num_acq):
                Acquisition = "Acquisition" + str(i)
                Acq = hdf.get(Acquisition)

                if (
                        Acq is not None and
                        read_image_type(Acq) == "Angle-resolved"
                ):
                    return True  # Early return if AR type is found

    except IOError as e:
        raise IOError(f"Unable to open the file: {filename}. Error: {e}")

    # Return False if no "Angle-resolved" image type is found
    return False


def load_image(Image: h5py.Dataset) -> np.ndarray:
    """Extract a 2D image from a 5D dataset."""
    # Check if Image is a h5py dataset
    if not isinstance(Image, h5py.Dataset):
        raise TypeError("The input 'Image' must be a h5 dataset.")

    # Check if Image has 5 dimensions
    if Image.ndim != 5:
        raise ValueError("The input 'Image' must be a 5D h5 dataset.")

    # Check for the expected dimensions
    if Image.shape[3] < 1 or Image.shape[4] < 1:
        raise ValueError(
            "The input 'Image' does not have the expected dimensions."
        )

    # Extract and transpose the data
    data = Image[0, 0, 0, :, :]

    return data


def load_hyperspectral(Image: h5py.Dataset) -> np.ndarray:
    """Extract a 3D hyperspectral dataset from a 5D dataset."""
    # Check if Image is a h5py dataset
    if not isinstance(Image, h5py._hl.dataset.Dataset):
        raise TypeError("The input 'Image' must be a h5 dataset.")

    # Check if Image has 5 dimensions
    if Image.ndim != 5:
        raise ValueError("The input 'Image' must be a 5D h5 dataset.")

    # Check for the expected dimensions
    if Image.shape[0] < 1 or Image.shape[3] < 1 or Image.shape[4] < 1:
        raise ValueError(
            "The input 'Image' does not have the expected dimensions."
        )

    # Extract and transpose the data
    if Image.shape[3] == 1 and Image.shape[4] == 1: 
        data = Image[:, 0, 0, 0, 0]
    else:
        data = Image[:, 0, 0, :, :].transpose(1, 2, 0)

    return data


def load_streak_camera_image(Image: h5py.Dataset) -> np.ndarray:
    """Extract a a 4D streak camera image from a 5D dataset."""
    # Check if Image is a h5py dataset
    if not isinstance(Image, h5py._hl.dataset.Dataset):
        raise TypeError("The input 'Image' must be a h5 dataset.")

    # Check if Image has 5 dimensions
    if Image.ndim != 5:
        raise ValueError("The input 'Image' must be a 5D h5 dataset.")

    # Check for the expected dimensions
    if (
        Image.shape[0] < 1
        or Image.shape[1] < 1
        or Image.shape[3] < 1
        or Image.shape[4] < 1
    ):
        raise ValueError(
            "The input 'Image' does not have the expected dimensions."
        )

    # Extract and transpose the data
    if Image.shape[3] == 1 and Image.shape[4] == 1: 
        data = Image[:, :, 0, 0, 0].transpose(1, 0)
    else:
        data = Image[:, :, 0, :, :].transpose(2, 3, 1, 0)
    # .transpose(3, 2, 0, 1)

    return data


def load_AR_spectrum(Image: h5py.Dataset) -> np.ndarray:
    """Extract an Angle-Resolved Spectrum (aka E-k) from a 5D dataset."""
    # Check if Image is a h5py dataset
    if not isinstance(Image, h5py.Dataset): ## XXX to remove
        raise TypeError("The input 'Image' must be a h5 dataset.") ## XXX to remove

    # Check if Image has 5 dimensions
    if Image.ndim != 5:
        raise ValueError("The input 'Image' must be a 5D h5 dataset.")

    # Check for the expected dimensions
    if (
        Image.shape[0] < 1
        or Image.shape[1] < 1
        or Image.shape[3] < 1
        or Image.shape[4] < 1
    ):
        raise ValueError(
            "The input 'Image' does not have the expected dimensions."
        )

    # Extract and transpose the data
    if Image.shape[3] == 1 and Image.shape[4] == 1: 
        data = Image[:, :, 0, 0, 0].transpose(1, 0)
    else:
        data = Image[:, :, 0, :, :].transpose(2, 3, 1, 0)

    return data


def load_temporal_trace(Image: h5py.Dataset) -> np.ndarray:
    """Extract a decay trace or g(2) curve from a 5D dataset."""
    # Check if Image is a h5py dataset
    if not isinstance(Image, h5py._hl.dataset.Dataset):
        raise TypeError("The input 'Image' must be a h5 dataset.")

    # Check if Image has 5 dimensions
    if Image.ndim != 5:
        raise ValueError("The input 'Image' must be a 5D h5 dataset.")

    # Check for the expected dimensions
    if Image.shape[1] < 1 or Image.shape[3] < 1 or Image.shape[4] < 1:
        raise ValueError(
            "The input 'Image' does not have the expected dimensions."
        )

    # Extract and transpose the data
    if Image.shape[3] == 1 and Image.shape[4] == 1: 
        data = Image[0, :, 0, 0, 0]
    else:
        data = Image[0, :, 0, :, :].transpose(1, 2, 0)

    return data


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


def make_axes(Acq):
    """Create a dictionary for the axes of a dataset."""
    ImgData = Acq.get("ImageData")
    Image = ImgData.get("Image")
    axes = []

    # Dynamically detect the presence of "T" or "A" in the data
    if "DimensionScaleT" in ImgData:
        time_axis = ("T", "DimensionScaleT", "TOffset")
    elif "DimensionScaleA" in ImgData:
        time_axis = ("A", "DimensionScaleA", "AOffset")
    else:
        time_axis = None

    # List of possible axes with corresponding scale and offset names,
    # reordered to X, Y, C (Wavelength), A (Angle)
    axis_info = [
        ("X", "DimensionScaleX", "XOffset"),  # X axis
        ("Y", "DimensionScaleY", "YOffset"),  # Y axis
        ("C", "DimensionScaleC", "COffset"),  # C axis for Wavelength
        time_axis,  # T axis for Time or A axis for Angle
    ]

    # Iterate over the axes, handling them appropriately
    for i, axis in enumerate(axis_info):
        if axis is None:  # Skip if no time or angle axis is detected
            continue
        axis_name, scale_key, offset_key = axis
        if scale_key in ImgData:
            Scale = np.array(ImgData.get(scale_key))

            if axis_name in ["C", "T"]:
                scale_value = np.mean(
                    Scale
                )
                prefix = get_unit_prefix(
                    scale_value
                )
            elif axis_name == "A":
                scale_value = None  # No scale value needed for "A" axis
                prefix = ""  # No prefix needed for "A" axis
            else:
                try:
                    scale_value = float(Scale)
                    prefix = get_unit_prefix(scale_value)
                except (TypeError, ValueError) as e:
                    raise TypeError(
                        f"Expected a numeric value for '{scale_key}', "
                        f"got {type(Scale)} instead."
                    ) from e

            if scale_value is not None and not (1e-15 < scale_value < 1):
                raise ValueError(
                    f"Scale value {scale_value} for axis {axis_name}"
                    f"is outside of expected ranges."
                )

            # Handle the special cases for the C, T, and A axes
            if axis_name == "C":
                axis_dict = {
                    "name": "Wavelength",
                    "axis": Scale * unit_factor(prefix),  # Full axis loaded
                    "units": prefix + "m",  # Adjusted unit prefix
                    "navigate": False,  # No navigation for the C axis
                }
            elif axis_name == "T":
                axis_dict = {
                    "name": "Time",
                    "axis": Scale * unit_factor(prefix),  # Full axis loaded
                    "units": prefix + "s",  # Adjusted unit prefix
                    "navigate": False,  # No navigation for the T axis
                }
            elif axis_name == "A":
                ScaleA = np.arange(0, Image.shape[1])
                axis_dict = {
                    "name": "Angle",
                    "axis": ScaleA,  # Full axis loaded, no scaling needed
                    "units": "",  # Unit for Angle
                    "navigate": False,  # No navigation for the A axis
                }
            else:
                axis_dict = {
                    "name": axis_name,
                    "size": Image.shape[
                        -(i + 1)
                    ],
                    "offset": float(np.array(ImgData.get(offset_key)))
                    * unit_factor(prefix),
                    "scale": scale_value * unit_factor(prefix),
                    "units": prefix + "m",
                    "navigate": True,
                }
            axes.append(axis_dict)

    # Reorder axes to X, Y, Wavelength, Angle
    axes_ordered = []
    for axis_name in ["Y", "X", "Time", "Angle", "Wavelength"]:
        for axis in axes:
            if axis["name"] == axis_name:
                axes_ordered.append(axis)
                break

    return axes_ordered

def make_signal_axes(Acq):
    """Create a dictionary for the axes of a dataset."""
    ImgData = Acq.get("ImageData")
    Image = ImgData.get("Image")
    axes = []

    # Dynamically detect the presence of "T" or "A" in the data
    if "DimensionScaleT" in ImgData:
        time_axis = ("T", "DimensionScaleT", "TOffset")
    elif "DimensionScaleA" in ImgData:
        time_axis = ("A", "DimensionScaleA", "AOffset")
    else:
        time_axis = None

    # List of possible axes with corresponding scale and offset names,
    # reordered to C (Wavelength), A (Angle)
    axis_info = [
        ("C", "DimensionScaleC", "COffset"),  # C axis for Wavelength
        time_axis,  # T axis for Time or A axis for Angle
    ]

    # Iterate over the axes, handling them appropriately
    for i, axis in enumerate(axis_info):
        if axis is None:  # Skip if no time or angle axis is detected
            continue
        axis_name, scale_key, offset_key = axis
        if scale_key in ImgData:
            Scale = np.array(ImgData.get(scale_key))

            if axis_name in ["C", "T"]:
                scale_value = np.mean(
                    Scale
                )
                prefix = get_unit_prefix(
                    scale_value
                )
            elif axis_name == "A":
                scale_value = None  # No scale value needed for "A" axis
                prefix = ""  # No prefix needed for "A" axis
            else:
                try:
                    scale_value = float(Scale)
                    prefix = get_unit_prefix(scale_value)
                except (TypeError, ValueError) as e:
                    raise TypeError(
                        f"Expected a numeric value for '{scale_key}', "
                        f"got {type(Scale)} instead."
                    ) from e

            if scale_value is not None and not (1e-15 < scale_value < 1):
                raise ValueError(
                    f"Scale value {scale_value} for axis {axis_name}"
                    f"is outside of expected ranges."
                )

            # Handle the special cases for the C, T, and A axes
            if axis_name == "C":
                axis_dict = {
                    "name": "Wavelength",
                    "axis": Scale * unit_factor(prefix),  # Full axis loaded
                    "units": prefix + "m",  # Adjusted unit prefix
                    "navigate": False,  # No navigation for the C axis
                }
            elif axis_name == "T":
                axis_dict = {
                    "name": "Time",
                    "axis": Scale * unit_factor(prefix),  # Full axis loaded
                    "units": prefix + "s",  # Adjusted unit prefix
                    "navigate": False,  # No navigation for the T axis
                }
            elif axis_name == "A":
                ScaleA = np.arange(0, Image.shape[1])
                axis_dict = {
                    "name": "Angle",
                    "axis": ScaleA,  # Full axis loaded, no scaling needed
                    "units": "",  # Unit for Angle
                    "navigate": False,  # No navigation for the A axis
                }
            else:
                continue
            axes.append(axis_dict)

    # Reorder axes to Wavelength, Angle
    axes_ordered = []
    for axis_name in ["Time", "Angle", "Wavelength"]:
        for axis in axes:
            if axis["name"] == axis_name:
                axes_ordered.append(axis)
                break

    return axes_ordered


def make_metadata(Acq):
    """Create a metadata dictionary for a given acquisition (Acq)."""

    metadata = {
        "Signal": {
            "quantity": "",
            "signal_type": "",
        }
    }

    img_type=read_image_type(Acq)

    if importlib.util.find_spec("lumispy") is None:
        metadata["Signal"]["quantity"] = "Counts"
        if img_type == "CL intensity":
            metadata["Signal"]["signal_type"] = "Signal2D"
        elif img_type == "Secondary electrons concurrent":
            metadata["Signal"]["signal_type"] = "BaseSignal"
        elif img_type == "Secondary electrons survey":
            metadata["Signal"]["signal_type"] = "BaseSignal"
        elif img_type.startswith("Spectrum"):
            metadata["Signal"]["signal_type"] = "Signal1D"
        elif img_type.startswith("Large Area"):
            metadata["Signal"]["signal_type"] = "Signal1D"
        elif img_type == "Time Correlator":
            metadata["Signal"]["signal_type"] = "Signal1D"
        elif img_type == "Temporal Spectrum":
            metadata["Signal"]["signal_type"] = "Signal2D"
        elif img_type == "Angle-resolved":
            metadata["Signal"]["signal_type"] = "Signal2D"
        elif img_type == "AR Spectrum":
            metadata["Signal"]["signal_type"] = "Signal2D"
        elif  img_type == "Anchor region":
            pass
    else:
        metadata["Signal"]["quantity"] = "Counts"
        if img_type == "CL intensity":
            metadata["Signal"]["signal_type"] = "Signal2D"
        elif img_type == "Secondary electrons concurrent":
            metadata["Signal"]["signal_type"] = "BaseSignal"
        elif img_type == "Secondary electrons survey":
            metadata["Signal"]["signal_type"] = "BaseSignal"
        elif img_type.startswith("Spectrum"):
            metadata["Signal"]["signal_type"] = "CLSEM"
        elif img_type.startswith("Large Area"):
            metadata["Signal"]["signal_type"] = "CLSEM"
        elif img_type == "Time Correlator":
            metadata["Signal"]["signal_type"] = "LumiTransient"
        elif img_type == "Temporal Spectrum":
            metadata["Signal"]["signal_type"] = "LumiTransientSpectrum"
        elif img_type == "Angle-resolved":
            metadata["Signal"]["signal_type"] = "Signal2D"
        elif img_type == "AR Spectrum":
            metadata["Signal"]["signal_type"] = "Signal2D"
        elif  img_type == "Anchor region":
            pass

    return metadata

def make_original_metadata(Acq: h5py.Group) -> str:
    """Create a dictionary for the original metadata of a dataset."""
    original_metadata = {}

    if "PhysicalData" in Acq.keys():
        PhysData = Acq.get("PhysicalData")
        if PhysData is not None:
            
            # Read all attributes in the group
            original_metadata = {key: value for key, value in PhysData.attrs.items()}
                
            # Read all datasets in the group, skipping unsupported items
            for item_name, item in PhysData.items():
                if isinstance(item, h5py.Dataset):  # Ensure the item is a Dataset
                    if item.shape is not None:  # Skip datasets with NULL dataspace
                        original_metadata[item_name] = item[()]  # Read the entire dataset
        else:
            original_metadata = {}  # Return an empty dictionary if the group doesn't exist
    
    return original_metadata


def make_original_AR_metadata(Acq, data):
    """Create a dictionary of the original metadata of an AR image dataset."""
    original_metadata = {}

    if "PhysicalData" in Acq.keys():
        PhysData = Acq.get("PhysicalData")
        if PhysData is not None:
            
            # Read all attributes in the group
            original_metadata = {key: value for key, value in PhysData.attrs.items()}
                
            # Read all datasets in the group, skipping unsupported items
            for item_name, item in PhysData.items():
                if isinstance(item, h5py.Dataset):  # Ensure the item is a Dataset
                    if item.shape is not None:  # Skip datasets with NULL dataspace
                        original_metadata[item_name] = item[()]  # Read the entire dataset
        else:
            original_metadata = {}  # Return an empty dictionary if the group doesn't exist
            
    return original_metadata


def load_AR(filename: str) -> np.ndarray:
    """Load angle-resolved (AR) image data from the specified HDF5 file."""
    try:
        with h5py.File(filename, "r") as hdf:
            num_acq = count_acquisitions(hdf)

            # Initialize dimensions for data array
            x, y, a, b = None, None, None, None
            data_shape_identified = False

            for i in range(min(3, num_acq)):
                Acquisition = "Acquisition" + str(i)

                if Acquisition in hdf:
                    Acq = hdf.get(Acquisition)
                    ImgData = Acq.get("ImageData")
                    Image = ImgData.get("Image")

                    img_type = read_image_type(Acq)

                    if img_type == "Secondary electrons concurrent":
                        x, y = Image[0, 0, 0, :, :].transpose().shape

                    elif img_type == "Secondary electrons survey":
                        continue  # Skip this image type

                    elif img_type == "Angle-resolved":
                        a, b = Image[0, 0, 0, :, :].transpose().shape
                        data_shape_identified = True
                        break  # Exit once dimensions are identified

                    else:
                        raise ValueError(f"Unknown data type: {img_type}")
                else:
                    raise ValueError(
                        f"Acquisition '{Acquisition}' is missing or invalid."
                    )

            # Ensure that dimensions have been identified
            if (
                not data_shape_identified
                or x is None
                or y is None
                or a is None
                or b is None
            ):
                raise ValueError(
                    "Could not determine data shape from initial acquisitions."
                )

            # Initialize the data array
            data = np.empty((x, y, a, b))

            # Reset the indices for data population
            j, k = 0, 0

            # Second loop: Populate the data array with AR image data
            for i in range(num_acq):
                Acquisition = "Acquisition" + str(i)

                if Acquisition in hdf:
                    Acq = hdf.get(Acquisition)
                    img_type = read_image_type(Acq)
                    if img_type == "Angle-resolved":
                        ImgData = Acq.get("ImageData")
                        Image = ImgData.get("Image")

                        data[j, k, :, :] = Image[0, 0, 0, :, :].transpose()

                        j += 1
                        if j == x:
                            k += 1
                            j = 0

                        # Check if the array is completely filled
                        if j == 0 and k == y:
                            data = data[::-1, :, :, :]
                            return data.transpose(1, 0, 3, 2)

                    elif img_type == "Secondary electrons concurrent":
                        continue

                    elif img_type == "Secondary electrons survey":
                        continue

                    elif img_type == "Anchor region":
                        continue

                    else:
                        raise ValueError(
                            f"Acquisition '{Acquisition}' is not an "
                            f"AR dataset, found: {read_image_type(Acq)}"
                        )
                else:
                    raise ValueError(
                        f"Acquisition '{Acquisition}' is missing or invalid."
                    )

        data = data[::-1, :, :, :]
        if data.shape[3] == 1 and data.shape[3] == 1: 
            data = data[::-1, :, 0, 0]
        else:
            data = data[::-1, :, :, :].transpose(1, 0, 3, 2)
        
        return data

    except IOError as e:
        raise IOError(f"Failed to load file '{filename}'. Error: {e}")


def load_AR_axes(hdf, acquisition, AR=True):
    """Load the axes information for an angle-resolved dataset."""
    axes = []

    # Load X and Y axes from SE concurrent image
    se_concurrent_acquisition = (
        "Acquisition1"
    )
    SE_Acq = hdf.get(se_concurrent_acquisition)
    if SE_Acq:
        SE_ImgData = SE_Acq.get("ImageData")
        SE_Image = SE_ImgData.get("Image")

        # Ensure SE Image shape is compatible by taking the last two dimensions
        if len(SE_Image.shape) >= 2:
            Y, X = SE_Image.shape[-2:]
        else:
            raise ValueError(
                "SE Image shape is invalid, expected at least 2 dimensions."
            )

        axis_info_se = [
            ("X", "DimensionScaleX", "XOffset"),
            ("Y", "DimensionScaleY", "YOffset"),
        ]

        for i, (axis_name, scale_key, offset_key) in enumerate(axis_info_se):
            Scale = np.array(SE_ImgData.get(scale_key))
            if Scale.ndim == 0:  # Check if Scale is a scalar
                scale_value = float(Scale)
            else:  # Handle Scale as an array
                scale_value = Scale[0]

            prefix = get_unit_prefix(scale_value)

            axis_dict = {
                "name": axis_name,
                "size": SE_Image.shape[
                    -(i + 1)
                ],  # Using the last two dimensions for X and Y
                "offset": float(np.array(SE_ImgData.get(offset_key)))
                * unit_factor(prefix),
                "scale": scale_value * unit_factor(prefix),
                "units": prefix + "m",
                "navigate": True,
            }
            axes.append(axis_dict)

    # Load C and A axes from the angle-resolved image
    ar_acquisition = acquisition
    AR_Acq = hdf.get(ar_acquisition)
    if AR_Acq:
        AR_ImgData = AR_Acq.get("ImageData")
        AR_Image = AR_ImgData.get("Image")

        axis_info_ar = [
            ("C", "DimensionScaleC", "COffset"),  # C axis for Wavelength
            ("A", "DimensionScaleA", "AOffset"),  # A axis for Angle
        ]

        for axis_name, scale_key, offset_key in axis_info_ar:
            ScaleC = np.arange(0, AR_Image.shape[3])
            ScaleA = np.arange(0, AR_Image.shape[4])

            if axis_name == "C":
                axis_dict = {
                    "name": "C",
                    "axis": ScaleC,  # Use full data array for Wavelength
                    "units": "",  # Appropriate units for Wavelength
                    "navigate": False,
                }
            elif axis_name == "A":
                axis_dict = {
                    "name": "Angle",
                    "axis": ScaleA,  # Use full data array for Angle
                    "units": "",  # Appropriate units for Angle
                    "navigate": False,
                }
            axes.append(axis_dict)

    # Reorder axes to X, Y, C, A
    axes_ordered = []
    for axis_name in ["Y", "X", "C", "Angle"]:
        for axis in axes:
            if axis["name"] == axis_name:
                axes_ordered.append(axis)
                break
    return axes_ordered

def load_AR_signal_axes(hdf, acquisition, AR=True):
    """Load the axes information for an angle-resolved dataset."""
    axes = []

    # Load C and A axes from the angle-resolved image
    ar_acquisition = acquisition
    AR_Acq = hdf.get(ar_acquisition)
    if AR_Acq:
        AR_ImgData = AR_Acq.get("ImageData")
        AR_Image = AR_ImgData.get("Image")

        axis_info_ar = [
            ("C", "DimensionScaleC", "COffset"),  # C axis for Wavelength
            ("A", "DimensionScaleA", "AOffset"),  # A axis for Angle
        ]

        for axis_name, scale_key, offset_key in axis_info_ar:
            ScaleC = np.arange(0, AR_Image.shape[3])
            ScaleA = np.arange(0, AR_Image.shape[4])

            if axis_name == "C":
                axis_dict = {
                    "name": "C",
                    "axis": ScaleC,  # Use full data array for Wavelength
                    "units": "",  # Appropriate units for Wavelength
                    "navigate": False,
                }
            elif axis_name == "A":
                axis_dict = {
                    "name": "Angle",
                    "axis": ScaleA,  # Use full data array for Angle
                    "units": "",  # Appropriate units for Angle
                    "navigate": False,
                }
            axes.append(axis_dict)

    # Reorder axes to C, A
    axes_ordered = []
    for axis_name in ["C", "Angle"]:
        for axis in axes:
            if axis["name"] == axis_name:
                axes_ordered.append(axis)
                break
    return axes_ordered

def file_reader(filename, signal=None, lazy=False):
    """
    Read a Delmic hdf5 hyperspectral image.

    Parameters
    ----------
    %s
    %s

    %s
    """
    signal_mapping = {
        "survey": 0,
        "SE": 1,
        "CL": 2,
        }

    if signal is None:
        signal = "CL"

    try:
        with h5py.File(filename, "r") as hdf:
            num_acq = count_acquisitions(
                hdf
            )

            dict_list = []

            AR = is_AR(filename)

            if num_acq > 4 and AR:

                for i in np.arange(0,2):

                    Acquisition = "Acquisition" + str(i)
                    Acq = hdf.get(Acquisition)
                    img_type = read_image_type(Acq)
                    if Acq is None:
                        raise ValueError(
                            f"Acquisition {Acquisition} not found in file."
                        )
        
                    ImgData = Acq.get("ImageData")
                    Image = ImgData.get("Image")
                    
                    if img_type in [
                        "Secondary electrons survey",
                        "Secondary electrons concurrent"
                    ]:
                        data = load_image(Image)
                        axes = make_axes(Acq)
                        metadata = make_metadata(Acq)
                        original_metadata = make_original_metadata(Acq)
                    
                        dictionnary = {
                            "data": data,
                            "axes": axes,
                            "metadata": metadata,
                            "original_metadata": original_metadata,
                        }

                        dict_list.append([dictionnary])

                    else:
                        pass

                data = load_AR(filename)
                
                Acquisition = "Acquisition2"
    
                Acq = hdf.get(Acquisition)
                
                if len(data.shape) < 3 and signal == "CL":
                    axes = load_AR_signal_axes(
                        hdf, Acquisition, AR=True
                    )
                else :
                    axes = load_AR_axes(
                        hdf, Acquisition, AR=True
                    )
                metadata = make_metadata(Acq)
                original_metadata = make_original_AR_metadata(Acq, data)

                dictionnary = {
                    "data": data,
                    "axes": axes,
                    "metadata": metadata,
                    "original_metadata": original_metadata,
                }
    
                dict_list.append([dictionnary])
            
            else:
                for i in np.arange(0,num_acq-1):             
                    
                    Acquisition = "Acquisition" + str(i)
    
                    Acq = hdf.get(Acquisition)
    
                    if Acq is None:
                        raise ValueError(
                            f"Acquisition {Acquisition} not found in file."
                        )
        
                    ImgData = Acq.get("ImageData")
                    Image = ImgData.get("Image")
        
                    dim = np.array(Image.shape)
                    C, T, Z, Y, X = dim
        
                    if dim.sum() < 5:
                        raise ValueError("Image dimensions are too small.")
        
                    if dim.sum() == 5:
                        if C == 1 and T == 1 and Z == 1 and Y == 1 and X == 1:
                            data = load_image(Image)
                            img_type = read_image_type(Acq)
                            axes = make_axes(Acq)
                            metadata = make_metadata(Acq)
                            original_metadata = make_original_metadata(Acq)
                            if img_type == "Secondary electrons survey":
                                pass  # Add additional processing if needed
                            elif img_type == "Secondary electrons concurrent":
                                pass  # Add additional processing if needed
                            else:
                                raise ValueError(
                                    f"Unknown data type: {img_type},"
                                    f"loaded as an image."
                                )
        
                    elif dim.sum() > 5:
                        if AR:
                            data = load_AR(filename)
                            if len(data.shape) < 4 and signal == "CL":
                                axes = load_AR_signal_axes(
                                hdf, Acquisition, AR=True
                            )
                            else :
                                axes = load_AR_axes(
                                    hdf, Acquisition, AR=True
                                )
                            metadata = make_metadata(Acq)
                            original_metadata = make_original_AR_metadata(Acq, data)
                        else:
                            if C > 1:
                                if T > 1:
                                    img_type = read_image_type(Acq)
                                    if img_type == "Temporal Spectrum":
                                        data = load_streak_camera_image(Image)
                                        if len(data.shape) < 3 and signal == "CL":
                                            axes = make_signal_axes(Acq)
                                        else:                 
                                            axes = make_axes(Acq)
                                        metadata = make_metadata(Acq)
                                        original_metadata = make_original_metadata(Acq)
                                    elif img_type == "AR Spectrum":
                                        data = load_AR_spectrum(Image)
                                        if len(data.shape) < 3 and signal == "CL":
                                            axes = make_signal_axes(Acq)
                                        else:                 
                                            axes = make_axes(Acq)
                                        metadata = make_metadata(Acq)
                                        original_metadata = make_original_metadata(Acq)
                                    elif img_type.startswith("Spectrum"):
                                        data = load_hyperspectral(
                                            Image
                                        )
                                        if len(data.shape) < 2 and signal == "CL":
                                            axes = make_signal_axes(Acq)
                                        else:                 
                                            axes = make_axes(Acq)
                                        metadata = make_metadata(Acq)
                                        original_metadata = make_original_metadata(Acq)
                                    else:
                                        raise ValueError(
                                            f"Unknown data type: {img_type},"
                                            f"loaded as a temporal spectrum."
                                        )
                                elif T == 1:
                                    data = load_hyperspectral(Image)
                                    img_type = read_image_type(Acq)
                                    if len(data.shape) < 2 and signal == "CL":
                                        axes = make_signal_axes(Acq)
                                    else:                 
                                        axes = make_axes(Acq)
                                    metadata = make_metadata(Acq)
                                    original_metadata = make_original_metadata(Acq)
                                    if (
                                        img_type.startswith("Spectrum") or
                                        img_type == (
                                            "Large Area Spectrum with Spectrometer"
                                        )
                                    ):
                                        pass  # Add additional processing if needed
                                    else:
                                        raise ValueError(
                                            f"Unknown data type: {img_type},"
                                            f"loaded as a hyperspectral dataset."
                                        )
                            elif C == 1:
                                if T > 1:
                                    data = load_temporal_trace(Image)
                                    img_type = read_image_type(Acq)
                                    if len(data.shape) < 2 and signal == "CL":
                                        axes = make_signal_axes(Acq)
                                    else:                 
                                        axes = make_axes(Acq)
                                    metadata = make_metadata(Acq)
                                    original_metadata = make_original_metadata(Acq)
                                    if img_type == "Time Correlator":
                                        pass  # Add additional processing if needed
                                    else:
                                        raise ValueError(
                                            f"Unknown data type: {img_type},"
                                            f"loaded as a temporal trace."
                                        )
                                elif T == 1:
                                    img_type = read_image_type(Acq)
                                    if img_type == "CL intensity":
                                        data = load_image(Image)
                                        axes = make_axes(Acq)
                                        metadata = make_metadata(Acq)
                                        original_metadata = make_original_metadata(Acq)
                                    elif img_type in [
                                        "Secondary electrons survey",
                                        "Secondary electrons concurrent",
                                    ]:
                                        data = load_image(Image)
                                        axes = make_axes(Acq)
                                        metadata = make_metadata(Acq)
                                        original_metadata = make_original_metadata(Acq)
                                    elif img_type == "Angle-resolved":
                                        raise ValueError(
                                            "Angle-resolved image type detected,"
                                            "use specific AR loading tool."
                                        )
                                    else:
                                        raise ValueError(
                                            f"Unknown data type: {img_type},"
                                            f"loaded as an image."
                                        )
                            elif Z != 1:
                                raise ValueError(
                                    "Invalid format: Z dimension should be 1."
                                )
                            else:
                                raise ValueError("Invalid format.")
        
                    dictionnary = {
                        "data": data,
                        "axes": axes,
                        "metadata": metadata,
                        "original_metadata": original_metadata,
                    }
    
                    dict_list.append([dictionnary])

            if signal == "all":
                return dict_list

            if isinstance(signal, str):
                signal = [signal]
                
            invalid_types = [s for s in signal if s not in signal_mapping]
            if invalid_types:
                raise ValueError(f"Invalid data type(s): {invalid_types}. Valid options are: {list(signal_mapping.keys())}")

            
            selected_data = [dict_list[signal_mapping[s]] for s in signal]

            return selected_data[0] if len(selected_data) == 1 else selected_data
    
    except IOError as e:
        raise IOError(f"Failed to load file '{filename}'. Error: {e}")


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)