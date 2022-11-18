# -*- coding: utf-8 -*-
# Copyright 2022 CEOS GmbH
# Copyright 2022 The HyperSpy developers
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
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/>.


from datetime import datetime as dt
import logging
import os

import numpy as np

from rsciio.utils.tools import DTBox

_logger = logging.getLogger(__name__)


# -----------------------
# File format description
# -----------------------
# The file consists of a compressed numpy file containing a n-dimensional numpy
# array (containing all data) with a dictionary containing all metadata.


def file_reader(filename, **kwds):
    prz_file = np.load(filename, allow_pickle=True)
    data = prz_file["data"]
    meta_data = prz_file["meta_data"][0]
    return import_pr(data, meta_data, filename)


def file_writer(filename, signal, **kwds):
    data, meta_data = export_pr(signal=signal)
    with open(filename, mode="wb") as f:
        # use open file to avoid numpy adding the npz extension
        np.savez_compressed(
            file=f,
            data=data,
            meta_data=[meta_data],
            file_format_version=2,
            data_model=[{}],
        )


def import_pr(data, meta_data, filename=None):
    """Converts metadata from PantaRhei to hyperspy format, and corrects
    the order of data axes if needed.

    Parameters
    ----------
    data: ndarray
        numerical data array, as loaded from file
    meta_data: dict
        dictionary containing the meta data in PantaRhei format
    filename: str or None
        name of the file being loaded

    Returns
    -------
    the list of dictionaries containing the data and metadata in hyperspy format
    """

    data_dimensions = len(data.shape)
    data_type = meta_data.get("type")
    content_type = meta_data.get("content.types")
    calibrations = []
    for axis in range(data_dimensions):
        try:
            calib = meta_data["device.calib"][axis]
        except (IndexError, KeyError):
            calib = None
        calibrations.append(calib)

    if content_type is None:
        if data_type in ("Stack", "3D"):
            content_type = [None, None, "Index"]
        elif data_type == "1D" and data_dimensions == 3:
            content_type = [None, "PlotIndex", "Index"]
        elif data_type == "1D":
            content_type = [None, "PlotIndex"]
        elif data_dimensions >= 3:
            content_type = [
                None,
                None,
            ] + ["Index" for _ in range(data_dimensions - 2)]
        else:
            content_type = [None for _ in range(data_dimensions)]
    elif data_type == "1D":
        if len(content_type) == 1 and data_dimensions == 2:
            content_type = [None, "PlotIndex"]
        elif len(content_type) == 2 and data_dimensions == 3:
            content_type = [None, "PlotIndex", "Index"]
        elif len(content_type) != data_dimensions:
            raise Exception(
                "Content type is not known for all dimensions "
                f"{content_type}, {data.shape}."
            )

    navigation_dimensions = ["ScanY", "ScanX", "Index", "PlotIndex"]
    signal_dimensions = ["CameraY", "CameraX", "Pixel", "Energy", "*"]

    content_type_np_order = content_type[::-1]
    calibrations_np_order = calibrations[::-1]
    if len(content_type_np_order) != data_dimensions:  # pragma: no cover
        raise RuntimeError(
            "Unsupported file, please report the error in the "
            "the HyperSpy issue tracker."
        )

    trivial_indices = [
        i
        for i, _ in enumerate(content_type_np_order)
        if data.shape[i] == 1 and "Index" in content_type_np_order[i]
    ]

    content_type_np_order = [
        c for i, c in enumerate(content_type_np_order) if i not in trivial_indices
    ]
    calibrations_np_order = [
        c for i, c in enumerate(calibrations_np_order) if i not in trivial_indices
    ]

    data = np.squeeze(data, axis=tuple(trivial_indices))

    def _navigation_first(i):
        order = navigation_dimensions + signal_dimensions
        if content_type_np_order[i] in order:
            return order.index(content_type_np_order[i])
        else:
            return len(order)

    new_order = sorted(range(len(content_type_np_order)), key=_navigation_first)
    default_labels = reversed(["X", "Y", "Z"][: content_type_np_order.count(None)])

    data_labels = [
        content_type_np_order[i]
        if content_type_np_order[i] is not None
        else next(default_labels)
        for i in new_order
    ]
    calibration_ordered = [calibrations_np_order[i] for i in new_order]
    data = np.moveaxis(data, new_order, list(range(len(content_type_np_order))))

    # TODO: Will have to be updated once CEOS adds selectable dispersion orientation
    if meta_data.get("filter.mode") == "EELS":
        flip_axis = tuple(i for i, label in enumerate(data_labels) if label == "Energy")
        if flip_axis:
            data = np.flip(data, flip_axis)
    else:
        flip_axis = ()

    for key in [
        "content.types",
        "user.calib",
        "inherited.calib",
        "device.calib",
        "size",
        "ref_size",
    ]:
        if key in meta_data:
            assert isinstance(meta_data[key], (list, tuple))
            if isinstance(meta_data[key], list):
                old_meta_data = meta_data[key].copy()
            else:
                old_meta_data = meta_data[key]
            if len(old_meta_data) == data.ndim + 1:
                item_in_numpy_order = old_meta_data[-2::-1]
            else:
                item_in_numpy_order = old_meta_data[::-1]
            meta_data[key] = []
            for i in range(data.ndim):
                try:
                    meta_data[key].append(item_in_numpy_order[new_order[i]])
                except Exception as e:  # pragma: no cover
                    raise Exception(
                        f"Could not load meta data: {key} " f"in hyperspy file: {e}."
                    )
    axes = []
    for i, (label, calib) in enumerate(zip(data_labels, calibration_ordered)):
        ax = {
            "navigate": label in navigation_dimensions,
            "name": label,
            "size": data.shape[i],
        }
        if calib:
            if "unit" in calib:
                ax["units"] = calib["unit"]
            if "value" in calib:
                ax["offset"] = calib["offset"] * calib["value"]
            else:
                ax["offset"] = calib["offset"]
            if "pixel_factor" in calib:
                ax["scale"] = calib["value"] * calib["pixel_factor"]
            else:
                ax["scale"] = calib["value"]
            if i in flip_axis:
                # TODO: Will have to be updated once CEOS adds selectable dispersion orientation
                new_offset = -(ax["offset"] + (data.shape[i] - 1) * ax["scale"])
                ax["offset"] = new_offset

        axes.append(ax)

    mapped = _metadata_converter_in(meta_data, axes, filename)

    dictionary = {
        "data": data,
        "axes": axes,
        "metadata": mapped.to_dict(),
        "original_metadata": meta_data,
    }
    file_data_list = [
        dictionary,
    ]

    return file_data_list


def export_pr(signal):
    """Extracts from the signal the data array and the metadata in PantaRhei format

    Parameters
    ----------
    signal: BaseSignal
        signal to be exported

    Returns
    -------
    data: ndarray
        numerical data of the signal
    meta_data: dict
        metadata dictionary in PantaRhei format
    """
    data = signal["data"]
    metadata = signal["metadata"]
    original_metadata = signal["original_metadata"]
    axes_info = signal["axes"]
    meta_data = _metadata_converter_out(metadata, original_metadata)
    if "ref_size" not in meta_data:
        meta_data["ref_size"] = data.shape[::-1]

    ref_size = meta_data["ref_size"][::-1]  # switch to numpy order
    pixel_factors = [ref_size[i] / data.shape[i] for i in range(data.ndim)]
    axes_meta_data = get_metadata_from_axes_info(axes_info, pixel_factors=pixel_factors)
    for k in axes_meta_data:
        meta_data[k] = axes_meta_data[k]
    return data, meta_data


def _metadata_converter_in(meta_data, axes, filename):
    mapped = DTBox(box_dots=True)

    signal_dimensions = 0
    for ax in axes:
        if ax["navigate"] == False:
            signal_dimensions += 1

    microscope_base_voltage = meta_data.get("electron_gun.voltage")
    convergence_angle = meta_data.get("condenser.convergence_semi_angle")
    collection_angle = meta_data.get("filter.collection_semi_angle")

    if microscope_base_voltage:
        total_voltage_shift = meta_data.get(
            "filter.ht_offset", meta_data.get("electron_gun.voltage_offset", 0)
        )
        beam_energy_keV = (microscope_base_voltage + total_voltage_shift) / 1000
        mapped.set_item("Acquisition_instrument.TEM.beam_energy", beam_energy_keV)

    if convergence_angle:
        convergence_angle_mrad = convergence_angle * 1e3
        mapped.set_item(
            "Acquisition_instrument.TEM.convergence_angle", convergence_angle_mrad
        )

    if collection_angle:
        collection_angle_mrad = collection_angle * 1e3
        mapped.set_item(
            "Acquisition_instrument.TEM.Detector.EELS.collection_angle",
            collection_angle_mrad,
        )

    if meta_data.get("filter.mode") == "EELS" and signal_dimensions == 1:
        mapped.set_item("Signal.signal_type", "EELS")

    name = meta_data.get("repo_id").split(".")[0]
    mapped.set_item("General.title", name)

    if filename is not None:
        mapped.set_item("General.original_filename", os.path.split(filename)[1])

    if "acquisition.time" in meta_data:
        timestamp = meta_data["acquisition.time"]
    elif "camera.time" in meta_data:
        timestamp = meta_data["camera.time"]
    if "timestamp" in locals():
        timestamp = dt.fromisoformat(timestamp)
        mapped.set_item("General.date", timestamp.date().isoformat())
        mapped.set_item("General.time", timestamp.time().isoformat())

    if "filter.aperture" in meta_data:
        aperture = meta_data["filter.aperture"]
        if "mm" in aperture:
            aperture = aperture.split("mm")[0]
            aperture = aperture.rstrip()
        mapped.set_item(
            "Acquisition_instrument.TEM.Detector.EELS.aperture_size", float(aperture)
        )

    source_type = meta_data.get("source.type")

    if source_type == "scan_generator":
        acquisition_mode = "STEM"
        key = "scan_driver"
    elif source_type == "camera":
        acquisition_mode = "TEM"
        key = "projector"
    else:
        acquisition_mode = None
        key = None
    magnification = meta_data.get(f"{key}.magnification")
    camera_length = meta_data.get("projector.camera_length")

    if acquisition_mode is not None:
        mapped.set_item("Acquisition_instrument.TEM.acquisition_mode", acquisition_mode)
    if magnification is not None:
        mapped.set_item("Acquisition_instrument.TEM.magnification", magnification)
    if camera_length is not None:
        mapped.set_item("Acquisition_instrument.TEM.camera_length", camera_length)

    return mapped


def _metadata_converter_out(metadata, original_metadata=None):
    metadata = DTBox(metadata, box_dots=True)
    original_metadata = DTBox(original_metadata, box_dots=True)
    original_fname = metadata.get("General.original_filename", "")
    original_extension = os.path.splitext(original_fname)[1]
    if original_metadata.get("ref_size"):
        PR_metadata_present = True
    else:
        PR_metadata_present = False

    if original_extension == ".prz" and PR_metadata_present:
        meta_data = original_metadata
        meta_data["ref_size"] = meta_data["ref_size"][::-1]
        for key in ["content.types", "user.calib", "inherited.calib", "device.calib"]:
            if key in meta_data:
                assert isinstance(meta_data[key], (list, tuple))
                if isinstance(meta_data[key], list):
                    old_meta_data = meta_data[key].copy()
                else:
                    old_meta_data = meta_data[key]
                meta_data[key] = old_meta_data[::-1]

    else:
        meta_data = {}
        if metadata.get("Signal.signal_type") == "EELS":
            meta_data["filter.mode"] = "EELS"

        name = metadata.get("General.title")
        if name is not None:
            meta_data["repo_id"] = name + ".0"

        date = metadata.get("General.date")
        time = metadata.get("General.time")
        if date is not None and time is not None:
            timestamp = date + "T" + time
            meta_data["acquisition.time"] = timestamp

        md_TEM = metadata.get("Acquisition_instrument.TEM")
        if md_TEM is not None:
            beam_energy = md_TEM.get("beam_energy")
            convergence_angle = md_TEM.get("convergence_angle")
            collection_angle = md_TEM.get("Detector.EELS.collection_angle")
            aperture = md_TEM.get("Detector.EELS.aperture_size")
            acquisition_mode = md_TEM.get("acquisition_mode")
            magnification = md_TEM.get("magnification")
            camera_length = md_TEM.get("camera_length")

            if aperture is not None:
                if type(aperture) in (float, int):
                    aperture = str(aperture) + " mm"
                meta_data["filter.aperture"] = aperture
            if beam_energy is not None:
                beam_energy_ev = beam_energy * 1e3
                meta_data["electron_gun.voltage"] = beam_energy_ev
            if convergence_angle is not None:
                convergence_angle_rad = convergence_angle / 1e3
                meta_data["condenser.convergence_semi_angle"] = convergence_angle_rad
            if collection_angle is not None:
                collection_angle_rad = collection_angle / 1e3
                meta_data["filter.collection_semi_angle"] = collection_angle_rad
            if camera_length is not None:
                meta_data["projector.camera_length"] = camera_length
            if acquisition_mode == "STEM":
                key = "scan_driver"
                meta_data["source.type"] = "scan_generator"
            else:
                key = "projector"
                meta_data["source.type"] = "camera"
            if magnification is not None:
                meta_data[f"{key}.magnification"] = magnification

    return meta_data


def get_metadata_from_axes_info(axes_info, pixel_factors=None):
    """
    Return a dict with calibration metadata obtained from the passed axes info.

    Parameters
    ----------
    axes_info: list of dict
    A list of dicts containing axis information. The list is sorted by the axis index,
    Each item in the list refers to one axis.

    Returns
    -------
    :param pixel_factors: A list of pixel factors.
        These are similar to binning factors, and are important when re-exporting dataset that where imported from
        PRZ files. They are relevant for Panta Rhei's internal handling of calibrations.

    """
    axis_name_to_content_type = {
        "Energy loss": "Energy",
        "Energy": "Energy",
        "ScanX": "ScanY",
        "ScanY": "ScanX",
    }
    nr_axes = len(axes_info)
    imported_calibs = [None] * nr_axes
    content_types = [None] * nr_axes
    navigate_axes = [None] * nr_axes
    for i in range(nr_axes):
        # Add content types if axes names are known.
        axis_info = axes_info[i]
        axis_label = None
        if "name" in axis_info:  # name is not always present
            axis_label = axis_info["name"]

        if axis_label in axis_name_to_content_type:
            content_types[i] = axis_name_to_content_type[axis_label]

        imported_calib_dict = {"scale": None, "offset": None, "units": None}
        for key in ("scale", "offset", "units"):
            if key in axis_info:
                imported_calib_dict[key] = axis_info[key]
        # If any part of a calibration is given
        # -> Create a default calibration and
        # set all available information.
        if any([v for k, v in imported_calib_dict.items()]):
            calib = {}

            if imported_calib_dict["scale"] is not None:
                calib["value"] = imported_calib_dict["scale"]
                # Apply pixel factor as calculated from meta data.
                if pixel_factors:
                    calib["value"] /= pixel_factors[i]
            if imported_calib_dict["offset"] is not None:
                calib["offset"] = imported_calib_dict["offset"]

                # PR expects offset in image pixels
                # not in calibrated values.
                calib["offset"] = calib["offset"] / calib["value"]
            if imported_calib_dict["units"] is not None:
                imported_unit = imported_calib_dict["units"]
                # unit may be in SI unit *with* prefix
                allowed_base_units = ["m", "A", "V", "rad", "s", "eV"]
                calib["value"], calib["unit"] = _guess_from_unit(
                    calib["value"], imported_unit, allowed_base_units=allowed_base_units
                )
                if calib["unit"] in allowed_base_units:
                    calib["use_prefix"] = True

            imported_calibs[i] = calib  # ['as_dict()
        # Get information whether axis is navigate axis
        # (which means 'display axis' in our terms).
        if "navigate" in axis_info:
            navigate_axes[i] = axis_info["navigate"]
        display_axes = [i for i, is_display in enumerate(navigate_axes) if is_display]
    axes_meta_data = {}
    # Calibrations and content types must be in reversed order,
    # because hyperspy uses numpy order,
    # while PR expects image order.
    if any(imported_calibs):
        axes_meta_data["inherited.calib"] = imported_calibs[::-1]
    if any(content_types):
        axes_meta_data["content.types"] = content_types[::-1]
    else:
        if len([nav for nav in navigate_axes if not nav]) == 1:
            axes_meta_data["type"] = "1D"

    if display_axes:
        # Only add display axes tag, if not all axes are displayed
        # (which means that data is 3D or 4D data).
        # If only one display axis is defined, ignore it,
        # because using plots as navigation tool for cubes
        # is currently not supported.
        if nr_axes > len(display_axes) and len(display_axes) > 1:
            axes_meta_data["display_axes"] = tuple(display_axes[::-1])

    return axes_meta_data


def _guess_from_unit(scale, unit, allowed_base_units=None):
    """Guess the base unit according to the passed unit (with possible prefix).

    Parameters
    ----------
    scale: float
        the calibration value as given by the imported format
    unit: str
        the calibration unit as given by the imported format.
        May start with a unit prefix (like 'm', 'u', etc.)
    allowed_base_units: list
        An optional list of allowed base units. If None is passed, all units
        that start with 'm', 'n', 'p' or 'u' are assumed to be units with prefixes.

    Returns
    -------
    scale: float
    the calibration value scaled with the prefix-factor.
    unit: str
    the base unit without the prefix

    """
    prefixes = {"m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "p": 1e-12}
    if isinstance(unit, str) and len(unit) > 1 and unit[0] in prefixes:
        if allowed_base_units is None or (unit[1:] in allowed_base_units):
            scale *= prefixes[unit[0]]
            unit = unit[1:]
    return scale, unit
