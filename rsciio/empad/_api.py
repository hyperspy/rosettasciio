# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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

import ast
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np

from rsciio._docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC
from rsciio.utils.xml import convert_xml_to_dict

_logger = logging.getLogger(__name__)


def _read_raw(info, fp, lazy=False):
    raw_height = info["raw_height"]
    width = info["width"]
    height = info["height"]

    if lazy:
        data = np.memmap(fp, dtype="<f4", mode="r")
    else:
        data = np.fromfile(fp, dtype="<f4")

    if "series_count" in info.keys():  # stack of images
        size = (info["series_count"], raw_height, width)
        data = data.reshape(size)[..., :height, :]

    else:  # 2D x 2D
        size = (info["scan_y"], info["scan_x"], raw_height, width)
        data = data.reshape(size)[..., :height, :]
    return data


def _parse_xml(filename):
    tree = ET.parse(filename)
    om = convert_xml_to_dict(tree.getroot())

    info = {
        "raw_filename": om.root.raw_file.filename,
        "width": 128,
        "height": 128,
        "raw_height": 130,
    }
    if om.has_item("root.scan_parameters.series_count"):
        # Stack of images
        info.update({"series_count": int(om.root.scan_parameters.series_count)})
    # The pix_x and pix_y are no longer used in the new EMPAD versions, but keep this line for backward compatibility
    elif om.has_item("root.pix_x") and om.has_item("root.pix_y"):
        # 2D x 2D
        info.update({"scan_x": int(om.root.pix_x), "scan_y": int(om.root.pix_y)})
    # in case root.pix_x and root.pix_y are not available
    elif om.has_item("root.scan_parameters.scan_resolution_x") and om.has_item(
        "root.scan_parameters.scan_resolution_y"
    ):
        info.update(
            {
                "scan_x": int(om.root.scan_parameters.scan_resolution_x),
                "scan_y": int(om.root.scan_parameters.scan_resolution_y),
            }
        )
    else:
        raise IOError("Unsupported Empad file: the scan parameters cannot be imported.")

    return om, info


def _convert_scale_units(value, units, factor=1):
    from rsciio.utils._units import _UREG

    v = float(value) * _UREG(units)
    converted_v = (factor * v).to_compact()
    converted_value = converted_v.magnitude / factor
    converted_units = "{:~}".format(converted_v.units)

    return converted_value, converted_units


def file_reader(filename, lazy=False, q_calibration=None, remove_nans=False):
    """
    Read file format used by the Electron Microscope Pixel Array Detector (EMPAD).

    Parameters
    ----------
    %s
    %s
    q_calibration : None or float, optional
        Specifies the calibration for the diffraction patterns in 1/nm.
        In EMPAD version v1.2.2 (default version shipped with most of the TFS TEMs),
        diffraction space calibration is no longer stored in the XML.
        Use this option to calibrate the diffraction patterns in 1/nm.
        If None, the data will be in pixel scales.
    remove_nans : bool, optional
        Sometimes the EMPAD data contains NaN values that cause issues in
        downstream processing. If True, these NaN values will be replaced
        with zeros. Default is False.

    %s

    Notes
    -----
    For EMPAD file v1.2.0+, the diffraction space calibration is no longer
    stored in the XML. The ``q_calibration`` parameter can be used to set
    the diffraction space calibration in 1/nm.

    Examples
    --------
    >>> from rsciio.empad import file_reader
    >>> s = file_reader("empad_file.xml", q_calibration=0.3315)
    """
    om, info = _parse_xml(filename)
    dname, fname = os.path.split(filename)

    md = {
        "General": {
            "original_filename": fname,
            "title": os.path.splitext(fname)[0],
        },
        "Signal": {"signal_type": "electron_diffraction"},
    }

    if om.has_item("root.timestamp.isoformat"):
        date, time = om.root.timestamp.isoformat.split("T")
        md["General"].update({"date": date, "time": time})

    units = [
        None,
    ] * 2
    scales = [
        1,
    ] * 2
    origins = [
        -64,
    ] * 2
    axes = []
    index_in_array = 0
    names = ["height", "width"]
    navigate = [False, False]

    if "series_count" in info.keys():
        names = ["series_count"] + names
        units.insert(0, "ms")
        scales.insert(0, 1)
        origins.insert(0, 0)
        navigate.insert(0, True)
    else:
        names = ["scan_y", "scan_x"] + names
        units.insert(0, "")
        units.insert(0, "")
        scales.insert(0, 1)
        scales.insert(0, 1)
        origins.insert(0, 0)
        origins.insert(0, 0)
        navigate.insert(0, True)
        navigate.insert(0, True)

    sizes = [info[name] for name in names]

    if "series_count" not in info.keys():
        # Try to read the scan size from the xml
        if om.has_item("root.iom_measurements.optics.get_full_scan_field_of_view"):
            # Keep this for backward compatibility
            fov = ast.literal_eval(
                om.root.iom_measurements.optics.get_full_scan_field_of_view
            )
            for i in range(2):
                value = fov[i] / sizes[i]
                scales[i], units[i] = _convert_scale_units(value, "m", sizes[i])

        elif om.has_item(
            "root.iom_measurements.full_scan_field_of_view.x"
        ) and om.has_item("root.iom_measurements.full_scan_field_of_view.scale_factor"):
            # The fov is stored under iom_measurements.full_scan_field_of_view.x and y,
            # but x and y are always the same, representing the larger dimension of the scan
            fov = ast.literal_eval(om.root.iom_measurements.full_scan_field_of_view.x)
            # The scale factor is to match the EMPAD fov to the internal scan, usually 0.72.
            # The fov is mistakenly stored as the internal fov * scale_factor
            scale_factor = ast.literal_eval(
                om.root.iom_measurements.full_scan_field_of_view.scale_factor
            )

            # Again, the fov is for the larger dimension of the scan
            n_pixels = max(sizes[0], sizes[1])
            value = fov / scale_factor / n_pixels
            scale, unit = _convert_scale_units(value, "m")
            for i in [0, 1]:
                scales[i] = scale
                units[i] = unit

        else:  # pragma: no cover
            _logger.warning("The scale of the navigation axes cannot be read.")

    # Try to set the pixel size for the diffraction patterns.
    # In newer EMPAD versions, the diffraction scale is no longer recorded in the xml. We provide the option to set the scale by providing a q_calibration value.
    if q_calibration is not None:
        dp_pixel_size = q_calibration
        dp_unit = "1/nm"
    # If q_calibration is not provided, check if the calibrated_pixelsize is available in the xml for backward compatibility.
    elif om.has_item("root.iom_measurements.calibrated_pixelsize"):
        dp_pixel_size = float(om.root.iom_measurements.calibrated_pixelsize) * 1e9
        dp_unit = "1/nm"
    else:
        # Fall back to pixel units if the diffraction scale cannot be read from the xml and q_calibration is not provided.
        dp_pixel_size = 1
        dp_unit = None
        _logger.info(
            "The scale of the diffraction axes is not set. The calibration can be set using the `q_calibration` parameter."
        )
    # Write the scale and unit for the axes.
    for i in [-1, -2]:
        scales[i] = dp_pixel_size
        units[i] = dp_unit
    for i in range(len(names)):
        if sizes[i] > 1:
            axes.append(
                {
                    "size": sizes[i],
                    "index_in_array": index_in_array,
                    "name": names[i],
                    "scale": scales[i],
                    "offset": origins[i] * scales[i],
                    "units": units[i],
                    "navigate": navigate[i],
                }
            )
            index_in_array += 1

    data = _read_raw(info, os.path.join(dname, info["raw_filename"]), lazy=lazy)

    # Remove NaN values if requested. Will add a few seconds for the loading time.
    if remove_nans:
        data = np.nan_to_num(data)

    dictionary = {
        "data": data.squeeze(),
        "axes": axes,
        "metadata": md,
        "original_metadata": om.to_dict(),
    }

    return [
        dictionary,
    ]


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)
