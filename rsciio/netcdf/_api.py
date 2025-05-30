# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
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

import logging
import os

import numpy as np

from rsciio._docstrings import FILENAME_DOC, LAZY_UNSUPPORTED_DOC, RETURNS_DOC

_logger = logging.getLogger(__name__)


try:
    from netCDF4 import Dataset as netcdf_file_reader

    netcdf_reader = "netCDF4"
except Exception:
    try:
        from scipy.io import netcdf_file as netcdf_file_reader

        netcdf_reader = "scipy"
    except Exception:
        netcdf_reader = None


attrib2netcdf = {
    "energyorigin": "energy_origin",
    "energyscale": "energy_scale",
    "energyunits": "energy_units",
    "xorigin": "x_origin",
    "xscale": "x_scale",
    "xunits": "x_units",
    "yorigin": "y_origin",
    "yscale": "y_scale",
    "yunits": "y_units",
    "zorigin": "z_origin",
    "zscale": "z_scale",
    "zunits": "z_units",
    "exposure": "exposure",
    "title": "title",
    "binning": "binning",
    "readout_frequency": "readout_frequency",
    "ccd_height": "ccd_height",
    "blanking": "blanking",
}

acquisition2netcdf = {
    "exposure": "exposure",
    "binning": "binning",
    "readout_frequency": "readout_frequency",
    "ccd_height": "ccd_height",
    "blanking": "blanking",
    "gain": "gain",
    "pppc": "pppc",
}

treatments2netcdf = {
    "dark_current": "dark_current",
    "readout": "readout",
}


def file_reader(filename, lazy=False):
    """
    Read netCDF ``.nc`` files saved using the HyperSpy predecessor EELSlab.

    Parameters
    ----------
    %s
    %s

    %s
    """
    if netcdf_reader is None:
        raise ImportError(
            "No netCDF library installed. "
            "To read EELSLab netcdf files install "
            "one of the following packages:"
            "netCDF4 or scipy."
        )

    if lazy is not False:
        raise NotImplementedError("Lazy loading is not supported.")

    ncfile = netcdf_file_reader(filename, "r")

    if (
        hasattr(ncfile, "file_format_version")
        and ncfile.file_format_version == "EELSLab 0.1"
    ):
        dictionary = nc_hyperspy_reader_0dot1(ncfile, filename)
    else:
        ncfile.close()
        raise IOError("Unsupported netCDF file")

    return (dictionary,)


file_reader.__doc__ %= (FILENAME_DOC, LAZY_UNSUPPORTED_DOC, RETURNS_DOC)


def nc_hyperspy_reader_0dot1(ncfile, filename):
    calibration_dict, acquisition_dict, treatments_dict = {}, {}, {}
    dc = ncfile.variables["data_cube"]
    data = dc[:]
    if "history" in calibration_dict:
        calibration_dict["history"] = eval(ncfile.history)
    for attrib in attrib2netcdf.items():
        if hasattr(dc, attrib[1]):
            value = eval("dc." + attrib[1])
            if isinstance(value, np.ndarray):
                calibration_dict[attrib[0]] = value[0]
            else:
                calibration_dict[attrib[0]] = value
        else:
            _logger.warning(
                "Warning: the attribute '%s' is not defined in the file '%s'",
                attrib[0],
                filename,
            )
    for attrib in acquisition2netcdf.items():
        if hasattr(dc, attrib[1]):
            value = eval("dc." + attrib[1])
            if isinstance(value, np.ndarray):
                acquisition_dict[attrib[0]] = value[0]
            else:
                acquisition_dict[attrib[0]] = value
        else:
            _logger.warning(
                "Warning: the attribute '%s' is not defined in the file '%s'",
                attrib[0],
                filename,
            )
    for attrib in treatments2netcdf.items():
        if hasattr(dc, attrib[1]):
            treatments_dict[attrib[0]] = eval("dc." + attrib[1])
        else:
            _logger.warning(
                "Warning: the attribute '%s' is not defined in the file '%s'",
                attrib[0],
                filename,
            )
    original_metadata = {
        "record_by": ncfile.type,
        "calibration": calibration_dict,
        "acquisition": acquisition_dict,
        "treatments": treatments_dict,
    }
    ncfile.close()
    # Now we'll map some parameters
    record_by = "image" if original_metadata["record_by"] == "image" else "spectrum"
    if record_by == "image":
        dim = len(data.shape)
        names = ["Z", "Y", "X"][3 - dim :]
        scaleskeys = ["zscale", "yscale", "xscale"]
        originskeys = ["zorigin", "yorigin", "xorigin"]
        unitskeys = ["zunits", "yunits", "xunits"]
        navigate = [True, False, False]

    elif record_by == "spectrum":
        dim = len(data.shape)
        names = ["Y", "X", "Energy"][3 - dim :]
        scaleskeys = ["yscale", "xscale", "energyscale"]
        originskeys = ["yorigin", "xorigin", "energyorigin"]
        unitskeys = ["yunits", "xunits", "energyunits"]
        navigate = [True, True, False]

    # The images are recorded in the Fortran order
    data = data.T.copy()
    try:
        scales = [calibration_dict[key] for key in scaleskeys[3 - dim :]]
    except KeyError:
        scales = [1, 1, 1][3 - dim :]
    try:
        origins = [calibration_dict[key] for key in originskeys[3 - dim :]]
    except KeyError:
        origins = [0, 0, 0][3 - dim :]
    try:
        units = [calibration_dict[key] for key in unitskeys[3 - dim :]]
    except KeyError:
        units = ["", "", ""]
    axes = [
        {
            "size": int(data.shape[i]),
            "index_in_array": i,
            "name": names[i],
            "scale": scales[i],
            "offset": origins[i],
            "units": units[i],
            "navigate": navigate[i],
        }
        for i in range(dim)
    ]
    metadata = {"General": {}, "Signal": {}}
    metadata["General"]["original_filename"] = os.path.split(filename)[1]
    metadata["General"]["signal_type"] = ""
    dictionary = {
        "data": data,
        "axes": axes,
        "metadata": metadata,
        "original_metadata": original_metadata,
    }

    return dictionary
