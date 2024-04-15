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

#  for more information on the RPL/RAW format, see
#  https://www.nist.gov/services-resources/software/lispix
#  and
#  https://www.nist.gov/services-resources/software/lispixdoc/image-file-formats/raw-file-format.htm

import codecs
import logging
import os.path
from io import StringIO

import numpy as np

from rsciio import __version__
from rsciio._docstrings import (
    ENCODING_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    MMAP_DOC,
    RETURNS_DOC,
    SIGNAL_DOC,
)
from rsciio.utils.tools import DTBox

_logger = logging.getLogger(__name__)


file_extensions = ("rpl", "RPL")
# The format only support the followng data types
newline = ("\n", "\r\n")
comment = ";"
sep = "\t"

dtype2keys = {
    "float64": ("float", 8),
    "float32": ("float", 4),
    "uint8": ("unsigned", 1),
    "uint16": ("unsigned", 2),
    "int32": ("signed", 4),
    "int64": ("signed", 8),
}

endianess2rpl = {"=": "dont-care", "<": "little-endian", ">": "big-endian"}

# Warning: for selection lists use tuples not lists.
rpl_keys = {
    # spectrum/image keys
    "width": int,
    "height": int,
    "depth": int,
    "offset": int,
    "data-length": ("1", "2", "4", "8"),
    "data-type": ("signed", "unsigned", "float"),
    "byte-order": ("little-endian", "big-endian", "dont-care"),
    "record-by": ("image", "vector", "dont-care"),
    # X-ray keys
    "ev-per-chan": float,  # usually 5 or 10 eV
    "detector-peak-width-ev": float,  # usually 150 eV
    # HyperSpy-specific keys
    "depth-origin": float,
    "depth-scale": float,
    "depth-units": str,
    "width-origin": float,
    "width-scale": float,
    "width-units": str,
    "height-origin": float,
    "height-scale": float,
    "height-units": str,
    "signal": str,
    # EELS HyperSpy keys
    "collection-angle": float,
    # TEM HyperSpy keys
    "convergence-angle": float,
    "beam-energy": float,
    # EDS HyperSpy keys
    "elevation-angle": float,
    "azimuth-angle": float,
    "live-time": float,
    # From 0.8.5 energy-resolution is deprecated as it is a duplicate of
    # detector-peak-width-ev of the ripple standard format. We keep it here
    # to keep compatibility with rpl file written by HyperSpy < 0.8.4
    "energy-resolution": float,
    "tilt-stage": float,
    "date": str,
    "time": str,
    "title": str,
}


def correct_INCA_format(fp):
    fp_list = list()
    fp.seek(0)
    if "(" in fp.readline():
        for line in fp:
            line = (
                line.replace("(MLX::", "")
                .replace(" : ", "\t")
                .replace(" :", "\t")
                .replace(" ", "\t")
                .lower()
                .strip()
                .replace(")", "\n")
            )
            if "record-by" in line:
                if "image" in line:
                    line = "record-by\timage"
                if "vector" in line:
                    line = "record-by\tvector"
                if "dont-care" in line:
                    line = "record-by\tdont-care"
            fp_list.append(line)
        fp = StringIO()
        fp.writelines(fp_list)
    fp.seek(0)
    return fp


def parse_ripple(fp):
    """
    Parse information from ripple (.rpl) file.
    Accepts file object 'fp. Returns dictionary rpl_info.
    """

    fp = correct_INCA_format(fp)

    rpl_info = {}
    for line in fp.readlines():
        # correct_brucker_format
        line = line.replace("data-Length", "data-length")
        if line[:2] not in newline and line[0] != comment:
            line = line.strip("\r\n")
            if comment in line:
                line = line[: line.find(comment)]
            if sep not in line:
                err = 'Separator in line "%s" is wrong, ' % line
                err += 'it should be a <TAB> ("\\t")'
                raise IOError(err)
            line = [seg.strip() for seg in line.split(sep)]  # now it's a list
            if (line[0] in rpl_keys) is True:
                value_type = rpl_keys[line[0]]
                if isinstance(value_type, tuple):  # is selection list
                    if line[1] not in value_type:
                        err = (
                            "Wrong value for key %s.\n"
                            "Value read is %s"
                            " but it should be one of %s"
                            % (line[0], line[1], str(value_type))
                        )
                        raise IOError(err)
                else:
                    # rpl_keys[line[0]] must then be a type
                    line[1] = value_type(line[1])

            rpl_info[line[0]] = line[1]

    if rpl_info["depth"] == 1 and rpl_info["record-by"] != "dont-care":
        err = '"depth" and "record-by" keys mismatch.\n'
        err += '"depth" cannot be "1" if "record-by" is "dont-care" '
        err += "and vice versa."
        err += "Check %s" % fp.name
        raise IOError(err)
    if rpl_info["data-type"] == "float" and int(rpl_info["data-length"]) < 4:
        err = '"data-length" for float "data-type" must be "4" or "8".\n'
        err += "Check %s" % fp.name
        raise IOError(err)
    if rpl_info["data-length"] == "1" and rpl_info["byte-order"] != "dont-care":
        err = '"data-length" and "byte-order" mismatch.\n'
        err += '"data-length" cannot be "1" if "byte-order" is not "dont-care"'
        err += " and vice versa."
        err += "Check %s" % fp.name
        raise IOError(err)
    return rpl_info


def read_raw(rpl_info, filename, mmap_mode="c"):
    """Read the raw file object 'fp' based on the information given in the
    'rpl_info' dictionary.

    Parameters
    ----------
    rpl_info : dict
        A dictionary containing the keywords as parsed by ``read_rpl``.
    filename : str
        The filename of the raw file.
    mmap_mode : str, default='c'
        The mmap_mode to use to read the file.
    """
    width = rpl_info["width"]
    height = rpl_info["height"]
    depth = rpl_info["depth"]
    offset = rpl_info["offset"]
    data_length = rpl_info["data-length"]
    data_type = rpl_info["data-type"]
    endian = rpl_info["byte-order"]
    record_by = rpl_info["record-by"]

    if data_type == "signed":
        data_type = "int"
    elif data_type == "unsigned":
        data_type = "uint"
    elif data_type == "float":
        pass
    else:
        raise TypeError('Unknown "data-type" string.')

    if endian == "big-endian":
        endian = ">"
    elif endian == "little-endian":
        endian = "<"
    else:
        endian = "="

    data_type += str(int(data_length) * 8)
    data_type = np.dtype(data_type)
    data_type = data_type.newbyteorder(endian)

    data = np.memmap(filename, offset=offset, dtype=data_type, mode=mmap_mode)

    if record_by == "vector":  # spectral image
        size = (height, width, depth)
        data = data.reshape(size)
    elif record_by == "image":  # stack of images
        size = (depth, height, width)
        data = data.reshape(size)
    elif record_by == "dont-care":  # stack of images
        size = (height, width)
        data = data.reshape(size)
    return data


def file_reader(
    filename, lazy=False, rpl_info=None, encoding="latin-1", mmap_mode=None
):
    """
    Read a ripple/raw file.
    Parse a lispix (https://www.nist.gov/services-resources/software/lispix)
    ripple (.rpl) file and reads the data from the corresponding raw (.raw) file;
    or, read a raw file if the dictionary ``rpl_info`` is provided.

    Parameters
    ----------
    %s
    %s
    rpl_info : dict, Default=None
        A dictionary containing the keywords in order to read a ``.raw`` file
        without corresponding ``.rpl`` file. If ``None``, the keywords are parsed
        automatically from the ``.rpl`` file.
    %s
    %s

    %s
    """
    if not rpl_info:
        if filename[-3:] in file_extensions:
            with codecs.open(filename, encoding=encoding, errors="replace") as f:
                rpl_info = parse_ripple(f)
        else:
            raise IOError('File has wrong extension: "%s"' % filename[-3:])

    for ext in ["raw", "RAW"]:
        rawfname = filename[:-3] + ext
        if os.path.exists(rawfname):
            break
        else:
            rawfname = ""
    if not rawfname:
        raise IOError(f'RAW file "{rawfname}" does not exists')

    if lazy:
        mmap_mode = "r"
    else:
        mmap_mode = "c"
    data = read_raw(rpl_info, rawfname, mmap_mode=mmap_mode)

    if rpl_info["record-by"] == "vector":
        _logger.info("Loading as Signal1D")
        navigate = [True, True, False]
    elif rpl_info["record-by"] == "image":
        _logger.info("Loading as Signal2D")
        navigate = [True, False, False]
    else:
        if len(data.shape) == 1:
            _logger.info("Loading as Signal1D")
            navigate = [True, True, False]
        else:
            _logger.info("Loading as Signal2D")
            navigate = [True, False, False]

    if rpl_info["record-by"] == "vector":
        idepth, iheight, iwidth = 2, 0, 1
        names = [
            "height",
            "width",
            "depth",
        ]
    else:
        idepth, iheight, iwidth = 0, 1, 2
        names = ["depth", "height", "width"]

    scales = [1, 1, 1]
    origins = [0, 0, 0]
    units = ["", "", ""]
    sizes = [rpl_info[names[i]] for i in range(3)]

    if "date" not in rpl_info:
        rpl_info["date"] = ""

    if "time" not in rpl_info:
        rpl_info["time"] = ""

    if "signal" not in rpl_info:
        rpl_info["signal"] = ""

    if "title" not in rpl_info:
        rpl_info["title"] = ""

    if "depth-scale" in rpl_info:
        scales[idepth] = rpl_info["depth-scale"]
    # ev-per-chan is the only calibration supported by the original ripple
    # format
    elif "ev-per-chan" in rpl_info:
        scales[idepth] = rpl_info["ev-per-chan"]

    if "depth-origin" in rpl_info:
        origins[idepth] = rpl_info["depth-origin"]

    if "depth-units" in rpl_info:
        units[idepth] = rpl_info["depth-units"]

    if "depth-name" in rpl_info:
        names[idepth] = rpl_info["depth-name"]

    if "width-origin" in rpl_info:
        origins[iwidth] = rpl_info["width-origin"]

    if "width-scale" in rpl_info:
        scales[iwidth] = rpl_info["width-scale"]

    if "width-units" in rpl_info:
        units[iwidth] = rpl_info["width-units"]

    if "width-name" in rpl_info:
        names[iwidth] = rpl_info["width-name"]

    if "height-origin" in rpl_info:
        origins[iheight] = rpl_info["height-origin"]

    if "height-scale" in rpl_info:
        scales[iheight] = rpl_info["height-scale"]

    if "height-units" in rpl_info:
        units[iheight] = rpl_info["height-units"]

    if "height-name" in rpl_info:
        names[iheight] = rpl_info["height-name"]

    mp = DTBox(
        {
            "General": {
                "original_filename": os.path.split(filename)[1],
                "date": rpl_info["date"],
                "time": rpl_info["time"],
                "title": rpl_info["title"],
            },
            "Signal": {"signal_type": rpl_info["signal"]},
        },
        box_dots=True,
    )
    if "convergence-angle" in rpl_info:
        mp.set_item(
            "Acquisition_instrument.TEM.convergence_angle",
            rpl_info["convergence-angle"],
        )
    if "tilt-stage" in rpl_info:
        mp.set_item(
            "Acquisition_instrument.TEM.Stage.tilt_alpha", rpl_info["tilt-stage"]
        )
    if "collection-angle" in rpl_info:
        mp.set_item(
            "Acquisition_instrument.TEM.Detector.EELS." + "collection_angle",
            rpl_info["collection-angle"],
        )
    if "beam-energy" in rpl_info:
        mp.set_item("Acquisition_instrument.TEM.beam_energy", rpl_info["beam-energy"])
    if "elevation-angle" in rpl_info:
        mp.set_item(
            "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
            rpl_info["elevation-angle"],
        )
    if "azimuth-angle" in rpl_info:
        mp.set_item(
            "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
            rpl_info["azimuth-angle"],
        )
    if "energy-resolution" in rpl_info:
        mp.set_item(
            "Acquisition_instrument.TEM.Detector.EDS." + "energy_resolution_MnKa",
            rpl_info["energy-resolution"],
        )
    if "detector-peak-width-ev" in rpl_info:
        mp.set_item(
            "Acquisition_instrument.TEM.Detector.EDS." + "energy_resolution_MnKa",
            rpl_info["detector-peak-width-ev"],
        )
    if "live-time" in rpl_info:
        mp.set_item(
            "Acquisition_instrument.TEM.Detector.EDS.live_time", rpl_info["live-time"]
        )

    units = [None if unit == "<undefined>" else unit for unit in units]

    axes = []
    index_in_array = 0
    for i in range(3):
        if sizes[i] > 1:
            axes.append(
                {
                    "size": sizes[i],
                    "index_in_array": index_in_array,
                    "name": names[i],
                    "scale": scales[i],
                    "offset": origins[i],
                    "units": units[i],
                    "navigate": navigate[i],
                }
            )
            index_in_array += 1

    dictionary = {
        "data": data.squeeze(),
        "axes": axes,
        "metadata": mp.to_dict(),
        "original_metadata": rpl_info,
    }
    return [
        dictionary,
    ]


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, ENCODING_DOC, MMAP_DOC, RETURNS_DOC)


def file_writer(filename, signal, encoding="latin-1"):
    """
    Write a ripple/raw file.
    Write a Lispix (https://www.nist.gov/services-resources/software/lispix)
    ripple (.rpl) file and saves the data in a corresponding raw (.raw) file.

    Parameters
    ----------
    %s
    %s
    %s
    """
    # Set the optional keys to None
    ev_per_chan = None

    # Check if the dtype is supported
    dc = signal["data"]
    md = DTBox(signal["metadata"], box_dots=True)
    dtype_name = dc.dtype.name
    if dtype_name not in dtype2keys.keys():
        supported_dtype = ", ".join(dtype2keys.keys())
        raise IOError(
            f"The ripple format does not support writing data of {dtype_name} type. "
            f"Supported data types are: {supported_dtype}."
        )
    # Check if the dimensions are supported
    dimension = len(dc.shape)
    if dimension > 3:
        raise IOError(f"This file format does not support {dimension} dimension data")

    # Gather the information to write the rpl
    data_type, data_length = dtype2keys[dtype_name]
    byte_order = endianess2rpl[dc.dtype.byteorder.replace("|", "=")]
    offset = 0

    signal_type = md.get("Signal.signal_type", "")
    date = md.get("General.date", "")
    time = md.get("General.time", "")
    title = md.get("General.title", "")

    sig_axes = [ax for ax in signal["axes"] if not ax["navigate"]][::-1]
    nav_axes = [ax for ax in signal["axes"] if ax["navigate"]][::-1]

    if len(sig_axes) == 1:
        record_by = "vector"
        depth_axis = sig_axes[0]
        ev_per_chan = int(round(depth_axis["scale"]))
        if dimension == 3:
            width_axis, height_axis = nav_axes
            depth, width, height = (
                depth_axis["size"],
                width_axis["size"],
                height_axis["size"],
            )
        elif dimension == 2:
            width_axis = nav_axes[0]
            depth, width, height = depth_axis["size"], width_axis["size"], 1
        elif dimension == 1:
            depth, width, height = depth_axis["size"], 1, 1

    elif len(sig_axes) == 2:
        width_axis, height_axis = sig_axes
        if dimension == 3:
            depth_axis = nav_axes[0]
            record_by = "image"
            depth, width, height = (
                depth_axis["size"],
                width_axis["size"],
                height_axis["size"],
            )
        elif dimension == 2:
            record_by = "dont-care"
            width, height, depth = width_axis["size"], height_axis["size"], 1
        elif dimension == 1:
            record_by = "dont-care"
            depth, width, height = width_axis["size"], 1, 1
    else:
        _logger.info("Only Signal1D and Signal2D objects can be saved")
        return

    # Fill the keys dictionary
    keys_dictionary = {
        "width": width,
        "height": height,
        "depth": depth,
        "offset": offset,
        "data-type": data_type,
        "data-length": data_length,
        "byte-order": byte_order,
        "record-by": record_by,
        "signal": signal_type,
        "date": date,
        "time": time,
        "title": title,
    }
    if ev_per_chan is not None:
        keys_dictionary["ev-per-chan"] = ev_per_chan
    keys = ["depth", "height", "width"]
    for key in keys:
        if eval(key) > 1:
            keys_dictionary[f"{key}-scale"] = eval(f"{key}_axis['scale']")
            keys_dictionary[f"{key}-origin"] = eval(f"{key}_axis['offset']")
            keys_dictionary[f"{key}-units"] = eval(f"{key}_axis['units']")
            keys_dictionary[f"{key}-name"] = eval(f"{key}_axis['name']")
    if signal_type == "EELS":
        mp = md.get("Acquisition_instrument.TEM")
        if mp is not None:
            if "beam_energy" in mp:
                keys_dictionary["beam-energy"] = mp.beam_energy
            if "convergence_angle" in mp:
                keys_dictionary["convergence-angle"] = mp.convergence_angle
            if "Detector.EELS.collection_angle" in mp:
                keys_dictionary["collection-angle"] = mp.Detector.EELS.collection_angle
    if "EDS" in signal_type:
        if signal_type == "EDS_SEM":
            mp = md.Acquisition_instrument.SEM
        elif signal_type == "EDS_TEM":
            mp = md.Acquisition_instrument.TEM
        if "beam_energy" in mp:
            keys_dictionary["beam-energy"] = mp.beam_energy
        if "Detector.EDS.elevation_angle" in mp:
            keys_dictionary["elevation-angle"] = mp.Detector.EDS.elevation_angle
        if "Stage.tilt_alpha" in mp:
            keys_dictionary["tilt-stage"] = mp.Stage.tilt_alpha
        if "Detector.EDS.azimuth_angle" in mp:
            keys_dictionary["azimuth-angle"] = mp.Detector.EDS.azimuth_angle
        if "Detector.EDS.live_time" in mp:
            keys_dictionary["live-time"] = mp.Detector.EDS.live_time
        if "Detector.EDS.energy_resolution_MnKa" in mp:
            keys_dictionary["detector-peak-width-ev"] = (
                mp.Detector.EDS.energy_resolution_MnKa
            )

    write_rpl(filename, keys_dictionary, encoding)
    write_raw(filename, signal, record_by, sig_axes, nav_axes)


file_writer.__doc__ %= (
    FILENAME_DOC.replace("read", "write to"),
    SIGNAL_DOC,
    ENCODING_DOC.replace("read", "write"),
)


def write_rpl(filename, keys_dictionary, encoding="ascii"):
    with codecs.open(filename, "w", encoding=encoding, errors="ignore") as f:
        f.write(f";File created by RosettaSciIO version {__version__}\n")
        f.write("key\tvalue\n")
        # Even if it is not necessary, we sort the keywords when writing
        # to make the rpl file more human friendly
        for key, value in iter(sorted(keys_dictionary.items())):
            if not isinstance(value, str):
                value = str(value)
            f.write(key + "\t" + value + "\n")


def write_raw(filename, signal, record_by, sig_axes, nav_axes):
    """
    Writes the raw file object

    Parameters
    ----------
    filename : str
        the filename, either with the extension or without it
    record_by : str
         'vector' or 'image'
    """
    filename = os.path.splitext(filename)[0] + ".raw"
    data = signal["data"]
    dshape = data.shape
    if len(dshape) == 3:
        if record_by == "vector":
            np.rollaxis(data, signal["axes"].index(sig_axes[0]), 3).ravel().tofile(
                filename
            )
        elif record_by == "image":
            np.rollaxis(data, signal["axes"].index(nav_axes[0]), 0).ravel().tofile(
                filename
            )
    elif len(dshape) == 2:
        if record_by == "vector":
            np.rollaxis(data, signal["axes"].index(sig_axes[0]), 2).ravel().tofile(
                filename
            )
        elif record_by in ("image", "dont-care"):
            data.ravel().tofile(filename)
    elif len(dshape) == 1:
        data.ravel().tofile(filename)
