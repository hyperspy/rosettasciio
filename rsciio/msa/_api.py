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

import codecs
import logging
import os
import warnings
from datetime import datetime as dt

import numpy as np

from rsciio._docstrings import (
    ENCODING_DOC,
    FILENAME_DOC,
    LAZY_UNSUPPORTED_DOC,
    RETURNS_DOC,
    SIGNAL_DOC,
)
from rsciio.utils.tools import DTBox

_logger = logging.getLogger(__name__)


# For a description of the EMSA/MSA format, including the meaning of the
# following keywords:
# https://www.amc.anl.gov/ANLSoftwareLibrary/02-MMSLib/XEDS/EMMFF/EMMFF.IBM/Emmff.Total

US_MONTHS_D2A = {
    "01": "JAN",
    "02": "FEB",
    "03": "MAR",
    "04": "APR",
    "05": "MAY",
    "06": "JUN",
    "07": "JUL",
    "08": "AUG",
    "09": "SEP",
    "10": "OCT",
    "11": "NOV",
    "12": "DEC",
}

US_MONTH_A2D = dict([reversed(i) for i in US_MONTHS_D2A.items()])

keywords = {
    # Required parameters
    "FORMAT": {"dtype": str, "mapped_to": None},
    "VERSION": {"dtype": str, "mapped_to": None},
    "TITLE": {"dtype": str, "mapped_to": "General.title"},
    "DATE": {"dtype": str, "mapped_to": None},
    "TIME": {"dtype": str, "mapped_to": None},
    "OWNER": {"dtype": str, "mapped_to": None},
    "NPOINTS": {"dtype": float, "mapped_to": None},
    "NCOLUMNS": {"dtype": float, "mapped_to": None},
    "DATATYPE": {"dtype": str, "mapped_to": None},
    "XPERCHAN": {"dtype": float, "mapped_to": None},
    "OFFSET": {"dtype": float, "mapped_to": None},
    # Optional parameters
    # Signal1D characteristics
    "SIGNALTYPE": {"dtype": str, "mapped_to": None},
    "XLABEL": {"dtype": str, "mapped_to": None},
    "YLABEL": {"dtype": str, "mapped_to": None},
    "XUNITS": {"dtype": str, "mapped_to": None},
    "YUNITS": {"dtype": str, "mapped_to": None},
    "CHOFFSET": {"dtype": float, "mapped_to": None},
    "COMMENT": {"dtype": str, "mapped_to": None},
    # Microscope
    "BEAMKV": {"dtype": float, "mapped_to": "Acquisition_instrument.TEM.beam_energy"},
    "EMISSION": {"dtype": float, "mapped_to": None},
    "PROBECUR": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.beam_current",
    },
    "BEAMDIAM": {"dtype": float, "mapped_to": None},
    "MAGCAM": {"dtype": float, "mapped_to": None},
    "OPERMODE": {"dtype": str, "mapped_to": None},
    "CONVANGLE": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.convergence_angle",
    },
    # Specimen
    "THICKNESS": {"dtype": float, "mapped_to": "Sample.thickness"},
    "XTILTSTGE": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.Stage.tilt_alpha",
    },
    "YTILTSTGE": {"dtype": float, "mapped_to": None},
    "XPOSITION": {"dtype": float, "mapped_to": None},
    "YPOSITION": {"dtype": float, "mapped_to": None},
    "ZPOSITION": {"dtype": float, "mapped_to": None},
    # EELS
    # in ms:
    "INTEGTIME": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.Detector.EELS.exposure",
    },
    # in ms:
    "DWELLTIME": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.Detector.EELS.dwell_time",
    },
    "COLLANGLE": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.Detector.EELS.collection_angle",
    },
    "ELSDET": {"dtype": str, "mapped_to": None},
    # EDS
    "ELEVANGLE": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
    },
    "AZIMANGLE": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
    },
    "SOLIDANGLE": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.Detector.EDS.solid_angle",
    },
    "LIVETIME": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.Detector.EDS.live_time",
    },
    "REALTIME": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.Detector.EDS.real_time",
    },
    "FWHMMNKA": {
        "dtype": float,
        "mapped_to": "Acquisition_instrument.TEM.Detector.EDS."
        + "energy_resolution_MnKa",
    },
    "TBEWIND": {"dtype": float, "mapped_to": None},
    "TAUWIND": {"dtype": float, "mapped_to": None},
    "TDEADLYR": {"dtype": float, "mapped_to": None},
    "TACTLYR": {"dtype": float, "mapped_to": None},
    "TALWIND": {"dtype": float, "mapped_to": None},
    "TPYWIND": {"dtype": float, "mapped_to": None},
    "TBNWIND": {"dtype": float, "mapped_to": None},
    "TDIWIND": {"dtype": float, "mapped_to": None},
    "THCWIND": {"dtype": float, "mapped_to": None},
    "EDSDET": {
        "dtype": str,
        "mapped_to": "Acquisition_instrument.TEM.Detector.EDS.EDS_det",
    },
}


def parse_msa_string(string, filename=None):
    """
    Parse an EMSA/MSA file content.

    Parameters
    ----------
    string : str
        It must complain with the EMSA/MSA standard.
    filename : str, None
        The filename.

    Returns
    -------
    list
        List of a single dictionary to be returned by ``file_reader``.
    """
    if not hasattr(string, "readlines"):
        string = string.splitlines()
    parameters = {}
    mapped = DTBox(box_dots=True)
    y = []
    # Read the keywords
    data_section = False
    for line in string:
        if data_section is False:
            if line[0] == "#":
                try:
                    key, value = line.split(": ")
                    value = value.strip()
                except ValueError:
                    key = line
                    value = None
                key = key.strip("#").strip()

                if key != "SPECTRUM":
                    parameters[key] = value
                else:
                    data_section = True
        else:
            # Read the data
            if line[0] != "#" and line.strip():
                if parameters["DATATYPE"] == "XY":
                    xy = line.replace(",", " ").strip().split()
                    y.append(float(xy[1]))
                elif parameters["DATATYPE"] == "Y":
                    data = [float(i) for i in line.replace(",", " ").strip().split()]
                    y.extend(data)
    # We rewrite the format value to be sure that it complies with the
    # standard, because it will be used by the writer routine
    parameters["FORMAT"] = "EMSA/MAS Spectral Data File"

    # Convert the parameters to the right type and map some
    # TODO: the msa format seems to support specifying the units of some
    # parametes. We should add this feature here
    for parameter, value in parameters.items():
        # Some parameters names can contain the units information
        # e.g. #AZIMANGLE-dg: 90.
        if "-" in parameter:
            clean_par, units = parameter.split("-")
            clean_par, units = clean_par.strip(), units.strip()
        else:
            clean_par, units = parameter, None
        if clean_par in keywords:
            type_ = keywords[clean_par]["dtype"]
            try:
                parameters[parameter] = type_(value)
            except Exception:
                error = f"The {parameter} keyword value, {value} could \
                    not be converted to the right type."
                if "e" in value.lower():
                    # Normally, the offending misspelling is a space in the
                    # scientific notation, e.g. 2.0 E-06
                    try:
                        parameters[parameter] = type_(value.replace(" ", ""))
                    except Exception:  # pragma: no cover
                        _logger.exception(error)
                else:
                    # Some files have two values separated by a space
                    # https://eelsdb.eu/wp-content/uploads/2017/03/Cu4O3-O-K.msa
                    try:
                        parameters[parameter] = type_(value.split(" ")[0])
                    except Exception:  # pragma: no cover
                        _logger.exception(error)

            if keywords[clean_par]["mapped_to"] is not None:
                mapped.set_item(keywords[clean_par]["mapped_to"], parameters[parameter])
                if units is not None:
                    mapped.set_item(keywords[clean_par]["mapped_to"] + "_units", units)
    if "TIME" in parameters and parameters["TIME"]:
        try:
            time = dt.strptime(parameters["TIME"], "%H:%M")
            mapped.set_item("General.time", time.time().isoformat())
        except ValueError as e:
            _logger.warning(
                "Possible malformed TIME field in msa file. The time "
                f"information could not be retrieved.: {e}"
            )

    malformed_date_error = "Possibly malformed DATE in msa file. The date information could not be retrieved."
    if "DATE" in parameters and parameters["DATE"]:
        try:
            day, month, year = parameters["DATE"].split("-")
            if month.upper() in US_MONTH_A2D:
                month = US_MONTH_A2D[month.upper()]
                date = dt.strptime("-".join((day, month, year)), "%d-%m-%Y")
                mapped.set_item("General.date", date.date().isoformat())
            else:
                _logger.warning(malformed_date_error)
        except (
            ValueError
        ) as e:  # Error raised if split does not return 3 elements in this case
            _logger.warning(malformed_date_error + ": %s" % e)

    axes = [
        {
            "size": len(y),
            "index_in_array": 0,
            "name": parameters["XLABEL"] if "XLABEL" in parameters else "",
            "scale": parameters["XPERCHAN"] if "XPERCHAN" in parameters else 1,
            "offset": parameters["OFFSET"] if "OFFSET" in parameters else 0,
            "units": parameters["XUNITS"] if "XUNITS" in parameters else "",
            "navigate": False,
        }
    ]
    if filename is not None:
        mapped.set_item("General.original_filename", os.path.split(filename)[1])
    if "SIGNALTYPE" in parameters and parameters["SIGNALTYPE"]:
        if parameters["SIGNALTYPE"] == "ELS":
            mapped.set_item("Signal.signal_type", "EELS")
        elif parameters["SIGNALTYPE"] == "CLS":
            mapped.set_item("Signal.signal_type", "CL")
        else:  # pragma: no cover
            if parameters["SIGNALTYPE"] not in [
                "EDS",
                "WDS",
                "AES",
                "PES",
                "XRF",
                "GAM",
            ]:
                warnings.warn(
                    """SIGNALTYPE does not correspond to any of the valid strings
                    according to the MSA file format definition."""
                )
            mapped.set_item("Signal.signal_type", parameters["SIGNALTYPE"])
    else:
        # Defaults to empty signal type, which is Signal1D for HyperSpy
        mapped.set_item("Signal.signal_type", "")
    if "YUNITS" in parameters.keys():
        yunits = "(%s)" % parameters["YUNITS"]
    else:
        yunits = ""
    if "YLABEL" in parameters.keys():
        quantity = "%s" % parameters["YLABEL"]
    else:
        if mapped.Signal.signal_type == "EELS":
            quantity = "Electrons"
            if not yunits:
                yunits = "(Counts)"
        elif "EDS" in mapped.Signal.signal_type:
            quantity = "X-rays"
            if not yunits:
                yunits = "(Counts)"
        else:
            quantity = ""
    if quantity or yunits:
        quantity_units = "%s %s" % (quantity, yunits)
        mapped.set_item("Signal.quantity", quantity_units.strip())

    dictionary = {
        "data": np.array(y),
        "axes": axes,
        "metadata": mapped.to_dict(),
        "original_metadata": parameters,
    }
    file_data_list = [
        dictionary,
    ]
    return file_data_list


def file_reader(filename, lazy=False, encoding="latin-1"):
    """
    Read an MSA file.

    Parameters
    ----------
    %s
    %s
    %s

    %s
    """
    if lazy is not False:
        raise NotImplementedError("Lazy loading is not supported.")

    with codecs.open(filename, encoding=encoding, errors="replace") as spectrum_file:
        return parse_msa_string(string=spectrum_file, filename=filename)


file_reader.__doc__ %= (FILENAME_DOC, LAZY_UNSUPPORTED_DOC, ENCODING_DOC, RETURNS_DOC)


def file_writer(filename, signal, format="Y", separator=", ", encoding="latin-1"):
    """
    Write signal to an MSA file.

    Parameters
    ----------
    %s
    %s
    format : str, default="Y"
        Specify whether the X-axis (energy/wavelength) should also be saved with
        the data. The default, ``"Y"`` omits the X-axis in the file. The alternative,
        ``"XY"``, saves the calibrated signal axis as first column.
    separator : str, Default=", "
        Change the column separator. However, if a different separator is chosen
        the resulting file will not comply with the MSA/EMSA standard and
        RosettaSciIO and other software may not be able to read it.
    %s

    Examples
    --------
    >>> from rsciio.msa import file_writer
    >>> file_writer("file.msa", signal, encoding="utf8")
    """
    loc_kwds = {}
    FORMAT = "EMSA/MAS Spectral Data File"
    md = DTBox(signal["metadata"], box_dots=True)
    if signal["original_metadata"].get("FORMAT", None) == FORMAT:
        loc_kwds = signal["original_metadata"]
        if format is not None:
            loc_kwds["DATATYPE"] = format
        else:
            if "DATATYPE" in loc_kwds:
                format = loc_kwds["DATATYPE"]
    else:
        if format is None:
            format = "Y"
        if "General.date" in md:
            date = dt.strptime(md.General.date, "%Y-%m-%d")
            date_str = date.strftime("%d-%m-%Y")
            day, month, year = date_str.split("-")
            month = US_MONTHS_D2A[month]
            loc_kwds["DATE"] = "-".join((day, month, year))
        if "General.item" in md:
            time = dt.strptime(md.General.time, "%H:%M:%S")
            loc_kwds["TIME"] = time.strftime("%H:%M")
    if md.Signal.signal_type in ["EDS_SEM", "EDS_TEM"]:
        loc_kwds["SIGNALTYPE"] = "EDS"
    elif md.Signal.signal_type in ["EELS"]:
        loc_kwds["SIGNALTYPE"] = "ELS"
    elif md.Signal.signal_type in ["CL", "CL_SEM", "CL_STEM"]:
        loc_kwds["SIGNALTYPE"] = "CLS"
    elif md.Signal.signal_type not in [
        "EDS",
        "WDS",
        "ELS",
        "AES",
        "PES",
        "XRF",
        "CLS",
        "GAM",
    ]:
        loc_kwds["SIGNALTYPE"] = ""
    else:
        loc_kwds["SIGNALTYPE"] = md.Signal.signal_type
    keys_from_signal = {
        # Required parameters
        "FORMAT": FORMAT,
        "VERSION": "1.0",
        # 'TITLE' : signal.title[:64] if hasattr(signal, "title") else '',
        "DATE": "",
        "TIME": "",
        "OWNER": "",
        "NPOINTS": signal["axes"][0]["size"],
        "NCOLUMNS": 1,
        "DATATYPE": format,
        "SIGNALTYPE": "",
        "XPERCHAN": signal["axes"][0]["scale"],
        "OFFSET": signal["axes"][0]["offset"],
        # Signal1D characteristics
        "XLABEL": signal["axes"][0]["name"],
        #        'YLABEL' : '',
        "XUNITS": signal["axes"][0]["units"],
        #        'YUNITS' : '',
        "COMMENT": "File created by RosettaSciIO version {__version__}",
        # Microscope
        #        'BEAMKV' : ,
        #        'EMISSION' : ,
        #        'PROBECUR' : ,
        #        'BEAMDIAM' : ,
        #        'MAGCAM' : ,
        #        'OPERMODE' : ,
        #        'CONVANGLE' : ,
        # Specimen
        #        'THICKNESS' : ,
        #        'XTILTSTGE' : ,
        #        'YTILTSTGE' : ,
        #        'XPOSITION' : ,
        #        'YPOSITION' : ,
        #        'ZPOSITION' : ,
        #
        # EELS
        # 'INTEGTIME' : , # in ms
        # 'DWELLTIME' : , # in ms
        #        'COLLANGLE' : ,
        #        'ELSDET' :  ,
    }

    # Update the loc_kwds with the information retrieved from the signal class
    for key, value in keys_from_signal.items():
        if key not in loc_kwds or value != "":
            loc_kwds[key] = value

    for key, dic in keywords.items():
        if dic["mapped_to"] is not None:
            if "SEM" in md.Signal.signal_type:
                dic["mapped_to"] = dic["mapped_to"].replace("TEM", "SEM")
            if dic["mapped_to"] in md:
                loc_kwds[key] = eval("md.%s" % dic["mapped_to"])

    with codecs.open(filename, "w", encoding=encoding, errors="ignore") as f:
        # Remove the following keys from loc_kwds if they are in
        # (although they shouldn't)
        for key in ["SPECTRUM", "ENDOFDATA"]:
            if key in loc_kwds:
                del loc_kwds[key]

        f.write("#%-12s: %s\u000d\u000a" % ("FORMAT", loc_kwds.pop("FORMAT")))
        f.write("#%-12s: %s\u000d\u000a" % ("VERSION", loc_kwds.pop("VERSION")))
        for keyword, value in loc_kwds.items():
            f.write("#%-12s: %s\u000d\u000a" % (keyword, value))

        f.write("#%-12s: Spectral Data Starts Here\u000d\u000a" % "SPECTRUM")

        if format == "XY":
            axis_dict = signal["axes"][0]
            axis = axis_dict["offset"] + axis_dict["scale"] * np.arange(
                axis_dict["size"]
            )
            for x, y in zip(axis, signal["data"]):
                f.write("%g%s%g" % (x, separator, y))
                f.write("\u000d\u000a")
        elif format == "Y":
            for y in signal["data"]:
                f.write("%f%s" % (y, separator))
                f.write("\u000d\u000a")
        else:
            raise ValueError("format must be one of: None, 'XY' or 'Y'")

        f.write("#%-12s: End Of Data and File" % "ENDOFDATA")


file_writer.__doc__ %= (
    FILENAME_DOC.replace("read", "write to"),
    SIGNAL_DOC,
    ENCODING_DOC.replace("read", "write"),
)
