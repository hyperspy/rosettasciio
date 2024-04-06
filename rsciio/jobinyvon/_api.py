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

import importlib.util
import logging
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np

from rsciio._docstrings import FILENAME_DOC, LAZY_UNSUPPORTED_DOC, RETURNS_DOC

_logger = logging.getLogger(__name__)


def _remove_none_from_dict(dict_in):
    for key, value in list(dict_in.items()):
        if isinstance(value, dict):
            _remove_none_from_dict(value)
        elif value is None:
            del dict_in[key]


class JobinYvonXMLReader:
    """Class to read Jobin Yvon .xml-files.

    The file is read using xml.etree.ElementTree.
    Each element can have the following attributes: attrib, tag, text.
    Moreover, non-leaf-elements are iterable (iterate over child-nodes).
    In this specific format, the tags do not contain useful information.
    Instead, the "ID"-entry in attrib is used to identify the sort of information.
    The IDs are consistent for the tested files.

    Parameters
    ----------
    file_path: pathlib.Path
        Path to the to be read file.

    use_uniform_signal_axis: bool, default=False
        Decides whether to use uniform or non-uniform signal-axis.

    Attributes
    ----------
    data, metadata, original_metadata, axes

    Methods
    -------
    parse_file, get_original_metadata, get_axes, get_data, map_metadata
    """

    def __init__(self, file_path, use_uniform_signal_axis=False):
        self._file_path = file_path
        self._use_uniform_signal_axis = use_uniform_signal_axis

        if importlib.util.find_spec("lumispy") is None:
            self._lumispy_installed = False
            _logger.warning(
                "Cannot find package lumispy, using BaseSignal1D as signal class."
            )
        else:
            self._lumispy_installed = True  # pragma: no cover

    @property
    def _signal_type(self):
        if self._lumispy_installed:
            return "Luminescence"  # pragma: no cover
        else:
            return ""

    @staticmethod
    def _get_id(xml_element):
        return xml_element.attrib["ID"]

    @staticmethod
    def _get_size(xml_element):
        return int(xml_element.attrib["Size"])

    def parse_file(self):
        """First parse through file to extract data/metadata positions."""
        tree = ET.parse(self._file_path)
        root = tree.getroot()

        lsx_tree_list = root.findall("LSX_Tree")
        if len(lsx_tree_list) > 1:
            _logger.critical(
                "File contains multiple positions to read metadata from.\n"
                "The first location is choosen."
            )  # pragma: no cover
        elif len(lsx_tree_list) == 0:
            _logger.critical("No metadata found.")  # pragma: no cover
        lsx_tree = lsx_tree_list[0]

        lsx_matrix_list = root.findall("LSX_Matrix")
        if len(lsx_matrix_list) > 1:
            _logger.critical(
                "File contains multiple positions to read data from.\n"
                "The first location is choosen."
            )  # pragma: no cover
        elif len(lsx_matrix_list) == 0:
            _logger.critical("No data found.")  # pragma: no cover
        self._lsx_matrix = lsx_matrix_list[0]

        for child in lsx_tree:
            id = self._get_id(child)
            if id == "0x6C62D4D9":
                self._metadata_root = child
            if id == "0x6C7469D9":
                self._title = child.text
            if id == "0x6D707974":
                self._measurement_type = child.text
            if id == "0x6C676EC6":
                for child2 in child:
                    if self._get_id(child2) == "0x0":
                        self._angle = child2.text
            if id == "0x7A74D9D6":
                for child2 in child:
                    if self._get_id(child2) == "0x7B697861":
                        self._axis_root = child2

        if not hasattr(self, "_metadata_root"):
            _logger.critical("Could not extract metadata")  # pragma: no cover
        if not hasattr(self, "_axis_root"):
            _logger.critical("Could not extract axis")  # pragma: no cover

    def _get_metadata_values(self, xml_element, tag):
        """Helper method to extract information from metadata xml-element.

        Parameters
        ----------
        xml_element: xml.etree.ElementTree.Element
            Head level metadata element.
        tag: Str
            Used as the corresponding key in the original_metadata dictionary.

        Examples
        --------
        <LSX Format="9" ID="0x6D7361DE" Size="5">
                        <LSX Format="7" ID="0x6D6D616E">Laser (nm)</LSX>
                        <LSX Format="7" ID="0x7D6C61DB">633 </LSX>
                        <LSX Format="5" ID="0x8736F70">632.817</LSX>
                        <LSX Format="4" ID="0x6A6DE7D3">0</LSX>
        </LSX>

        This corresponds to child-level (xml-element would be multiple of those elements).
        ID="0x6D6D616E" -> key name for original metadata
        ID="0x7D6C61DB" -> first value
        ID="0x8736F70" -> second value

        Which value is used is decided in _clean_up_metadata (in this case the second value is used).
        """
        metadata_xml_element = dict()
        for child in xml_element:
            values = {}
            for child2 in child:
                if self._get_id(child2) == "0x6D6D616E":
                    key = child2.text
                if self._get_id(child2) == "0x7D6C61DB":
                    values["1"] = child2.text
                if self._get_id(child2) == "0x8736F70":
                    values["2"] = child2.text
            metadata_xml_element[key] = values
        self.original_metadata[tag] = metadata_xml_element

    def _clean_up_metadata(self):
        """Cleans up original metadata to meet standardized format.

        This means converting numbers from strings to floats,
        deciding which value shall be used (when 2 values are extracted,
        see _get_metadata_values() for more information).
        Moreover, some names are slightly modified.
        """
        convert_to_numeric = [
            "Acq. time (s)",
            "Accumulations",
            "Delay time (s)",
            "Binning",
            "Detector temperature (°C)",
            "Objective",
            "Grating",
            "ND Filter",
            "Laser (nm)",
            "Spectro (nm)",
            "Hole",
            "Laser Pol. (°)",
            "Raman Pol. (°)",
            "X (µm)",
            "Y (µm)",
            "Z (µm)",
            "Full time(s)",
            "rotation angle (rad)",
            "Windows",
        ]

        change_to_second_value = [
            "Objective",
            "Grating",
            "ND Filter",
            "Laser (nm)",
            "Spectro (nm)",
        ]

        ## use second extracted value
        for key in change_to_second_value:
            try:
                self.original_metadata["experimental_setup"][key] = (
                    self.original_metadata["experimental_setup"][key]["2"]
                )
            except KeyError:
                pass

        ## use first extracted value
        for key, value in self.original_metadata["experimental_setup"].items():
            if isinstance(value, dict):
                # only if there is an entry/value
                if bool(value):
                    self.original_metadata["experimental_setup"][key] = (
                        self.original_metadata["experimental_setup"][key]["1"]
                    )

        for key, value in self.original_metadata["date"].items():
            if isinstance(value, dict):
                if bool(value):
                    self.original_metadata["date"][key] = self.original_metadata[
                        "date"
                    ][key]["1"]

        for key, value in self.original_metadata["file_information"].items():
            if isinstance(value, dict):
                if bool(value):
                    self.original_metadata["file_information"][key] = (
                        self.original_metadata["file_information"][key]["1"]
                    )

        ## convert strings to float
        for key in convert_to_numeric:
            try:
                self.original_metadata["experimental_setup"][key] = float(
                    self.original_metadata["experimental_setup"][key]
                )
            except KeyError:
                pass

        ## move the unit from grating to the key name
        try:
            self.original_metadata["experimental_setup"]["Grating (gr/mm)"] = (
                self.original_metadata["experimental_setup"].pop("Grating")
            )
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover

        ## add percentage for filter key name
        try:
            self.original_metadata["experimental_setup"]["ND Filter (%)"] = (
                self.original_metadata["experimental_setup"].pop("ND Filter")
            )
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover

    def get_original_metadata(self):
        """Extracts metadata from file."""
        self.original_metadata = {}
        for child in self._metadata_root:
            id = self._get_id(child)
            if id == "0x7CECDBD7":
                date = child
            if id == "0x8716361":
                metadata = child
            if id == "0x7C73E2D2":
                file_specs = child

        ## setup tree structure original_metadata -> date{...}, experimental_setup{...}, file_information{...}
        ## based on structure in file
        self._get_metadata_values(date, "date")
        self._get_metadata_values(metadata, "experimental_setup")
        self._get_metadata_values(file_specs, "file_information")
        try:
            self.original_metadata["experimental_setup"]["measurement_type"] = (
                self._measurement_type
            )
        except AttributeError:  # pragma: no cover
            pass  # pragma: no cover
        try:
            self.original_metadata["experimental_setup"]["title"] = self._title
        except AttributeError:  # pragma: no cover
            pass  # pragma: no cover
        try:
            self.original_metadata["experimental_setup"]["rotation angle (rad)"] = (
                self._angle
            )
        except AttributeError:
            pass
        self._clean_up_metadata()

    def _set_signal_type(self, xml_element):
        """Sets signal type and units based on metadata from file.

        Extra method, because this information is stored seperate from the rest of the metadata.

        Parameters
        ----------
        xml_element: xml.etree.ElementTree.Element
            Head level metadata element.
        """
        for child in xml_element:
            id = self._get_id(child)
            ## contains also intensity-minima/maxima-values for each data-row (ignored by this reader)
            if id == "0x6D707974":
                self.original_metadata["experimental_setup"]["signal type"] = child.text
            if id == "0x7C696E75":
                self.original_metadata["experimental_setup"]["signal units"] = (
                    child.text
                )

    def _set_nav_axis(self, xml_element, tag):
        """Helper method for setting navigation axes.

        Parameters
        ----------
        xml_element: xml.etree.ElementTree.Element
            Head level metadata element.
        tag: Str
            Axis name.
        """
        has_nav = True
        nav_dict = dict()
        for child in xml_element:
            id = self._get_id(child)
            if id == "0x6D707974":
                nav_dict["name"] = child.text
            if id == "0x7C696E75":
                nav_dict["units"] = child.text
            if id == "0x7D6CD4DB":
                nav_array = np.fromstring(child.text.strip(), sep=" ")
                nav_size = nav_array.size
                if nav_size < 2:
                    has_nav = False
                else:
                    nav_dict["scale"] = nav_array[1] - nav_array[0]
                    nav_dict["offset"] = nav_array[0]
                    nav_dict["size"] = nav_size
                    nav_dict["navigate"] = True
        if has_nav:
            self.axes[tag] = nav_dict
        return has_nav, nav_size

    def _set_signal_axis(self, xml_element):
        """Helper method to extract signal axis information.

        Parameters
        ----------
        xml_element: xml.etree.ElementTree.Element
            Head level metadata element.
        tag: Str
            Axis name.
        """
        signal_dict = dict()
        signal_dict["navigate"] = False
        for child in xml_element:
            id = self._get_id(child)
            if id == "0x7D6CD4DB":
                signal_array = np.fromstring(child.text.strip(), sep=" ")
                if signal_array.size > 1:
                    if signal_array[0] > signal_array[1]:
                        signal_array = signal_array[::-1]
                        self._reverse_signal = True
                    else:
                        self._reverse_signal = False
                    if self._use_uniform_signal_axis:
                        offset, scale = np.polynomial.polynomial.polyfit(
                            np.arange(signal_array.size), signal_array, deg=1
                        )
                        signal_dict["offset"] = offset
                        signal_dict["scale"] = scale
                        signal_dict["size"] = signal_array.size
                        scale_compare = 100 * np.max(
                            np.abs(np.diff(signal_array) - scale) / scale
                        )
                        if scale_compare > 1:
                            _logger.warning(
                                f"The relative variation of the signal-axis-scale ({scale_compare:.2f}%) exceeds 1%.\n"
                                "                              "
                                "Using a non-uniform-axis is recommended."
                            )
                    else:
                        signal_dict["axis"] = signal_array
                else:  # pragma: no cover
                    if self._use_uniform_signal_axis:  # pragma: no cover
                        _logger.warning(  # pragma: no cover
                            "Signal only contains one entry.\n"  # pragma: no cover
                            "                              "  # pragma: no cover
                            "Using non-uniform-axis independent of use_uniform_signal_axis setting"  # pragma: no cover
                        )  # pragma: no cover
                    signal_dict["axis"] = signal_array  # pragma: no cover
            if id == "0x7C696E75":
                units = child.text
                if "/" in units and units[-3:] == "abs":
                    signal_dict["name"] = "Wavenumber"
                    signal_dict["units"] = units[:-4]
                elif "/" in units and units[-1] == "m":
                    signal_dict["name"] = "Raman Shift"
                    signal_dict["units"] = units
                elif units[-2:] == "eV":
                    signal_dict["name"] = "Energy"
                    signal_dict["units"] = units
                elif "/" not in units and units[-1] == "m":
                    signal_dict["name"] = "Wavelength"
                    signal_dict["units"] = units
                else:
                    _logger.warning(
                        "Cannot extract type of signal axis from units, using wavelength as name."
                    )  # pragma: no cover
                    signal_dict["name"] = "Wavelength"  # pragma: no cover
                    signal_dict["units"] = units  # pragma: no cover
        self.axes["signal_dict"] = signal_dict

    def _sort_nav_axes(self):
        """Sort the navigation/signal axes, such that (X, Y, Spectrum) = (1, 0, 2) (for map)
        or (X/Y, Spectrum) = (0, 1) or (Spectrum) = (0) (for linescan/spectrum).
        """
        self.axes["signal_dict"]["index_in_array"] = len(self.axes) - 1
        if self._has_nav2:
            self.axes["nav2_dict"]["index_in_array"] = 0
            if self._has_nav1:
                self.axes["nav1_dict"]["index_in_array"] = 1
        elif self._has_nav1 and not self._has_nav2:
            self.axes["nav1_dict"]["index_in_array"] = 0
        self.axes = sorted(self.axes.values(), key=lambda item: item["index_in_array"])

    def get_axes(self):
        """Extract navigation/signal axes data from file."""
        self.axes = dict()
        self._has_nav1 = False
        self._has_nav2 = False
        for child in self._axis_root:
            if self._get_id(child) == "0x0":
                self._set_signal_type(child)
            if self._get_id(child) == "0x1":
                self._set_signal_axis(child)
            if self._get_id(child) == "0x2":
                self._has_nav1, self._nav1_size = self._set_nav_axis(child, "nav1_dict")
            if self._get_id(child) == "0x3":
                self._has_nav2, self._nav2_size = self._set_nav_axis(child, "nav2_dict")

        self._sort_nav_axes()

    def get_data(self):
        """Extract data from file."""
        data_raw = self._lsx_matrix.findall("LSX_Row")
        ## lexicographical ordering -> 3x3 map -> 9 rows
        num_rows = len(data_raw)
        if num_rows == 0:
            _logger.critical("No data found.")  # pragma: no cover
        elif num_rows == 1:
            ## Spectrum
            self.data = np.fromstring(data_raw[0].text.strip(), sep=" ")
            if self._reverse_signal:
                self.data = self.data[::-1]
        else:
            ## linescan or map
            num_cols = self._get_size(data_raw[0])
            self.data = np.empty((num_rows, num_cols))
            for i, row in enumerate(data_raw):
                row_array = np.fromstring(row.text.strip(), sep=" ")
                if self._reverse_signal:
                    row_array = row_array[::-1]
                self.data[i, :] = row_array
            ## reshape the array (lexicographic -> cartesian)
            ## reshape depends on available axes
            if self._has_nav2:
                if self._has_nav1:
                    self.data = np.reshape(
                        self.data, (self._nav2_size, self._nav1_size, num_cols)
                    )
                else:
                    self.data = np.reshape(
                        self.data, (self._nav2_size, num_cols)
                    )  # pragma: no cover
            elif self._has_nav1 and not self._has_nav2:
                self.data = np.reshape(self.data, (self._nav1_size, num_cols))

    def map_metadata(self):
        """Maps original_metadata to metadata dictionary."""
        general = {}
        signal = {}
        laser = {"Filter": {}, "Polarizer": {}}
        spectrometer = {"Grating": {}, "Polarizer": {}}
        detector = {"processing": {}}
        spectral_image = {}
        sample = {}

        sample["description"] = self.original_metadata["file_information"].get("Sample")

        general["title"] = self._title
        general["original_filename"] = self._file_path.name
        general["notes"] = self.original_metadata["file_information"].get("Remark")
        try:
            date, time = self.original_metadata["date"]["Acquired"].split(" ")
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover
        else:
            general["date"] = date
            general["time"] = time

        signal["signal_type"] = self._signal_type
        signal["signal_dimension"] = 1
        try:
            intensity_axis = self.original_metadata["experimental_setup"]["signal type"]
            intensity_units = self.original_metadata["experimental_setup"][
                "signal units"
            ]
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover
        else:
            if intensity_axis == "Intens":
                intensity_axis = "Intensity"
            if intensity_units == "Cnt/sec":
                intensity_units = "Counts/s"
            if intensity_units == "Cnt":
                intensity_units = "Counts"
            signal["quantity"] = f"{intensity_axis} ({intensity_units})"

        laser["wavelength"] = self.original_metadata["experimental_setup"].get(
            "Laser (nm)"
        )
        laser["objective_magnification"] = self.original_metadata[
            "experimental_setup"
        ].get("Objective")
        laser["Filter"]["optical_density"] = self.original_metadata[
            "experimental_setup"
        ].get("ND Filter (%)")
        laser["Polarizer"]["polarizer_type"] = self.original_metadata[
            "experimental_setup"
        ].get("Laser. Pol.")
        laser["Polarizer"]["angle"] = self.original_metadata["experimental_setup"].get(
            "Laser Pol. (°)"
        )

        spectrometer["central_wavelength"] = self.original_metadata[
            "experimental_setup"
        ].get("Spectro (nm)")
        spectrometer["model"] = self.original_metadata["experimental_setup"].get(
            "Instrument"
        )
        spectrometer["Grating"]["groove_density"] = self.original_metadata[
            "experimental_setup"
        ].get("Grating (gr/mm)")
        spectrometer["entrance_slit_width"] = self.original_metadata[
            "experimental_setup"
        ].get("Hole")
        spectrometer["spectral_range"] = self.original_metadata[
            "experimental_setup"
        ].get("Range")
        spectrometer["Polarizer"]["polarizer_type"] = self.original_metadata[
            "experimental_setup"
        ].get("Raman. Pol.")
        spectrometer["Polarizer"]["angle"] = self.original_metadata[
            "experimental_setup"
        ].get("Raman Pol. (°)")

        detector["model"] = self.original_metadata["experimental_setup"].get("Detector")
        detector["delay_time"] = self.original_metadata["experimental_setup"].get(
            "Delay time (s)"
        )
        detector["binning"] = self.original_metadata["experimental_setup"].get(
            "Binning"
        )
        detector["temperature"] = self.original_metadata["experimental_setup"].get(
            "Detector temperature (°C)"
        )
        detector["exposure_per_frame"] = self.original_metadata[
            "experimental_setup"
        ].get("Acq. time (s)")
        detector["frames"] = self.original_metadata["experimental_setup"].get(
            "Accumulations"
        )
        detector["processing"]["autofocus"] = self.original_metadata[
            "experimental_setup"
        ].get("Autofocus")
        detector["processing"]["swift"] = self.original_metadata[
            "experimental_setup"
        ].get("SWIFT")
        detector["processing"]["auto_exposure"] = self.original_metadata[
            "experimental_setup"
        ].get("AutoExposure")
        detector["processing"]["spike_filter"] = self.original_metadata[
            "experimental_setup"
        ].get("Spike filter")
        detector["processing"]["de_noise"] = self.original_metadata[
            "experimental_setup"
        ].get("DeNoise")
        detector["processing"]["ics_correction"] = self.original_metadata[
            "experimental_setup"
        ].get("ICS correction")
        detector["processing"]["dark_correction"] = self.original_metadata[
            "experimental_setup"
        ].get("Dark correction")
        detector["processing"]["inst_process"] = self.original_metadata[
            "experimental_setup"
        ].get("Inst. Process")

        ## extra units here, because rad vs. deg
        if "rotation angle (rad)" in self.original_metadata["experimental_setup"]:
            spectral_image["rotation_angle"] = self.original_metadata[
                "experimental_setup"
            ]["rotation angle (rad)"] * (180 / np.pi)
            spectral_image["rotation_angle_units"] = "°"

        ## settings for glued spectra
        if "Windows" in self.original_metadata["experimental_setup"]:
            detector["glued_spectrum"] = True
            detector["glued_spectrum_windows"] = self.original_metadata[
                "experimental_setup"
            ]["Windows"]
        else:
            detector["glued_spectrum"] = False

        ## calculate and set integration time
        try:
            integration_time = (
                self.original_metadata["experimental_setup"]["Accumulations"]
                * self.original_metadata["experimental_setup"]["Acq. time (s)"]
            )
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover
        else:
            detector["integration_time"] = integration_time

        ## convert filter range from percentage (0-100) to (0-1)
        try:
            laser["Filter"]["optical_density"] /= 100
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover

        ## convert entrance_hole_width to mm
        try:
            spectrometer["entrance_slit_width"] /= 100
            spectrometer["pinhole"] = spectrometer["entrance_slit_width"]
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover

        self.metadata = {
            "General": general,
            "Signal": signal,
            "Sample": sample,
            "Acquisition_instrument": {
                "Laser": laser,
                "Spectrometer": spectrometer,
                "Detector": detector,
                "Spectral_image": spectral_image,
            },
        }

        _remove_none_from_dict(self.metadata)


def file_reader(filename, lazy=False, use_uniform_signal_axis=False):
    """
    Read data from .xml files saved using Horiba Jobin Yvon's LabSpec software.

    Parameters
    ----------
    %s
    %s
    use_uniform_signal_axis : bool, default=False
        Can be specified to choose between non-uniform or uniform signal axis.
        If ``True``, the ``scale`` attribute is calculated from the average delta
        along the signal axis and a warning is raised in case the delta varies
        by more than 1 percent.

    %s
    """
    if lazy is not False:
        raise NotImplementedError("Lazy loading is not supported.")
    if not isinstance(filename, Path):
        filename = Path(filename)
    jy = JobinYvonXMLReader(
        file_path=filename, use_uniform_signal_axis=use_uniform_signal_axis
    )
    jy.parse_file()
    jy.get_original_metadata()
    jy.get_axes()
    jy.get_data()
    jy.map_metadata()
    dictionary = {
        "data": jy.data,
        "axes": jy.axes,
        "metadata": deepcopy(jy.metadata),
        "original_metadata": deepcopy(jy.original_metadata),
    }
    return [
        dictionary,
    ]


file_reader.__doc__ %= (FILENAME_DOC, LAZY_UNSUPPORTED_DOC, RETURNS_DOC)
