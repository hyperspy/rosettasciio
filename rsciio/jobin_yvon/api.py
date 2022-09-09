# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from copy import deepcopy

_logger = logging.getLogger(__name__)


class JobinYvonXMLReader:
    def __init__(self, file_path, use_uniform_signal_axis=False):
        self._file_path = file_path

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

        Example
        -------
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
                self.original_metadata["experimental_setup"][
                    key
                ] = self.original_metadata["experimental_setup"][key]["2"]
            except KeyError:
                pass

        ## use first extracted value
        for key, value in self.original_metadata["experimental_setup"].items():
            if isinstance(value, dict):
                # only if there is an entry/value
                if bool(value):
                    self.original_metadata["experimental_setup"][
                        key
                    ] = self.original_metadata["experimental_setup"][key]["1"]

        for key, value in self.original_metadata["date"].items():
            if isinstance(value, dict):
                if bool(value):
                    self.original_metadata["date"][key] = self.original_metadata[
                        "date"
                    ][key]["1"]

        for key, value in self.original_metadata["file_information"].items():
            if isinstance(value, dict):
                if bool(value):
                    self.original_metadata["file_information"][
                        key
                    ] = self.original_metadata["file_information"][key]["1"]

        ## convert strings to float
        for key in convert_to_numeric:
            try:
                self.original_metadata["experimental_setup"][key] = float(
                    self.original_metadata["experimental_setup"][key]
                )
            except KeyError:
                pass

        ## move the unit from grating to the key name
        new_grating_key_name = "Grating (gr/mm)"
        try:
            self.original_metadata["experimental_setup"][
                new_grating_key_name
            ] = self.original_metadata["experimental_setup"]["Grating"]
            del self.original_metadata["experimental_setup"]["Grating"]
        except KeyError:
            pass  # pragma: no cover

        ## add percentage for filter key name
        new_filter_key_name = "ND Filter (%)"
        try:
            self.original_metadata["experimental_setup"][
                new_filter_key_name
            ] = self.original_metadata["experimental_setup"]["ND Filter"]
            del self.original_metadata["experimental_setup"]["ND Filter"]
        except KeyError:
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
            self.original_metadata["experimental_setup"][
                "measurement_type"
            ] = self._measurement_type
        except AttributeError:
            pass  # pragma: no cover
        try:
            self.original_metadata["experimental_setup"]["title"] = self._title
        except AttributeError:
            pass  # pragma: no cover
        try:
            self.original_metadata["experimental_setup"][
                "rotation angle (rad)"
            ] = self._angle
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
                self.original_metadata["experimental_setup"][
                    "signal units"
                ] = child.text

