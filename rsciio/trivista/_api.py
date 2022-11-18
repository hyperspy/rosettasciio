# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

import xml.etree.ElementTree as ET
import logging
import importlib.util
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import numpy as np
_logger = logging.getLogger(__name__)

def _etree_to_dict(t, only_top_lvl=False):
    """Recursive conversion from xml.etree.ElementTree to a dictionary."""
    d = {t.tag: {} if t.attrib else None}
    if not only_top_lvl:
        children = list(t)
        if children:
            dd = defaultdict(list)
            for dc in map(_etree_to_dict, children):
                for k, v in dc.items():
                    dd[k].append(v)
            d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
        if t.text:
            if children or t.attrib:
                pass
            else:
                d[t.tag] = t.text.strip()
    if t.attrib:
        d[t.tag].update((k, v) for k, v in t.attrib.items())
    return d


def _process_info_serialized(head):
    """Recursive processing designed for the InfoSerialized entry from the original_metadata."""
    result = {}
    if not isinstance(head, list):
        head = [head]
    save_name_for_rename = set()
    for idx, entry in enumerate(head):
        entry_name = entry["Name"]
        if entry["Groups"] is not None:
            result[entry_name] = _process_info_serialized(entry["Groups"]["Group"])
        else:
            entry_dict_old = entry["Items"]["Item"]
            entry_dict_new = dict()
            if isinstance(entry_dict_old, list):
                for item in entry_dict_old:
                    entry_dict_new[item["Name"]] = item["Value"]
            elif isinstance(entry_dict_old, dict):
                entry_dict_new[entry_dict_old["Name"]] = entry_dict_old["Value"]
            if entry_name in result:
                save_name_for_rename.add(entry_name)
                entry_name += str(idx + 1)
            result[entry_name] = entry_dict_new
    for name in save_name_for_rename:
        result[name + "1"] = result.pop(name)
    return result


class TrivistaTVFReader:
    """Class to read Trivista's .tvf-files.

    The file is read using xml.etree.ElementTree.

    Parameters
    ----------
    file_path: pathlib.Path
        Path to the to be read file.

    Attributes
    ----------
    data, metadata, original_metadata, axes

    Methods
    -------
    read_file
    """

    def __init__(self, file_path):
        self._file_path = file_path

        if importlib.util.find_spec("lumispy") is None:
            self._lumispy_installed = False
            _logger.warning(
                "Cannot find package lumispy, using BaseSignal as signal_type."
            )
        else:
            self._lumispy_installed = True  # pragma: no cover

    @property
    def _signal_type(self):
        if self._lumispy_installed:
            return "Luminescence"  # pragma: no cover
        else:
            return ""

    def parse_file_structure(self, filter_original_metadata):
        """Initial parse through the file to extract original metadata and the data location.
        In general the file structure looks something like this.
        - root_level
            - root_level metadata
                - FileInfoSerialized
            - Document
                - data/signal_axis
                - document metadata
                    - InfoSerialized
            - Hardware

        Most of the usable metadata is in the InfoSerialized section.
        InfoSerialized and FileInfoSerialized need to be converted extra
        with _etree_to_dict().

        The metadata from the hardware section contains information for all
        available hardware (i.e. multiple objectives even though only one is used.
        In the filtering process, the metadata of the actual used objective is extracted).

        Parameters
        ----------
        filter_original_metadata: bool
            if True then 2 seperate dicts are created
            (original_metadata, unfiltered_original_metadata).
            Otherwise only the former is created.
            Filtering original_metadata is done independent
            of the value of this parameter.
            This ensures independence of the metadata mapping
            on this setting.
        """
        self.original_metadata = dict()
        et_root = ET.parse(self._file_path).getroot()

        ## root level metadata
        self.original_metadata.update(_etree_to_dict(et_root, only_top_lvl=True))
        et_fileInfoSerialized = ET.fromstring(
            self.original_metadata["XmlMain"]["FileInfoSerialized"]
        )
        fileInfoSerialized = _etree_to_dict(et_fileInfoSerialized, only_top_lvl=False)
        self.original_metadata["XmlMain"]["FileInfoSerialized"] = fileInfoSerialized

        ## Documents / Document section
        self.data_head = et_root[1][0]
        self.original_metadata.update(_etree_to_dict(self.data_head, only_top_lvl=True))
        et_infoSerialized = ET.fromstring(
            self.original_metadata["Document"]["InfoSerialized"]
        )
        infoSerialized = _etree_to_dict(et_infoSerialized, only_top_lvl=False)
        infoSerialized_processed = _process_info_serialized(
            infoSerialized["Info"]["Groups"]["Group"]
        )
        self.original_metadata["Document"]["InfoSerialized"] = infoSerialized_processed

        ## Hardware section
        metadata_head = et_root[0]
        metadata_hardware = _etree_to_dict(metadata_head, only_top_lvl=False)
        if not filter_original_metadata:
            self.unfiltered_original_metadata = deepcopy(self.original_metadata)
            self.unfiltered_original_metadata.update(deepcopy(metadata_hardware))

        ## filter LightSources section (Laser) via wavelength
        for laser in metadata_hardware["Hardware"]["LightSources"]["LightSource"]:
            try:
                calibration_wl = float(
                    infoSerialized_processed["Calibration"]["Laser_Wavelength"]
                )
                laser_wl = float(laser["Wavelengths"]["Value_0"])
            except KeyError:
                pass
            else:
                if np.isclose(calibration_wl, laser_wl) and not np.isclose(laser_wl, 0):
                    metadata_hardware["Hardware"]["LightSources"]["LightSource"] = laser

        ## filter Detector section via "name"
        for detector in metadata_hardware["Hardware"]["Detectors"]["Detector"]:
            if detector["Name"] == infoSerialized_processed["Detector"]["Name"]:
                metadata_hardware["Hardware"]["Detectors"]["Detector"] = detector

        ## filter microscope section (objective) via isEnabled tag
        for microscope in metadata_hardware["Hardware"]["Microscopes"]["Microscope"]:
            for objective in microscope["Objectives"]["Objective"]:
                if objective["IsEnabled"] == "True":
                    metadata_hardware["Hardware"]["Microscopes"][
                        "Microscope"
                    ] = microscope
                    metadata_hardware["Hardware"]["Microscopes"]["Microscope"][
                        "Objectives"
                    ]["Objective"] = objective

        ## get serialnumbers for all used spectrometers
        ## contrary to the other parts, multiple spectrometers can be used
        spectrometer_serial_numbers = []
        spectrometer_serialized_list = []
        for key, val in infoSerialized_processed["Spectrometers"].items():
            spectrometer_serial_numbers.append(val["Serialnumber"])
            spectrometer_serialized_list.append(key)

        ## filter spectrometers via serialnumber
        ## result for one spectrometer:
        ## Spectrometers
        ##     - Spectrometer
        ##         - ...
        ## result for 2 spectrometers:
        ## Spectrometers
        ##     - Spectrometer1
        ##         - ...
        ##     - Spectrometer2
        ##         - ...
        for spectrometer in metadata_hardware["Hardware"]["Spectrometers"][
            "Spectrometer"
        ]:
            if spectrometer["Serialnumber"] in spectrometer_serial_numbers:
                idx = spectrometer_serial_numbers.index(spectrometer["Serialnumber"])
                spectrometer_name = spectrometer_serialized_list[idx]
                metadata_hardware["Hardware"]["Spectrometers"][
                    spectrometer_name
                ] = spectrometer
                ## filter grating via groove density
                gratings_root = spectrometer["Gratings"]["Grating"]
                for grating in gratings_root:
                    if (
                        grating["GrooveDensity"]
                        == infoSerialized_processed["Spectrometers"][spectrometer_name][
                            "Groove_Density"
                        ]
                    ):
                        metadata_hardware["Hardware"]["Spectrometers"][
                            spectrometer_name
                        ]["Gratings"]["Grating"] = grating
        if not spectrometer_name == "Spectrometer":
            del metadata_hardware["Hardware"]["Spectrometers"]["Spectrometer"]
        self.original_metadata.update(metadata_hardware)
