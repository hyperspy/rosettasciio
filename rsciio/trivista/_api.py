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


def _error_handling_find_location(len_entry, name):
    if len_entry > 1:
        _logger.error(  # pragma: no cover
            f"File contains multiple positions to read {name} from."
            "                            "
            "The first location is choosen."
        )
    elif len_entry == 0:
        raise RuntimeError(
            f"Could not find location to read {name} from."
        )  # pragma: no cover


def _convert_float(input):
    """Handle None-values when converting strings to float."""
    if input is None:
        return None  # pragma: no cover
    else:
        return float(input)


def _remove_none_from_dict(dict_in):
    """Recursive removal of None-values from a dictionary."""
    for key, value in list(dict_in.items()):
        if isinstance(value, dict):
            _remove_none_from_dict(value)
        elif value is None:
            del dict_in[key]


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

    @property
    def num_datasets(self):
        return int(self.data_head.findall("Childs")[0].attrib["Count"])

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

    def _map_general_md(self):
        general = {}
        general["title"] = self._file_path.name.split(".")[0]
        general["original_filename"] = self._file_path.name
        try:
            date, time = self.original_metadata["Document"]["RecordTime"].split(" ")
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover
        else:
            date_split = date.split("/")
            date = date_split[-1] + "-" + date_split[0] + "-" + date_split[1]
            general["date"] = date
            general["time"] = time.split(".")[0]
        return general

    def _map_signal_md(self):
        signal = {}
        signal["signal_type"] = self._signal_type
        try:
            quantity = self.original_metadata["Document"]["Label"]
            quantity_unit = self.original_metadata["Document"]["DataLabel"]
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover
        else:
            signal["quantity"] = f"{quantity} ({quantity_unit})"
        return signal

    def _map_detector_md(self):
        detector = {"processing": {}}
        detector_original = self.original_metadata["Document"]["InfoSerialized"][
            "Detector"
        ]
        try:
            experiment_original = self.original_metadata["Document"]["InfoSerialized"][
                "Experiment"
            ]
        except KeyError:
            pass
        else:
            if "Overlap (%)" in experiment_original:
                detector["glued_spectrum"] = True
                detector["glued_spectrum_overlap"] = float(
                    experiment_original.get("Overlap (%)")
                )
                detector["glued_spectrum_windows"] = self.num_datasets
            else:
                detector["glued_spectrum"] = False
        detector["temperature"] = _convert_float(
            detector_original.get("Detector_Temperature")
        )
        detector["exposure_per_frame"] = (
            _convert_float(detector_original.get("Exposure_Time_(ms)")) / 1000
        )
        detector["frames"] = _convert_float(
            detector_original.get("No_of_Accumulations")
        )
        detector["processing"]["calc_average"] = detector_original.get("Calc_Average")

        try:
            detector["exposure_per_frame"] = (
                _convert_float(detector_original.get("Exposure_Time_(ms)")) / 1000
            )
        except TypeError:  # pragma: no cover
            detector["exposure_per_frame"] = None  # pragma: no cover

        if detector["processing"]["calc_average"] == "False":
            try:
                detector["integration_time"] = (
                    detector["exposure_per_frame"] * detector["frames"]
                )
            except TypeError:  # pragma: no cover
                pass  # pragma: no cover
        elif detector["processing"]["calc_average"] == "True":
            detector["integration_time"] = detector["exposure_per_frame"]

        self._total_frames = int(detector_original.get("No_of_Frames"))

        return detector

    def _map_laser_md(self, laser_wavelength):
        laser = {}
        laser["objective_magnification"] = float(
            self.original_metadata["Hardware"]["Microscopes"]["Microscope"][
                "Objectives"
            ]["Objective"]["Magnification"]
        )
        if not laser_wavelength is None:
            laser["wavelength"] = laser_wavelength
        return laser

    def _map_spectrometer_md(self, central_wavelength):
        all_spectrometers_dict = {}

        spectrometers_original = self.original_metadata["Document"]["InfoSerialized"][
            "Spectrometers"
        ]

        for key, entry in spectrometers_original.items():
            spectro_dict_tmp = {"Grating": {}}
            spectro_dict_tmp["central_wavelength"] = central_wavelength
            blaze = self.original_metadata["Hardware"]["Spectrometers"][key][
                "Gratings"
            ]["Grating"]["Blaze"]
            if blaze[-2:] == "NM":
                blaze = float(blaze.split("N")[0])
            spectro_dict_tmp["Grating"]["blazing_wavelength"] = blaze
            spectro_dict_tmp["model"] = entry.get("Model")
            try:
                groove_density = entry["Groove_Density"]
            except KeyError:  # pragma: no cover
                groove_density = None  # pragma: no cover
            else:
                groove_density = float(groove_density.split(" ")[0])
            spectro_dict_tmp["Grating"]["groove_density"] = groove_density
            slit_entrance_front = (
                _convert_float(entry.get("Slit_Entrance-Front")) / 1000
            )
            slit_entrance_side = _convert_float(entry.get("Slit_Entrance-Side")) / 1000
            slit_exit_front = _convert_float(entry.get("Slit_Exit-Front")) / 1000
            slit_exit_side = _convert_float(entry.get("Slit_Exit-Side")) / 1000
            ## using the maximum here, because
            ## only one entrance/exit should be in use anyways
            spectro_dict_tmp["entrance_slit_width"] = max(
                slit_entrance_front, slit_entrance_side
            )
            spectro_dict_tmp["exit_slit_width"] = max(slit_exit_front, slit_exit_side)
            all_spectrometers_dict[key] = spectro_dict_tmp

        return all_spectrometers_dict

    def _get_calibration_md(self):
        try:
            calibration_original = self.original_metadata["Document"]["InfoSerialized"][
                "Calibration"
            ]
        except KeyError:
            central_wavelength = None
            laser_wavelength = None
        else:
            central_wavelength = _convert_float(
                calibration_original.get("Center_Wavelength")
            )
            laser_wavelength = _convert_float(
                calibration_original.get("Laser_Wavelength")
            )
            if laser_wavelength is not None:
                if np.isclose(laser_wavelength, 0):
                    laser_wavelength = None
        return central_wavelength, laser_wavelength

    def map_metadata(self):
        """Maps original_metadata to metadata."""
        general = self._map_general_md()
        signal = self._map_signal_md()
        detector = self._map_detector_md()
        central_wavelength, laser_wavelength = self._get_calibration_md()
        laser = self._map_laser_md(laser_wavelength)
        spectrometer = self._map_spectrometer_md(central_wavelength)

        acquisition_instrument = {
            "Detector": detector,
            "Laser": laser,
        }
        acquisition_instrument.update(spectrometer)

        self.metadata = {
            "Acquisition_instrument": acquisition_instrument,
            "General": general,
            "Signal": signal,
        }
        _remove_none_from_dict(self.metadata)

    def _parse_data(self, data_pos):
        """Extracts data from file."""
        data_list = data_pos.findall("Data")
        _error_handling_find_location(len(data_list), "data")  # pragma: no cover

        ## dtype=np.int64 instead of int here,
        ## because on windows python int defaults to 32bit
        ## the timestamp is given as windows filetime
        ## -> number is too large for 32bit
        data_array = []
        time_array = []
        for frame in data_list[0]:
            time_frame = np.fromstring(
                frame.attrib["TimeStamp"], sep=" ", dtype=np.int64
            )
            data_frame = [float(x) for x in frame.text.split(";")]
            time_array.append(time_frame)
            data_array.append(data_frame)
        data = np.array(data_array).ravel()
        time = (np.array(time_array, dtype=np.int64) - time_array[0]).ravel() / 1e7
        return data, time

    def load_glued_data_stack(self):
        num_datasets_list = self.data_head.findall("Childs")
        _error_handling_find_location(
            len(num_datasets_list), "glued datasets"
        )  # pragma: no cover
        data_array = []
        self.signal_axis_list = []
        time_array = []
        for dataset in num_datasets_list[0]:
            signal_axis = self._get_signal_axis(dataset)
            self.signal_axis_list.append(signal_axis)
            data, time = self._parse_data(dataset)
            data_array.append(data)
            time_array.append(time)
        self.data = np.array(data_array)
        self.time = np.array(time_array)

    def get_data(self, glued_data_as_stack):
        if glued_data_as_stack and self.num_datasets != 0:
            self.load_glued_data_stack()
        else:
            data, self.time = self._parse_data(self.data_head)
            ## extra surrounding list here
            ## to ensure compatibility with glued_data_as_stack=True
            ## for file_reader(), reshape_data()
            self.data = [data]
