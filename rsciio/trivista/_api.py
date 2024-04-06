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

import importlib.util
import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyfit

from rsciio._docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC

_logger = logging.getLogger(__name__)


def _error_handling_find_location(len_entry, name):
    if len_entry > 1:
        _logger.error(  # pragma: no cover
            f"File contains multiple positions to read {name} from."
            "                            "
            "The first location is chosen."
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
                ## in this case, the text is ignored
                ## if children=True -> text is just empty space in test data
                ## if t.attrib=True and children=False doesn't occur in test data
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
    """Class to read Trivista's .tvf-files using xml.etree.ElementTree.

    Attributes
    ----------
    data, metadata, original_metadata, axes
    """

    def __init__(
        self,
        file_path,
        use_uniform_signal_axis=False,
        glued_data_as_stack=False,
        filter_original_metadata=True,
    ):
        self._file_path = file_path
        self._use_uniform_signal_axis = use_uniform_signal_axis
        self._glued_data_as_stack = glued_data_as_stack

        (
            data_head,
            filtered_original_metadata,
            unfiltered_original_metadata,
        ) = self.parse_file_structure(filter_original_metadata)

        self._num_datasets = int(data_head.findall("Childs")[0].attrib["Count"])
        data, time, signal_axis = self.get_data_and_signal(data_head=data_head)

        self.axes = self.set_axes(
            axis_head=filtered_original_metadata["Document"]["InfoSerialized"],
            signal_axis=signal_axis,
            time=time,
        )

        self.data = self.reshape_data(data, self.axes)
        self.metadata = self.map_metadata(filtered_original_metadata)

        if filter_original_metadata:
            self.original_metadata = filtered_original_metadata
        else:
            self.original_metadata = unfiltered_original_metadata

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
                - hardware metadata

        Most of the usable metadata is in the InfoSerialized section.
        InfoSerialized and FileInfoSerialized need to be converted extra
        with ET.fromstring().

        The metadata from the hardware section contains information for all
        available hardware (i.e. multiple objectives even though only one is used.
        In the filtering process, the metadata of the actual used objective is extracted).

        Parameters
        ----------
        filter_original_metadata: bool
            if True, then unfiltered_original_metadata will
            contain a copy of the original_metadata before
            processing InfoSerialized and filtering the metadata
            from the Hardware section. Otherwise unfiltered_original_metadata
            will be empty.
            Independent of this parameter, the filtered and processed
            metadata is stored in filtered_original_metadata.
            This ensures that the metadata mapping doesn't
            depend on this setting.

        Returns
        -------
        data_head: ET
            file position where data/signal can be read from

        filtered_original_metadata: dict
            filtered + processed metadata

        unfiltered_original_metadata: dict
        """
        filtered_original_metadata = dict()
        unfiltered_original_metadata = dict()
        et_root = ET.parse(self._file_path).getroot()

        ## root level metadata
        filtered_original_metadata.update(_etree_to_dict(et_root, only_top_lvl=True))
        et_fileInfoSerialized = ET.fromstring(
            filtered_original_metadata["XmlMain"]["FileInfoSerialized"]
        )
        fileInfoSerialized = _etree_to_dict(et_fileInfoSerialized, only_top_lvl=False)
        filtered_original_metadata["XmlMain"]["FileInfoSerialized"] = fileInfoSerialized

        ## Documents / Document section
        data_head = et_root[1][0]
        filtered_original_metadata.update(_etree_to_dict(data_head, only_top_lvl=True))
        et_infoSerialized = ET.fromstring(
            filtered_original_metadata["Document"]["InfoSerialized"]
        )
        infoSerialized = _etree_to_dict(et_infoSerialized, only_top_lvl=False)

        ## Hardware section
        metadata_head = et_root[0]
        metadata_hardware = _etree_to_dict(metadata_head, only_top_lvl=False)

        if not filter_original_metadata:
            unfiltered_original_metadata = deepcopy(filtered_original_metadata)
            unfiltered_original_metadata["Document"]["InfoSerialized"] = deepcopy(
                infoSerialized
            )
            unfiltered_original_metadata.update(deepcopy(metadata_hardware))

        ## processing/filtering
        infoSerialized_processed = _process_info_serialized(
            infoSerialized["Info"]["Groups"]["Group"]
        )
        filtered_original_metadata["Document"]["InfoSerialized"] = (
            infoSerialized_processed
        )

        ## these methods alter metadata_hardware
        self._filter_laser_metadata(infoSerialized_processed, metadata_hardware)
        self._filter_detector_metadata(infoSerialized_processed, metadata_hardware)
        self._filter_objectives_metadata(metadata_hardware)
        self._filter_spectrometers_metadata(infoSerialized_processed, metadata_hardware)

        filtered_original_metadata.update(metadata_hardware)

        return data_head, filtered_original_metadata, unfiltered_original_metadata

    @staticmethod
    def _filter_laser_metadata(infoSerialized_processed, metadata_hardware):
        """Filter LightSources section (Laser) via wavelength if possible."""
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

    @staticmethod
    def _filter_detector_metadata(infoSerialized_processed, metadata_hardware):
        """Filter Detector section via name"""
        for detector in metadata_hardware["Hardware"]["Detectors"]["Detector"]:
            if detector["Name"] == infoSerialized_processed["Detector"]["Name"]:
                metadata_hardware["Hardware"]["Detectors"]["Detector"] = detector

    @staticmethod
    def _filter_objectives_metadata(metadata_hardware):
        """Filter microscope section (objective) via isEnabled tag"""
        for microscope in metadata_hardware["Hardware"]["Microscopes"]["Microscope"]:
            for objective in microscope["Objectives"]["Objective"]:
                if objective["IsEnabled"] == "True":
                    metadata_hardware["Hardware"]["Microscopes"]["Microscope"] = (
                        microscope
                    )
                    metadata_hardware["Hardware"]["Microscopes"]["Microscope"][
                        "Objectives"
                    ]["Objective"] = objective

    @staticmethod
    def _filter_spectrometers_metadata(infoSerialized_processed, metadata_hardware):
        """Filter spectrometers via serialnumbers"""
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
                metadata_hardware["Hardware"]["Spectrometers"][spectrometer_name] = (
                    spectrometer
                )
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

    def _map_general_md(self, original_metadata):
        general = {}
        general["title"] = self._file_path.name.split(".")[0]
        general["original_filename"] = self._file_path.name
        try:
            date, time = original_metadata["Document"]["RecordTime"].split(" ")
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover
        else:
            date_split = date.split("/")
            date = date_split[-1] + "-" + date_split[0] + "-" + date_split[1]
            general["date"] = date
            general["time"] = time.split(".")[0]
        return general

    def _map_signal_md(self, original_metadata):
        signal = {}

        if importlib.util.find_spec("lumispy") is None:
            _logger.warning(
                "Cannot find package lumispy, using BaseSignal as signal_type."
            )
            signal["signal_type"] = ""
        else:
            signal["signal_type"] = "Luminescence"  # pragma: no cover

        try:
            quantity = original_metadata["Document"]["Label"]
            quantity_unit = original_metadata["Document"]["DataLabel"]
        except KeyError:  # pragma: no cover
            pass  # pragma: no cover
        else:
            signal["quantity"] = f"{quantity} ({quantity_unit})"
        return signal

    def _map_detector_md(self, original_metadata):
        detector = {"processing": {}}
        detector_original = original_metadata["Document"]["InfoSerialized"]["Detector"]
        try:
            experiment_original = original_metadata["Document"]["InfoSerialized"][
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
                detector["glued_spectrum_windows"] = self._num_datasets
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

        return detector

    @staticmethod
    def _map_laser_md(original_metadata, laser_wavelength):
        laser = {}
        laser["objective_magnification"] = float(
            original_metadata["Hardware"]["Microscopes"]["Microscope"]["Objectives"][
                "Objective"
            ]["Magnification"]
        )
        if laser_wavelength is not None:
            laser["wavelength"] = laser_wavelength
        return laser

    @staticmethod
    def _map_spectrometer_md(original_metadata, central_wavelength):
        all_spectrometers_dict = {}

        spectrometers_original = original_metadata["Document"]["InfoSerialized"][
            "Spectrometers"
        ]

        for key, entry in spectrometers_original.items():
            spectro_dict_tmp = {"Grating": {}}
            spectro_dict_tmp["central_wavelength"] = central_wavelength
            blaze = original_metadata["Hardware"]["Spectrometers"][key]["Gratings"][
                "Grating"
            ]["Blaze"]
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

    @staticmethod
    def _get_calibration_md(original_metadata):
        try:
            calibration_original = original_metadata["Document"]["InfoSerialized"][
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

    def map_metadata(self, original_metadata):
        """Maps original_metadata to metadata."""
        general = self._map_general_md(original_metadata)
        signal = self._map_signal_md(original_metadata)
        detector = self._map_detector_md(original_metadata)
        central_wavelength, laser_wavelength = self._get_calibration_md(
            original_metadata
        )
        laser = self._map_laser_md(original_metadata, laser_wavelength)
        spectrometer = self._map_spectrometer_md(original_metadata, central_wavelength)

        acquisition_instrument = {
            "Detector": detector,
            "Laser": laser,
        }
        acquisition_instrument.update(spectrometer)

        metadata = {
            "Acquisition_instrument": acquisition_instrument,
            "General": general,
            "Signal": signal,
        }
        _remove_none_from_dict(metadata)
        return metadata

    def _get_signal_axis(self, axis_pos):
        """Helper method to read and set signal axis."""
        axis_list = axis_pos.findall("xDim")
        _error_handling_find_location(
            len(axis_list), "signal axis information"
        )  # pragma: no cover
        axis = axis_list[0]
        signal_data = axis[0].attrib["ValueArray"].split("|")
        if len(signal_data) != (int(signal_data[0]) + 1):
            _logger.critical(
                "Signal data size does not match expected size."
            )  # pragma: no cover
        signal_data = np.array([float(x) for x in signal_data[1:]])
        signal_dict = {}
        signal_dict["name"] = "Wavelength"
        unit = axis[0].attrib["Unit"]
        if unit == "Nanometer":
            signal_dict["units"] = "nm"
        else:
            signal_dict["units"] = unit  # pragma: no cover
        signal_dict["navigate"] = False
        if signal_data.size > 1:
            if self._use_uniform_signal_axis:
                offset, scale = polyfit(np.arange(signal_data.size), signal_data, deg=1)
                signal_dict["offset"] = offset
                signal_dict["scale"] = scale
                signal_dict["size"] = signal_data.size
                scale_compare = 100 * np.max(
                    np.abs(np.diff(signal_data) - scale) / scale
                )
                if scale_compare > 1:
                    _logger.warning(
                        f"The relative variation of the signal-axis-scale ({scale_compare:.2f}%) exceeds 1%.\n"
                        "                            "
                        "Using a non-uniform-axis is recommended."
                    )
            else:
                signal_dict["axis"] = signal_data
        else:  # pragma: no cover
            if self._use_uniform_signal_axis:  # pragma: no cover
                _logger.warning(  # pragma: no cover
                    "Signal only contains one entry.\n"  # pragma: no cover
                    "                            "  # pragma: no cover
                    "Using non-uniform-axis independent of use_uniform_signal_axis setting"  # pragma: no cover
                )  # pragma: no cover
            signal_dict["axis"] = signal_data  # pragma: no cover
        return signal_dict

    @staticmethod
    def _get_time_axis(time, axes_dict):
        scale = time[1]
        size = time.size
        ## inconsistency between timestamps and metadata
        ## (Document/InfoSerialized/Experiment6)
        ## in timeseries example file:
        ## exposure time: 1 sec
        ## delay: 3 sec
        ## accumulations: 2
        ## frames: 10
        ## total time: 56 sec
        ## timestamp scale: 4 sec
        ## -> max timestamp: 36 sec < 56 sec
        ## Here the timestamp is used for scale

        axes_dict["time"] = {
            "name": "time",
            "units": "s",
            "size": size,
            "offset": 0,
            "scale": scale,
            "navigate": False,
        }
        axes_dict["time"]["index_in_array"] = len(axes_dict) - 1

    @staticmethod
    def _get_nav_axis(name, axis):
        """Helper method to read and set navigation axes."""
        nav_dict = {}
        nav_dict["offset"] = float(axis["From"])
        nav_dict["scale"] = float(axis["Step"])
        nav_dict["size"] = int(axis["Points"])
        nav_dict["navigate"] = True
        nav_dict["name"] = name
        nav_dict["units"] = "µm"
        return nav_dict

    def set_axes(self, axis_head, signal_axis, time):
        """Extracts signal and navigation axes."""
        axes = dict()
        has_y = False
        has_x = False
        num_xy_frames = 0
        if "Y-Axis" in axis_head.keys():
            axes["Y"] = self._get_nav_axis("Y", axis_head["Y-Axis"])
            num_xy_frames += axes["Y"]["size"]
            axes["Y"]["index_in_array"] = 0
            has_y = True
        if "X-Axis" in axis_head.keys():
            axes["X"] = self._get_nav_axis("X", axis_head["X-Axis"])
            has_x = True
            if has_y:
                num_xy_frames *= axes["X"]["size"]
                axes["X"]["index_in_array"] = 1
            else:
                num_xy_frames += axes["X"]["size"]
                axes["X"]["index_in_array"] = 0

        ## adding time-axis entry if appropriate
        ## this is done inplace (the argument "axes" itself is altered)
        total_frames = int(axis_head["Detector"]["No_of_Frames"])
        if (total_frames - num_xy_frames) == 0 or total_frames == 1:
            has_time = False
        else:
            has_time = True
            self._get_time_axis(time, axes)
        if (has_x or has_y) and has_time:
            raise NotImplementedError(  # pragma: no cover
                "Reading a combination of timeseries and map or linescan is not implemented."
            )

        if self._glued_data_as_stack and self._num_datasets != 0:
            if has_time or has_x or has_y:
                _logger.warning(  # pragma: no cover
                    "Loading glued data as stack in combination with multiple axis (time, linescan or map)"
                    "                            "
                    "is not tested and may lead to false results."
                    "                            "
                    "Please use glued_data_as_stack=False for loading the file."
                )

            axes_list = []
            for signal_axis in signal_axis:
                axes_tmp = deepcopy(axes)
                axes_tmp["signal_dict"] = signal_axis
                axes_tmp["signal_dict"]["index_in_array"] = len(axes) - 1
                axes_tmp = sorted(
                    axes_tmp.values(), key=lambda item: item["index_in_array"]
                )
                axes_list.append(axes_tmp)
        else:
            axes["signal_dict"] = signal_axis[0]
            axes["signal_dict"]["index_in_array"] = len(axes) - 1
            ## extra surrounding list here to ensure compatibility
            ## with glued_data_as_stack for file_reader()
            axes_list = [sorted(axes.values(), key=lambda item: item["index_in_array"])]

        return axes_list

    @staticmethod
    def _parse_data(data_pos):
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

    def _load_glued_data_stack(self, data_head):
        num_datasets_list = data_head.findall("Childs")
        _error_handling_find_location(
            len(num_datasets_list), "glued datasets"
        )  # pragma: no cover
        data_array = []
        signal_axis_list = []
        time_array = []
        for dataset in num_datasets_list[0]:
            signal_axis = self._get_signal_axis(dataset)
            signal_axis_list.append(signal_axis)
            data, time = self._parse_data(dataset)
            data_array.append(data)
            time_array.append(time)
        data = np.array(data_array)
        time = np.array(time_array)
        return data, time, signal_axis_list

    def get_data_and_signal(self, data_head):
        if self._glued_data_as_stack and self._num_datasets != 0:
            data, time, signal_axis = self._load_glued_data_stack(data_head)
        else:
            data, time = self._parse_data(data_head)
            signal_axis = [self._get_signal_axis(data_head)]
            data = [data]
            ## extra surrounding list here
            ## to ensure compatibility with glued_data_as_stack=True
            ## for file_reader(), reshape_data()
        return data, time, signal_axis

    def reshape_data(self, data, axes):
        """Reshapes data according to axes sizes."""
        if self._use_uniform_signal_axis:
            wavelength_size = axes[0][-1]["size"]
        else:
            wavelength_size = axes[0][-1]["axis"].size
        shape_sizes = []
        for i in range(len(axes[0]) - 1):
            shape_sizes.append(axes[0][i]["size"])
        shape_sizes.append(wavelength_size)

        for i, dataset in enumerate(data):
            dataset_reshaped = np.reshape(dataset, shape_sizes)
            data[i] = dataset_reshaped
        return data


def file_reader(
    filename,
    lazy=False,
    use_uniform_signal_axis=False,
    glued_data_as_stack=False,
    filter_original_metadata=True,
):
    """
    Read TriVista's ``.tvf`` file.

    Parameters
    ----------
    %s
    %s
    use_uniform_signal_axis : bool, default=False
        Can be specified to choose between non-uniform or uniform signal axes.
        If `True`, the ``scale`` attribute is calculated from the average delta
        along the signal axis and a warning is raised in case the delta varies
        by more than 1%%.
    glued_data_as_stack : bool, default=False
        Using the mode `Step & Glue` results in measurements performed
        at different wavelength ranges with some overlap between them.
        The file then contains the individual spectra as well as
        the "glued" spectrum. The latter is represented as one spectrum,
        which covers the complete wavelength range. Stitching the datasets
        together in the overlap region is already done by the setup.
        If this setting is set to `True`, then the individual datasets will be loaded
        as a stack. Otherwise, only the "glued" spectrum is loaded.
    filter_original_metadata : bool, default=True
        Decides whether to process the original_metadata.
        If `True`, then non-relevant metadata will be excluded.
        For example, the metadata usually contains information
        for multiple objectives, even though only one is used.
        In this case, only the metadata from the used objective
        will be added to original_metadata.
        This setting only affects the ``original_metadata`` attribute
        and not the ``metadata`` attribute.

    %s
    """
    if lazy is not False:
        raise NotImplementedError("Lazy loading is not supported.")

    t = TrivistaTVFReader(
        Path(filename),
        use_uniform_signal_axis=use_uniform_signal_axis,
        glued_data_as_stack=glued_data_as_stack,
        filter_original_metadata=filter_original_metadata,
    )

    result = []
    for dataset, axes in zip(t.data, t.axes):
        result.append(
            {
                "data": dataset,
                "axes": axes,
                "metadata": deepcopy(t.metadata),
                "original_metadata": deepcopy(t.original_metadata),
            }
        )
    return result


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)
