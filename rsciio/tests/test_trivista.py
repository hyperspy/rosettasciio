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

import gc
import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")

TEST_DATA_PATH = Path(__file__).parent / "data" / "trivista"

testfile_spec_path = TEST_DATA_PATH / "spec_1s_1acc_1frame_average.tvf"
testfile_spec_2frames_path = TEST_DATA_PATH / "spec_3s_1acc_2frames_average.tvf"
testfile_spec_2acc_path = TEST_DATA_PATH / "spec_3s_2acc_1frame_average.tvf"
testfile_spec_2acc_no_average_path = (
    TEST_DATA_PATH / "spec_3s_2acc_1frame_no_average.tvf"
)
testfile_spec_timeseries_path = TEST_DATA_PATH / "spec_timeseries_2x1s_delta3s.tvf"
testfile_map_path = TEST_DATA_PATH / "map.tvf"
testfile_step_and_glue_path = TEST_DATA_PATH / "spec_step_and_glue.tvf"
testfile_triple_add_path = TEST_DATA_PATH / "spec_multiple_spectrometers.tvf"
testfile_linescan_path = TEST_DATA_PATH / "linescan.tvf"


if importlib.util.find_spec("lumispy") is None:
    lumispy_installed = False
else:
    lumispy_installed = True


class TestSpec:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_spec_path,
            reader="TriVista",
            use_uniform_signal_axis=True,
            filter_original_metadata=True,
        )
        cls.s_non_uniform_unfiltered = hs.load(
            testfile_spec_path,
            reader="TriVista",
            use_uniform_signal_axis=False,
            filter_original_metadata=False,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        del cls.s_non_uniform_unfiltered
        gc.collect()

    def test_data(self):
        expected_data = np.array([27732, 39135, 35343, -420, 32464])
        np.testing.assert_allclose(expected_data, self.s.isig[:5].data)
        np.testing.assert_allclose(
            expected_data, self.s_non_uniform_unfiltered.isig[:5].data
        )

    def test_axes(self):
        expected_axis = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
                "size": 1024,
            }
        }

        expected_axis_non_uniform = {
            "axis-0": {
                "_type": "DataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
            }
        }

        expected_values_non_uniform_axis = np.array(
            [794.220731002166, 794.261531331698, 794.302331149879]
        )

        uniform_axes_manager = deepcopy(self.s.axes_manager.as_dictionary())
        non_uniform_axis_manager = deepcopy(
            self.s_non_uniform_unfiltered.axes_manager.as_dictionary()
        )

        np.testing.assert_allclose(
            uniform_axes_manager["axis-0"].pop("scale"), 0.040515, atol=0.000005
        )
        np.testing.assert_allclose(
            uniform_axes_manager["axis-0"].pop("offset"), 794.22, atol=0.1
        )
        np.testing.assert_allclose(
            expected_values_non_uniform_axis,
            non_uniform_axis_manager["axis-0"].pop("axis")[:3],
        )
        assert expected_axis == uniform_axes_manager
        assert expected_axis_non_uniform == non_uniform_axis_manager

    def test_metadata(self):
        metadata = deepcopy(self.s.metadata.as_dictionary())
        metadata_non_uniform = deepcopy(
            self.s_non_uniform_unfiltered.metadata.as_dictionary()
        )

        assert metadata["General"]["FileIO"]["0"]["io_plugin"] == "rsciio.trivista"
        assert (
            metadata_non_uniform["General"]["FileIO"]["0"]["io_plugin"]
            == "rsciio.trivista"
        )

        assert metadata["General"]["date"] == "2022-06-14"
        assert (
            metadata["General"]["original_filename"]
            == "spec_1s_1acc_1frame_average.tvf"
        )
        assert metadata["General"]["time"] == "13:34:27"
        assert (
            metadata["General"]["title"]
            == metadata["General"]["original_filename"][:-4]
        )

        assert metadata["Signal"]["quantity"] == "Intensity (Counts)"
        if lumispy_installed:
            assert metadata["Signal"]["signal_type"] == "Luminescence"
        else:
            assert metadata["Signal"]["signal_type"] == ""

        assert metadata["Acquisition_instrument"]["Detector"]["glued_spectrum"] is False
        assert (
            metadata["Acquisition_instrument"]["Detector"]["processing"]["calc_average"]
            == "True"
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["exposure_per_frame"], 1
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["frames"], 1
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["integration_time"], 1
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["temperature"], -25
        )

        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Laser"]["objective_magnification"], 100
        )

        assert (
            metadata["Acquisition_instrument"]["Spectrometer"]["Grating"][
                "blazing_wavelength"
            ]
            == "H-NIR"
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Spectrometer"]["Grating"][
                "groove_density"
            ],
            750,
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Spectrometer"]["central_wavelength"],
            815,
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Spectrometer"]["entrance_slit_width"],
            0.2,
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Spectrometer"]["exit_slit_width"], 0
        )
        assert (
            metadata["Acquisition_instrument"]["Spectrometer"]["model"] == "SP-2-750i"
        )

        del metadata["General"]["FileIO"]
        del metadata_non_uniform["General"]["FileIO"]

        assert metadata == metadata_non_uniform

    def test_original_metadata(self):
        expected_metadata = {
            "XmlMain": {
                "Version": "1.0.1",
                "Filename": "C:\\Users\\optik\\Desktop\\Testspectrum.tvf",
                "DateTime": "06/14/2022 13:35:01",
                "PlugInName": "",
                "PlugInData": "",
                "FileInfoSerialized": {"Info": {"Groups": None}},
            },
            "Document": {
                "Version": "2",
                "Label": "Intensity",
                "DataLabel": "Counts",
                "DocType": "Spectra",
                "RecordTime": "06/14/2022 13:34:27.453",
                "ModeName": "",
                "DataSource": "nothing",
                "Encoding": "Ascii",
                "ColorMode": "Grayscale",
                "ViewDisplayMode": "Graph",
                "ViewImageColorMode": "False",
                "InfoSerialized": {
                    "Document": {"Record Time": "6/14/2022 1:34 PM"},
                    "Experiment": {"Used Setup": "PL_Stage3_750g"},
                    "Spectrometers": {
                        "Spectrometer": {
                            "Serialnumber": "27580185",
                            "Model": "SP-2-750i",
                            "Stage_Number": "1",
                            "Focallength": "749",
                            "Inclusion_Angle": "6.5",
                            "Detector_Angle": "0.68",
                            "Groove_Density": "750 g/mm",
                            "Order": "1",
                            "Slit_Entrance-Front": "200",
                            "Slit_Entrance-Side": "0",
                            "Slit_Exit-Front": "0",
                            "Slit_Exit-Side": "0",
                        }
                    },
                    "Detector": {
                        "Name": "Camera1",
                        "Serialnumber": None,
                        "Detector_Size": "1024;1",
                        "Detector_Temperature": "-25",
                        "Exposure_Time_(ms)": "1000",
                        "Exposure_Mode": None,
                        "No_of_Accumulations": "1",
                        "Calc_Average": "True",
                        "No_of_Frames": "1",
                        "ADC__Readout_Port": "Normal",
                        "ADC__Rate_Resolution": "1 MHz",
                        "ADC__Gain": "2",
                        "Clearing__Mode": None,
                        "Clearing__No_of_Cleans": "1",
                        "Region_of_Interests": "1|1;1024;1;1;1;1",
                    },
                    "Calibration": {
                        "Center_Wavelength": "815.000",
                        "Laser_Wavelength": "0.000",
                    },
                },
            },
            "Hardware": {
                "Spectrometers": {
                    "Spectrometer": {
                        "Gratings": {
                            "Grating": {
                                "Offsets": {"Count": "0"},
                                "GrooveDensity": "750 g/mm",
                                "Blaze": "H-NIR",
                                "Turret": "1",
                            },
                            "Count": "3",
                        },
                        "MirrorEntrance": {
                            "Name": "M1",
                            "State": "isMotorized",
                            "Position": "front",
                        },
                        "MirrorExit": {
                            "Name": "M2",
                            "State": "isMotorized",
                            "Position": "side",
                        },
                        "SlitEntranceSide": {
                            "Name": "S1",
                            "State": "isMotorized",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "SlitEntranceFront": {
                            "Name": "S2",
                            "State": "isMotorized",
                            "MaxWidth": "12000",
                            "Position": "500",
                        },
                        "SlitExitFront": {
                            "Name": "S3",
                            "State": "disabled",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "SlitExitSide": {
                            "Name": "S4",
                            "State": "disabled",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "FilterWheel": {
                            "Name": "",
                            "State": "disabled",
                            "Position": "0",
                            "FilterStrings": "6|#1|#2|#3|#4|#5|#6",
                        },
                        "Shutter": {
                            "Name": "",
                            "State": "disabled",
                            "Position": "open",
                            "ShutterMode": "OpenForExperiment",
                            "ShutterClosedForFilterwheel": "True",
                            "ShutterClosedForSetupChange": "True",
                            "ShutterClosedForLaserCrossing": "True",
                            "ShutterClosedForLaserCrossingValue": "0.000",
                            "ShutterClosedForLaserCrossingUnit": "Nanometer",
                        },
                        "GUID": "37a2855b-db41-4268-970b-bb75d96e5654",
                        "Model": "SP-2-750i",
                        "Serialnumber": "27580185",
                        "FocalLength": "749",
                        "InclusionAngle": "6.500",
                        "InclusionAngleAdditive": "39.000",
                        "DetectorAngle": "0.680",
                        "PixelWidthExitFront": "25.000",
                        "PixelWidthExitSide": "-1.000",
                        "Backlash": "ScanUp",
                        "DriveDirection": "Normal",
                        "ComPortSpec": "COM5",
                        "ComPortEntrance": "",
                        "ComPortExit": "",
                        "DemoMode": "False",
                        "UsedAsLightsource": "False",
                        "UseBackwards": "False",
                        "Wavelength": "850.184",
                        "Grating": "2",
                        "ReadTimeout": "35000",
                        "Activated": "False",
                        "IsFixedCalibration": "False",
                        "FixedCalibrationPoints": "0",
                    },
                    "Count": "3",
                },
                "Detectors": {
                    "Detector": {
                        "GUID": "c1bc9fd2-135e-4098-b681-d30f6ea8be82",
                        "Driver": "PvCam32 Driver",
                        "Name": "Camera1",
                        "DisplayName": "Camera1",
                        "SerialNumber": "",
                        "DetectorType": "None",
                        "Size": "1024;1",
                        "OpenAtStartup": "False",
                        "Usage": "Default",
                    },
                    "Count": "3",
                },
                "SinglePointDetectors": {"Count": "0"},
                "LightSources": {
                    "LightSource": [
                        {
                            "Wavelengths": {"Count": "1", "Value_0": "0.000"},
                            "GUID": "a57ac71a-c667-4501-a24a-596bc54b2c8c",
                            "Name": "Calibration Lamp",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "False",
                            "Type": "CalibrationLamp",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "0"},
                            "GUID": "cac03351-735e-4592-9bb2-c17f64b3649f",
                            "Name": "Calibration Lamp2",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "False",
                            "Type": "CalibrationLamp",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "0"},
                            "GUID": "84a08bee-227a-46f0-b1e1-0f171808a155",
                            "Name": "Calibration Lamp3",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "False",
                            "Type": "CalibrationLamp",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "1", "Value_0": "0.000"},
                            "GUID": "a0601d22-23a0-4587-bb79-35d5823916c4",
                            "Name": "Laser",
                            "Serialnumber": "",
                            "FixedWavelength": "False",
                            "CanChange": "False",
                            "Type": "Laser",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "1", "Value_0": "790.000"},
                            "GUID": "a36b29da-0d54-4ab7-860c-967269f76217",
                            "Name": "tisa",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "True",
                            "Type": "Laser",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                    ],
                    "Count": "5",
                },
                "ArduinoHardware": {"Count": "0"},
                "Additionals": {"Count": "0"},
                "MotorizedStages": {
                    "MotorizedStage": {
                        "GUID": "c551372e-56af-4686-b7c1-48ca00dfdf27",
                        "DriverName": "Tango Driver",
                        "Name": "XY-Stage",
                        "ConnectionString": "COM15",
                        "Serialnumber": "144012056",
                        "IsTriggerModeAvailable": "False",
                    },
                    "Count": "1",
                },
                "Microscopes": {
                    "Microscope": {
                        "Objectives": {
                            "Objective": {
                                "IsEnabled": "True",
                                "Name": "100x",
                                "Magnification": "100.000",
                                "FactorMeasuredWithResolution": "1280",
                                "Factor": "34.700",
                                "LaserOffsetX": "0.000",
                                "LaserOffsetY": "0.000",
                                "OffsetX": "0.000",
                                "OffsetY": "0.000",
                                "LineLaserOffsetX": "0.000",
                                "LineLaserOffsetY": "0.000",
                                "LineOffsetX": "0.000",
                                "LineOffsetY": "0.000",
                                "LineFocusFactor": "1.130",
                            },
                            "Count": "10",
                        },
                        "GUID": "841bfef9-b969-43f0-9ca3-2522c223a31f",
                        "Name": "Microscope",
                        "StageID": "c551372e-56af-4686-b7c1-48ca00dfdf27",
                        "ImagingCameraID": "",
                        "RamanBoxAvailable": "True",
                        "RamanBoxLinearStage.Available": "False",
                        "RamanBoxLinearStage.ConnectedHarwareID": "",
                        "RamanBoxFilterWheel.Available": "False",
                        "RamanBoxFilterWheel.ConnectedHarwareID": "",
                        "RamanBoxBeamSplitter.Available": "False",
                        "RamanBoxBeamSplitter.ConnectedHarwareID": "",
                        "RamanBoxNotchFilter.Available": "False",
                        "RamanBoxNotchFilter.ConnectedHarwareID": "",
                        "ActiveObjectiveIndex": "0",
                    },
                    "Count": "2",
                },
                "Cryostats": {"Count": "0"},
                "Version": "2",
            },
        }

        assert expected_metadata == self.s.original_metadata.as_dictionary()

    def test_unfiltered_original_metadata(self):
        expected_metadata = {
            "XmlMain": {
                "Version": "1.0.1",
                "Filename": "C:\\Users\\optik\\Desktop\\Testspectrum.tvf",
                "DateTime": "06/14/2022 13:35:01",
                "PlugInName": "",
                "PlugInData": "",
                "FileInfoSerialized": {"Info": {"Groups": None}},
            },
            "Document": {
                "Version": "2",
                "Label": "Intensity",
                "DataLabel": "Counts",
                "DocType": "Spectra",
                "RecordTime": "06/14/2022 13:34:27.453",
                "ModeName": "",
                "DataSource": "nothing",
                "Encoding": "Ascii",
                "ColorMode": "Grayscale",
                "ViewDisplayMode": "Graph",
                "ViewImageColorMode": "False",
                "InfoSerialized": {
                    "Info": {
                        "Groups": {
                            "Group": [
                                {
                                    "Name": "Document",
                                    "Items": {
                                        "Item": {
                                            "Name": "Record Time",
                                            "Value": "6/14/2022 1:34 PM",
                                            "IsVisible": "true",
                                        }
                                    },
                                    "Groups": None,
                                    "IsVisible": "true",
                                },
                                {
                                    "Name": "Experiment",
                                    "Items": {
                                        "Item": {
                                            "Name": "Used Setup",
                                            "Value": "PL_Stage3_750g",
                                            "IsVisible": "true",
                                        }
                                    },
                                    "Groups": None,
                                    "IsVisible": "true",
                                },
                                {
                                    "Name": "Spectrometers",
                                    "Items": None,
                                    "Groups": {
                                        "Group": {
                                            "Name": "Spectrometer",
                                            "Items": {
                                                "Item": [
                                                    {
                                                        "Name": "Serialnumber",
                                                        "Value": "27580185",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Model",
                                                        "Value": "SP-2-750i",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Stage_Number",
                                                        "Value": "1",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Focallength",
                                                        "Value": "749",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Inclusion_Angle",
                                                        "Value": "6.5",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Detector_Angle",
                                                        "Value": "0.68",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Groove_Density",
                                                        "Value": "750 g/mm",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Order",
                                                        "Value": "1",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Slit_Entrance-Front",
                                                        "Value": "200",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Slit_Entrance-Side",
                                                        "Value": "0",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Slit_Exit-Front",
                                                        "Value": "0",
                                                        "IsVisible": "true",
                                                    },
                                                    {
                                                        "Name": "Slit_Exit-Side",
                                                        "Value": "0",
                                                        "IsVisible": "true",
                                                    },
                                                ]
                                            },
                                            "Groups": None,
                                            "IsVisible": "true",
                                        }
                                    },
                                    "IsVisible": "true",
                                },
                                {
                                    "Name": "Detector",
                                    "Items": {
                                        "Item": [
                                            {
                                                "Name": "Name",
                                                "Value": "Camera1",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "Serialnumber",
                                                "Value": None,
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "Detector_Size",
                                                "Value": "1024;1",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "Detector_Temperature",
                                                "Value": "-25",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "Exposure_Time_(ms)",
                                                "Value": "1000",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "Exposure_Mode",
                                                "Value": None,
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "No_of_Accumulations",
                                                "Value": "1",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "Calc_Average",
                                                "Value": "True",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "No_of_Frames",
                                                "Value": "1",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "ADC__Readout_Port",
                                                "Value": "Normal",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "ADC__Rate_Resolution",
                                                "Value": "1 MHz",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "ADC__Gain",
                                                "Value": "2",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "Clearing__Mode",
                                                "Value": None,
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "Clearing__No_of_Cleans",
                                                "Value": "1",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "Region_of_Interests",
                                                "Value": "1|1;1024;1;1;1;1",
                                                "IsVisible": "true",
                                            },
                                        ]
                                    },
                                    "Groups": None,
                                    "IsVisible": "true",
                                },
                                {
                                    "Name": "Calibration",
                                    "Items": {
                                        "Item": [
                                            {
                                                "Name": "Center_Wavelength",
                                                "Value": "815.000",
                                                "IsVisible": "true",
                                            },
                                            {
                                                "Name": "Laser_Wavelength",
                                                "Value": "0.000",
                                                "IsVisible": "true",
                                            },
                                        ]
                                    },
                                    "Groups": None,
                                    "IsVisible": "true",
                                },
                            ]
                        }
                    }
                },
            },
            "Hardware": {
                "Spectrometers": {
                    "Spectrometer": [
                        {
                            "Gratings": {
                                "Grating": [
                                    {
                                        "Offsets": {"Count": "0"},
                                        "GrooveDensity": "1800 g/mm",
                                        "Blaze": "H-VIS",
                                        "Turret": "1",
                                    },
                                    {
                                        "Offsets": {"Count": "0"},
                                        "GrooveDensity": "900 g/mm",
                                        "Blaze": "550NM",
                                        "Turret": "1",
                                    },
                                    {
                                        "Offsets": {"Count": "0"},
                                        "GrooveDensity": "750 g/mm",
                                        "Blaze": "H-NIR",
                                        "Turret": "1",
                                    },
                                ],
                                "Count": "3",
                            },
                            "MirrorEntrance": {
                                "Name": "M1",
                                "State": "isMotorized",
                                "Position": "front",
                            },
                            "MirrorExit": {
                                "Name": "M2",
                                "State": "isMotorized",
                                "Position": "side",
                            },
                            "SlitEntranceSide": {
                                "Name": "S1",
                                "State": "isMotorized",
                                "MaxWidth": "12000",
                                "Position": "0",
                            },
                            "SlitEntranceFront": {
                                "Name": "S2",
                                "State": "isMotorized",
                                "MaxWidth": "12000",
                                "Position": "200",
                            },
                            "SlitExitFront": {
                                "Name": "S3",
                                "State": "isMotorized",
                                "MaxWidth": "12000",
                                "Position": "0",
                            },
                            "SlitExitSide": {
                                "Name": "S4",
                                "State": "disabled",
                                "MaxWidth": "12000",
                                "Position": "0",
                            },
                            "FilterWheel": {
                                "Name": "",
                                "State": "disabled",
                                "Position": "0",
                                "FilterStrings": "6|#1|#2|#3|#4|#5|#6",
                            },
                            "Shutter": {
                                "Name": "",
                                "State": "disabled",
                                "Position": "open",
                                "ShutterMode": "OpenForExperiment",
                                "ShutterClosedForFilterwheel": "True",
                                "ShutterClosedForSetupChange": "True",
                                "ShutterClosedForLaserCrossing": "True",
                                "ShutterClosedForLaserCrossingValue": "0.000",
                                "ShutterClosedForLaserCrossingUnit": "Nanometer",
                            },
                            "GUID": "c05fbec6-7cc0-4546-a009-06850868f43d",
                            "Model": "SP-2-500i",
                            "Serialnumber": "25580419",
                            "FocalLength": "500",
                            "InclusionAngle": "8.600",
                            "InclusionAngleAdditive": "51.600",
                            "DetectorAngle": "0.000",
                            "PixelWidthExitFront": "-1.000",
                            "PixelWidthExitSide": "-1.000",
                            "Backlash": "ScanUp",
                            "DriveDirection": "Normal",
                            "ComPortSpec": "COM3",
                            "ComPortEntrance": "",
                            "ComPortExit": "",
                            "DemoMode": "False",
                            "UsedAsLightsource": "False",
                            "UseBackwards": "False",
                            "Wavelength": "925.010",
                            "Grating": "2",
                            "ReadTimeout": "35000",
                            "Activated": "False",
                            "IsFixedCalibration": "False",
                            "FixedCalibrationPoints": "0",
                        },
                        {
                            "Gratings": {
                                "Grating": [
                                    {
                                        "Offsets": {"Count": "0"},
                                        "GrooveDensity": "1800 g/mm",
                                        "Blaze": "H-VIS",
                                        "Turret": "1",
                                    },
                                    {
                                        "Offsets": {"Count": "0"},
                                        "GrooveDensity": "900 g/mm",
                                        "Blaze": "550NM",
                                        "Turret": "1",
                                    },
                                    {
                                        "Offsets": {"Count": "0"},
                                        "GrooveDensity": "750 g/mm",
                                        "Blaze": "H-NIR",
                                        "Turret": "1",
                                    },
                                ],
                                "Count": "3",
                            },
                            "MirrorEntrance": {
                                "Name": "M1",
                                "State": "isMotorized",
                                "Position": "side",
                            },
                            "MirrorExit": {
                                "Name": "M2",
                                "State": "isMotorized",
                                "Position": "side",
                            },
                            "SlitEntranceSide": {
                                "Name": "S1",
                                "State": "isMotorized",
                                "MaxWidth": "12000",
                                "Position": "10000",
                            },
                            "SlitEntranceFront": {
                                "Name": "S2",
                                "State": "isMotorized",
                                "MaxWidth": "12000",
                                "Position": "0",
                            },
                            "SlitExitFront": {
                                "Name": "S3",
                                "State": "isMotorized",
                                "MaxWidth": "12000",
                                "Position": "0",
                            },
                            "SlitExitSide": {
                                "Name": "S4",
                                "State": "disabled",
                                "MaxWidth": "12000",
                                "Position": "0",
                            },
                            "FilterWheel": {
                                "Name": "",
                                "State": "disabled",
                                "Position": "0",
                                "FilterStrings": "6|#1|#2|#3|#4|#5|#6",
                            },
                            "Shutter": {
                                "Name": "",
                                "State": "disabled",
                                "Position": "open",
                                "ShutterMode": "OpenForExperiment",
                                "ShutterClosedForFilterwheel": "True",
                                "ShutterClosedForSetupChange": "True",
                                "ShutterClosedForLaserCrossing": "True",
                                "ShutterClosedForLaserCrossingValue": "0.000",
                                "ShutterClosedForLaserCrossingUnit": "Nanometer",
                            },
                            "GUID": "39ade9ad-caa8-48c2-a3d0-4f79cc44bda7",
                            "Model": "SP-2-500i",
                            "Serialnumber": "25580420",
                            "FocalLength": "500",
                            "InclusionAngle": "8.600",
                            "InclusionAngleAdditive": "51.600",
                            "DetectorAngle": "0.000",
                            "PixelWidthExitFront": "1.000",
                            "PixelWidthExitSide": "-1.000",
                            "Backlash": "ScanUp",
                            "DriveDirection": "Normal",
                            "ComPortSpec": "COM4",
                            "ComPortEntrance": "",
                            "ComPortExit": "",
                            "DemoMode": "False",
                            "UsedAsLightsource": "False",
                            "UseBackwards": "False",
                            "Wavelength": "-924.910",
                            "Grating": "2",
                            "ReadTimeout": "35000",
                            "Activated": "False",
                            "IsFixedCalibration": "False",
                            "FixedCalibrationPoints": "0",
                        },
                        {
                            "Gratings": {
                                "Grating": [
                                    {
                                        "Offsets": {"Count": "0"},
                                        "GrooveDensity": "1800 g/mm",
                                        "Blaze": "H-VIS",
                                        "Turret": "1",
                                    },
                                    {
                                        "Offsets": {"Count": "0"},
                                        "GrooveDensity": "900 g/mm",
                                        "Blaze": "550NM",
                                        "Turret": "1",
                                    },
                                    {
                                        "Offsets": {"Count": "0"},
                                        "GrooveDensity": "750 g/mm",
                                        "Blaze": "H-NIR",
                                        "Turret": "1",
                                    },
                                ],
                                "Count": "3",
                            },
                            "MirrorEntrance": {
                                "Name": "M1",
                                "State": "isMotorized",
                                "Position": "front",
                            },
                            "MirrorExit": {
                                "Name": "M2",
                                "State": "isMotorized",
                                "Position": "side",
                            },
                            "SlitEntranceSide": {
                                "Name": "S1",
                                "State": "isMotorized",
                                "MaxWidth": "12000",
                                "Position": "0",
                            },
                            "SlitEntranceFront": {
                                "Name": "S2",
                                "State": "isMotorized",
                                "MaxWidth": "12000",
                                "Position": "500",
                            },
                            "SlitExitFront": {
                                "Name": "S3",
                                "State": "disabled",
                                "MaxWidth": "12000",
                                "Position": "0",
                            },
                            "SlitExitSide": {
                                "Name": "S4",
                                "State": "disabled",
                                "MaxWidth": "12000",
                                "Position": "0",
                            },
                            "FilterWheel": {
                                "Name": "",
                                "State": "disabled",
                                "Position": "0",
                                "FilterStrings": "6|#1|#2|#3|#4|#5|#6",
                            },
                            "Shutter": {
                                "Name": "",
                                "State": "disabled",
                                "Position": "open",
                                "ShutterMode": "OpenForExperiment",
                                "ShutterClosedForFilterwheel": "True",
                                "ShutterClosedForSetupChange": "True",
                                "ShutterClosedForLaserCrossing": "True",
                                "ShutterClosedForLaserCrossingValue": "0.000",
                                "ShutterClosedForLaserCrossingUnit": "Nanometer",
                            },
                            "GUID": "37a2855b-db41-4268-970b-bb75d96e5654",
                            "Model": "SP-2-750i",
                            "Serialnumber": "27580185",
                            "FocalLength": "749",
                            "InclusionAngle": "6.500",
                            "InclusionAngleAdditive": "39.000",
                            "DetectorAngle": "0.680",
                            "PixelWidthExitFront": "25.000",
                            "PixelWidthExitSide": "-1.000",
                            "Backlash": "ScanUp",
                            "DriveDirection": "Normal",
                            "ComPortSpec": "COM5",
                            "ComPortEntrance": "",
                            "ComPortExit": "",
                            "DemoMode": "False",
                            "UsedAsLightsource": "False",
                            "UseBackwards": "False",
                            "Wavelength": "850.184",
                            "Grating": "2",
                            "ReadTimeout": "35000",
                            "Activated": "False",
                            "IsFixedCalibration": "False",
                            "FixedCalibrationPoints": "0",
                        },
                    ],
                    "Count": "3",
                },
                "Detectors": {
                    "Detector": [
                        {
                            "GUID": "c1bc9fd2-135e-4098-b681-d30f6ea8be82",
                            "Driver": "PvCam32 Driver",
                            "Name": "Camera1",
                            "DisplayName": "Camera1",
                            "SerialNumber": "",
                            "DetectorType": "None",
                            "Size": "1024;1",
                            "OpenAtStartup": "False",
                            "Usage": "Default",
                        },
                        {
                            "GUID": "bbfbd5b4-6ea8-43de-8760-3b91c5867333",
                            "Driver": "SpectraHub-Driver",
                            "Name": "Spectra-Hub_COM7",
                            "DisplayName": "Name",
                            "SerialNumber": "210367",
                            "DetectorType": "SinglePointDetector",
                            "Size": "1;1",
                            "OpenAtStartup": "False",
                            "Usage": "Default",
                        },
                        {
                            "GUID": "98abd847-6843-4c4c-b88c-7340de27ce13",
                            "Driver": "SpectraHub-Driver",
                            "Name": "Spectra-Hub_COM8",
                            "DisplayName": "Name",
                            "SerialNumber": "210238",
                            "DetectorType": "SinglePointDetector",
                            "Size": "1;1",
                            "OpenAtStartup": "False",
                            "Usage": "Default",
                        },
                    ],
                    "Count": "3",
                },
                "SinglePointDetectors": {"Count": "0"},
                "LightSources": {
                    "LightSource": [
                        {
                            "Wavelengths": {"Count": "1", "Value_0": "0.000"},
                            "GUID": "a57ac71a-c667-4501-a24a-596bc54b2c8c",
                            "Name": "Calibration Lamp",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "False",
                            "Type": "CalibrationLamp",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "0"},
                            "GUID": "cac03351-735e-4592-9bb2-c17f64b3649f",
                            "Name": "Calibration Lamp2",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "False",
                            "Type": "CalibrationLamp",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "0"},
                            "GUID": "84a08bee-227a-46f0-b1e1-0f171808a155",
                            "Name": "Calibration Lamp3",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "False",
                            "Type": "CalibrationLamp",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "1", "Value_0": "0.000"},
                            "GUID": "a0601d22-23a0-4587-bb79-35d5823916c4",
                            "Name": "Laser",
                            "Serialnumber": "",
                            "FixedWavelength": "False",
                            "CanChange": "False",
                            "Type": "Laser",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "1", "Value_0": "790.000"},
                            "GUID": "a36b29da-0d54-4ab7-860c-967269f76217",
                            "Name": "tisa",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "True",
                            "Type": "Laser",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                    ],
                    "Count": "5",
                },
                "ArduinoHardware": {"Count": "0"},
                "Additionals": {"Count": "0"},
                "MotorizedStages": {
                    "MotorizedStage": {
                        "GUID": "c551372e-56af-4686-b7c1-48ca00dfdf27",
                        "DriverName": "Tango Driver",
                        "Name": "XY-Stage",
                        "ConnectionString": "COM15",
                        "Serialnumber": "144012056",
                        "IsTriggerModeAvailable": "False",
                    },
                    "Count": "1",
                },
                "Microscopes": {
                    "Microscope": [
                        {
                            "Objectives": {
                                "Objective": [
                                    {
                                        "IsEnabled": "True",
                                        "Name": "100x",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                ],
                                "Count": "10",
                            },
                            "GUID": "841bfef9-b969-43f0-9ca3-2522c223a31f",
                            "Name": "Microscope",
                            "StageID": "c551372e-56af-4686-b7c1-48ca00dfdf27",
                            "ImagingCameraID": "",
                            "RamanBoxAvailable": "True",
                            "RamanBoxLinearStage.Available": "False",
                            "RamanBoxLinearStage.ConnectedHarwareID": "",
                            "RamanBoxFilterWheel.Available": "False",
                            "RamanBoxFilterWheel.ConnectedHarwareID": "",
                            "RamanBoxBeamSplitter.Available": "False",
                            "RamanBoxBeamSplitter.ConnectedHarwareID": "",
                            "RamanBoxNotchFilter.Available": "False",
                            "RamanBoxNotchFilter.ConnectedHarwareID": "",
                            "ActiveObjectiveIndex": "0",
                        },
                        {
                            "Objectives": {
                                "Objective": [
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                    {
                                        "IsEnabled": "False",
                                        "Name": "",
                                        "Magnification": "100.000",
                                        "FactorMeasuredWithResolution": "1280",
                                        "Factor": "34.700",
                                        "LaserOffsetX": "0.000",
                                        "LaserOffsetY": "0.000",
                                        "OffsetX": "0.000",
                                        "OffsetY": "0.000",
                                        "LineLaserOffsetX": "0.000",
                                        "LineLaserOffsetY": "0.000",
                                        "LineOffsetX": "0.000",
                                        "LineOffsetY": "0.000",
                                        "LineFocusFactor": "1.130",
                                    },
                                ],
                                "Count": "10",
                            },
                            "GUID": "ee496960-089d-4040-a5ca-cde5f9d1b0f8",
                            "Name": "Microscope",
                            "StageID": "",
                            "ImagingCameraID": "",
                            "RamanBoxAvailable": "False",
                            "RamanBoxLinearStage.Available": "False",
                            "RamanBoxLinearStage.ConnectedHarwareID": "",
                            "RamanBoxFilterWheel.Available": "False",
                            "RamanBoxFilterWheel.ConnectedHarwareID": "",
                            "RamanBoxBeamSplitter.Available": "False",
                            "RamanBoxBeamSplitter.ConnectedHarwareID": "",
                            "RamanBoxNotchFilter.Available": "False",
                            "RamanBoxNotchFilter.ConnectedHarwareID": "",
                            "ActiveObjectiveIndex": "0",
                        },
                    ],
                    "Count": "2",
                },
                "Cryostats": {"Count": "0"},
                "Version": "2",
            },
        }
        assert (
            expected_metadata
            == self.s_non_uniform_unfiltered.original_metadata.as_dictionary()
        )


class TestLinescan:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_linescan_path, reader="TriVista", use_uniform_signal_axis=True
        )
        cls.s_non_uniform = hs.load(
            testfile_linescan_path, reader="TriVista", use_uniform_signal_axis=False
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        del cls.s_non_uniform
        gc.collect()

    def test_data(self):
        expected_data_line1 = np.array([448, 482, 433, 407, 379])
        np.testing.assert_allclose(expected_data_line1, self.s.inav[0].isig[:5].data)
        np.testing.assert_allclose(
            expected_data_line1, self.s_non_uniform.inav[0].isig[:5].data
        )
        expected_data_last_line = np.array([430, 437, 378, 408, 488])
        np.testing.assert_allclose(
            expected_data_last_line, self.s.inav[-1].isig[-5:].data
        )
        np.testing.assert_allclose(
            expected_data_last_line, self.s_non_uniform.inav[-1].isig[-5:].data
        )

    def test_axes(self):
        expected_axis = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "X",
                "units": "µm",
                "navigate": True,
                "is_binned": False,
                "size": 21,
                "scale": 0.001,
                "offset": -0.01,
            },
            "axis-1": {
                "_type": "UniformDataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
                "size": 97,
            },
        }

        expected_axis_non_uniform = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "X",
                "units": "µm",
                "navigate": True,
                "is_binned": False,
                "size": 21,
                "scale": 0.001,
                "offset": -0.01,
            },
            "axis-1": {
                "_type": "DataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
            },
        }
        expected_values_non_uniform_axis = np.array(
            [910.310424804688, 910.350219726563, 910.390014648438]
        )

        uniform_axes_manager = deepcopy(self.s.axes_manager.as_dictionary())
        non_uniform_axis_manager = deepcopy(
            self.s_non_uniform.axes_manager.as_dictionary()
        )

        np.testing.assert_allclose(
            uniform_axes_manager["axis-1"].pop("scale"), 0.0397771, atol=0.0000005
        )
        np.testing.assert_allclose(
            uniform_axes_manager["axis-1"].pop("offset"), 910.3104, atol=0.005
        )
        np.testing.assert_allclose(
            expected_values_non_uniform_axis,
            non_uniform_axis_manager["axis-1"].pop("axis")[:3],
        )
        assert expected_axis == uniform_axes_manager
        assert expected_axis_non_uniform == non_uniform_axis_manager

    def test_original_metadata(self):
        assert (
            self.s.original_metadata.as_dictionary()
            == self.s_non_uniform.original_metadata.as_dictionary()
        )

        metadata_experiment = (
            self.s.original_metadata.Document.InfoSerialized.Experiment.as_dictionary()
        )

        assert metadata_experiment["Mode"] == "Point by Point"
        assert metadata_experiment["Stage Mode"] == "LineScanX"
        assert metadata_experiment["Used Time"] == "00:00:21"


class TestMap:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_map_path, reader="TriVista", use_uniform_signal_axis=True
        )
        cls.s_non_uniform = hs.load(
            testfile_map_path, reader="TriVista", use_uniform_signal_axis=False
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        del cls.s_non_uniform
        gc.collect()

    def test_data(self):
        expected_data_00 = np.array([531, 499, 442, 446, 460])
        np.testing.assert_allclose(expected_data_00, self.s.inav[0, 0].isig[:5].data)
        np.testing.assert_allclose(
            expected_data_00, self.s_non_uniform.inav[0, 0].isig[:5].data
        )

        expected_data_10 = np.array([532, 516, 425, 438, 433])
        np.testing.assert_allclose(expected_data_10, self.s.inav[1, 0].isig[:5].data)
        np.testing.assert_allclose(
            expected_data_10, self.s_non_uniform.inav[1, 0].isig[:5].data
        )

        expected_data_01 = np.array([525, 498, 447, 433, 452])
        np.testing.assert_allclose(expected_data_01, self.s.inav[0, 1].isig[:5].data)
        np.testing.assert_allclose(
            expected_data_01, self.s_non_uniform.inav[0, 1].isig[:5].data
        )

        expected_data_99 = np.array([452, 478, 687, 1320, 1145])
        np.testing.assert_allclose(expected_data_99, self.s.inav[-1, -1].isig[-5:].data)
        np.testing.assert_allclose(
            expected_data_99, self.s_non_uniform.inav[-1, -1].isig[-5:].data
        )

    def test_axes(self):
        expected_axis = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "Y",
                "units": "µm",
                "navigate": True,
                "is_binned": False,
                "size": 9,
                "scale": 0.025,
                "offset": -0.1,
            },
            "axis-1": {
                "_type": "UniformDataAxis",
                "name": "X",
                "units": "µm",
                "navigate": True,
                "is_binned": False,
                "size": 9,
                "scale": 0.025,
                "offset": -0.1,
            },
            "axis-2": {
                "_type": "UniformDataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
                "size": 1024,
            },
        }

        expected_axis_non_uniform = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "Y",
                "units": "µm",
                "navigate": True,
                "is_binned": False,
                "size": 9,
                "scale": 0.025,
                "offset": -0.1,
            },
            "axis-1": {
                "_type": "UniformDataAxis",
                "name": "X",
                "units": "µm",
                "navigate": True,
                "is_binned": False,
                "size": 9,
                "scale": 0.025,
                "offset": -0.1,
            },
            "axis-2": {
                "_type": "DataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
            },
        }
        expected_values_non_uniform_axis = np.array(
            [891.894470214844, 891.9345703125, 891.974670410156]
        )

        uniform_axes_manager = deepcopy(self.s.axes_manager.as_dictionary())
        non_uniform_axis_manager = deepcopy(
            self.s_non_uniform.axes_manager.as_dictionary()
        )

        np.testing.assert_allclose(
            uniform_axes_manager["axis-2"].pop("scale"), 0.039770, atol=0.000005
        )
        np.testing.assert_allclose(
            uniform_axes_manager["axis-2"].pop("offset"), 891.9, atol=0.05
        )
        np.testing.assert_allclose(
            expected_values_non_uniform_axis,
            non_uniform_axis_manager["axis-2"].pop("axis")[:3],
        )
        assert expected_axis == uniform_axes_manager
        assert expected_axis_non_uniform == non_uniform_axis_manager

    def test_original_metadata(self):
        assert (
            self.s.original_metadata.as_dictionary()
            == self.s_non_uniform.original_metadata.as_dictionary()
        )

        metadata_experiment = (
            self.s.original_metadata.Document.InfoSerialized.Experiment.as_dictionary()
        )

        assert metadata_experiment["Mode"] == "Point by Point"
        assert metadata_experiment["Stage Mode"] == "MappingXY"
        assert metadata_experiment["Used Time"] == "00:01:27"


class Test3Spectrometers:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_triple_add_path,
            reader="TriVista",
            use_uniform_signal_axis=True,
            filter_original_metadata=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        gc.collect()

    def test_original_metadata(self):
        expected_metadata = {
            "XmlMain": {
                "Version": "1.0.1",
                "Filename": "D:\\Data\\test-spectra\\spec_triple-add.tvf",
                "DateTime": "06/27/2022 17:56:15",
                "PlugInName": "",
                "PlugInData": "",
                "FileInfoSerialized": {"Info": {"Groups": None}},
            },
            "Document": {
                "Version": "2",
                "Label": "Intensity",
                "DataLabel": "Counts",
                "DocType": "Spectra",
                "RecordTime": "06/27/2022 17:55:55.294",
                "ModeName": "",
                "DataSource": "nothing",
                "Encoding": "Ascii",
                "ColorMode": "Grayscale",
                "ViewDisplayMode": "Graph",
                "ViewImageColorMode": "False",
                "InfoSerialized": {
                    "Document": {"Record Time": "6/27/2022 5:55 PM"},
                    "Experiment": {"Used Setup": "PL_add777"},
                    "Spectrometers": {
                        "Spectrometer2": {
                            "Serialnumber": "25580420",
                            "Model": "SP-2-500i",
                            "Stage_Number": "2",
                            "Focallength": "500",
                            "Inclusion_Angle": "8.6",
                            "Detector_Angle": "0",
                            "Groove_Density": "750 g/mm",
                            "Order": "1",
                            "Slit_Entrance-Front": "0",
                            "Slit_Entrance-Side": "6000",
                            "Slit_Exit-Front": "0",
                            "Slit_Exit-Side": "0",
                        },
                        "Spectrometer3": {
                            "Serialnumber": "27580185",
                            "Model": "SP-2-750i",
                            "Stage_Number": "3",
                            "Focallength": "749",
                            "Inclusion_Angle": "6.5",
                            "Detector_Angle": "0.68",
                            "Groove_Density": "750 g/mm",
                            "Order": "1",
                            "Slit_Entrance-Front": "0",
                            "Slit_Entrance-Side": "12000",
                            "Slit_Exit-Front": "0",
                            "Slit_Exit-Side": "0",
                        },
                        "Spectrometer1": {
                            "Serialnumber": "25580419",
                            "Model": "SP-2-500i",
                            "Stage_Number": "1",
                            "Focallength": "500",
                            "Inclusion_Angle": "8.6",
                            "Detector_Angle": "0",
                            "Groove_Density": "750 g/mm",
                            "Order": "1",
                            "Slit_Entrance-Front": "300",
                            "Slit_Entrance-Side": "0",
                            "Slit_Exit-Front": "0",
                            "Slit_Exit-Side": "0",
                        },
                    },
                    "Detector": {
                        "Name": "Camera1",
                        "Serialnumber": None,
                        "Detector_Size": "1024;1",
                        "Detector_Temperature": "-25",
                        "Exposure_Time_(ms)": "1000",
                        "Exposure_Mode": None,
                        "No_of_Accumulations": "1",
                        "Calc_Average": "True",
                        "No_of_Frames": "1",
                        "ADC__Readout_Port": "Normal",
                        "ADC__Rate_Resolution": "1 MHz",
                        "ADC__Gain": "2",
                        "Clearing__Mode": None,
                        "Clearing__No_of_Cleans": "1",
                        "Region_of_Interests": "1|1;1024;1;1;1;1",
                    },
                    "Calibration": {
                        "Center_Wavelength": "820.000",
                        "Laser_Wavelength": "0.000",
                    },
                },
            },
            "Hardware": {
                "Spectrometers": {
                    "Count": "3",
                    "Spectrometer1": {
                        "Gratings": {
                            "Grating": {
                                "Offsets": {"Count": "0"},
                                "GrooveDensity": "750 g/mm",
                                "Blaze": "H-NIR",
                                "Turret": "1",
                            },
                            "Count": "3",
                        },
                        "MirrorEntrance": {
                            "Name": "M1",
                            "State": "isMotorized",
                            "Position": "front",
                        },
                        "MirrorExit": {
                            "Name": "M2",
                            "State": "isMotorized",
                            "Position": "side",
                        },
                        "SlitEntranceSide": {
                            "Name": "S1",
                            "State": "isMotorized",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "SlitEntranceFront": {
                            "Name": "S2",
                            "State": "isMotorized",
                            "MaxWidth": "12000",
                            "Position": "200",
                        },
                        "SlitExitFront": {
                            "Name": "S3",
                            "State": "isMotorized",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "SlitExitSide": {
                            "Name": "S4",
                            "State": "disabled",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "FilterWheel": {
                            "Name": "",
                            "State": "disabled",
                            "Position": "0",
                            "FilterStrings": "6|#1|#2|#3|#4|#5|#6",
                        },
                        "Shutter": {
                            "Name": "",
                            "State": "disabled",
                            "Position": "open",
                            "ShutterMode": "OpenForExperiment",
                            "ShutterClosedForFilterwheel": "True",
                            "ShutterClosedForSetupChange": "True",
                            "ShutterClosedForLaserCrossing": "True",
                            "ShutterClosedForLaserCrossingValue": "0.000",
                            "ShutterClosedForLaserCrossingUnit": "Nanometer",
                        },
                        "GUID": "c05fbec6-7cc0-4546-a009-06850868f43d",
                        "Model": "SP-2-500i",
                        "Serialnumber": "25580419",
                        "FocalLength": "500",
                        "InclusionAngle": "8.600",
                        "InclusionAngleAdditive": "51.600",
                        "DetectorAngle": "0.000",
                        "PixelWidthExitFront": "-1.000",
                        "PixelWidthExitSide": "-1.000",
                        "Backlash": "ScanUp",
                        "DriveDirection": "Normal",
                        "ComPortSpec": "COM3",
                        "ComPortEntrance": "",
                        "ComPortExit": "",
                        "DemoMode": "False",
                        "UsedAsLightsource": "False",
                        "UseBackwards": "False",
                        "Wavelength": "925.010",
                        "Grating": "2",
                        "ReadTimeout": "35000",
                        "Activated": "False",
                        "IsFixedCalibration": "False",
                        "FixedCalibrationPoints": "0",
                    },
                    "Spectrometer2": {
                        "Gratings": {
                            "Grating": {
                                "Offsets": {"Count": "0"},
                                "GrooveDensity": "750 g/mm",
                                "Blaze": "H-NIR",
                                "Turret": "1",
                            },
                            "Count": "3",
                        },
                        "MirrorEntrance": {
                            "Name": "M1",
                            "State": "isMotorized",
                            "Position": "side",
                        },
                        "MirrorExit": {
                            "Name": "M2",
                            "State": "isMotorized",
                            "Position": "side",
                        },
                        "SlitEntranceSide": {
                            "Name": "S1",
                            "State": "isMotorized",
                            "MaxWidth": "12000",
                            "Position": "10000",
                        },
                        "SlitEntranceFront": {
                            "Name": "S2",
                            "State": "isMotorized",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "SlitExitFront": {
                            "Name": "S3",
                            "State": "isMotorized",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "SlitExitSide": {
                            "Name": "S4",
                            "State": "disabled",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "FilterWheel": {
                            "Name": "",
                            "State": "disabled",
                            "Position": "0",
                            "FilterStrings": "6|#1|#2|#3|#4|#5|#6",
                        },
                        "Shutter": {
                            "Name": "",
                            "State": "disabled",
                            "Position": "open",
                            "ShutterMode": "OpenForExperiment",
                            "ShutterClosedForFilterwheel": "True",
                            "ShutterClosedForSetupChange": "True",
                            "ShutterClosedForLaserCrossing": "True",
                            "ShutterClosedForLaserCrossingValue": "0.000",
                            "ShutterClosedForLaserCrossingUnit": "Nanometer",
                        },
                        "GUID": "39ade9ad-caa8-48c2-a3d0-4f79cc44bda7",
                        "Model": "SP-2-500i",
                        "Serialnumber": "25580420",
                        "FocalLength": "500",
                        "InclusionAngle": "8.600",
                        "InclusionAngleAdditive": "51.600",
                        "DetectorAngle": "0.000",
                        "PixelWidthExitFront": "1.000",
                        "PixelWidthExitSide": "-1.000",
                        "Backlash": "ScanUp",
                        "DriveDirection": "Normal",
                        "ComPortSpec": "COM4",
                        "ComPortEntrance": "",
                        "ComPortExit": "",
                        "DemoMode": "False",
                        "UsedAsLightsource": "False",
                        "UseBackwards": "False",
                        "Wavelength": "-924.910",
                        "Grating": "2",
                        "ReadTimeout": "35000",
                        "Activated": "False",
                        "IsFixedCalibration": "False",
                        "FixedCalibrationPoints": "0",
                    },
                    "Spectrometer3": {
                        "Gratings": {
                            "Grating": {
                                "Offsets": {"Count": "0"},
                                "GrooveDensity": "750 g/mm",
                                "Blaze": "H-NIR",
                                "Turret": "1",
                            },
                            "Count": "3",
                        },
                        "MirrorEntrance": {
                            "Name": "M1",
                            "State": "isMotorized",
                            "Position": "front",
                        },
                        "MirrorExit": {
                            "Name": "M2",
                            "State": "isMotorized",
                            "Position": "side",
                        },
                        "SlitEntranceSide": {
                            "Name": "S1",
                            "State": "isMotorized",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "SlitEntranceFront": {
                            "Name": "S2",
                            "State": "isMotorized",
                            "MaxWidth": "12000",
                            "Position": "500",
                        },
                        "SlitExitFront": {
                            "Name": "S3",
                            "State": "disabled",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "SlitExitSide": {
                            "Name": "S4",
                            "State": "disabled",
                            "MaxWidth": "12000",
                            "Position": "0",
                        },
                        "FilterWheel": {
                            "Name": "",
                            "State": "disabled",
                            "Position": "0",
                            "FilterStrings": "6|#1|#2|#3|#4|#5|#6",
                        },
                        "Shutter": {
                            "Name": "",
                            "State": "disabled",
                            "Position": "open",
                            "ShutterMode": "OpenForExperiment",
                            "ShutterClosedForFilterwheel": "True",
                            "ShutterClosedForSetupChange": "True",
                            "ShutterClosedForLaserCrossing": "True",
                            "ShutterClosedForLaserCrossingValue": "0.000",
                            "ShutterClosedForLaserCrossingUnit": "Nanometer",
                        },
                        "GUID": "37a2855b-db41-4268-970b-bb75d96e5654",
                        "Model": "SP-2-750i",
                        "Serialnumber": "27580185",
                        "FocalLength": "749",
                        "InclusionAngle": "6.500",
                        "InclusionAngleAdditive": "39.000",
                        "DetectorAngle": "0.680",
                        "PixelWidthExitFront": "25.000",
                        "PixelWidthExitSide": "-1.000",
                        "Backlash": "ScanUp",
                        "DriveDirection": "Normal",
                        "ComPortSpec": "COM5",
                        "ComPortEntrance": "",
                        "ComPortExit": "",
                        "DemoMode": "False",
                        "UsedAsLightsource": "False",
                        "UseBackwards": "False",
                        "Wavelength": "850.184",
                        "Grating": "2",
                        "ReadTimeout": "35000",
                        "Activated": "False",
                        "IsFixedCalibration": "False",
                        "FixedCalibrationPoints": "0",
                    },
                },
                "Detectors": {
                    "Detector": {
                        "GUID": "c1bc9fd2-135e-4098-b681-d30f6ea8be82",
                        "Driver": "PvCam32 Driver",
                        "Name": "Camera1",
                        "DisplayName": "Camera1",
                        "SerialNumber": "",
                        "DetectorType": "None",
                        "Size": "1024;1",
                        "OpenAtStartup": "False",
                        "Usage": "Default",
                    },
                    "Count": "3",
                },
                "SinglePointDetectors": {"Count": "0"},
                "LightSources": {
                    "LightSource": [
                        {
                            "Wavelengths": {"Count": "1", "Value_0": "0.000"},
                            "GUID": "a57ac71a-c667-4501-a24a-596bc54b2c8c",
                            "Name": "Calibration Lamp",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "False",
                            "Type": "CalibrationLamp",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "0"},
                            "GUID": "cac03351-735e-4592-9bb2-c17f64b3649f",
                            "Name": "Calibration Lamp2",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "False",
                            "Type": "CalibrationLamp",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "0"},
                            "GUID": "84a08bee-227a-46f0-b1e1-0f171808a155",
                            "Name": "Calibration Lamp3",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "False",
                            "Type": "CalibrationLamp",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "1", "Value_0": "0.000"},
                            "GUID": "a0601d22-23a0-4587-bb79-35d5823916c4",
                            "Name": "Laser",
                            "Serialnumber": "",
                            "FixedWavelength": "False",
                            "CanChange": "False",
                            "Type": "Laser",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                        {
                            "Wavelengths": {"Count": "1", "Value_0": "790.000"},
                            "GUID": "a36b29da-0d54-4ab7-860c-967269f76217",
                            "Name": "tisa",
                            "Serialnumber": "",
                            "FixedWavelength": "True",
                            "CanChange": "True",
                            "Type": "Laser",
                            "ConnectionString": "None",
                            "IsMirrorPositionValid": "False",
                            "MirrorsAlignmentPositionTop": "0;0",
                            "MirrorsAlignmentPositionBottom": "0;0",
                        },
                    ],
                    "Count": "5",
                },
                "ArduinoHardware": {"Count": "0"},
                "Additionals": {"Count": "0"},
                "MotorizedStages": {
                    "MotorizedStage": {
                        "GUID": "c551372e-56af-4686-b7c1-48ca00dfdf27",
                        "DriverName": "Tango Driver",
                        "Name": "XY-Stage",
                        "ConnectionString": "COM15",
                        "Serialnumber": "144012056",
                        "IsTriggerModeAvailable": "False",
                    },
                    "Count": "1",
                },
                "Microscopes": {
                    "Microscope": {
                        "Objectives": {
                            "Objective": {
                                "IsEnabled": "True",
                                "Name": "100x",
                                "Magnification": "100.000",
                                "FactorMeasuredWithResolution": "1280",
                                "Factor": "34.700",
                                "LaserOffsetX": "0.000",
                                "LaserOffsetY": "0.000",
                                "OffsetX": "0.000",
                                "OffsetY": "0.000",
                                "LineLaserOffsetX": "0.000",
                                "LineLaserOffsetY": "0.000",
                                "LineOffsetX": "0.000",
                                "LineOffsetY": "0.000",
                                "LineFocusFactor": "1.130",
                            },
                            "Count": "10",
                        },
                        "GUID": "841bfef9-b969-43f0-9ca3-2522c223a31f",
                        "Name": "Microscope",
                        "StageID": "c551372e-56af-4686-b7c1-48ca00dfdf27",
                        "ImagingCameraID": "",
                        "RamanBoxAvailable": "True",
                        "RamanBoxLinearStage.Available": "False",
                        "RamanBoxLinearStage.ConnectedHarwareID": "",
                        "RamanBoxFilterWheel.Available": "False",
                        "RamanBoxFilterWheel.ConnectedHarwareID": "",
                        "RamanBoxBeamSplitter.Available": "False",
                        "RamanBoxBeamSplitter.ConnectedHarwareID": "",
                        "RamanBoxNotchFilter.Available": "False",
                        "RamanBoxNotchFilter.ConnectedHarwareID": "",
                        "ActiveObjectiveIndex": "0",
                    },
                    "Count": "2",
                },
                "Cryostats": {"Count": "0"},
                "Version": "2",
            },
        }

        assert self.s.original_metadata.as_dictionary() == expected_metadata

    def test_metadata_acquisition_instrument(self):
        expected_metadata = {
            "Detector": {
                "processing": {"calc_average": "True"},
                "temperature": -25.0,
                "exposure_per_frame": 1.0,
                "frames": 1.0,
                "glued_spectrum": False,
                "integration_time": 1.0,
            },
            "Laser": {"objective_magnification": 100.0},
            "Spectrometer1": {
                "Grating": {"blazing_wavelength": "H-NIR", "groove_density": 750.0},
                "central_wavelength": 820.0,
                "model": "SP-2-500i",
                "entrance_slit_width": 0.3,
                "exit_slit_width": 0.0,
            },
            "Spectrometer2": {
                "Grating": {"blazing_wavelength": "H-NIR", "groove_density": 750.0},
                "central_wavelength": 820.0,
                "model": "SP-2-500i",
                "entrance_slit_width": 6.0,
                "exit_slit_width": 0.0,
            },
            "Spectrometer3": {
                "Grating": {"blazing_wavelength": "H-NIR", "groove_density": 750.0},
                "central_wavelength": 820.0,
                "model": "SP-2-750i",
                "entrance_slit_width": 12.0,
                "exit_slit_width": 0.0,
            },
        }

        assert (
            self.s.metadata.Acquisition_instrument.as_dictionary() == expected_metadata
        )


class TestStepAndGlue:
    @classmethod
    def setup_class(cls):
        cls.glued = hs.load(
            testfile_step_and_glue_path,
            reader="TriVista",
            use_uniform_signal_axis=True,
            filter_original_metadata=True,
            glued_data_as_stack=False,
        )
        cls.stack = hs.load(
            testfile_step_and_glue_path,
            reader="TriVista",
            use_uniform_signal_axis=False,
            filter_original_metadata=True,
            glued_data_as_stack=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.glued
        del cls.stack
        gc.collect()

    def test_data_glued(self):
        assert np.isclose(self.glued.isig[0].data, 1071)
        assert np.isclose(self.glued.isig[17999].data, 978)

    def test_data_stack(self):
        data_stack1 = np.array([1071, 1134, 1092])
        data_stack19 = np.array([1064, 1150, 1111])
        np.testing.assert_allclose(self.stack[0].isig[:3].data, data_stack1)
        np.testing.assert_allclose(self.stack[18].isig[:3].data, data_stack19)

    def test_axes_stack(self):
        expected_axis = {
            "axis-0": {
                "_type": "DataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
            }
        }
        for frame in self.stack:
            axes_manager = frame.axes_manager.as_dictionary()
            data_axis = axes_manager["axis-0"].pop("axis")
            assert data_axis.size == 1024
            assert axes_manager == expected_axis
        assert np.isclose(
            self.stack[0].axes_manager.as_dictionary()["axis-0"]["axis"][0],
            900.027,
            atol=0.001,
        )

    def test_metadata(self):
        original_metadata_glued = self.glued.original_metadata.Document.InfoSerialized.Experiment.as_dictionary()  # noqa: E501

        metadata_detector = self.glued.metadata.Acquisition_instrument.Detector

        assert original_metadata_glued["From"] == "900.000 nm"
        assert original_metadata_glued["To"] == "1500.000 nm"
        assert original_metadata_glued["Mode:"] == "Step & Glue"
        assert original_metadata_glued["Overlap (%)"] == "15"
        assert original_metadata_glued["Skipped Pixel Left"] == "0"
        assert original_metadata_glued["Skipped Pixel Right"] == "0"

        assert metadata_detector.glued_spectrum is True
        assert np.isclose(metadata_detector.glued_spectrum_overlap, 15)
        assert np.isclose(metadata_detector.glued_spectrum_windows, 19)

        metadata_glued = deepcopy(self.glued.metadata.as_dictionary())
        del metadata_glued["General"]["FileIO"]
        for frame in self.stack:
            assert (
                self.glued.original_metadata.as_dictionary()
                == frame.original_metadata.as_dictionary()
            )
            metadata_frame = deepcopy(frame.metadata.as_dictionary())
            del metadata_frame["General"]["FileIO"]
            assert metadata_frame == metadata_glued


class TestTimeSeries:
    @classmethod
    def setup_class(cls):
        cls.timeseries = hs.load(
            testfile_spec_timeseries_path,
            reader="TriVista",
            use_uniform_signal_axis=True,
            filter_original_metadata=True,
        )
        cls.frames = hs.load(
            testfile_spec_2frames_path,
            reader="TriVista",
            use_uniform_signal_axis=True,
            filter_original_metadata=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.timeseries
        del cls.frames
        gc.collect()

    def test_data_timeseries(self):
        data_ts_1 = np.array([27381, 39153, 35022, -425, 32430])
        data_ts_2 = np.array([27449.5, 38971.5, 35168.5, -425, 32484.5])
        np.testing.assert_allclose(self.timeseries.isig[:5, 0].data, data_ts_1)
        np.testing.assert_allclose(self.timeseries.isig[:5, 1].data, data_ts_2)

    def test_data_frames(self):
        data_frame_1 = np.array([49868, 65074, 60325, 10241, 52539])
        data_frame_2 = np.array([49909, 65074, 60017, 10022, 52639])
        np.testing.assert_allclose(self.frames.isig[:5, 0].data, data_frame_1)
        np.testing.assert_allclose(self.frames.isig[:5, 1].data, data_frame_2)

    def test_axes_timeseries(self):
        expected_axis = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "time",
                "units": "s",
                "navigate": False,
                "is_binned": False,
                "size": 2,
                "offset": 0.0,
            },
            "axis-1": {
                "_type": "UniformDataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
                "size": 1024,
            },
        }

        axes_manager = deepcopy(self.timeseries.axes_manager.as_dictionary())

        np.testing.assert_allclose(axes_manager["axis-0"].pop("scale"), 4.04, atol=0.01)
        np.testing.assert_allclose(axes_manager["axis-1"].pop("scale"), 0.03, atol=0.01)
        np.testing.assert_allclose(axes_manager["axis-1"].pop("offset"), 934, atol=1)
        assert expected_axis == axes_manager

    def test_axes_frames(self):
        expected_axis = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "time",
                "units": "s",
                "navigate": False,
                "is_binned": False,
                "size": 2,
                "offset": 0.0,
            },
            "axis-1": {
                "_type": "UniformDataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
                "size": 1024,
            },
        }

        axes_manager = deepcopy(self.frames.axes_manager.as_dictionary())

        np.testing.assert_allclose(axes_manager["axis-1"].pop("scale"), 0.03, atol=0.01)
        np.testing.assert_allclose(axes_manager["axis-1"].pop("offset"), 934, atol=1)
        np.testing.assert_allclose(axes_manager["axis-0"].pop("scale"), 6, atol=0.01)
        assert expected_axis == axes_manager


class TestSpecIntegrationTime:
    @classmethod
    def setup_class(cls):
        cls.s_2acc = hs.load(
            testfile_spec_2acc_path,
            reader="TriVista",
            use_uniform_signal_axis=True,
            filter_original_metadata=True,
            glued_data_as_stack=True,
        )
        ## glued_data_as_stack set to True here
        ## to ensure that this setting doesn't affect
        ## non-glued datasets
        cls.s_2acc_no_average = hs.load(
            testfile_spec_2acc_no_average_path,
            reader="TriVista",
            use_uniform_signal_axis=True,
            filter_original_metadata=True,
            glued_data_as_stack=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s_2acc
        del cls.s_2acc_no_average
        gc.collect()

    def test_metadata_2acc(self):
        metadata_detector = self.s_2acc.metadata.Acquisition_instrument.Detector
        assert np.isclose(metadata_detector.exposure_per_frame, 3)
        assert np.isclose(metadata_detector.frames, 2)
        assert np.isclose(metadata_detector.integration_time, 3)
        assert metadata_detector.processing.calc_average == "True"

    def test_metadata_2acc_no_average(self):
        metadata_detector = (
            self.s_2acc_no_average.metadata.Acquisition_instrument.Detector
        )
        assert np.isclose(metadata_detector.exposure_per_frame, 3)
        assert np.isclose(metadata_detector.frames, 2)
        assert np.isclose(metadata_detector.integration_time, 6)
        assert metadata_detector.processing.calc_average == "False"
