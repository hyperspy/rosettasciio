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
import importlib
from pathlib import Path

import numpy as np
import pytest

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")

testfile_dir = (Path(__file__).parent / "data" / "hamamatsu").resolve()

testfile_focus_mode_path = (testfile_dir / "focus_mode.img").resolve()
testfile_operate_mode_path = (testfile_dir / "operate_mode.img").resolve()
testfile_photon_count_path = (testfile_dir / "photon_counting.img").resolve()
testfile_shading_path = (testfile_dir / "shading_file.img").resolve()


class TestOperate:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(testfile_operate_mode_path, reader="Hamamatsu")

    @classmethod
    def teardown_class(cls):
        del cls.s
        gc.collect()

    def test_data_deflection(self):
        expected_data_start = [
            [0, 0, 715],
            [246, 161, 475],
            [106, 899, 0],
        ]
        np.testing.assert_allclose(expected_data_start, self.s.isig[:3, :3].data)

    def test_axes_deflection(self):
        axes = self.s.axes_manager
        assert axes.signal_dimension == 2
        assert axes.navigation_dimension == 0
        ax0 = axes[0]
        ax1 = axes[1]
        assert ax0.name == "Wavelength"
        assert ax1.name == "Time"
        assert ax0.size == 672
        assert ax1.size == 512
        assert ax0.units == "nm"
        assert ax1.units == "µs"

        expected_data_start_X = [472.252, 472.33337, 472.41473, 472.4961, 472.57745]
        expected_data_start_Y = [0.0, 0.031080816, 0.062164314, 0.09325048, 0.12433932]
        np.testing.assert_allclose(ax0.axis[:5], expected_data_start_X)
        np.testing.assert_allclose(ax1.axis[:5], expected_data_start_Y)

    def test_original_metadata_no_comment(self):
        original_metadata_no_comment = {
            k: v
            for (k, v) in self.s.original_metadata.as_dictionary().items()
            if k != "Comment"
        }
        expected_metadata = {
            "character_im": "IM",
            "offset_x": 0,
            "offset_y": 0,
            "file_type": "bit32",
            "num_images_in_channel": 0,
            "num_additional_channels": 0,
            "channel_number": 0,
            "timestamp": 0.0,
            "marker": "",
            "additional_info": "",
            "image_width_px": 672,
            "image_height_lines": 512,
            "comment_length": 3311,
        }
        assert expected_metadata == original_metadata_no_comment

    def test_original_metadata_comment(self):
        original_metadata = self.s.original_metadata.Comment.as_dictionary()
        expected_metadata = {
            "Application": {
                "Enconding": "UTF-8",
                "Date": "20/01/2023",
                "Time": "11:34:55.147",
                "Software": "HPD-TA",
                "Application": "2",
                "ApplicationTitle": "High Performance Digital Temporal Analyzer",
                "SoftwareVersion": "9.5 pf10",
                "SoftwareDate": "15.12.2020",
            },
            "Camera": {
                "CameraName": "Orca R2",
                "SerialNumber": "S/N: 891892",
                "Type": "37",
                "ScanMode": "1",
                "Subarray": "0",
                "Binning": "2",
                "BitsPerChannel": "0",
                "HighDynamicRangeMode": "0",
                "ScanSpeed": "0",
                "SubarrayHpos": "0",
                "SubarrayHsize": "0",
                "SubarrayVpos": "0",
                "SubarrayVsize": "0",
                "TriggerMode": "3",
                "TriggerModeKeyVal": "Internal",
                "TriggerPolarity": "1",
                "TriggerPolarityKeyVal": "neg.",
                "CoolerSwitch": "-1",
                "Gain": "0",
                "Prop_SensorMode": "1",
                "Prop_Colortype": "1",
                "Prop_TriggerTimes": "1",
                "Prop_ExposureTimeControl": "2",
                "Prop_TimingMinTriggerInterval": "0.500139",
                "Prop_TimingExposure": "2",
                "Prop_ImageTopOffsetBytes": "0",
                "Prop_ImagePixelType": "2",
                "Prop_BufferRowbytes": "1344",
                "Prop_BufferFramebytes": "688128",
                "Prop_BufferTopOffsetBytes": "0",
                "Prop_BufferPixelType": "2",
                "Prop_RecordFixedBytesPerFile": "256",
                "Prop_RecordFixedBytesPerSession": "784",
                "Prop_RecordFixedBytesPerFrame": "688176",
                "Prop_NumberOfOutputTriggerConnector": "1",
                "Prop_OutputTriggerPolarity": "1",
                "Prop_OutputTriggerActive": "1",
                "Prop_OutputTriggerDelay": "0",
                "Prop_OutputTriggerPeriod": "0.0001",
                "Prop_SystemAlive": "2",
                "Prop_ImageDetectorPixelWidth": "6.45",
                "Prop_ImageDetectorPixelHeight": "6.45",
                "Prop_TimeStampProducer": "2",
                "Prop_FrameStampProducer": "2",
            },
            "Acquisition": {
                "NrExposure": "60",
                "NrTrigger": "0",
                "ExposureTime": "5 s",
                "AcqMode": "4",
                "DataType": "8",
                "DataTypeOfSingleImage": "7",
                "CurveCorr": "0",
                "DefectCorrection": "0",
                "areSource": "0,0,672,512",
                "areGRBScan": "0,0,672,512",
                "pntOrigCh": "0,0",
                "pntOrigFB": "0,0",
                "pntBinning": "2,2",
                "BytesPerPixel": "4",
                "IsLineData": "0",
                "BacksubCorr": "-1",
                "ShadingCorr": "0",
                "ZAxisLabel": "Intensity",
                "ZAxisUnit": "Count",
                "miMirrorRotate": "0",
            },
            "Grabber": {"Type": "5", "SubType": "0"},
            "DisplayLUT": {
                "EntrySize": "9",
                "LowerValue": "764",
                "UpperValue": "3562",
                "LowerValueEx": "0",
                "UpperValueEx": "32767",
                "BitRange": "16x bit",
                "Color": "2",
                "LUTType": "0",
                "LUTInverted": "0",
                "AutoLutInLiveMode": "0",
                "DisplayNegative": "0",
                "Gamma": "1",
                "First812OvlCol": "1",
                "Lut16xShift": "7",
                "Lut16xOvlVal": "3932100",
            },
            "ExternalDevices": {
                "TriggerDelay": "150",
                "PostTriggerTime": "10",
                "ExposureTime": "10",
                "TDStatusCableConnected": "0",
                "ConnectMonitorOut": "0",
                "ConnectResetIn": "0",
                "TriggerMethod": "2",
                "UseDTBE": "0",
                "ExpTimeAddMultiple": "-1",
                "DontSendReset": "0",
                "MultipleOfSweep": "1",
                "A6538Connected": "0",
                "CounterBoardInstalled": "0",
                "MotorizedSlitInstalled": "0",
                "UseSpecAsMono": "0",
                "GPIBInstalled": "-1",
                "CounterBoardIOBase": "0",
                "MotorizedSlitPortID": "1",
                "GPIBIOBase": "0",
            },
            "Streak camera": {
                "UseDevice": "-1",
                "DeviceName": "C5680",
                "PluginName": "M5677",
                "GPIBCableConnected": "-1",
                "GPIBBase": "8",
                "Time Range": "20 us",
                "Mode": "Operate",
                "Gate Mode": "Normal",
                "MCP Gain": "50",
                "Shutter": "Open",
                "Gate Time": "0",
                "Trig. Mode": "Cont",
                "Trigger status": "Ready",
                "Trig. level": "1",
                "Trig. slope": "Rising",
                "FocusTimeOver": "11",
            },
            "Spectrograph": {
                "UseDevice": "-1",
                "DeviceName": "Andor SG",
                "PluginName": "Kymera 328i",
                "GPIBCableConnected": "0",
                "GPIBBase": "13",
                "Wavelength": "500",
                "Grating": "300 g/mm",
                "Blaze": "500",
                "Ruling": "300",
                "Exit Mirror": "Front",
                "Side Ent. Slitw.": "10",
                "Turret": "1",
                "Focus Mirror": "234",
                "Side Entry Iris": "26",
            },
            "Delay box": {
                "UseDevice": "-1",
                "DeviceName": "C4792-01",
                "PluginName": "",
                "GPIBCableConnected": "0",
                "Trig. Mode": "Int. Trig.",
                "Repetition Rate": "1 Hz",
                "L-Pulsewidth": "300 ns",
                "Delay Int.Trig": "-999999999900",
                "Delay Ext.Trig": "0",
                "Dly Mode-Lock": "0",
                "Dly1 DmpMode": "0",
                "Dly2 DmpMode": "0",
            },
            "Delay2 box": {"UseDevice": "0"},
            "Filter wheel": {"UseDevice": "0"},
            "Scaling": {
                "ScalingXType": "2",
                "ScalingXScale": "1",
                "ScalingXUnit": "nm",
                "ScalingXScalingFile": "#1379631,0672",
                "ScalingYType": "2",
                "ScalingYScale": "1",
                "ScalingYUnit": "us",
                "ScalingYScalingFile": "#1382319,0512",
            },
            "Comment": {"UserComment": ""},
        }

        assert expected_metadata == original_metadata

    def test_metadata(self):
        metadata = self.s.metadata

        detector = self.s.metadata.Acquisition_instrument.Detector
        spectrometer = self.s.metadata.Acquisition_instrument.Spectrometer

        assert metadata.General.date == "2023-01-20"
        assert metadata.General.time == "11:34:55"
        assert metadata.General.original_filename == "operate_mode.img"
        assert metadata.General.title == metadata.General.original_filename[:-4]

        assert metadata.Signal.quantity == "Intensity (Counts)"
        if importlib.util.find_spec("lumispy") is None:
            signal_type = ""
        else:
            signal_type = "Luminescence"

        assert metadata.Signal.signal_type == signal_type

        assert isinstance(detector.binning, tuple)
        assert len(detector.binning) == 2
        assert detector.binning[0] == 2
        assert detector.binning[1] == 2
        assert detector.detector_type == "StreakCamera"
        assert detector.model == "C5680"
        assert detector.frames == 60
        np.testing.assert_allclose(detector.integration_time, 300)
        assert detector.processing.background_correction is True
        assert detector.processing.curvature_correction is False
        assert detector.processing.defect_correction is False
        assert detector.processing.shading_correction is False
        np.testing.assert_allclose(detector.time_range, 20)
        assert detector.time_range_units == "µs"
        np.testing.assert_allclose(detector.mcp_gain, 50)
        assert detector.acquisition_mode == "analog_integration"

        np.testing.assert_allclose(spectrometer.entrance_slit_width, 10)
        assert spectrometer.model == "Andor SG"
        assert spectrometer.Grating.blazing_wavelength == 500
        assert spectrometer.Grating.groove_density == 300
        np.testing.assert_allclose(spectrometer.central_wavelength, 500)


class TestFocus:
    @classmethod
    def setup_class(cls):
        cls.s_focus = hs.load(testfile_focus_mode_path, reader="Hamamatsu")

    @classmethod
    def teardown_class(cls):
        del cls.s_focus
        gc.collect()

    def test_data_focus(self):
        expected_data_end = [[0, 0, 36], [0, 0, 0], [21, 0, 0]]
        np.testing.assert_allclose(expected_data_end, self.s_focus.isig[-3:, -3:].data)

    def test_axes_focus(self):
        axes = self.s_focus.axes_manager
        assert axes.signal_dimension == 2
        assert axes.navigation_dimension == 0
        ax0 = axes[0]
        ax1 = axes[1]
        assert ax0.name == "Wavelength"
        assert ax1.name == "Vertical CCD Position"
        assert ax1.units == "px"
        assert ax0.units == "nm"
        assert ax0.size == 672
        assert ax1.size == 512

        np.testing.assert_allclose(ax1.scale, 1)
        np.testing.assert_allclose(ax1.offset, 0)

        expected_data_start_X = [472.252, 472.33337, 472.41473, 472.4961, 472.57745]
        np.testing.assert_allclose(ax0.axis[:5], expected_data_start_X)


class TestPhotonCount:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(testfile_photon_count_path, reader="Hamamatsu")

    @classmethod
    def teardown_class(cls):
        del cls.s
        gc.collect()

    def test_data(self):
        expected_data = [0, 0, 0]
        np.testing.assert_allclose(self.s.isig[-3:, 0].data, expected_data)

    def test_metadata(self):
        metadata = self.s.metadata
        assert metadata.General.date == "2018-08-29"
        assert (
            metadata.Acquisition_instrument.Detector.acquisition_mode
            == "photon_counting"
        )
        assert "Grating" not in list(
            self.s.metadata.Acquisition_instrument.Spectrometer.as_dictionary()
        )
        assert self.s.original_metadata.file_type == "bit16"


class TestShading:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(testfile_shading_path, reader="Hamamatsu")

    @classmethod
    def teardown_class(cls):
        del cls.s
        gc.collect()

    def test_metadata(self):
        np.testing.assert_allclose(
            self.s.metadata.Acquisition_instrument.Detector.time_range, 4
        )

    def test_data(self):
        expected_data = [9385, 8354, 7658]
        np.testing.assert_allclose(self.s.isig[:3, 0].data, expected_data)
