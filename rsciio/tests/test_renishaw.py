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

import gc
import pytest
from pathlib import Path
from copy import deepcopy

import numpy as np

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")

testfile_dir = (Path(__file__).parent / "renishaw_data").resolve()

testfile_spec = (testfile_dir / "renishaw_test_spectrum.wdf").resolve()
testfile_linescan = (testfile_dir / "renishaw_test_linescan.wdf").resolve()
testfile_map = (testfile_dir / "renishaw_test_map.wdf").resolve()
testfile_zscan = (testfile_dir / "renishaw_test_zscan.wdf").resolve()
testfile_undefined = (testfile_dir / "renishaw_test_undefined.wdf").resolve()
testfile_streamline = (testfile_dir / "renishaw_test_streamline.wdf").resolve()
testfile_map2 = (testfile_dir / "renishaw_test_map2.wdf").resolve()


class TestSpec:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_spec,
            reader="Renishaw",
            use_uniform_signal_axis=True,
        )

        cls.s_non_uniform = hs.load(
            testfile_spec,
            reader="Renishaw",
            use_uniform_signal_axis=False,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        del cls.s_non_uniform
        gc.collect()

    def test_data(self):
        expected_data_start = [
            65.36617,
            67.51018,
            60.00703,
            62.75381,
            64.59786,
            61.309525,
        ]
        expected_data_end = [
            63.337173,
            64.90067,
            61.871353,
            59.75822,
            67.45442,
            68.10285,
        ]
        np.testing.assert_allclose(expected_data_start, self.s.isig[:6].data)
        np.testing.assert_allclose(expected_data_end, self.s.isig[-6:].data)
        np.testing.assert_allclose(self.s.data, self.s_non_uniform.data)
        assert len(self.s.data) == 36

    def test_axes(self):
        expected_axis = {
            "axis-0": {
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "size": 36,
            }
        }

        axes_manager = self.s.axes_manager.as_dictionary()
        axes_manager["axis-0"].pop("_type", None)
        axes_manager["axis-0"].pop("is_binned", None)
        np.testing.assert_allclose(
            axes_manager["axis-0"].pop("scale"), 0.0847, rtol=0.0003
        )
        np.testing.assert_allclose(axes_manager["axis-0"].pop("offset"), 326.01, 0.01)
        assert axes_manager == expected_axis

        expected_non_uniform_axis_values = [
            326.01633,
            326.101,
            326.18573,
            326.27042,
            326.35513,
            326.43985,
        ]
        non_uniform_axes_manager = self.s_non_uniform.axes_manager.as_dictionary()
        non_uniform_axes_manager["axis-0"].pop("_type", None)
        non_uniform_axes_manager["axis-0"].pop("is_binned", None)
        assert len(non_uniform_axes_manager["axis-0"]["axis"]) == 36
        np.testing.assert_allclose(
            non_uniform_axes_manager["axis-0"].pop("axis")[:6],
            expected_non_uniform_axis_values,
        )
        axes_manager["axis-0"].pop("size")
        assert axes_manager == non_uniform_axes_manager

    def test_original_metadata_WDF1(self):
        original_metadata = deepcopy(self.s.original_metadata.as_dictionary())
        original_metadata_non_uniform = deepcopy(
            self.s_non_uniform.original_metadata.as_dictionary()
        )
        assert np.isclose(original_metadata["WDF1_1"].pop("laser_wavenumber"), 30769.2)
        assert np.isclose(
            original_metadata_non_uniform["WDF1_1"].pop("laser_wavenumber"), 30769.2
        )
        assert original_metadata_non_uniform == original_metadata

        expected_metadata = {
            "flags": 0,
            "uuid": "3846669335-1129807005-716570026-4266516655",
            "ntracks": 0,
            "file_status_error_code": 0,
            "capacity": 1,
            "app_name": "WiRE",
            "app_version": "5-5-0-22400",
            "scan_type": "Static",
            "time_start": "2022-02-16#13:01:31",
            "time_end": "2022-02-16#13:13:31",
            "quantity_unit": "counts",
            "username": "optik",
            "title": "Single scan measurement 7",
            "points_per_spectrum": 36,
            "num_spectra": 1,
            "accumulations_per_spectrum": 4,
            "YLST_length": 1,
            "XLST_length": 36,
            "num_ORGN": 3,
            "measurement_type": "Single",
        }
        assert expected_metadata == original_metadata["WDF1_1"]

    def test_original_metadata_YLST(self):
        expected_YLST = {"name": "Spatial_Y", "units": "px", "size": 1, "data": 25.0}
        assert self.s.original_metadata.YLST_0.as_dictionary() == expected_YLST

    def test_original_metadata_WXIS(self):
        expected_WXIS = {
            "Instrument type": "InVia",
            "logicalMotorPositions": {
                "Beam Expander": 0.0,
                "CCD lensfocus": 12.1875,
                "Entrance Filter 1": 7500.0,
                "Entrance Filter 2": 7500.0,
                "Grating Motor": 11.859422492401215,
                "Holographic Notch Filter": -0.7598784194528876,
                "Laser Shutter": 1.0,
                "ND Set Flag 1": 0.0,
                "ND Set Flag 2": -0.0,
                "ND Set Flag 3": -0.0,
                "ND Set Flag 4": 0.0,
                "Pinhole": -0.0,
                "Podule AStop": -5000.0,
                "Podule FStop": -5000.0,
                "Podule Lower Selector Wheel": 3.0,
                "Podule Silicon Ref tilt": 7500.0,
                "Podule Upper Selector Wheel": 1.0,
                "Post Beam Expander Axis 1": 738.0,
                "Post Beam Expander Axis 2": 572.0,
                "Post slit lensfocus": 12.1875,
                "Pre Beam Expander Axis 1": 461.0,
                "Pre Beam Expander Axis 2": 776.0,
                "Pre slit lensfocus": 5.96875,
                "Slit Master": 1698.2421875,
                "Slit Slave": 1754.0690104166667,
                "Waveplate Motor": 0.0,
                "XYZ Stage X Motor": 25219.8,
                "XYZ Stage Y Motor": -41827.5,
                "XYZ Stage Z Motor": -19023.2,
            },
            "Microscope": {"Field of view X": 9096.0, "Field of view Y": 5761.0},
            "Motor Positions": {
                "Beam Expander": "0 Percent",
                "CCD lensfocus": "12.1875 mm",
                "Entrance Filter 1": "7500 Steps",
                "Entrance Filter 2": "7500 Steps",
                "Grating Motor": "11.8594 Degrees",
                "Holographic Notch Filter": "-0.759878 Degrees",
                "Laser Shutter": "Open",
                "ND Set Flag 1": "100",
                "ND Set Flag 2": "100",
                "ND Set Flag 3": "100",
                "ND Set Flag 4": "100",
                "Pinhole": "Out",
                "Podule AStop": "-5000 Steps",
                "Podule FStop": "-5000 Steps",
                "Podule Lower Selector Wheel": "Sample",
                "Podule Silicon Ref tilt": "7500 Steps",
                "Podule Upper Selector Wheel": "Laser",
                "Post Beam Expander Axis 1": "738 Steps",
                "Post Beam Expander Axis 2": "572 Steps",
                "Post slit lensfocus": "12.1875 mm",
                "Pre Beam Expander Axis 1": "461 Steps",
                "Pre Beam Expander Axis 2": "776 Steps",
                "Pre slit lensfocus": "5.96875 mm",
                "Slit Master": "1698.24 microns",
                "Slit Slave": "1754.07 microns",
                "Waveplate Motor": "Normal",
                "XYZ Stage X Motor": "25219.8 microns",
                "XYZ Stage Y Motor": "-41827.5 microns",
                "XYZ Stage Z Motor": "-19023.2 microns",
            },
            "ND Transmission %": "100",
            "Podule": {
                "Calibration lamp": "Off",
                "Calibration lamp intensity": 0,
                "Illumination lamp": "Off",
                "Illumination lamp intensity": 0,
                "Neon lamp": "Off",
                "Podule name": "Main podule",
                "Podule Temperature": "23°C",
                "podule type": 3,
                "Podule type": "Podule mk2",
            },
            "Serial Numbers": {},
            "Slits": {"Bias": "0µm", "Opening": "66µm", "SlitBeamCentre": "1731µm"},
            "Stage": {"X": 25219.8, "Y": -41827.5, "Z": -19023.2},
            "System Configuration": {
                "beamPath": 2,
                "detectorID": 1,
                "focusMode": 1,
                "FocusMode": "Regular",
                "Grating": "1200 l/mm (325nm PL / 532nm )",
                "key_Grating": "grating0",
                "key_Laser": "laser0",
            },
        }

        assert expected_WXIS == self.s.original_metadata.WXIS_0.as_dictionary()

    def test_original_metadata_WXCS(self):
        expected_WXCS = {
            "AreakeyDependants": {
                "Microscope imaging factor": {
                    "532 nm edge, Renishaw Centrus 2URV61": "800",
                    "Default": "800",
                    "mask": 5,
                    "Mask": "laser detector ",
                    "Use default": "true",
                },
                "Response calibration target level (raw counts)": {
                    "Default": "30000",
                    "mask": 4,
                    "Mask": "detector ",
                    "Use default": "true",
                },
                "Zero level & dark current": {
                    "mask": 4,
                    "Mask": "detector ",
                    "Renishaw Centrus 2URV61": "<object>",
                    "Use default": "false",
                },
            },
            "CCD": {
                "Areas": {
                    "Laser0, grating0, mode: Regular, path: Grating, det 1.": {
                        "Area bottom": 55,
                        "Area left": 18,
                        "Area right": 1032,
                        "Area top": 25,
                        "Centre pixel": 517,
                        "X binning": 1,
                        "Y binning": 108,
                    }
                },
                "CCD": "Renishaw Centrus 2URV61",
                "Centrus": {
                    "Channel 0": {
                        "ChannelGain": 0.83,
                        "ChannelGainName": "high",
                        "ChannelIndex": 0,
                        "ChannelName": "High sensitivity mode",
                        "ChannelSpeed": 33.0,
                        "ChannelSpeedName": "low",
                    },
                    "Channel 1": {
                        "ChannelGain": 2.4,
                        "ChannelGainName": "medium",
                        "ChannelIndex": 1,
                        "ChannelName": "Standard mode",
                        "ChannelSpeed": 33.0,
                        "ChannelSpeedName": "low",
                    },
                    "Channel 2": {
                        "ChannelGain": 9.54,
                        "ChannelGainName": "low",
                        "ChannelIndex": 2,
                        "ChannelName": "High range mode",
                        "ChannelSpeed": 33.0,
                        "ChannelSpeedName": "low",
                    },
                    "Channel 3": {
                        "ChannelGain": 0.84,
                        "ChannelGainName": "high",
                        "ChannelIndex": 3,
                        "ChannelName": "High sensitivity fast readout mode",
                        "ChannelSpeed": 200.0,
                        "ChannelSpeedName": "medium",
                    },
                    "Channel 4": {
                        "ChannelGain": 2.44,
                        "ChannelGainName": "medium",
                        "ChannelIndex": 4,
                        "ChannelName": "Standard fast readout mode",
                        "ChannelSpeed": 200.0,
                        "ChannelSpeedName": "medium",
                    },
                    "Channel 5": {
                        "ChannelGain": 9.77,
                        "ChannelGainName": "low",
                        "ChannelIndex": 5,
                        "ChannelName": "High range fast readout mode ",
                        "ChannelSpeed": 200.0,
                        "ChannelSpeedName": "medium",
                    },
                    "Channel 6": {
                        "ChannelGain": 0.69,
                        "ChannelGainName": "high",
                        "ChannelIndex": 6,
                        "ChannelName": "High sensitivity fastest readout mode",
                        "ChannelSpeed": 2000.0,
                        "ChannelSpeedName": "high",
                    },
                    "Channel 7": {
                        "ChannelGain": 2.47,
                        "ChannelGainName": "medium",
                        "ChannelIndex": 7,
                        "ChannelName": "Standard fastest readout mode",
                        "ChannelSpeed": 2000.0,
                        "ChannelSpeedName": "high",
                    },
                    "Channel 8": {
                        "ChannelGain": 6.81,
                        "ChannelGainName": "low",
                        "ChannelIndex": 8,
                        "ChannelName": "High range fastest readout mode",
                        "ChannelSpeed": 2000.0,
                        "ChannelSpeedName": "high",
                    },
                    "DisplayName": "Renishaw Centrus 2URV61",
                    "EMGainSupport": "false",
                    "Identifier": "2URV61",
                    "Type": "{D42848CA-072A-4ABA-B9CA-1D1AB50EFFA2}",
                },
                "Charge clear mode": "once per exposure",
                "Gain factor (high)": 2.5,
                "Gain factor (low)": 10.0,
                "Height (pixels)": 256,
                "Pixel height (mm)": -0.026,
                "Pixel width (mm)": -0.026,
                "Preread": 18,
                "Preread_Ext": 3,
                "Readout axis": "X",
                "Readout dirn": "positive",
                "Readout from X1": "false",
                "Shutter always open": "false",
                "Shutter present": "false",
                "Target temperature": -70,
                "Use pixel intensity": "true",
                "Width (pixels)": 1040,
            },
            "Gratings": {
                "1200 l/mm (325nm PL / 532nm )": {
                    "325nm_PL": {
                        "CalibrationType": "RM Standard",
                        "Focal Length 0": 255.052664563221,
                        "Focal Length 1": 2.82660353662461e-05,
                        "Grating Order": 1,
                        "Grating Status": "calibrated",
                        "Groove Density (lines/mm)": 1200,
                        "Include Angle (Rads)": 0.515767,
                        "Lens Distortion 0": 0.0,
                        "Lens Distortion 1": 1.0,
                        "Lens Distortion 2": 2.88731884159059e-06,
                        "Lens Distortion 3": 2.06664681416031e-08,
                        "Lens Distortion 4": -5.75777927691806e-12,
                        "Lens Distortion 5": -8.62270489788888e-16,
                        "Polynomial 0": 0.0,
                        "Polynomial 1": 1.0,
                        "Polynomial 2": 0.0,
                        "Polynomial 3": 0.0,
                        "Zero Offset (Rads)": 0.0003818160379531949,
                    },
                    "Grating key": "grating0",
                }
            },
            "Lasers": {
                "325nm_PL": {
                    "Laser key": "laser0",
                    "Laser path": 2,
                    "Wavenumber": 30769.2,
                }
            },
            "Motor Calibration": {
                "Beam Expander": {
                    "boardType": "Serial",
                    "motor Address": 60,
                    "Reference Position (steps)": -1250,
                    "Steps Per Unit": 10.0,
                },
                "CCD lensfocus": {
                    "boardType": "Serial",
                    "motor Address": 84,
                    "Reference Position (steps)": -64000,
                    "Steps Per Unit": -3200.0,
                },
                "Entrance Filter 1": {
                    "boardType": "Serial",
                    "motor Address": 51,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 1.0,
                },
                "Entrance Filter 2": {
                    "boardType": "Serial",
                    "motor Address": 52,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 1.0,
                },
                "Grating Motor": {
                    "boardType": "Serial",
                    "motor Address": 61,
                    "Reference Position (steps)": -4091971,
                    "Steps Per Unit": 1316.0,
                },
                "Holographic Notch Filter": {
                    "boardType": "Serial",
                    "motor Address": 62,
                    "Reference Position (steps)": -4000000,
                    "Steps Per Unit": 1316.0,
                },
                "Laser Shutter": {
                    "boardType": "Serial",
                    "motor Address": 7,
                    "Reference Position (steps)": -4475,
                    "Steps Per Unit": -2544.0,
                },
                "ND Set Flag 1": {
                    "boardType": "Serial",
                    "motor Address": 31,
                    "Reference Position (steps)": -5000,
                    "Steps Per Unit": 3891.0,
                },
                "ND Set Flag 2": {
                    "boardType": "Serial",
                    "motor Address": 32,
                    "Reference Position (steps)": -10000,
                    "Steps Per Unit": -3954.0,
                },
                "ND Set Flag 3": {
                    "boardType": "Serial",
                    "motor Address": 33,
                    "Reference Position (steps)": -10000,
                    "Steps Per Unit": -4182.0,
                },
                "ND Set Flag 4": {
                    "boardType": "Serial",
                    "motor Address": 34,
                    "Reference Position (steps)": -5000,
                    "Steps Per Unit": 3979.0,
                },
                "Pinhole": {
                    "boardType": "Serial",
                    "motor Address": 27,
                    "Reference Position (steps)": -7650,
                    "Steps Per Unit": -2700.0,
                },
                "Podule AStop": {
                    "boardType": "Serial",
                    "motor Address": 37,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": -1.0,
                },
                "Podule FStop": {
                    "boardType": "Serial",
                    "motor Address": 39,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": -1.0,
                },
                "Podule Lower Selector Wheel": {
                    "boardType": "Serial",
                    "motor Address": 44,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 1.0,
                },
                "Podule Silicon Ref tilt": {
                    "boardType": "Serial",
                    "motor Address": 35,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 1.0,
                },
                "Podule Upper Selector Wheel": {
                    "boardType": "Serial",
                    "motor Address": 43,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 1.0,
                },
                "Post Beam Expander Axis 1": {
                    "boardType": "Serial",
                    "motor Address": 19,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 1.0,
                },
                "Post Beam Expander Axis 2": {
                    "boardType": "Serial",
                    "motor Address": 20,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 1.0,
                },
                "Post slit lensfocus": {
                    "boardType": "Serial",
                    "motor Address": 83,
                    "Reference Position (steps)": -64000,
                    "Steps Per Unit": -3200.0,
                },
                "Pre Beam Expander Axis 1": {
                    "boardType": "Serial",
                    "motor Address": 21,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 1.0,
                },
                "Pre Beam Expander Axis 2": {
                    "boardType": "Serial",
                    "motor Address": 22,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 1.0,
                },
                "Pre slit lensfocus": {
                    "boardType": "Serial",
                    "motor Address": 82,
                    "Reference Position (steps)": -64000,
                    "Steps Per Unit": -3200.0,
                },
                "Slit Master": {
                    "boardType": "Serial",
                    "motor Address": 2,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 6.144,
                },
                "Slit Slave": {
                    "boardType": "Serial",
                    "motor Address": 3,
                    "Reference Position (steps)": 0,
                    "Steps Per Unit": 6.144,
                },
                "Waveplate Motor": {
                    "boardType": "Serial",
                    "motor Address": 50,
                    "Reference Position (steps)": -6880,
                    "Steps Per Unit": 932.0,
                },
                "XYZ Stage X Motor": {
                    "boardType": "Serial",
                    "motor Address": 12,
                    "Reference Position (steps)": -3867487,
                    "Steps Per Unit": 10.0,
                },
                "XYZ Stage Y Motor": {
                    "boardType": "Serial",
                    "motor Address": 13,
                    "Reference Position (steps)": -3376226,
                    "Steps Per Unit": -10.0,
                },
                "XYZ Stage Z Motor": {
                    "boardType": "Serial",
                    "motor Address": 14,
                    "Reference Position (steps)": -3802311,
                    "Steps Per Unit": -10.0,
                },
            },
            "Slits": {
                "325nm_PL": {
                    "Regular beam centre": "1730.7µm",
                    "Regular opening": "65.0µm",
                }
            },
        }

        assert expected_WXCS == self.s.original_metadata.WXCS_0.as_dictionary()

    def test_original_metadata_WXDM(self):
        expected_WXDM = {"ExportToSPC": 1, "ExportToTXT": 1}
        assert expected_WXDM == self.s.original_metadata.WXDM_0.as_dictionary()

    def test_original_metadata_WXDA(self):
        expected_WXDA = {
            "ScanDuration(ms)": 720530,
            "CCD Status 1": 1,
            "CCD Status 2": 1,
            "lastScanCompletionStatus": "Incomplete",
            "AutoExportErrorCode": 0,
            "AutoExportTxtError": "No Error",
            "AutoExportSpcError": "No Error",
            "Use CV mode": 0,
            "CentrusMapIndex": 1,
            "abortState": 0,
            "CCD firmware version": "4.99",
            "CCD not emulating": 1,
            "CCD serial number": "2URV61",
            "CCD SYSID": 17051001,
            "cycling": 0,
            "InterlockChainAtEnd": 221,
            "InterlockOKEnd": 0,
            "InterlockOKStart": 1,
            "InterlockPeripheralsAtEnd": 0,
            "InterlockRelaysEnd": -1,
            "InterlockRelaysStart": 3,
            "LUT_Auto": -1,
            "minimizeLaserExposure": 0,
            "PlugIns": {},
            "SaturationProtectionUsed": 0,
            "SequenceDurationMS": 2098,
            "UnsaturatedExposureTime": 0,
            "UnsaturatedNDPercent": 0,
            "useExternalSignalMapping": 0,
            "usePerformanceQualification": 0,
            "ErrorSource": "No error",
            "ErrorDescription": "No error",
            "ErrorLine": 0,
            "MultiMeasurementAbort": 0,
            "ScanRangeOverridden": 0,
        }
        assert expected_WXDA == self.s.original_metadata.WXDA_0.as_dictionary()

    def test_original_metadata_ZLDC(self):
        expected_ZLDC = {
            "Type": "ZeroLevelAndDarkCurrent",
            "ProcessingOperation": -1,
            "ReadOnly": -1,
        }
        assert expected_ZLDC == self.s.original_metadata.ZLDC_0.as_dictionary()

    def test_original_metadata_WARP(self):
        expected_WARP0 = {"ProcessingOperation": 1, "SpectrumIndex": -1}
        expected_WARP1 = {"ProcessingOperation": 1, "SpectrumIndex": -1}

        assert expected_WARP0 == self.s.original_metadata.WARP_0.as_dictionary()
        assert expected_WARP1 == self.s.original_metadata.WARP_1.as_dictionary()

    def test_original_metadata_TEXT(self):
        expected_TEXT = "A single scan measurement generated by the WiRE spectral acquisition wizard."
        assert expected_TEXT == self.s.original_metadata.TEXT_0

    def test_original_metadata_ORGN(self):
        expected_ORGN = {
            "Time": {
                "units": "ns",
                "annotation": "Time",
                "data": np.array([0], dtype=np.int64),
            },
            "BitFlags": {
                "units": "",
                "annotation": "Flags",
                "data": np.array([0.0]),
            },
            "SpectrumDataChecksum": {
                "units": "",
                "annotation": "Checksum",
                "data": np.array([0.0]),
            },
        }
        assert expected_ORGN == self.s.original_metadata.ORGN_0.as_dictionary()

    def test_metadata(self):
        metadata = deepcopy(self.s.metadata.as_dictionary())
        metadata_non_uniform = deepcopy(self.s_non_uniform.metadata.as_dictionary())

        assert metadata["General"]["date"] == "2022-02-16"
        assert metadata["General"]["time"] == "13:01:31"
        assert metadata["General"]["original_filename"] == "renishaw_test_spectrum.wdf"
        assert metadata["General"]["title"] == "Single scan measurement 7"

        assert metadata["Signal"]["quantity"] == "Intensity (Counts)"
        assert metadata["Signal"]["signal_type"] == ""

        assert metadata["Acquisition_instrument"]["Detector"]["detector_type"] == "CCD"
        assert metadata["Acquisition_instrument"]["Detector"]["frames"] == 4
        assert (
            metadata["Acquisition_instrument"]["Detector"]["model"]
            == "Renishaw Centrus 2URV61"
        )
        assert np.isclose(
            metadata["Acquisition_instrument"]["Detector"]["temperature"], -70
        )
        assert np.isclose(
            metadata["Acquisition_instrument"]["Laser"]["wavelength"], 325, atol=1e-3
        )
        assert np.isclose(
            metadata["Acquisition_instrument"]["Spectrometer"]["Grating"][
                "groove_density"
            ],
            1200,
        )

        del metadata["General"]["FileIO"]
        del metadata_non_uniform["General"]["FileIO"]

        assert metadata_non_uniform == metadata


class TestLinescan:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_linescan,
            reader="Renishaw",
            use_uniform_signal_axis=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        gc.collect()

    def test_data(self):
        expected_column0_end = [
            1.4914633,
            1.8651184,
            0.74636304,
            1.120018,
            -0.7469943,
            0.0,
        ]
        expected_column0_start = [0.0, 1.1029433, 1.1034185, 0.0]
        expected_column4_end = [
            0.74573165,
            1.4920948,
            0.37318152,
            1.4933574,
            0.7469943,
            1.1209648,
        ]
        expected_column4_start = [
            0.7349788,
            1.4705911,
            0.36780614,
            -0.73592895,
        ]

        np.testing.assert_allclose(expected_column0_start, self.s.inav[0].isig[:4])
        np.testing.assert_allclose(expected_column0_end, self.s.inav[0].isig[-6:])
        np.testing.assert_allclose(expected_column4_start, self.s.inav[-1].isig[:4])
        np.testing.assert_allclose(expected_column4_end, self.s.inav[-1].isig[-6:])

        assert self.s.data.shape == (5, 40)

    def test_axes(self):
        expected_axis = {
            "axis-0": {
                "name": "Abs. Distance",
                "units": "µm",
                "navigate": True,
                "size": 5,
            },
            "axis-1": {
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "size": 40,
            },
        }

        axes_manager = self.s.axes_manager.as_dictionary()

        assert np.isclose(axes_manager["axis-0"].pop("offset"), 0)
        assert np.isclose(axes_manager["axis-0"].pop("scale"), 30)

        assert np.isclose(axes_manager["axis-1"].pop("offset"), 361.3127556405415)
        assert np.isclose(axes_manager["axis-1"].pop("scale"), 0.08534686462516697)

        for key in axes_manager.keys():
            axes_manager[key].pop("_type", None)
            axes_manager[key].pop("is_binned", None)

        assert axes_manager == expected_axis


class TestMap:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_map,
            reader="Renishaw",
            use_uniform_signal_axis=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        gc.collect()

    def test_data(self):
        expected_data_00_end = [
            1.3878582,
            1.3884968,
            2.0837028,
            1.3897737,
            2.0856183,
        ]
        expected_data_00_start = [
            2.722048,
            2.3829105,
        ]

        np.testing.assert_allclose(
            expected_data_00_start, self.s.inav[0, 0].isig[:2].data
        )
        np.testing.assert_allclose(
            expected_data_00_end, self.s.inav[0, 0].isig[-5:].data
        )

        expected_data_10_end = [
            0.34696454,
            2.082745,
            1.3891352,
            2.7795475,
            0.6952061,
        ]
        expected_data_10_start = [
            1.361024,
            2.0424948,
        ]

        np.testing.assert_allclose(
            expected_data_10_start, self.s.inav[1, 0].isig[:2].data
        )
        np.testing.assert_allclose(
            expected_data_10_end, self.s.inav[1, 0].isig[-5:].data
        )

        expected_data_01_end = [
            1.3878582,
            2.7769935,
            1.3891352,
            2.0846605,
            2.0856183,
        ]
        expected_data_01_start = [
            0.680512,
            1.702079,
        ]

        np.testing.assert_allclose(
            expected_data_01_start, self.s.inav[0, 1].isig[:2].data
        )
        np.testing.assert_allclose(
            expected_data_01_end, self.s.inav[0, 1].isig[-5:].data
        )

        expected_data_22_end = [
            1.3878582,
            0.6942484,
            0.6945676,
            0.0,
            1.0428091,
        ]
        expected_data_22_start = [
            0.680512,
            1.3616632,
        ]

        np.testing.assert_allclose(
            expected_data_22_start, self.s.inav[2, 2].isig[:2].data
        )
        np.testing.assert_allclose(
            expected_data_22_end, self.s.inav[2, 2].isig[-5:].data
        )

        assert self.s.data.shape == (3, 3, 47)

    def test_axes(self):
        expected_axis = {
            "axis-0": {
                "name": "Y",
                "units": "µm",
                "navigate": True,
                "size": 3,
            },
            "axis-1": {
                "name": "X",
                "units": "µm",
                "navigate": True,
                "size": 3,
            },
            "axis-2": {
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "size": 47,
            },
        }

        axes_manager = self.s.axes_manager.as_dictionary()

        assert np.isclose(axes_manager["axis-0"].pop("offset"), -100)
        assert np.isclose(axes_manager["axis-0"].pop("scale"), 100)
        assert np.isclose(axes_manager["axis-1"].pop("offset"), -100)
        assert np.isclose(axes_manager["axis-1"].pop("scale"), 100)

        assert np.isclose(axes_manager["axis-2"].pop("scale"), 0.0848807)
        assert np.isclose(axes_manager["axis-2"].pop("offset"), 346.7724758716342)

        for key in axes_manager.keys():
            axes_manager[key].pop("_type", None)
            axes_manager[key].pop("is_binned", None)

        assert expected_axis == axes_manager


class TestZscan:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_zscan,
            reader="Renishaw",
            use_uniform_signal_axis=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        gc.collect()

    def test_axes(self):
        axes_manager = self.s.axes_manager.as_dictionary()
        assert len(axes_manager) == 2
        z_axis = axes_manager.pop("axis-0")
        scale = z_axis.pop("scale")
        offset = z_axis.pop("offset")

        expected_axis = {
            "_type": "UniformDataAxis",
            "name": "Z",
            "units": "µm",
            "navigate": True,
            "is_binned": False,
            "size": 40,
        }

        assert np.isclose(scale, 0.5)
        assert np.isclose(offset, -10)
        assert z_axis == expected_axis
        assert axes_manager["axis-1"]["units"] == "1/cm"
        assert axes_manager["axis-1"]["name"] == "Wavenumber"

    def test_data(self):
        np.testing.assert_allclose(self.s.inav[0].isig[:3].data, [0, 0, 0])
        np.testing.assert_allclose(self.s.inav[0].isig[-3:].data, [0, 0, 0])
        np.testing.assert_allclose(
            self.s.inav[-1].isig[-3:].data, [1.8109605, -1.8113279, 9.05664]
        )
        np.testing.assert_allclose(
            self.s.inav[-1].isig[:3].data, [1.454424, -1.4547517, 4.365239]
        )

    def test_measurement_type(self):
        assert self.s.original_metadata.WDF1_1.measurement_type == "Series"


class TestUndefined:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_undefined,
            reader="Renishaw",
            use_uniform_signal_axis=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        gc.collect()

    def test_data(self):
        np.testing.assert_allclose(
            self.s.isig[-3:].data, [262.90552, 262.82877, 262.752]
        )
        np.testing.assert_allclose(
            self.s.isig[:3].data, [349.54773, 349.45215, 349.3566]
        )

    def test_measurement_type(self):
        assert self.s.original_metadata.WDF1_1.measurement_type == "Unspecified"


class TestStreamline:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_streamline,
            reader="Renishaw",
            use_uniform_signal_axis=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        gc.collect()

    def test_data(self):
        np.testing.assert_allclose(
            self.s.inav[0, 0].isig[:3].data, [418.35907, 424.54782, 409.82785]
        )

        np.testing.assert_allclose(
            self.s.inav[44, 48].isig[-3:].data, [587.48083, 570.73505, 583.5814]
        )

    def test_original_metadata_WHTL(self):
        expected_WTHL = {
            "FocalPlaneResolutionUnit": "µm",
            "FocalPlaneXResolution": 445.75,
            "FocalPlaneYResolution": 270.85,
            "FocalPlaneXOrigin": -8325.176,
            "FocalPlaneYOrigin": -1334.639,
            "ImageDescription": "white-light image",
            "Make": "Renishaw",
            "Unknown": 20.0,
            "FieldOfViewXY": (8915.0, 5417.0),
        }

        metadata_WHTL = deepcopy(self.s.original_metadata.WHTL_0.as_dictionary())
        img = metadata_WHTL.pop("image")
        assert metadata_WHTL == expected_WTHL

    def test_original_metadata_WMAP(self):
        expected_WMAP = {
            "linefocus_size": 0,
            "flag": 2,
        }
        metadata_WMAP = deepcopy(self.s.original_metadata.WMAP_0.as_dictionary())
        np.testing.assert_allclose(
            metadata_WMAP.pop("offset_xyz"), [-8100.3, -1201.3, 0.0]
        )
        np.testing.assert_allclose(metadata_WMAP.pop("scale_xyz"), [2.6, 2.6, 1.0])
        np.testing.assert_allclose(metadata_WMAP.pop("size_xyz"), [45, 49, 1])
        assert expected_WMAP == metadata_WMAP

    def test_original_metadata_WDF1(self):
        expected_WDF1 = {
            "flags": 0,
            "uuid": "1899852918-1261200088-3066841267-2010995264",
            "ntracks": 0,
            "file_status_error_code": 0,
            "capacity": 2205,
            "accumulations_per_spectrum": 1,
            "XLST_length": 394,
            "app_name": "WiRE",
            "app_version": "4-2-0-5037",
            "scan_type": "StreamLine",
            "time_start": "2017-10-05#11:17:33",
            "time_end": "2017-10-05#11:21:24",
            "quantity_unit": "counts",
            "username": "Raman",
            "title": "StreamLine image acquisition 5",
            "points_per_spectrum": 394,
            "num_spectra": 2205,
            "YLST_length": 1,
            "num_ORGN": 5,
            "measurement_type": "Mapping",
        }

        metadata_WDF1 = deepcopy(self.s.original_metadata.WDF1_1.as_dictionary())

        assert np.isclose(metadata_WDF1.pop("laser_wavenumber"), 12738.9)
        assert metadata_WDF1 == expected_WDF1


class TestMap2:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_map2,
            reader="Renishaw",
            use_uniform_signal_axis=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        gc.collect()

    def test_original_metadata_MAP(self):
        metadata_MAP0 = self.s.original_metadata.MAP_0
        assert metadata_MAP0.npoints == 400
        assert metadata_MAP0.guid == "{57AAA72E-55A7-4F79-9FE3-EB4603813AB9}"
        np.testing.assert_allclose(metadata_MAP0.dataRange, [45.172123, 719.9313])
        np.testing.assert_allclose(
            metadata_MAP0.data[:3], [66.674965, 62.949856, 107.18564]
        )
        np.testing.assert_allclose(
            metadata_MAP0.data[-3:], [381.15073, 593.5397, 431.03043]
        )
        assert len(metadata_MAP0.data) == 400

        metadata_MAP1 = self.s.original_metadata.MAP_1
        assert metadata_MAP1.npoints == 400
        assert metadata_MAP1.guid == "{6cf9115b-c452-404f-b0f2-e654d9a05aea}"
        np.testing.assert_allclose(metadata_MAP1.dataRange, [42.882656, 518.5028])
        np.testing.assert_allclose(
            metadata_MAP1.data[:3], [53.033577, 54.36599, 81.270424]
        )
        np.testing.assert_allclose(
            metadata_MAP1.data[-3:], [224.7265, 380.83423, 261.48608]
        )
        assert len(metadata_MAP1.data) == 400
