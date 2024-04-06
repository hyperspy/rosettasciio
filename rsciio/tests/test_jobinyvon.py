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

import gc
import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")

testfile_dir = Path(__file__).parent / "data" / "jobinyvon"

testfile_spec_wavelength_path = testfile_dir / "jobinyvon_test_spec.xml"
testfile_spec_wavenumber_path = testfile_dir / "jobinyvon_test_spec_3s_cm-1.xml"
testfile_spec_abs_wavenumber_path = testfile_dir / "jobinyvon_test_spec_3s_abs-cm-1.xml"
testfile_spec_energy_path = testfile_dir / "jobinyvon_test_spec_3s_eV.xml"
testfile_linescan_path = testfile_dir / "jobinyvon_test_linescan.xml"
testfile_map_path = testfile_dir / "jobinyvon_test_map_x3-y2.xml"
testfile_map_rotated_path = testfile_dir / "jobinyvon_test_map_x2-y2_rotated.xml"
testfile_glue_path = testfile_dir / "jobinyvon_test_spec_range.xml"
testfile_spec_count_path = testfile_dir / "jobinyvon_test_spec_3s_counts.xml"


if importlib.util.find_spec("lumispy") is None:
    lumispy_installed = False
else:
    lumispy_installed = True


class TestSpec:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_spec_wavelength_path,
            reader="JobinYvon",
            use_uniform_signal_axis=True,
        )
        cls.s_non_uniform = hs.load(
            testfile_spec_wavelength_path,
            reader="JobinYvon",
            use_uniform_signal_axis=False,
        )
        cls.s_wn = hs.load(
            testfile_spec_wavenumber_path,
            reader="JobinYvon",
            use_uniform_signal_axis=True,
        )
        cls.s_abs_wn = hs.load(
            testfile_spec_abs_wavenumber_path,
            reader="JobinYvon",
            use_uniform_signal_axis=True,
        )
        cls.s_ev = hs.load(
            testfile_spec_energy_path,
            reader="JobinYvon",
            use_uniform_signal_axis=True,
        )
        cls.s_count = hs.load(
            testfile_spec_count_path,
            reader="JobinYvon",
            use_uniform_signal_axis=True,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        del cls.s_non_uniform
        del cls.s_wn
        del cls.s_abs_wn
        del cls.s_ev
        gc.collect()

    @pytest.mark.skipif(lumispy_installed, reason="lumispy is installed")
    def test_signal1D(self):
        hyperspy = pytest.importorskip("hyperspy", reason="hyperspy not installed")
        assert isinstance(self.s, hyperspy._signals.signal1d.Signal1D)

    def test_lumispectrum(self):
        lum = pytest.importorskip("lumispy", reason="lumispy not installed")
        s_lum = hs.load(
            testfile_spec_wavelength_path,
            reader="JobinYvon",
            use_uniform_signal_axis=True,
        )
        assert isinstance(s_lum, lum.signals.luminescence_spectrum.LumiSpectrum)

    def test_intensity_count_unit(self):
        assert self.s_count.metadata.Signal.quantity == "Intensity (Counts)"

    def test_signal_units(self):
        assert self.s_wn.axes_manager.as_dictionary()["axis-0"]["units"] == "1/cm"
        assert self.s_abs_wn.axes_manager.as_dictionary()["axis-0"]["units"] == "1/cm"
        assert self.s_ev.axes_manager.as_dictionary()["axis-0"]["units"] == "eV"

    def test_signal_names(self):
        assert self.s_wn.axes_manager.as_dictionary()["axis-0"]["name"] == "Raman Shift"
        assert (
            self.s_abs_wn.axes_manager.as_dictionary()["axis-0"]["name"] == "Wavenumber"
        )
        assert self.s_ev.axes_manager.as_dictionary()["axis-0"]["name"] == "Energy"

    def test_integration_time(self):
        np.testing.assert_allclose(
            self.s_wn.metadata["Acquisition_instrument"]["Detector"][
                "integration_time"
            ],
            3,
        )
        np.testing.assert_allclose(
            self.s_wn.metadata["Acquisition_instrument"]["Detector"][
                "exposure_per_frame"
            ],
            3,
        )
        np.testing.assert_allclose(
            self.s_wn.metadata["Acquisition_instrument"]["Detector"]["frames"], 1
        )

    def test_data(self):
        spec_data = [
            1496,
            1242,
            1094,
            986,
            948,
            900,
            858,
            855,
            840,
            822,
            824,
            820,
            810,
            809,
            791,
            781,
            771,
            782,
            795,
            790,
            777,
            771,
            769,
            767,
            756,
            755,
            759,
            740,
            743,
            763,
            759,
            727,
            764,
            760,
        ]
        assert spec_data[::-1] == self.s.data.tolist()
        np.testing.assert_allclose(self.s.data, self.s_non_uniform.data)

    def test_axes(self):
        spec_axes = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
                "size": 34,
            }
        }

        spec_axes_non_uniform = {
            "axis-0": {
                "_type": "DataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
            }
        }

        non_uniform_axis_values = np.array(
            [
                537.361,
                536.918,
                536.474,
                536.031,
                535.586,
                535.142,
                534.697,
                534.252,
                533.807,
                533.361,
                532.915,
                532.468,
                532.022,
                531.575,
                531.128,
                530.68,
                530.232,
                529.784,
                529.336,
                528.887,
                528.438,
                527.988,
                527.539,
                527.089,
                526.639,
                526.188,
                525.737,
                525.286,
                524.835,
                524.383,
                523.931,
                523.479,
                523.027,
                522.574,
            ]
        )

        uniform_axis_manager = deepcopy(self.s.axes_manager.as_dictionary())
        non_uniform_axis_manager = deepcopy(
            self.s_non_uniform.axes_manager.as_dictionary()
        )
        np.testing.assert_allclose(
            uniform_axis_manager["axis-0"].pop("scale"), 0.4481, atol=0.0001
        )
        np.testing.assert_allclose(
            uniform_axis_manager["axis-0"].pop("offset"), 522.6, atol=0.05
        )

        np.testing.assert_allclose(
            non_uniform_axis_values[::-1],
            non_uniform_axis_manager["axis-0"].pop("axis"),
        )
        assert spec_axes == uniform_axis_manager
        assert spec_axes_non_uniform == non_uniform_axis_manager

    def test_original_metadata(self):
        spec_original_metadata = {
            "date": {"Acquired": "27.06.2022 16:26:24"},
            "experimental_setup": {
                "Acq. time (s)": 1.0,
                "Accumulations": 2.0,
                "Range": "Visible",
                "Autofocus": "Off",
                "AutoExposure": "Off",
                "Spike filter": "Multiple accum.",
                "Delay time (s)": 0.0,
                "Binning": 30.0,
                "Readout mode": "Signal",
                "DeNoise": "Off",
                "ICS correction": "Off",
                "Dark correction": "Off",
                "Inst. Process": "Off",
                "Detector temperature (°C)": -118.94,
                "Instrument": "LabRAM HR Evol",
                "Detector": "Symphony VIS",
                "Objective": 100.0,
                "Grating (gr/mm)": 1800.0,
                "ND Filter (%)": 10.0,
                "Laser (nm)": 632.817,
                "Spectro (nm)": 530.0006245,
                "Hole": 100.02125,
                "Laser Pol. (°)": 0.0,
                "Raman Pol. (°)": 0.0,
                "StageXY": "Marzhauser",
                "StageZ": "Marzhauser",
                "X (µm)": 0.0,
                "Y (µm)": 0.0,
                "Z (µm)": 0.0,
                "Full time(s)": 3.0,
                "measurement_type": "Spectrum",
                "title": "jobinyvon_test_spec",
                "signal type": "Intens",
                "signal units": "Cnt/sec",
            },
            "file_information": {
                "Project": "A",
                "Sample": "test",
                "Site": "C",
                "Title": "ev21738_1",
                "Remark": "PL",
                "Date": "27.06.2022 16:26",
            },
        }
        assert spec_original_metadata == self.s.original_metadata.as_dictionary()
        assert (
            spec_original_metadata
            == self.s_non_uniform.original_metadata.as_dictionary()
        )

    def test_metadata(self):
        metadata = deepcopy(self.s.metadata.as_dictionary())
        metadata_non_uniform = deepcopy(self.s_non_uniform.metadata.as_dictionary())
        assert (
            metadata_non_uniform["General"]["FileIO"]["0"]["io_plugin"]
            == "rsciio.jobinyvon"
        )
        assert metadata["General"]["FileIO"]["0"]["io_plugin"] == "rsciio.jobinyvon"
        assert metadata["General"]["date"] == "27.06.2022"
        assert metadata["General"]["original_filename"] == str(
            testfile_spec_wavelength_path.name
        )
        assert metadata["General"]["time"] == "16:26:24"
        assert metadata["Signal"]["quantity"] == "Intensity (Counts/s)"
        assert metadata["Signal"]["signal_dimension"] == 1
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["binning"], 30
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["exposure_per_frame"], 1
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["frames"], 2
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["integration_time"], 2
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["temperature"], -118.94
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["delay_time"], 0
        )
        assert metadata["Acquisition_instrument"]["Detector"]["model"] == "Symphony VIS"
        assert (
            metadata["Acquisition_instrument"]["Detector"]["processing"][
                "auto_exposure"
            ]
            == "Off"
        )
        assert (
            metadata["Acquisition_instrument"]["Detector"]["processing"]["autofocus"]
            == "Off"
        )
        assert (
            metadata["Acquisition_instrument"]["Detector"]["processing"][
                "dark_correction"
            ]
            == "Off"
        )
        assert (
            metadata["Acquisition_instrument"]["Detector"]["processing"]["de_noise"]
            == "Off"
        )
        assert (
            metadata["Acquisition_instrument"]["Detector"]["processing"][
                "ics_correction"
            ]
            == "Off"
        )
        assert (
            metadata["Acquisition_instrument"]["Detector"]["processing"]["inst_process"]
            == "Off"
        )
        assert (
            metadata["Acquisition_instrument"]["Detector"]["processing"]["spike_filter"]
            == "Multiple accum."
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Laser"]["objective_magnification"], 100
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Laser"]["wavelength"], 632.817
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Laser"]["Filter"]["optical_density"],
            0.1,
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Spectrometer"]["Grating"][
                "groove_density"
            ],
            1800,
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Spectrometer"]["central_wavelength"],
            530.0006245,
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Spectrometer"]["entrance_slit_width"],
            1.0002125,
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Spectrometer"]["pinhole"],
            1.0002125,
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Spectrometer"]["Polarizer"]["angle"],
            0,
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Laser"]["Polarizer"]["angle"],
            0,
        )
        assert (
            metadata["Acquisition_instrument"]["Spectrometer"]["model"]
            == "LabRAM HR Evol"
        )
        assert (
            metadata["Acquisition_instrument"]["Spectrometer"]["spectral_range"]
            == "Visible"
        )
        assert (
            metadata["Acquisition_instrument"]["Spectrometer"]["Polarizer"][
                "polarizer_type"
            ]
            == "No"
        )
        assert (
            metadata["Acquisition_instrument"]["Laser"]["Polarizer"]["polarizer_type"]
            == "L/2 vis L"
        )

        ## remove FileIO for comparison (timestamp varies)
        ## plugin is already tested for both (above)
        del metadata["General"]["FileIO"]
        del metadata_non_uniform["General"]["FileIO"]

        assert metadata == metadata_non_uniform


class TestLinescan:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_linescan_path,
            reader="JobinYvon",
            use_uniform_signal_axis=True,
        )
        cls.s_non_uniform = hs.load(
            testfile_linescan_path,
            reader="JobinYvon",
            use_uniform_signal_axis=False,
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        del cls.s_non_uniform
        gc.collect()

    def test_data(self):
        linescan_row0 = [
            1614,
            1317,
            1140,
            1035,
            970,
            931,
            901,
            868,
            864,
            845,
            843,
            847,
            831,
            834,
            813,
            810,
            807,
            817,
            807,
            798,
            800,
            797,
            788,
            804,
            767,
            778,
            778,
            787,
            775,
            790,
            769,
            778,
            780,
            783,
        ]
        linescan_row1 = [
            1509,
            1251,
            1087,
            1002,
            934,
            896,
            866,
            830,
            837,
            831,
            815,
            792,
            784,
            811,
            796,
            799,
            794,
            784,
            788,
            773,
            780,
            787,
            797,
            779,
            780,
            765,
            770,
            757,
            758,
            758,
            741,
            769,
            765,
            775,
        ]
        linescan_row2 = [
            1546,
            1292,
            1124,
            1020,
            950,
            901,
            890,
            865,
            865,
            847,
            837,
            824,
            827,
            808,
            818,
            809,
            810,
            814,
            798,
            784,
            790,
            785,
            771,
            790,
            786,
            786,
            773,
            771,
            772,
            768,
            782,
            762,
            757,
            781,
        ]
        assert linescan_row0[::-1] == self.s.data.tolist()[0]
        assert linescan_row1[::-1] == self.s.data.tolist()[1]
        assert linescan_row2[::-1] == self.s.data.tolist()[2]
        np.testing.assert_allclose(self.s.data, self.s_non_uniform.data)

    def test_axes(self):
        linescan_axes = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "Y",
                "units": "µm",
                "navigate": True,
                "is_binned": False,
                "size": 3,
                "scale": 0.5,
                "offset": 0.0,
            },
            "axis-1": {
                "_type": "UniformDataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
                "size": 34,
            },
        }

        linescan_axes_non_uniform = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "Y",
                "units": "µm",
                "navigate": True,
                "is_binned": False,
                "size": 3,
                "scale": 0.5,
                "offset": 0.0,
            },
            "axis-1": {
                "_type": "DataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
            },
        }

        non_uniform_axis_values = np.array(
            [
                537.361,
                536.918,
                536.474,
                536.031,
                535.586,
                535.142,
                534.697,
                534.252,
                533.807,
                533.361,
                532.915,
                532.468,
                532.022,
                531.575,
                531.128,
                530.68,
                530.232,
                529.784,
                529.336,
                528.887,
                528.438,
                527.988,
                527.539,
                527.089,
                526.639,
                526.188,
                525.737,
                525.286,
                524.835,
                524.383,
                523.931,
                523.479,
                523.027,
                522.574,
            ]
        )

        uniform_axis_manager = deepcopy(self.s.axes_manager.as_dictionary())
        non_uniform_axis_manager = deepcopy(
            self.s_non_uniform.axes_manager.as_dictionary()
        )

        np.testing.assert_allclose(
            uniform_axis_manager["axis-1"].pop("scale"), 0.4481, atol=0.0001
        )
        np.testing.assert_allclose(
            uniform_axis_manager["axis-1"].pop("offset"), 522.6, atol=0.05
        )
        np.testing.assert_allclose(
            non_uniform_axis_values[::-1],
            non_uniform_axis_manager["axis-1"].pop("axis"),
        )
        assert linescan_axes_non_uniform == non_uniform_axis_manager
        assert linescan_axes == uniform_axis_manager


class TestMap:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_map_path, reader="JobinYvon", use_uniform_signal_axis=True
        )
        cls.s_non_uniform = hs.load(
            testfile_map_path, reader="JobinYvon", use_uniform_signal_axis=False
        )
        cls.s_rotated = hs.load(testfile_map_rotated_path, reader="JobinYvon")

    @classmethod
    def teardown_class(cls):
        del cls.s
        del cls.s_non_uniform
        del cls.s_rotated
        gc.collect()

    def test_rotation_angle(self):
        original_metadata = self.s_rotated.original_metadata.as_dictionary()
        np.testing.assert_allclose(
            original_metadata["experimental_setup"]["rotation angle (rad)"],
            -0.532322511232846,
        )

        metadata = self.s_rotated.metadata.as_dictionary()
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Spectral_image"]["rotation_angle"],
            -30.499833233447433,
        )

        assert (
            metadata["Acquisition_instrument"]["Spectral_image"]["rotation_angle_units"]
            == "°"
        )

    def test_data(self):
        map_row0 = [
            275.5,
            214.5,
            206.5,
            184,
            168,
            171.5,
            158,
            160,
            169.5,
            154.5,
            156.5,
            147,
            152.5,
            159,
            144,
            147.5,
            142,
            155,
            145,
            143,
            149,
            147,
            147,
            152,
            151.5,
            142.5,
            141,
            139.5,
            143,
            143,
            139.5,
            133.5,
            141.5,
            144,
        ]

        map_row1 = [
            295.5,
            234,
            206.5,
            183,
            165,
            164.5,
            145.5,
            143,
            163.5,
            152,
            142,
            146.5,
            146.5,
            142.5,
            145.5,
            138.5,
            137.5,
            132.5,
            147.5,
            138.5,
            140,
            143,
            141,
            131.5,
            142.5,
            132,
            133,
            145.5,
            141,
            132,
            123.5,
            136.5,
            128.5,
            139.5,
        ]
        map_row2 = [
            257.5,
            210,
            198,
            171,
            168,
            168,
            147.5,
            161.5,
            174.5,
            161,
            157.5,
            152.5,
            157,
            144,
            151.5,
            152,
            148,
            149.5,
            154.5,
            153.5,
            155.5,
            149.5,
            144,
            149,
            147,
            153.5,
            146,
            150,
            149.5,
            145.5,
            146,
            140.5,
            133.5,
            136,
        ]
        map_row3 = [
            257,
            202,
            176,
            146,
            143,
            138,
            126,
            128,
            145.5,
            130,
            124.5,
            114,
            122.5,
            119,
            120.5,
            125,
            134.5,
            125.5,
            118,
            126,
            122,
            120,
            120.5,
            125,
            115.5,
            114.5,
            116,
            122.5,
            119.5,
            115,
            121,
            116.5,
            112.5,
            108.5,
        ]
        map_row4 = [
            262,
            206.5,
            183,
            157.5,
            158,
            143.5,
            137,
            135,
            154,
            145.5,
            138.5,
            128.5,
            152,
            152.5,
            160,
            156.5,
            150.5,
            154,
            160.5,
            147,
            152,
            158.5,
            148.5,
            142.5,
            156.5,
            147.5,
            157,
            147,
            149.5,
            146.5,
            144.5,
            142.5,
            135.5,
            136.5,
        ]
        map_row5 = [
            254.5,
            206.5,
            178,
            169,
            148.5,
            139,
            140.5,
            136.5,
            147.5,
            148.5,
            135,
            137.5,
            131.5,
            130,
            120,
            125,
            122.5,
            124,
            124.5,
            133.5,
            129,
            125.5,
            115.5,
            114,
            110,
            120,
            108.5,
            121,
            114,
            116.5,
            115.5,
            119.5,
            108.5,
            108.5,
        ]

        assert map_row0[::-1] == self.s.inav[0, 0].data.tolist()
        assert map_row1[::-1] == self.s.inav[1, 0].data.tolist()
        assert map_row2[::-1] == self.s.inav[2, 0].data.tolist()
        assert map_row3[::-1] == self.s.inav[0, 1].data.tolist()
        assert map_row4[::-1] == self.s.inav[1, 1].data.tolist()
        assert map_row5[::-1] == self.s.inav[2, 1].data.tolist()
        np.testing.assert_allclose(self.s.data, self.s_non_uniform.data)

    def test_axes(self):
        map_axes = {
            "axis-0": {
                "_type": "UniformDataAxis",
                "name": "Y",
                "units": "µm",
                "navigate": True,
                "is_binned": False,
                "size": 2,
                "scale": 2,
                "offset": -1,
            },
            "axis-1": {
                "_type": "UniformDataAxis",
                "name": "X",
                "units": "µm",
                "navigate": True,
                "is_binned": False,
                "size": 3,
                "scale": 2,
                "offset": -2,
            },
            "axis-2": {
                "_type": "UniformDataAxis",
                "name": "Wavelength",
                "units": "nm",
                "navigate": False,
                "is_binned": False,
                "size": 34,
            },
        }

        uniform_axis_manager = deepcopy(self.s.axes_manager.as_dictionary())

        np.testing.assert_allclose(
            uniform_axis_manager["axis-2"].pop("scale"), 1.5416, atol=0.0001
        )
        np.testing.assert_allclose(
            uniform_axis_manager["axis-2"].pop("offset"), 720.967, atol=0.05
        )

        assert map_axes == uniform_axis_manager


class TestGlue:
    @classmethod
    def setup_class(cls):
        cls.s = hs.load(
            testfile_glue_path, reader="JobinYvon", use_uniform_signal_axis=True
        )
        cls.s_non_uniform = hs.load(
            testfile_glue_path, reader="JobinYvon", use_uniform_signal_axis=False
        )

    @classmethod
    def teardown_class(cls):
        del cls.s
        del cls.s_non_uniform
        gc.collect()

    def test_data(self):
        assert np.isclose(self.s.isig[0].data, 238)
        assert np.isclose(self.s.isig[-1].data, 254.265)

    def test_metadata(self):
        original_metadata = self.s.original_metadata.as_dictionary()
        metadata = self.s.metadata.as_dictionary()

        np.testing.assert_allclose(
            original_metadata["experimental_setup"]["Windows"], 4
        )
        np.testing.assert_allclose(
            metadata["Acquisition_instrument"]["Detector"]["glued_spectrum_windows"], 4
        )
        assert metadata["Acquisition_instrument"]["Detector"]["glued_spectrum"] is True
