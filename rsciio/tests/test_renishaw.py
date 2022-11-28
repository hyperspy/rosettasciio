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
            68.10285,
            67.45442,
            59.75822,
            61.871353,
            64.90067,
            63.337173,
        ]
        expected_data_end = [
            61.309525,
            64.59786,
            62.75381,
            60.00703,
            67.51018,
            65.36617,
        ]
        np.testing.assert_allclose(expected_data_start, self.s.isig[:6].data)
        np.testing.assert_allclose(expected_data_end, self.s.isig[-6:].data)
        np.testing.assert_allclose(self.s.data, self.s_non_uniform.data)

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
            axes_manager["axis-0"].pop("scale"), -0.0847, rtol=0.0003
        )
        np.testing.assert_allclose(axes_manager["axis-0"].pop("offset"), 328.98, 0.01)
        assert axes_manager == expected_axis

        expected_non_uniform_axis_values = [
            328.98077,
            328.8961,
            328.81137,
            328.72668,
            328.642,
            328.55728,
        ]
        non_uniform_axes_manager = self.s_non_uniform.axes_manager.as_dictionary()
        non_uniform_axes_manager["axis-0"].pop("_type", None)
        non_uniform_axes_manager["axis-0"].pop("is_binned", None)
        np.testing.assert_allclose(
            non_uniform_axes_manager["axis-0"].pop("axis")[:6],
            expected_non_uniform_axis_values,
        )
        axes_manager["axis-0"].pop("size")
        assert axes_manager == non_uniform_axes_manager


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
        expected_column0_start = [
            0.0,
            -0.7469943,
            1.120018,
            0.74636304,
            1.8651184,
            1.4914633,
        ]
        expected_column0_end = [0.0, 1.1034185, 1.1029433, 0.0]
        expected_column4_start = [
            1.1209648,
            0.7469943,
            1.4933574,
            0.37318152,
            1.4920948,
            0.74573165,
        ]
        expected_column4_end = [-0.73592895, 0.36780614, 1.4705911, 0.7349788]

        np.testing.assert_allclose(expected_column0_start, self.s.inav[0].isig[:6])
        np.testing.assert_allclose(expected_column0_end, self.s.inav[0].isig[-4:])
        np.testing.assert_allclose(expected_column4_start, self.s.inav[-1].isig[:6])
        np.testing.assert_allclose(expected_column4_end, self.s.inav[-1].isig[-4:])

    def test_axes(self):
        expected_axis = {
            "axis-0": {
                "name": "X",
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

        assert np.isclose(axes_manager["axis-0"].pop("offset"), -50)
        assert np.isclose(axes_manager["axis-0"].pop("scale"), 21.21320343017578)

        assert np.isclose(axes_manager["axis-1"].pop("offset"), 364.64128336092307)
        assert np.isclose(axes_manager["axis-1"].pop("scale"), -0.08534686462516697)

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
        expected_data_00_start = [2.0856183, 1.3897737, 2.0837028, 1.3884968, 1.3878582]
        expected_data_00_end = [2.3829105, 2.722048]

        np.testing.assert_allclose(
            expected_data_00_start, self.s.inav[0, 0].isig[:5].data
        )
        np.testing.assert_allclose(
            expected_data_00_end, self.s.inav[0, 0].isig[-2:].data
        )

        expected_data_10_start = [0.6952061, 2.7795475, 1.3891352, 2.082745, 0.34696454]
        expected_data_10_end = [2.0424948, 1.361024]

        np.testing.assert_allclose(
            expected_data_10_start, self.s.inav[1, 0].isig[:5].data
        )
        np.testing.assert_allclose(
            expected_data_10_end, self.s.inav[1, 0].isig[-2:].data
        )

        expected_data_01_start = [2.0856183, 2.0846605, 1.3891352, 2.7769935, 1.3878582]
        expected_data_01_end = [1.702079, 0.680512]

        np.testing.assert_allclose(
            expected_data_01_start, self.s.inav[0, 1].isig[:5].data
        )
        np.testing.assert_allclose(
            expected_data_01_end, self.s.inav[0, 1].isig[-2:].data
        )

        expected_data_22_start = [1.0428091, 0.0, 0.6945676, 0.6942484, 1.3878582]
        expected_data_22_end = [1.3616632, 0.680512]

        np.testing.assert_allclose(
            expected_data_22_start, self.s.inav[2, 2].isig[:5].data
        )
        np.testing.assert_allclose(
            expected_data_22_end, self.s.inav[2, 2].isig[-2:].data
        )

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

        assert np.isclose(axes_manager["axis-2"].pop("scale"), -0.0848807)
        assert np.isclose(axes_manager["axis-2"].pop("offset"), 350.676988)

        for key in axes_manager.keys():
            axes_manager[key].pop("_type", None)
            axes_manager[key].pop("is_binned", None)

        assert expected_axis == axes_manager
