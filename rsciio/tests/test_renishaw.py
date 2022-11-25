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
                "is_binned": False,
                "size": 36,
            }
        }

        axes_manager = self.s.axes_manager.as_dictionary()
        axes_manager["axis-0"].pop("_type")
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
        non_uniform_axes_manager["axis-0"].pop("_type")
        np.testing.assert_allclose(
            non_uniform_axes_manager["axis-0"].pop("axis")[:6],
            expected_non_uniform_axis_values,
        )
        axes_manager["axis-0"].pop("size")
        assert axes_manager == non_uniform_axes_manager
