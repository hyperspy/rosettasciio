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


from pathlib import Path

import numpy as np
import pytest

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")

TEST_DATA_DIR = Path(__file__).parent / "data" / "dens"
FILE1 = TEST_DATA_DIR / "file1.dens"
FILE2 = TEST_DATA_DIR / "file2.dens"
FILE3 = TEST_DATA_DIR / "file3.dens"


ref_T = np.array([15.091, 16.828, 13.232, 50.117, 49.927, 49.986, 49.981])
ref_t = np.array([15.091, 16.828, 13.232, 50.117, 49.927, 49.986, 49.981])


def test_read1():
    s = hs.load(FILE1)
    np.testing.assert_allclose(s.data, ref_T)
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.33)
    np.testing.assert_allclose(s.axes_manager[0].offset, 50077.68)
    assert s.axes_manager[0].units == "s"
    ref_date, ref_time = "2015-04-16", "13:53:00"
    assert s.metadata.General.date == ref_date
    assert s.metadata.General.time == ref_time
    assert s.metadata.Signal.signal_type == ""
    assert s.metadata.Signal.quantity == "Temperature (Celsius)"


def test_read_convert_units():
    s = hs.load(FILE1, convert_units=None)
    np.testing.assert_allclose(s.data, ref_T)
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.33)
    np.testing.assert_allclose(s.axes_manager[0].offset, 50077.68)
    assert s.axes_manager[0].units == "s"

    s = hs.load(FILE1, convert_units=False)
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.33)
    np.testing.assert_allclose(s.axes_manager[0].offset, 50077.68)
    assert s.axes_manager[0].units == "s"

    s = hs.load(FILE1, convert_units=True)
    np.testing.assert_allclose(s.data, ref_T)
    np.testing.assert_allclose(s.axes_manager[0].scale, 330.0)
    np.testing.assert_allclose(s.axes_manager[0].offset, 50077680.0)
    assert s.axes_manager[0].units == "ms"


def test_read2():
    with pytest.raises(AssertionError):
        hs.load(FILE2)


def test_read3():
    with pytest.raises(AssertionError):
        hs.load(FILE3)
