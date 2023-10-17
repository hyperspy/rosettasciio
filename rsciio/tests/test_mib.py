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

import os
from pathlib import Path
import shutil
import zipfile

import pytest

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")


TEST_DATA_DIR = Path(__file__).parent / "data" / "quantumdetector"
ZIP_FILE = TEST_DATA_DIR / "Merlin_Single_Quad.zip"
TEST_DATA_DIR_UNZIPPED = TEST_DATA_DIR / "unzipped"


SINGLE_CHIP_FNAME_LIST = [
    f"Single_{frame}_Frame_CounterDepth_{depth}_Rows_256.mib"
    for frame in [1, 9]
    for depth in [1, 6, 12, 24]
]


QUAD_CHIP_FNAME_LIST = [
    f"Quad_{frame}_Frame_CounterDepth_{depth}_Rows_256.mib"
    for frame in [1, 9]
    for depth in [1, 6, 12, 24]
]


def filter_list(fname_list, string):
    return [fname for fname in fname_list if string in fname]


def setup_module():
    if ZIP_FILE.exists() and not TEST_DATA_DIR_UNZIPPED.exists():
        with zipfile.ZipFile(ZIP_FILE, "r") as zipped:
            zipped.extractall(TEST_DATA_DIR_UNZIPPED)


def teardown_module():
    shutil.rmtree(TEST_DATA_DIR_UNZIPPED)


@pytest.mark.parametrize(
    ("fname", "reshape"),
    zip(
        SINGLE_CHIP_FNAME_LIST + filter_list(SINGLE_CHIP_FNAME_LIST, "9_Frames"),
        [False] * len(SINGLE_CHIP_FNAME_LIST)
        + [True] * len(filter_list(SINGLE_CHIP_FNAME_LIST, "9_Frames")),
    ),
)
def test_single_chip(fname, reshape):
    if "9_Frame" in fname:
        navigation_shape = (3, 3) if reshape else (9,)
    else:
        navigation_shape = ()

    nav_shape = navigation_shape if reshape else None
    s = hs.load(TEST_DATA_DIR_UNZIPPED / fname, navigation_shape=nav_shape)
    assert s.data.shape == navigation_shape + (256, 256)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == navigation_shape

    for axis in s.axes_manager.signal_axes:
        assert axis.scale == 1
        assert axis.offset == 0
        assert axis.units == ""


@pytest.mark.parametrize("fname", QUAD_CHIP_FNAME_LIST)
def test_quad_chip(fname):
    s = hs.load(TEST_DATA_DIR_UNZIPPED / fname)
    if "9_Frame" in fname:
        navigation_shape = (9,)
    else:
        navigation_shape = ()
    assert s.data.shape == navigation_shape + (512, 512)
    assert s.axes_manager.signal_shape == (512, 512)
    assert s.axes_manager.navigation_shape == navigation_shape

    for axis in s.axes_manager.signal_axes:
        assert axis.scale == 1
        assert axis.offset == 0
        assert axis.units == ""
