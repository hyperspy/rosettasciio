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
from pathlib import Path
import shutil
import zipfile

import pytest

from rsciio.quantumdetector._api import MIBProperties

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")


TEST_DATA_DIR = Path(__file__).parent / "data" / "quantumdetector"
ZIP_FILE = TEST_DATA_DIR / "Merlin_Single_Quad.zip"
ZIP_FILE2 = TEST_DATA_DIR / "Merlin_navigation4x2_ROI.zip"
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
    if not TEST_DATA_DIR_UNZIPPED.exists():
        if ZIP_FILE.exists():
            with zipfile.ZipFile(ZIP_FILE, "r") as zipped:
                zipped.extractall(TEST_DATA_DIR_UNZIPPED)

        if ZIP_FILE2.exists():
            with zipfile.ZipFile(ZIP_FILE2, "r") as zipped:
                zipped.extractall(TEST_DATA_DIR_UNZIPPED)


def teardown_module():
    # necessary on windows, to help closing the files...
    gc.collect()
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


def test_mib_properties():
    fname = TEST_DATA_DIR_UNZIPPED / "Single_9_Frame_CounterDepth_1_Rows_256.mib"
    mib_prop = MIBProperties()
    mib_prop.parse_file(fname)
    print(mib_prop)


def test_interrupted_acquisition():
    fname = TEST_DATA_DIR_UNZIPPED / "Single_9_Frame_CounterDepth_1_Rows_256.mib"
    with pytest.raises(ValueError):
        s = hs.load(fname, navigation_shape=(2, 5))

    s = hs.load(TEST_DATA_DIR_UNZIPPED / fname, navigation_shape=(2, 4))
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (2, 4)


def test_non_square():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"
    s = hs.load(fname, navigation_shape=(4, 2))
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (4, 2)
