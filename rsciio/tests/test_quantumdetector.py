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

import numpy as np
import pytest

from rsciio.quantumdetector._api import (
    MIBProperties,
    load_mib_data,
    parse_exposures,
    parse_timestamps,
)

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


def test_mib_properties_single__repr__():
    fname = TEST_DATA_DIR_UNZIPPED / "Single_9_Frame_CounterDepth_1_Rows_256.mib"
    mib_prop = MIBProperties()
    mib_prop.parse_file(str(fname))
    assert "\nPath: " == mib_prop.__repr__()[:7]


def test_mib_properties_quad__repr__():
    fname = TEST_DATA_DIR_UNZIPPED / "Quad_9_Frame_CounterDepth_1_Rows_256.mib"
    mib_prop = MIBProperties()
    mib_prop.parse_file(str(fname))
    assert "\nPath: " == mib_prop.__repr__()[:7]


def test_interrupted_acquisition():
    fname = TEST_DATA_DIR_UNZIPPED / "Single_9_Frame_CounterDepth_1_Rows_256.mib"
    # There is only 9 frames, simulate interrupted acquisition using 10 lines
    s = hs.load(fname, navigation_shape=(10, 2))
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (4, 2)

    s = hs.load(TEST_DATA_DIR_UNZIPPED / fname, navigation_shape=(2, 4))
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (2, 4)


def test_non_square():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"
    s = hs.load(fname, navigation_shape=(4, 2))
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (4, 2)


def test_load_mib_data():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"
    data = load_mib_data(str(fname))
    assert data.shape == (8, 256, 256)

    data = load_mib_data(str(fname), navigation_shape=(4, 2))
    assert data.shape == (2, 4, 256, 256)

    data, header = load_mib_data(str(fname), return_header=True)
    assert data.shape == (8, 256, 256)
    assert header.shape == (8,)


def test_test_load_mib_data_from_buffer():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    with open(fname, mode="rb") as f:
        data = load_mib_data(f.read())

    assert data.shape == (8, 256, 256)

    with open(fname, mode="rb") as f:
        data = load_mib_data(f.read(), navigation_shape=(4, 2))

    assert data.shape == (2, 4, 256, 256)


def test_parse_exposures():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    data, header = load_mib_data(str(fname), return_header=True)
    exposures = parse_exposures(header[0])
    assert exposures == [100.0]

    exposures = parse_exposures(header)
    assert exposures == [100.0] * header.shape[0]


def test_parse_timestamps():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    data, header = load_mib_data(str(fname), return_header=True)
    timestamps = parse_timestamps(header[0])
    assert timestamps == ["2021-05-07T16:51:10.905800928Z"]

    timestamps = parse_timestamps(header)
    assert len(timestamps) == len(header)


def test_metadata():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    s = hs.load(fname)
    md_gen = s.metadata.General
    assert md_gen.date == "2021-05-07"
    assert md_gen.time == "16:51:10.905800928"
    assert md_gen.time_zone == "UTC"
    np.testing.assert_allclose(s.metadata.Acquisition_instrument.dwell_time, 1e-4)


def test_print_info():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    _ = hs.load(fname, print_info=True)
