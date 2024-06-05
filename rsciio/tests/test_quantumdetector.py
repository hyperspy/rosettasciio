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
import shutil
import zipfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest
from dask.array.core import normalize_chunks

from rsciio.quantumdetector._api import (
    MIBProperties,
    load_mib_data,
    parse_exposures,
    parse_hdr_file,
    parse_timestamps,
)

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
zarr = pytest.importorskip("zarr", reason="zarr not installed")


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


def _get_expected_dtype_from_fname(fname):
    counter_depth = int(fname.split("CounterDepth_")[1].split("_Rows")[0])
    if counter_depth in [1, 6]:
        dtype = np.dtype(">u1")
    elif counter_depth == 12:
        dtype = np.dtype(">u2")
    else:
        dtype = np.dtype(">u4")
    return dtype


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
    assert s.data.dtype == _get_expected_dtype_from_fname(fname)
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
        if "24_Rows_256" in fname:
            # Unknow why the timestamps of this file are not consistent
            # with others
            navigation_shape = (3, 3)
        else:
            navigation_shape = (9,)
    else:
        navigation_shape = ()
    assert s.data.shape == navigation_shape + (512, 512)
    assert s.data.dtype == _get_expected_dtype_from_fname(fname)
    assert s.axes_manager.signal_shape == (512, 512)
    assert s.axes_manager.navigation_shape == navigation_shape

    for axis in s.axes_manager.signal_axes:
        assert axis.scale == 1
        assert axis.offset == 0
        assert axis.units == ""


@pytest.mark.parametrize(
    "chunks", ("auto", (3, 3, 128, 128), ("auto", "auto", 128, 128))
)
def test_chunks(chunks):
    fname = TEST_DATA_DIR_UNZIPPED / "Quad_9_Frame_CounterDepth_24_Rows_256.mib"
    s = hs.load(fname, lazy=True, chunks=chunks)
    chunks = normalize_chunks(chunks, shape=s.data.shape, dtype=s.data.dtype)
    assert s.data.chunks == chunks


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
    s = hs.load(fname, navigation_shape=(4, 3))
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (4, 2)

    s = hs.load(TEST_DATA_DIR_UNZIPPED / fname, navigation_shape=(2, 4))
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (2, 4)


def test_interrupted_acquisition_first_frame():
    fname = TEST_DATA_DIR_UNZIPPED / "Single_9_Frame_CounterDepth_1_Rows_256.mib"
    # There is only 9 frames, simulate interrupted acquisition using 10 lines
    s = hs.load(fname, navigation_shape=(10, 2), first_frame=1)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (8,)

    s = hs.load(fname, navigation_shape=(10, 2), first_frame=2)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (7,)


@pytest.mark.parametrize("navigation_shape", (None, (8,), (4, 2)))
def test_non_square(navigation_shape):
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"
    s = hs.load(fname, navigation_shape=navigation_shape)
    assert s.axes_manager.signal_shape == (256, 256)
    if navigation_shape is None:
        navigation_shape = (4, 2)
    assert s.axes_manager.navigation_shape == navigation_shape


def test_no_hdr():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"
    fname2 = str(fname).replace(".mib", "-copy.mib")
    shutil.copyfile(fname, fname2)
    s = hs.load(fname2)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (4, 2)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"first_frame": None, "last_frame": None},
        {"first_frame": 0, "last_frame": 9},
        {"first_frame": -9, "last_frame": 9},
        {"first_frame": -9},
        {"first_frame": 0},
        {"last_frame": None},
        {"last_frame": 9},
    ),
)
def test_first_last_frame_all9(kwargs):
    fname = TEST_DATA_DIR_UNZIPPED / "Single_9_Frame_CounterDepth_1_Rows_256.mib"
    s = hs.load(fname, **kwargs)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (9,)
    assert s.data.shape == (9, 256, 256)


@pytest.mark.parametrize("navigation_shape", ((8,), (4, 2)))
@pytest.mark.parametrize(
    "kwargs",
    (
        {"first_frame": 0, "last_frame": -1},
        {"first_frame": 0, "last_frame": 8},
        {"first_frame": 1, "last_frame": 9},
        {"first_frame": -8, "last_frame": 9},
        {"first_frame": 1},
        {"last_frame": -1},
        {"last_frame": 8},
    ),
)
def test_first_last_frame_8(kwargs, navigation_shape):
    fname = TEST_DATA_DIR_UNZIPPED / "Single_9_Frame_CounterDepth_1_Rows_256.mib"
    s = hs.load(fname, navigation_shape=navigation_shape, **kwargs)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == navigation_shape
    assert s.data.shape == navigation_shape[::-1] + (256, 256)


def test_first_last_frame_8_nav_shape_None():
    fname = TEST_DATA_DIR_UNZIPPED / "Single_9_Frame_CounterDepth_1_Rows_256.mib"
    # the navigation_shape will be obtained from the hdf file
    s = hs.load(fname, navigation_shape=None, first_frame=None, last_frame=-1)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (8,)
    assert s.data.shape == (8, 256, 256)

    s = hs.load(fname, navigation_shape=None, first_frame=1, last_frame=None)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (8,)
    assert s.data.shape == (8, 256, 256)


@pytest.mark.parametrize("return_mmap", (True, False))
@pytest.mark.parametrize("lazy", (True, False))
def test_load_mib_data(lazy, return_mmap):
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"
    data = load_mib_data(str(fname), lazy=lazy, return_mmap=return_mmap)
    assert data.shape == (8, 256, 256)
    if return_mmap or not lazy:
        # Even when lazy, it should still be an instance of
        # np.ndarray because it should return the memmap
        assert isinstance(data, np.ndarray)
    else:
        assert isinstance(data, da.Array)

    data = load_mib_data(str(fname), navigation_shape=(4, 2))
    assert data.shape == (2, 4, 256, 256)

    data, headers = load_mib_data(str(fname), return_headers=True)
    assert data.shape == (8, 256, 256)
    assert headers.shape == (8,)


@pytest.mark.parametrize("lazy", (True, False))
def test_load_mib_data_return_mmap_default(lazy):
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"
    data = load_mib_data(str(fname), lazy=lazy)
    # Even if this lazy, it should still be an instance of np.ndarray
    # because it should return the memmap
    assert isinstance(data, np.ndarray)


def test_test_load_mib_data_from_buffer():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    with open(fname, mode="rb") as f:
        data = load_mib_data(f.read())

    assert data.shape == (8, 256, 256)

    with open(fname, mode="rb") as f:
        data = load_mib_data(f.read(), navigation_shape=(4, 2))

    assert data.shape == (2, 4, 256, 256)

    with open(fname, mode="rb") as f:
        with pytest.raises(ValueError):
            # loading lazy memory buffer is not supported
            _ = load_mib_data(f.read(), lazy=True)


@pytest.mark.parametrize("return_mmap", (True, False))
def test_parse_exposures(return_mmap):
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    data, headers = load_mib_data(
        str(fname), return_headers=True, return_mmap=return_mmap
    )
    exposures = parse_exposures(headers[0])
    assert exposures == [100.0]

    exposures = parse_exposures(headers)
    assert exposures == [100.0] * headers.shape[0]


@pytest.mark.parametrize("return_mmap", (True, False))
def test_parse_timestamps(return_mmap):
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    data, headers = load_mib_data(
        str(fname), return_headers=True, return_mmap=return_mmap
    )
    timestamps = parse_timestamps(headers[0])
    assert timestamps == ["2021-05-07T16:51:10.905800928Z"]

    timestamps = parse_timestamps(headers)
    assert len(timestamps) == len(headers)


def test_metadata():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    s = hs.load(fname)
    md_gen = s.metadata.General
    assert md_gen.date == "2021-05-07"
    assert md_gen.time == "16:51:10.905800928"
    assert md_gen.time_zone == "UTC"
    np.testing.assert_allclose(s.metadata.Acquisition_instrument.dwell_time, 1e-1)


def test_print_info():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    _ = hs.load(fname, print_info=True)


def test_navigation_shape_list_error():
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"

    with pytest.raises(TypeError):
        _ = hs.load(fname, navigation_shape=[4, 2])


def test_load_save_cycle(tmp_path):
    fname = TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib"
    s = hs.load(fname, navigation_shape=(4, 2))
    fname2 = tmp_path / "test.zspy"
    s.save(fname2)
    s2 = hs.load(fname2)

    np.testing.assert_allclose(s.data, s2.data)
    assert s.axes_manager.navigation_shape == s2.axes_manager.navigation_shape
    assert s.axes_manager.signal_shape == s2.axes_manager.signal_shape
    assert s.data.dtype == s2.data.dtype


def test_frames_in_acquisition_zero():
    # Some hdr file have entry "Frames per Trigger (Number): 0"
    # Possibly for "continuous and indefinite" acquisition
    # Copy and edit a file with corresponding changes
    base_fname = TEST_DATA_DIR_UNZIPPED / "Single_1_Frame_CounterDepth_6_Rows_256"
    fname = f"{base_fname}_zero_frames_in_acquisition"
    # Create test file using existing test file
    shutil.copyfile(f"{base_fname}.mib", f"{fname}.mib")
    hdf_dict = parse_hdr_file(f"{base_fname}.hdr")
    hdf_dict["Frames in Acquisition (Number)"] = 0
    with open(f"{fname}.hdr", "w") as f:
        f.write("HDR\n")
        for k, v in hdf_dict.items():
            f.write(f"{k}:\t{v}\n")
        f.write("End\t")

    s = hs.load(f"{fname}.mib")
    assert s.axes_manager.navigation_shape == ()


@pytest.mark.parametrize("lazy", (True, False))
def test_distributed(lazy):
    s = hs.load(
        TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib",
        distributed=False,
        lazy=lazy,
    )
    s2 = hs.load(
        TEST_DATA_DIR_UNZIPPED / "001_4x2_6bit.mib",
        distributed=True,
        lazy=lazy,
    )
    if lazy:
        s.compute()
        s2.compute()
    np.testing.assert_array_equal(s.data, s2.data)
