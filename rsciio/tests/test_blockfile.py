# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
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
import warnings
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from rsciio.utils.date_time_tools import serial_date_to_ISO_format
from rsciio.utils.tests import assert_deep_almost_equal
from rsciio.utils.tools import sarray2dict

try:
    WindowsError
except NameError:
    WindowsError = None

pytest.importorskip("skimage", reason="scikit-image not installed")
hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")

from rsciio.blockfile._api import get_default_header  # noqa: E402

TEST_DATA_DIR = Path(__file__).parent / "data" / "blockfile"
FILE1 = TEST_DATA_DIR / "test1.blo"
FILE2 = TEST_DATA_DIR / "test2.blo"


@pytest.fixture()
def fake_signal():
    fake_data = np.arange(300, dtype=np.uint8).reshape(3, 4, 5, 5)
    fake_signal = hs.signals.Signal2D(fake_data)
    fake_signal.axes_manager[0].scale_as_quantity = "1 mm"
    fake_signal.axes_manager[1].scale_as_quantity = "1 mm"
    fake_signal.axes_manager[2].scale_as_quantity = "1 mm"
    fake_signal.axes_manager[3].scale_as_quantity = "1 mm"
    return fake_signal


@pytest.fixture()
def save_path(tmp_path):
    filepath = tmp_path / "save_temp.blo"
    yield filepath
    # Force files release (required in Windows)
    gc.collect()


ref_data2 = np.array(
    [
        [
            [
                [20, 23, 25, 25, 27],
                [29, 23, 23, 0, 29],
                [24, 0, 0, 22, 18],
                [0, 14, 19, 17, 26],
                [19, 21, 22, 27, 20],
            ],
            [
                [28, 25, 29, 15, 29],
                [12, 15, 12, 25, 24],
                [25, 26, 26, 18, 27],
                [19, 18, 20, 23, 28],
                [28, 18, 22, 25, 0],
            ],
            [
                [21, 29, 25, 19, 18],
                [30, 15, 20, 22, 26],
                [23, 18, 26, 15, 25],
                [22, 25, 24, 15, 20],
                [22, 15, 15, 21, 23],
            ],
        ],
        [
            [
                [28, 25, 26, 24, 26],
                [26, 17, 0, 24, 12],
                [17, 18, 21, 19, 21],
                [21, 24, 19, 17, 0],
                [17, 14, 25, 15, 26],
            ],
            [
                [25, 18, 20, 15, 24],
                [19, 13, 23, 18, 11],
                [0, 25, 0, 0, 14],
                [26, 22, 22, 11, 14],
                [21, 0, 15, 13, 19],
            ],
            [
                [24, 18, 20, 22, 21],
                [13, 25, 20, 28, 29],
                [15, 17, 24, 23, 23],
                [22, 21, 21, 22, 18],
                [24, 25, 18, 18, 27],
            ],
        ],
    ],
    dtype=np.uint8,
)

axes1 = {
    "axis-0": {
        "_type": "UniformDataAxis",
        "name": "y",
        "navigate": True,
        "offset": 0.0,
        "scale": 12.8,
        "size": 3,
        "units": "nm",
        "is_binned": False,
    },
    "axis-1": {
        "_type": "UniformDataAxis",
        "name": "x",
        "navigate": True,
        "offset": 0.0,
        "scale": 12.8,
        "size": 2,
        "units": "nm",
        "is_binned": False,
    },
    "axis-2": {
        "_type": "UniformDataAxis",
        "name": "dy",
        "navigate": False,
        "offset": 0.0,
        "scale": 0.016061676839061997,
        "size": 144,
        "units": "cm",
        "is_binned": False,
    },
    "axis-3": {
        "_type": "UniformDataAxis",
        "name": "dx",
        "navigate": False,
        "offset": 0.0,
        "scale": 0.016061676839061997,
        "size": 144,
        "units": "cm",
        "is_binned": False,
    },
}

axes2 = {
    "axis-0": {
        "_type": "UniformDataAxis",
        "name": "y",
        "navigate": True,
        "offset": 0.0,
        "scale": 64.0,
        "size": 2,
        "units": "nm",
        "is_binned": False,
    },
    "axis-1": {
        "_type": "UniformDataAxis",
        "name": "x",
        "navigate": True,
        "offset": 0.0,
        "scale": 64.0,
        "size": 3,
        "units": "nm",
        "is_binned": False,
    },
    "axis-2": {
        "_type": "UniformDataAxis",
        "name": "dy",
        "navigate": False,
        "offset": 0.0,
        "scale": 0.016061676839061997,
        "size": 5,
        "units": "cm",
        "is_binned": False,
    },
    "axis-3": {
        "_type": "UniformDataAxis",
        "name": "dx",
        "navigate": False,
        "offset": 0.0,
        "scale": 0.016061676839061997,
        "size": 5,
        "units": "cm",
        "is_binned": False,
    },
}

axes2_converted = {
    "axis-0": {
        "_type": "UniformDataAxis",
        "name": "y",
        "navigate": True,
        "offset": 0.0,
        "scale": 64.0,
        "size": 2,
        "units": "nm",
        "is_binned": False,
    },
    "axis-1": {
        "_type": "UniformDataAxis",
        "name": "x",
        "navigate": True,
        "offset": 0.0,
        "scale": 64.0,
        "size": 3,
        "units": "nm",
        "is_binned": False,
    },
    "axis-2": {
        "_type": "UniformDataAxis",
        "name": "dy",
        "navigate": False,
        "offset": 0.0,
        "scale": 160.61676839061997,
        "size": 5,
        "units": "µm",
        "is_binned": False,
    },
    "axis-3": {
        "_type": "UniformDataAxis",
        "name": "dx",
        "navigate": False,
        "offset": 0.0,
        "scale": 160.61676839061997,
        "size": 5,
        "units": "µm",
        "is_binned": False,
    },
}


def test_load1():
    s = hs.load(FILE1)
    assert s.data.shape == (3, 2, 144, 144)
    assert s.axes_manager.as_dictionary() == axes1


@pytest.mark.parametrize(("convert_units"), (True, False))
def test_load2(convert_units):
    s = hs.load(FILE2, convert_units=convert_units)
    assert s.data.shape == (2, 3, 5, 5)
    axes = axes2_converted if convert_units else axes2
    np.testing.assert_equal(s.axes_manager.as_dictionary(), axes)
    np.testing.assert_allclose(s.data, ref_data2)


@pytest.mark.parametrize(("convert_units"), (True, False))
def test_save_load_cycle(save_path, convert_units):
    sig_reload = None
    signal = hs.load(FILE2, convert_units=convert_units)
    serial = signal.original_metadata["blockfile_header"]["Acquisition_time"]
    date, time, timezone = serial_date_to_ISO_format(serial)
    assert signal.metadata.General.original_filename == "test2.blo"
    assert signal.metadata.General.date == date
    assert signal.metadata.General.time == time
    assert signal.metadata.General.time_zone == timezone
    assert (
        signal.metadata.General.notes
        == "Precession angle : \r\nPrecession Frequency : \r\nCamera gamma : on"
    )
    signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path, convert_units=convert_units)
    np.testing.assert_equal(signal.data, sig_reload.data)
    assert (
        signal.axes_manager.as_dictionary() == sig_reload.axes_manager.as_dictionary()
    )
    assert (
        signal.original_metadata.as_dictionary()
        == sig_reload.original_metadata.as_dictionary()
    )
    # change original_filename to make the metadata of both signals equals
    sig_reload.metadata.General.original_filename = (
        signal.metadata.General.original_filename
    )
    # assert file reading tests here, then delete so we can compare
    # entire metadata structure at once:
    plugin = "rsciio.blockfile"
    assert signal.metadata.General.FileIO.Number_0.operation == "load"
    assert signal.metadata.General.FileIO.Number_0.io_plugin == plugin
    assert signal.metadata.General.FileIO.Number_1.operation == "save"
    assert signal.metadata.General.FileIO.Number_1.io_plugin == plugin
    assert sig_reload.metadata.General.FileIO.Number_0.operation == "load"
    assert sig_reload.metadata.General.FileIO.Number_0.io_plugin == plugin
    del signal.metadata.General.FileIO
    del sig_reload.metadata.General.FileIO

    assert_deep_almost_equal(
        signal.metadata.as_dictionary(), sig_reload.metadata.as_dictionary()
    )
    assert signal.metadata.General.date == sig_reload.metadata.General.date
    assert signal.metadata.General.time == sig_reload.metadata.General.time
    assert isinstance(signal, hs.signals.Signal2D)
    # Delete reference to close memmap file!
    del sig_reload


def test_different_x_y_scale_units(save_path):
    # perform load and save cycle with changing the scale on y
    signal = hs.load(FILE2)
    signal.axes_manager[0].scale = 50.0
    signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    np.testing.assert_allclose(sig_reload.axes_manager[0].scale, 50.0, rtol=1e-5)
    np.testing.assert_allclose(sig_reload.axes_manager[1].scale, 64.0, rtol=1e-5)
    np.testing.assert_allclose(sig_reload.axes_manager[2].scale, 0.0160616, rtol=1e-5)


def test_inconvertible_units(save_path, fake_signal):
    fake_signal.axes_manager[2].units = "1/A"
    fake_signal.axes_manager[3].units = "1/A"
    with pytest.warns(UserWarning):
        fake_signal.save(save_path, overwrite=True)


def test_overflow(save_path, fake_signal):
    fake_signal.change_dtype(np.uint16)
    with pytest.warns(UserWarning):
        fake_signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    np.testing.assert_allclose(sig_reload.data, fake_signal.data.astype(np.uint8))


def test_dtype_lims(save_path, fake_signal):
    fake_signal.data = fake_signal.data * 100
    fake_signal.change_dtype(np.uint16)
    fake_signal.save(save_path, intensity_scaling="dtype", overwrite=True)
    sig_reload = hs.load(save_path)
    np.testing.assert_allclose(
        sig_reload.data, (fake_signal.data / 65535 * 255).astype(np.uint8)
    )


def test_dtype_float_fail(save_path, fake_signal):
    fake_signal.change_dtype(np.float32)
    with pytest.raises(ValueError):
        fake_signal.save(save_path, intensity_scaling="dtype", overwrite=True)


def test_minmax_lims(save_path, fake_signal):
    fake_signal.save(save_path, intensity_scaling="minmax", overwrite=True)
    sig_reload = hs.load(save_path)
    np.testing.assert_allclose(
        sig_reload.data,
        (fake_signal.data / fake_signal.data.max() * 255).astype(np.uint8),
    )


def test_crop_lims(save_path, fake_signal):
    fake_signal.save(save_path, intensity_scaling="crop", overwrite=True)
    sig_reload = hs.load(save_path)
    compare = fake_signal.data
    compare[compare > 255] = 255
    np.testing.assert_allclose(sig_reload.data, compare)


def test_tuple_limits(save_path, fake_signal):
    skimage = pytest.importorskip("skimage", reason="scikit-image not installed")
    fake_signal.save(save_path, intensity_scaling=(5, 200), overwrite=True)
    sig_reload = hs.load(save_path)
    compare = skimage.exposure.rescale_intensity(
        fake_signal.data, in_range=(5, 200), out_range=np.uint8
    )
    np.testing.assert_allclose(sig_reload.data, compare)


def test_lazy_save(save_path, fake_signal):
    fake_signal = fake_signal.as_lazy()
    fake_signal.save(save_path, intensity_scaling="minmax", overwrite=True)
    sig_reload = hs.load(save_path)
    compare = (fake_signal.data / fake_signal.data.max() * 255).astype(np.uint8)
    np.testing.assert_allclose(sig_reload.data, compare)


@pytest.mark.parametrize("navigator", [None, "navigator", "array"])
def test_vbfs(save_path, fake_signal, navigator):
    fake_signal = fake_signal.as_lazy()
    if navigator in ["navigator", "array"]:
        fake_signal.compute_navigator()
    if navigator == "array":
        navigator = fake_signal.navigator.data
    fake_signal.save(
        save_path, intensity_scaling=None, navigator=navigator, overwrite=True
    )
    sig_reload = hs.load(save_path)
    np.testing.assert_allclose(sig_reload.data, fake_signal.data)


def test_invalid_vbf(save_path, fake_signal):
    with pytest.raises(ValueError):
        fake_signal.save(
            save_path,
            navigator=hs.signals.Signal2D(np.zeros((10, 10))),
            overwrite=True,
        )


def test_default_header():
    # Simply check that no exceptions are raised
    header = get_default_header()
    assert header is not None


def test_non_square(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(10, 3, 5, 6)).astype(np.uint8))
    with pytest.warns(UserWarning):
        # warning about expect cm units
        with pytest.raises(ValueError):
            signal.save(save_path, overwrite=True)


def test_load_lazy():
    s = hs.load(FILE2, lazy=True)
    assert isinstance(s.data, da.Array)
    s.compute()
    s2 = hs.load(FILE2)
    np.testing.assert_allclose(s.data, s2.data)


def test_load_to_memory():
    s = hs.load(FILE2, lazy=False)
    assert isinstance(s.data, np.ndarray)
    assert not isinstance(s.data, np.memmap)


def test_write_fresh(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8))
    signal.axes_manager["sig"].set(units="cm")
    signal.axes_manager["nav"].set(units="nm")
    signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    np.testing.assert_equal(signal.data, sig_reload.data)
    header = sarray2dict(get_default_header())
    header.update(
        {
            "NX": 3,
            "NY": 10,
            "DP_SZ": 5,
            "SX": 1,
            "SY": 1,
            "SDP": 100,
            "Data_offset_2": 10 * 3 + header["Data_offset_1"],
            "Note": "",
        }
    )
    header["Data_offset_2"] += header["Data_offset_2"] % 16
    assert sig_reload.original_metadata.blockfile_header.as_dictionary() == header


def test_write_data_line(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(3, 5, 5)).astype(np.uint8))
    with pytest.warns(UserWarning):
        # expected units warning
        signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    np.testing.assert_equal(signal.data, sig_reload.data)


def test_write_data_single(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(5, 5)).astype(np.uint8))
    with pytest.warns(UserWarning):
        # expected units warning
        signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    np.testing.assert_equal(signal.data, sig_reload.data)


def test_write_data_am_mismatch(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8))
    signal.axes_manager.navigation_axes[1].size = 4
    with pytest.warns(UserWarning):
        # expected units warning
        with pytest.raises(ValueError):
            signal.save(save_path, overwrite=True)


def test_unrecognized_header_warning(save_path, fake_signal):
    fake_signal.save(save_path, overwrite=True)
    # change magic number
    with open(save_path, "r+b") as f:
        f.seek(6)
        f.write((0xAFAF).to_bytes(2, byteorder="big", signed=False))
    with pytest.warns(UserWarning, match=r"Blockfile has unrecognized .*"):
        hs.load(save_path)


def test_write_cutoff(save_path):
    signal = hs.signals.Signal2D((255 * np.random.rand(10, 3, 5, 5)).astype(np.uint8))
    signal.axes_manager.navigation_axes[0].size = 20
    signal.axes_manager["sig"].set(units="cm")
    signal.axes_manager["nav"].set(units="nm")
    signal.save(save_path, overwrite=True)
    # Test that it raises a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sig_reload = hs.load(save_path)
        # There can be other warnings so >=
        assert len(w) >= 1
        warning_blockfile = [
            "Blockfile header" in str(warning.message) for warning in w
        ]
        assert True in warning_blockfile
        assert issubclass(w[warning_blockfile.index(True)].category, UserWarning)
    cut_data = signal.data.flatten()
    pw = [(0, 17 * 10 * 5 * 5)]
    cut_data = np.pad(cut_data, pw, mode="constant")
    cut_data = cut_data.reshape((10, 20, 5, 5))
    np.testing.assert_equal(cut_data, sig_reload.data)


def test_crop_notes(save_path):
    note_len = 0x1000 - 0xF0
    note = "test123" * 1000  # > note_len
    signal = hs.signals.Signal2D((255 * np.random.rand(2, 3, 2, 2)).astype(np.uint8))
    signal.original_metadata.add_node("blockfile_header.Note")
    signal.original_metadata.blockfile_header.Note = note
    with pytest.warns(UserWarning):
        # expected units warning
        signal.save(save_path, overwrite=True)
    sig_reload = hs.load(save_path)
    assert sig_reload.original_metadata.blockfile_header.Note == note[:note_len]


def test_blo_chunking_lazy(save_path):
    data = np.zeros((90, 121, 144, 144), dtype=np.uint8)
    sig = hs.signals.Signal2D(data)
    sig.save(save_path, overwrite=True)
    new_s = hs.load(save_path, lazy=True, chunks=(1, 1, 144 // 2, 144 // 2))
    assert isinstance(new_s.data, da.Array)
    assert new_s.data.shape == (90, 121, 144, 144)
    assert new_s.data.chunks[0] == (1,) * 90
    assert new_s.data.chunks[1] == (1,) * 121
    assert new_s.data.chunks[2] == (72, 72)
    assert new_s.data.chunks[3] == (72, 72)
