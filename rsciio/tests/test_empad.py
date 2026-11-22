# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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
import importlib
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
from packaging.version import Version

from rsciio.empad import file_reader
from rsciio.utils._tests import normalize_micro

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
t = pytest.importorskip("traits.api", reason="traits not installed")


DATA_DIR = Path(__file__).parent / "data" / "empad"
FILENAME_STACK_RAW = DATA_DIR / "series_x10.raw"
FILENAME_MAP_RAW = DATA_DIR / "scan_x4_y4.raw"
# Version 1.2.0 and 1.2.2 xml files are identical
# keep 1.2.2 version with non-square scan dimensions for testing
FILENAME_MAP_RAW_VERSION_1_2_2 = DATA_DIR / "scan_x32_y64.raw"


if importlib.util.find_spec("pyxem") and Version(version("pyxem")) >= Version("0.19"):
    EXPECTED_UNSPECIFIED_UNITS = "px"
else:
    EXPECTED_UNSPECIFIED_UNITS = t.Undefined


def _create_raw_data(filename, shape):
    size = np.prod(shape)
    data = np.arange(size).reshape(shape).astype("float32")
    data.tofile(filename)


def setup_module(module):
    _create_raw_data(FILENAME_STACK_RAW, (166400,))
    _create_raw_data(FILENAME_MAP_RAW, (4 * 4 * 130 * 128))
    _create_raw_data(FILENAME_MAP_RAW_VERSION_1_2_2, (64 * 32 * 130 * 128))


def teardown_module(module):
    # run garbage collection to release file on windows
    gc.collect()

    fs = [
        f
        for f in [FILENAME_STACK_RAW, FILENAME_MAP_RAW, FILENAME_MAP_RAW_VERSION_1_2_2]
        if f.exists()
    ]

    for f in fs:
        # remove file
        f.unlink()


@pytest.mark.parametrize("lazy", (False, True))
def test_read_stack(lazy):
    # xml file version 0.51 211118
    s = hs.load(DATA_DIR / "stack_images.xml", lazy=lazy, file_format="EMPAD")
    assert s.data.dtype == "float32"
    ref_data = np.arange(166400).reshape((10, 130, 128))[..., :128, :]
    np.testing.assert_allclose(s.data, ref_data.astype("float32"))
    signal_axes = s.axes_manager.signal_axes
    assert signal_axes[0].name == "width"
    assert signal_axes[0].size == 128
    assert signal_axes[1].name == "height"
    assert signal_axes[1].size == 128
    for axis in signal_axes:
        units = EXPECTED_UNSPECIFIED_UNITS
        assert axis.units == units
        assert axis.scale == 1.0
        assert axis.offset == -64
    navigation_axes = s.axes_manager.navigation_axes
    assert navigation_axes[0].name == "series_count"
    assert navigation_axes[0].units == "ms"
    assert navigation_axes[0].scale == 1.0
    assert navigation_axes[0].offset == 0.0

    assert s.metadata.General.date == "2019-06-07"
    assert s.metadata.General.time == "13:17:22.590279"
    assert s.metadata.Signal.signal_type == "electron_diffraction"


@pytest.mark.parametrize("lazy", (False, True))
def test_read_map(lazy):
    # xml file version 0.51 211118
    s = hs.load(DATA_DIR / "map4x4.xml", lazy=lazy, file_format="EMPAD")
    assert s.data.dtype == "float32"
    ref_data = np.arange(266240).reshape((4, 4, 130, 128))[..., :128, :]
    np.testing.assert_allclose(s.data, ref_data.astype("float32"))
    signal_axes = s.axes_manager.signal_axes
    assert signal_axes[0].name == "width"
    assert signal_axes[1].name == "height"
    for axis in signal_axes:
        assert axis.units == "1/nm"
        np.testing.assert_allclose(axis.scale, 0.1826537)
        np.testing.assert_allclose(axis.offset, -11.689837)
    navigation_axes = s.axes_manager.navigation_axes
    assert navigation_axes[0].name == "scan_x"
    assert navigation_axes[1].name == "scan_y"
    for axis in navigation_axes:
        assert normalize_micro(axis.units) == "μm"
        np.testing.assert_allclose(axis.scale, 1.1415856)
        np.testing.assert_allclose(axis.offset, 0.0)

    assert s.metadata.General.date == "2019-06-06"
    assert s.metadata.General.time == "13:30:00.164675"
    assert s.metadata.Signal.signal_type == "electron_diffraction"


@pytest.mark.parametrize("lazy", (False, True))
@pytest.mark.parametrize("q_calibration", [None, 0.3315])
def test_read_map_1_2_2(lazy, q_calibration):
    # xml file version 1.2.2 (2021-08-20))
    filename = DATA_DIR / "EMPAD_v1.2.2_x32_y64_size0.8.xml"
    s = hs.load(filename, lazy=lazy, q_calibration=q_calibration, file_format="EMPAD")
    assert s.data.dtype == "float32"
    ref_data = np.arange(34078720).reshape((64, 32, 130, 128))[..., :128, :]
    np.testing.assert_allclose(s.data, ref_data.astype("float32"))
    signal_axes = s.axes_manager.signal_axes
    assert signal_axes[0].name == "width"
    assert signal_axes[0].index_in_array == 3
    assert signal_axes[0].size == 128
    assert signal_axes[1].name == "height"
    assert signal_axes[1].index_in_array == 2
    assert signal_axes[1].size == 128

    for axis in signal_axes:
        if q_calibration is not None:
            assert axis.units == "1/nm"
            np.testing.assert_allclose(axis.scale, q_calibration)
            np.testing.assert_allclose(axis.offset, -64 * q_calibration)
        else:
            np.testing.assert_allclose(axis.scale, 1)
            np.testing.assert_allclose(axis.offset, -64)
            assert axis.units == EXPECTED_UNSPECIFIED_UNITS
    navigation_axes = s.axes_manager.navigation_axes
    assert navigation_axes[0].name == "scan_x"
    assert navigation_axes[0].index_in_array == 1
    assert navigation_axes[0].size == 32
    assert navigation_axes[1].name == "scan_y"
    assert navigation_axes[1].index_in_array == 0
    assert navigation_axes[1].size == 64
    for axis in navigation_axes:
        assert axis.units == "nm"
        np.testing.assert_allclose(axis.scale, 27.6131150)
        np.testing.assert_allclose(axis.offset, 0.0)

    assert s.metadata.General.date == "2026-03-25"
    assert s.metadata.General.time == "16:47:52.586728"
    assert s.metadata.Signal.signal_type == "electron_diffraction"


@pytest.mark.parametrize("lazy", (False, True))
def test_remove_nans(lazy):
    # xml file version 1.2.2 (2021-08-20))
    filename = DATA_DIR / "EMPAD_v1.2.2_x32_y64_size0.8.xml"
    s = hs.load(filename, lazy=lazy, remove_nans=True, file_format="EMPAD")
    assert np.isnan(s.data).sum() == 0


def _write_empad_raw(path):
    raw = np.arange(130 * 128, dtype="<f4").reshape(130, 128)
    raw.tofile(path)
    return raw[:128]


def _write_empad_xml(path, raw_filename):
    path.write_text(
        '<root name="poc">'
        '<timestamp timestamp="1559820600" isoformat="2019-06-06T13:30:00"/>'
        "<software_version>0.51</software_version>"
        "<pix_x>1</pix_x><pix_y>1</pix_y>"
        f'<raw_file filename="{raw_filename}"/></root>'
    )


def _build_empad_escape(tmp_path, raw_filename):
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    outside = tmp_path / "outside.raw"
    _write_empad_raw(outside)
    xml_path = scan_dir / "scan.xml"
    _write_empad_xml(xml_path, raw_filename)
    return xml_path, outside, scan_dir


def test_read_raw_inside_scan_directory(tmp_path):
    scan_dir = tmp_path / "scan"
    scan_dir.mkdir()
    expected = _write_empad_raw(scan_dir / "inside.raw")
    xml_path = scan_dir / "scan.xml"
    _write_empad_xml(xml_path, "inside.raw")

    result = file_reader(str(xml_path))

    assert result[0]["data"].dtype == np.float32
    np.testing.assert_allclose(result[0]["data"], expected)


def test_reject_absolute_raw_filename_outside_scan_directory(tmp_path):
    xml_path, _, _ = _build_empad_escape(tmp_path, str(tmp_path / "outside.raw"))

    with pytest.raises(ValueError, match="outside the scan directory"):
        file_reader(str(xml_path))


def test_reject_parent_raw_filename_outside_scan_directory(tmp_path):
    xml_path, _, _ = _build_empad_escape(tmp_path, "../outside.raw")

    with pytest.raises(ValueError, match="outside the scan directory"):
        file_reader(str(xml_path))


@pytest.mark.skipif(
    os.name == "nt", reason="creating symlinks may require elevated privileges"
)
def test_reject_symlinked_raw_filename_outside_scan_directory(tmp_path):
    xml_path, outside, scan_dir = _build_empad_escape(tmp_path, "link.raw")
    (scan_dir / "link.raw").symlink_to(outside)

    with pytest.raises(ValueError, match="outside the scan directory"):
        file_reader(str(xml_path))
