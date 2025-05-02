# -*- coding: utf-8 -*-
# Copyright 2024 The HyperSpy developers
#
# This file is part of rosettasciio.
#
# rosettasciio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# rosettasciio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with rosettasciio. If not, see <https://www.gnu.org/licenses/#GPL>.


import numpy as np
import pytest

from rsciio.tests.registry import TEST_DATA_REGISTRY

h5py = pytest.importorskip("h5py", reason="h5py not installed")
hdf5plugin = pytest.importorskip("hdf5plugin", reason="hdf5plugin not installed")


from rsciio.arina import file_reader  # noqa: E402


@pytest.fixture
def test_file():
    """Get the path to the test file."""
    return TEST_DATA_REGISTRY.fetch("arina/test_00_master.h5")


def test_file_reader(test_file):
    """Test basic file reading functionality."""
    result = file_reader(test_file)
    assert isinstance(result, list)
    assert len(result) == 1
    result = result[0]  # Get the first dictionary

    assert isinstance(result, dict)
    assert "data" in result
    assert "axes" in result
    assert "metadata" in result
    assert "original_metadata" in result
    assert "post_process" in result
    assert "mapping" in result

    # Check data structure
    assert isinstance(result["data"], np.ndarray)
    assert len(result["data"].shape) == 4

    # Check axes information
    assert len(result["axes"]) == 4
    for axis in result["axes"]:
        assert "name" in axis
        assert "scale" in axis
        assert "offset" in axis
        assert "units" in axis
        assert "size" in axis
        assert axis["offset"] == 0.0
        if axis["name"] in ["x", "y"]:
            assert axis["units"] == "A"
        else:
            assert axis["units"] == "1"


def test_file_reader_with_binning(test_file):
    """Test file reading with binning."""
    result = file_reader(test_file, rebin_diffraction=2)[0]
    assert isinstance(result["data"], np.ndarray)
    assert len(result["data"].shape) == 4
    # Check that the last two dimensions are halved
    original_shape = file_reader(test_file)[0]["data"].shape
    assert result["data"].shape[-2] == original_shape[-2] // 2
    assert result["data"].shape[-1] == original_shape[-1] // 2


def test_file_reader_with_dtype(test_file):
    """Test file reading with specified dataset dtype."""
    result = file_reader(test_file, dtype=np.float32)[0]
    assert result["data"].dtype == np.float32


def test_file_reader_with_flatfield(test_file):
    """Test file reading with flatfield correction."""
    # Create a simple flatfield
    with h5py.File(test_file, "r") as f:
        shape = f["entry/data/data_000001"].shape[1:]
    flatfield = np.ones(shape, dtype=np.float32)
    result = file_reader(test_file, flatfield=flatfield)[0]
    assert result["data"].dtype == np.float32


def test_file_reader_nonexistent_file():
    """Test file reading with nonexistent file."""
    with pytest.raises(FileNotFoundError):
        file_reader("nonexistent_file.h5")


def test_file_reader_lazy_not_implemented(test_file):
    """Test that lazy loading is not implemented."""
    with pytest.raises(NotImplementedError):
        file_reader(test_file, lazy=True)


def test_arina_reader_navigation_shape(test_file):
    """Test that navigation_shape parameter is correctly applied."""
    s = file_reader(test_file, navigation_shape=(4, 4))
    assert s[0]["data"].shape == (4, 4, 192, 192)


def test_file_reader_invalid_navigation_shape(test_file):
    """Test file reading with invalid navigation_shape."""
    # Get the total number of images
    with h5py.File(test_file, "r") as f:
        nimages = sum(f["entry/data"][dset].shape[0] for dset in f["entry/data"])
    # Use a scan width that doesn't divide evenly into the total number of images
    invalid_navigation_shape = nimages + 1
    with pytest.raises(ValueError):
        file_reader(test_file, navigation_shape=(invalid_navigation_shape, "auto"))
