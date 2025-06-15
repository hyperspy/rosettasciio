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

import dask.array as da
import numpy as np
import pytest

from rsciio.numpy import file_reader, file_writer
from rsciio.numpy._api import inspect_npy_file


def test_numpy_write_read_cycle(tmp_path):
    """
    Test the file_writer and file_reader functions by writing a numpy array to a file
    and then reading it back, ensuring the data integrity is maintained.
    """
    # Create a sample numpy array
    original_array = np.arange(100 * 100).reshape((100, 100))

    # Write the array to a file
    filename = tmp_path / "test_write_read_cycle.npy"
    file_writer(filename, {"data": original_array})
    d = file_reader(filename)[0]

    read_array = d["data"]
    for axis in d["axes"]:
        assert not axis["navigate"]

    # Check if the original and read arrays are equal
    np.testing.assert_allclose(original_array, read_array)


@pytest.mark.parametrize("navigation_axes", ([0], [0, 1], [], None))
def test_numpy_file_reader_navigation_axes(tmp_path, navigation_axes):
    """
    Test the file_reader function to ensure it correctly reads a numpy file
    and returns the expected array.
    """
    # Create a sample numpy array
    original_array = np.arange(10**4).reshape((10, 10, 10, 10))

    # Write the array to a file
    filename = tmp_path / "test_navigation_axes.npy"
    file_writer(filename, {"data": original_array})
    d = file_reader(filename, navigation_axes=navigation_axes)[0]

    # Check if the original and read arrays are equal
    np.testing.assert_array_equal(original_array, d["data"])
    if navigation_axes is None:
        navigation_axes = []
    for i, axis in enumerate(d["axes"]):
        assert axis["navigate"] == (i in navigation_axes)


def test_numpy_file_reader_lazy(tmp_path):
    # Create a sample numpy array
    original_array = np.arange(10**2).reshape((10, 10))
    filename = tmp_path / "test_read_lazy.npy"
    file_writer(filename, {"data": original_array})

    d = file_reader(filename, lazy=True)[0]
    read_array = d["data"]
    assert isinstance(read_array, da.Array)

    read_array.compute()
    np.testing.assert_allclose(original_array, read_array)


def test_write_lazy(tmp_path):
    # Create a sample numpy array
    original_array = da.arange(10**2).reshape((10, 10))
    filename = tmp_path / "test_write_lazy.npy"
    with pytest.raises(TypeError, match="Lazy signal are not supported"):
        file_writer(filename, {"data": original_array})


@pytest.mark.parametrize("version", ((1, 0), (2, 0), (3, 0)))
def test_numpy_file_reader_version(tmp_path, version):
    original_array = np.arange(100 * 100).reshape((100, 100))

    # Write the array to a file
    filename = tmp_path / "test_write_read_cycle.npy"
    with open(filename, "wb") as f:
        # Use np.lib.format.write_array to write the array with the specified version
        np.lib.format.write_array(f, original_array, version=version)

    # Read the array back
    d = file_reader(filename, lazy=True)[0]
    read_array = d["data"]
    for axis in d["axes"]:
        assert not axis["navigate"]

    # Check if the original and read arrays are equal
    np.testing.assert_allclose(original_array, read_array)


def test_numpy_unicore_header(tmp_path):
    filename = tmp_path / "test_utf8_containing_dtype.npy"
    # Create a sample numpy array with UTF-8 string dtype
    # Define a structured dtype with a unicode field
    dtype = np.dtype([("ΔT", "f4"), ("Time", "f4")])

    # Create a structured array with sample data
    original_array = np.array([(1.5, 0.0), (2.3, 1.0)], dtype=dtype)
    with pytest.warns(UserWarning, match="Stored array in format 3.0"):
        np.save(filename, original_array)

    offset, shape, dtype = inspect_npy_file(filename)
    assert offset == 128  # Check the offset for the header
    assert shape == (2,)  # Check the shape of the array
    assert dtype == [("ΔT", "<f4"), ("Time", "<f4")]  # Check the dtype of the array
