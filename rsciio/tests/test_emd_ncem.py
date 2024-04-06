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


# The EMD format is a hdf5 standard proposed at Lawrence Berkeley
# National Lab (see https://emdatasets.com/ for more information).
# NOT to be confused with the FEI EMD format which was developed later.

import os
import tempfile
from pathlib import Path

import dask.array as da
import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py not installed")
hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")

TEST_DATA_PATH = Path(__file__).parent / "data" / "emd"


# Reference data:
data_signal = np.arange(27).reshape((3, 3, 3)).T
data_image = np.arange(9).reshape((3, 3)).T
data_spectrum = np.arange(3).T
data_save = np.arange(24).reshape((2, 3, 4))
sig_metadata = {"a": 1, "b": 2}
user = {
    "name": "John Doe",
    "institution": "TestUniversity",
    "department": "Microscopy",
    "email": "johndoe@web.de",
}
microscope = {"name": "Titan", "voltage": "300kV"}
sample = {"material": "TiO2", "preparation": "FIB"}
comments = {"comment": "Test"}
test_title = "This is a test!"


@pytest.mark.parametrize("lazy", (True, False))
def test_signal_3d_loading(lazy):
    signal = hs.load(TEST_DATA_PATH / "example_signal.emd", lazy=lazy)
    if lazy:
        signal.compute(close_file=True)
    np.testing.assert_equal(signal.data, data_signal)
    assert isinstance(signal, hs.signals.BaseSignal)


def test_image_2d_loading():
    signal = hs.load(TEST_DATA_PATH / "example_image.emd")
    np.testing.assert_equal(signal.data, data_image)
    assert isinstance(signal, hs.signals.Signal2D)


def test_spectrum_1d_loading():
    signal = hs.load(TEST_DATA_PATH / "example_spectrum.emd")
    np.testing.assert_equal(signal.data, data_spectrum)
    assert isinstance(signal, hs.signals.Signal1D)


def test_metadata():
    signal = hs.load(TEST_DATA_PATH / "example_metadata.emd")
    om = signal.original_metadata
    np.testing.assert_equal(signal.data, data_image)
    np.testing.assert_equal(signal.metadata.General.title, test_title)
    np.testing.assert_equal(om.user.as_dictionary(), user)
    np.testing.assert_equal(om.microscope.as_dictionary(), microscope)
    np.testing.assert_equal(om.sample.as_dictionary(), sample)
    np.testing.assert_equal(om.comments.as_dictionary(), comments)
    assert isinstance(signal, hs.signals.Signal2D)


def test_metadata_with_bytes_string():
    pytest.importorskip("natsort", minversion="5.1.0")
    filename = TEST_DATA_PATH / "example_bytes_string_metadata.emd"
    f = h5py.File(filename, "r")
    dim1 = f["test_group"]["data_group"]["dim1"]
    dim1_name = dim1.attrs["name"]
    dim1_units = dim1.attrs["units"]
    f.close()
    assert isinstance(dim1_name, np.bytes_)
    assert isinstance(dim1_units, np.bytes_)
    _ = hs.load(TEST_DATA_PATH / filename)


def test_data_numpy_object_dtype():
    filename = TEST_DATA_PATH / "example_object_dtype_data.emd"
    signal = hs.load(filename)
    np.testing.assert_equal(signal.data, np.array([["a, 2, test1", "a, 2, test1"]]))


def test_data_axis_length_1():
    filename = TEST_DATA_PATH / "example_axis_len_1.emd"
    signal = hs.load(filename)
    assert signal.data.shape == (5, 1, 5)


class TestDatasetName:
    def setup_method(self):
        tmpdir = tempfile.TemporaryDirectory()
        hdf5_dataset_path = os.path.join(tmpdir.name, "test_dataset.emd")
        f = h5py.File(hdf5_dataset_path, mode="w")
        f.attrs.create("version_major", 0)
        f.attrs.create("version_minor", 2)

        dataset_path_list = [
            "/experimental/science_data_0/data",
            "/experimental/science_data_1/data",
            "/processed/science_data_0/data",
        ]
        data_size_list = [(50, 50), (20, 10), (16, 32)]

        for dataset_path, data_size in zip(dataset_path_list, data_size_list):
            group = f.create_group(os.path.dirname(dataset_path))
            group.attrs.create("emd_group_type", 1)
            group.create_dataset(name="data", data=np.random.random(data_size))
            group.create_dataset(name="dim1", data=range(data_size[0]))
            group.create_dataset(name="dim2", data=range(data_size[1]))

        f.close()

        self.hdf5_dataset_path = hdf5_dataset_path
        self.tmpdir = tmpdir
        self.dataset_path_list = dataset_path_list
        self.data_size_list = data_size_list

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_load_with_dataset_path(self):
        s = hs.load(self.hdf5_dataset_path)
        assert len(s) == len(self.dataset_path_list)
        for dataset_path, data_size in zip(self.dataset_path_list, self.data_size_list):
            s = hs.load(self.hdf5_dataset_path, dataset_path=dataset_path)
            title = os.path.basename(os.path.dirname(dataset_path))
            assert s.metadata.General.title == title
            assert s.data.shape == data_size[::-1]

    def test_load_with_dataset_path_several(self):
        dataset_path = self.dataset_path_list[0:2]
        s = hs.load(self.hdf5_dataset_path, dataset_path=dataset_path)
        assert len(s) == len(dataset_path)
        assert s[0].metadata.General.title in dataset_path[0]
        assert s[1].metadata.General.title in dataset_path[1]

    def test_wrong_dataset_path(self):
        with pytest.raises(IOError):
            hs.load(self.hdf5_dataset_path, dataset_path="a_wrong_name")
        with pytest.raises(IOError):
            hs.load(
                self.hdf5_dataset_path,
                dataset_path=[self.dataset_path_list[0], "a_wrong_name"],
            )


def test_minimal_save(tmp_path):
    signal = hs.signals.Signal1D([0, 1])
    signal.save(tmp_path / "testfile.emd")


def test_load_file(tmp_path):
    hdf5_dataset_path = tmp_path / "test_dataset.emd"
    f = h5py.File(hdf5_dataset_path, mode="w")
    f.attrs.create("version_major", 0)
    f.attrs.create("version_minor", 2)

    group_path_list = ["/exp/data_0/data", "/exp/data_1/data", "/calc/data_0/data"]

    for group_path in group_path_list:
        group = f.create_group(group_path)
        group.attrs.create("emd_group_type", 1)
        data = np.random.random((128, 128))
        group.create_dataset(name="data", data=data)
        group.create_dataset(name="dim1", data=range(128))
        group.create_dataset(name="dim2", data=range(128))

    f.close()

    s = hs.load(hdf5_dataset_path)
    assert len(s) == len(group_path_list)
    for _s, path in zip(s, group_path_list):
        assert _s.metadata.General.title in path


@pytest.mark.parametrize("lazy", (True, False))
def test_save_and_read(lazy, tmp_path):
    signal_ref = hs.signals.BaseSignal(data_save)
    signal_ref.metadata.General.title = test_title
    signal_ref.axes_manager[0].name = "x"
    signal_ref.axes_manager[1].name = "y"
    signal_ref.axes_manager[2].name = "z"
    signal_ref.axes_manager[0].scale = 2
    signal_ref.axes_manager[1].scale = 3
    signal_ref.axes_manager[2].scale = 4
    signal_ref.axes_manager[0].offset = 10
    signal_ref.axes_manager[1].offset = 20
    signal_ref.axes_manager[2].offset = 30
    signal_ref.axes_manager[0].units = "nm"
    signal_ref.axes_manager[1].units = "µm"
    signal_ref.axes_manager[2].units = "mm"
    signal_ref.original_metadata.add_dictionary({"user": user})
    signal_ref.original_metadata.add_dictionary({"microscope": microscope})
    signal_ref.original_metadata.add_dictionary({"sample": sample})
    signal_ref.original_metadata.add_dictionary({"comments": comments})

    signal_ref.save(tmp_path / "example_temp.emd", overwrite=True)
    signal = hs.load(tmp_path / "example_temp.emd", lazy=lazy)
    if lazy:
        signal.compute(close_file=True)
    om = signal.original_metadata
    np.testing.assert_equal(signal.data, signal_ref.data)
    np.testing.assert_equal(signal.axes_manager[0].name, "x")
    np.testing.assert_equal(signal.axes_manager[1].name, "y")
    np.testing.assert_equal(signal.axes_manager[2].name, "z")
    np.testing.assert_equal(signal.axes_manager[0].scale, 2)
    np.testing.assert_almost_equal(signal.axes_manager[1].scale, 3.0)
    np.testing.assert_almost_equal(signal.axes_manager[2].scale, 4.0)
    np.testing.assert_equal(signal.axes_manager[0].offset, 10)
    np.testing.assert_almost_equal(signal.axes_manager[1].offset, 20.0)
    np.testing.assert_almost_equal(signal.axes_manager[2].offset, 30.0)
    np.testing.assert_equal(signal.axes_manager[0].units, "nm")
    np.testing.assert_equal(signal.axes_manager[1].units, "µm")
    np.testing.assert_equal(signal.axes_manager[2].units, "mm")
    np.testing.assert_equal(signal.metadata.General.title, test_title)
    np.testing.assert_equal(om.user.as_dictionary(), user)
    np.testing.assert_equal(om.microscope.as_dictionary(), microscope)
    np.testing.assert_equal(om.sample.as_dictionary(), sample)
    np.testing.assert_equal(om.comments.as_dictionary(), comments)

    assert isinstance(signal, hs.signals.BaseSignal)


def test_chunking_saving_lazy(tmp_path):
    s = hs.signals.Signal2D(da.zeros((50, 100, 100))).as_lazy()
    s.data = s.data.rechunk([50, 25, 25])
    filename = tmp_path / "test_chunking_saving_lazy.emd"
    filename2 = tmp_path / "test_chunking_saving_lazy_chunks_True.emd"
    filename3 = tmp_path / "test_chunking_saving_lazy_chunks_specify.emd"
    s.save(filename)
    s1 = hs.load(filename, lazy=True)
    assert s.data.chunks == s1.data.chunks

    # with chunks=True, use h5py chunking
    s.save(filename2, chunks=True)
    s2 = hs.load(filename2, lazy=True)
    assert tuple([c[0] for c in s2.data.chunks]) == (13, 25, 13)
    s1.close_file()
    s2.close_file()

    # Specify chunks
    chunks = (50, 20, 20)
    s.save(filename3, chunks=chunks)
    s3 = hs.load(filename3, lazy=True)
    assert tuple([c[0] for c in s3.data.chunks]) == chunks
