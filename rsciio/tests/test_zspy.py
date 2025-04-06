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

import logging
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
# zarr (because of numcodecs) is only supported on x86_64 machines
zarr = pytest.importorskip("zarr", reason="zarr not installed")

dname = Path(os.path.dirname(__file__)) / "data" / "zspy"


class TestZspy:
    @pytest.fixture
    def signal(self):
        data = np.ones((10, 10, 10, 10))
        s = hs.signals.Signal1D(data)
        return s

    @pytest.mark.parametrize("store_class", [zarr.N5Store, zarr.ZipStore])
    def test_save_store(self, signal, tmp_path, store_class):
        filename = tmp_path / "test_save_store.zspy"
        store = store_class(path=filename)
        signal.save(store)

        if store_class is zarr.ZipStore:
            assert os.path.isfile(filename)
        else:
            assert os.path.isdir(filename)

        store2 = store_class(path=filename)
        signal2 = hs.load(store2)

        np.testing.assert_array_equal(signal2.data, signal.data)

    @pytest.mark.parametrize("close_file", [True, False])
    def test_save_ZipStore_close_file(self, signal, tmp_path, close_file):
        filename = tmp_path / "test_zip_Store.zspy"
        store = zarr.ZipStore(path=filename)
        signal.save(store, close_file=close_file)

        assert os.path.isfile(filename)

        s2 = hs.load(filename)
        np.testing.assert_array_equal(s2.data, signal.data)

    def test_save_ZipStore_mode_warning(self, signal, tmp_path, caplog):
        filename = tmp_path / "test.zspy"
        store = zarr.ZipStore(path=filename)
        signal.save(store)

        with caplog.at_level(logging.WARNING):
            _ = hs.load(filename, mode="r+")
            assert "Specifying `mode` " in caplog.text

    def test_save_wrong_store(self, signal, tmp_path, caplog):
        filename = tmp_path / "testmodels.zspy"
        store = zarr.N5Store(path=filename)
        signal.save(store)

        store2 = zarr.N5Store(path=filename)
        s2 = hs.load(store2)
        np.testing.assert_array_equal(s2.data, signal.data)

        store2 = zarr.NestedDirectoryStore(path=filename)
        with pytest.raises(Exception):
            with caplog.at_level(logging.ERROR):
                _ = hs.load(store2)

    @pytest.mark.parametrize("overwrite", [None, True, False])
    def test_overwrite(self, signal, overwrite, tmp_path):
        filename = tmp_path / "testmodels.zspy"
        signal.save(filename=filename)
        signal2 = signal * 2
        signal2.save(filename=filename, overwrite=overwrite)
        if overwrite is None:
            np.testing.assert_array_equal(signal.data, hs.load(filename).data)
        elif overwrite:
            np.testing.assert_array_equal(signal2.data, hs.load(filename).data)
        else:
            np.testing.assert_array_equal(signal.data, hs.load(filename).data)

    def test_compression_opts(self, tmp_path):
        self.filename = tmp_path / "testfile.zspy"
        from numcodecs import Blosc

        comp = Blosc(cname="zstd", clevel=1, shuffle=Blosc.SHUFFLE)
        hs.signals.BaseSignal([1, 2, 3]).save(self.filename, compressor=comp)
        f = zarr.open(self.filename.__str__(), mode="r+")
        d = f["Experiments/__unnamed__/data"]
        assert d.compressor == comp

    @pytest.mark.parametrize("compressor", (None, "default", "blosc"))
    def test_compression(self, compressor, tmp_path):
        if compressor == "blosc":
            from numcodecs import Blosc

            compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        s = hs.signals.Signal1D(np.ones((3, 3)))
        s.save(
            tmp_path / "test_compression.zspy", overwrite=True, compressor=compressor
        )
        _ = hs.load(tmp_path / "test_compression.zspy")


def test_non_valid_zspy(tmp_path, caplog):
    filename = tmp_path / "testfile.zspy"
    data = np.arange(10)

    f = zarr.group(filename)
    f.create_dataset("dataset", data=data)

    with pytest.raises(IOError):
        with caplog.at_level(logging.ERROR):
            _ = hs.load(filename)


@pytest.mark.parametrize(
    "fname",
    [
        "signal1d_10x10-DirectoryStore.zspy",
        "signal1d_10x10-NestedDirectoryStore.zspy",
        "signal1d_10x10-ZipStore.zspy",
    ],
)
def test_read_zspy_saved_with_zarr_v2(fname):
    """Test reading a zspy file saved with zarr v2 and different stores."""
    fname = dname / fname

    s = hs.load(fname)
    assert s.data.shape == (10, 10)
    assert s.axes_manager.signal_shape == (10,)


def test_read_zspy_saved_with_zarr_v2_ragged_markers():
    """
    Test reading a zspy file with ragged markers saved with zarr v2.

    File created with the following code:

    import hyperspy.api as hs
    import numpy as np

    data = np.arange(20*10*10).reshape((20, 10, 10))
    s = hs.signals.Signal2D(data)

    offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
    for ind in np.ndindex(offsets.shape):
        i = ind[0] / 2
        offsets[ind] = [i, i]
    m = hs.plot.markers.Points(
        offsets=offsets,
        facecolor='orange',
        )

    s.plot()
    s.add_marker(m, permanent=True)
    s.save("signal2d_20x10x10-ragged_markers.zspy", store_type="zip")
    """
    fname = dname / "signal2d_20x10x10-ragged_markers.zspy"

    s = hs.load(fname)
    assert s.data.shape == (20, 10, 10)
    m = s.metadata.Markers.Points
    # The position of the markers increase by 0.5 in each dimension
    # with increase the navigation position
    for i, array in enumerate(m.kwargs["offsets"]):
        np.testing.assert_allclose(array, [i / 2, i / 2])


def test_read_zspy_saved_with_zarr_v2_ragged_markers_unicode():
    """
    Test reading a zspy file with ragged markers and unicode dtype
    saved with zarr v2.

    File created with the following code:

    import hyperspy.api as hs
    import numpy as np

    data = np.ones((5, 10, 10))
    s = hs.signals.Signal2D(data)

    offsets = np.empty(s.axes_manager.navigation_shape, dtype=object)
    texts = np.empty(s.axes_manager.navigation_shape, dtype=object)
    for index in np.ndindex(offsets.shape):
        i = index[0]
        offsets[index] = [i/2, i/2]
        texts[index] = np.array(["a" * (i + 1), "b",][: i + 2])
    m = hs.plot.markers.Texts(
        offsets=offsets,
        texts=texts,
        sizes=3,
        facecolor="black",
    )

    s.add_marker(m, permanent=True)
    s.save("signal2d_20x10x10-ragged_markers_unicode.zspy", store_type="zip")
    """
    fname = dname / "signal2d_20x10x10-ragged_markers_unicode.zspy"

    s = hs.load(fname)
    assert s.data.shape == (5, 10, 10)
    m = s.metadata.Markers.Texts
    assert m.kwargs["offsets"].dtype == object
    assert m.kwargs["texts"].dtype == object
    np.testing.assert_equal(m.kwargs["texts"][0], ["a", "b"])
    np.testing.assert_equal(m.kwargs["texts"][1], ["aa", "b"])


def test_read_zspy_saved_with_zarr_v2_structured_array():
    """
    Test reading a zspy file with structured arrays saved with zarr v2.
    A multidimensional model is good example of a structured array, because
    the parameters map is a structured array.

    import numpy as np
    import hyperspy.api as hs

    # Generate the data and make the spectrum
    data = np.arange(1000, dtype=np.int64).reshape((10, 100))
    s = hs.signals.Signal1D(data)
    s.add_poissonian_noise(random_state=0)

    m = s.create_model()
    line = hs.model.components1D.Expression("a * x + b", name="Affine")
    m.append(line)

    m.multifit()
    m.store()
    s.save("signal1d_10x100-model.zspy", store_type="zip")

    """
    fname = dname / "signal1d_10x100-model.zspy"

    s = hs.load(fname)
    m = s.models.restore("a")

    assert m[0].a.map.dtype == np.dtype(
        [("values", "<f8"), ("std", "<f8"), ("is_set", bool)]
    )


@pytest.mark.parametrize("store_type", ["local", "zip"])
def test_save_store(store_type, tmp_path):
    filename = tmp_path / f"test_save_{store_type}.zspy"

    data = np.ones((10, 10, 10, 10))
    s = hs.signals.Signal1D(data)
    s.save(filename, store_type=store_type)

    if store_type == "zip":
        assert zipfile.is_zipfile(filename)
    else:
        assert filename.is_dir()

    s2 = hs.load(filename)
    np.testing.assert_allclose(s2.data, s.data)


def test_save_store_error(tmp_path):
    # ValueError if store_type is not None and a zarr store
    # is passed to filename
    data = np.ones((10, 10, 10, 10))
    s = hs.signals.Signal1D(data)

    with pytest.raises(ValueError, match="one of 'local' or 'zip'."):
        s.save(tmp_path / "test0.zspy", store_type="unsupported_store_type")

    store = zarr.storage.ZipStore(tmp_path / "test1.zspy")
    with pytest.raises(
        ValueError, match="The `store_type` parameter must be None if a zarr "
    ):
        s.save(filename=store, store_type="zip")
