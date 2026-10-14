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

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from rsciio.nexus import file_reader  # noqa: E402


def _write_nexus(
    path, root_default=False, entry_default=True, entries=1, root_attributes=True
):
    with h5py.File(path, "w") as f:
        if root_attributes:
            f.attrs["NX_class"] = "NXroot"
        if root_default:
            f.attrs["default"] = "entry0"
        for index in range(entries):
            entry = f.create_group(f"entry{index}")
            entry.attrs["NX_class"] = "NXentry"
            if entry_default:
                entry.attrs["default"] = "data"
            data = entry.create_group("data")
            data.attrs["NX_class"] = "NXdata"
            data.attrs["signal"] = "counts"
            data.attrs["axes"] = ["x"]
            data["counts"] = np.arange(8, dtype="float32") + 100 * index
            data["x"] = np.arange(8, dtype="float64")
            instrument = entry.create_group("instrument")
            instrument.attrs["NX_class"] = "NXinstrument"
            instrument["extra"] = np.full(8, -3, dtype="float32")


def _assert_selected_signal(signals, lazy):
    assert len(signals) == 1
    signal = signals[0]
    assert signal["metadata"]["General"]["title"] == "data"
    data = signal["data"]
    if lazy:
        import dask.array as da

        assert isinstance(data, da.Array)
        data = data.compute()
    assert data.dtype == np.float32
    np.testing.assert_array_equal(data, np.arange(8, dtype="float32"))


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("root_default", [False, True])
def test_single_entry_default(tmp_path, root_default, lazy):
    path = tmp_path / "single_entry.nxs"
    _write_nexus(path, root_default=root_default)

    _assert_selected_signal(file_reader(path, use_default=True, lazy=lazy), lazy)
    assert len(file_reader(path, use_default=False, lazy=lazy)) == 3


@pytest.mark.parametrize("lazy", [False, True])
def test_explicit_default_with_multiple_entries(tmp_path, lazy):
    path = tmp_path / "explicit_default.nxs"
    _write_nexus(path, root_default=True, entries=2)

    _assert_selected_signal(file_reader(path, use_default=True, lazy=lazy), lazy)


@pytest.mark.parametrize("lazy", [False, True])
def test_multiple_entries_without_root_default(tmp_path, lazy):
    path = tmp_path / "multiple_entries.nxs"
    _write_nexus(path, entries=2)

    signals = file_reader(path, use_default=True, lazy=lazy)
    assert len(signals) == 6
    assert sum(s["metadata"]["General"]["title"] == "data" for s in signals) == 2


@pytest.mark.parametrize("lazy", [False, True])
def test_single_entry_without_entry_default(tmp_path, lazy):
    path = tmp_path / "no_default.nxs"
    _write_nexus(path, entry_default=False)

    signals = file_reader(path, use_default=True, lazy=lazy)
    assert len(signals) == 3
    assert {s["metadata"]["General"]["title"] for s in signals} == {
        "data",
        "entry0_data_x",
        "entry0_instrument_extra",
    }


@pytest.mark.parametrize("lazy", [False, True])
def test_attribute_free_root_default(tmp_path, lazy):
    path = tmp_path / "attribute_free_root.nxs"
    _write_nexus(path, root_attributes=False)

    _assert_selected_signal(file_reader(path, use_default=True, lazy=lazy), lazy)
    assert len(file_reader(path, use_default=False, lazy=lazy)) == 3


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("root_attributes", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
def test_hdf5_without_entry_attributes(tmp_path, grouped, root_attributes, lazy):
    path = tmp_path / "plain_hdf5.nxs"
    expected = np.arange(8, dtype="float32")
    with h5py.File(path, "w") as f:
        if root_attributes:
            f.attrs["creator"] = "test"
        parent = f.create_group("plain") if grouped else f
        parent["counts"] = expected

    for use_default in (False, True):
        signals = file_reader(path, use_default=use_default, lazy=lazy)
        assert len(signals) == 1
        data = signals[0]["data"]
        if lazy:
            import dask.array as da

            assert isinstance(data, da.Array)
            data = data.compute()
        assert data.dtype == np.float32
        np.testing.assert_array_equal(data, expected)
