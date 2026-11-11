# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py not installed")
da = pytest.importorskip("dask.array", reason="dask not installed")

from rsciio.usid import file_reader, file_writer  # noqa: E402


def _import_pyusid():
    # pyUSID is only used to check compatibility with the reference
    # implementation; it does not support numpy 2.
    if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
        return None
    try:
        import pyUSID
    except Exception:
        return None
    return pyUSID


pyUSID = _import_pyusid()
requires_pyusid = pytest.mark.skipif(
    pyUSID is None, reason="pyUSID not installed or not usable with numpy >= 2"
)


def write_usid(
    filename,
    data_2d,
    pos_dims,
    spec_dims,
    group="Measurement_000/Channel_000",
    name="Raw_Data",
    fast_to_slow=False,
):
    """Write a USID Main dataset by hand, following the USID specification.

    Dimensions are ``(label, units, values)`` tuples, slowest first. With
    ``fast_to_slow``, they are stored in the reverse order in the ancillary
    datasets, as done by some writers.
    """
    with h5py.File(filename, "a") as f:
        grp = f.require_group(group)
        main = grp.create_dataset(name, data=data_2d)
        main.attrs["quantity"] = "Current"
        main.attrs["units"] = "nA"
        for base, dims, transpose in (
            ("Position", pos_dims, True),
            ("Spectroscopic", spec_dims, False),
        ):
            sizes = [len(values) for _, _, values in dims]
            inds = np.indices(sizes).reshape(len(dims), -1)
            vals = np.stack([np.asarray(v)[i] for (_, _, v), i in zip(dims, inds)])
            labels = [label.encode() for label, _, _ in dims]
            units = [unit.encode() for _, unit, _ in dims]
            if fast_to_slow:
                inds, vals, labels, units = (
                    inds[::-1],
                    vals[::-1],
                    labels[::-1],
                    units[::-1],
                )
            if transpose:
                inds, vals = inds.T, vals.T
            for kind, array in (("Indices", inds.astype(np.uint32)), ("Values", vals)):
                dset = grp.create_dataset(f"{name}_{base}_{kind}", data=array)
                dset.attrs["labels"] = np.array(labels, dtype="S")
                dset.attrs["units"] = np.array(units, dtype="S")
                main.attrs[f"{base}_{kind}"] = dset.ref


def make_signal(data, axes):
    """Signal dictionary with axes given as (name, units, scale, offset, navigate)."""
    return {
        "data": data,
        "axes": [
            {
                "name": name,
                "units": units,
                "scale": scale,
                "offset": offset,
                "navigate": navigate,
                "size": size,
            }
            for (name, units, scale, offset, navigate), size in zip(axes, data.shape)
        ],
        "metadata": {"General": {"title": "test"}, "Signal": {"signal_type": ""}},
        "original_metadata": {"acquisition": {"exposure": 0.5}},
    }


def check_axes(axes, expected):
    """Compare axis dictionaries to (name, units, scale, offset, size) tuples."""
    assert len(axes) == len(expected)
    for axis, (name, units, scale, offset, size) in zip(axes, expected):
        assert "navigate" not in axis
        assert (axis["name"], axis["units"], axis["size"]) == (name, units, size)
        np.testing.assert_allclose(
            [axis["scale"], axis["offset"]], [scale, offset], atol=1e-6
        )


POS_DIMS = [("Y", "um", np.arange(7) * 0.5 + 1), ("X", "um", np.arange(2) * 0.5)]
SPEC_DIMS = [("Bias", "V", np.linspace(-1, 1, 5)), ("Cycle", "", np.arange(3))]
EXPECTED_AXES = [
    ("Y", "um", 0.5, 1, 7),
    ("X", "um", 0.5, 0, 2),
    ("Bias", "V", 0.5, -1, 5),
    ("Cycle", "", 1, 0, 3),
]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@pytest.fixture
def four_d_file(tmp_path):
    data = np.random.default_rng(0).random((7 * 2, 5 * 3))
    filename = tmp_path / "four_d.h5"
    write_usid(filename, data, POS_DIMS, SPEC_DIMS)
    return filename, data.reshape(7, 2, 5, 3)


@pytest.mark.parametrize("lazy", [False, True])
def test_read_position_and_spectroscopic_dims(four_d_file, lazy):
    filename, expected = four_d_file
    signals = file_reader(filename, lazy=lazy)
    assert isinstance(signals, list) and len(signals) == 1
    s = signals[0]

    assert isinstance(s["data"], da.Array) == lazy
    np.testing.assert_allclose(np.asarray(s["data"]), expected)
    check_axes(s["axes"], EXPECTED_AXES)
    assert s["metadata"]["General"]["title"] == "Raw_Data"
    assert Path(s["metadata"]["General"]["original_filename"]) == filename
    assert s["metadata"]["Signal"]["signal_type"] == ""
    om = s["original_metadata"]
    assert (om["quantity"], om["units"]) == ("Current", "nA")
    assert om["dataset_path"] == "/Measurement_000/Channel_000/Raw_Data"
    assert om["original_file_type"] == "USID HDF5"


def test_read_fast_to_slow_storage(tmp_path):
    # The same data with the dimensions stored fastest first
    data = np.random.default_rng(0).random((7 * 2, 5 * 3))
    write_usid(tmp_path / "f.h5", data, POS_DIMS, SPEC_DIMS, fast_to_slow=True)
    s = file_reader(str(tmp_path / "f.h5"))[0]
    np.testing.assert_allclose(s["data"], data.reshape(7, 2, 5, 3))
    check_axes(s["axes"], EXPECTED_AXES)


def test_read_irregular_grid(tmp_path):
    filename = tmp_path / "f.h5"
    write_usid(filename, np.ones((4, 3)), [("X", "m", np.arange(4))], SPEC_DIMS[1:])
    with h5py.File(filename, "a") as f:
        f["Measurement_000/Channel_000/Raw_Data_Position_Indices"][2, 0] = 0
    with pytest.raises(ValueError, match="regular grid"):
        file_reader(filename)


@pytest.mark.parametrize("positions", [True, False])
def test_read_single_value_dims_are_dropped(tmp_path, positions):
    data = np.arange(12.0)
    dims = [("Y", "nm", np.arange(3)), ("X", "nm", np.arange(4))]
    single = [("Arb", "a. u.", [0])]
    if positions:
        write_usid(tmp_path / "f.h5", data.reshape(12, 1), dims, single)
    else:
        write_usid(tmp_path / "f.h5", data.reshape(1, 12), single, dims)

    s = file_reader(tmp_path / "f.h5")[0]
    np.testing.assert_allclose(s["data"], data.reshape(3, 4))
    check_axes(s["axes"], [("Y", "nm", 1, 0, 3), ("X", "nm", 1, 0, 4)])


def test_read_complex(tmp_path):
    data = (np.arange(6) + 1j * np.arange(6)[::-1]).astype(np.complex64)
    write_usid(
        tmp_path / "f.h5",
        data.reshape(2, 3),
        [("X", "m", np.arange(2))],
        [("Freq", "Hz", np.arange(3))],
    )
    s = file_reader(tmp_path / "f.h5")[0]
    assert np.iscomplexobj(s["data"])
    np.testing.assert_allclose(s["data"], data.reshape(2, 3))


@pytest.mark.parametrize("lazy", [False, True])
def test_read_compound(tmp_path, lazy):
    data = np.zeros((4, 5), dtype=[("amp", np.float16), ("phase", np.float32)])
    data["amp"] = np.arange(20).reshape(4, 5)
    data["phase"] = -np.arange(20).reshape(4, 5)
    write_usid(
        tmp_path / "f.h5",
        data,
        [("X", "m", np.arange(4))],
        [("Freq", "Hz", np.arange(5))],
    )

    signals = file_reader(tmp_path / "f.h5", lazy=lazy)
    assert len(signals) == 2
    for s, field in zip(signals, ["amp", "phase"]):
        np.testing.assert_allclose(np.asarray(s["data"]), data[field])
        assert s["original_metadata"]["quantity"] == field
        assert s["original_metadata"]["units"] == ""
        assert s["metadata"]["Signal"]["signal_type"] == "Current"
        assert s["metadata"]["General"]["title"] == "Raw_Data"


@pytest.fixture
def non_uniform_file(tmp_path):
    bias = np.sin(np.linspace(0, np.pi / 2, 10))
    write_usid(
        tmp_path / "f.h5",
        np.ones((3, 10)),
        [("X", "m", np.arange(3))],
        [("Bias", "V", bias)],
    )
    return tmp_path / "f.h5"


def test_read_non_uniform_dim_warns(non_uniform_file):
    with pytest.warns(UserWarning, match="Ignoring non-uniformity of dimension: Bias"):
        s = file_reader(non_uniform_file)[0]
    check_axes(s["axes"], [("X", "m", 1, 0, 3), ("Bias", "V", 1, 0, 10)])


def test_read_non_uniform_dim_raises(non_uniform_file):
    with pytest.raises(ValueError, match="Bias was varied non-uniformly"):
        file_reader(non_uniform_file, ignore_non_uniform_dims=False)


@pytest.fixture
def two_dataset_file(tmp_path):
    filename = tmp_path / "f.h5"
    write_usid(
        filename,
        np.ones((2, 3)),
        [("X", "m", np.arange(2))],
        [("E", "eV", np.arange(3))],
    )
    write_usid(
        filename,
        np.zeros((4, 5)),
        [("Y", "m", np.arange(4))],
        [("t", "s", np.arange(5))],
        group="Measurement_001",
        name="Map",
    )
    with h5py.File(filename, "a") as f:
        f.create_dataset("not_usid", data=np.ones((2, 3)))
    return filename


def test_read_all_datasets(two_dataset_file):
    signals = file_reader(two_dataset_file)
    paths = [s["original_metadata"]["dataset_path"] for s in signals]
    assert paths == ["/Measurement_000/Channel_000/Raw_Data", "/Measurement_001/Map"]
    assert [s["data"].shape for s in signals] == [(2, 3), (4, 5)]
    # Not in a Channel group, so no group parameters
    assert signals[1]["original_metadata"]["parameters"] == {}


def test_read_dataset_path(two_dataset_file):
    signals = file_reader(two_dataset_file, dataset_path="/Measurement_001/Map")
    assert len(signals) == 1
    assert signals[0]["metadata"]["General"]["title"] == "Map"
    np.testing.assert_allclose(signals[0]["data"], np.zeros((4, 5)))


def test_read_group_parameters(two_dataset_file):
    with h5py.File(two_dataset_file, "a") as f:
        f["Measurement_000"].attrs["temperature"] = 300
        f["Measurement_000"].attrs["sample"] = b"Si"
        f["Measurement_000/Channel_000"].attrs["gain"] = 2
    parameters = file_reader(two_dataset_file)[0]["original_metadata"]["parameters"]
    assert parameters == {"temperature": 300, "sample": "Si", "gain": 2}


def test_read_errors(two_dataset_file, tmp_path):
    with pytest.raises(FileNotFoundError):
        file_reader(tmp_path / "missing.h5")
    with pytest.raises(TypeError, match="dataset_path"):
        file_reader(two_dataset_file, dataset_path=1)
    with pytest.raises(KeyError):
        file_reader(two_dataset_file, dataset_path="/does/not/exist")
    with pytest.raises(ValueError, match="not a USID Main dataset"):
        file_reader(two_dataset_file, dataset_path="/not_usid")


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

AXES = [
    ("y", "nm", 0.5, 10.0, True),
    ("x", "nm", 0.25, 0.0, True),
    ("ky", "1/nm", 0.1, -0.4, False),
    ("kx", "1/nm", 0.2, -0.6, False),
]


def check_written_layout(filename):
    with h5py.File(filename, "r") as f:
        assert list(f) == ["Measurement_000"]
        assert list(f["Measurement_000"]) == ["Channel_000"]
        assert set(f["Measurement_000/Channel_000"]) == {
            "Raw_Data",
            "Position_Indices",
            "Position_Values",
            "Spectroscopic_Indices",
            "Spectroscopic_Values",
        }


@pytest.mark.parametrize(
    "axes",
    [AXES, AXES[:2], AXES[2:], AXES[1:3]],
    ids=["nav_and_sig", "nav_only", "sig_only", "one_each"],
)
def test_write_round_trip(tmp_path, axes):
    shape = (4, 3, 5, 6)[: len(axes)]
    data = np.random.default_rng(0).random(shape)
    filename = tmp_path / "out.h5"
    file_writer(filename, make_signal(data, axes))
    check_written_layout(filename)

    s = file_reader(filename)[0]
    np.testing.assert_allclose(s["data"], data)
    check_axes(
        s["axes"],
        [(n, u, sc, off, size) for (n, u, sc, off, _), size in zip(axes, shape)],
    )
    assert s["original_metadata"]["quantity"] == "Unknown Quantity"
    assert s["original_metadata"]["units"] == "Unknown Units"


def test_write_lazy(tmp_path):
    data = da.random.random((4, 3, 5, 6), chunks=(2, 3, 5, 6))
    filename = tmp_path / "out.h5"
    file_writer(filename, make_signal(data, AXES))
    check_written_layout(filename)
    np.testing.assert_allclose(file_reader(filename)[0]["data"], data.compute())


def test_write_kwds(tmp_path):
    file_writer(
        tmp_path / "out.h5", make_signal(np.ones((4, 5)), AXES[1:3]), compression="gzip"
    )
    with h5py.File(tmp_path / "out.h5", "r") as f:
        assert f["Measurement_000/Channel_000/Raw_Data"].compression == "gzip"


def test_write_default_names_and_units(tmp_path):
    axes = [(None, None, 1, 0, True), ("  ", "", 1, 0, True), ("", None, 1, 0, False)]
    file_writer(tmp_path / "out.h5", make_signal(np.ones((2, 3, 4)), axes))
    s = file_reader(tmp_path / "out.h5")[0]
    # Unnamed dimensions are numbered from the fastest one
    assert [a["name"] for a in s["axes"]] == ["Pos_Dim_1", "Pos_Dim_0", "Spec_Dim_0"]
    assert {a["units"] for a in s["axes"]} == {"a. u."}


def test_write_ignores_axis_size(tmp_path):
    signal = make_signal(np.ones((2, 3)), AXES[1:3])
    for axis in signal["axes"]:
        axis["size"] = 100
    file_writer(tmp_path / "out.h5", signal)
    assert file_reader(tmp_path / "out.h5")[0]["data"].shape == (2, 3)


def test_write_axes_mismatch(tmp_path):
    signal = make_signal(np.ones((2, 3)), AXES[2:])
    signal["axes"].append({**signal["axes"][-1], "name": "extra"})
    with pytest.raises(ValueError, match="number of axes"):
        file_writer(tmp_path / "out.h5", signal)


def test_write_appends_to_existing_file(tmp_path):
    filename = tmp_path / "out.h5"
    file_writer(filename, make_signal(np.ones((2, 3)), AXES[:2]))
    file_writer(filename, make_signal(np.zeros((5,)), AXES[3:]))

    signals = file_reader(filename)
    assert [s["original_metadata"]["dataset_path"] for s in signals] == [
        "/Measurement_000/Channel_000/Raw_Data",
        "/Measurement_001/Channel_000/Raw_Data",
    ]
    np.testing.assert_allclose(signals[1]["data"], np.zeros(5))


def test_write_metadata_as_attributes(tmp_path, caplog):
    filename = tmp_path / "out.h5"
    signal = make_signal(np.ones((2, 3)), AXES[1:3])
    signal["metadata"]["General"]["authors"] = ["A", "B"]
    signal["metadata"]["General"]["unset"] = None
    signal["original_metadata"]["unsupported"] = object()
    file_writer(filename, signal)
    assert "'Original-unsupported'" in caplog.text

    with h5py.File(filename, "r") as f:
        attrs = f["Measurement_000"].attrs
        assert attrs["General-title"] == "test"
        assert attrs["Original-acquisition-exposure"] == 0.5
        assert "General-unset" not in attrs
    # and they are read back as group parameters
    parameters = file_reader(filename)[0]["original_metadata"]["parameters"]
    assert parameters["General-title"] == "test"
    assert parameters["General-authors"] == ["A", "B"]


# ---------------------------------------------------------------------------
# Compatibility with pyUSID, the reference implementation
# ---------------------------------------------------------------------------


@requires_pyusid
def test_pyusid_reads_written_file(tmp_path):
    data = np.random.default_rng(0).random((4, 3, 5, 6))
    filename = tmp_path / "out.h5"
    file_writer(filename, make_signal(data, AXES))
    with h5py.File(filename, "r") as f:
        (main,) = pyUSID.hdf_utils.get_all_main(f)
        dataset = pyUSID.USIDataset(main)
        assert list(dataset.pos_dim_labels) == ["y", "x"]
        assert list(dataset.spec_dim_labels) == ["ky", "kx"]
        np.testing.assert_allclose(dataset.get_n_dim_form(), data)
        np.testing.assert_allclose(
            dataset.get_spec_values("kx"), np.arange(6) * 0.2 - 0.6, rtol=1e-6
        )


@requires_pyusid
@pytest.mark.parametrize("slow_to_fast", [True, False])
def test_read_pyusid_file(tmp_path, slow_to_fast):
    data = np.random.default_rng(0).random((7 * 2, 5 * 3))
    dims = [
        [pyUSID.Dimension(label, units, values) for label, units, values in d]
        for d in (POS_DIMS, SPEC_DIMS)
    ]
    if not slow_to_fast:
        dims = [d[::-1] for d in dims]
    pyUSID.ArrayTranslator().translate(
        str(tmp_path / "f.h5"),
        "Raw_Data",
        data,
        "Current",
        "nA",
        *dims,
        slow_to_fast=slow_to_fast,
    )
    s = file_reader(tmp_path / "f.h5")[0]
    np.testing.assert_allclose(s["data"], data.reshape(7, 2, 5, 3))
    check_axes(s["axes"], EXPECTED_AXES)
    assert s["original_metadata"]["quantity"] == "Current"
