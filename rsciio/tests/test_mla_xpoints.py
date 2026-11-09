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

"""Tests of the MLA ``.xpoints`` reader.

All files are synthetic, written by the builders in
``generate_mla_xpoints_test_files.py``, so no test data is downloaded.
"""

import logging
import pickle
import struct
from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest

import rsciio
from rsciio.mla_xpoints import _api, _records, file_reader
from rsciio.mla_xpoints._jkf import read_point_index, unpack_packbits
from rsciio.tests.generate_mla_xpoints_test_files import (
    FRAME0,
    FRAME_CENTERS,
    FRAME_SIZES,
    IMAGE0,
    LOCATIONS,
    N_CH,
    PARTICLE0,
    RECORD_BYTES,
    SPECTRA,
    jkf_file,
    jkf_for,
    jkf_frame,
    jkf_particle,
    jkf_point,
    packbits,
    random_spectra,
    xpoints_record,
)

ENERGY_KEV = (-10.04 + 10.0163 * np.arange(N_CH)) / 1000
HEADER_FIELDS = ("x", "y", "width", "height", "bse_mode", "bse_mode_fraction")


@pytest.fixture
def dataset(tmp_path):
    """Five spectra (the fourth needs uint16) and an X-Ray stage ``.jkf`` for them."""
    spectra = random_spectra(5)
    spectra[3, 100] = 400
    records = [xpoints_record(s, *pos) for s, pos in zip(spectra, SPECTRA)]
    sizes = [len(r) for r in records]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).tolist()
    path = tmp_path / "sample_spx.xpoints"
    path.write_bytes(b"".join(records))
    jkf_path = path.with_suffix(".jkf")
    jkf_path.write_bytes(jkf_for(offsets, sizes))
    return SimpleNamespace(
        path=path,
        jkf=jkf_path,
        spectra=spectra,
        offsets=offsets,
        sizes=sizes,
        records=records,
    )


def write_xpoints(tmp_path, records, name="file.xpoints"):
    path = tmp_path / name
    path.write_bytes(b"".join(records))
    return path


# ---------------------------------------------------------------------------
# Spectra, axes and metadata
# ---------------------------------------------------------------------------


def test_plugin_specification():
    (plugin,) = [p for p in rsciio.IO_PLUGINS if p["name"] == "MLA_xpoints"]
    assert plugin["api"] == "rsciio.mla_xpoints"
    assert "xpoints" in plugin["file_extensions"]
    assert plugin["writes"] is False


def test_read_uint8_file(tmp_path):
    spectra = random_spectra(5, peak=200)
    records = [
        xpoints_record(
            s, x=i, y=10 * i, size=(1, 2) if i % 2 else (10, 20), bse=70.0 + i
        )
        for i, s in enumerate(spectra)
    ]
    path = write_xpoints(tmp_path, records)

    (signal,) = file_reader(path)

    data = signal["data"]
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.uint16
    np.testing.assert_array_equal(data, spectra)
    headers = signal["original_metadata"]["spectrum_headers"]
    np.testing.assert_array_equal(headers["x"], np.arange(5))
    np.testing.assert_array_equal(headers["y"], 10 * np.arange(5))
    np.testing.assert_array_equal(headers["width"], [10, 1, 10, 1, 10])
    np.testing.assert_array_equal(headers["height"], [20, 2, 20, 2, 20])
    np.testing.assert_allclose(headers["bse_mode"], 70.0 + np.arange(5))
    np.testing.assert_allclose(headers["bse_mode_fraction"], 0.113)
    assert set(headers) == set(HEADER_FIELDS)


def test_read_mixed_uint8_and_uint16_records(tmp_path):
    spectra = random_spectra(6, peak=200)
    spectra[2, 500] = 300
    spectra[4, 7] = 65535
    path = write_xpoints(tmp_path, [xpoints_record(s) for s in spectra])

    (signal,) = file_reader(path)

    np.testing.assert_array_equal(signal["data"], spectra)
    assert signal["data"].dtype == np.uint16


def test_uint16_records_do_not_break_a_file_that_looks_uniform(tmp_path):
    # 26 uint16 records add 26 * 2000 bytes = 25 * 2080 bytes, so the size of the
    # file is a multiple of the uint8 record size.
    spectra = random_spectra(30)
    spectra[:26, 0] = 1000
    path = write_xpoints(tmp_path, [xpoints_record(s) for s in spectra])
    assert path.stat().st_size % RECORD_BYTES == 0

    (signal,) = file_reader(path)

    assert signal["data"].shape == (30, N_CH)
    np.testing.assert_array_equal(signal["data"], spectra)


@pytest.mark.parametrize("block", [100, 2090, 4097, 1 << 20])
def test_records_that_straddle_the_blocks_read_while_scanning(
    tmp_path, monkeypatch, block
):
    monkeypatch.setattr(_records, "_BLOCK", block)
    spectra = random_spectra(12)
    spectra[::4, 3] = 700
    path = write_xpoints(
        tmp_path, [xpoints_record(s, x=i) for i, s in enumerate(spectra)]
    )

    (signal,) = file_reader(path)

    np.testing.assert_array_equal(signal["data"], spectra)
    np.testing.assert_array_equal(
        signal["original_metadata"]["spectrum_headers"]["x"], np.arange(12)
    )


def test_block_smaller_than_a_header(tmp_path, monkeypatch):
    monkeypatch.setattr(_records, "_BLOCK", 10)
    path = write_xpoints(tmp_path, [xpoints_record(random_spectra(1)[0])])
    with pytest.raises(ValueError, match="changed while it was read"):
        file_reader(path)


def test_padding_bytes_are_ignored(tmp_path):
    rng = np.random.default_rng(3)
    spectra = random_spectra(4)
    records = [
        xpoints_record(s, x=i, garbage=rng.integers(0, 256, 80, np.uint8).tobytes())
        for i, s in enumerate(spectra)
    ]
    path = write_xpoints(tmp_path, records)

    (signal,) = file_reader(path)

    np.testing.assert_array_equal(signal["data"], spectra)
    np.testing.assert_array_equal(
        signal["original_metadata"]["spectrum_headers"]["x"], np.arange(4)
    )


def test_scanned_headers_are_the_record_headers(dataset):
    offsets, payload_bytes, n_channels, headers = _records.scan_records(dataset.path)

    np.testing.assert_array_equal(offsets, dataset.offsets)
    np.testing.assert_array_equal(payload_bytes + 80, dataset.sizes)
    assert n_channels == N_CH
    np.testing.assert_array_equal(headers["tag"], 0x5A)
    np.testing.assert_array_equal(headers["version_a"], 3)
    np.testing.assert_array_equal(headers["version_b"], 2)
    np.testing.assert_array_equal(headers["unknown_44"], 1)
    np.testing.assert_array_equal(headers["n_channels"], N_CH)
    np.testing.assert_array_equal(headers["payload_bytes"], payload_bytes)


def test_single_record(tmp_path):
    path = write_xpoints(tmp_path, [xpoints_record(np.arange(N_CH) % 200)])

    (signal,) = file_reader(path)

    assert signal["data"].shape == (1, N_CH)
    assert signal["axes"][0]["size"] == 1


def test_axes_and_metadata(dataset):
    (signal,) = file_reader(dataset.path)

    navigation_axis, signal_axis = signal["axes"]
    assert navigation_axis == {
        "name": "Spectrum",
        "size": 5,
        "index_in_array": 0,
        "scale": 1.0,
        "offset": 0.0,
        "units": "",
        "navigate": True,
    }
    assert signal_axis["name"] == "Energy"
    assert signal_axis["size"] == N_CH
    assert signal_axis["index_in_array"] == 1
    assert signal_axis["units"] == "keV"
    assert signal_axis["navigate"] is False
    np.testing.assert_allclose(signal_axis["scale"], 0.0100163)
    np.testing.assert_allclose(signal_axis["offset"], -0.01004)
    energy = signal_axis["offset"] + signal_axis["scale"] * np.arange(N_CH)
    np.testing.assert_allclose(energy, ENERGY_KEV)

    assert signal["metadata"] == {
        "General": {"original_filename": "sample_spx.xpoints", "title": "sample_spx"},
        "Signal": {"signal_type": "EDS_SEM", "quantity": "X-rays (Counts)"},
    }


def test_accepts_str_and_path(dataset):
    (from_path,) = file_reader(dataset.path)
    (from_str,) = file_reader(str(dataset.path))
    np.testing.assert_array_equal(from_path["data"], from_str["data"])


def test_other_channel_count_is_not_calibrated(tmp_path, caplog):
    spectra = random_spectra(3, n_channels=1000)
    path = write_xpoints(tmp_path, [xpoints_record(s) for s in spectra])

    with caplog.at_level(logging.WARNING):
        (signal,) = file_reader(path)

    np.testing.assert_array_equal(signal["data"], spectra)
    axis = signal["axes"][1]
    assert (axis["name"], axis["size"], axis["scale"], axis["offset"]) == (
        "Channel",
        1000,
        1.0,
        0.0,
    )
    assert "energy calibration is only known for 2000 channels" in caplog.text


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


def test_lazy_matches_eager(dataset):
    (eager,) = file_reader(dataset.path)
    (lazy,) = file_reader(dataset.path, lazy=True)

    assert isinstance(lazy["data"], da.Array)
    assert lazy["data"].dtype == np.uint16
    assert lazy["data"].shape == (5, N_CH)
    np.testing.assert_array_equal(lazy["data"].compute(), eager["data"])
    np.testing.assert_array_equal(lazy["data"].compute(), dataset.spectra)


@pytest.mark.parametrize(
    "chunks, expected",
    [
        (2, (2, 2, 1)),
        ((2,), (2, 2, 1)),
        ((3, 5), (3, 2)),
        (1, (1,) * 5),
        ("auto", (5,)),
    ],
)
def test_lazy_chunks_hold_complete_spectra(dataset, chunks, expected):
    (signal,) = file_reader(dataset.path, lazy=True, chunks=chunks)

    assert signal["data"].chunks == (expected, (N_CH,))
    np.testing.assert_array_equal(signal["data"].compute(), dataset.spectra)


def test_lazy_chunks_across_many_uint16_records(tmp_path):
    spectra = random_spectra(23)
    spectra[::3, 1] = 999
    path = write_xpoints(tmp_path, [xpoints_record(s) for s in spectra])

    (signal,) = file_reader(path, lazy=True, chunks=4)

    assert signal["data"].chunks[0] == (4, 4, 4, 4, 4, 3)
    np.testing.assert_array_equal(signal["data"].compute(), spectra)
    np.testing.assert_array_equal(signal["data"][5:9].compute(), spectra[5:9])


def test_lazy_data_can_be_pickled_for_other_processes(dataset):
    (signal,) = file_reader(dataset.path, lazy=True, chunks=2)

    copy = pickle.loads(pickle.dumps(signal["data"]))

    np.testing.assert_array_equal(copy.compute(), dataset.spectra)


def test_eager_loading_in_several_blocks(dataset, monkeypatch):
    monkeypatch.setattr(_api, "_EAGER_BLOCK", 2)
    (signal,) = file_reader(dataset.path)
    np.testing.assert_array_equal(signal["data"], dataset.spectra)


def test_lazy_file_that_shrinks_after_loading(dataset):
    (signal,) = file_reader(dataset.path, lazy=True)
    with open(dataset.path, "r+b") as fh:
        fh.truncate(dataset.sizes[0] * 2)
    with pytest.raises(ValueError, match="shorter than its record table"):
        signal["data"].compute()


def test_read_spectra_without_records(dataset):
    from rsciio.mla_xpoints._records import read_spectra

    empty = read_spectra(dataset.path, np.zeros(0, np.int64), np.zeros(0, np.uint32), 7)
    assert empty.shape == (0, 7)
    assert empty.dtype == np.uint16


def test_lazy_reads_the_file_on_compute(dataset):
    (signal,) = file_reader(dataset.path, lazy=True)
    # Nothing is read or cached when loading lazily: only the path is kept
    dataset.path.rename(dataset.path.with_name("moved.xpoints"))
    with pytest.raises(FileNotFoundError):
        signal["data"].compute()


# ---------------------------------------------------------------------------
# Invalid files
# ---------------------------------------------------------------------------


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        file_reader(tmp_path / "missing.xpoints")


def test_empty_file(tmp_path):
    path = write_xpoints(tmp_path, [])
    with pytest.raises(ValueError, match="empty"):
        file_reader(path)


def test_wrong_tag(tmp_path):
    spectra = random_spectra(3)
    records = [bytearray(xpoints_record(s)) for s in spectra]
    records[1][0] = 0x00
    path = write_xpoints(tmp_path, records)
    with pytest.raises(ValueError, match="Record 1 at byte 2080: tag 0x00"):
        file_reader(path)


def test_truncated_header(tmp_path):
    path = write_xpoints(tmp_path, [xpoints_record(random_spectra(1)[0]), bytes(10)])
    with pytest.raises(ValueError, match="truncated header"):
        file_reader(path)


def test_truncated_payload(tmp_path):
    record = xpoints_record(random_spectra(1)[0])
    path = write_xpoints(tmp_path, [record[:-100]])
    with pytest.raises(ValueError, match="payload runs past the end"):
        file_reader(path)


def test_payload_size_does_not_match_channel_count(tmp_path):
    record = bytearray(xpoints_record(random_spectra(1)[0]))
    struct.pack_into("<I", record, 76, 1234)
    path = write_xpoints(tmp_path, [record])
    with pytest.raises(ValueError, match="payload_bytes=1234 does not match"):
        file_reader(path)


def test_zero_channels(tmp_path):
    record = bytearray(xpoints_record(random_spectra(1)[0]))
    struct.pack_into("<H", record, 70, 0)
    path = write_xpoints(tmp_path, [record])
    with pytest.raises(ValueError, match="n_channels=0"):
        file_reader(path)


def test_channel_count_changes_within_file(tmp_path):
    records = [
        xpoints_record(random_spectra(1)[0]),
        xpoints_record(random_spectra(1, n_channels=1000)[0]),
    ]
    path = write_xpoints(tmp_path, records)
    with pytest.raises(ValueError, match="n_channels=1000, the file uses 2000"):
        file_reader(path)


def test_not_an_xpoints_file(tmp_path):
    path = tmp_path / "text.xpoints"
    path.write_text("this is not an xpoints file" * 10)
    with pytest.raises(ValueError, match="not a MLA .xpoints file"):
        file_reader(path)


# ---------------------------------------------------------------------------
# X-Ray stage .jkf index
# ---------------------------------------------------------------------------


def test_jkf_is_read_when_it_sits_next_to_the_file(dataset):
    (signal,) = file_reader(dataset.path)

    jkf = signal["original_metadata"]["jkf"]
    assert jkf["filename"] == "sample_spx.jkf"
    frames, particles, grains = zip(*LOCATIONS)
    np.testing.assert_array_equal(jkf["frame"], frames)
    np.testing.assert_array_equal(jkf["particle"], particles)
    np.testing.assert_array_equal(jkf["grain"], grains)
    x, y = np.array([spectrum[:2] for spectrum in SPECTRA]).T
    cx, cy = np.array([FRAME_CENTERS[f] for f in frames]).T
    frame_width, frame_height = np.array([FRAME_SIZES[f] for f in frames]).T
    # image x runs along stage +X, image rows along stage -Y, 3 um per pixel
    np.testing.assert_allclose(
        jkf["stage_x_um"], cx + (x + 0.5 - frame_width / 2) * 3.0
    )
    np.testing.assert_allclose(
        jkf["stage_y_um"], cy - (y + 0.5 - frame_height / 2) * 3.0
    )
    assert jkf["frame"].dtype == np.int32
    assert jkf["grain"].dtype == np.int16


def test_jkf_arrays_have_one_value_per_spectrum(dataset):
    (signal,) = file_reader(dataset.path, lazy=True)
    original_metadata = signal["original_metadata"]
    for group, names in (
        ("spectrum_headers", HEADER_FIELDS),
        ("jkf", ("frame", "particle", "grain", "stage_x_um", "stage_y_um")),
    ):
        for name in names:
            assert original_metadata[group][name].shape == (5,), (group, name)


@pytest.mark.parametrize("jkf", [False, None])
def test_jkf_can_be_switched_off(dataset, jkf):
    (signal,) = file_reader(dataset.path, jkf=jkf)
    assert list(signal["original_metadata"]) == ["spectrum_headers"]


def test_no_jkf_next_to_the_file(dataset):
    dataset.jkf.unlink()
    (signal,) = file_reader(dataset.path)
    assert list(signal["original_metadata"]) == ["spectrum_headers"]


def test_explicit_jkf_path(dataset, tmp_path):
    other = tmp_path / "elsewhere" / "index.bin"
    other.parent.mkdir()
    dataset.jkf.rename(other)

    (signal,) = file_reader(dataset.path)
    assert "jkf" not in signal["original_metadata"]

    for jkf in (other, str(other)):
        (signal,) = file_reader(dataset.path, jkf=jkf)
        assert signal["original_metadata"]["jkf"]["filename"] == "index.bin"
        np.testing.assert_array_equal(
            signal["original_metadata"]["jkf"]["frame"], [10, 10, 10, 11, 11]
        )


def test_missing_explicit_jkf_path(dataset, tmp_path):
    with pytest.raises(FileNotFoundError):
        file_reader(dataset.path, jkf=tmp_path / "missing.jkf")


@pytest.mark.parametrize("rows", [[1, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 3], [0, 2, 4]])
def test_jkf_may_skip_records(dataset, caplog, rows):
    # The index of sample 5-1a of the dataset skips frames that were dropped in
    # processing, so spectra are matched by byte offset and not by position.
    dataset.jkf.write_bytes(jkf_for(dataset.offsets, dataset.sizes, rows=rows))
    missing = [i for i in range(5) if i not in rows]

    with caplog.at_level(logging.WARNING):
        (signal,) = file_reader(dataset.path)

    jkf = signal["original_metadata"]["jkf"]
    for name, column in (("frame", 0), ("particle", 1), ("grain", 2)):
        expected = [-1 if i in missing else LOCATIONS[i][column] for i in range(5)]
        np.testing.assert_array_equal(jkf[name], expected, err_msg=name)
    assert np.isnan(jkf["stage_x_um"][missing]).all()
    assert np.isnan(jkf["stage_y_um"][missing]).all()
    assert np.isfinite(np.delete(jkf["stage_x_um"], missing)).all()
    assert f"{len(missing)} of 5 spectra have no entry in sample_spx.jkf" in caplog.text
    # the spectra themselves are all read, including the ones the index skips
    np.testing.assert_array_equal(signal["data"], dataset.spectra)


@pytest.mark.parametrize("field", ["x", "y", "width", "height"])
def test_jkf_of_another_file_is_ignored(dataset, caplog, field):
    # one field of the index records differs from the record headers
    dataset.jkf.write_bytes(jkf_for(dataset.offsets, dataset.sizes, shift={field: 1}))

    with caplog.at_level(logging.WARNING):
        (signal,) = file_reader(dataset.path)

    assert "jkf" not in signal["original_metadata"]
    assert "Ignoring sample_spx.jkf" in caplog.text
    assert "does not belong to this file" in caplog.text
    with pytest.raises(ValueError, match="does not belong to this file"):
        file_reader(dataset.path, jkf=dataset.jkf)


def test_jkf_with_other_record_sizes_is_ignored(dataset):
    sizes = [s + 2000 for s in dataset.sizes]
    dataset.jkf.write_bytes(jkf_for(dataset.offsets, sizes))
    with pytest.raises(ValueError, match="does not belong to this file"):
        file_reader(dataset.path, jkf=dataset.jkf)


def test_jkf_without_matching_offsets_is_ignored(dataset):
    dataset.jkf.write_bytes(jkf_for([o + 1 for o in dataset.offsets], dataset.sizes))
    with pytest.raises(ValueError, match="No entry of sample_spx.jkf matches"):
        file_reader(dataset.path, jkf=dataset.jkf)
    (signal,) = file_reader(dataset.path)
    assert "jkf" not in signal["original_metadata"]


def test_jkf_without_frames(dataset):
    dataset.jkf.write_bytes(jkf_file([]))
    with pytest.raises(ValueError, match="No entry of sample_spx.jkf matches"):
        file_reader(dataset.path, jkf=dataset.jkf)


def test_jkf_without_spectra(dataset):
    dataset.jkf.write_bytes(jkf_for(dataset.offsets, dataset.sizes, rows=[]))
    with pytest.raises(ValueError, match="No entry of sample_spx.jkf matches"):
        file_reader(dataset.path, jkf=dataset.jkf)


def test_corrupt_jkf_is_ignored_with_a_warning(dataset, caplog):
    dataset.jkf.write_bytes(dataset.jkf.read_bytes()[:-300])

    with caplog.at_level(logging.WARNING):
        (signal,) = file_reader(dataset.path)

    assert "jkf" not in signal["original_metadata"]
    assert "Ignoring sample_spx.jkf" in caplog.text
    np.testing.assert_array_equal(signal["data"], dataset.spectra)


@pytest.mark.parametrize("reverse", [False, True])
def test_jkf_index_is_sorted_by_offset(dataset, reverse):
    dataset.jkf.write_bytes(jkf_for(dataset.offsets, dataset.sizes, reverse=reverse))

    index = read_point_index(dataset.jkf)

    np.testing.assert_array_equal(index["offset"], dataset.offsets)
    np.testing.assert_array_equal(index["record_bytes"], dataset.sizes)
    np.testing.assert_array_equal(index["width"], [s[2][0] for s in SPECTRA])
    np.testing.assert_array_equal(index["height"], [s[2][1] for s in SPECTRA])
    np.testing.assert_array_equal(index["frame"], [loc[0] for loc in LOCATIONS])
    (signal,) = file_reader(dataset.path)
    np.testing.assert_array_equal(
        signal["original_metadata"]["jkf"]["grain"], [loc[2] for loc in LOCATIONS]
    )


def test_grain_is_unknown_outside_the_particle_image(tmp_path):
    # (x, y) = (4, 1) and (1, 4) lie just beyond the right and bottom edge of the
    # 4 x 4 particle image
    positions = [(1, 1), (4, 1), (1, 4)]
    spectra = random_spectra(3)
    records = [xpoints_record(s, x=x, y=y) for s, (x, y) in zip(spectra, positions)]
    points = [
        jkf_point(i * RECORD_BYTES, RECORD_BYTES, x, y, 10, 88.0, 0.113)
        for i, (x, y) in enumerate(positions)
    ]
    particle = jkf_particle(1, (0, 0, 4, 4), np.full((4, 4), 3), points)
    path = write_xpoints(tmp_path, records, "edge.xpoints")
    path.with_suffix(".jkf").write_bytes(
        jkf_file([jkf_frame(1, (0.0, 0.0), [particle])])
    )

    (signal,) = file_reader(path)

    np.testing.assert_array_equal(
        signal["original_metadata"]["jkf"]["grain"], [3, -1, -1]
    )


def test_jkf_entries_with_huge_offsets_are_not_matched(dataset, caplog):
    # offsets of 2**63 or more must neither match nor disturb the search
    data = bytearray(dataset.jkf.read_bytes())
    (image_length,) = struct.unpack_from("<Q", data, IMAGE0 + 28)
    first_point = IMAGE0 + 40 + image_length + 16
    struct.pack_into("<Q", data, first_point + 8, 2**63 + 5)
    struct.pack_into("<Q", data, first_point + 64 + 8, 2**64 - 1)
    dataset.jkf.write_bytes(bytes(data))

    with caplog.at_level(logging.WARNING):
        (signal,) = file_reader(dataset.path)

    jkf = signal["original_metadata"]["jkf"]
    np.testing.assert_array_equal(jkf["frame"], [-1, -1, 10, 11, 11])
    np.testing.assert_array_equal(jkf["grain"], [-1, -1, 7, 3, -1])
    assert "2 of 5 spectra have no entry" in caplog.text


def test_jkf_with_out_of_range_numbers_is_ignored(dataset, caplog):
    data = bytearray(dataset.jkf.read_bytes())
    struct.pack_into("<I", data, FRAME0 + 0x10, 3_000_000_000)  # frame number
    dataset.jkf.write_bytes(bytes(data))

    with caplog.at_level(logging.WARNING):
        (signal,) = file_reader(dataset.path)

    assert "jkf" not in signal["original_metadata"]
    assert "frame number 3000000000 is out of range" in caplog.text


def test_jkf_with_a_single_frame_and_no_trailing_bytes(tmp_path):
    spectra = random_spectra(1)
    record = xpoints_record(spectra[0], x=1, y=1)
    label = np.full((2, 2), 4)
    point = jkf_point(0, len(record), 1, 1, 10, 88.0, 0.113)
    frame = jkf_frame(1, (0.0, 0.0), [jkf_particle(1, (0, 0, 2, 2), label, [point])])
    path = write_xpoints(tmp_path, [record], "one.xpoints")
    path.with_suffix(".jkf").write_bytes(jkf_file([frame]))

    (signal,) = file_reader(path)

    np.testing.assert_array_equal(signal["original_metadata"]["jkf"]["grain"], [4])


def test_jkf_without_palette_block_is_found_by_scanning(dataset):
    data = bytearray(dataset.jkf.read_bytes())
    struct.pack_into("<I", data, 0x120 + 4, 0)  # palette block type is wrong
    struct.pack_into("<I", data, 0x200, 0x5DC)  # and a stray frame type is in front
    dataset.jkf.write_bytes(bytes(data))

    (signal,) = file_reader(dataset.path)

    np.testing.assert_array_equal(
        signal["original_metadata"]["jkf"]["frame"], [10, 10, 10, 11, 11]
    )


@pytest.mark.parametrize(
    "position, fmt, value, message",
    [
        (FRAME0 + 4, "<I", 0x1234, "not followed by a frame block"),
        (FRAME0 + 8, "<Q", 1 << 40, "bad frame block"),
        (FRAME0 + 0x18, "<I", 1000, "cannot fit"),
        (FRAME0 + 0x18, "<I", 0, "particle chain ends"),
        (PARTICLE0 + 4, "<I", 0x1234, "expected a particle block"),
        (PARTICLE0 + 8, "<Q", 10, "does not fit its frame"),
        (PARTICLE0 + 36 + 8, "<I", 999, "outside the 500x400 frame"),
        (PARTICLE0 + 36 + 12, "<I", 450, "outside the 500x400 frame"),  # > height
        (IMAGE0 + 4, "<I", 0x1234, "expected an image block"),
        (IMAGE0 + 16, "<I", 5, "does not match its bounding box"),
        (PARTICLE0, "<I", 3_000_000_000, "particle number"),
        (PARTICLE0 + 36 + 8, "<I", 10, "outside the 500x400 frame"),  # no pixels
        (IMAGE0 + 24, "<I", 3, "does not match its bounding box"),  # stride < width
        (IMAGE0 + 24, "<I", 1 << 20, "too large to be unpacked"),  # decompression bomb
        (IMAGE0 + 28, "<Q", 1, "inconsistent"),
        (IMAGE0 + 40, "<B", 127, "PackBits literal"),
    ],
)
def test_corrupt_jkf_blocks(dataset, position, fmt, value, message):
    data = bytearray(dataset.jkf.read_bytes())
    struct.pack_into(fmt, data, position, value)
    dataset.jkf.write_bytes(bytes(data))

    with pytest.raises(ValueError, match=message):
        file_reader(dataset.path, jkf=dataset.jkf)


def test_corrupt_points_block(dataset):
    data = bytearray(dataset.jkf.read_bytes())
    (image_length,) = struct.unpack_from("<Q", data, IMAGE0 + 28)
    struct.pack_into("<I", data, IMAGE0 + 40 + image_length + 8, 99)  # points length
    dataset.jkf.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="malformed points block"):
        file_reader(dataset.path, jkf=dataset.jkf)


def test_truncated_particle_block(dataset, tmp_path):
    # 5 bytes are left in the particle block after its image
    particle = bytearray(jkf_particle(1, (0, 0, 2, 2), np.full((2, 2), 4), []))
    particle += bytes(5)
    struct.pack_into("<Q", particle, 8, len(particle))
    path = tmp_path / "short.bin"
    path.write_bytes(jkf_file([jkf_frame(1, (0.0, 0.0), [bytes(particle)])]))

    with pytest.raises(ValueError, match="field runs past its block"):
        file_reader(dataset.path, jkf=path)


def test_not_a_jkf_file(dataset, tmp_path):
    path = tmp_path / "other.bin"
    path.write_bytes(b"XX" + bytes(2000))
    with pytest.raises(ValueError, match="no 'JK' magic"):
        file_reader(dataset.path, jkf=path)

    path.write_bytes(b"JK" + bytes(50))
    with pytest.raises(ValueError, match="too short"):
        file_reader(dataset.path, jkf=path)

    for size in (0x125, 0x800):  # without and with room for a palette block header
        path.write_bytes(b"JK" + bytes(size - 2))
        with pytest.raises(ValueError, match="no frame block found"):
            file_reader(dataset.path, jkf=path)


# ---------------------------------------------------------------------------
# PackBits
# ---------------------------------------------------------------------------


def test_packbits_round_trip():
    rng = np.random.default_rng(0)
    data = bytes(rng.integers(0, 3, 1000, np.uint8)) + bytes(300) + bytes(range(200))
    packed = packbits(data)
    assert unpack_packbits(packed, 0, len(packed), len(data)) == data
    padded = b"abc" + packed + b"xyz"
    assert unpack_packbits(padded, 3, 3 + len(packed), len(data)) == data


def test_packbits_no_op_control_byte():
    assert unpack_packbits(bytes([128, 0, 5]), 0, 3, 1) == b"\x05"


@pytest.mark.parametrize(
    "src, start, end, expected, message",
    [
        (bytes(10), 5, 20, 1, "outside the file"),
        (bytes(10), 6, 4, 1, "outside the file"),
        (bytes([5, 1, 2]), 0, 3, 6, "literal at byte 0 overruns"),
        (bytes([254]), 0, 1, 3, "repeat at byte 0 overruns"),
        (bytes([254, 7]), 0, 2, 2, "output exceeds the expected 2"),
        (bytes([0, 7]), 0, 2, 5, "decoded 1 bytes, expected 5"),
    ],
)
def test_packbits_errors(src, start, end, expected, message):
    with pytest.raises(ValueError, match=message):
        unpack_packbits(src, start, end, expected)


# ---------------------------------------------------------------------------
# HyperSpy
# ---------------------------------------------------------------------------


@pytest.fixture
def hs():
    return pytest.importorskip("hyperspy.api", reason="hyperspy not installed")


def test_load_with_hyperspy(hs, dataset):
    s = hs.load(dataset.path)

    assert s.axes_manager.navigation_shape == (5,)
    assert s.axes_manager.signal_shape == (N_CH,)
    np.testing.assert_array_equal(s.data, dataset.spectra)
    energy = s.axes_manager.signal_axes[0]
    assert energy.units == "keV"
    np.testing.assert_allclose(energy.axis, ENERGY_KEV)
    assert s.metadata.Signal.signal_type == "EDS_SEM"
    assert s.metadata.Signal.quantity == "X-rays (Counts)"
    assert s.metadata.General.title == "sample_spx"
    np.testing.assert_array_equal(
        s.original_metadata.spectrum_headers.x, [spectrum[0] for spectrum in SPECTRA]
    )
    np.testing.assert_array_equal(s.original_metadata.jkf.frame, [10, 10, 10, 11, 11])


def test_load_lazy_with_hyperspy(hs, dataset):
    s = hs.load(dataset.path, lazy=True, chunks=2, jkf=False)

    assert s._lazy
    assert s.data.chunks[0] == (2, 2, 1)
    assert not s.original_metadata.has_item("jkf")
    s.compute()
    np.testing.assert_array_equal(s.data, dataset.spectra)
