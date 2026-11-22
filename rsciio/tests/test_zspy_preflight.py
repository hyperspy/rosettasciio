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

import json
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

_zspy_api = pytest.importorskip("rsciio.zspy._api")
zarr = pytest.importorskip("zarr")
numcodecs = pytest.importorskip("numcodecs")

file_reader = _zspy_api.file_reader
_UnsafeCodecError = _zspy_api._UnsafeCodecError
_validate_zspy_codecs = _zspy_api._validate_zspy_codecs

DATA = Path(__file__).parent / "data" / "zspy"
ARRAY = "Experiments/security/data"


class _Payload:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (open, (self.marker, "w"))


def _n5_zszy(root, compression, payload=None):
    store = zarr.N5Store(str(root))
    group = zarr.open_group(store=store, mode="w")
    group.attrs["file_format"] = "ZSpy"
    group.attrs["file_format_version"] = "3.3"
    experiment = group.require_group("Experiments/security")
    for name in ("metadata", "original_metadata", "axis-0"):
        experiment.require_group(name)
    experiment.create_dataset(
        "data",
        shape=(4,),
        chunks=(4,),
        dtype="f8",
        data=np.arange(4.0),
        compressor=None,
    )
    store = zarr.N5Store(str(root))
    key = f"{ARRAY}/attributes.json"
    attributes = json.loads(bytes(store[key]))
    attributes["compression"] = compression
    store[key] = json.dumps(attributes).encode()
    if payload is not None:
        header = struct.pack(">H", 0) + struct.pack(">H", 1) + struct.pack(">I", 4)
        store[f"{ARRAY}/0"] = header + pickle.dumps(payload)
    return root


def _zarr_zszy(root):
    store = zarr.DirectoryStore(str(root))
    group = zarr.open_group(store=store, mode="w")
    group.create_dataset(
        "Experiments/a/data",
        shape=(4,),
        chunks=(4,),
        dtype="f8",
        data=np.arange(4.0),
    )
    return store


def _set_zarray(store, key, **changes):
    metadata = json.loads(store[key])
    metadata.update(changes)
    store[key] = json.dumps(metadata).encode()


@pytest.mark.parametrize(
    "compression",
    [
        {"type": "pickle"},
        {"type": "unused", "id": "pickle", "protocol": 5},
    ],
    ids=["declared-type", "id-override"],
)
def test_n5_pickle_codec_rejected(tmp_path, compression):
    root = _n5_zszy(tmp_path / "s.zspy", compression)

    with pytest.raises(_UnsafeCodecError, match=ARRAY):
        _validate_zspy_codecs(zarr.N5Store(str(root)))


def test_n5_pickle_codec_never_reaches_a_decoder(tmp_path, monkeypatch):
    monkeypatch.setattr(numcodecs.pickles, "ALLOW_UNSAFE_PICKLE_DECODE", True)
    marker = tmp_path / "PWNED"
    root = _n5_zszy(
        tmp_path / "s.zspy",
        {"type": "unused", "id": "pickle", "protocol": 5},
        payload=_Payload(str(marker)),
    )

    with pytest.raises(_UnsafeCodecError):
        file_reader(zarr.N5Store(str(root)))

    assert not marker.exists()


@pytest.mark.parametrize(
    "compression",
    [
        {"type": "gzip", "level": 1},
        {"type": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
        {"type": "zstd", "level": 3},
        {"type": "msgpack"},
        {"type": "raw"},
    ],
    ids=["gzip", "blosc", "zstd", "msgpack", "raw"],
)
def test_safe_n5_compressions_pass_the_preflight(tmp_path, compression):
    root = _n5_zszy(tmp_path / "s.zspy", compression)

    _validate_zspy_codecs(zarr.N5Store(str(root)))


@pytest.mark.parametrize(
    "filename",
    [
        "signal1d_10x10-DirectoryStore.zspy",
        "signal1d_10x10-NestedDirectoryStore.zspy",
        "signal1d_10x10-ZipStore.zspy",
    ],
)
def test_ordinary_zspy_files_still_load(filename):
    result = file_reader(str(DATA / filename))

    np.testing.assert_array_equal(result[0]["data"], np.arange(100).reshape(10, 10))


def test_zarr_pickle_compressor_rejected(tmp_path):
    store = _zarr_zszy(tmp_path / "s.zspy")
    _set_zarray(store, "Experiments/a/data/.zarray", compressor={"id": "pickle"})

    with pytest.raises(_UnsafeCodecError, match="Experiments/a/data"):
        file_reader(str(tmp_path / "s.zspy"))


def test_zarr_pickle_filter_rejected(tmp_path):
    store = _zarr_zszy(tmp_path / "s.zspy")
    _set_zarray(store, "Experiments/a/data/.zarray", filters=[{"id": "pickle"}])

    with pytest.raises(_UnsafeCodecError, match="Experiments/a/data"):
        file_reader(str(tmp_path / "s.zspy"))


def test_malformed_metadata_names_the_array(tmp_path):
    store = _zarr_zszy(tmp_path / "s.zspy")
    store["Experiments/a/data/.zarray"] = b"{not json"

    with pytest.raises(ValueError, match="Experiments/a/data.*could not read"):
        file_reader(str(tmp_path / "s.zspy"))


def test_selected_subtree_ignores_arrays_outside_it(tmp_path):
    store = _zarr_zszy(tmp_path / "s.zspy")
    group = zarr.open_group(store=store, mode="r+")
    group.create_dataset(
        "Experiments/b/data",
        shape=(4,),
        chunks=(4,),
        dtype="f8",
        data=np.arange(4.0),
    )
    _set_zarray(store, "Experiments/a/data/.zarray", compressor={"id": "pickle"})

    _validate_zspy_codecs(store, path="Experiments/b")
    with pytest.raises(_UnsafeCodecError, match="Experiments/a/data"):
        _validate_zspy_codecs(store)


def test_owned_zipstore_is_closed_when_the_file_is_rejected(tmp_path, monkeypatch):
    root = tmp_path / "src"
    store = _zarr_zszy(root)
    _set_zarray(store, "Experiments/a/data/.zarray", compressor={"id": "pickle"})
    filename = tmp_path / "hostile.zspy"
    with zipfile.ZipFile(filename, "w") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))

    closed = []
    real_close = zarr.storage.ZipStore.close
    monkeypatch.setattr(
        zarr.storage.ZipStore,
        "close",
        lambda self: (closed.append(self), real_close(self)),
    )

    with pytest.raises(_UnsafeCodecError):
        file_reader(str(filename))

    assert len(closed) == 1


@pytest.mark.parametrize("store_name", ["N5Store", "N5FSStore"])
def test_physical_zarray_inside_an_n5_store_fails_closed(tmp_path, store_name):
    root = tmp_path / "n5-physical"
    store = zarr.N5Store(str(root))
    store["a/attributes.json"] = json.dumps(
        {
            "dimensions": [4],
            "blockSize": [4],
            "dataType": "<f8",
            "compression": {"type": "raw"},
        }
    ).encode()
    # N5Store translates ".zarray" to attributes.json on read, so the raw
    # file is written through a plain DirectoryStore.
    zarr.storage.DirectoryStore(str(root))["a/.zarray"] = json.dumps(
        {
            "zarr_format": 2,
            "shape": [4],
            "chunks": [4],
            "dtype": "<f8",
            "compressor": {"id": "pickle"},
            "filters": None,
            "fill_value": 0,
            "order": "C",
        }
    ).encode()

    with pytest.raises(_UnsafeCodecError, match="'a'"):
        _validate_zspy_codecs(getattr(zarr, store_name)(str(root)))
