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
import logging
import zipfile
from collections.abc import Mapping, MutableMapping

import numcodecs
import numpy as np
import zarr
from zarr.n5 import compressor_config_to_zarr

from rsciio._docstrings import (
    CHUNKS_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    RETURNS_DOC,
    SHOW_PROGRESSBAR_DOC,
    SIGNAL_DOC,
)
from rsciio._hierarchical import HierarchicalReader, HierarchicalWriter, version
from rsciio.utils._array import is_dask_array
from rsciio.utils._context_manager import get_progress_bar_context_manager

_logger = logging.getLogger(__name__)


# -----------------------
# File format description
# -----------------------
# The root must contain a group called Experiments
# The experiments group can contain any number of subgroups
# Each subgroup is an experiment or signal
# Each subgroup must contain at least one dataset called data
# The data is an array of arbitrary dimension
# In addition a number equal to the number of dimensions of the data
# dataset + 1 of empty groups called coordinates followed by a number
# must exists with the following attributes:
#    'name'
#    'offset'
#    'scale'
#    'units'
#    'size'
#    'index_in_array'
# The experiment group contains a number of attributes that will be
# directly assigned as class attributes of the Signal instance. In
# addition the experiment groups may contain 'original_metadata' and
# 'metadata'subgroup that will be
# assigned to the same name attributes of the Signal instance as a
# Dictionary Browsers
# The Experiments group can contain attributes that may be common to all
# the experiments and that will be accessible as attributes of the
# Experiments instance


class ZspyReader(HierarchicalReader):
    _file_type = "zspy"

    def __init__(self, file):
        super().__init__(file)
        self.Dataset = zarr.Array
        self.Group = zarr.Group


class ZspyWriter(HierarchicalWriter):
    target_size = 1e8
    _file_type = "zspy"
    _unicode_kwds = dict(dtype=str)

    def __init__(self, file, signal, expg, **kwargs):
        super().__init__(file, signal, expg, **kwargs)
        self.Dataset = zarr.Array

    @staticmethod
    def _get_object_dset(group, data, key, chunks, dtype=None, **kwds):
        """Creates a Zarr Array object for saving ragged data

        Forces the number of chunks span the array if not a dask array as
        calculating the chunks for a ragged array is not supported. See
        https://github.com/hyperspy/rosettasciio/issues/168 for more details.
        """
        if not is_dask_array(data):
            chunks = data.shape
        these_kwds = kwds.copy()
        these_kwds.update(dict(dtype=object, exact=True, chunks=chunks))

        if dtype is None:
            test_data = data[data.ndim * (0,)]
            if is_dask_array(test_data):
                test_data = test_data.compute()
            if hasattr(test_data, "dtype"):
                # this is a numpy array
                dtype = test_data.dtype
            else:
                dtype = type(test_data)

        # For python type, JSON / MsgPack codecs, otherwise
        # use VLenArray with specific numpy dtype
        if (
            np.issubdtype(dtype, str)
            or np.issubdtype(dtype, list)
            or np.issubdtype(dtype, tuple)
        ):
            object_codec = numcodecs.MsgPack()
        else:
            object_codec = numcodecs.VLenArray(dtype)

        dset = group.require_dataset(
            key,
            data.shape,
            object_codec=object_codec,
            **these_kwds,
        )
        return dset

    @staticmethod
    def _store_data(data, dset, group, key, chunks, show_progressbar=True):
        # Tuple of dask arrays can also be passed, in which case the task graphs
        # are merged and the data is written in a single `da.store` call.
        # This is useful when saving a ragged array, where we need to write
        # the data and the shape at the same time as the ragged array must have
        # only one dimension.
        if isinstance(data, tuple):
            data = list(data)
        elif not isinstance(data, list):
            data = [
                data,
            ]
            dset = [
                dset,
            ]
        for i, (data_, dset_) in enumerate(zip(data, dset)):
            if is_dask_array(data_):
                if data_.chunks != dset_.chunks:
                    data[i] = data_.rechunk(dset_.chunks)
                # for performance reason, we write the data later, with all data
                # at the same time in a single `da.store` call
            else:
                dset_[:] = data_
        if is_dask_array(data[0]):
            with get_progress_bar_context_manager(show_progressbar)():
                import dask.array as da

                # lock=False is necessary with the distributed scheduler
                # da.store of tuple helps to merge task graphs and avoid computing twice
                da.store(data, dset, lock=False)


def _consolidate_metadata(store):
    """Consolidate zarr metadata, handling both zarr v2 and v3.

    Calls the appropriate consolidated metadata function for the installed
    zarr version. In zarr v2, it's ``zarr.convenience.consolidate_metadata``.
    In zarr v3+, it's ``zarr.consolidate_metadata``.
    """
    try:
        # zarr v3+
        from zarr import consolidate_metadata  # noqa: F811

        consolidate_metadata(store)
    except ImportError:
        # zarr v2
        zarr.convenience.consolidate_metadata(store)


def file_writer(
    filename,
    signal,
    chunks=None,
    compressor=None,
    close_file=True,
    write_dataset=True,
    store_type=None,
    show_progressbar=True,
    consolidate=True,
    **kwds,
):
    """
    Write data to HyperSpy's zarr format.

    Parameters
    ----------
    %s
    %s
    %s
    compressor : numcodecs.abc.Codec or None, default=None
        A compressor can be passed to the save function to compress the data
        efficiently, see `Numcodecs codec <https://numcodecs.readthedocs.io/en/stable>`_.
        If None, use a Blosc compressor.
    close_file : bool, default=True
        Close the file after writing. Only relevant for some zarr storages
        (:py:class:`zarr.storage.ZipStore`, :py:class:`zarr.storage.DBMStore`)
        requiring store to flush data to disk. If ``False``, doesn't close the
        file after writing. The file should not be closed if the data needs to be
        accessed lazily after saving.
    write_dataset : bool, default=True
        If ``False``, doesn't write the dataset when writing the file. This can
        be useful to overwrite signal attributes only (for example ``axes_manager``)
        without having to write the whole dataset, which can take time.
    store_type : str, "local" or "zip" or None
        If "local", uses a :class:`zarr.storage.NestedDirectoryStore`
        to save the file in a local directory. If "zip", uses a
        :class:`zarr.storage.ZipStore` to save the file in a zip archive.
        If ``None``, the default store is used (:class:`~zarr.storage.NestedDirectoryStore`)
        is used. Specifying this parameter is incompatible with passing an instance of
        a zarr store to the ``filename`` parameter. Default is None.
    %s
    consolidate : bool, default=True
        If ``True``, call :func:`zarr.convenience.consolidate_metadata`
        to bundle all metadata into a single ``.zmetadata`` file. This
        significantly reduces the number of HTTP requests needed when
        opening the file on remote or cloud storage (S3, GCS, etc.),
        improving performance for cloud workflows. When saving to local
        filesystems, the overhead is negligible. If the store does not
        support writes or the consolidation fails for any reason, a
        warning is logged and the file remains valid — metadata is
        simply read from individual files on open.
    **kwds
        The keyword arguments are passed to the
        :py:meth:`zarr.hierarchy.Group.require_dataset` function.

    Examples
    --------
    >>> from numcodecs import Blosc
    >>> compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE) # Used by default
    >>> file_writer("test.zspy", signal_dict, compressor=compressor) # will save with Blosc compression

    Using a :class:`zarr.storage.ZipStore` store:

    >>> file_writer("test.zspy", signal_dict, store_type="zip")
    """
    if compressor is None:
        compressor = numcodecs.Blosc(
            cname="zstd", clevel=1, shuffle=numcodecs.Blosc.SHUFFLE
        )
    if not isinstance(write_dataset, bool):
        raise ValueError("`write_dataset` argument must a boolean.")

    if isinstance(filename, MutableMapping):
        # a store is passed for the filename
        store = filename
        if store_type is not None:
            raise ValueError(
                "The `store_type` parameter must be None if a zarr "
                "store is passed to the `filename` parameter."
            )
    else:
        if store_type in ["local", None]:
            store = zarr.storage.NestedDirectoryStore(filename)
        elif store_type == "zip":
            store = zarr.storage.ZipStore(filename)
        else:
            raise ValueError(
                "The `store_type` argument must be one of 'local' or 'zip'."
            )

    mode = "w" if write_dataset else "a"

    _logger.debug(f"File mode: {mode}")
    _logger.debug(f"Zarr store: {store}")

    f = zarr.open_group(store=store, mode=mode)
    f.attrs["file_format"] = "ZSpy"
    f.attrs["file_format_version"] = version
    exps = f.require_group("Experiments")
    title = signal["metadata"]["General"]["title"]
    group_name = title if title else "__unnamed__"
    # / is a invalid character, see https://github.com/hyperspy/hyperspy/issues/942
    if "/" in group_name:
        group_name = group_name.replace("/", "-")
    expg = exps.require_group(group_name)

    writer = ZspyWriter(
        f,
        signal,
        expg,
        chunks=chunks,
        compressor=compressor,
        write_dataset=write_dataset,
        show_progressbar=show_progressbar,
        **kwds,
    )
    writer.write()

    if consolidate:
        try:
            _consolidate_metadata(store)
        except Exception as e:
            _logger.warning(
                f"Failed to consolidate metadata: {e}. "
                "The file is still valid, but opening it on remote/cloud "
                "storage will require more HTTP requests."
            )

    if isinstance(store, (zarr.ZipStore, zarr.DBMStore, zarr.LMDBStore)):
        if close_file:
            store.close()
        else:
            store.flush()


file_writer.__doc__ %= (
    FILENAME_DOC.replace("read", "write to"),
    SIGNAL_DOC,
    CHUNKS_DOC,
    SHOW_PROGRESSBAR_DOC,
)


class _UnsafeCodecError(ValueError):
    """Raised when an array selects the canonical Pickle codec."""

    def __init__(self, array_path, codec_id):
        self.array_path = array_path
        self.codec_id = codec_id
        display = array_path or "<root>"
        super().__init__(
            f"Unsafe ZSPY codec {codec_id!r} is not allowed for array "
            f"{display!r}; refusing to open the file."
        )


def _normalize_zspy_store(store, mode, storage_options=None):
    """Normalize a Zarr store exactly once for validation and opening."""
    if getattr(store, "_store_version", 2) != 2:
        raise ValueError("The ZSPY reader only supports Zarr format 2 stores.")
    normalize = getattr(zarr.storage, "normalize_store_arg", None)
    if normalize is not None:
        store = normalize(store, storage_options=storage_options, mode=mode)
    else:  # Zarr 2.10.x
        store = zarr.creation.normalize_store_arg(
            store, clobber=True, storage_options=storage_options, mode=mode
        )
    if getattr(store, "_store_version", 2) != 2:
        raise ValueError("The ZSPY reader only supports Zarr format 2 stores.")
    return store


def _logical_array_path(metadata_key):
    if metadata_key == ".zarray":
        return ""
    if metadata_key.endswith("/.zarray"):
        return metadata_key[: -len("/.zarray")]
    return None


def _parse_raw_metadata(raw):
    """Parse byte-like or text metadata without Zarr, NumPy, or codecs."""
    if isinstance(raw, Mapping):
        return raw
    text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
    return json.loads(text)


def _malformed_metadata_error(array_path, reason):
    display = array_path or "<root>"
    return ValueError(f"Malformed ZSPY array metadata for {display!r}: {reason}")


def _raise_malformed_metadata(array_path, reason):
    raise _malformed_metadata_error(array_path, reason)


def _has_n5_chunk(keys, array_path):
    prefix = f"{array_path}/" if array_path else ""
    for key in keys:
        if not key.startswith(prefix):
            continue
        relative = key[len(prefix) :]
        if relative and all(part.isdigit() for part in relative.split("/")):
            return True
    return False


def _iter_candidate_arrays(keys, store, selected_path):
    """Read the selected raw metadata of every candidate array."""
    keys = [str(key) for key in keys]
    n5_store = isinstance(store, (zarr.N5Store, zarr.N5FSStore))
    candidates = []

    def selected(array_path):
        return not selected_path or (
            array_path == selected_path or array_path.startswith(selected_path + "/")
        )

    for key in keys:
        array_path = _logical_array_path(key)
        if array_path is not None:
            origin = "n5_zarray" if n5_store else "zarray"
        elif n5_store and (
            key == "attributes.json" or key.endswith("/attributes.json")
        ):
            array_path = (
                key[: -len("/attributes.json")] if key != "attributes.json" else ""
            )
            origin = "n5"
        else:
            continue
        if not selected(array_path):
            continue
        item = {
            "path": array_path,
            "key": key,
            "origin": origin,
            "raw": None,
            "problem": None,
            "group": False,
        }
        try:
            if origin == "n5_zarray":
                # Bypass the N5 adapter's translation to inspect raw bytes.
                if isinstance(store, zarr.N5FSStore):
                    raw = store.map[key]
                else:
                    raw = zarr.storage.NestedDirectoryStore.__getitem__(store, key)
            else:
                raw = store[key]
            # Always parse with the stdlib boundary: handing bytes to
            # Zarr/numcodecs would route them through NumPy and codecs before
            # the codec decision this preflight makes.
            item["raw"] = _parse_raw_metadata(raw)
        except Exception as exc:
            reason = (
                "could not determine N5 array metadata"
                if origin == "n5"
                else "could not read metadata"
            )
            error = _malformed_metadata_error(array_path, reason)
            error.__cause__ = exc
            error.__suppress_context__ = True
            item["problem"] = error
        if (
            origin == "n5"
            and isinstance(item["raw"], Mapping)
            and "dimensions" not in item["raw"]
        ):
            item["group"] = not _has_n5_chunk(keys, array_path)
        candidates.append(item)
    return sorted(candidates, key=lambda item: (item["path"], item["key"]))


def _n5_compressor_config(compression, array_path):
    """Return the codec config Zarr will build for an N5 compression mapping."""
    if not isinstance(compression, Mapping):
        _raise_malformed_metadata(
            array_path, "could not determine N5 array metadata: invalid 'compression'"
        )
    if not isinstance(compression.get("type"), str):
        _raise_malformed_metadata(
            array_path,
            "could not determine N5 array metadata: invalid compression 'type'",
        )
    # Zarr's converter applies a supplied "id" last, so a mapping such as
    # {"type": "unused", "id": "pickle"} surfaces as id "pickle" here instead
    # of being normalized away into a harmless-looking type name.
    return compressor_config_to_zarr(compression)


def _inspect_codec(codec, array_path):
    """Inspect a codec mapping's own id, recursing only N5 wrapper config."""
    if not isinstance(codec, Mapping):
        _raise_malformed_metadata(array_path, "codec entry is not a mapping")
    codec_id = codec.get("id")
    if not isinstance(codec_id, str):
        _raise_malformed_metadata(array_path, "codec entry has no string 'id'")
    if codec_id == "pickle":
        raise _UnsafeCodecError(array_path, codec_id)
    if codec_id == "n5_wrapper":
        nested = codec.get("compressor_config")
        if nested is None:
            return
        if isinstance(nested, Mapping):
            _inspect_codec(nested, array_path)
            return
        _raise_malformed_metadata(
            array_path, "n5_wrapper compressor_config is not a mapping"
        )


def _scan_codec_entry(codec, array_path, problems):
    try:
        _inspect_codec(codec, array_path)
    except _UnsafeCodecError:
        raise
    except Exception as exc:
        problems.append(exc)


def _scan_candidate_codecs(item):
    """Inspect every codec entry of one candidate and collect problems."""
    problems = []
    raw = item["raw"]
    if not isinstance(raw, Mapping):
        return problems
    if item["origin"] == "n5":
        if item.get("group"):
            # A group's attributes.json carries saved metadata, not array
            # codecs: Zarr builds codecs from array attributes only, so
            # scanning group attributes would reject valid metadata.
            return problems
        if "compression" not in raw:
            return problems
        try:
            config = _n5_compressor_config(raw["compression"], item["path"])
        except _UnsafeCodecError:
            raise
        except Exception as exc:
            error = _malformed_metadata_error(
                item["path"], "could not determine N5 array metadata"
            )
            error.__cause__ = exc
            error.__suppress_context__ = True
            problems.append(error)
            return problems
        if config is not None:
            _scan_codec_entry(config, item["path"], problems)
        return problems
    compressor = raw.get("compressor")
    if compressor is not None:
        _scan_codec_entry(compressor, item["path"], problems)
    filters = raw.get("filters")
    if isinstance(filters, (list, tuple)):
        for codec in filters:
            _scan_codec_entry(codec, item["path"], problems)
    return problems


def _validate_shape_values(value, field, positive, array_path):
    if not isinstance(value, (list, tuple)) or any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or (item <= 0 if positive else item < 0)
        for item in value
    ):
        _raise_malformed_metadata(
            array_path, f"invalid {field!r}: expected integer list"
        )


def _inspect_array_metadata(metadata, array_path):
    """Validate the Zarr array document before Zarr constructs anything."""
    if not isinstance(metadata, Mapping):
        _raise_malformed_metadata(array_path, "metadata is not a mapping")
    required = (
        "zarr_format",
        "shape",
        "chunks",
        "dtype",
        "compressor",
        "fill_value",
        "order",
        "filters",
    )
    for field in required:
        if field not in metadata:
            _raise_malformed_metadata(array_path, f"missing {field!r}")
    if metadata["zarr_format"] != 2:
        _raise_malformed_metadata(
            array_path,
            f"unsupported zarr_format {metadata['zarr_format']!r}",
        )
    _validate_shape_values(metadata["shape"], "shape", False, array_path)
    _validate_shape_values(metadata["chunks"], "chunks", True, array_path)
    if len(metadata["shape"]) != len(metadata["chunks"]):
        _raise_malformed_metadata(
            array_path, "invalid 'chunks': rank differs from 'shape'"
        )
    if metadata["order"] not in ("C", "F"):
        _raise_malformed_metadata(array_path, "invalid 'order': expected 'C' or 'F'")
    if metadata.get("dimension_separator", ".") not in (".", "/"):
        _raise_malformed_metadata(
            array_path, "invalid 'dimension_separator': expected '.' or '/'"
        )
    if metadata["filters"] is not None and not isinstance(
        metadata["filters"], (list, tuple)
    ):
        _raise_malformed_metadata(array_path, "filters is not a list")


def _decode_array_metadata(metadata, array_path):
    """Decode dtype and fill with Zarr, keeping the array path on failure."""
    try:
        zarr.meta.decode_array_metadata(json.dumps(metadata))
    except Exception as exc:
        raise ValueError(
            "Malformed ZSPY array metadata for "
            f"{array_path or '<root>'!r}: could not decode array metadata"
        ) from exc


def _inspect_n5_metadata(metadata, array_path):
    """Validate the N5 array document before Zarr constructs anything."""
    if not isinstance(metadata, Mapping):
        _raise_malformed_metadata(array_path, "metadata is not a mapping")
    for field in ("dimensions", "blockSize", "dataType", "compression"):
        if field not in metadata:
            _raise_malformed_metadata(
                array_path,
                f"could not determine N5 array metadata: missing {field!r}",
            )
    for field in ("dimensions", "blockSize"):
        if not isinstance(metadata[field], (list, tuple)):
            _raise_malformed_metadata(
                array_path,
                f"could not determine N5 array metadata: invalid {field!r}",
            )
    _validate_shape_values(metadata["dimensions"], "dimensions", False, array_path)
    _validate_shape_values(metadata["blockSize"], "blockSize", True, array_path)
    if len(metadata["dimensions"]) != len(metadata["blockSize"]):
        _raise_malformed_metadata(
            array_path, "invalid 'blockSize': rank differs from 'dimensions'"
        )
    if not isinstance(metadata["dataType"], str):
        _raise_malformed_metadata(
            array_path, "could not determine N5 array metadata: invalid 'dataType'"
        )
    try:
        np.dtype(metadata["dataType"])
    except Exception as exc:
        raise _malformed_metadata_error(
            array_path, "could not determine N5 array metadata: invalid 'dataType'"
        ) from exc
    if not isinstance(metadata["compression"], Mapping):
        _raise_malformed_metadata(
            array_path,
            "could not determine N5 array metadata: invalid 'compression'",
        )


def _validate_zspy_codecs(store, path=None):
    """Reject Pickle codecs using the selected raw metadata only."""
    selected_path = zarr.util.normalize_storage_path(path)
    try:
        keys = [str(key) for key in store.keys()]
    except Exception as exc:
        raise ValueError(
            "Malformed ZSPY array metadata for '<store>': "
            "could not enumerate metadata keys"
        ) from exc
    if selected_path:
        # Nothing outside the selected subtree can influence the decision,
        # so a small selection does not need to hold every chunk key of a
        # large store in memory.
        prefix = selected_path + "/"
        keys = [key for key in keys if key.startswith(prefix)]
    candidates = _iter_candidate_arrays(keys, store, selected_path)
    # A hostile codec anywhere in the selection wins over an earlier array's
    # unreadable or malformed metadata, so scan every candidate first.
    for item in candidates:
        item["codec_problems"] = _scan_candidate_codecs(item)
    for item in candidates:
        if item["problem"] is not None:
            raise item["problem"]
        if item["codec_problems"]:
            raise item["codec_problems"][0]
        if item["origin"] == "n5":
            if not item.get("group"):
                _inspect_n5_metadata(item["raw"], item["path"])
        else:
            _inspect_array_metadata(item["raw"], item["path"])
            _decode_array_metadata(item["raw"], item["path"])


def file_reader(filename, lazy=False, **kwds):
    """
    Read data from zspy files saved with the HyperSpy zarr format
    specification.

    Parameters
    ----------
    %s
    %s
    **kwds : dict, optional
        Pass keyword arguments to the :py:func:`zarr.convenience.open` function.

    %s
    """
    # check that this is a not zarr store before checking if it is a zip file
    owns_zip_store = False
    if not isinstance(filename, MutableMapping) and zipfile.is_zipfile(filename):
        filename = zarr.storage.ZipStore(filename, mode="r")
        owns_zip_store = True

    completed = False
    try:
        if isinstance(filename, zarr.storage.ZipStore) and "mode" in kwds.keys():
            _logger.warning(
                "Specifying `mode` when opening a zspy file with a ZipStore is not supported."
            )

        # if lazy is True, we need to open the file in read/write mode
        # to be able to write later
        kwds.setdefault("mode", "r+" if lazy else "r")

        storage_options = kwds.pop("storage_options", None)
        store = _normalize_zspy_store(filename, kwds["mode"], storage_options)
        _validate_zspy_codecs(store, kwds.get("path"))
        f = zarr.open(store, **kwds)
        to_return = ZspyReader(f).read(lazy=lazy)
        completed = True
    except Exception:
        _logger.error(
            "The file can't be read. It may be possible that the zspy file is "
            "saved with a different store than a zarr directory store. Try "
            "passing a different zarr store instead of the file name."
        )
        raise
    finally:
        # A store this reader opened must not leak when the preflight rejects
        # the file or the read fails; a lazy read keeps it usable.
        if owns_zip_store and not (completed and lazy):
            filename.close()

    return to_return


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)
