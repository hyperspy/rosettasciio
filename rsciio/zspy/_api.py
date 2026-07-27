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

import logging
import zipfile
from collections.abc import MutableMapping

import numcodecs
import numpy as np
import zarr
from packaging.version import Version

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

ZARR_V3 = Version(zarr.__version__).major >= 3

# zarr 3 stores are no longer ``MutableMapping`` subclasses; they derive from
# ``zarr.abc.store.Store`` instead. Detecting "a store was passed instead of a
# path" has to accept both, otherwise a zarr 3 store is mistaken for a filename.
if ZARR_V3:  # pragma: no cover - depends on the installed zarr
    from zarr.abc.store import Store as _ZarrStore

    _STORE_TYPES = (MutableMapping, _ZarrStore)
else:  # pragma: no cover
    _STORE_TYPES = (MutableMapping,)

# Stores that buffer writes and need an explicit close/flush to land on disk.
# The v2-only ones simply don't exist under zarr 3 (`DBMStore`/`LMDBStore` were
# dropped rather than renamed), so build the tuple from whatever is available.
_BUFFERED_STORES = tuple(
    store
    for store in (
        getattr(zarr.storage, "ZipStore", None),
        getattr(zarr, "DBMStore", None),
        getattr(zarr, "LMDBStore", None),
    )
    if store is not None
)


def _as_v3_codec(compressor):
    """
    Translate a classic :mod:`numcodecs` codec to the zarr 3 codec wrapping it.

    zarr 3 rejects classic codecs (``TypeError: Expected a BytesBytesCodec``)
    and wants its own wrappers instead. Keeping the documented ``compressor=``
    parameter working means translating rather than making callers pass a
    different type depending on their zarr version.
    """
    if compressor is None or isinstance(compressor, (list, tuple)):
        return compressor
    try:
        # zarr >= 3.1.3 exposes the wrappers itself; `numcodecs.zarr3` is
        # deprecated and slated for removal.
        from zarr.codecs import numcodecs as zarr_numcodecs
    except ImportError:  # pragma: no cover - older zarr 3
        from numcodecs import zarr3 as zarr_numcodecs

    name = type(compressor).__name__
    wrapper = getattr(zarr_numcodecs, name, None)
    if wrapper is None:
        raise ValueError(
            f"The compressor {compressor!r} has no zarr 3 equivalent in "
            f"{zarr_numcodecs.__name__}. Pass a codec from that module "
            "directly, or use `compressor=None`."
        )
    config = {k: v for k, v in compressor.get_config().items() if k != "id"}
    return wrapper(**config)


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

    @classmethod
    def _require_dataset(cls, group, key, **kwds):
        if not ZARR_V3:
            return group.require_dataset(key, **kwds)
        # zarr 3 renamed `require_dataset` to `require_array`, dropped the
        # `exact` argument (shape and dtype are matched regardless), and takes
        # a list of its own codecs rather than a single classic one.
        kwds.pop("exact", None)
        compressor = kwds.pop("compressor", None)
        if isinstance(compressor, numcodecs.abc.Codec):
            kwds["compressors"] = [_as_v3_codec(compressor)]
        elif compressor is None:
            # zarr 2 spells "no compression" as `compressor=None`.
            kwds["compressors"] = None
        # Anything else (notably zarr 2's ``"default"`` sentinel) is left to
        # zarr to interpret, which for an unset ``compressors`` means its own
        # default pipeline.
        if kwds.get("chunks") is True:
            # h5py spells "pick chunks for me" as True, zarr 3 as "auto".
            kwds["chunks"] = "auto"
        return group.require_array(name=key, **kwds)

    @staticmethod
    def _get_object_dset(group, data, key, chunks, dtype=None, **kwds):
        """Creates a Zarr Array object for saving ragged data

        Forces the number of chunks span the array if not a dask array as
        calculating the chunks for a ragged array is not supported. See
        https://github.com/hyperspy/rosettasciio/issues/168 for more details.
        """
        if ZARR_V3:
            raise NotImplementedError(
                "Saving ragged arrays (such as variable-length markers or "
                "diffraction vectors) to zspy is not supported with zarr 3 "
                "yet. It relies on zarr 2 object codecs (`VLenArray`, "
                "`MsgPack`), which zarr 3 has no equivalent for. Install "
                "`zarr<3` to save this signal, or save it as `.hspy` instead."
            )
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


def file_writer(
    filename,
    signal,
    chunks=None,
    compressor=None,
    close_file=True,
    write_dataset=True,
    store_type=None,
    show_progressbar=True,
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

    if isinstance(filename, _STORE_TYPES):
        # a store is passed for the filename
        store = filename
        if store_type is not None:
            raise ValueError(
                "The `store_type` parameter must be None if a zarr "
                "store is passed to the `filename` parameter."
            )
    else:
        if store_type in ["local", None]:
            if ZARR_V3:
                # `NestedDirectoryStore` is v2-only. The mode is set by the
                # `zarr.open_group(store=store, mode=mode)` call below, and
                # `LocalStore` takes no `mode` argument.
                store = zarr.storage.LocalStore(filename)
            else:
                store = zarr.storage.NestedDirectoryStore(filename)
        elif store_type == "zip":
            if ZARR_V3:
                # zarr 3's ZipStore defaults to read-only, so the mode has to
                # be given up front rather than only to `open_group` below.
                store = zarr.storage.ZipStore(
                    filename, mode="w" if write_dataset else "a"
                )
            else:
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

    if _BUFFERED_STORES and isinstance(store, _BUFFERED_STORES):
        if close_file:
            store.close()
        elif hasattr(store, "flush"):
            # zarr 3 stores have no `flush`; the data is already written.
            store.flush()


file_writer.__doc__ %= (
    FILENAME_DOC.replace("read", "write to"),
    SIGNAL_DOC,
    CHUNKS_DOC,
    SHOW_PROGRESSBAR_DOC,
)


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
    if not isinstance(filename, _STORE_TYPES) and zipfile.is_zipfile(filename):
        filename = zarr.storage.ZipStore(filename, mode="r")

    if isinstance(filename, zarr.storage.ZipStore) and "mode" in kwds.keys():
        _logger.warning(
            "Specifying `mode` when opening a zspy file with a ZipStore is not supported."
        )

    # if lazy is True, we need to open the file in read/write mode
    # to be able to write later
    kwds.setdefault("mode", "r+" if lazy else "r")

    try:
        f = zarr.open(filename, **kwds)
    except Exception:
        _logger.error(
            "The file can't be read. It may be possible that the zspy file is "
            "saved with a different store than a zarr directory store. Try "
            "passing a different zarr store instead of the file name."
        )
        raise

    try:
        to_return = ZspyReader(f).read(lazy=lazy)
    except ValueError as err:
        if ZARR_V3 and "object_codec_id" in str(err):
            # zarr 3 can't resolve the v2 object-codec metadata that ragged
            # arrays (variable-length markers, diffraction vectors) were
            # written with. This is zarr's own v2-compatibility layer, so no
            # amount of fixing the write path here makes such a file readable.
            raise ValueError(
                "This file contains ragged arrays (such as variable-length "
                "markers or diffraction vectors) written with zarr 2 object "
                "codecs, which zarr 3 cannot read. Install `zarr<3` to read "
                "this file, and re-save it if you need it readable under "
                f"zarr 3. Original error: {err}"
            ) from err
        raise
    if not lazy and isinstance(filename, zarr.storage.ZipStore):
        # Close the file if not lazy
        filename.close()

    return to_return


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)
