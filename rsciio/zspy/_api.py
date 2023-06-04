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

import logging
from collections.abc import MutableMapping

import dask.array as da
import numcodecs
import zarr

from rsciio._docstrings import (
    FILENAME_DOC,
    LAZY_DOC,
    RETURNS_DOC,
    SIGNAL_DOC,
)
from rsciio._hierarchical import HierarchicalWriter, HierarchicalReader, version


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

    def __init__(self, file, signal, expg, **kwargs):
        super().__init__(file, signal, expg, **kwargs)
        self.Dataset = zarr.Array
        self.unicode_kwds = {"dtype": object, "object_codec": numcodecs.JSON()}
        self.ragged_kwds = {
            "dtype": object,
            "object_codec": numcodecs.VLenArray(int),
            "exact": True,
        }

    @staticmethod
    def _get_object_dset(group, data, key, chunks, **kwds):
        """Creates a Zarr Array object for saving ragged data"""
        these_kwds = kwds.copy()
        these_kwds.update(dict(dtype=object, exact=True, chunks=chunks))
        dset = group.require_dataset(
            key, data.shape, object_codec=numcodecs.VLenArray(int), **these_kwds
        )
        return dset

    @staticmethod
    def _store_data(data, dset, group, key, chunks):
        """Write data to zarr format."""
        if isinstance(data, da.Array):
            if data.chunks != dset.chunks:
                data = data.rechunk(dset.chunks)
            # lock=False is necessary with the distributed scheduler
            data.store(dset, lock=False)
        else:
            dset[:] = data


def file_writer(filename, signal, close_file=True, **kwds):
    """Writes data to HyperSpy's zarr format.

    Parameters
    ----------
    %s
    %s
    close_file : bool, default=True
        Close the file after writing. Only relevant for some zarr storages
        (:py:class:`zarr.storage.ZipStore`, :py:class:`zarr.storage.DBMStore`)
        requiring store to flush data to disk. If ``False``, doesn't close the
        file after writing. The file should not be closed if the data needs to be
        accessed lazily after saving.
    chunks : tuple of integer or None, default=None
        Define the chunking used for saving the dataset. If None, calculates
        chunks for the signal, with preferably at least one chunk per signal
        space.
    compressor : numcodecs compression, optional
        A compressor can be passed to the save function to compress the data
        efficiently, see `Numcodecs codec <https://numcodecs.readthedocs.io/en/stable>`_.
        The default is to use a Blosc compressor.
    write_dataset : bool, default=True
        If ``False``, doesn't write the dataset when writing the file. This can
        be useful to overwrite signal attributes only (for example ``axes_manager``)
        without having to write the whole dataset, which can take time.
    **kwds
        The keyword arguments are passed to the
        :py:meth:`zarr.hierarchy.Group.require_dataset` function.

    Examples
    --------
    >>> from numcodecs import Blosc
    >>> compressor=Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE) # Used by default
    >>> file_writer('test.zspy', s, compressor = compressor) # will save with Blosc compression
    """
    if "compressor" not in kwds:
        kwds["compressor"] = numcodecs.Blosc(
            cname="zstd", clevel=1, shuffle=numcodecs.Blosc.SHUFFLE
        )

    if isinstance(filename, MutableMapping):
        store = filename
    else:
        store = zarr.storage.NestedDirectoryStore(
            filename,
        )
    write_dataset = kwds.get("write_dataset", True)
    if not isinstance(write_dataset, bool):
        raise ValueError("`write_dataset` argument must a boolean.")
    mode = "w" if kwds.get("write_dataset", True) else "a"

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

    # Add record_by metadata for backward compatibility
    signal_dimension = len([axis for axis in signal["axes"] if not axis["navigate"]])
    smd = signal["metadata"]["Signal"]
    if signal_dimension == 1:
        smd["record_by"] = "spectrum"
    elif signal_dimension == 2:
        smd["record_by"] = "image"
    else:
        smd["record_by"] = ""

    try:
        writer = ZspyWriter(f, signal, expg, **kwds)
        writer.write()
    except Exception:
        raise
    finally:
        del smd["record_by"]

    if isinstance(store, (zarr.ZipStore, zarr.DBMStore, zarr.LMDBStore)):
        if close_file:
            store.close()
        else:
            store.flush()


file_writer.__doc__ %= (FILENAME_DOC.replace("read", "write to"), SIGNAL_DOC)


def file_reader(filename, lazy=False, **kwds):
    """Read data from zspy files saved with the HyperSpy zarr format
    specification.

    Parameters
    ----------
    %s
    %s
    **kwds: optional
        Pass keyword arguments to the :py:meth:`zarr.open` function.

    %s
    """
    mode = kwds.pop("mode", "r")
    try:
        f = zarr.open(filename, mode=mode, **kwds)
    except Exception:
        _logger.error(
            "The file can't be read. It may be possible that the zspy file is "
            "saved with a different store than a zarr directory store. Try "
            "passing a different zarr store instead of the file name."
        )
        raise

    reader = ZspyReader(f)

    return reader.read(lazy=lazy)


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)
