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
from pathlib import Path

import dask.array as da
import h5py
from dask.diagnostics import ProgressBar
from packaging.version import Version

from rsciio._docstrings import (
    CHUNKS_DOC,
    COMPRESSION_HDF5_DOC,
    COMPRESSION_HDF5_NOTES_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    RETURNS_DOC,
    SHOW_PROGRESSBAR_DOC,
    SIGNAL_DOC,
)
from rsciio._hierarchical import HierarchicalReader, HierarchicalWriter, version
from rsciio.utils.tools import dummy_context_manager, get_file_handle

_logger = logging.getLogger(__name__)

not_valid_format = "The file is not a valid HyperSpy hdf5 file"

current_file_version = None  # Format version of the file being read
default_version = Version(version)


class HyperspyReader(HierarchicalReader):
    _file_type = "hspy"
    _is_hdf5 = True

    def __init__(self, file):
        super().__init__(file)
        self.Dataset = h5py.Dataset
        self.Group = h5py.Group


class HyperspyWriter(HierarchicalWriter):
    """
    An object used to simplify and organize the process for
    writing a hyperspy signal.  (.hspy format)
    """

    target_size = 1e6
    _unicode_kwds = {"dtype": h5py.string_dtype()}
    _is_hdf5 = True

    def __init__(self, file, signal, expg, **kwds):
        super().__init__(file, signal, expg, **kwds)
        self.Dataset = h5py.Dataset
        self.Group = h5py.Group

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
            if isinstance(data_, da.Array):
                if data_.chunks != dset_.chunks:
                    data[i] = data_.rechunk(dset_.chunks)
                if data_.ndim == 1 and data_.dtype == object:
                    # https://github.com/hyperspy/rosettasciio/issues/198
                    raise ValueError(
                        "Saving a 1-D ragged dask array to hspy is not supported yet. "
                        "Please use the .zspy extension."
                    )
                # for performance reason, we write the data later, with all data
                # at the same time in a single `da.store` call
            # "write_direct" doesn't play well with empty array
            elif data_.flags.c_contiguous and data_.shape != (0,):
                dset_.write_direct(data_)
            else:
                dset_[:] = data_
        if isinstance(data[0], da.Array):
            cm = ProgressBar if show_progressbar else dummy_context_manager
            with cm():
                # da.store of tuple helps to merge task graphs and avoid computing twice
                da.store(data, dset)

    @staticmethod
    def _get_object_dset(group, data, key, chunks, dtype=None, **kwds):
        """Creates a h5py dataset object for saving ragged data"""
        if chunks is None:  # pragma: no cover
            chunks = 1

        if dtype is None:
            test_data = data[data.ndim * (0,)]
            if isinstance(test_data, da.Array):
                test_data = test_data.compute()
            dtype = test_data.dtype

        dset = group.require_dataset(
            key, data.shape, dtype=h5py.vlen_dtype(dtype), chunks=chunks, **kwds
        )
        return dset


def file_reader(filename, lazy=False, **kwds):
    """
    Read data from hdf5-files saved with the HyperSpy hdf5-format
    specification (``.hspy``).

    Parameters
    ----------
    %s
    %s
    **kwds : dict, optional
        The keyword arguments are passed to :py:class:`h5py.File`.

    %s
    """
    try:
        # in case blosc compression is used
        # module needs to be imported to register plugin
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    mode = kwds.pop("mode", "r")
    f = h5py.File(filename, mode=mode, **kwds)

    reader = HyperspyReader(f)
    # Use try, except, finally to close file when an error is raised
    try:
        exp_dict_list = reader.read(lazy=lazy)
    except BaseException as err:
        raise err
    finally:
        if not lazy:
            f.close()

    return exp_dict_list


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)


def file_writer(
    filename,
    signal,
    chunks=None,
    compression="gzip",
    close_file=True,
    write_dataset=True,
    show_progressbar=True,
    **kwds,
):
    """
    Write data to HyperSpy's hdf5-format (``.hspy``).

    Parameters
    ----------
    %s
    %s
    %s
    %s
    close_file : bool, default=True
        Close the file after writing.  The file should not be closed if the data
        needs to be accessed lazily after saving.
    write_dataset : bool, default=True
        If True, write the dataset, otherwise, don't write it. Useful to
        overwrite attributes (for example ``axes_manager``) only without having
        to write the whole dataset.
    %s
    **kwds
        The keyword argument are passed to the
        :external+h5py:meth:`h5py.Group.require_dataset` function.

    Notes
    -----
    %s
    """
    if not isinstance(write_dataset, bool):
        raise ValueError("`write_dataset` argument has to be a boolean.")

    if "shuffle" not in kwds:
        # Use shuffle by default to improve compression
        kwds["shuffle"] = True

    folder = signal["tmp_parameters"].get("original_folder", "")
    fname = signal["tmp_parameters"].get("original_filename", "")
    ext = signal["tmp_parameters"].get("original_extension", "")
    original_path = Path(folder, f"{fname}.{ext}")

    f = None
    if signal["attributes"]["_lazy"] and Path(filename).absolute() == original_path:
        f = get_file_handle(signal["data"], warn=False)
        if f is not None and f.mode == "r":
            # when the file is read only, force to reopen it in writing mode
            raise OSError(
                "File opened in read only mode. To overwrite file "
                "with lazy signal, use `mode='a'` when loading the "
                "signal."
            )

    if f is None:
        # with "write_dataset=False", we need mode='a', otherwise the dataset
        # will be flushed with using 'w' mode
        mode = kwds.get("mode", "w" if write_dataset else "a")
        if mode != "a" and not write_dataset:
            raise ValueError("`mode='a'` is required to use " "`write_dataset=False`.")
        f = h5py.File(filename, mode=mode)

    f.attrs["file_format"] = "HyperSpy"
    f.attrs["file_format_version"] = version
    exps = f.require_group("Experiments")
    title = signal["metadata"]["General"]["title"]
    group_name = title if title else "__unnamed__"
    # / is a invalid character, see https://github.com/hyperspy/hyperspy/issues/942
    if "/" in group_name:
        group_name = group_name.replace("/", "-")
    expg = exps.require_group(group_name)

    writer = HyperspyWriter(
        f,
        signal,
        expg,
        chunks=chunks,
        compression=compression,
        write_dataset=write_dataset,
        show_progressbar=show_progressbar,
        **kwds,
    )
    # Use try, except, finally to close file when an error is raised
    try:
        writer.write()
    except BaseException as err:
        raise err
    finally:
        if close_file:
            f.close()


file_writer.__doc__ %= (
    FILENAME_DOC.replace("read", "write to"),
    SIGNAL_DOC,
    CHUNKS_DOC,
    COMPRESSION_HDF5_DOC,
    SHOW_PROGRESSBAR_DOC,
    COMPRESSION_HDF5_NOTES_DOC,
)


overwrite_dataset = HyperspyWriter.overwrite_dataset
