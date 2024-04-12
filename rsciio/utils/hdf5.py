"""HDF5 file inspection."""

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
#

import json
import pprint

import h5py

from rsciio._docstrings import FILENAME_DOC, LAZY_DOC
from rsciio.nexus._api import (
    _check_search_keys,
    _find_data,
    _find_search_keys_in_dict,
    _load_metadata,
)


def read_metadata_from_file(
    filename, lazy=False, metadata_key=None, verbose=False, skip_array_metadata=False
):
    """
    Read the metadata from a NeXus or ``.hdf`` file.

    This method iterates through the hdf5 file and returns a dictionary of
    the entries.
    This is a convenience method to inspect a file for a value
    rather than loading the file as a signal.

    Parameters
    ----------
    %s
    %s
    metadata_key : None, str, list of str, default=None
        None will return all datasets found including linked data.
        Providing a string or list of strings will only return items
        which contain the string(s).
        For example, search_keys = ["instrument","Fe"] will return
        hdf entries with "instrument" or "Fe" in their hdf path.
    verbose : bool, default=False
        Pretty print the results to screen.
    skip_array_metadata : bool, default=False
        Whether to skip loading array metadata. This is useful as a lot of
        large array may be present in the metadata and it is redundant with
        dataset itself.

    Returns
    -------
    dict
        Metadata dictionary.

    See Also
    --------
    rsciio.utils.hdf5.list_datasets_in_file : Convenience function to list
        datasets present in a file.
    """
    search_keys = _check_search_keys(metadata_key)
    fin = h5py.File(filename, "r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    stripped_metadata = _load_metadata(
        fin, lazy=lazy, skip_array_metadata=skip_array_metadata
    )
    stripped_metadata = _find_search_keys_in_dict(
        stripped_metadata, search_keys=search_keys
    )
    if verbose:
        pprint.pprint(stripped_metadata)

    fin.close()
    return stripped_metadata


read_metadata_from_file.__doc__ %= (FILENAME_DOC, LAZY_DOC)


def list_datasets_in_file(
    filename, dataset_key=None, hardlinks_only=False, verbose=True
):
    """
    Read from a NeXus or ``.hdf`` file and return a list of the dataset paths.

    This method is used to inspect the contents of an hdf5 file.
    The method iterates through group attributes and returns NXdata or
    hdf datasets of size >=2 if they're not already NXdata blocks
    and returns a list of the entries.
    This is a convenience method to inspect a file to list datasets
    present rather than loading all the datasets in the file as signals.

    Parameters
    ----------
    %s
    dataset_key : str, list of str, None , default=None
        If a str or list of strings is provided only return items whose
        path contain the strings.
        For example, dataset_key = ["instrument", "Fe"] will only return
        hdf entries with "instrument" or "Fe" somewhere in their hdf path.
    hardlinks_only : bool, default=False
        If true any links (soft or External) will be ignored when loading.
    verbose : bool, default=True
        Prints the results to screen.

    Returns
    -------
    list
        List of paths to datasets.

    See Also
    --------
    rsciio.utils.hdf5.read_metadata_from_file : Convenience function to read
        metadata present in a file.
    """
    search_keys = _check_search_keys(dataset_key)
    fin = h5py.File(filename, "r")
    # search for NXdata sets...
    # strip out the metadata (basically everything other than NXdata)
    nexus_data_paths, hdf_dataset_paths = _find_data(
        fin, search_keys=search_keys, hardlinks_only=hardlinks_only
    )
    if verbose:
        if nexus_data_paths:
            print("NXdata found")
            for nxd in nexus_data_paths:
                print(nxd)
        else:
            print("NXdata not found")
        if hdf_dataset_paths:
            print("Unique HDF datasets found")
            for hdfd in hdf_dataset_paths:
                print(hdfd, fin[hdfd].shape)
        else:
            print("No HDF datasets not found or data is captured by NXdata")
    fin.close()
    return nexus_data_paths, hdf_dataset_paths


list_datasets_in_file.__doc__ %= FILENAME_DOC


def _get_keys_from_group(group):
    # Return a list of ids of items contains in the group
    return list(group.keys())


def _parse_sub_data_group_metadata(sub_data_group):
    metadata_array = sub_data_group["Metadata"][:, 0].T
    mdata_string = metadata_array.tobytes().decode("utf-8")
    return json.loads(mdata_string.rstrip("\x00"))


def _parse_metadata(data_group, sub_group_key):
    return _parse_sub_data_group_metadata(data_group[sub_group_key])


__all__ = [
    "read_metadata_from_file",
    "list_datasets_in_file",
]


def __dir__():
    return sorted(__all__)
