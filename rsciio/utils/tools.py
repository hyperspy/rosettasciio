# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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
import xml.etree.ElementTree as ET
from pathlib import Path
import os
from collections import OrderedDict
from contextlib import contextmanager
import importlib

import dask.array as da
import numpy as np
from box import Box
from pint import UnitRegistry

_UREG = UnitRegistry()


_logger = logging.getLogger(__name__)


@contextmanager
def dummy_context_manager(*args, **kwargs):
    yield


def seek_read(file, dtype, pos):
    file.seek(pos)
    data = np.squeeze(np.fromfile(file, dtype, count=1))[()]
    if type(data) == np.uint32 or np.int32:
        data = int(data)
    if type(data) == np.bool_:
        data =bool(data)
    return data


def read_binary_metadata(file, mapping_dict):
    """ This function reads binary metadata in a batch like process.
    The mapping dict is passed as dictionary with a "key":[data,location]"
    format.
    """
    try:
        with open(file, mode="rb") as f:
            metadata = {
                m: seek_read(f, mapping_dict[m][0], mapping_dict[m][1])
                for m in mapping_dict
            }
        return metadata
    except FileNotFoundError:
        _logger.warning(
            msg="File " + file + " not found. Please"
            "move it to the same directory to read"
            " the metadata "
        )
        return None


def xml_branch(child):
    new_dict = {}
    if len(child) != 0:
        for c in child:
            new_dict[c.tag] = xml_branch(c)
        return new_dict
    else:
        new_dict = child.attrib
        for key in new_dict:
            try:
                new_dict[key] = float(new_dict[key])
            except ValueError:
                pass
        return new_dict


def parse_xml(file):
    try:
        tree = ET.parse(file)
        xml_dict = xml_branch(tree.getroot())
    except FileNotFoundError:
        _logger.warning(
            msg="File " + file + " not found. Please"
            "move it to the same directory to read"
            " the metadata "
        )
        return None
    return xml_dict


def get_chunk_slice(shape,
                    chunks="auto",
                    block_size_limit=None,
                    dtype=None,
                    ):
    chunks = da.core.normalize_chunks(
        chunks=chunks, shape=shape, limit=block_size_limit, dtype=dtype
    )
    chunks_shape = tuple([len(c) for c in zip(chunks, shape)])
    slices = np.empty(shape=chunks_shape, dtype=object)
    for ind in np.ndindex(chunks_shape):
        current_chunk = [chunk[i] for i, chunk in zip(ind, chunks)]
        starts = [int(np.sum(chunk[:i])) for i, chunk in zip(ind, chunks)]
        stops = [s+c for s, c in zip(starts, current_chunk)]
        slices[ind] = tuple([slice(start, stop)for start, stop in zip(starts, stops)])
    return da.from_array(slices, chunks=1), chunks


def slice_memmap(slice, file, dtypes, key=None, **kwargs):
    slice = np.squeeze(slice)[()]
    print(slice)

    data = np.memmap(file, dtypes, **kwargs)
    if key is not None:
        data = data[key]
    return data[slice]


def memmap_distributed(file, dtype,
                       offset=0, shape=None,
                       order="C", chunks="auto", block_size_limit=None,
                       key="Array"):
    """ Drop in replacement for `np.memmap` allowing for distributed loading of data.
    This always loads the data using dask which can be beneficial in many cases, but
    may not be ideal in others.

    The chunks and block_size_limit are for describing an ideal chunk shape and size
    as defined using the `da.core.normalize_chunks` function.

    Notes
    -----
    Currently `da.map_blocks` does not allow for multiple outputs.  As a result one "Key" is
    allowed which can be used when the give dtpye has a keyed input.  For example:
    dtype = (("Array", int, (128,128)),
             ("sec", "<u4"),("ms", "<u2"),("mis", "<u2"),("empty", bytes, empty),)
    """
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)
    if dtype.names is not None:
        array_dtype = dtype[key].base
        sub_array_shape = dtype[key].shape
    else:
        array_dtype = dtype.base
        sub_array_shape = dtype.shape
    if shape is None:
        unit_size = np.dtype(dtype).itemsize
        shape = int(os.path.getsize(file)/unit_size)
    if not isinstance(shape, tuple):
        shape = (shape,)
    full_shape = shape + sub_array_shape

    # Separates slices into appropriately sized chunks.
    chunked_slices, data_chunks = get_chunk_slice(shape=full_shape,
                                     chunks=chunks,
                                     block_size_limit=block_size_limit,
                                     dtype=array_dtype)

    data = da.map_blocks(slice_memmap,
                         chunked_slices,
                         file=file,
                         dtype=array_dtype,
                         shape=shape,
                         order=order,
                         mode="r",
                         dtypes=dtype,
                         offset=offset,
                         chunks=data_chunks,
                         key=key)
    return data


def get_chunk_index(shape,
                    signal_axes=(-1, -2),
                    chunks="auto",
                    block_size_limit=None,
                    dtype=None,
                    ):
    """Returns a chunk index for distributed chunking of some dataset. This is
    particularly useful with np.memmap and creating arrays from binary data
    which work with dask.distributed.  Note that for almost all cases `get_chunk_slice`
    is preferred. This is particularly useful for when data is nested or otherwise
    non-uniformly distributed

    Parameters
    ----------
    shape: tuple
        The shape of the resulting array. This is used to determine the chunk
        size for some dataset. Based on the underlying datatype this automatically
        creates chunks of around ~100 mb.
    signal_axes: tuple
        The signal axes are the axes t
    """
    nav_shape = np.delete(np.array(shape), signal_axes)
    num_frames = np.prod(nav_shape)
    indexes = da.arange(num_frames)
    indexes = da.reshape(indexes, nav_shape)
    chunks = da.core.normalize_chunks(
        chunks=chunks, shape=shape, limit=block_size_limit, dtype=dtype
    )

    nav_chunks = tuple(np.delete(np.array(chunks, dtype=object), signal_axes))
    indexes = da.rechunk(indexes, chunks=nav_chunks)
    return indexes


def dump_dictionary(
    file, dic, string="root", node_separator=".", value_separator=" = "
):
    for key in list(dic.keys()):
        if isinstance(dic[key], dict):
            dump_dictionary(file, dic[key], string + node_separator + key)
        else:
            file.write(
                string + node_separator + key + value_separator + str(dic[key]) + "\n"
            )


def append2pathname(filename, to_append):
    """Append a string to a path name

    Parameters
    ----------
    filename : str
    to_append : str

    """
    p = Path(filename)
    return Path(p.parent, p.stem + to_append, p.suffix)


def incremental_filename(filename, i=1):
    """If a file with the same file name exists, returns a new filename that
    does not exists.

    The new file name is created by appending `-n` (where `n` is an integer)
    to path name

    Parameters
    ----------
    filename : str
    i : int
       The number to be appended.
    """
    filename = Path(filename)

    if filename.is_file():
        new_filename = append2pathname(filename, "-{i}")
        if new_filename.is_file():
            return incremental_filename(filename, i + 1)
        else:
            return new_filename
    else:
        return filename


def ensure_directory(path):
    """Check if the path exists and if it does not, creates the directory."""
    # If it's a file path, try the parent directory instead
    p = Path(path)
    p = p.parent if p.is_file() else p

    try:
        p.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        _logger.debug(f"Directory {p} already exists. Doing nothing.")


def overwrite(fname):
    """If file exists 'fname', ask for overwriting and return True or False,
    else return True.

    Parameters
    ----------
    fname : str or pathlib.Path
        File to check for overwriting.

    Returns
    -------
    bool :
        Whether to overwrite file.

    """
    if Path(fname).is_file() or (
        Path(fname).is_dir() and os.path.splitext(fname)[1] == ".zspy"
    ):
        message = f"Overwrite '{fname}' (y/n)?\n"
        try:
            answer = input(message)
            answer = answer.lower()
            while (answer != "y") and (answer != "n"):
                print("Please answer y or n.")
                answer = input(message)
            if answer.lower() == "y":
                return True
            elif answer.lower() == "n":
                return False
        except:
            # We are running in the IPython notebook that does not
            # support raw_input
            _logger.info(
                "Your terminal does not support raw input. "
                "Not overwriting. "
                "To overwrite the file use `overwrite=True`"
            )
            return False
    else:
        return True


def xml2dtb(et, dictree):
    if et.text:
        dictree.set_item(et.tag, et.text)
        return
    else:
        dictree.add_node(et.tag)
        if et.attrib:
            dictree[et.tag].merge_update(et.attrib)
        for child in et:
            xml2dtb(child, dictree[et.tag])


class DTBox(Box):
    def add_node(self, path):
        keys = path.split(".")
        for key in keys:
            if self.get(key) is None:
                self[key] = {}
            self = self[key]

    def set_item(self, path, value):
        if self.get(path) is None:
            self.add_node(path)
        self[path] = value

    def has_item(self, path):
        return self.get(path) is not None


def convert_xml_to_dict(xml_object):
    if isinstance(xml_object, str):
        xml_object = ET.fromstring(xml_object)
    op = DTBox(box_dots=True)
    xml2dtb(xml_object, op)
    return op


def sarray2dict(sarray, dictionary=None):
    """Converts a struct array to an ordered dictionary

    Parameters
    ----------
    sarray: struct array
    dictionary: None or dict
        If dictionary is not None the content of sarray will be appended to the
        given dictonary

    Returns
    -------
    Ordered dictionary

    """
    if dictionary is None:
        dictionary = OrderedDict()
    for name in sarray.dtype.names:
        dictionary[name] = sarray[name][0] if len(sarray[name]) == 1 else sarray[name]
    return dictionary


def dict2sarray(dictionary, sarray=None, dtype=None):
    """Populates a struct array from a dictionary

    Parameters
    ----------
    dictionary: dict
    sarray: struct array or None
        Either sarray or dtype must be given. If sarray is given, it is
        populated from the dictionary.
    dtype: None, numpy dtype or dtype list
        If sarray is None, dtype must be given. If so, a new struct array
        is created according to the dtype, which is then populated.

    Returns
    -------
    Structure array

    """
    if sarray is None:
        if dtype is None:
            raise ValueError("Either sarray or dtype need to be specified.")
        sarray = np.zeros((1,), dtype=dtype)
    for name in set(sarray.dtype.names).intersection(set(dictionary.keys())):
        if len(sarray[name]) == 1:
            sarray[name][0] = dictionary[name]
        else:
            sarray[name] = dictionary[name]
    return sarray


def convert_units(value, units, to_units):
    return (value * _UREG(units)).to(to_units).magnitude


def get_object_package_info(obj):
    """Get info about object package

    Returns
    -------
    dic: dict
        Dictionary containing ``package`` and ``package_version`` (if available)
    """
    dic = {}
    # Note that the following can be "__main__" if the component was user
    # defined
    dic["package"] = obj.__module__.split(".")[0]
    if dic["package"] != "__main__":
        try:
            dic["package_version"] = importlib.import_module(dic["package"]).__version__
        except AttributeError:
            dic["package_version"] = ""
            _logger.warning(
                "The package {package} does not set its version in "
                + "{package}.__version__. Please report this issue to the "
                + "{package} developers.".format(package=dic["package"])
            )
    else:
        dic["package_version"] = ""
    return dic


def ensure_unicode(stuff, encoding="utf8", encoding2="latin-1"):
    if not isinstance(stuff, (bytes, np.string_)):
        return stuff
    else:
        string = stuff
    try:
        string = string.decode(encoding)
    except BaseException:
        string = string.decode(encoding2, errors="ignore")
    return string


def get_file_handle(data, warn=True):
    """Return file handle of a dask array when possible; currently only hdf5 file are
    supported.
    """
    arrkey = None
    for key in data.dask.keys():
        # The if statement with both "array-original" and "original-array"
        # is due to dask changing the name of this key. After dask-2022.1.1
        # the key is "original-array", before it is "array-original"
        if ("array-original" in key) or ("original-array" in key):
            arrkey = key
            break
    if arrkey:
        try:
            return data.dask[arrkey].file
        except (AttributeError, ValueError):
            if warn:
                _logger.warning(
                    "Failed to retrieve file handle, either "
                    "the file is already closed or it is not "
                    "an hdf5 file."
                )
