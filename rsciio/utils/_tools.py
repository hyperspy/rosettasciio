# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
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

import ast
import importlib
import logging
import struct
from typing import IO

import numpy as np
from packaging.version import Version

_logger = logging.getLogger(__name__)


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
    if not isinstance(stuff, (bytes, np.bytes_)):
        return stuff
    else:
        string = stuff
    try:
        string = string.decode(encoding)
    except Exception:
        string = string.decode(encoding2, errors="ignore")
    return string


def get_file_handle(data, warn=True):
    """
    Return file handle of a dask array when possible.
    Currently only hdf5 and tiff file are supported.

    Parameters
    ----------
    data : dask.array.Array
        The dask array from which the file handle
        will be retrieved.
    warn : bool
        Whether to warn or not when the file handle
        can't be retrieved. Default is True.

    Returns
    -------
    File handle or None
        The file handle of the file when possible.
    """
    arrkey_hdf5 = None
    arrkey_tifffile = None
    for key in data.dask.keys():
        # The if statement with both "array-original" and "original-array"
        # is due to dask changing the name of this key. After dask-2022.1.1
        # the key is "original-array", before it is "array-original"
        if ("array-original" in key) or ("original-array" in key):
            arrkey_hdf5 = key
            break
        # For tiff files, use _load_data key
        if "_load_data" in key:
            arrkey_tifffile = key
    if arrkey_hdf5:
        try:
            return data.dask[arrkey_hdf5].file
        except (AttributeError, ValueError):  # pragma: no cover
            if warn:
                _logger.warning(
                    "Failed to retrieve file handle, either the file is "
                    "already closed or it is not an hdf5 file."
                )
    if arrkey_tifffile:
        try:
            import dask

            # access the filehandle through the pages or series
            # interfaces of tifffile
            # this may be brittle and may need maintenance as
            # dask or tifffile evolve
            if Version(dask.__version__) >= Version("2025.4.0"):
                tiff_pages_series = data.dask[arrkey_tifffile].args[1]
            else:
                tiff_pages_series = data.dask[arrkey_tifffile][2][0]
            return tiff_pages_series.parent.filehandle._fh
        # can raise IndexError or AttributeError
        except BaseException:  # pragma: no cover
            if warn:
                _logger.warning(
                    "Failed to retrieve file handle, either the file is "
                    "already closed or it is not a supported tiff file."
                )

    return None


def jit_ifnumba(*decorator_args, **decorator_kwargs):
    try:
        import numba

        decorator_kwargs.setdefault("nopython", True)
        return numba.jit(*decorator_args, **decorator_kwargs)
    except ImportError:
        _logger.warning(
            "Falling back to slow pure python code, because `numba` is not installed."
        )

        def wrap(func):
            def wrapper_func(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper_func

        return wrap


def inspect_npy_bytes(file_: IO[bytes]) -> tuple[int, tuple[int, ...], str]:
    """
    Inspect a .npy byte stream to extract metadata such as data offset, shape, and dtype.

    .. warning::
       After calling this function, the ``file_`` stream position advanced to the
       start of the data.

    Parameters
    ----------
    file_ : File handle
        The .npy byte stream to inspect.

    Returns
    -------
    tuple
        A tuple containing:

        - ``offset`` (int): The byte offset where the data starts.
        - ``shape`` (tuple): The shape of the array stored in the file.
        - ``dtype`` (str): The data type of the array elements.

    Examples
    --------
    >>> with open('example.npy', 'rb') as f:
    ...     offset, shape, dtype = inspect_npy_bytes(f)
    """

    # Read magic string and version
    _ = file_.read(6)
    major, _ = struct.unpack("BB", file_.read(2))

    encoding = "latin1"  # Default encoding for .npy files version 1.0 and 2.0
    # Read header length
    if major == 1:
        header_len_size = 2
        header_len = struct.unpack("<H", file_.read(header_len_size))[0]
    elif major in (2, 3):
        header_len_size = 4
        header_len = struct.unpack("<I", file_.read(header_len_size))[0]
        if major == 3:
            encoding = "utf8"  # Version 3.0 uses UTF-8 encoding
    else:  # pragma: no cover
        raise ValueError("Unsupported .npy version")

    header_offset = 6 + 2 + header_len_size

    # Read and parse header
    header = file_.read(header_len).decode(encoding)
    header_dict = ast.literal_eval(header)

    # Extract metadata
    shape = header_dict["shape"]
    dtype = header_dict["descr"]
    offset = header_offset + header_len

    return offset, shape, dtype
