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


import dask.array as da
import numpy as np

from rsciio._docstrings import (
    CHUNKS_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    RETURNS_DOC,
    SIGNAL_DOC,
    UNSUPPORTED_METADATA_DOC,
)
from rsciio.utils.distributed import memmap_distributed
from rsciio.utils.tools import inspect_npy_bytes


def inspect_npy_file(filename):
    """
    Inspect a .npy file to extract metadata such as data offset, shape, and dtype.

    Parameters
    ----------
    filename : str
        Path to the .npy file to inspect.

    Returns
    -------
    tuple
        A tuple containing:
        - data_offset (int): The byte offset where the data starts.
        - shape (tuple): The shape of the array stored in the file.
        - dtype (str): The data type of the array elements.

    Example
    -------
    >>> offset, shape, dtype = inspect_npy_file('example.npy')
    >>> print(f"Data offset: {offset} bytes")
    >>> print(f"Shape: {shape}")
    >>> print(f"Dtype: {dtype}")
    """
    with open(filename, "rb") as f:
        return inspect_npy_bytes(f)


def file_writer(filename, signal, **kwargs):
    """
    Write data to npy files.

    Parameters
    ----------
    %s
    %s
    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`numpy.save`.

    %s
    """
    array = signal["data"]
    if isinstance(array, da.Array):
        raise TypeError("Lazy signal are not supported for writing to npy files.")

    np.save(filename, array, **kwargs)


file_writer.__doc__ %= (
    FILENAME_DOC.replace("read", "write to"),
    SIGNAL_DOC,
    UNSUPPORTED_METADATA_DOC,
)


def file_reader(filename, lazy=False, chunks="auto", navigation_axes=None, **kwargs):
    """
    Read data from npy files.

    Parameters
    ----------
    %s
    %s
    %s
    navigation_axes : list, optional
        List of axes that should be treated as navigation axes. If not provided,
        all axes will be treated as signal axes.
    **kwargs : dict, optional
        Pass keyword arguments to the :func:`numpy.load`, when
        lazy is False, otherwise to :func:`rsciio.utils.distributed.memmap_distributed`.

    %s

    %s

    Examples
    --------
    To load a numpy file with lazy loading and specified chunks:

    >>> from rsciio.numpy import file_reader
    >>> d = file_reader('data.npy', lazy=True, chunks=("auto", "auto", 250, 250), navigation_axes=[0, 1])
    >>> d[0]['data']
    dask.array<array, shape=(1000, 1000, 500, 500), dtype=float64, chunksize=(100, 100, 250, 250)>
    """
    if lazy:
        offset, shape, dtype = inspect_npy_file(filename)
        data = memmap_distributed(
            filename,
            offset=offset,
            shape=shape,
            dtype=np.dtype(dtype),
            chunks=chunks,
            **kwargs,
        )
    else:
        data = np.load(filename, **kwargs)

    axes = []
    index_in_array = 0
    if navigation_axes is None:
        navigation_axes = []
    for axis, length in enumerate(data.shape):
        if length > 1:
            axes.append(
                {
                    "size": length,
                    "index_in_array": index_in_array,
                    "name": "",
                    "scale": 1.0,
                    "offset": 0.0,
                    "units": "",
                    "navigate": axis in navigation_axes,
                }
            )
            index_in_array += 1

    return [
        {
            "data": data.squeeze(),
            "axes": axes,
            "metadata": {},
            "original_metadata": {},
        },
    ]


file_reader.__doc__ %= (
    FILENAME_DOC,
    LAZY_DOC,
    CHUNKS_DOC,
    RETURNS_DOC,
    UNSUPPORTED_METADATA_DOC,
)
