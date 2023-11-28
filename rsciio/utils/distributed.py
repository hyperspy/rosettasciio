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


import numpy as np
import dask.array as da


def get_chunk_slice(
    shape,
    chunks="auto",
    block_size_limit=None,
    dtype=None,
):
    """
    Get chunk slices for the :func:`rsciio.utils.distributed.slice_memmap` function.

    Takes a shape and chunks and returns a dask array of the slices to be used with the
    :func:`rsciio.utils.distributed.slice_memmap` function. This is useful for loading data from a memmaped file in a
    distributed manner.

    Parameters
    ----------
    shape : tuple
        Shape of the data.
    chunks : tuple, optional
        Chunk shape. The default is "auto".
    block_size_limit : int, optional
        Maximum size of a block in bytes. The default is None.
    dtype : numpy.dtype, optional
        Data type. The default is None.

    Returns
    -------
    dask.array.Array
        Dask array of the slices.
    """

    chunks = da.core.normalize_chunks(
        chunks=chunks, shape=shape, limit=block_size_limit, dtype=dtype
    )
    chunks_shape = tuple([len(c) for c in chunks])
    slices = np.empty(
        shape=chunks_shape
        + (
            len(chunks_shape),
            2,
        ),
        dtype=int,
    )
    for ind in np.ndindex(chunks_shape):
        current_chunk = [chunk[i] for i, chunk in zip(ind, chunks)]
        starts = [int(np.sum(chunk[:i])) for i, chunk in zip(ind, chunks)]
        stops = [s + c for s, c in zip(starts, current_chunk)]
        slices[ind] = [[start, stop] for start, stop in zip(starts, stops)]
    return da.from_array(slices, chunks=(1,) * len(shape) + slices.shape[-2:]), chunks


def slice_memmap(sl, file, dtypes, shape, **kwargs):
    """
    Slice a memmaped file using a tuple of slices.

    This is useful for loading data from a memmaped file in a distributed manner. This takes
    a slice of the dimensions of the data and returns the data from the memmaped file sliced.

    Parameters
    ----------
    sl : array-like
        An array of the slices to use. The dimensions of the array should be
        (n,2) where n is the number of dimensions of the data. The first column
        is the start of the slice and the second column is the stop of the slice.
    file : str
        Path to the file.
    dtypes : numpy.dtype
        Data type of the data for memmap function.
    shape : tuple
        Shape of the data to be read.
    **kwargs : dict
        Additional keyword arguments to pass to the memmap function.

    Returns
    -------
    numpy.ndarray
        Array of the data from the memmaped file sliced using the provided slice.
    """
    sl = np.squeeze(sl)[()]
    data = np.memmap(file, dtypes, shape=shape, **kwargs)
    slics = tuple([slice(s[0], s[1]) for s in sl])
    return data[slics]


def memmap_distributed(
    file,
    dtype,
    offset=0,
    shape=None,
    order="C",
    chunks="auto",
    block_size_limit=None,
):
    """
    Drop in replacement for py:func:`numpy.memmap` allowing for distributed loading of data.

    This always loads the data using dask which can be beneficial in many cases, but
    may not be ideal in others. The chunks and block_size_limit are for describing an ideal chunk shape and size
    as defined using the `da.core.normalize_chunks` function.

    Parameters
    ----------
    file : str
        Path to the file.
    dtype : numpy.dtype
        Data type of the data for memmap function.
    offset : int, optional
        Offset in bytes. The default is 0.
    shape : tuple, optional
        Shape of the data to be read. The default is None.
    order : str, optional
        Order of the data. The default is "C" see py:func:`numpy.memmap` for more details.
    chunks : tuple, optional
        Chunk shape. The default is "auto".
    block_size_limit : int, optional
        Maximum size of a block in bytes. The default is None.

    Returns
    -------
    dask.array.Array
        Dask array of the data from the memmaped file and with the specified chunks.
    """
    # Separates slices into appropriately sized chunks.
    chunked_slices, data_chunks = get_chunk_slice(
        shape=shape,
        chunks=chunks,
        block_size_limit=block_size_limit,
        dtype=dtype,
    )
    num_dim = len(shape)
    data = da.map_blocks(
        slice_memmap,
        chunked_slices,
        file=file,
        dtype=dtype,
        shape=shape,
        order=order,
        mode="r",
        dtypes=dtype,
        offset=offset,
        chunks=data_chunks,
        drop_axis=(
            num_dim,
            num_dim + 1,
        ),  # Dask 2021.10.0 minimum to use negative indexing
    )
    return data
