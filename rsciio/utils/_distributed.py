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

import os

import dask.array as da
import numpy as np


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
    chunks : tuple or str, optional
        Define the chunk shape. This argument is passed to :func:`dask.array.core.normalize_chunks`.
        The default is "auto".
    block_size_limit : int, optional
        Maximum size of a block in bytes. The default is None. This is passed
        to the :py:func:`dask.array.core.normalize_chunks` function when chunks == "auto".
    dtype : numpy.dtype, optional
        Data type. The default is None. This is passed to the
        :py:func:`dask.array.core.normalize_chunks` function when chunks == "auto".

    Returns
    -------
    dask.array.Array
        Dask array of the slices.
    tuple
        Tuple of the chunks.
    """

    chunks = da.core.normalize_chunks(
        chunks=chunks, shape=shape, limit=block_size_limit, dtype=dtype
    )
    chunks_shape = tuple([len(c) for c in chunks])
    slices = np.empty(
        shape=chunks_shape + (len(chunks_shape), 2),
        dtype=int,
    )
    for ind in np.ndindex(chunks_shape):
        current_chunk = [chunk[i] for i, chunk in zip(ind, chunks)]
        starts = [int(np.sum(chunk[:i])) for i, chunk in zip(ind, chunks)]
        stops = [s + c for s, c in zip(starts, current_chunk)]
        slices[ind] = [[start, stop] for start, stop in zip(starts, stops)]

    return da.from_array(slices, chunks=(1,) * len(shape) + slices.shape[-2:]), chunks


def get_arbitrary_chunk_slice(
    positions,
    shape,
    chunks="auto",
    block_size_limit=None,
    dtype=None,
):
    """
    Get chunk slices for the :func:`rsciio.utils.distributed.slice_memmap` function. From arbitrary positions
    given by a list of x, y coordinates.

    Parameters
    ----------
    positions : array-like
        A numpy array in the form [[x1, y1], [x2, y2], ...] where x, y map the frame to the
        real space coordinate of the data.
    shape : tuple
        Shape of the signal data.
    chunks : tuple, optional
        Chunk shape. The default is "auto".
    block_size_limit : int, optional
        Maximum size of a block in bytes. The default is None. This is passed
        to the :py:func:`dask.array.core.normalize_chunks` function when chunks == "auto".
    dtype : numpy.dtype, optional
        Data type. The default is None. This is passed to the
        :py:func:`dask.array.core.normalize_chunks` function when chunks == "auto".

    Returns
    -------
    dask.array.Array
        Dask array of the slices.
    """
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    if chunks == "auto":
        chunks = ("auto",) * (len(shape) - 2) + (-1, -1)
    elif chunks[-2:] != (-1, -1):
        raise ValueError("Last two dimensions of chunks must be -1")
    chunks = da.core.normalize_chunks(
        chunks=chunks, shape=shape, limit=block_size_limit, dtype=dtype
    )
    pos_mapping = np.zeros(shape=shape[:-2] + (1, 1), dtype=int)

    for i, p in enumerate(positions):
        pos_mapping[tuple(p)] = i + 1
    pos_mapping = pos_mapping - 1  # 0 based indexing, -1 for the empty frames

    # Now we chunk the pos_mapping array.  In the case each frame remains in a single chunk and we only
    # return the navigation dimensions.  Later when we populate the data we will use the pos_mapping array
    # map some frame index to the position within a dense array.
    return da.from_array(pos_mapping, chunks=chunks[:-2] + (1, 1)), chunks


def slice_memmap(slices, file, dtypes, shape, key=None, positions=False, **kwargs):
    """
    Slice a memory mapped file using a tuple of slices.

    This is useful for loading data from a memory mapped file in a distributed manner. The function
    first creates a memory mapped array of the entire dataset and then uses the ``slices`` to slice the
    memory mapped array.  The slices can be used to build a ``dask`` array as each slice translates to one
    chunk for the ``dask`` array.

    Parameters
    ----------
    slices : array-like of int
        An array of the slices to use. The dimensions of the array should be
        (n,2) where n is the number of dimensions of the data. The first column
        is the start of the slice and the second column is the stop of the slice.
    file : str
        Path to the file.
    dtypes : numpy.dtype
        Data type of the data for :class:`numpy.memmap` function.
    shape : tuple
        Shape of the entire dataset. Passed to the :class:`numpy.memmap` function.
    key : None, str
        For structured dtype only. Specify the key of the structured dtype to use.
    positions : bool, optional
        If True, the slices include indexes for positions which are then used to
        create a custom scan pattern. The default is False.
    **kwargs : dict
        Additional keyword arguments to pass to the :class:`numpy.memmap` function.

    Returns
    -------
    numpy.ndarray
        Array of the data from the memory mapped file sliced using the provided slice.
    """
    slices_ = np.squeeze(slices)[()]
    data = np.memmap(file, dtypes, shape=shape, **kwargs)
    if key is not None:
        data = data[key]
    if positions:
        # We have arbitrary positions.
        if -1 in slices_:  # -1 means empty frame we will return 0.
            result = data[slices_]
            result[slices_ == -1] = 0
            return result
        else:
            return data[slices_]
    else:
        slices_ = tuple([slice(s[0], s[1]) for s in slices_])
        return data[slices_]


def memmap_distributed(
    filename,
    dtype,
    positions=None,
    offset=0,
    shape=None,
    order="C",
    chunks="auto",
    block_size_limit=None,
    key=None,
):
    """
    Drop in replacement for py:func:`numpy.memmap` allowing for distributed
    loading of data.

    This always loads the data using dask which can be beneficial in many
    cases, but may not be ideal in others. The ``chunks`` and ``block_size_limit``
    are for describing an ideal chunk shape and size as defined using the
    :func:`dask.array.core.normalize_chunks` function.

    Parameters
    ----------
    filename : str
        Path to the file.
    dtype : numpy.dtype
        Data type of the data for memmap function.
    positions : array-like, optional
        A numpy array in the form [[x1, y1], [x2, y2], ...] where x, y map the frame to the
        real space coordinate of the data. The default is None.
    offset : int, optional
        Offset in bytes. The default is 0.
    shape : tuple, optional
        Shape of the data to be read. The default is None.
    order : str, optional
        Order of the data. The default is "C" see :class:`numpy.memmap` for more details.
    chunks : tuple, optional
        Chunk shape. The default is "auto".
    block_size_limit : int, optional
        Maximum size of a block in bytes. The default is None.
    key : None, str
        For structured dtype only. Specify the key of the structured dtype to use.

    Returns
    -------
    dask.array.Array
        Dask array of the data from the memmaped file and with the specified chunks.

    Notes
    -----
    Currently :func:`dask.array.map_blocks` does not allow for multiple outputs.
    As a result, in case of structured dtype, the key of the structured dtype need
    to be specified.
    For example: with dtype = (("data", int, (128, 128)), ("sec", "<u4", 512)),
    "data" or "sec" will need to be specified.
    """

    if dtype.names is not None:
        # Structured dtype
        array_dtype = dtype[key].base
        sub_array_shape = dtype[key].shape
    else:
        array_dtype = dtype.base
        sub_array_shape = dtype.shape

    if shape is None:
        unit_size = np.dtype(dtype).itemsize
        shape = int(os.path.getsize(filename) / unit_size)
    if not isinstance(shape, tuple):
        shape = (shape,)

    num_dim = len(shape + sub_array_shape)
    if positions is not None:
        # We have arbitrary positions
        chunked_slices, data_chunks = get_arbitrary_chunk_slice(
            positions=positions,
            shape=shape + sub_array_shape,
            chunks=chunks,
            block_size_limit=block_size_limit,
            dtype=array_dtype,
        )
        drop_axes = None
        use_positions = True
        shape = (len(positions),) + shape[-2:]  # update the shape to be linear
    else:
        # Separates slices into appropriately sized chunks.
        chunked_slices, data_chunks = get_chunk_slice(
            shape=shape + sub_array_shape,
            chunks=chunks,
            block_size_limit=block_size_limit,
            dtype=array_dtype,
        )
        drop_axes = (
            num_dim,
            num_dim + 1,
        )  # Dask 2021.10.0 minimum to use negative indexing
        use_positions = False
    data = da.map_blocks(
        slice_memmap,
        chunked_slices,
        file=filename,
        dtype=array_dtype,
        shape=shape,
        order=order,
        mode="r",
        dtypes=dtype,
        offset=offset,
        chunks=data_chunks,
        drop_axis=drop_axes,
        positions=use_positions,
        key=key,
    )
    return data
