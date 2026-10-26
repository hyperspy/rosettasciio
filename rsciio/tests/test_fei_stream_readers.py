# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

"""Test the FEI stream readers.

Because there is no official description of the format, these tests just tests
consistency between ``array_to_stream`` and ``stream_to*array``. In the
particular case of stream to sparse array, we use dask to compute the array
in order to mimic the usage in the FEI EMD reader.

"""

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("sparse")

from rsciio.utils._fei_stream_readers import (  # noqa: E402
    array_to_stream,
    stream_to_array,
    stream_to_sparse_COO_array,
)


@pytest.mark.parametrize("lazy", (True, False))
def test_dense_stream(lazy):
    arr = np.random.randint(0, 65535, size=(2, 3, 4, 5)).astype("uint16")
    stream = array_to_stream(arr)
    if lazy:
        arrs = stream_to_sparse_COO_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5, last_frame=2
        )
        arrs = arrs.compute()
        assert (arrs == arr).all()
    else:
        arrs = stream_to_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5, last_frame=2
        )
        assert (arrs == arr).all()


@pytest.mark.parametrize("lazy", (True, False))
def test_empty_stream(lazy):
    arr = np.zeros((2, 3, 4, 5), dtype="uint16")
    stream = array_to_stream(arr)
    if lazy:
        arrs = stream_to_sparse_COO_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5, last_frame=2
        )
        arrs = arrs.compute()
        assert not arrs.any()
    else:
        arrs = stream_to_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5, last_frame=2
        )
        assert not arrs.any()


@pytest.mark.parametrize("lazy", (True, False))
def test_sparse_stream(lazy):
    arr = np.zeros((2, 3, 4, 5), dtype="uint16")
    arr[0, 0, 0, 0] = 1
    arr[-1, -1, -1, -1] = 2
    arr[1, 1, 3, 3] = 3
    stream = array_to_stream(arr)
    if lazy:
        arrs = stream_to_sparse_COO_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5, last_frame=2
        )
        arrs = arrs.compute()
        assert (arrs == arr).all()
    else:
        arrs = stream_to_array(
            stream, spatial_shape=(3, 4), sum_frames=False, channels=5, last_frame=2
        )
        assert (arrs == arr).all()
