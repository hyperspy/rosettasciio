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
from dask.array import Array
from dask.diagnostics import ProgressBar

from rsciio._docstrings import SHOW_PROGRESSBAR_DOC
from rsciio.utils.array import get_numpy_kwargs
from rsciio.utils.tools import dummy_context_manager

rgba8 = np.dtype({"names": ["R", "G", "B", "A"], "formats": ["u1", "u1", "u1", "u1"]})
rgb8 = np.dtype({"names": ["R", "G", "B"], "formats": ["u1", "u1", "u1"]})

rgba16 = np.dtype({"names": ["R", "G", "B", "A"], "formats": ["u2", "u2", "u2", "u2"]})
rgb16 = np.dtype({"names": ["R", "G", "B"], "formats": ["u2", "u2", "u2"]})
rgb_dtypes = {"rgb8": rgb8, "rgb16": rgb16, "rgba8": rgba8, "rgba16": rgba16}


def is_rgba(array):
    if array.dtype in (rgba8, rgba16):
        return True
    else:
        return False


def is_rgb(array):
    if array.dtype in (rgb8, rgb16):
        return True
    else:
        return False


def is_rgbx(array):
    if is_rgb(array) or is_rgba(array):
        return True
    else:
        return False


def rgbx2regular_array(data, plot_friendly=False, show_progressbar=True):
    """Transforms a RGBx array into a standard one

    Parameters
    ----------
    data : numpy array of RGBx dtype
    plot_friendly : bool
        If True change the dtype to float when dtype is not uint8 and
        normalize the array so that it is ready to be plotted by matplotlib.
    %s
    """
    # Make sure that the data is contiguous
    if isinstance(data, Array):
        cm = ProgressBar if show_progressbar else dummy_context_manager
        with cm():
            data = data.compute()
    if data.flags["C_CONTIGUOUS"] is False:
        if np.ma.is_masked(data):
            data = data.copy(order="C")
        else:
            data = np.ascontiguousarray(data, **get_numpy_kwargs(data))
    if is_rgba(data) is True:
        dt = data.dtype.fields["B"][0]
        data = data.view((dt, 4))
    elif is_rgb(data) is True:
        dt = data.dtype.fields["B"][0]
        data = data.view((dt, 3))
    else:
        return data
    if plot_friendly is True and data.dtype == np.dtype("uint16"):
        data = data.astype("float")
        data /= 2**16 - 1
    return data


rgbx2regular_array.__doc__ %= (SHOW_PROGRESSBAR_DOC,)


def regular_array2rgbx(data):
    # Make sure that the data is contiguous
    if data.flags["C_CONTIGUOUS"] is False:
        if np.ma.is_masked(data):
            data = data.copy(order="C")
        else:
            data = np.ascontiguousarray(data, **get_numpy_kwargs(data))
    if data.shape[-1] == 3:
        names = rgb8.names
    elif data.shape[-1] == 4:
        names = rgba8.names
    else:
        raise ValueError("The last dimension size of the array must be 3 or 4")
    if data.dtype in (np.dtype("u1"), np.dtype("u2")):
        formats = [data.dtype] * len(names)
    else:
        raise ValueError("The data dtype must be uint16 or uint8")
    return data.view(np.dtype({"names": names, "formats": formats})).reshape(
        data.shape[:-1]
    )
