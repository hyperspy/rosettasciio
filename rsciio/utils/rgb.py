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
"""Utility functions for RGB array handling."""

import numpy as np

from rsciio._docstrings import SHOW_PROGRESSBAR_DOC
from rsciio.utils._array import get_numpy_kwargs

__all__ = [
    "is_rgba",
    "is_rgb",
    "is_rgbx",
    "rgbx2regular_array",
    "regular_array2rgbx",
    "RGB_DTYPES",
]


def __dir__():
    return sorted(__all__)


_RGBA8 = np.dtype({"names": ["R", "G", "B", "A"], "formats": ["u1", "u1", "u1", "u1"]})
_RGB8 = np.dtype({"names": ["R", "G", "B"], "formats": ["u1", "u1", "u1"]})

_RGBA16 = np.dtype({"names": ["R", "G", "B", "A"], "formats": ["u2", "u2", "u2", "u2"]})
_RGB16 = np.dtype({"names": ["R", "G", "B"], "formats": ["u2", "u2", "u2"]})
RGB_DTYPES = {"rgb8": _RGB8, "rgb16": _RGB16, "rgba8": _RGBA8, "rgba16": _RGBA16}
"""
Mapping of RGB color space names to their corresponding numpy structured dtypes.

:meta hide-value:
"""


_DOCSTRING_BASE = """Check if the array is a {} structured numpy array.

    Parameters
    ----------
    array : numpy.ndarray
        The array to check.

    Returns
    -------
    bool
        True if the array is {}, False otherwise.
    """


def is_rgba(array):
    """
    %s
    """
    if array.dtype in (_RGBA8, _RGBA16):
        return True
    else:
        return False


is_rgba.__doc__ %= _DOCSTRING_BASE.format("RGBA", "RGBA")


def is_rgb(array):
    """
    %s
    """
    if array.dtype in (_RGB8, _RGB16):
        return True
    else:
        return False


is_rgb.__doc__ %= _DOCSTRING_BASE.format("RGB", "RGB")


def is_rgbx(array):
    """
    %s
    """
    if is_rgb(array) or is_rgba(array):
        return True
    else:
        return False


is_rgbx.__doc__ %= _DOCSTRING_BASE.format("RGB or RGBA", "RGB or RGBA")


def rgbx2regular_array(data, plot_friendly=False, show_progressbar=True):
    """
    Transform a RGBx structured numpy array into a standard one with
    an additional dimension for the color channel.

    Parameters
    ----------
    data : numpy.ndarray or dask.array.Array
        The RGB array to be transformed.
    plot_friendly : bool
        If True change the dtype to float when dtype is not uint8 and
        normalize the array so that it is ready to be plotted by matplotlib.
    %s

    Returns
    -------
    numpy.ndarray or dask.array.Array
        The transformed array with additional dimension for the color channel.
    """
    # lazy import dask.array
    from dask.array import Array
    from dask.diagnostics import ProgressBar

    from rsciio.utils._tools import dummy_context_manager

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


rgbx2regular_array.__doc__ %= SHOW_PROGRESSBAR_DOC


def regular_array2rgbx(data):
    """
    Transform a regular numpy array with an additional dimension for the
    color channel into a RGBx structured numpy array.

    Parameters
    ----------
    data : numpy.ndarray or dask.array.Array
        The regular array to be transformed.

    Returns
    -------
    numpy.ndarray or dask.array.Array
        The transformed RGBx structured array.
    """
    # Make sure that the data is contiguous
    if data.flags["C_CONTIGUOUS"] is False:
        if np.ma.is_masked(data):
            data = data.copy(order="C")
        else:
            data = np.ascontiguousarray(data, **get_numpy_kwargs(data))
    if data.shape[-1] == 3:
        names = _RGB8.names
    elif data.shape[-1] == 4:
        names = _RGBA8.names
    else:
        raise ValueError("The last dimension size of the array must be 3 or 4")
    if data.dtype in (np.dtype("u1"), np.dtype("u2")):
        formats = [data.dtype] * len(names)
    else:
        raise ValueError("The data dtype must be uint16 or uint8")
    return data.view(np.dtype({"names": names, "formats": formats})).reshape(
        data.shape[:-1]
    )
