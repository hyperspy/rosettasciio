# -*- coding: utf-8 -*-
# Copyright 2025 CEOS GmbH
# Copyright 2022-2025 The HyperSpy developers
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
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/>.
"""Implements an unpickler, which restricts the allowed modules which pickle can load."""

import pickle
from logging import getLogger

import numpy

from rsciio.utils.tools import inspect_npy_bytes


class InvalidPickleError(Exception):
    pass


class RestrictedUnpickler(pickle.Unpickler):
    """
    By overriding `find_class` limits the modules which can be loaded onto the
    pickle stack by the following `white_list`:
    - numpy._core.multiarray._reconstruct
    - numpy._core.multiarray.scalar
    - numpy.ndarray
    - numpy.dtype
    - _codecs.encode
    """

    # The two almost identical NumPy API `core` entries cover older
    # versions (<1.26) and were tested up to NumPy 2.3.
    # Might need additional adaptation for later NumPy >2.3 versions.
    white_list = {
        "numpy._core.multiarray": ["_reconstruct", "scalar"],
        "numpy.core.multiarray": ["_reconstruct", "scalar"],  # numpy <1.26
        "numpy": ["ndarray", "dtype"],
        "_codecs": [
            "encode",
        ],
    }

    def find_class(self, module, name):
        if (module in self.white_list) and (name in self.white_list[module]):
            getLogger(__name__).info(
                "find_class: module = {}, name = {} TRUSTED".format(module, name)
            )
            return super().find_class(module, name)
        else:
            getLogger(__name__).info(
                "find_class: module = {}, name = {} FAILED".format(module, name)
            )
            raise InvalidPickleError(
                f"Invalid names in pickle detected:\n`{module}`, `{name}`"
            )


def read_pickled_array(fp):
    """
    Read pickled data from a NPY file using the RestrictedUnpickler.

    Parameters
    ----------
    fp : file_like object
        The NPY file to read the pickled data from.

    Returns
    -------
    array : ndarray
        The data read from the file.
    """
    # advances the byte stream past the numpy header
    _, _, dtype_str = inspect_npy_bytes(fp)
    dtype = numpy.dtype(dtype_str)
    if not dtype.hasobject:
        raise ValueError("File does not contain pickled data.")

    return RestrictedUnpickler(fp).load()
