# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.
"""Utility functions for file handling."""


# ruff: noqa: F822

import importlib

__all__ = [
    "get_file_handle",
    "inspect_npy_bytes",
    "memmap_distributed",
]


def __dir__():
    return sorted(__all__)


_import_mapping = {
    "get_file_handle": "_tools",
    "inspect_npy_bytes": "_tools",
    "memmap_distributed": "_distributed",
}


def __getattr__(name):
    if name in __all__:
        submodule = _import_mapping[name]
        return getattr(importlib.import_module(f"rsciio.utils.{submodule}"), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
