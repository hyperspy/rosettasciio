# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.
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
