# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

import importlib

__all__ = [
    "file",
    "hdf5",
    "path",
    "rgb",
    "tools",
    "xml",
]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        return importlib.import_module(f".{name}", "rsciio.utils")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
