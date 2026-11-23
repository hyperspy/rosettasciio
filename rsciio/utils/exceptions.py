# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

# ruff: noqa: F822

import importlib
import warnings

import rsciio.exceptions

# This module is deprecated and will be removed in version 1.0.

__all__ = [
    "MountainsMapFileError",
    "ByteOrderError",
    "DM3FileVersionError",
    "DM3TagError",
    "DM3DataTypeError",
    "DM3TagTypeError",
    "DM3TagIDError",
    "VisibleDeprecationWarning",
    "LazyCupyConversion",
]


def __dir__():
    return sorted(__all__)


warnings.warn(
    "The module `rsciio.utils.exceptions` has been moved to `rsciio.exceptions` "
    "and it will be removed in version 1.0.",
    rsciio.exceptions.VisibleDeprecationWarning,
)


def __getattr__(name):
    if name in __all__:
        warnings.warn(
            f"{name} has been moved to `rsciio.exceptions` and it will be removed from "
            "`rsciio.utils.exceptions` in version 1.0.",
            rsciio.exceptions.VisibleDeprecationWarning,
        )  # pragma: no cover
        return getattr(importlib.import_module("rsciio.exceptions"), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
