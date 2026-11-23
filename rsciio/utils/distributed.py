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

from rsciio.exceptions import VisibleDeprecationWarning

# This module is deprecated and will be removed in version 1.0.

__all__ = [
    "memmap_distributed",
]


def __dir__():
    return sorted(__all__)


warnings.warn(
    "The module `rsciio.utils.distributed` has been deprecated and will be removed in version 1.0.",
    VisibleDeprecationWarning,
)


def __getattr__(name):
    if name in __all__:
        warnings.warn(
            f"{name} has been moved to `rsciio.utils.file` and will be removed from "
            "`rsciio.utils.distributed` in version 1.0.",
            VisibleDeprecationWarning,
        )  # pragma: no cover
        return getattr(importlib.import_module("rsciio.utils.file"), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
