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
    "is_rgba",
    "is_rgb",
    "is_rgbx",
    "rgb8",
    "rgba8",
    "rgb16",
    "rgba16",
    "rgb_dtypes",
    "rgbx2regular_array",
    "regular_array2rgbx",
]


def __dir__():
    return sorted(__all__)


warnings.warn(
    "The module `rsciio.utils.rgb_tools` has been renamed to `rsciio.utils.rgb` "
    "and it will be removed in version 1.0.",
    VisibleDeprecationWarning,
)


def __getattr__(name):
    if name in __all__:
        text = ""
        if name == "rgb_dtypes":
            name = "RGB_DTYPES"
            text = " Use `RGB_DTYPES` instead"
        if name in ["rgb8", "rgba8", "rgb16", "rgba16"]:
            name = "_" + name.upper()
            text = " Use `RGB_DTYPES` instead"
        warnings.warn(
            "The module `rsciio.utils.rgb_tools` has been renamed to `rsciio.utils.rgb` "
            "and it will be removed in version 1.0." + text,
            VisibleDeprecationWarning,
        )
        return getattr(importlib.import_module("rsciio.utils.rgb"), name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
