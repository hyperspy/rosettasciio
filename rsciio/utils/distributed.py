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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

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
