# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

import importlib

from ._logger import set_log_level

# Default to warning
set_log_level("WARNING")


__all__ = ["__version__", "IO_PLUGINS", "set_log_level"]


_import_mapping = {
    "__version__": "._version",
    "IO_PLUGINS": "._io_plugins",
}


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in _import_mapping.keys():
        import_path = "rsciio" + _import_mapping.get(name)
        return getattr(importlib.import_module(import_path), name)

    if name in __all__:
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
