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
