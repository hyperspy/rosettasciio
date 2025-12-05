# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
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

__all__ = [
    "dummy_context_manager",
    "sanitize_msxml_float",
    "dump_dictionary",
    "XmlToDict",
    "xml2dtb",
    "DTBox",
    "convert_xml_to_dict",
    "sarray2dict",
    "dict2sarray",
    "convert_units",
    "get_object_package_info",
    "ensure_unicode",
    "get_file_handle",
    "inspect_npy_bytes",
    "jit_ifnumba",
    "append2pathname",
    "incremental_filename",
    "ensure_directory",
    "overwrite",
]


_import_mapping = {
    "dummy_context_manager": "._tools",
    "sanitize_msxml_float": ".xml",
    "XmlToDict": ".xml",
    "xml2dtb": ".xml",
    "convert_xml_to_dict": ".xml",
    "dump_dictionary": ".dictionary",
    "DTBox": ".dictionary",
    "sarray2dict": ".dictionary",
    "dict2sarray": ".dictionary",
    "convert_units": ".units",
    "get_object_package_info": "._tools",
    "ensure_unicode": "._tools",
    "get_file_handle": "._tools",
    "inspect_npy_bytes": "_tools",
    "jit_ifnumba": "._tools",
    "append2pathname": ".path",
    "incremental_filename": ".path",
    "ensure_directory": ".path",
    "overwrite": ".path",
}


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    if name in __all__:
        if name in _import_mapping.keys():
            import_path = "rsciio.utils" + _import_mapping.get(name)
            return getattr(importlib.import_module(import_path), name)
        else:
            return importlib.import_module("." + name, "rsciio.utils")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
