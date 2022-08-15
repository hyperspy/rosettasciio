# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

import importlib

import pytest


@pytest.mark.no_hyperspy
def test_import_version():
    from rsciio import __version__


@pytest.mark.no_hyperspy
def test_import_minimal_no_hyperspy():
    from rsciio import IO_PLUGINS

    for plugin in IO_PLUGINS:
        if plugin["format_name"] not in [
            "Blockfile",
            "Electron Microscopy Data (EMD)",
            "image",
            "MRCZ",
            "Phenom Element Identification (ELID)",
            "TIFF",
            "TVIPS",
            "USID",
            "ZSpy",
        ]:
            importlib.import_module(plugin["api"])


def test_import_all():
    from rsciio import IO_PLUGINS

    for plugin in IO_PLUGINS:
        importlib.import_module(plugin["api"])
