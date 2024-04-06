# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

PLUGIN_LAZY_NOT_IMPLEMENTED = [
    # "bruker", # SPX only
    "dens",
    "digitalsurf",
    "impulse",
    "jobinyvon",
    "msa",
    "netcdf",
    "pantarhei",
    "phenom",
    "protochips",
    "renishaw",
    "trivista",
]


@pytest.mark.parametrize("plugin", PLUGIN_LAZY_NOT_IMPLEMENTED)
def test_lazy_not_implemented(plugin):
    fname = "fname"
    if plugin == "bruker":
        fname = "fname.spx"
    if plugin == "phenom":
        pytest.importorskip("tifffile")
    if plugin == "netcdf":
        pytest.importorskip("scipy")

    file_reader = importlib.import_module(f"rsciio.{plugin}").file_reader

    with pytest.raises(NotImplementedError):
        file_reader(fname, lazy=True)
