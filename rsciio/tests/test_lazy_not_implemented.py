# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

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
