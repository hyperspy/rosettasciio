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

import rsciio


def test_import_version():
    from rsciio import __version__  # noqa


def test_rsciio_dir():
    assert dir(rsciio) == ["IO_PLUGINS", "__version__", "set_log_level"]


def test_rsciio_utils():
    pytest.importorskip("h5py")
    from rsciio.utils import hdf5 as utils_hdf5

    assert dir(utils_hdf5) == ["list_datasets_in_file", "read_metadata_from_file"]


def test_import_all():
    plugin_name_to_remove = []

    # Remove plugins which require not installed optional dependencies
    h5py = importlib.util.find_spec("h5py")
    if h5py is None:
        plugin_name_to_remove.extend(["EMD", "HSPY", "NeXus"])

    imageio = importlib.util.find_spec("imageio")
    if imageio is None:
        plugin_name_to_remove.extend(["Image"])

    sparse = importlib.util.find_spec("sparse")
    if sparse is None:
        plugin_name_to_remove.extend(["EMD", "JEOL"])

    skimage = importlib.util.find_spec("skimage")
    if skimage is None:
        plugin_name_to_remove.append("Blockfile")

    mrcz = importlib.util.find_spec("mrcz")
    if mrcz is None:
        plugin_name_to_remove.append("MRCZ")

    tifffile = importlib.util.find_spec("tifffile")
    if tifffile is None:
        plugin_name_to_remove.append("TIFF")
        plugin_name_to_remove.append("Phenom")

    pyUSID = importlib.util.find_spec("pyUSID")
    if pyUSID is None:
        plugin_name_to_remove.append("USID")

    zarr = importlib.util.find_spec("zarr")
    if zarr is None:
        plugin_name_to_remove.append("ZSPY")

    IO_PLUGINS_ = list(
        filter(
            lambda plugin: plugin["name"] not in set(plugin_name_to_remove),
            rsciio.IO_PLUGINS,
        )
    )

    for plugin in IO_PLUGINS_:
        importlib.import_module(plugin["api"])


def test_format_name_aliases():
    for reader in rsciio.IO_PLUGINS:
        assert isinstance(reader["name"], str)
        assert isinstance(reader["name_aliases"], list)
        if reader["name_aliases"]:
            for aliases in reader["name_aliases"]:
                assert isinstance(aliases, str)
        assert isinstance(reader["description"], str)
        assert isinstance(reader["full_support"], bool)
        assert isinstance(reader["file_extensions"], list)
        for extensions in reader["file_extensions"]:
            assert isinstance(extensions, str)
        assert isinstance(reader["default_extension"], int)
        if isinstance(reader["writes"], list):
            for i in reader["writes"]:
                assert isinstance(i, list)
        else:
            assert isinstance(reader["writes"], bool)
        assert isinstance(reader["non_uniform_axis"], bool)


@pytest.mark.parametrize("plugin", rsciio.IO_PLUGINS)
def test_dir_plugins(plugin):
    plugin_string = "rsciio.%s" % plugin["name"].lower()
    # skip for missing optional dependencies
    if plugin["name"] == "Blockfile":
        pytest.importorskip("skimage")
    elif plugin["name"] == "Image":
        pytest.importorskip("imageio")
    elif plugin["name"] == "MRCZ":
        pytest.importorskip("mrcz")
    elif plugin["name"] in ["TIFF", "Phenom"]:
        pytest.importorskip("tifffile")
    elif plugin["name"] == "USID":
        pytest.importorskip("pyUSID")
    elif plugin["name"] == "ZSPY":
        pytest.importorskip("zarr")
    elif plugin["name"] in ["EMD", "HSPY", "NeXus"]:
        pytest.importorskip("h5py")
    plugin_module = importlib.import_module(plugin_string)

    if plugin["name"] == "MSA":
        assert dir(plugin_module) == [
            "file_reader",
            "file_writer",
            "parse_msa_string",
        ]
    elif plugin["name"] == "QuantumDetector":
        assert dir(plugin_module) == [
            "file_reader",
            "load_mib_data",
            "parse_exposures",
            "parse_timestamps",
        ]
    elif plugin["writes"] is False:
        assert dir(plugin_module) == ["file_reader"]
    else:
        assert dir(plugin_module) == ["file_reader", "file_writer"]
