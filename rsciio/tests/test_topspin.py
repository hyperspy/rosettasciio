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

from pathlib import Path

import h5py
import numpy as np
import pytest

from rsciio.topspin._api import _parse_app5_xml, file_reader

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")


data_directory = Path(__file__).parent / "data" / "topspin"
file_from_disk = data_directory / "topspin_from_file.app5"
file_from_export = data_directory / "topspin_from_export.app5"


def test_xml_parser():
    # make sure we can open a metadata file
    f1 = h5py.File(file_from_disk, "r")
    meta_dict_1 = _parse_app5_xml(f1["Metadata"][()].decode())
    f1.close()
    # make sure we can open a nested metadata file
    f2 = h5py.File(file_from_export, "r")
    for grp in [x for i, x in enumerate(f2.keys()) if i in [0, 2]]:
        metadata_string = f2[grp]["Metadata"][()].decode()
        meta_dict_2 = _parse_app5_xml(metadata_string)
    f2.close()


def test_read_from_file():
    out = file_reader(file_from_disk)


def test_read_from_export():
    out = file_reader(file_from_export)
