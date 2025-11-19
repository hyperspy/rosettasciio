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
import pytest

from rsciio.topspin._api import _parse_app5_xml, file_reader

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")


data_directory = Path(__file__).parent / "data" / "topspin"
file_A = data_directory / "topspin_test_A.app5"
file_B = data_directory / "topspin_test_B.app5"
file_C = data_directory / "topspin_test_C.app5"
file_Cstr = str(file_C)


def test_xml_parser():
    # Explicitly test the metadata file reader before testing the loader.
    # this will get ran as part of the following tests, but since it is
    # the most likely thing to break, it's worth testing verbosely at
    # the start
    f1 = h5py.File(file_A, "r")  # unnested
    meta_dict = _parse_app5_xml(f1["Metadata"][()].decode())
    f1.close()
    assert len(meta_dict) == 18
    assert isinstance(meta_dict["ProcedureData"], dict)
    assert isinstance(meta_dict["Id"], str)

    f2 = h5py.File(file_B, "r")  # nested
    for grp in [x for i, x in enumerate(f2.keys()) if i in [0, 2]]:
        metadata_string = f2[grp]["Metadata"][()].decode()
        meta_dict = _parse_app5_xml(metadata_string)
        assert len(meta_dict) == 18
        assert isinstance(meta_dict["ProcedureData"], dict)
        assert isinstance(meta_dict["Id"], str)
    f2.close()


def test_file_reader():
    out_A = file_reader(file_A)
    out_B = file_reader(file_B)
    out_C = file_reader(file_C)
    assert len(out_A) == 2
    assert len(out_B) == 4
    assert len(out_C) == 3
    for out in [out_A, out_B, out_C]:
        for x in out:
            assert isinstance(x, dict)
            assert "axes" in x
            assert "data" in x
    assert out_A[0]["data"].shape == (2, 5, 32, 32)
    assert out_A[1]["data"].shape == (2, 5)
    assert out_B[0]["data"].shape == (3, 7, 29, 29)
    assert out_B[1]["data"].shape == (11, 13)
    assert out_B[2]["data"].shape == (13, 17)
    assert out_B[3]["data"].shape == (11, 13)
    assert out_C[0]["data"].shape == (3, 7)
    assert out_C[1]["data"].shape == (3, 5, 8, 8)
    assert out_C[2]["data"].shape == (13, 17)


def test_dryrun():
    out_A = file_reader(file_A, dryrun=True)
    out_B = file_reader(file_B, dryrun=True)
    out_C = file_reader(file_C, dryrun=True)
    for out in [out_A, out_B, out_C]:
        assert out == []
