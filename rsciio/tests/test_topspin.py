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

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import h5py
import pytest

from rsciio.topspin._api import _parse_app5_xml, file_reader

h5py = pytest.importorskip("h5py", reason="h5py not installed")

# Notes to self: add check that all axes values are expected type

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
    assert out_A[0]["data"].shape == (11, 13)
    assert out_A[1]["data"].shape == (2, 5, 16, 16)
    assert out_B[0]["data"].shape == (3, 7, 37, 37)
    assert out_B[1]["data"].shape == (11, 13)
    assert out_B[2]["data"].shape == (11, 13)
    assert out_B[3]["data"].shape == (11, 13)
    assert out_C[0]["data"].shape == (11, 13)
    assert out_C[1]["data"].shape == (3, 5, 16, 16)
    assert out_C[2]["data"].shape == (11, 13)
    # TODO: Verify the Metadata is correct
    # TODO: Verify the axes are correct


# TODO: load one or all of these into hyperspy and make sure they work
# def TODO_hs_test_function():
#     hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
#     return hs


def test_dryrun():
    correct_sizes = [
        ["11, 13", "2, 5, 16, 16"],
        ["3, 7, 37, 37", "11, 13", "11, 13", "11, 13"],
        ["11, 13", "3, 5, 16, 16", "11, 13"],
    ]
    for i, f in enumerate([file_A, file_B, file_C]):
        buffer = StringIO()
        with redirect_stdout(buffer):
            out = file_reader(f, dryrun=True)
        assert out == []
        txt = buffer.getvalue()
        # make sure the output is correctly estimating the object shapes
        dims_str = [x.split("]")[0] for x in txt.split("[")[1:]]
        assert dims_str == correct_sizes[i]
