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

import xml.etree.cElementTree as ET
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py not installed")

from rsciio.topspin._api import _parse_app5_xml, file_reader  # noqa: E402

# locations for test data, both in path and str format
data_directory = Path(__file__).parent / "data" / "topspin"
file_A = data_directory / "topspin_test_A.app5"
file_B = data_directory / "topspin_test_B.app5"
file_C = data_directory / "topspin_test_C.app5"
file_Cstr = str(file_C)


def test_xml_parser():
    # Explicitly test the metadata file reader before testing the loader.
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
    # Test failed read warning
    f1 = h5py.File(file_A, "r")
    root = ET.fromstring(f1["Metadata"][()].decode())
    f1.close()
    root[16][4][1].attrib["Serializer"] = "aaa"
    _parse_app5_xml(ET.tostring(root))


def test_file_reader():
    out_A = file_reader(file_A, show_progressbar=False)
    out_B = file_reader(file_B, show_progressbar=False)
    out_C = file_reader(file_C, show_progressbar=False)
    # Test subset reader for single session
    sub_b1 = file_reader(
        file_B, "18d9446f-22bf-4fb1-8d13-338174e75d20", show_progressbar=False
    )
    # Test subset reader for single dataset
    sub_b1a = file_reader(
        file_B,
        "18d9446f-22bf-4fb1-8d13-338174e75d20"
        + "/3526f008-a687-41fb-a21e-c21362241492",
        show_progressbar=False,
    )
    # Check everything loaded
    assert len(out_A) == 2
    assert len(out_B) == 4
    assert len(out_C) == 3
    assert len(sub_b1) == 3
    assert len(sub_b1a) == 1

    # Check the loaded data is the expected size
    for out in [out_A, out_B, out_C]:
        for x in out:
            assert isinstance(x, dict)
            assert "axes" in x
            assert "data" in x
    assert out_A[0]["data"].shape == (8, 56)
    assert out_A[1]["data"].shape == (2, 5, 14, 12)
    assert out_B[0]["data"].shape == (3, 7, 31, 26)
    assert out_B[1]["data"].shape == (8, 6)
    assert out_B[2]["data"].shape == (83, 65)
    assert out_B[3]["data"].shape == (12, 9)
    assert out_C[0]["data"].shape == (6, 23)
    assert out_C[1]["data"].shape == (3, 5, 14, 12)
    assert out_C[2]["data"].shape == (29, 23)

    # Check identical data loaded with the subset call are identical
    assert np.all(sub_b1a[0]["data"] == sub_b1[0]["data"])
    assert np.all(sub_b1a[0]["data"] == out_B[0]["data"])

    # Check hyperspy metadata is populated for all test datasets
    for out in [out_A, out_B, out_C, sub_b1, sub_b1a]:
        for dset in out:
            md = dset["metadata"]
            assert "General" in md
            assert "title" in md["General"]
            assert isinstance(md["General"]["FileIO"], dict)
            for key in md.keys():
                assert md[key] is not None  # exists
                if isinstance(md[key], str):
                    assert len(md[key]) > 1  # has non-default data

    # Check axes
    for out in [out_A, out_B, out_C, sub_b1, sub_b1a]:
        for dset in out:
            ad_all = dset["axes"]
            for ad in ad_all:
                assert "name" in ad.keys()
                assert "units" in ad.keys()
                assert "size" in ad.keys()
                assert "scale" in ad.keys()
                assert "offset" in ad.keys()
                assert "navigate" in ad.keys()
                assert "index_in_array" in ad.keys()
                for k in ["size", "scale", "offset"]:
                    assert np.abs(ad[k]) > 0
                if ad["name"] in ["x", "y"]:
                    assert ad["navigate"]
                else:
                    assert not ad["navigate"]
            names = np.array([x["name"] for x in ad_all])
            idxs = np.array([x["index_in_array"] for x in ad_all])
            assert np.all(np.isin(np.unique(names), ["x", "y", "kx", "ky"]))
            assert np.max(np.unique(names, return_counts=True)[1]) == 1
            assert np.max(np.unique(idxs, return_counts=True)[1]) == 1


def test_dryrun():
    correct_sizes = [
        ["8, 56", "2, 5, 14, 12"],
        ["3, 7, 31, 26", "8, 6", "83, 65", "12, 9"],
        ["6, 23", "3, 5, 14, 12", "29, 23"],
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


def test_with_hyperspy():
    hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
    # test all files can be converted to hyperspy
    for f in [file_A, file_B, file_C]:
        out = file_reader(f, show_progressbar=False)
        for dset in out:
            signal = hs.signals.Signal2D(
                data=dset["data"], axes=dset["axes"], metadata=dset["metadata"]
            )
            assert isinstance(signal, hs.signals.Signal2D)
    # for file_A, test axis values are as expected
    out = file_reader(file_A, show_progressbar=False)
    s1 = hs.signals.Signal2D(
        data=out[1]["data"],
        axes=out[1]["axes"],
        metadata=out[1]["metadata"],
    )
    s2 = hs.signals.Signal2D(
        data=out[0]["data"],
        axes=out[0]["axes"],
        metadata=out[0]["metadata"],
    )
    # Test 1 SPED dataset
    assert s1.axes_manager["y"].size == 2
    assert np.around(s1.axes_manager["y"].offset) == 184
    assert s1.axes_manager["y"].units == "nm"
    assert s1.axes_manager["y"].scale == 2

    assert s1.axes_manager["x"].size == 5
    assert np.around(s1.axes_manager["x"].offset) == 173
    assert s1.axes_manager["x"].units == "nm"
    assert s1.axes_manager["x"].scale == 2

    assert s1.axes_manager["ky"].size == 14
    assert np.around(s1.axes_manager["ky"].offset * 1e6) == -2730
    assert s1.axes_manager["ky"].units == "Angle"
    assert np.around(s1.axes_manager["ky"].scale * 1e6) == 21

    assert s1.axes_manager["kx"].size == 12
    assert np.around(s1.axes_manager["kx"].offset * 1e6) == -2730
    assert s1.axes_manager["kx"].units == "Angle"
    assert np.around(s1.axes_manager["kx"].scale * 1e6) == 21

    # Test 1 STEM dataset
    assert s2.axes_manager["y"].size == 8
    assert np.around(s2.axes_manager["y"].offset) == 64
    assert s2.axes_manager["y"].units == "nm"
    assert s2.axes_manager["y"].scale == 2

    assert s2.axes_manager["x"].size == 56
    assert np.around(s2.axes_manager["x"].offset) == 163
    assert s2.axes_manager["x"].units == "nm"
    assert s2.axes_manager["x"].scale == 2
