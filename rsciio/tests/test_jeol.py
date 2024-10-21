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

import gc
import zipfile
from pathlib import Path

import numpy as np
import pytest

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")


def teardown_module(module):
    """
    Run a garbage collection cycle at the end of the test of this module
    to avoid any memory issue when continuing running the test suite.
    """
    gc.collect()


TESTS_FILE_PATH = Path(__file__).resolve().parent / "data" / "jeol"
TESTS_FILE_PATH2 = TESTS_FILE_PATH / "InvalidFrame"

TEST_FILES = [
    "rawdata.ASW",
    "View000_0000000.img",
    "View000_0000001.map",
    "View000_0000002.map",
    "View000_0000003.map",
    "View000_0000004.map",
    "View000_0000005.map",
    "View000_0000006.pts",
]

TEST_FILES2 = [
    "dummy2.ASW",
    "Dummy-Data_0000000.img",
    "Dummy-Data_0000001.map",
    "Dummy-Data_0000002.map",
    "Dummy-Data_0000003.map",
    "Dummy-Data_0000004.map",
    "Dummy-Data_0000005.map",
    "Dummy-Data_0000006.map",
    "Dummy-Data_0000007.pts",
    "Dummy-Data_0000008.apb",
    "Dummy-Data_0000009.map",
    "Dummy-Data_0000010.map",
    "Dummy-Data_0000011.map",
    "Dummy-Data_0000012.map",
    "Dummy-Data_0000013.map",
    "Dummy-Data_0000014.map",
    "Dummy-Data_0000015.pts",
    "Dummy-Data_0000016.apb",
    "Dummy-Data_0000017.map",
    "Dummy-Data_0000018.map",
    "Dummy-Data_0000019.map",
    "Dummy-Data_0000020.map",
    "Dummy-Data_0000021.map",
    "Dummy-Data_0000022.map",
    "Dummy-Data_0000023.pts",
    "Dummy-Data_0000024.APB",
]


def test_load_project():
    pytest.importorskip("numba")
    # test load all elements of the project rawdata.ASW
    filename = TESTS_FILE_PATH / TEST_FILES[0]
    s = hs.load(filename, reader="JEOL")
    # first file is always a 16bit image of the work area
    assert s[0].data.dtype == np.uint8
    assert s[0].data.shape == (512, 512)
    assert s[0].axes_manager.signal_dimension == 2
    assert s[0].axes_manager[0].units == "µm"
    assert s[0].axes_manager[0].name == "x"
    assert s[0].axes_manager[1].units == "µm"
    assert s[0].axes_manager[1].name == "y"
    # 1 to 16 files are a 16bit image of work area and elemental maps
    for elmap in s[:-1]:
        assert elmap.data.dtype == np.uint8
        assert elmap.data.shape == (512, 512)
        assert elmap.axes_manager.signal_dimension == 2
        assert elmap.axes_manager[0].units == "µm"
        assert elmap.axes_manager[0].name == "x"
        assert elmap.axes_manager[1].units == "µm"
        assert elmap.axes_manager[1].name == "y"
    # last file is the datacube
    assert s[-1].data.dtype == np.uint8
    assert s[-1].data.shape == (512, 512, 4096)
    assert s[-1].axes_manager.signal_dimension == 1
    assert s[-1].axes_manager.navigation_dimension == 2
    assert s[-1].axes_manager[0].units == "µm"
    assert s[-1].axes_manager[0].name == "x"
    assert s[-1].axes_manager[1].units == "µm"
    assert s[-1].axes_manager[1].name == "y"
    assert s[-1].axes_manager[2].units == "keV"
    np.testing.assert_allclose(
        s[-1].axes_manager[2].offset, -0.000789965 - 0.00999866 * 96
    )
    np.testing.assert_allclose(s[-1].axes_manager[2].scale, 0.00999866)
    assert s[-1].axes_manager[2].name == "Energy"

    # check scale (image)
    filename = TESTS_FILE_PATH / "Sample" / "00_View000" / TEST_FILES[1]
    s1 = hs.load(filename, reader="JEOL")
    np.testing.assert_allclose(s[0].axes_manager[0].scale, s1.axes_manager[0].scale)
    assert s[0].axes_manager[0].units == s1.axes_manager[0].units
    # check scale (pts)
    filename = TESTS_FILE_PATH / "Sample" / "00_View000" / TEST_FILES[7]
    s2 = hs.load(filename, reader="JEOL")
    np.testing.assert_allclose(s[6].axes_manager[0].scale, s2.axes_manager[0].scale)
    assert s[6].axes_manager[0].units == s2.axes_manager[0].units


def test_load_image():
    # test load work area haadf image
    filename = TESTS_FILE_PATH / "Sample" / "00_View000" / TEST_FILES[1]
    s = hs.load(filename, reader="JEOL")
    assert s.data.dtype == np.uint8
    assert s.data.shape == (512, 512)
    assert s.axes_manager.signal_dimension == 2
    assert s.axes_manager[0].units == "µm"
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.00869140587747097)
    assert s.axes_manager[0].name == "x"
    assert s.axes_manager[1].units == "µm"
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.00869140587747097)
    assert s.axes_manager[1].name == "y"


@pytest.mark.parametrize("SI_dtype", [np.int8, np.uint8])
def test_load_datacube(SI_dtype):
    pytest.importorskip("numba")
    # test load eds datacube
    filename = TESTS_FILE_PATH / "Sample" / "00_View000" / TEST_FILES[7]
    s = hs.load(filename, SI_dtype=SI_dtype, cutoff_at_kV=5, reader="JEOL")
    assert s.data.dtype == SI_dtype
    assert s.data.shape == (512, 512, 596)
    assert s.axes_manager.signal_dimension == 1
    assert s.axes_manager.navigation_dimension == 2
    assert s.axes_manager[0].units == "µm"
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.00869140587747097)
    assert s.axes_manager[0].name == "x"
    assert s.axes_manager[1].units == "µm"
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.00869140587747097)
    assert s.axes_manager[1].name == "y"
    assert s.axes_manager[2].units == "keV"
    np.testing.assert_allclose(s.axes_manager[2].offset, -0.000789965 - 0.00999866 * 96)
    np.testing.assert_allclose(s.axes_manager[2].scale, 0.00999866)
    assert s.axes_manager[2].name == "Energy"


def test_load_datacube_rebin_energy():
    pytest.importorskip("numba")
    filename = TESTS_FILE_PATH / "Sample" / "00_View000" / TEST_FILES[7]
    s = hs.load(filename, cutoff_at_kV=0.1, reader="JEOL")
    s_sum = s.sum()

    ref_data = hs.signals.Signal1D(np.array([3, 23, 77, 200, 487, 984, 1599, 2391]))
    np.testing.assert_allclose(s_sum.data[88:96], ref_data.data)

    rebin_energy = 8
    s2 = hs.load(filename, rebin_energy=rebin_energy, reader="JEOL")
    s2_sum = s2.sum()

    np.testing.assert_allclose(s2_sum.data[11:12], ref_data.data.sum())

    with pytest.raises(ValueError, match="must be a divisor"):
        _ = hs.load(filename, rebin_energy=10, reader="JEOL")


def test_load_datacube_cutoff_at_kV():
    pytest.importorskip("numba")
    gc.collect()
    cutoff_at_kV = 10.0
    filename = TESTS_FILE_PATH / "Sample" / "00_View000" / TEST_FILES[7]
    s = hs.load(filename, cutoff_at_kV=None, reader="JEOL")
    s2 = hs.load(filename, cutoff_at_kV=cutoff_at_kV, reader="JEOL")

    assert s2.axes_manager[-1].size == 1096
    np.testing.assert_allclose(s2.axes_manager[2].scale, 0.00999866)
    np.testing.assert_allclose(s2.axes_manager[2].offset, -0.9606613)

    np.testing.assert_allclose(s.sum().isig[:cutoff_at_kV].data, s2.sum().data)


def test_load_datacube_downsample():
    pytest.importorskip("numba")
    downsample = 8
    filename = TESTS_FILE_PATH / TEST_FILES[0]
    s = hs.load(filename, downsample=1, reader="JEOL")[-1]
    s2 = hs.load(filename, downsample=downsample, reader="JEOL")[-1]

    s_sum = s.sum(-1).rebin(scale=(downsample, downsample))
    s2_sum = s2.sum(-1)

    assert s2.axes_manager[-1].size == 4096
    np.testing.assert_allclose(s2.axes_manager[2].scale, 0.00999866)
    np.testing.assert_allclose(s2.axes_manager[2].offset, -0.9606613)

    for axis in s2.axes_manager.navigation_axes:
        assert axis.size == 64
        np.testing.assert_allclose(axis.scale, 0.069531247)
        np.testing.assert_allclose(axis.offset, 0.0)

    np.testing.assert_allclose(s_sum.data, s2_sum.data)

    with pytest.raises(ValueError, match="must be a divisor"):
        _ = hs.load(filename, downsample=10, reader="JEOL")[-1]

    with pytest.raises(
        ValueError,
        match="`downsample` can't be an iterable of length different from 2.",
    ):
        _ = hs.load(filename, downsample=[2, 2, 2], reader="JEOL")[-1]

    downsample = [8, 16]
    s = hs.load(filename, downsample=downsample, reader="JEOL")[-1]
    assert s.axes_manager["x"].size * downsample[0] == 512
    assert s.axes_manager["y"].size * downsample[1] == 512

    with pytest.raises(ValueError, match="must be a divisor"):
        _ = hs.load(filename, downsample=[256, 100], reader="JEOL")[-1]

    with pytest.raises(ValueError, match="must be a divisor"):
        _ = hs.load(filename, downsample=[100, 256], reader="JEOL")[-1]


def test_load_datacube_frames():
    pytest.importorskip("numba")
    rebin_energy = 2048
    filename = TESTS_FILE_PATH / "Sample" / "00_View000" / TEST_FILES[7]
    s = hs.load(filename, sum_frames=True, rebin_energy=rebin_energy, reader="JEOL")
    assert s.data.shape == (512, 512, 2)
    s_frame = hs.load(
        filename, sum_frames=False, rebin_energy=rebin_energy, reader="JEOL"
    )
    assert s_frame.data.shape == (14, 512, 512, 2)
    np.testing.assert_allclose(s_frame.sum(axis="Frame").data, s.data)
    np.testing.assert_allclose(
        s_frame.sum(axis=["x", "y", "Energy"]).data,
        np.array(
            [
                22355,
                21975,
                22038,
                21904,
                21846,
                22115,
                22021,
                21917,
                22123,
                21919,
                22141,
                22024,
                22086,
                21797,
            ]
        ),
    )


@pytest.mark.parametrize("filename_as_string", [True, False])
def test_load_eds_file(filename_as_string):
    pytest.importorskip("numba")
    pytest.importorskip("exspy", reason="exspy not installed.")
    filename = TESTS_FILE_PATH / "met03.EDS"
    if filename_as_string:
        filename = str(filename)
    s = hs.load(filename, reader="JEOL")
    assert s.metadata.Signal.signal_type == "EDS_TEM"
    assert isinstance(s, hs.signals.Signal1D)
    assert s.data.shape == (2048,)
    axis = s.axes_manager[0]
    assert axis.name == "Energy"
    assert axis.size == 2048
    assert axis.offset == -0.00176612
    assert axis.scale == 0.0100004

    # delete timestamp from metadata since it's runtime dependent
    del s.metadata.General.FileIO.Number_0.timestamp

    md_dict = s.metadata.as_dictionary()
    assert md_dict["General"] == {
        "original_filename": "met03.EDS",
        "time": "14:14:51",
        "date": "2018-06-25",
        "title": "EDX",
        "FileIO": {
            "0": {
                "operation": "load",
                "hyperspy_version": hs.__version__,
                "io_plugin": "rsciio.jeol",
            }
        },
    }
    TEM_dict = md_dict["Acquisition_instrument"]["TEM"]
    assert TEM_dict == {
        "beam_energy": 200.0,
        "Detector": {
            "EDS": {
                "azimuth_angle": 90.0,
                "detector_type": "EX24075JGT",
                "elevation_angle": 22.299999237060547,
                "energy_resolution_MnKa": 138.0,
                "live_time": 30.0,
            }
        },
        "Stage": {"tilt_alpha": 0.0},
    }


def test_shift_jis_encoding():
    # See https://github.com/hyperspy/hyperspy/issues/2812
    filename = TESTS_FILE_PATH / "181019-BN.ASW"
    # make sure we can open the file
    with open(filename, "br"):
        pass
    try:
        _ = hs.load(filename, reader="JEOL")
    except FileNotFoundError:
        # we don't have the other files required to open the data
        pass


def test_number_of_frames():
    pytest.importorskip("numba")
    dir1 = TESTS_FILE_PATH / "Sample" / "00_View000"
    dir2 = TESTS_FILE_PATH / "InvalidFrame" / "Sample" / "00_Dummy-Data"

    test_list = [  # dir, file, num_frames, num_valid_frames
        [dir1, TEST_FILES[7], 14, 14],
        [dir2, TEST_FILES2[8], 1, 0],
        [dir2, TEST_FILES2[16], 2, 1],
        [dir2, TEST_FILES2[24], 1, 1],
    ]

    for item in test_list:
        dirname, filename, frames, valid = item
        fname = str(dirname / filename)

        # Count number of frames including incomplete frame
        data = hs.load(
            fname,
            sum_frames=False,
            only_valid_data=False,
            downsample=[32, 32],
            rebin_energy=512,
            SI_dtype=np.int32,
            reader="JEOL",
        )
        assert data.axes_manager["Frame"].size == frames

        # Count number of valid frames
        data = hs.load(
            fname,
            sum_frames=False,
            only_valid_data=True,
            downsample=[32, 32],
            rebin_energy=512,
            SI_dtype=np.int32,
            reader="JEOL",
        )
        assert data.axes_manager["Frame"].size == valid


def test_em_image_in_pts():
    pytest.importorskip("numba")
    dir1 = TESTS_FILE_PATH
    dir2 = TESTS_FILE_PATH / "InvalidFrame"
    dir2p = dir2 / "Sample" / "00_Dummy-Data"

    # no SEM/STEM image
    s = hs.load(
        dir1 / TEST_FILES[0],
        read_em_image=False,
        only_valid_data=False,
        cutoff_at_kV=1,
        reader="JEOL",
    )
    assert len(s) == 7

    s = hs.load(
        dir1 / TEST_FILES[0],
        read_em_image=True,
        only_valid_data=False,
        cutoff_at_kV=1,
        reader="JEOL",
    )
    assert len(s) == 7

    # with SEM/STEM image
    s = hs.load(
        dir2 / TEST_FILES2[0],
        read_em_image=False,
        only_valid_data=False,
        cutoff_at_kV=1,
        reader="JEOL",
    )
    assert len(s) == 22
    s = hs.load(
        dir2 / TEST_FILES2[0],
        read_em_image=True,
        only_valid_data=False,
        cutoff_at_kV=1,
        reader="JEOL",
    )
    assert len(s) == 25
    assert (
        s[8].metadata.General.title
        == "S(T)EM Image extracted from " + s[8].metadata.General.original_filename
    )
    assert s[8].data[38, 15] == 87
    assert s[8].data[38, 16] == 0

    # integrate SEM/STEM image along frame axis
    s = hs.load(
        dir2p / TEST_FILES2[16],
        read_em_image=True,
        only_valid_data=False,
        sum_frames=True,
        cutoff_at_kV=1,
        frame_list=[0, 0, 0, 1],
        reader="JEOL",
    )
    assert s[1].data[0, 0] == 87 * 4
    assert s[1].data[63, 63] == 87 * 3

    s = hs.load(
        dir2p / TEST_FILES2[16],
        read_em_image=True,
        only_valid_data=False,
        sum_frames=False,
        cutoff_at_kV=1,
        reader="JEOL",
    )
    s2 = hs.load(
        dir2p / TEST_FILES2[16],
        read_em_image=True,
        only_valid_data=False,
        sum_frames=True,
        cutoff_at_kV=1,
        reader="JEOL",
    )
    s1 = [s[0].data.sum(axis=0), s[1].data.sum(axis=0)]
    assert np.array_equal(s1[0], s2[0].data)
    assert np.array_equal(s1[1], s2[1].data)


def test_pts_lazy():
    pytest.importorskip("sparse")
    dir2 = TESTS_FILE_PATH / "InvalidFrame"
    dir2p = dir2 / "Sample" / "00_Dummy-Data"
    s = hs.load(
        dir2p / TEST_FILES2[16],
        read_em_image=True,
        only_valid_data=False,
        sum_frames=False,
        lazy=True,
        reader="JEOL",
    )
    s1 = [s[0].data.sum(axis=0).compute(), s[1].data.sum(axis=0).compute()]
    s2 = hs.load(
        dir2p / TEST_FILES2[16],
        read_em_image=True,
        only_valid_data=False,
        sum_frames=True,
        lazy=False,
        reader="JEOL",
    )
    assert np.array_equal(s1[0], s2[0].data)
    assert np.array_equal(s1[1], s2[1].data)


def test_pts_frame_shift():
    pytest.importorskip("sparse")
    file = TESTS_FILE_PATH2 / "Sample" / "00_Dummy-Data" / TEST_FILES2[16]

    # without frame shift
    ref = hs.load(
        file,
        read_em_image=True,
        only_valid_data=False,
        sum_frames=False,
        lazy=False,
        reader="JEOL",
    )
    #         x, y, en
    points = [[24, 23, 106], [21, 16, 106]]
    values = [3, 1]
    targets = np.asarray([[2, 3, 106], [20, 3, 100], [4, 20, 100]], dtype=np.int16)

    # check values before shift
    d0 = np.zeros(len(points), dtype=np.int16)
    d1 = np.zeros(len(points), dtype=np.int16)
    d2 = np.zeros(len(points), dtype=np.int16)
    for frame, p in enumerate(points):
        d0[frame] = ref[0].data[frame, p[1], p[0], p[2]]
        assert d0[frame] == values[frame]

    for target in targets:
        sfts = np.zeros((ref[0].axes_manager["Frame"].size, 3), dtype=np.int16)
        for frame in range(ref[0].axes_manager["Frame"].size):
            origin = points[frame]
            sfts[frame] = np.asarray(target) - np.asarray(origin)
        shifts = sfts[:, [1, 0, 2]]

        # test frame shifts for dense (normal) loading
        s0 = hs.load(
            file,
            read_em_image=True,
            only_valid_data=False,
            sum_frames=False,
            frame_shifts=shifts,
            lazy=False,
            reader="JEOL",
        )

        for frame in range(s0[0].axes_manager["Frame"].size):
            origin = points[frame]
            sfts0 = s0[0].original_metadata.jeol_pts_frame_shifts[frame]
            pos = [origin[0] + sfts0[1], origin[1] + sfts0[0], origin[2] + sfts0[2]]
            d1[frame] = s0[0].data[frame, pos[1], pos[0], pos[2]]
            assert d1[frame] == d0[frame]

        # test frame shifts for lazy loading
        s1 = hs.load(
            file,
            read_em_image=True,
            only_valid_data=False,
            sum_frames=False,
            frame_shifts=shifts,
            lazy=True,
            reader="JEOL",
        )
        dt = s1[0].data.compute()
        for frame in range(s0[0].axes_manager["Frame"].size):
            origin = points[frame]
            sfts0 = s0[0].original_metadata.jeol_pts_frame_shifts[frame]
            pos = [origin[0] + sfts0[1], origin[1] + sfts0[0], origin[2] + sfts0[2]]
            d2[frame] = dt[frame, pos[1], pos[0], pos[2]]
            assert d2[frame] == d0[frame]

    # test frame shift with default values (no energy shift)
    sfts = np.array([[1, 2], [10, 3]])
    max_sfts = sfts.max(axis=0)
    min_sfts = sfts.min(axis=0)
    fs = sfts - max_sfts
    s = hs.load(
        file, frame_shifts=sfts, sum_frames=False, only_valid_data=False, reader="JEOL"
    )
    sz = min_sfts - max_sfts + ref[0].data.shape[1:3]
    assert s.data.shape == (2, sz[0], sz[1], 4096)
    for fr, sft in enumerate(fs):
        assert np.array_equal(
            s.data[fr, 20 + sft[0] : 30 + sft[0], 20 + sft[1] : 30 + sft[1], 106],
            ref[0].data[fr, 20:30, 20:30, 106],
        )


def test_broken_files(tmp_path):
    pytest.importorskip("numba")
    TEST_BROKEN_FILES = ["test.asw", "test.pts", "test.img"]
    for _file in TEST_BROKEN_FILES:
        file = tmp_path / _file
        with open(file, "w") as fd:
            fd.write("aaaaaaaa")
        if file.suffix == ".asw":
            # in case of asw, valid data can not be obtained
            with pytest.raises(ValueError, match="Not a valid JEOL asw format"):
                _ = hs.load(file, reader="JEOL")
        else:
            # just skipping broken files
            s = hs.load(file, reader="JEOL")
            assert s == []


def test_seq_eds_files(tmp_path):
    pos0 = [0.0, 0.0, -0.000132, 0.000132]
    pos = [
        [0.0, 0.0, 0.0, 0.0],
        [2.04070450e-05, -4.77886497e-05, 1.05909980e-05, -3.87475538e-05],
        [1.91154599e-05, -3.07397260e-05, -5.45048924e-05, 5.16634051e-05],
    ]
    memo = ["", "030", "035"]
    test_file = TESTS_FILE_PATH / "jeol_seq_eds_files.zip"

    with zipfile.ZipFile(test_file, "r") as zipped:
        zipped.extractall(tmp_path)

    # test reading sequential acuired EDS spectrum
    s = hs.load(tmp_path / "1" / "1.ASW", reader="JEOL")
    # check if three subfiles are in file (img, eds, eds)
    assert len(s) == 3
    # check positional information in subfiles
    for i, p in enumerate(pos):
        sampleinfo = s[i].original_metadata["asw"]["SampleInfo"]["0"]
        viewinfo = sampleinfo["ViewInfo"]["0"]
        np.testing.assert_allclose(viewinfo["PositionMM2"], pos0)
        viewdata_asw = viewinfo["ViewData"]
        viewdata = s[i].original_metadata["asw_viewdata"]
        np.testing.assert_allclose(viewdata["PositionMM2"], p)
        np.testing.assert_allclose(
            viewdata["PositionMM2"], viewdata_asw[i]["PositionMM2"]
        )
        assert viewdata["Memo"] == memo[i]
    for s_ in s[1:3]:
        assert s_.metadata.Signal.signal_type == "EDS_TEM"
        assert isinstance(s_, hs.signals.Signal1D)

    # test with broken asw file
    fname = tmp_path / "1" / "1.ASW"
    fname2 = tmp_path / "1" / "2.ASW"
    with open(fname, "rb") as f:
        data = bytearray(f.read())

    # No ViewData
    data2 = data.copy()
    data2[0x42D] = 0x30
    with open(fname2, "wb") as f:
        f.write(data2)
    dat = hs.load(fname2, reader="JEOL")
    assert len(dat) == 0

    # No ViewInfo
    data2 = data.copy()
    data2[0x1AD] = 0x30
    with open(fname2, "wb") as f:
        f.write(data2)
    dat = hs.load(fname2, reader="JEOL")
    assert len(dat) == 0

    # No SampleInfo
    data2 = data.copy()
    data2[0x6E] = 0x30
    with open(fname2, "wb") as f:
        f.write(data2)
    dat = hs.load(fname2, reader="JEOL")
    assert len(dat) == 0

    # test read for pseudo SEM eds/img data
    sub_dir = tmp_path / "1" / "Sample" / "00_View002"
    test_files = ["View002_0000000.img", "View002_0000001.eds"]

    # rewrite AccV  200 kV to 20 kV to generate pseudo SEM data
    # .img
    with open(sub_dir / test_files[0], "rb") as f:
        data = bytearray(f.read())
        data[0x75BC] = 0xA0
        data[0x75BD] = 0x41
    with open(sub_dir / ("x" + test_files[0]), "wb") as f:
        f.write(data)
    s = hs.load(sub_dir / ("x" + test_files[0]), reader="JEOL")
    assert "SEM" in s.metadata["Acquisition_instrument"]

    # .eds
    with open(sub_dir / test_files[1], "rb") as f:
        data = bytearray(f.read())
        data[0x4B13] = 0x34
    with open(sub_dir / ("x" + test_files[1]), "wb") as f:
        f.write(data)
    s = hs.load(sub_dir / ("x" + test_files[1]), reader="JEOL")
    assert s.metadata.Signal.signal_type == "EDS_SEM"
    assert isinstance(s, hs.signals.Signal1D)
    assert "SEM" in s.metadata["Acquisition_instrument"]


def test_frame_start_index(tmp_path):
    pytest.importorskip("numba")
    file = TESTS_FILE_PATH / "Sample" / "00_View000" / TEST_FILES[7]
    frame_start_index_ref = [
        0,
        49660,
        98602,
        147633,
        196414,
        245078,
        294263,
        343283,
        392081,
        441310,
        490126,
        539395,
        588409,
        637523,
        686084,
    ]

    ref = hs.load(
        file,
        sum_frames=False,
        downsample=[32, 32],
        rebin_energy=512,
        SI_dtype=np.int32,
        reader="JEOL",
    )
    frame_start_index = ref.original_metadata.jeol_pts_frame_start_index
    assert np.array_equal(frame_start_index, frame_start_index_ref)

    s = hs.load(
        file,
        frame_list=[2, 5],
        downsample=[32, 32],
        rebin_energy=512,
        SI_dtype=np.int32,
        reader="JEOL",
    )
    frame_start_index = s.original_metadata.jeol_pts_frame_start_index
    assert np.array_equal(frame_start_index[0:6], frame_start_index_ref[0:6])
    assert np.all(frame_start_index[6:] == -1)

    s = hs.load(
        file,
        frame_list=[4, 9],
        frame_start_index=frame_start_index,
        downsample=[32, 32],
        rebin_energy=512,
        SI_dtype=np.int32,
        reader="JEOL",
    )
    frame_start_index = s.original_metadata.jeol_pts_frame_start_index
    assert np.array_equal(frame_start_index[0:10], frame_start_index_ref[0:10])
    assert np.all(frame_start_index[10:] == -1)

    s = hs.load(
        file,
        frame_list=[11, 5, 20],
        sum_frames=False,
        frame_start_index=frame_start_index,
        downsample=[32, 32],
        rebin_energy=512,
        SI_dtype=np.int32,
        reader="JEOL",
    )
    assert s.data.shape == (2, 16, 16, 8)

    # test with pseudo "SEM" data
    test_file = tmp_path / "test.pts"
    with open(file, "rb") as f:
        data = bytearray(f.read())
        # AcckV = 20 kV
        data[0x1116] = 0xA0
        data[0x1117] = 0x41
    with open(test_file, "wb") as f:
        f.write(data)
        s = hs.load(
            test_file,
            downsample=[32, 32],
            rebin_energy=512,
            SI_dtype=np.int32,
            reader="JEOL",
        )
    assert s.metadata["Signal"]["signal_type"] == "EDS_SEM"
