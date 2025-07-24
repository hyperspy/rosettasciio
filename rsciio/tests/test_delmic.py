# -*- coding: utf-8 -*-
# Copyright 2024-2025 Delmic
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

import numpy as np
import pytest

try:
    import hyperspy.api as hs
    from hyperspy._signals.signal1d import Signal1D
    from hyperspy._signals.signal2d import Signal2D
    from hyperspy.signal import BaseSignal
except ImportError:
    pytest.skip("hyperspy not installed", allow_module_level=True)

try:
    import lumispy
except ImportError:
    lumispy = None

pytest.importorskip("h5py", reason="h5py not installed")

testfile_dir = (Path(__file__).parent / "data" / "delmic").resolve()
testfile_intensity_path = (testfile_dir / "sparc-intensity.h5").resolve()
testfile_intensity_data_survey_path = (
    testfile_dir / "sparc-intensity-data-survey.npy"
).resolve()
testfile_intensity_drift_path = (testfile_dir / "sparc-intensity-drift.h5").resolve()
testfile_hyperspectral_path = (testfile_dir / "sparc-hyperspectral.h5").resolve()
testfile_hyperspectral_data_path = (
    testfile_dir / "sparc-hyperspectral-data.npy"
).resolve()
testfile_hyperspectral_data_survey_path = (
    testfile_dir / "sparc-hyperspectral-data-survey.npy"
).resolve()
testfile_hyperspectral_wavelengths_path = (
    testfile_dir / "sparc-hyperspectral-wavelengths.npy"
).resolve()
testfile_hyperspectral_spot_path = (
    testfile_dir / "sparc-hyperspectral-spot.h5"
).resolve()
testfile_temporaltrace_path = (testfile_dir / "sparc-time-correlator.h5").resolve()
testfile_temporaltrace_data_path = (
    testfile_dir / "sparc-time-correlator-data.npy"
).resolve()
testfile_temporaltrace_data_survey_path = (
    testfile_dir / "sparc-time-correlator-data-survey.npy"
).resolve()
testfile_temporaltrace_timelist_path = (
    testfile_dir / "sparc-time-correlator-timelist.npy"
).resolve()
testfile_temporaltrace_spot_path = (
    testfile_dir / "sparc-time-correlator-spot.h5"
).resolve()
testfile_streakcamera_path = (testfile_dir / "sparc-streak-camera.h5").resolve()
testfile_streakcamera_data_path = (
    testfile_dir / "sparc-streak-camera-data.npy"
).resolve()
testfile_streakcamera_data_survey_path = (
    testfile_dir / "sparc-streak-camera-data-survey.npy"
).resolve()
testfile_streakcamera_timelist_path = (
    testfile_dir / "sparc-streak-camera-timelist.npy"
).resolve()
testfile_streakcamera_wavelengths_path = (
    testfile_dir / "sparc-streak-camera-wavelengths.npy"
).resolve()
testfile_streakcamera_spot_path = (
    testfile_dir / "sparc-streak-camera-spot.h5"
).resolve()
testfile_ek_path = (testfile_dir / "sparc-e-k.h5").resolve()
testfile_ek_data_path = (testfile_dir / "sparc-e-k-data.npy").resolve()
testfile_ek_data_survey_path = (testfile_dir / "sparc-e-k-data-survey.npy").resolve()
testfile_ek_channels_path = (testfile_dir / "sparc-e-k-channels.npy").resolve()
testfile_ek_wavelengths_path = (testfile_dir / "sparc-e-k-wavelengths.npy").resolve()
testfile_ek_spot_path = (testfile_dir / "sparc-e-k-spot.h5").resolve()
testfile_AR_path = (testfile_dir / "sparc-angle-resolved.h5").resolve()
testfile_AR_data_path = (testfile_dir / "sparc-angle-resolved-data.npy").resolve()
testfile_AR_data_survey_path = (
    testfile_dir / "sparc-angle-resolved-data-survey.npy"
).resolve()
testfile_AR_angles_path = (testfile_dir / "sparc-angle-resolved-angles.npy").resolve()
testfile_AR_channels_path = (
    testfile_dir / "sparc-angle-resolved-channels.npy"
).resolve()
testfile_ar_pol_spot_path = (
    testfile_dir / "sparc-angle-resolved-pol-spot.h5"
).resolve()

testfile_hspy_path = (
    Path(__file__).parent / "data" / "hspy" / "example1_v1.2.hdf5"
).resolve()
testfile_arina_path = (
    Path(__file__).parent / "data" / "arina" / "test_00.h5"
).resolve()


# Intensity dataset
def test_read_data_intensity():
    """Test reading data for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path, reader="Delmic", signal="all")

    assert len(s) == 3
    assert s[0].metadata.General.title == "CL intensity"
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[-1].metadata.General.title == "Secondary electrons survey"


def test_read_data_intensity_CL():
    """Test reading data for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path, reader="Delmic")  # Default signal is "cl"
    assert isinstance(s, BaseSignal)

    # Expect 2x2 pixels
    x = np.array([27147, 28907])
    y = np.array([26964, 27695])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate

    np.testing.assert_allclose(s.axes_manager[0].scale, 200)
    np.testing.assert_allclose(s.axes_manager[0].offset, -100)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    # test Y axis
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate

    np.testing.assert_allclose(s.axes_manager[1].scale, -200)
    np.testing.assert_allclose(s.axes_manager[1].offset, 100)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)

    # Metadata
    assert s.metadata["Signal"]["quantity"] == "Counts"

    # Original metadata
    np.testing.assert_almost_equal(s.original_metadata.Magnification, 555555.55555555)
    assert s.original_metadata.SVIData.Company == "Delmic"


def test_read_data_intensity_SE():
    """Test reading data for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path, reader="Delmic", signal="se")
    assert isinstance(s, BaseSignal)

    x = np.array([27308, 28592])
    y = np.array([27851, 27958])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate

    np.testing.assert_allclose(s.axes_manager[0].scale, 200)
    np.testing.assert_allclose(s.axes_manager[0].offset, -100)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    # test Y axis
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate

    np.testing.assert_allclose(s.axes_manager[1].scale, -200)
    np.testing.assert_allclose(s.axes_manager[1].offset, 100)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)

    # Metadata
    assert s.metadata["Signal"]["quantity"] == "Counts"

    # Original metadata
    np.testing.assert_almost_equal(s.original_metadata.Magnification, 555555.55555555)


def test_read_data_intensity_survey():
    """Test reading data for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path, reader="Delmic", signal="survey")
    assert isinstance(s, BaseSignal)

    # 256x256 px
    data = np.load(testfile_intensity_data_survey_path)
    np.testing.assert_allclose(s.data, data)

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate

    np.testing.assert_allclose(s.axes_manager[0].scale, 1.7578125)
    np.testing.assert_allclose(s.axes_manager[0].offset, -224.121094)
    np.testing.assert_allclose(s.axes_manager[0].size, 256)

    # test Y axis
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate

    np.testing.assert_allclose(s.axes_manager[1].scale, -1.7578125)
    np.testing.assert_allclose(s.axes_manager[1].offset, 224.121094)
    np.testing.assert_allclose(s.axes_manager[1].size, 256)

    assert s.metadata["Signal"]["quantity"] == "Counts"

    np.testing.assert_almost_equal(s.original_metadata.Magnification, 555555.55555555)


def test_read_data_intensity_drift():
    """Test reading data for a CL intensity dataset with drift correction."""
    s = hs.load(testfile_intensity_drift_path, reader="Delmic", signal="all")

    assert len(s) == 4
    assert s[0].metadata.General.title == "CL intensity"
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[2].metadata.General.title == "Anchor region"
    assert s[3].metadata.General.title == "Secondary electrons survey"

    anchor_data = s[2]
    assert anchor_data.data.shape == (134, 122, 4)


# Hyperspectral dataset
def test_read_data_hyperspectral():
    """Test reading data for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic", signal="all")
    assert len(s) == 3
    assert s[0].metadata.General.title.startswith("Spectrum")
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[2].metadata.General.title == "Secondary electrons survey"


def test_read_data_hyperspectral_spot():
    """Test reading data for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_spot_path, reader="Delmic", signal="all")
    assert len(s) == 3
    assert s[0].metadata.General.title.startswith("Spectrum")
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[2].metadata.General.title == "Secondary electrons survey"


def test_read_data_hyperspectral_CL():
    """Test reading data for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic", signal="cl")
    if lumispy:
        assert isinstance(s, lumispy.signals.CLSEMSpectrum)
    else:
        assert isinstance(s, Signal1D)

    data = np.load(testfile_hyperspectral_data_path)
    np.testing.assert_allclose(s.data, data)

    # Axes
    ref = np.load(testfile_hyperspectral_wavelengths_path)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate

    np.testing.assert_allclose(s.axes_manager[0].scale, 963.862901465797)
    np.testing.assert_allclose(s.axes_manager[0].offset, -893.135842)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -963.862901465797)
    np.testing.assert_allclose(s.axes_manager[1].offset, 1326.3523814)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    assert s.axes_manager[2].name == "Wavelength"
    assert s.axes_manager[2].units == "nm"
    assert not s.axes_manager[2].navigate
    np.testing.assert_allclose(s.axes_manager[2].axis, ref)

    # Metadata
    assert s.metadata["Signal"]["quantity"] == "Counts"

    # Original metadata
    assert isinstance(s.original_metadata.AcquisitionDate, float)


def test_read_data_hyperspectral_spot_CL():
    """Test reading data for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_spot_path, reader="Delmic", signal="cl")

    np.testing.assert_allclose(s.data.shape, 335)

    assert s.axes_manager[0].name == "Wavelength"
    assert s.axes_manager[0].units == "nm"
    assert not s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].size, 335)


def test_read_data_hyperspectral_SE():
    """Test reading data for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic", signal="se")
    assert isinstance(s, BaseSignal)

    x = np.array([27265, 27598])
    y = np.array([27892, 28124])
    z = np.array([28299, 28468])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)
    np.testing.assert_allclose(s.data[2], z)

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 963.862901465797)
    np.testing.assert_allclose(s.axes_manager[0].offset, -893.135842)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -963.862901465797)
    np.testing.assert_allclose(s.axes_manager[1].offset, 1326.3523814)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    # Metadata
    assert s.metadata["Signal"]["quantity"] == "Counts"

    # Original metadata
    assert s.original_metadata.Magnification == 10000.0


def test_read_data_hyperspectral_survey():
    """Test reading data for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic", signal="survey")
    assert isinstance(s, BaseSignal)
    data = np.load(testfile_hyperspectral_data_survey_path)

    np.testing.assert_allclose(s.data, data)

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 27.343750000000004)
    np.testing.assert_allclose(s.axes_manager[0].offset, -6986.328125)
    np.testing.assert_allclose(s.axes_manager[0].size, 512)

    # test Y axis
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -27.343750000000004)
    np.testing.assert_allclose(s.axes_manager[1].offset, 6986.328125)
    np.testing.assert_allclose(s.axes_manager[1].size, 512)

    # Metadata
    assert s.metadata["Signal"]["quantity"] == "Counts"

    # Original metadata
    assert s.original_metadata.Magnification == 10000.0


# Time-resolved dataset
def test_read_data_temporaltrace():
    """Test reading data for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic", signal="all")
    assert len(s) == 3
    assert s[0].metadata.General.title == "Time Correlator"
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[2].metadata.General.title == "Secondary electrons survey"


def test_read_data_temporaltrace_spot():
    """Test reading data for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_spot_path, reader="Delmic", signal="all")
    assert len(s) == 3
    assert s[0].metadata.General.title == "Time Correlator"
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[2].metadata.General.title == "Secondary electrons survey"


def test_read_data_temporaltrace_CL():
    """Test reading data for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic", signal="cl")
    if lumispy:
        assert isinstance(s, lumispy.signals.LumiTransient)
    else:
        assert isinstance(s, Signal1D)

    data = np.load(testfile_temporaltrace_data_path)
    np.testing.assert_allclose(s.data, data)

    ref = np.load(testfile_temporaltrace_timelist_path)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 917.1372563963987)
    np.testing.assert_allclose(s.axes_manager[0].offset, -773.568628)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -917.1372563963987)
    np.testing.assert_allclose(s.axes_manager[1].offset, 1267.1372563964)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    assert s.axes_manager[2].name == "Time"
    assert s.axes_manager[2].units == "ns"
    assert not s.axes_manager[2].navigate
    np.testing.assert_allclose(s.axes_manager[2].axis, ref)

    assert s.metadata.General.title == "Time Correlator"
    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert isinstance(s.original_metadata.AcquisitionDate, float)


def test_read_data_temporaltrace_CL_spot():
    """Test reading data for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_spot_path, reader="Delmic", signal="cl")
    if lumispy:
        assert isinstance(s, lumispy.signals.LumiTransient)
    else:
        assert isinstance(s, Signal1D)

    np.testing.assert_allclose(s.data.shape, 65536)

    np.testing.assert_allclose(s.axes_manager[0].size, 65536)
    assert s.axes_manager[0].name == "Time"
    assert s.axes_manager[0].units == "ns"
    assert not s.axes_manager[0].navigate


def test_read_data_temporaltrace_SE():
    """Test reading data for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic", signal="se")
    assert isinstance(s, BaseSignal)

    x = np.array([32766, 32766])
    y = np.array([32766, 32766])
    z = np.array([32766, 32766])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)
    np.testing.assert_allclose(s.data[2], z)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 917.1372563963987)
    np.testing.assert_allclose(s.axes_manager[0].offset, -773.568628)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -917.1372563963987)
    np.testing.assert_allclose(s.axes_manager[1].offset, 1267.1372563964)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    assert s.metadata.General.title == "Secondary electrons concurrent"
    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert s.original_metadata.Magnification == 10000.0


def test_read_data_temporaltrace_survey():
    """Test reading data for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic", signal="survey")
    assert isinstance(s, BaseSignal)

    data = np.load(testfile_temporaltrace_data_survey_path)
    np.testing.assert_allclose(s.data, data)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 27.343750000000004)
    np.testing.assert_allclose(s.axes_manager[0].offset, -6986.328125)
    np.testing.assert_allclose(s.axes_manager[0].size, 512)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -27.343750000000004)
    np.testing.assert_allclose(s.axes_manager[1].offset, 6986.328125)
    np.testing.assert_allclose(s.axes_manager[1].size, 512)

    assert s.metadata.General.title == "Secondary electrons survey"
    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert s.original_metadata.Magnification == 10000.0


# Streak camera dataset
def test_read_data_streakcamera():
    """Test reading data for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic", signal="all")
    assert len(s) == 3
    assert s[0].metadata.General.title == "Temporal Spectrum"
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[2].metadata.General.title == "Secondary electrons survey"


def test_read_data_streakcamera_spot():
    """Test reading data for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_spot_path, reader="Delmic", signal="all")
    assert len(s) == 3
    assert s[0].metadata.General.title == "Temporal Spectrum"
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[2].metadata.General.title == "Secondary electrons survey"


def test_read_data_streakcamera_CL():
    """Test reading data for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic", signal="cl")
    if lumispy:
        assert isinstance(s, lumispy.signals.LumiTransientSpectrum)
    else:
        assert isinstance(s, Signal2D)

    data = np.load(testfile_streakcamera_data_path)
    np.testing.assert_allclose(s.data, data)

    # Axes
    ref_t = np.load(testfile_streakcamera_timelist_path)
    ref_w = np.load(testfile_streakcamera_wavelengths_path)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "µm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 3.10737984559)
    np.testing.assert_allclose(s.axes_manager[0].offset, -5.518681)
    np.testing.assert_allclose(s.axes_manager[0].size, 3)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "µm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -3.10737984559)
    np.testing.assert_allclose(s.axes_manager[1].offset, 2.12726981)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)

    assert s.axes_manager[2].name == "Wavelength"
    assert s.axes_manager[2].units == "nm"
    assert not s.axes_manager[2].navigate
    np.testing.assert_allclose(s.axes_manager[2].axis, ref_w)

    assert s.axes_manager[3].name == "Time"
    assert s.axes_manager[3].units == "ns"
    assert not s.axes_manager[3].navigate
    np.testing.assert_allclose(s.axes_manager[3].axis, ref_t)

    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert isinstance(s.original_metadata.AcquisitionDate, float)


def test_read_data_streakcamera_CL_spot():
    """Test reading data for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_spot_path, reader="Delmic", signal="cl")

    np.testing.assert_allclose(s.data.shape, 256)

    # Axes
    np.testing.assert_allclose(s.axes_manager[0].size, 256)
    assert s.axes_manager[0].name == "Wavelength"
    assert s.axes_manager[0].units == "nm"
    assert not s.axes_manager[0].navigate

    np.testing.assert_allclose(s.axes_manager[1].size, 256)
    assert s.axes_manager[1].name == "Time"
    assert s.axes_manager[1].units == "ns"
    assert not s.axes_manager[1].navigate


def test_read_data_streakcamera_SE():
    """Test reading data for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic", signal="se")
    assert isinstance(s, BaseSignal)

    x = np.array([5443, 90, 5318])
    y = np.array([241, 5256, 174])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)

    # Axes
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "µm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 3.10737984559)
    np.testing.assert_allclose(s.axes_manager[0].offset, -5.518681)
    np.testing.assert_allclose(s.axes_manager[0].size, 3)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "µm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -3.10737984559)
    np.testing.assert_allclose(s.axes_manager[1].offset, 2.12726981)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)

    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert s.original_metadata.Magnification == 10000.0


def test_read_data_streakcamera_survey():
    """Test reading data for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic", signal="survey")

    data = np.load(testfile_streakcamera_data_survey_path)
    np.testing.assert_allclose(s.data, data)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 48.828125)
    np.testing.assert_allclose(s.axes_manager[0].offset, -12475.585938)
    np.testing.assert_allclose(s.axes_manager[0].size, 512)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -48.828125)
    np.testing.assert_allclose(s.axes_manager[1].offset, 12475.585938)
    np.testing.assert_allclose(s.axes_manager[1].size, 512)

    assert s.metadata["General"]["title"] == "Secondary electrons survey"
    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert s.original_metadata.Magnification == 10000.0


# E-k dataset
def test_read_data_ek():
    """Test reading data for a CL AR Spectrum (E-k) dataset."""
    s = hs.load(testfile_ek_path, reader="Delmic", signal="all")
    assert len(s) == 3
    assert s[0].metadata.General.title == "AR Spectrum"
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[2].metadata.General.title == "Secondary electrons survey"


def test_read_data_ek_spot():
    """Test reading data for a CL AR Spectrum (E-k) dataset."""
    s = hs.load(testfile_ek_spot_path, reader="Delmic", signal="all")
    assert len(s) == 3
    assert s[0].metadata.General.title == "AR Spectrum"
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[2].metadata.General.title == "Secondary electrons survey"


def test_read_data_ek_CL():
    """Test reading data for a CL AR Spectrum (E-k) dataset."""
    s = hs.load(testfile_ek_path, reader="Delmic", signal="cl")
    assert isinstance(s, Signal2D)

    data = np.load(testfile_ek_data_path)
    np.testing.assert_allclose(s.data, data)

    ref_a = np.load(testfile_ek_channels_path)
    ref_w = np.load(testfile_ek_wavelengths_path)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 299.0099783670874)
    np.testing.assert_allclose(s.axes_manager[0].offset, -483.594683)
    np.testing.assert_allclose(s.axes_manager[0].size, 3)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -299.0099783670874)
    np.testing.assert_allclose(s.axes_manager[1].offset, 232.865286374114)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)

    assert s.axes_manager[2].name == "Wavelength"
    assert s.axes_manager[2].units == "nm"
    assert not s.axes_manager[2].navigate
    np.testing.assert_allclose(s.axes_manager[2].axis, ref_w)

    assert s.axes_manager[3].name == "Angle"
    assert s.axes_manager[3].units == ""
    assert not s.axes_manager[3].navigate
    np.testing.assert_allclose(s.axes_manager[3].axis, ref_a)

    assert s.metadata.General.title == "AR Spectrum"
    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert s.original_metadata.PolePosition == [15.25, 18.125]


def test_read_data_ek_CL_spot():
    """Test reading data for a CL AR Spectrum (E-k) dataset."""
    s = hs.load(testfile_ek_spot_path, reader="Delmic", signal="cl")

    assert s.data.shape == (270, 320)

    np.testing.assert_allclose(s.axes_manager[0].size, 320)
    assert s.axes_manager[0].name == "Wavelength"
    assert s.axes_manager[0].units == "nm"
    assert not s.axes_manager[0].navigate

    np.testing.assert_allclose(s.axes_manager[1].size, 270)
    assert s.axes_manager[1].name == "Angle"
    assert s.axes_manager[1].units == ""
    assert not s.axes_manager[1].navigate


def test_read_data_ek_SE():
    """Test reading data for a CL AR Spectrum (E-k) dataset."""
    s = hs.load(testfile_ek_path, reader="Delmic", signal="se")
    assert isinstance(s, BaseSignal)

    x = np.array([32766, 32766, 32766])
    y = np.array([32766, 32766, 32766])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 299.0099783670874)
    np.testing.assert_allclose(s.axes_manager[0].offset, -483.594683)
    np.testing.assert_allclose(s.axes_manager[0].size, 3)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -299.0099783670874)
    np.testing.assert_allclose(s.axes_manager[1].offset, 232.865286374114)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)

    assert s.metadata.General.title == "Secondary electrons concurrent"
    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert s.original_metadata.Magnification == 10000.0


def test_read_data_ek_survey():
    """Test reading data for a CL AR Spectrum (E-k) dataset."""
    s = hs.load(testfile_ek_path, reader="Delmic", signal="survey")

    data = np.load(testfile_ek_data_survey_path)
    np.testing.assert_allclose(s.data, data)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 48.828125)
    np.testing.assert_allclose(s.axes_manager[0].offset, -12475.585938)
    np.testing.assert_allclose(s.axes_manager[0].size, 512)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -48.828125)
    np.testing.assert_allclose(s.axes_manager[1].offset, 12475.585938)
    np.testing.assert_allclose(s.axes_manager[1].size, 512)

    assert s.metadata.General.title == "Secondary electrons survey"
    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert s.original_metadata.Magnification == 10000.0


# Angle-resolved dataset
def test_read_data_AR():
    """Test reading data for a CL AR dataset."""
    s = hs.load(testfile_AR_path, reader="Delmic", signal="all")
    assert len(s) == 3
    assert s[0].metadata.General.title == "Angle-resolved"
    assert s[1].metadata.General.title == "Secondary electrons concurrent"
    assert s[2].metadata.General.title == "Secondary electrons survey"


def test_read_data_AR_CL():
    """Test reading data for a CL AR dataset."""
    s = hs.load(testfile_AR_path, reader="Delmic", signal="cl")
    assert isinstance(s, Signal2D)

    data = np.load(testfile_AR_data_path)
    np.testing.assert_allclose(s.data, data)

    ref_a = np.load(testfile_AR_angles_path)
    ref_b = np.load(testfile_AR_channels_path)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 963.8629014657938)
    np.testing.assert_allclose(s.axes_manager[0].offset, -885.49392)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -963.8629014657942)
    np.testing.assert_allclose(s.axes_manager[1].offset, 1356.92001)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    assert s.axes_manager[2].name == "Angle A"
    assert s.axes_manager[2].units == ""
    assert not s.axes_manager[2].navigate
    np.testing.assert_allclose(s.axes_manager[2].axis, ref_a)

    assert s.axes_manager[3].name == "Angle B"
    assert s.axes_manager[3].units == ""
    assert not s.axes_manager[3].navigate
    np.testing.assert_allclose(s.axes_manager[3].axis, ref_b)

    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert s.original_metadata.PolePosition == [114.5, 129.75]


def test_read_data_AR_SE():
    """Test reading data for a CL AR dataset."""
    s = hs.load(testfile_AR_path, reader="Delmic", signal="se")
    assert isinstance(s, BaseSignal)

    x = np.array([32932, 33065])
    y = np.array([33203, 32495])
    z = np.array([32431, 32565])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)
    np.testing.assert_allclose(s.data[2], z)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 963.8629014657938)
    np.testing.assert_allclose(s.axes_manager[0].offset, -885.49392)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -963.8629014657942)
    np.testing.assert_allclose(s.axes_manager[1].offset, 1356.92001)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    assert s.metadata["Signal"]["quantity"] == "Counts"

    assert s.original_metadata.Magnification == 10000.0


def test_read_data_AR_survey():
    """Test reading data for a CL AR dataset."""
    s = hs.load(testfile_AR_path, reader="Delmic", signal="survey")
    assert isinstance(s, BaseSignal)

    data = np.load(testfile_AR_data_survey_path)
    np.testing.assert_allclose(s.data, data)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate
    np.testing.assert_allclose(s.axes_manager[0].scale, 27.343750000000004)
    np.testing.assert_allclose(s.axes_manager[0].offset, -6986.328125)
    np.testing.assert_allclose(s.axes_manager[0].size, 512)

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate
    np.testing.assert_allclose(s.axes_manager[1].scale, -27.343750000000004)
    np.testing.assert_allclose(s.axes_manager[1].offset, 6986.328125)
    np.testing.assert_allclose(s.axes_manager[1].size, 512)

    assert s.metadata["Signal"]["quantity"] == "Counts"
    assert s.metadata["General"]["title"] == "Secondary electrons survey"

    assert s.original_metadata.Magnification == 10000.0


def test_read_data_ar_pol():
    """Test reading data for a CL AR polarized dataset."""
    s = hs.load(testfile_ar_pol_spot_path, reader="Delmic", signal="all")
    assert len(s) == 8

    # We expect 6 AR images, each with a different polarizations
    polarizations = set()
    for d in s[:6]:
        assert d.metadata.General.title == "Angle-resolved"
        assert d.data.shape == (128, 128)
        pol = d.metadata.Acquisition_instrument.Spectrometer.Filter.position
        polarizations.add(pol)

    # If set is not of length 6, then some polarizations are identical
    assert len(polarizations) == 6

    assert s[-2].metadata.General.title == "Secondary electrons concurrent"
    assert s[-1].metadata.General.title == "Secondary electrons survey"


def test_wrong_arguments():
    """
    Attempt to load an HDF5 file with wrong arguments should raise a ValueError
    """
    with pytest.raises(ValueError):
        hs.load(testfile_intensity_path, reader="Delmic", signal="wrong_signal")

    with pytest.raises(NotImplementedError):
        hs.load(testfile_hyperspectral_path, reader="Delmic", signal="cl", lazy=True)


def test_wrong_format():
    """
    Attempt to load an HDF5 file not of the correct format should raise an IOError
    """
    with pytest.raises(IOError):
        hs.load(testfile_arina_path, reader="Delmic", signal="all")

    with pytest.raises(IOError):
        hs.load(testfile_hspy_path, reader="Delmic", signal="all")
