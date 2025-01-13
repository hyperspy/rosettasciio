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

from pathlib import Path

import numpy as np
import pytest

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
pytest.importorskip("h5py", reason="h5py not installed")

testfile_dir = (
    Path(__file__).parent / "data" / "delmic"
).resolve()
testfile_intensity_path = (
    testfile_dir / "sparc-intensity.h5"
).resolve()
testfile_intensity_data_survey_path = (
    testfile_dir / "sparc-intensity-data-survey.npy"
).resolve()
testfile_hyperspectral_path = (
    testfile_dir / "sparc-hyperspectral.h5"
).resolve()
testfile_hyperspectral_data_path = (
    testfile_dir / "sparc-hyperspectral-data.npy"
).resolve()
testfile_hyperspectral_data_survey_path = (
    testfile_dir / "sparc-hyperspectral-data-survey.npy"
).resolve()
testfile_hyperspectral_wavelengths_path = (
    testfile_dir / "sparc-hyperspectral-wavelengths.npy"
).resolve()
testfile_temporaltrace_path = (
    testfile_dir / "sparc-time-correlator.h5"
).resolve()
testfile_temporaltrace_data_path = (
    testfile_dir / "sparc-time-correlator-data.npy"
).resolve()
testfile_temporaltrace_data_survey_path = (
    testfile_dir / "sparc-time-correlator-data-survey.npy"
).resolve()
testfile_temporaltrace_timelist_path = (
    testfile_dir / "sparc-time-correlator-timelist.npy"
).resolve()
testfile_streakcamera_path = (
    testfile_dir / "sparc-streak-camera.h5"
).resolve()
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
testfile_ek_path = (
    testfile_dir / "sparc-e-k.h5"
).resolve()
testfile_ek_data_path = (
    testfile_dir / "sparc-e-k-data.npy"
).resolve()
testfile_ek_channels_path = (
    testfile_dir / "sparc-e-k-channels.npy"
).resolve()
testfile_ek_wavelengths_path = (
    testfile_dir / "sparc-e-k-wavelengths.npy"
).resolve()
testfile_AR_path = (
    testfile_dir / "sparc-angle-resolved.h5"
).resolve()
testfile_AR_data_path = (
    testfile_dir / "sparc-angle-resolved-data.npy"
).resolve()
testfile_AR_angles_path = (
    testfile_dir / "sparc-angle-resolved-angles.npy"
).resolve()
testfile_AR_channels_path = (
    testfile_dir / "sparc-angle-resolved-channels.npy"
).resolve()


# Intensity dataset
def test_read_data_intensity():
    """Test reading data for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path, reader="Delmic")

    x = np.array([27147, 28907])
    y = np.array([26964, 27695])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)
    
def test_read_data_intensity_CL():
    """Test reading data for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path, reader="Delmic",signal="CL")

    x = np.array([27147, 28907])
    y = np.array([26964, 27695])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)
    
def test_read_data_intensity_SE():
        """Test reading data for a CL intensity dataset."""
        s = hs.load(testfile_intensity_path, reader="Delmic",signal="SE")

        x = np.array([27308, 28592])
        y = np.array([27851, 27958])
        np.testing.assert_allclose(s.data[0], x)
        np.testing.assert_allclose(s.data[1], y)
        
def test_read_data_intensity_survey():
        """Test reading data for a CL intensity dataset."""
        s = hs.load(testfile_intensity_path, reader="Delmic",signal="survey")
        data = np.load(testfile_intensity_data_survey_path)

        np.testing.assert_allclose(s.data, data)
        


def test_read_axes_intensity():
    """Test reading axes for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path,reader="Delmic")

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate == True

    np.testing.assert_allclose(s.axes_manager[0].scale, 200)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    # test Y axis
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate == True

    np.testing.assert_allclose(s.axes_manager[1].scale, 200)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)
    
def test_read_axes_intensity_CL():
    """Test reading axes for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path,reader="Delmic",signal='CL')

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate == True

    np.testing.assert_allclose(s.axes_manager[0].scale, 200)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    # test Y axis
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate == True

    np.testing.assert_allclose(s.axes_manager[1].scale, 200)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)  
    
def test_read_axes_intensity_SE():
    """Test reading axes for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path,reader="Delmic",signal='SE')

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate == True

    np.testing.assert_allclose(s.axes_manager[0].scale, 200)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    # test Y axis
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate == True

    np.testing.assert_allclose(s.axes_manager[1].scale, 200)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)
    
def test_read_axes_intensity_survey():
    """Test reading axes for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path,reader="Delmic",signal='survey')

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate == True

    np.testing.assert_allclose(s.axes_manager[0].scale, 1.7578125)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[0].size, 256)

    # test Y axis
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate == True

    np.testing.assert_allclose(s.axes_manager[1].scale, 1.7578125)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[1].size, 256)


#def test_read_metadata_intensity():
#    """Test reading metadata for a CL intensity dataset."""
#    s = hs.load(testfile_intensity_path, reader="Delmic")

#    assert s.metadata["General"]["title"] == ""
#    assert s.metadata["Signal"]["quantity"] == "Counts"


def test_read_original_metadata_intensity():
    """Test reading original metadata for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path, reader="Delmic")

    assert s.original_metadata
    
def test_read_original_metadata_intensity_CL():
    """Test reading original metadata for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path, reader="Delmic",signal='CL')

    assert s.original_metadata    
    
def test_read_original_metadata_intensity_SE():
    """Test reading original metadata for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path, reader="Delmic",signal='SE')

    assert s.original_metadata 
    
def test_read_original_metadata_intensity_survey():
    """Test reading original metadata for a CL intensity dataset."""
    s = hs.load(testfile_intensity_path, reader="Delmic",signal='survey')

    assert s.original_metadata 


# Hyperspectral dataset
def test_read_data_hyperspectral():
    """Test reading data for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic")
    data = np.load(testfile_hyperspectral_data_path)

    np.testing.assert_allclose(s.data, data)
    
def test_read_data_hyperspectral_CL():
    """Test reading data for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic",signal='CL')
    data = np.load(testfile_hyperspectral_data_path)

    np.testing.assert_allclose(s.data, data)
    
def test_read_data_hyperspectral_SE():
    """Test reading data for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic",signal='SE')
    
    x = np.array([27265, 27598])
    y = np.array([27892, 28124])
    z = np.array([28299, 28468])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)
    np.testing.assert_allclose(s.data[2], z)
    
def test_read_data_hyperspectral_survey():
    """Test reading data for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic",signal='survey')
    data = np.load(testfile_hyperspectral_data_survey_path)
    
    np.testing.assert_allclose(s.data, data)

def test_read_axes_hyperspectral():
    """Test reading axes for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic")
    ref = np.load(testfile_hyperspectral_wavelengths_path)

    np.testing.assert_allclose(s.axes_manager[0].scale, 963.862901465797)
    np.testing.assert_allclose(s.axes_manager[0].offset, -411.20439112265274)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    np.testing.assert_allclose(s.axes_manager[1].scale, 963.862901465797)
    np.testing.assert_allclose(s.axes_manager[1].offset, 362.48947994366983)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    np.testing.assert_allclose(s.axes_manager[2].axis, ref)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"

    assert s.axes_manager[2].name == "Wavelength"
    assert s.axes_manager[2].units == "nm"
    
def test_read_axes_hyperspectral_SE():
    """Test reading axes for a CL intensity dataset."""
    s = hs.load(testfile_hyperspectral_path,reader="Delmic",signal='SE')

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate == True

    np.testing.assert_allclose(s.axes_manager[0].scale, 963.862901465797)
    np.testing.assert_allclose(s.axes_manager[0].offset, -411.20439112265274)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    # test Y axis
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate == True

    np.testing.assert_allclose(s.axes_manager[1].scale, 963.862901465797)
    np.testing.assert_allclose(s.axes_manager[1].offset, 362.48947994366983)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

def test_read_axes_hyperspectral_survey():
    """Test reading axes for a CL intensity dataset."""
    s = hs.load(testfile_hyperspectral_path,reader="Delmic",signal='survey')

    # test X axis
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"
    assert s.axes_manager[0].navigate == True

    np.testing.assert_allclose(s.axes_manager[0].scale, 27.343750000000004)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[0].size, 512)

    # test Y axis
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"
    assert s.axes_manager[1].navigate == True

    np.testing.assert_allclose(s.axes_manager[1].scale, 27.343750000000004)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[1].size, 512)

#def test_read_metadata_hyperspectral():
#    """Test reading metadata for a CL hyperspectral dataset."""
#    s = hs.load(testfile_hyperspectral_path, reader="Delmic")

#    assert s.metadata["General"]["title"] == ""
#    assert s.metadata["Signal"]["quantity"] == "Counts"


def test_read_original_metadata_hyperspectral():
    """Test reading original metadata for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic")

    assert s.original_metadata
    
def test_read_original_metadata_hyperspectral_CL():
    """Test reading original metadata for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic",signal='CL')

    assert s.original_metadata
    
def test_read_original_metadata_hyperspectral_SE():
    """Test reading original metadata for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic",signal='SE')

    assert s.original_metadata

def test_read_original_metadata_hyperspectral_survey():
    """Test reading original metadata for a CL hyperspectral dataset."""
    s = hs.load(testfile_hyperspectral_path, reader="Delmic",signal='survey')

    assert s.original_metadata

# Time-resolved dataset
def test_read_data_temporaltrace():
    """Test reading data for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic")
    data = np.load(testfile_temporaltrace_data_path)

    np.testing.assert_allclose(s.data, data)

def test_read_data_temporaltrace_CL():
    """Test reading data for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic",signal='CL')
    data = np.load(testfile_temporaltrace_data_path)

    np.testing.assert_allclose(s.data, data)
    
def test_read_data_temporaltrace_SE():
    """Test reading data for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic",signal='SE')
    
    x = np.array([32766, 32766])
    y = np.array([32766, 32766])
    z = np.array([32766, 32766])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)
    np.testing.assert_allclose(s.data[2], z)

def test_read_data_temporaltrace_survey():
    """Test reading data for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic",signal='survey')
    data = np.load(testfile_temporaltrace_data_survey_path)

    np.testing.assert_allclose(s.data, data)


def test_read_axes_temporaltrace():
    """Test reading axes for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic")
    ref = np.load(testfile_temporaltrace_timelist_path)

    np.testing.assert_allclose(s.axes_manager[0].scale, 917.1372563963987)
    np.testing.assert_allclose(s.axes_manager[0].offset, -314.999999999998)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    np.testing.assert_allclose(s.axes_manager[1].scale, 917.1372563963987)
    np.testing.assert_allclose(s.axes_manager[1].offset, 350.00000000000034)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    np.testing.assert_allclose(s.axes_manager[2].axis, ref)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"

    assert s.axes_manager[2].name == "Time"
    assert s.axes_manager[2].units == "ns"

def test_read_axes_temporaltrace_CL():
    """Test reading axes for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic",signal='CL')
    ref = np.load(testfile_temporaltrace_timelist_path)

    np.testing.assert_allclose(s.axes_manager[0].scale, 917.1372563963987)
    np.testing.assert_allclose(s.axes_manager[0].offset, -314.999999999998)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    np.testing.assert_allclose(s.axes_manager[1].scale, 917.1372563963987)
    np.testing.assert_allclose(s.axes_manager[1].offset, 350.00000000000034)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    np.testing.assert_allclose(s.axes_manager[2].axis, ref)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"

    assert s.axes_manager[2].name == "Time"
    assert s.axes_manager[2].units == "ns"
    
def test_read_axes_temporaltrace_SE():
    """Test reading axes for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic",signal='SE')

    np.testing.assert_allclose(s.axes_manager[0].scale, 917.1372563963987)
    np.testing.assert_allclose(s.axes_manager[0].offset, -314.999999999998)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    np.testing.assert_allclose(s.axes_manager[1].scale, 917.1372563963987)
    np.testing.assert_allclose(s.axes_manager[1].offset, 350.00000000000034)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"

    
def test_read_axes_temporaltrace_survey():
    """Test reading axes for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic",signal='survey')

    np.testing.assert_allclose(s.axes_manager[0].scale, 27.343750000000004)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[0].size, 512)

    np.testing.assert_allclose(s.axes_manager[1].scale, 27.343750000000004)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[1].size, 512)


    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"


#def test_read_metadata_temporaltrace():
#    """Test reading metadata for a CL decay trace or g(2) datasets."""
#    s = hs.load(testfile_temporaltrace_path, reader="Delmic")

#    assert s.metadata["General"]["title"] == ""
#    assert s.metadata["Signal"]["quantity"] == "Counts"


def test_read_original_metadata_temporaltrace():
    """Test reading original metadata for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic")

    assert s.original_metadata
    
def test_read_original_metadata_temporaltrace_CL():
    """Test reading original metadata for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic",signal='CL')

    assert s.original_metadata
    
def test_read_original_metadata_temporaltrace_SE():
    """Test reading original metadata for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic",signal='SE')

    assert s.original_metadata

def test_read_original_metadata_temporaltrace_survey():
    """Test reading original metadata for a CL decay trace or g(2) datasets."""
    s = hs.load(testfile_temporaltrace_path, reader="Delmic",signal='survey')

    assert s.original_metadata


# Streak camera dataset
def test_read_data_streakcamera():
    """Test reading data for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic")
    data = np.load(testfile_streakcamera_data_path)

    np.testing.assert_allclose(s.data, data)
    
def test_read_data_streakcamera_CL():
    """Test reading data for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic",signal='CL')
    data = np.load(testfile_streakcamera_data_path)

    np.testing.assert_allclose(s.data, data)
    
def test_read_data_streakcamera_SE():
    """Test reading data for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic",signal='SE')
    
    x = np.array([5443,   90, 5318])
    y = np.array([241, 5256,  174])
    np.testing.assert_allclose(s.data[0], x)
    np.testing.assert_allclose(s.data[1], y)
    
def test_read_data_streakcamera_survey():
    """Test reading data for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic",signal='survey')
    data = np.load(testfile_streakcamera_data_survey_path)

    np.testing.assert_allclose(s.data, data)


def test_read_axes_streakcamera():
    """Test reading axes for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic")
    ref_t = np.load(testfile_streakcamera_timelist_path)
    ref_w = np.load(testfile_streakcamera_wavelengths_path)

    np.testing.assert_allclose(s.axes_manager[0].scale, 3.1073798455896666)
    np.testing.assert_allclose(s.axes_manager[0].offset, -2.411301009767636)
    np.testing.assert_allclose(s.axes_manager[0].size, 3)

    np.testing.assert_allclose(s.axes_manager[1].scale, 3.1073798455896666)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.573579891248638)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)

    np.testing.assert_allclose(s.axes_manager[3].axis, ref_t)
    np.testing.assert_allclose(s.axes_manager[2].axis, ref_w)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "µm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "µm"

    assert s.axes_manager[3].name == "Time"
    assert s.axes_manager[3].units == "s"

    assert s.axes_manager[2].name == "Wavelength"
    assert s.axes_manager[2].units == "nm"

def test_read_axes_streakcamera_CL():
    """Test reading axes for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic",signal='CL')
    ref_t = np.load(testfile_streakcamera_timelist_path)
    ref_w = np.load(testfile_streakcamera_wavelengths_path)

    np.testing.assert_allclose(s.axes_manager[0].scale, 3.1073798455896666)
    np.testing.assert_allclose(s.axes_manager[0].offset, -2.411301009767636)
    np.testing.assert_allclose(s.axes_manager[0].size, 3)

    np.testing.assert_allclose(s.axes_manager[1].scale, 3.1073798455896666)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.573579891248638)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)

    np.testing.assert_allclose(s.axes_manager[3].axis, ref_t)
    np.testing.assert_allclose(s.axes_manager[2].axis, ref_w)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "µm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "µm"

    assert s.axes_manager[3].name == "Time"
    assert s.axes_manager[3].units == "s"

    assert s.axes_manager[2].name == "Wavelength"
    assert s.axes_manager[2].units == "nm"

def test_read_axes_streakcamera_SE():
    """Test reading axes for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic",signal="SE")
    ref_t = np.load(testfile_streakcamera_timelist_path)
    ref_w = np.load(testfile_streakcamera_wavelengths_path)

    np.testing.assert_allclose(s.axes_manager[0].scale, 3.1073798455896666)
    np.testing.assert_allclose(s.axes_manager[0].offset, -2.411301009767636)
    np.testing.assert_allclose(s.axes_manager[0].size, 3)

    np.testing.assert_allclose(s.axes_manager[1].scale, 3.1073798455896666)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.573579891248638)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "µm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "µm"

def test_read_axes_streakcamera_survey():
    """Test reading axes for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic",signal='survey')
    ref_t = np.load(testfile_streakcamera_timelist_path)
    ref_w = np.load(testfile_streakcamera_wavelengths_path)

    np.testing.assert_allclose(s.axes_manager[0].scale, 48.828125)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[0].size, 512)

    np.testing.assert_allclose(s.axes_manager[1].scale, 48.828125)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[1].size, 512)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"


#def test_read_metadata_streakcamera():
#    """Test reading metadata for a CL streack camera dataset."""
#    s = hs.load(testfile_streakcamera_path, reader="Delmic")

#    assert s.metadata["General"]["title"] == ""
#    assert s.metadata["Signal"]["quantity"] == "Counts"


def test_read_original_metadata_streakcamera():
    """Test reading original metadata for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic")

    assert s.original_metadata
    
def test_read_original_metadata_streakcamera_CL():
    """Test reading original metadata for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic",signal='CL')

    assert s.original_metadata
    
def test_read_original_metadata_streakcamera_SE():
    """Test reading original metadata for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic",signal='SE')

    assert s.original_metadata
    
def test_read_original_metadata_streakcamera_survey():
    """Test reading original metadata for a CL streak camera dataset."""
    s = hs.load(testfile_streakcamera_path, reader="Delmic",signal='survey')

    assert s.original_metadata


# E-k dataset
def test_read_data_ek():
    """Test reading data for a CL AR Spectrum (E-k) dataset."""
    s = hs.load(testfile_ek_path, reader="Delmic")
    data = np.load(testfile_ek_data_path)

    np.testing.assert_allclose(s.data, data)


def test_read_axes_ek():
    """Test reading axes for a CL AR Spectrum (E-k) dataset."""
    s = hs.load(testfile_ek_path, reader="Delmic")
    ref_a = np.load(testfile_ek_channels_path)
    ref_w = np.load(testfile_ek_wavelengths_path)

    np.testing.assert_allclose(s.axes_manager[0].scale, 299.0099783670874)
    np.testing.assert_allclose(s.axes_manager[0].offset, -184.58470506959597)
    np.testing.assert_allclose(s.axes_manager[0].size, 3)

    np.testing.assert_allclose(s.axes_manager[1].scale, 299.0099783670874)
    np.testing.assert_allclose(s.axes_manager[1].offset, 83.36029719057113)
    np.testing.assert_allclose(s.axes_manager[1].size, 2)

    np.testing.assert_allclose(s.axes_manager[3].axis, ref_a)
    np.testing.assert_allclose(s.axes_manager[2].axis, ref_w)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"

    assert s.axes_manager[2].name == "Wavelength"
    assert s.axes_manager[2].units == "nm"

    assert s.axes_manager[3].name == "Angle"
    assert s.axes_manager[3].units == ""


#def test_read_metadata_ek():
#    """Test reading metadata for a CL AR Spectrum (E-k) dataset."""
#    s = hs.load(testfile_ek_path, reader="Delmic")

#    assert s.metadata["General"]["title"] == ""
#    assert s.metadata["Signal"]["quantity"] == "Counts"


def test_read_original_metadata_ek():
    """Test reading original metadata for a CL AR Spectrum (E-k) dataset."""
    s = hs.load(testfile_ek_path, reader="Delmic")

    assert s.original_metadata


# Angle-resolved dataset
def test_read_data_AR():
    """Test reading data for a CL AR dataset."""
    s = hs.load(testfile_AR_path, reader="Delmic")
    data = np.load(testfile_AR_data_path)

    np.testing.assert_allclose(s.data, data)


def test_read_axes_AR():
    """Test reading axes for a CL AR dataset."""
    s = hs.load(testfile_AR_path, reader="Delmic")
    ref_a = np.load(testfile_AR_angles_path)
    ref_w = np.load(testfile_AR_channels_path)

    np.testing.assert_allclose(s.axes_manager[0].scale, 963.8629014657938)
    np.testing.assert_allclose(s.axes_manager[0].offset, -403.5624697252699)
    np.testing.assert_allclose(s.axes_manager[0].size, 2)

    np.testing.assert_allclose(s.axes_manager[1].scale, 963.8629014657942)
    np.testing.assert_allclose(s.axes_manager[1].offset, 393.0571655331861)
    np.testing.assert_allclose(s.axes_manager[1].size, 3)

    np.testing.assert_allclose(s.axes_manager[2].axis, ref_a)
    np.testing.assert_allclose(s.axes_manager[3].axis, ref_w)

    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "nm"

    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "nm"

    assert s.axes_manager[3].name == "C"
    assert s.axes_manager[3].units == ""

    assert s.axes_manager[2].name == "Angle"
    assert s.axes_manager[2].units == ""


#def test_read_metadata_AR():
#    """Test reading metadata for a CL AR dataset."""
#    s = hs.load(testfile_AR_path, reader="Delmic")

#    assert s.metadata["General"]["title"] == ""
#    assert s.metadata["Signal"]["quantity"] == "Counts"


def test_read_original_metadata_AR():
    """Test reading original metadata for a CL AR dataset."""
    s = hs.load(testfile_AR_path, reader="Delmic")

    assert s.original_metadata
