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

from rsciio.digitalsurf._api import DigitalSurfHandler, MountainsMapFileError

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")

TEST_DATA_PATH = Path(__file__).parent / "data" / "digitalsurf"

header_keys = [
    "H01_Signature",
    "H02_Format",
    "H03_Number_of_Objects",
    "H04_Version",
    "H05_Object_Type",
    "H06_Object_Name",
    "H07_Operator_Name",
    "H08_P_Size",
    "H09_Acquisition_Type",
    "H10_Range_Type",
    "H11_Special_Points",
    "H12_Absolute",
    "H13_Gauge_Resolution",
    "H14_W_Size",
    "H15_Size_of_Points",
    "H16_Zmin",
    "H17_Zmax",
    "H18_Number_of_Points",
    "H19_Number_of_Lines",
    "H20_Total_Nb_of_Pts",
    "H21_X_Spacing",
    "H22_Y_Spacing",
    "H23_Z_Spacing",
    "H24_Name_of_X_Axis",
    "H25_Name_of_Y_Axis",
    "H26_Name_of_Z_Axis",
    "H27_X_Step_Unit",
    "H28_Y_Step_Unit",
    "H29_Z_Step_Unit",
    "H30_X_Length_Unit",
    "H31_Y_Length_Unit",
    "H32_Z_Length_Unit",
    "H33_X_Unit_Ratio",
    "H34_Y_Unit_Ratio",
    "H35_Z_Unit_Ratio",
    "H36_Imprint",
    "H37_Inverted",
    "H38_Levelled",
    "H39_Obsolete",
    "H40_Seconds",
    "H41_Minutes",
    "H42_Hours",
    "H43_Day",
    "H44_Month",
    "H45_Year",
    "H46_Day_of_week",
    "H47_Measurement_duration",
    "H48_Compressed_data_size",
    "H49_Obsolete",
    "H50_Comment_size",
    "H51_Private_size",
    "H52_Client_zone",
    "H53_X_Offset",
    "H54_Y_Offset",
    "H55_Z_Offset",
    "H56_T_Spacing",
    "H57_T_Offset",
    "H58_T_Axis_Name",
    "H59_T_Step_Unit",
    "H60_Comment",
]

atto_head_keys = [
    "WAFER",
    "SITE IMAGE",
    "SEM",
    "CHANNELS",
    "SPECTROMETER",
    "SCAN",
]

atto_wafer_keys = [
    "Lot Number",
    "ID",
    "Type",
    "Center Position X",
    "Center Position X_units",
    "Center Position Y",
    "Center Position Y_units",
    "Orientation",
    "Orientation_units",
    "Diameter",
    "Diameter_units",
    "Flat Length",
    "Flat Length_units",
    "Edge Exclusion",
    "Edge Exclusion_units",
]

atto_scan_keys = [
    "Mode",
    "HYP Dwelltime",
    "HYP Dwelltime_units",
    "Resolution_X",
    "Resolution_X_units",
    "Resolution_Y",
    "Resolution_Y_units",
    "Reference_Size_X",
    "Reference_Size_Y",
    "Voltage Calibration Range_X",
    "Voltage Calibration Range_X_units",
    "Voltage Calibration Range_Y",
    "Voltage Calibration Range_Y_units",
    "Start_X",
    "Size_X",
    "Start_Y",
    "Size_Y",
    "Rotate",
    "Rotate_units",
]


def test_invalid_data():
    dsh = DigitalSurfHandler()

    with pytest.raises(MountainsMapFileError):
        dsh._Object_type = "INVALID"
        dsh._build_sur_dict()

    dsh._list_sur_file_content = [{"img1": None}, {"img2": None}]

    with pytest.raises(MountainsMapFileError):
        dsh._build_hyperspectral_map()

    with pytest.raises(MountainsMapFileError):
        dsh._build_general_1D_data()

    with pytest.raises(MountainsMapFileError):
        dsh._build_surface()

    dsh.signal_dict = {}
    dsh.signal_dict["original_metadata"] = {}
    res = dsh._map_SEM_metadata()
    assert res == {}

    res = dsh._map_Spectrometer_metadata()
    assert res == {}

    res = dsh._map_spectral_detector_metadata()
    assert res == {}


def test_load_profile():
    # Signal loading
    fname = TEST_DATA_PATH / "test_profile.pro"
    s = hs.load(fname)

    # Verifying signal shape and axes dimensions, navigation (not data themselves)
    assert s.data.shape == (128,)
    assert s.data.dtype == np.dtype(float)
    np.testing.assert_allclose(s.axes_manager[0].scale, 8.252197e-05)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.0)
    assert s.axes_manager[0].name == "Width"
    assert s.axes_manager[0].units == "mm"
    assert s.axes_manager[0].size == 128
    assert s.axes_manager[0].navigate is False

    # Metadata verification
    md = s.metadata
    assert md.Signal.quantity == "CL Intensity (a.u.)"

    # Original metadata. We verify that the correct structure is given
    # and the right headers but not the values
    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ["Object_0_Channel_0"]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ["Header"]
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) == header_keys


def test_load_RGB():
    fname = TEST_DATA_PATH / "test_RGB.sur"
    s = hs.load(fname)
    assert s.data.shape == (200, 200)
    assert s.data.dtype == np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])

    np.testing.assert_allclose(s.axes_manager[0].scale, 0.35277777)
    np.testing.assert_allclose(s.axes_manager[0].offset, 208.8444519)
    np.testing.assert_allclose(s.axes_manager[1].scale, 0.35277777)
    np.testing.assert_allclose(s.axes_manager[1].offset, 210.608337)
    assert s.axes_manager[0].name == "X"
    assert s.axes_manager[0].units == "mm"
    assert s.axes_manager[1].name == "Y"
    assert s.axes_manager[1].units == "mm"
    assert s.axes_manager[0].size == 200
    assert s.axes_manager[0].navigate is False
    assert s.axes_manager[1].size == 200
    assert s.axes_manager[1].navigate is False

    md = s.metadata
    assert md.Signal.quantity == "Z"

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == [
        "Object_0_Channel_0",
        "Object_0_Channel_1",
        "Object_0_Channel_2",
    ]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ["Header"]
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) == header_keys


def test_load_spectra():
    fname = TEST_DATA_PATH / "test_spectra.pro"
    s = hs.load(fname)

    assert s.data.shape == (65, 512)
    assert s.data.dtype == np.dtype("float64")

    md = s.metadata
    assert md.Signal.quantity == "CL Intensity (a.u.)"
    np.testing.assert_allclose(s.axes_manager[0].scale, 0.00011458775406936184)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[1].scale, 1.084000246009964e-06)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.00017284281784668565)
    assert s.axes_manager[0].name == "Spectrum positi"
    assert s.axes_manager[0].units == "mm"
    assert s.axes_manager[1].name == "Wavelength"
    assert s.axes_manager[1].units == "mm"
    assert s.axes_manager[0].size == 65
    assert s.axes_manager[0].navigate is True
    assert s.axes_manager[1].size == 512
    assert s.axes_manager[1].navigate is False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == [
        "Object_0_Channel_0",
    ]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ["Header"]
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) == header_keys


def test_load_spectral_map_compressed():
    fname = TEST_DATA_PATH / "test_spectral_map_compressed.sur"
    s = hs.load(fname)

    assert s.data.shape == (12, 10, 281)
    assert s.data.dtype == np.dtype("float64")

    md = s.metadata
    assert md.Signal.quantity == "CL Intensity (a.u.)"
    np.testing.assert_allclose(s.axes_manager[0].scale, 8.252198e-05)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.005694016348570585)
    np.testing.assert_allclose(s.axes_manager[1].scale, 8.252198e-05)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.0054464503191411495)
    np.testing.assert_allclose(s.axes_manager[2].scale, 1.084000246009964e-06)
    np.testing.assert_allclose(s.axes_manager[2].offset, 0.00034411484375596046)
    assert s.axes_manager[0].name == "Width"
    assert s.axes_manager[0].units == "mm"
    assert s.axes_manager[1].name == "Height"
    assert s.axes_manager[1].units == "mm"
    assert s.axes_manager[2].name == "Wavelength"
    assert s.axes_manager[2].units == "mm"
    assert s.axes_manager[0].size == 10
    assert s.axes_manager[0].navigate is True
    assert s.axes_manager[1].size == 12
    assert s.axes_manager[1].navigate is True
    assert s.axes_manager[2].size == 281
    assert s.axes_manager[2].navigate is False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == [
        "Object_0_Channel_0",
    ]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ["Header", "Parsed"]
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) == header_keys

    assert list(omd.Object_0_Channel_0.Parsed.as_dictionary().keys()) == atto_head_keys

    assert (
        list(omd.Object_0_Channel_0.Parsed.WAFER.as_dictionary().keys())
        == atto_wafer_keys
    )

    assert (
        list(omd.Object_0_Channel_0.Parsed.SCAN.as_dictionary().keys())
        == atto_scan_keys
    )


def test_load_spectral_map():
    fname = TEST_DATA_PATH / "test_spectral_map.sur"
    s = hs.load(fname)

    assert s.data.shape == (12, 10, 310)
    assert s.data.dtype == np.dtype("float64")

    md = s.metadata
    assert md.Signal.quantity == "CL Intensity (a.u.)"
    np.testing.assert_allclose(s.axes_manager[0].scale, 8.252197585534304e-05)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.00701436772942543)
    np.testing.assert_allclose(s.axes_manager[1].scale, 8.252197585534304e-05)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.003053313121199608)
    np.testing.assert_allclose(s.axes_manager[2].scale, 1.084000246009964e-6)
    np.testing.assert_allclose(s.axes_manager[2].offset, 0.0003332748601678759)
    assert s.axes_manager[0].name == "Width"
    assert s.axes_manager[0].units == "mm"
    assert s.axes_manager[1].name == "Height"
    assert s.axes_manager[1].units == "mm"
    assert s.axes_manager[2].name == "Wavelength"
    assert s.axes_manager[2].units == "mm"
    assert s.axes_manager[0].size == 10
    assert s.axes_manager[0].navigate is True
    assert s.axes_manager[1].size == 12
    assert s.axes_manager[1].navigate is True
    assert s.axes_manager[2].size == 310
    assert s.axes_manager[2].navigate is False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == [
        "Object_0_Channel_0",
    ]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ["Header", "Parsed"]
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) == header_keys

    assert list(omd.Object_0_Channel_0.Parsed.as_dictionary().keys()) == atto_head_keys

    assert (
        list(omd.Object_0_Channel_0.Parsed.WAFER.as_dictionary().keys())
        == atto_wafer_keys
    )

    assert (
        list(omd.Object_0_Channel_0.Parsed.SCAN.as_dictionary().keys())
        == atto_scan_keys
    )


def test_load_spectrum_compressed():
    fname = TEST_DATA_PATH / "test_spectrum_compressed.pro"
    s = hs.load(fname)
    md = s.metadata
    assert md.Signal.quantity == "CL Intensity (a.u.)"
    assert s.data.shape == (512,)
    # np.testing.assert_allclose(s.axes_manager[0].scale,1.0)
    # np.testing.assert_allclose(s.axes_manager[0].offset,0.0)
    np.testing.assert_allclose(s.axes_manager[0].scale, 1.084000246009964e-6)
    np.testing.assert_allclose(s.axes_manager[0].offset, 172.84281784668565e-6)

    # assert s.axes_manager[0].name == 'T'
    # assert s.axes_manager[0].units == ''
    assert s.axes_manager[0].name == "Wavelength"
    assert s.axes_manager[0].units == "mm"
    # assert s.axes_manager[0].size == 1
    # assert s.axes_manager[0].navigate == True
    assert s.axes_manager[0].size == 512
    assert s.axes_manager[0].navigate is False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ["Object_0_Channel_0"]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ["Header"]
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) == header_keys


def test_load_spectrum():
    fname = TEST_DATA_PATH / "test_spectrum.pro"
    s = hs.load(fname)
    assert s.data.shape == (512,)

    md = s.metadata
    assert md.Signal.quantity == "CL Intensity (a.u.)"
    # np.testing.assert_allclose(s.axes_manager[0].scale,1.0)
    # np.testing.assert_allclose(s.axes_manager[0].offset,0.0)
    np.testing.assert_allclose(s.axes_manager[0].scale, 1.084000246009964e-6)
    np.testing.assert_allclose(s.axes_manager[0].offset, 172.84281784668565e-6)

    # assert s.axes_manager[0].name == 'T'
    # assert s.axes_manager[0].units == ''
    assert s.axes_manager[0].name == "Wavelength"
    assert s.axes_manager[0].units == "mm"
    # assert s.axes_manager[0].size == 1
    # assert s.axes_manager[0].navigate == True
    assert s.axes_manager[0].size == 512
    assert s.axes_manager[0].navigate is False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ["Object_0_Channel_0"]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ["Header"]
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) == header_keys


def test_load_surface():
    fname = TEST_DATA_PATH / "test_surface.sur"
    s = hs.load(fname)
    md = s.metadata
    assert md.Signal.quantity == "CL Intensity (a.u.)"
    assert s.data.shape == (128, 128)
    np.testing.assert_allclose(s.axes_manager[0].scale, 8.252198e-05)
    np.testing.assert_allclose(s.axes_manager[0].offset, 0.0)
    np.testing.assert_allclose(s.axes_manager[1].scale, 8.252198e-05)
    np.testing.assert_allclose(s.axes_manager[1].offset, 0.0)

    assert s.axes_manager[0].name == "Width"
    assert s.axes_manager[0].units == "mm"
    assert s.axes_manager[1].name == "Height"
    assert s.axes_manager[1].units == "mm"
    assert s.axes_manager[0].size == 128
    assert s.axes_manager[0].navigate is False
    assert s.axes_manager[1].size == 128
    assert s.axes_manager[1].navigate is False

    omd = s.original_metadata
    assert list(omd.as_dictionary().keys()) == ["Object_0_Channel_0"]
    assert list(omd.Object_0_Channel_0.as_dictionary().keys()) == ["Header"]
    assert list(omd.Object_0_Channel_0.Header.as_dictionary().keys()) == header_keys


def test_choose_signal_type():
    reader = DigitalSurfHandler()

    # Empty dict should not raise error but return empty string
    mock_dict = {}
    assert not reader._choose_signal_type(mock_dict)
    # Correct behaviour
    mock_dict = {"_26_Name_of_Z_Axis": "CL Intensity"}
    assert reader._choose_signal_type(mock_dict) == "CL"
    # Other behaviour
    mock_dict = {"_26_Name_of_Z_Axis": "Hairy Monster"}
    assert not reader._choose_signal_type(mock_dict)


def test_metadata_mapping():
    fname = TEST_DATA_PATH / "test_spectral_map_compressed.sur"

    # Initialize  reader
    reader = DigitalSurfHandler(fname)
    reader._read_sur_file()
    assert not reader.signal_dict["metadata"]

    dict_from_sur_object = reader._list_sur_file_content[0]

    # reader._build_sur_dict()
    generic_metadata = reader._build_generic_metadata(dict_from_sur_object)
    # By default no signal specific metadata should be created
    assert "General" in generic_metadata
    assert "Signal" in generic_metadata
    assert "Acquisition Instrument" not in generic_metadata

    # Assert correct parsing from date
    dict_from_sur_object["_45_Year"] = 1993
    dict_from_sur_object["_44_Month"] = 5
    dict_from_sur_object["_43_Day"] = 27

    # Assert correct parsing from time
    dict_from_sur_object["_42_Hours"] = 8
    dict_from_sur_object["_41_Minutes"] = 45
    dict_from_sur_object["_40_Seconds"] = 27

    generic_metadata = reader._build_generic_metadata(dict_from_sur_object)
    assert generic_metadata["General"]["date"] == "1993-05-27"
    assert generic_metadata["General"]["time"] == "08:45:27"

    # Fake a generic signal
    dict_from_sur_object["_26_Name_of_Z_Axis"] = "NothingSpecial1D"
    reader._set_metadata_and_original_metadata(dict_from_sur_object)
    assert not reader.signal_dict["metadata"]["Signal"]["signal_type"]
    assert "Acquisition Instrument" not in reader.signal_dict["metadata"]

    # Now with a CL signal
    dict_from_sur_object["_26_Name_of_Z_Axis"] = "CL Intensity"
    reader._set_metadata_and_original_metadata(dict_from_sur_object)
    assert reader.signal_dict["metadata"]["Signal"]["signal_type"] == "CL"
    assert "Acquisition_instrument" in reader.signal_dict["metadata"]
    assert (
        reader.signal_dict["metadata"]["Acquisition_instrument"]["Spectrometer"][
            "exit_slit_width"
        ]
        == 7000
    )


def test_get_n_obj_chn():

    omd = {"Object_0_Channel_0":{},
           "Object_1_Channel_0":{},
           "Object_2_Channel_0":{},
           "Object_2_Channel_1":{},
           "Object_2_Channel_2":{},
           "Object_3_Channel_0":{},}

    assert DigitalSurfHandler._get_nobjects(omd)==3


def test_compressdata():

    testdat = np.arange(120, dtype=np.int32)

    #Refuse too many / neg streams
    with pytest.raises(MountainsMapFileError):
        DigitalSurfHandler._compress_data(testdat,nstreams=9)
    with pytest.raises(MountainsMapFileError):
        DigitalSurfHandler._compress_data(testdat,nstreams=-1)
    
    # Accept 1 (dft) or several streams
    bcomp = DigitalSurfHandler._compress_data(testdat)
    assert bcomp.startswith(b'\x01\x00\x00\x00\xe0\x01\x00\x00')
    bcomp = DigitalSurfHandler._compress_data(testdat,nstreams=2)
    assert bcomp.startswith(b'\x02\x00\x00\x00\xf0\x00\x00\x00_\x00\x00\x00')

    # Accept 16-bits int as well as 32
    testdat = np.arange(120, dtype=np.int16)
    bcomp = DigitalSurfHandler._compress_data(testdat)
    assert bcomp.startswith(b'\x01\x00\x00\x00\xf0\x00\x00\x00')


    # Also streams non-perfectly divided data
    testdat = np.arange(120, dtype=np.int16)
    bcomp = DigitalSurfHandler._compress_data(testdat)
    assert bcomp.startswith(b'\x01\x00\x00\x00\xf0\x00\x00\x00')

    testdat = np.arange(127, dtype=np.int16)
    bcomp = DigitalSurfHandler._compress_data(testdat,nstreams=3)
    assert bcomp.startswith(b'\x03\x00\x00\x00V\x00\x00\x00C\x00\x00\x00'+
                            b'V\x00\x00\x00F\x00\x00\x00'+
                            b'R\x00\x00\x00B\x00\x00\x00')


def test_get_comment_dict():
    tdh = DigitalSurfHandler()
    tdh.signal_dict={'original_metadata':{
        'Object_0_Channel_0':{
            'Parsed':{
                'key_1': 1,
                'key_2':'2'
            }
        }
    }}

    assert tdh._get_comment_dict('auto')=={'key_1': 1,'key_2':'2'}
    assert tdh._get_comment_dict('off')=={}
    assert tdh._get_comment_dict('raw')=={'Object_0_Channel_0':{'Parsed':{'key_1': 1,'key_2':'2'}}}
    assert tdh._get_comment_dict('custom',custom={'a':0}) == {'a':0}

    #Goes to second dict if only this one's valid
    tdh.signal_dict={'original_metadata':{
        'Object_0_Channel_0':{'Header':{}},
        'Object_0_Channel_1':{'Header':'ObjHead','Parsed':{'key_1': '0'}},
    }}
    assert tdh._get_comment_dict('auto') == {'key_1': '0'}

    #Return empty if none valid
    tdh.signal_dict={'original_metadata':{
        'Object_0_Channel_0':{'Header':{}},
        'Object_0_Channel_1':{'Header':'ObjHead'},
    }}
    assert tdh._get_comment_dict('auto') == {}

    #Return dict-cast if a single field is named 'Parsed' (weird case)
    tdh.signal_dict={'original_metadata':{
        'Object_0_Channel_0':{'Header':{}},
        'Object_0_Channel_1':{'Header':'ObjHead','Parsed':'SomeContent'},
    }}
    assert tdh._get_comment_dict('auto') == {'Parsed':'SomeContent'}

@pytest.mark.parametrize("test_object", ["test_profile.pro", "test_spectra.pro", "test_spectral_map.sur", "test_spectral_map_compressed.sur", "test_spectrum.pro", "test_spectrum_compressed.pro", "test_surface.sur"])
def test_writetestobjects(tmp_path,test_object):
    """Test data integrity of load/save functions. Starting from externally-generated data (i.e. not from hyperspy)"""

    df = TEST_DATA_PATH.joinpath(test_object)

    d = hs.load(df)
    fn = tmp_path.joinpath(test_object)
    d.save(fn,is_special=False)
    d2 = hs.load(fn)
    d2.save(fn,is_special=False)
    d3 = hs.load(fn)

    assert np.allclose(d2.data,d.data)
    assert np.allclose(d2.data,d3.data)
    
    a = d.axes_manager.navigation_axes
    b = d2.axes_manager.navigation_axes
    c = d3.axes_manager.navigation_axes

    for ax,ax2,ax3 in zip(a,b,c):
        assert np.allclose(ax.axis,ax2.axis)
        assert np.allclose(ax.axis,ax3.axis)

    a = d.axes_manager.signal_axes
    b = d2.axes_manager.signal_axes
    c = d3.axes_manager.signal_axes

    for ax,ax2,ax3 in zip(a,b,c):
        assert np.allclose(ax.axis,ax2.axis)
        assert np.allclose(ax.axis,ax3.axis)

def test_writeRGB(tmp_path):
    
    df = TEST_DATA_PATH.joinpath("test_RGB.sur")
    d = hs.load(df)
    fn = tmp_path.joinpath("test_RGB.sur")
    d.save(fn,is_special=False)
    d2 = hs.load(fn)
    d2.save(fn,is_special=False)
    d3 = hs.load(fn)

    for k in ['R','G','B']:
        assert np.allclose(d2.data[k],d.data[k])
        assert np.allclose(d3.data[k],d.data[k])

    a = d.axes_manager.navigation_axes
    b = d2.axes_manager.navigation_axes
    c = d3.axes_manager.navigation_axes

    for ax,ax2,ax3 in zip(a,b,c):
        assert np.allclose(ax.axis,ax2.axis)
        assert np.allclose(ax.axis,ax3.axis)

    a = d.axes_manager.signal_axes
    b = d2.axes_manager.signal_axes
    c = d3.axes_manager.signal_axes

    for ax,ax2,ax3 in zip(a,b,c):
        assert np.allclose(ax.axis,ax2.axis)
        assert np.allclose(ax.axis,ax3.axis)

@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float64, np.uint8, np.uint16])
def test_writegeneric_validtypes(tmp_path,dtype):

    gen = hs.signals.Signal1D(np.arange(24,dtype=dtype))+25
    fgen = tmp_path.joinpath('test.pro')
    gen.save(fgen,overwrite=True)