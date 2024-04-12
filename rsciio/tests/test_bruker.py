import json
from pathlib import Path

import numpy as np
import pytest

from rsciio.bruker import file_reader
from rsciio.utils.tests import assert_deep_almost_equal

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")


test_files = [
    "30x30_instructively_packed_16bit_compressed.bcf",
    "16x16_12bit_packed_8bit.bcf",
    "P45_the_default_job.bcf",
    "test_TEM.bcf",
    "Hitachi_TM3030Plus.bcf",
    "over16bit.bcf",
    "bcf_v2_50x50px.bcf",
    "bcf-edx-ebsd.bcf",
]
np_files = ["30x30_16bit.npy", "30x30_16bit_ds.npy"]
spx_files = ["extracted_from_bcf.spx", "bruker_nano.spx"]

TEST_DATA_DIR = Path(__file__).parent / "data" / "bruker"


def test_load_16bit():
    # test bcf from hyperspy hs.load function level
    # some of functions can be not covered
    # it cant use cython parsing implementation, as it is not compiled
    filename = TEST_DATA_DIR / test_files[0]
    print("testing bcf instructively packed 16bit...")
    s = hs.load(filename)
    bse, hype = s
    # Bruker saves all images in true 16bit:
    assert bse.data.dtype == np.uint16
    assert bse.data.shape == (30, 30)
    np_filename = TEST_DATA_DIR / np_files[0]
    np.testing.assert_array_equal(hype.data[:, :, 222:224], np.load(np_filename))
    assert hype.data.shape == (30, 30, 2048)
    assert hype.axes_manager.navigation_shape == (30, 30)
    assert hype.axes_manager.signal_shape == (2048,)
    assert bse.metadata.get_item("Stage.x", full_path=False) == 66940.81
    assert hype.metadata.get_item("Stage.x", full_path=False) == 66940.81


def test_load_16bit_reduced():
    filename = TEST_DATA_DIR / test_files[0]
    print("testing downsampled 16bit bcf...")
    s = hs.load(filename, downsample=4, cutoff_at_kV=10)
    bse, hype = s
    # sem images are never downsampled
    assert bse.data.shape == (30, 30)
    assert bse.axes_manager.signal_shape == (30, 30)
    assert bse.axes_manager.navigation_shape == ()
    np_filename = TEST_DATA_DIR / np_files[1]
    np.testing.assert_array_equal(hype.data[:, :, 222:224], np.load(np_filename))
    assert hype.data.shape == (8, 8, 1047)
    assert hype.axes_manager.navigation_shape == (8, 8)
    assert hype.axes_manager.signal_shape == (1047,)
    # Bruker saves all images in true 16bit:
    assert bse.data.dtype == np.uint16
    # hypermaps should always return unsigned integers:
    assert str(hype.data.dtype)[0] == "u"


def test_load_16bit_cutoff_zealous():
    filename = TEST_DATA_DIR / test_files[0]
    print("testing downsampled 16bit bcf with cutoff_at_kV=zealous...")
    hype = hs.load(filename, cutoff_at_kV="zealous", select_type="spectrum_image")
    assert hype.data.shape == (30, 30, 2048)
    assert hype.axes_manager.navigation_shape == (30, 30)
    assert hype.axes_manager.signal_shape == (2048,)


def test_load_16bit_cutoff_auto():
    filename = TEST_DATA_DIR / test_files[0]
    print("testing downsampled 16bit bcf with cutoff_at_kV=auto...")
    hype = hs.load(filename, cutoff_at_kV="auto", select_type="spectrum_image")
    assert hype.data.shape == (30, 30, 2048)
    assert hype.axes_manager.navigation_shape == (30, 30)
    assert hype.axes_manager.signal_shape == (2048,)


def test_load_8bit():
    bse_sig_shapes = [(16, 16), (100, 75)]  # identical to hype nav shapes
    hype_sig_shapes = [(2048,), (2048,)]
    for i, bcffile in enumerate(test_files[1:3]):
        filename = TEST_DATA_DIR / bcffile
        print("testing simple 8bit bcf...")
        s = hs.load(filename)
        bse, hype = s[0], s[-1]
        # Bruker saves all images in true 16bit:
        assert bse.data.dtype == np.uint16
        assert bse.axes_manager.navigation_shape == ()
        assert bse.axes_manager.signal_shape == bse_sig_shapes[i]
        # hypermaps should always return unsigned integers:
        assert str(hype.data.dtype)[0] == "u"
        assert hype.axes_manager.navigation_shape == bse.axes_manager.signal_shape
        assert hype.axes_manager.signal_shape == hype_sig_shapes[i]


def test_hyperspy_wrap():
    pytest.importorskip("exspy", reason="exspy not installed.")
    filename = TEST_DATA_DIR / test_files[0]
    print("testing bcf wrap to hyperspy signal...")

    hype = hs.load(filename, select_type="spectrum_image")
    np.testing.assert_allclose(hype.axes_manager[0].scale, 1.66740910949362, atol=1e-12)
    np.testing.assert_allclose(hype.axes_manager[1].scale, 1.66740910949362, atol=1e-12)
    assert hype.axes_manager[1].units == "µm"
    np.testing.assert_allclose(hype.axes_manager[2].scale, 0.009999)
    np.testing.assert_allclose(hype.axes_manager[2].offset, -0.47225277)
    assert hype.axes_manager[2].units == "keV"
    assert hype.axes_manager[2].is_binned is True

    md_ref = {
        "Acquisition_instrument": {
            "SEM": {
                "beam_energy": 20,
                "magnification": 1819.22595,
                "Detector": {
                    "EDS": {
                        "elevation_angle": 35.0,
                        "detector_type": "XFlash 6|10",
                        "azimuth_angle": 90.0,
                        "real_time": 70.07298,
                        "energy_resolution_MnKa": 130.0,
                    }
                },
                "Stage": {
                    "tilt_alpha": 0.0,
                    "rotation": 326.10089,
                    "x": 66940.81,
                    "y": 54233.16,
                    "z": 39194.77,
                },
            }
        },
        "General": {
            "original_filename": "30x30_instructively_packed_16bit_compressed.bcf",
            "title": "EDX",
            "date": "2018-10-04",
            "time": "13:02:07",
            "FileIO": {
                "0": {
                    "operation": "load",
                    "hyperspy_version": hs.__version__,
                    "io_plugin": "rsciio.bruker",
                }
            },
        },
        "Sample": {
            "name": "chevkinite",
            "elements": [
                "Al",
                "C",
                "Ca",
                "Ce",
                "Fe",
                "Gd",
                "K",
                "Mg",
                "Na",
                "Nd",
                "O",
                "P",
                "Si",
                "Sm",
                "Th",
                "Ti",
            ],
            "xray_lines": [
                "Al_Ka",
                "C_Ka",
                "Ca_Ka",
                "Ce_La",
                "Fe_Ka",
                "Gd_La",
                "K_Ka",
                "Mg_Ka",
                "Na_Ka",
                "Nd_La",
                "O_Ka",
                "P_Ka",
                "Si_Ka",
                "Sm_La",
                "Th_Ma",
                "Ti_Ka",
            ],
        },
        "Signal": {"quantity": "X-rays (Counts)", "signal_type": "EDS_SEM"},
        "_HyperSpy": {
            "Folding": {
                "original_axes_manager": None,
                "original_shape": None,
                "signal_unfolded": False,
                "unfolded": False,
            }
        },
    }

    filename_omd = TEST_DATA_DIR / "30x30_original_metadata.json"
    with open(filename_omd) as fn:
        # original_metadata:
        omd_ref = json.load(fn)
    # delete FileIO timestamp since it's runtime dependent
    del hype.metadata.General.FileIO.Number_0.timestamp
    assert_deep_almost_equal(hype.metadata.as_dictionary(), md_ref)
    assert_deep_almost_equal(hype.original_metadata.as_dictionary(), omd_ref)
    assert hype.metadata.General.date == "2018-10-04"
    assert hype.metadata.General.time == "13:02:07"
    assert hype.metadata.Signal.quantity == "X-rays (Counts)"


def test_hyperspy_wrap_downsampled():
    filename = TEST_DATA_DIR / test_files[0]
    print("testing bcf wrap to hyperspy signal...")
    hype = hs.load(filename, select_type="spectrum_image", downsample=5)
    np.testing.assert_allclose(
        hype.axes_manager[0].scale, 8.337045547468101, atol=1e-12
    )
    np.testing.assert_allclose(
        hype.axes_manager[1].scale, 8.337045547468101, atol=1e-12
    )
    assert hype.axes_manager[1].units == "µm"


def test_get_mode():
    filename = TEST_DATA_DIR / test_files[0]
    s = hs.load(filename, select_type="spectrum_image", instrument="SEM")
    assert s.metadata.Signal.signal_type == "EDS_SEM"
    assert isinstance(s, hs.signals.Signal1D)

    filename = TEST_DATA_DIR / test_files[0]
    s = hs.load(filename, select_type="spectrum_image", instrument="TEM")
    assert s.metadata.Signal.signal_type == "EDS_TEM"
    assert isinstance(s, hs.signals.Signal1D)

    filename = TEST_DATA_DIR / test_files[0]
    s = hs.load(filename, select_type="spectrum_image")
    assert s.metadata.Signal.signal_type == "EDS_SEM"
    assert isinstance(s, hs.signals.Signal1D)

    filename = TEST_DATA_DIR / test_files[3]
    s = hs.load(filename, select_type="spectrum_image")
    assert s.metadata.Signal.signal_type == "EDS_TEM"
    assert isinstance(s, hs.signals.Signal1D)


def test_wrong_file():
    filename = TEST_DATA_DIR / "Nope.bcf"
    with pytest.raises(TypeError):
        hs.load(filename)


def test_fast_bcf():
    thingy = pytest.importorskip("rsciio.bruker.unbcf_fast")
    from rsciio.bruker import _api

    for bcffile in test_files:
        filename = TEST_DATA_DIR / bcffile
        thingy = _api.BCF_reader(filename)
        for j in range(2, 5, 1):
            print("downsampling:", j)
            _api.fast_unbcf = True  # manually enabling fast parsing
            hmap1 = thingy.parse_hypermap(downsample=j)  # using cython
            _api.fast_unbcf = False  # manually disabling fast parsing
            hmap2 = thingy.parse_hypermap(downsample=j)  # py implementation
            np.testing.assert_array_equal(hmap1, hmap2)


def test_decimal_regex():
    from rsciio.utils.tools import sanitize_msxml_float

    dummy_xml_positive = [
        b"<dummy_tag>85,658</dummy_tag>",
        b"<dummy_tag>85,658E-8</dummy_tag>",
        b"<dummy_tag>-85,658E-8</dummy_tag>",
        b"<dum_tag>-85.658</dum_tag>",  # negative check
        b"<dum_tag>85.658E-8</dum_tag>",
    ]  # negative check
    dummy_xml_negative = [
        b"<dum_tag>12,25,23,45,56,12,45</dum_tag>",
        b"<dum_tag>12e1,23,-24E-5</dum_tag>",
    ]
    for i in dummy_xml_positive:
        assert b"85.658" in sanitize_msxml_float(i)
    for j in dummy_xml_negative:
        assert b"." not in sanitize_msxml_float(j)


def test_all_spx_loads():
    signal_shape = [(1548,), (4096,)]
    for i, spxfile in enumerate(spx_files):
        filename = TEST_DATA_DIR / spxfile
        s = hs.load(filename)
        assert s.data.dtype == np.uint64
        assert s.metadata.Signal.signal_type == "EDS_SEM"
        assert s.axes_manager.navigation_shape == ()
        assert s.axes_manager.signal_shape == signal_shape[i]


def test_stand_alone_spx():
    filename = TEST_DATA_DIR / "bruker_nano.spx"
    s = hs.load(filename)
    assert s.metadata.Sample.elements == ["Fe", "S", "Cu"]
    assert s.metadata.Acquisition_instrument.SEM.Detector.EDS.live_time == 7.385


def test_bruker_XRF():
    # See https://github.com/hyperspy/hyperspy/issues/2689
    # Bruker M6 Jetstream SPX
    filename = TEST_DATA_DIR / "bruker_m6_jetstream_file_example.spx"
    s = hs.load(filename)
    assert s.metadata.Acquisition_instrument.TEM.Detector.EDS.live_time == 28.046
    assert s.metadata.Acquisition_instrument.TEM.beam_energy == 50
    assert s.axes_manager.signal_shape == (4096,)
    assert s.axes_manager.navigation_shape == ()


def test_unsupported_extension():
    with pytest.raises(ValueError):
        file_reader("fname.unsupported_extension")
