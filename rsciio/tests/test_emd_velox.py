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


# The EMD format is a hdf5 standard proposed at Lawrence Berkeley
# National Lab (see https://emdatasets.com/ for more information).
# NOT to be confused with the FEI EMD format which was developed later.

import gc
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from dateutil import tz

from rsciio.utils.tests import assert_deep_almost_equal

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
pytest.importorskip("sparse")


TEST_DATA_PATH = Path(__file__).parent / "data" / "emd"


def _generate_parameters():
    parameters = []
    for lazy in [True, False]:
        for sum_EDS_detectors in [True, False]:
            parameters.append([lazy, sum_EDS_detectors])
    return parameters


class TestFeiEMD:
    fei_files_path = TEST_DATA_PATH / "fei_emd_files"

    @classmethod
    def setup_class(cls):
        import zipfile

        zipf = TEST_DATA_PATH / "fei_emd_files.zip"
        with zipfile.ZipFile(zipf, "r") as zipped:
            zipped.extractall(cls.fei_files_path)

    @classmethod
    def teardown_class(cls):
        gc.collect()
        shutil.rmtree(cls.fei_files_path)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_image(self, lazy):
        stage = {
            "tilt_alpha": 0.006,
            "tilt_beta": 0.000,
            "x": -0.000009,
            "y": 0.000144,
            "z": 0.000029,
        }
        md = {
            "Acquisition_instrument": {
                "TEM": {
                    "beam_energy": 200.0,
                    "camera_length": 98.0,
                    "magnification": 40000.0,
                    "microscope": "Talos",
                    "Stage": stage,
                }
            },
            "General": {
                "original_filename": "fei_emd_image.emd",
                "date": "2017-03-06",
                "time": "09:56:41",
                "time_zone": "BST",
                "title": "HAADF",
                "FileIO": {
                    "0": {
                        "operation": "load",
                        "hyperspy_version": hs.__version__,
                        "io_plugin": "rsciio.emd",
                    }
                },
            },
            "Signal": {"signal_type": ""},
            "_HyperSpy": {
                "Folding": {
                    "original_axes_manager": None,
                    "original_shape": None,
                    "signal_unfolded": False,
                    "unfolded": False,
                }
            },
        }

        # Update time and time_zone to local ones
        md["General"]["time_zone"] = tz.tzlocal().tzname(datetime.today())
        dt = datetime.fromtimestamp(1488794201, tz=tz.tzutc())
        date, time = dt.astimezone(tz.tzlocal()).isoformat().split("+")[0].split("T")
        md["General"]["date"] = date
        md["General"]["time"] = time

        signal = hs.load(self.fei_files_path / "fei_emd_image.emd", lazy=lazy)
        # delete timestamp from metadata since it's runtime dependent
        del signal.metadata.General.FileIO.Number_0.timestamp
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        fei_image = np.load(self.fei_files_path / "fei_emd_image.npy")
        assert signal.axes_manager[0].name == "x"
        assert signal.axes_manager[0].units == "µm"
        assert signal.axes_manager[0].is_binned is False
        np.testing.assert_allclose(signal.axes_manager[0].scale, 0.00530241, rtol=1e-5)
        assert signal.axes_manager[1].name == "y"
        assert signal.axes_manager[1].units == "µm"
        assert signal.axes_manager[1].is_binned is False
        np.testing.assert_allclose(signal.axes_manager[1].scale, 0.00530241, rtol=1e-5)
        np.testing.assert_allclose(signal.data, fei_image)
        assert_deep_almost_equal(signal.metadata.as_dictionary(), md)
        assert isinstance(signal, hs.signals.Signal2D)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_spectrum(self, lazy):
        signal = hs.load(self.fei_files_path / "fei_emd_spectrum.emd", lazy=lazy)
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        fei_spectrum = np.load(self.fei_files_path / "fei_emd_spectrum.npy")
        np.testing.assert_equal(signal.data, fei_spectrum)
        assert isinstance(signal, hs.signals.Signal1D)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_si(self, lazy):
        signal = hs.load(self.fei_files_path / "fei_emd_si.emd", lazy=lazy)
        if lazy:
            assert signal[1]._lazy
            signal[1].compute(close_file=True)
        fei_si = np.load(self.fei_files_path / "fei_emd_si.npy")
        np.testing.assert_equal(signal[1].data, fei_si)
        assert isinstance(signal[1], hs.signals.Signal1D)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_si_non_square_10frames(self, lazy):
        s = hs.load(
            self.fei_files_path / "fei_SI_SuperX-HAADF_10frames_10x50.emd",
            lazy=lazy,
        )
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert signal.metadata.Signal.signal_type == "EDS_TEM"
        assert isinstance(signal, hs.signals.Signal1D)
        assert signal.axes_manager[0].name == "x"
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[1].name == "y"
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[2].name == "X-ray energy"
        assert signal.axes_manager[2].size == 4096
        assert signal.axes_manager[2].units == "keV"
        np.testing.assert_allclose(signal.axes_manager[2].scale, 0.005, atol=1e-5)

        signal0 = s[0]
        if lazy:
            assert signal0._lazy
            signal0.compute(close_file=True)
        assert isinstance(signal0, hs.signals.Signal2D)
        assert signal0.axes_manager[0].name == "x"
        assert signal0.axes_manager[0].size == 10
        assert signal0.axes_manager[0].units == "nm"
        np.testing.assert_allclose(signal0.axes_manager[0].scale, 1.234009, atol=1e-5)
        assert signal0.axes_manager[1].name == "y"
        assert signal0.axes_manager[1].size == 50
        assert signal0.axes_manager[1].units == "nm"

        s = hs.load(
            self.fei_files_path / "fei_SI_SuperX-HAADF_10frames_10x50.emd",
            lazy=lazy,
            load_SI_image_stack=True,
        )
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert signal.metadata.Signal.signal_type == "EDS_TEM"
        assert isinstance(signal, hs.signals.Signal1D)
        assert signal.axes_manager[0].name == "x"
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[1].name == "y"
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[2].name == "X-ray energy"
        assert signal.axes_manager[2].size == 4096
        assert signal.axes_manager[2].units == "keV"
        np.testing.assert_allclose(signal.axes_manager[2].scale, 0.005, atol=1e-5)

        signal0 = s[0]
        if lazy:
            assert signal0._lazy
            signal0.compute(close_file=True)
        assert isinstance(signal0, hs.signals.Signal2D)
        assert signal0.axes_manager[0].name == "Time"
        assert signal0.axes_manager[0].size == 10
        assert signal0.axes_manager[0].units == "s"
        assert signal0.axes_manager[1].name == "x"
        assert signal0.axes_manager[1].size == 10
        assert signal0.axes_manager[1].units == "nm"
        np.testing.assert_allclose(signal0.axes_manager[1].scale, 1.234009, atol=1e-5)
        assert signal0.axes_manager[2].name == "y"
        assert signal0.axes_manager[2].size == 50
        assert signal0.axes_manager[2].units == "nm"

        s = hs.load(
            self.fei_files_path / "fei_SI_SuperX-HAADF_10frames_10x50.emd",
            sum_frames=False,
            SI_dtype=np.uint8,
            rebin_energy=256,
            lazy=lazy,
        )
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert signal.metadata.Signal.signal_type == "EDS_TEM"
        assert isinstance(signal, hs.signals.Signal1D)
        assert signal.axes_manager.navigation_shape == (10, 50, 10)
        assert signal.axes_manager[0].name == "x"
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[1].name == "y"
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[2].name == "Time"
        assert signal.axes_manager[2].size == 10
        assert signal.axes_manager[2].units == "s"
        np.testing.assert_allclose(signal.axes_manager[2].scale, 0.76800, atol=1e-5)
        assert signal.axes_manager[3].name == "X-ray energy"
        assert signal.axes_manager[3].size == 16
        assert signal.axes_manager[3].units == "keV"
        np.testing.assert_allclose(signal.axes_manager[3].scale, 1.28, atol=1e-5)

        s = hs.load(
            self.fei_files_path / "fei_SI_SuperX-HAADF_10frames_10x50.emd",
            sum_frames=False,
            last_frame=5,
            SI_dtype=np.uint8,
            rebin_energy=256,
            lazy=lazy,
        )
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert signal.metadata.Signal.signal_type == "EDS_TEM"
        assert isinstance(signal, hs.signals.Signal1D)
        assert signal.axes_manager.navigation_shape == (10, 50, 5)
        assert signal.axes_manager[0].name == "x"
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[1].name == "y"
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[2].name == "Time"
        assert signal.axes_manager[2].size == 5
        assert signal.axes_manager[2].units == "s"
        np.testing.assert_allclose(signal.axes_manager[2].scale, 0.76800, atol=1e-5)
        assert signal.axes_manager[3].name == "X-ray energy"
        assert signal.axes_manager[3].size == 16
        assert signal.axes_manager[3].units == "keV"
        np.testing.assert_allclose(signal.axes_manager[3].scale, 1.28, atol=1e-5)

        s = hs.load(
            self.fei_files_path / "fei_SI_SuperX-HAADF_10frames_10x50.emd",
            sum_frames=False,
            first_frame=4,
            SI_dtype=np.uint8,
            rebin_energy=256,
            lazy=lazy,
        )
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert signal.metadata.Signal.signal_type == "EDS_TEM"
        assert isinstance(signal, hs.signals.Signal1D)
        assert signal.axes_manager.navigation_shape == (10, 50, 6)
        assert signal.axes_manager[0].name == "x"
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[1].name == "y"
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[2].name == "Time"
        assert signal.axes_manager[2].size == 6
        assert signal.axes_manager[2].units == "s"
        np.testing.assert_allclose(signal.axes_manager[2].scale, 0.76800, atol=1e-5)
        assert signal.axes_manager[3].name == "X-ray energy"
        assert signal.axes_manager[3].size == 16
        assert signal.axes_manager[3].units == "keV"
        np.testing.assert_allclose(signal.axes_manager[3].scale, 1.28, atol=1e-5)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_si_non_square_20frames(self, lazy):
        s = hs.load(
            self.fei_files_path / "fei_SI_SuperX-HAADF_20frames_10x50.emd",
            lazy=lazy,
        )
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert signal.metadata.Signal.signal_type == "EDS_TEM"
        assert isinstance(signal, hs.signals.Signal1D)
        assert signal.axes_manager[0].name == "x"
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[1].name == "y"
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[2].name == "X-ray energy"
        assert signal.axes_manager[2].size == 4096
        assert signal.axes_manager[2].units == "keV"
        np.testing.assert_allclose(signal.axes_manager[2].scale, 0.005, atol=1e-5)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_si_non_square_20frames_2eV(self, lazy):
        s = hs.load(
            self.fei_files_path / "fei_SI_SuperX-HAADF_20frames_10x50_2ev.emd",
            lazy=lazy,
        )
        signal = s[1]
        if lazy:
            assert signal._lazy
            signal.compute(close_file=True)
        assert signal.metadata.Signal.signal_type == "EDS_TEM"
        assert isinstance(signal, hs.signals.Signal1D)
        assert signal.axes_manager[0].name == "x"
        assert signal.axes_manager[0].size == 10
        assert signal.axes_manager[0].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[0].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[1].name == "y"
        assert signal.axes_manager[1].size == 50
        assert signal.axes_manager[1].units == "nm"
        np.testing.assert_allclose(signal.axes_manager[1].scale, 1.234009, atol=1e-5)
        assert signal.axes_manager[2].name == "X-ray energy"
        assert signal.axes_manager[2].size == 4096
        assert signal.axes_manager[2].units == "keV"
        np.testing.assert_allclose(signal.axes_manager[2].scale, 0.002, atol=1e-5)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_fei_emd_si_frame_range(self, lazy):
        signal = hs.load(
            self.fei_files_path / "fei_emd_si.emd",
            first_frame=2,
            last_frame=4,
            lazy=lazy,
        )
        fei_si = np.load(self.fei_files_path / "fei_emd_si_frame.npy")
        if lazy:
            assert signal[1]._lazy
            signal[1].compute(close_file=True)
        np.testing.assert_equal(signal[1].data, fei_si)
        assert isinstance(signal[1], hs.signals.Signal1D)

    @pytest.mark.parametrize(["lazy", "sum_EDS_detectors"], _generate_parameters())
    def test_fei_si_4detectors(self, lazy, sum_EDS_detectors):
        fname = self.fei_files_path / "fei_SI_EDS-HAADF-4detectors_2frames.emd"
        signal = hs.load(fname, sum_EDS_detectors=sum_EDS_detectors, lazy=lazy)
        if lazy:
            assert signal[1]._lazy
            signal[1].compute(close_file=True)
        length = 6
        if not sum_EDS_detectors:
            length += 3
        assert len(signal) == length
        # TODO: add parsing azimuth_angle

    def test_fei_emd_ceta_camera(self):
        signal = hs.load(self.fei_files_path / "1532 Camera Ceta.emd")
        np.testing.assert_allclose(signal.data, np.zeros((64, 64)))
        assert isinstance(signal, hs.signals.Signal2D)
        date, time = self._convert_datetime(1512055942.914275).split("T")
        assert signal.metadata.General.date == date
        assert signal.metadata.General.time == time
        assert signal.metadata.General.time_zone == self._get_local_time_zone()

        signal = hs.load(self.fei_files_path / "1854 Camera Ceta.emd")
        np.testing.assert_allclose(signal.data, np.zeros((64, 64)))
        assert isinstance(signal, hs.signals.Signal2D)

    def _convert_datetime(self, unix_time):
        # Since we don't know the actual time zone of where the data have been
        # acquired, we convert the datetime to the local time for convenience
        dt = datetime.fromtimestamp(float(unix_time), tz=tz.tzutc())
        return dt.astimezone(tz.tzlocal()).isoformat().split("+")[0]

    def _get_local_time_zone(self):
        return tz.tzlocal().tzname(datetime.today())

    def time_loading_frame(self):
        # Run this function to check the loading time when loading EDS data
        import time

        frame_number = 100
        point_measurement = 15
        frame_offsets = np.arange(0, point_measurement * frame_number, frame_number)
        time_data = np.zeros_like(frame_offsets)
        path = Path("path to large dataset")
        for i, frame_offset in enumerate(frame_offsets):
            print(frame_offset + frame_number)
            t0 = time.time()
            hs.load(
                path / "large dataset.emd",
                first_frame=frame_offset,
                last_frame=frame_offset + frame_number,
            )
            t1 = time.time()
            time_data[i] = t1 - t0
        import matplotlib.pyplot as plt

        plt.plot(frame_offsets, time_data)
        plt.xlabel("Frame offset")
        plt.xlabel("Loading time (s)")


def test_fei_complex_loading():
    signal = hs.load(TEST_DATA_PATH / "fei_example_complex_fft.emd")
    assert isinstance(signal, hs.signals.ComplexSignal2D)


def test_fei_complex_loading_lazy():
    signal = hs.load(TEST_DATA_PATH / "fei_example_complex_fft.emd", lazy=True)
    assert isinstance(signal, hs.signals.ComplexSignal2D)


def test_fei_no_frametime():
    signal = hs.load(TEST_DATA_PATH / "fei_example_tem_stack.emd")
    assert isinstance(signal, hs.signals.Signal2D)
    assert signal.data.shape == (2, 3, 3)
    assert signal.axes_manager["Time"].scale == 0.8


def test_fei_dpc_loading():
    signals = hs.load(TEST_DATA_PATH / "fei_example_dpc_titles.emd")
    assert signals[0].metadata.General.title == "B-D"
    assert signals[1].metadata.General.title == "DPC"
    assert signals[2].metadata.General.title == "iDPC"
    assert signals[3].metadata.General.title == "DF4-C"
    assert signals[4].metadata.General.title == "DF4-B"
    assert signals[5].metadata.General.title == "DF4-A"
    assert signals[6].metadata.General.title == "A-C"
    assert signals[7].metadata.General.title == "DF4-D"
    assert signals[8].metadata.General.title == "Filtered iDPC"
    assert isinstance(signals[0], hs.signals.Signal2D)
    assert isinstance(signals[1], hs.signals.ComplexSignal2D)
    assert isinstance(signals[2], hs.signals.Signal2D)
    assert isinstance(signals[3], hs.signals.Signal2D)
    assert isinstance(signals[4], hs.signals.Signal2D)
    assert isinstance(signals[5], hs.signals.Signal2D)
    assert isinstance(signals[6], hs.signals.Signal2D)
    assert isinstance(signals[7], hs.signals.Signal2D)
    assert isinstance(signals[8], hs.signals.Signal2D)


@pytest.mark.parametrize("fname", ["FFTComplexEven.emd", "FFTComplexOdd.emd"])
def test_velox_fft_odd_number(fname):
    s = hs.load(TEST_DATA_PATH / fname)
    assert len(s) == 2

    shape = (36, 70) if fname == "FFTComplexEven.emd" else (32, 64)
    assert s[0].axes_manager.signal_shape == shape
    assert np.issubdtype(s[0].data.dtype, np.complex64)

    assert s[1].axes_manager.signal_shape == (128, 128)
    assert np.issubdtype(s[1].data.dtype, float)


class TestVeloxEMDv11:
    fei_files_path = TEST_DATA_PATH / "velox_emd_version11"

    @classmethod
    def setup_class(cls):
        import zipfile

        zipf = TEST_DATA_PATH / "velox_emd_version11.zip"
        with zipfile.ZipFile(zipf, "r") as zipped:
            zipped.extractall(cls.fei_files_path)

    @classmethod
    def teardown_class(cls):
        gc.collect()
        shutil.rmtree(cls.fei_files_path)

    @pytest.mark.parametrize("lazy", (True, False))
    def test_spectrum_images(self, lazy):
        s = hs.load(self.fei_files_path / "Test SI 16x16 215 kx.emd", lazy=lazy)
        assert s[-1].metadata.Sample.elements == ["C", "O", "Ca", "Cu"]
        assert len(s) == 10
        for i, v in enumerate(["C", "Ca", "O", "Cu", "HAADF", "EDS"]):
            assert s[i + 4].metadata.General.title == v

        assert s[-1].data.shape == (16, 16, 4096)

    def test_prune_data(self, caplog):
        with caplog.at_level(logging.WARNING):
            _ = hs.load(self.fei_files_path / "Test SI 16x16 ReducedData 215 kx.emd")

        assert "No spectrum stream is present" in caplog.text
