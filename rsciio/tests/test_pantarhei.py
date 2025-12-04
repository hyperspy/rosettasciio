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


import io
import logging
from pathlib import Path

import numpy as np
import pytest

from rsciio.pantarhei._restricted_unpickling import (
    InvalidPickleError,
    read_pickled_array,
)
from rsciio.utils.tests import assert_deep_almost_equal

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
exspy = pytest.importorskip("exspy", reason="exspy not installed")


TEST_DATA_PATH = Path(__file__).parent / "data" / "pantarhei"


def test_metadata_prz():
    md = {
        "General": {"title": "O", "original_filename": "panta_rhei_sample.prz"},
        "Signal": {"signal_type": ""},
        "Acquisition_instrument": {
            "TEM": {
                "beam_energy": 200.0,
                "acquisition_mode": "STEM",
                "magnification": 10000000,
                "camera_length": 0.02,
            }
        },
    }
    am = {
        "axis-0": {
            "_type": "UniformDataAxis",
            "name": "Y",
            "units": "m",
            "navigate": False,
            "is_binned": False,
            "size": 16,
            "scale": 7.795828292907633e-09,
            "offset": 0.0,
        },
        "axis-1": {
            "_type": "UniformDataAxis",
            "name": "X",
            "units": "m",
            "navigate": False,
            "is_binned": False,
            "size": 16,
            "scale": 7.795828292907633e-09,
            "offset": 0.0,
        },
    }

    s = hs.load(TEST_DATA_PATH / "panta_rhei_sample.prz")

    md_file = s.metadata.as_dictionary()
    md_file.pop("_HyperSpy")
    md_file["General"].pop("FileIO")
    assert_deep_almost_equal(md_file, md)
    assert_deep_almost_equal(s.axes_manager.as_dictionary(), am)
    assert s.data.shape == (16, 16)
    assert s.data.max() == 40571
    assert s.data.min() == 36193
    np.testing.assert_almost_equal(s.data.std(), 1025.115644550)


def test_save_load_cycle(tmp_path):
    fname = tmp_path / "test_file.prz"

    s = hs.load(TEST_DATA_PATH / "panta_rhei_sample.prz")
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)
    assert s2.metadata.Signal.signal_type == s.metadata.Signal.signal_type


def test_save_load_cycle_new_signal_1D_nav1(tmp_path):
    fname = tmp_path / "test_file_new_signal_1D_nav1.prz"
    data = np.arange(20).reshape(2, 10)
    s = hs.signals.Signal1D(data)
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname, allow_restricted_pickle=True)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)


def test_save_load_cycle_new_signal_1D_nav2(tmp_path):
    fname = tmp_path / "test_file_new_signal1D_nav2.prz"
    data = np.arange(100).reshape(2, 5, 10)
    s = hs.signals.Signal2D(data)
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)


def test_save_load_cycle_new_signal_2D(tmp_path):
    fname = tmp_path / "test_file_new_signal2D.prz"
    data = np.arange(100).reshape(2, 5, 10)
    s = hs.signals.Signal2D(data)
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)


def test_save_load_cycle_new_signal_EELS(tmp_path):
    fname = tmp_path / "test_file_new_signal2D.prz"
    data = np.arange(100).reshape(2, 5, 10)
    s = exspy.signals.EELSSpectrum(data)
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname, allow_restricted_pickle=True)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)


def test_save_load_cycle_new_signal_EELS_aperture_out(tmp_path):
    fname = tmp_path / "test_file_new_signal2D_aperture_out.prz"
    data = np.arange(100).reshape(2, 5, 10)
    s = exspy.signals.EELSSpectrum(data)
    s.metadata.set_item("Acquisition_instrument.TEM.Detector.EELS.aperture", "Out")
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname, allow_restricted_pickle=True)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)
    assert s2.metadata.Acquisition_instrument.TEM.Detector.EELS.aperture == "Out"


def test_metadata_STEM(tmp_path):
    fname = tmp_path / "test_file_new_signal_metadata_STEM.prz"
    data = np.arange(20).reshape(2, 10)
    s = exspy.signals.EELSSpectrum(data)
    # Set some metadata
    md = {
        "Acquisition_instrument": {
            "TEM": {
                "beam_energy": 200.0,
                "acquisition_mode": "STEM",
                "magnification": 500000,
                "camera_length": 200,
                "convergence_angle": 20,
                "Detector": {
                    "EELS": {"collection_angle": 60, "aperture": 5},
                },
            },
        },
    }

    s.metadata.add_dictionary(md)
    s.metadata.General.add_dictionary({"date": "2022-07-08", "time": "16:00"})
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname, allow_restricted_pickle=True)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)

    assert_deep_almost_equal(
        s2.metadata.Acquisition_instrument.as_dictionary(),
        s.metadata.Acquisition_instrument.as_dictionary(),
    )


def test_metadata_TEM(tmp_path):
    fname = tmp_path / "test_file_new_signal_metadata_TEM.prz"
    data = np.arange(20).reshape(2, 10)
    s = exspy.signals.EELSSpectrum(data)
    # Set some metadata
    md = {
        "Acquisition_instrument": {
            "TEM": {
                "beam_energy": 200.0,
                "acquisition_mode": "TEM",
                "magnification": 500000,
                "camera_length": 200,
                "Detector": {
                    "EELS": {"collection_angle": 60, "aperture": 5},
                },
            },
        },
    }

    s.metadata.add_dictionary(md)
    s.metadata.General.add_dictionary({"date": "2022-07-08", "time": "16:00"})
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname, allow_restricted_pickle=True)
    np.testing.assert_allclose(s2.data, s.data)
    assert isinstance(s2, s.__class__)

    assert_deep_almost_equal(
        s2.metadata.Acquisition_instrument.as_dictionary(),
        s.metadata.Acquisition_instrument.as_dictionary(),
    )


def test_legacy_prz_allow_restricted_pickle_flag(caplog):
    """
    For prz files saved with Panta Rhei <24.03
    pickle must be allowed to read the meta data
    """
    with caplog.at_level(logging.WARNING):
        s1 = hs.load(
            TEST_DATA_PATH / "panta_rhei_sample_legacy.prz",
            allow_restricted_pickle=True,
        )
        assert "security risk" in caplog.text
    assert "TEM" in s1.metadata["Acquisition_instrument"]

    with caplog.at_level(logging.WARNING):
        s2 = hs.load(
            TEST_DATA_PATH / "panta_rhei_sample_legacy.prz",
            allow_restricted_pickle=False,
        )
        assert "no meta data" in caplog.text
    # signal data is the same, but without pickle
    # the original meta data can't be loaded
    np.testing.assert_allclose(s2.data, s1.data)
    assert "Acquisition_instrument" not in s2.metadata


def test_restricted_unpickler():
    md = {
        "some": {"thing": "ok"},
        "forbidden": Path("/a/path"),  # Path is NOT whitelisted
    }
    fp = io.BytesIO()
    # create infamous object array which needs to be unpickled
    np.save(fp, np.array([md], dtype=object))
    fp.seek(0)
    # exception message contains offending class name `Path`
    with pytest.raises(InvalidPickleError, match="Path"):
        read_pickled_array(fp)

    # non-object arrays don't need to be unpickled
    fp2 = io.BytesIO()
    np.save(fp2, np.arange(3))
    fp2.seek(0)
    with pytest.raises(ValueError, match="not contain pickled data"):
        read_pickled_array(fp2)
