# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

import gc
import logging
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py not installed")

from rsciio.tests.generate_tofwerk_test_files import (  # noqa: E402
    NPEAKS,
    NSAMPLES,
    NSEGS,
    NWRITES,
    NX,
    make_opened_fixture,
    make_raw_fixture,
)
from rsciio.tofwerk import available_signals, file_reader  # noqa: E402

TEST_DATA = Path(__file__).parent / "data" / "tofwerk"
OPENED_FILE = TEST_DATA / "fib_sims_opened.h5"
RAW_FILE = TEST_DATA / "fib_sims_raw.h5"


def setup_module():
    TEST_DATA.mkdir(parents=True, exist_ok=True)
    make_raw_fixture(
        RAW_FILE, nwrites=NWRITES, nsegs=NSEGS, nx=NX, npeaks=NPEAKS, nsamples=NSAMPLES
    )
    make_opened_fixture(
        OPENED_FILE,
        nwrites=NWRITES,
        nsegs=NSEGS,
        nx=NX,
        npeaks=NPEAKS,
        nsamples=NSAMPLES,
    )


def teardown_module():
    gc.collect()
    for f in [OPENED_FILE, RAW_FILE]:
        if f.exists():
            f.unlink()


# FIBParams.ViewField = 1e-5 m = 10 µm; pixel_size = 10 µm / 16 = 0.625 µm
PIXEL_SIZE_UM = 10.0 / NX


class TestOpenedFile:
    """
    Opened file default returns 1 signal (sum spectrum).
    Pass signal="peak_data" or signal=["sum_spectrum", "peak_data"] for more.
    """

    def test_returns_one_signal_by_default(self):
        assert len(file_reader(OPENED_FILE)) == 1

    def test_returns_one_signal_with_peak_data(self):
        assert len(file_reader(OPENED_FILE, signal="peak_data")) == 1

    def test_returns_two_signals_with_list(self):
        assert len(file_reader(OPENED_FILE, signal=["sum_spectrum", "peak_data"])) == 2

    # ── Signal [0] default: sum spectrum ─────────────────────────────────

    def test_sum_spectrum_shape(self):
        assert file_reader(OPENED_FILE)[0]["data"].shape == (NSAMPLES,)

    def test_sum_spectrum_title(self):
        meta = file_reader(OPENED_FILE)[0]["metadata"]
        assert "sum spectrum" in meta["General"]["title"]

    def test_sum_spectrum_axis(self):
        ax = file_reader(OPENED_FILE)[0]["axes"][0]
        assert ax["name"] == "m/z"
        assert ax["units"] == "Da"
        assert "axis" in ax
        assert len(ax["axis"]) == NSAMPLES

    # ── Signal [0] with signal="peak_data": 4D peak data ─────────────────

    def test_peak_data_shape(self):
        assert file_reader(OPENED_FILE, signal="peak_data")[0]["data"].shape == (
            NWRITES,
            NSEGS,
            NX,
            NPEAKS,
        )

    def test_peak_data_dtype(self):
        assert (
            file_reader(OPENED_FILE, signal="peak_data")[0]["data"].dtype == np.float32
        )

    def test_peak_data_title(self):
        meta = file_reader(OPENED_FILE, signal="peak_data")[0]["metadata"]
        assert "peak data" in meta["General"]["title"]

    def test_four_axes(self):
        assert len(file_reader(OPENED_FILE, signal="peak_data")[0]["axes"]) == 4

    def test_axis_names(self):
        names = [
            ax["name"] for ax in file_reader(OPENED_FILE, signal="peak_data")[0]["axes"]
        ]
        assert names == ["depth", "y", "x", "m/z"]

    def test_depth_axis(self):
        ax = file_reader(OPENED_FILE, signal="peak_data")[0]["axes"][0]
        assert ax["units"] == "slice"
        assert ax["navigate"] is True
        assert ax["size"] == NWRITES
        assert ax["scale"] == 1
        assert ax["offset"] == 0

    def test_spatial_axes_units(self):
        axes = file_reader(OPENED_FILE, signal="peak_data")[0]["axes"]
        assert axes[1]["units"] == "µm"
        assert axes[2]["units"] == "µm"

    def test_spatial_axes_navigate(self):
        axes = file_reader(OPENED_FILE, signal="peak_data")[0]["axes"]
        assert axes[1]["navigate"] is True
        assert axes[2]["navigate"] is True

    def test_pixel_size_from_viewfield(self):
        # FIBParams.ViewField = 1e-5 m = 10 µm; 10 / 16 = 0.625 µm/pixel
        axes = file_reader(OPENED_FILE, signal="peak_data")[0]["axes"]
        np.testing.assert_allclose(axes[1]["scale"], PIXEL_SIZE_UM)
        np.testing.assert_allclose(axes[2]["scale"], PIXEL_SIZE_UM)

    def test_mass_axis_units(self):
        ax = file_reader(OPENED_FILE, signal="peak_data")[0]["axes"][3]
        assert ax["units"] == "Da"

    def test_mass_axis_navigate(self):
        ax = file_reader(OPENED_FILE, signal="peak_data")[0]["axes"][3]
        assert ax["navigate"] is False

    def test_mass_axis_values(self):
        ax = file_reader(OPENED_FILE, signal="peak_data")[0]["axes"][3]
        assert "axis" in ax
        np.testing.assert_allclose(ax["axis"], np.arange(1, NPEAKS + 1, dtype=float))

    # ── Metadata ─────────────────────────────────────────────────────────

    def test_signal_type(self):
        meta = file_reader(OPENED_FILE, signal="peak_data")[0]["metadata"]
        assert meta["Signal"]["signal_type"] == "FIB-SIMS"

    def test_binned_flag(self):
        # is_binned is set on the m/z signal axis, not in metadata.Signal
        ax = file_reader(OPENED_FILE, signal="peak_data")[0]["axes"][3]
        assert ax["is_binned"] is True

    def test_file_type_flag(self):
        meta = file_reader(OPENED_FILE, signal="peak_data")[0]["metadata"]
        assert (
            meta["Acquisition_instrument"]["FIB_SIMS"]["file_type"] == "pre-processed"
        )

    def test_voltage_kv(self):
        fib = file_reader(OPENED_FILE, signal="peak_data")[0]["metadata"][
            "Acquisition_instrument"
        ]["FIB_SIMS"]["FIB"]
        np.testing.assert_allclose(fib["voltage_kV"], 30.0)

    def test_fib_hardware(self):
        fib = file_reader(OPENED_FILE, signal="peak_data")[0]["metadata"][
            "Acquisition_instrument"
        ]["FIB_SIMS"]["FIB"]
        assert fib["hardware"] == "Tescan"

    def test_ion_mode(self):
        tof = file_reader(OPENED_FILE, signal="peak_data")[0]["metadata"][
            "Acquisition_instrument"
        ]["FIB_SIMS"]["ToF"]
        assert tof["ion_mode"] == "positive"

    def test_creation_date(self):
        assert (
            file_reader(OPENED_FILE)[0]["metadata"]["General"]["date"] == "2025-01-01"
        )

    def test_creation_time(self):
        assert file_reader(OPENED_FILE)[0]["metadata"]["General"]["time"] == "12:00:00"

    def test_creation_timezone(self):
        assert (
            file_reader(OPENED_FILE)[0]["metadata"]["General"]["time_zone"] == "-05:00"
        )

    def test_original_metadata_present(self):
        assert "TofDAQ Version" in file_reader(OPENED_FILE)[0]["original_metadata"]

    @pytest.mark.parametrize("lazy", [True, False])
    def test_lazy_loading(self, lazy):
        import dask.array as da

        result = file_reader(
            OPENED_FILE, lazy=lazy, signal=["sum_spectrum", "peak_data"]
        )
        # Sum spectrum: lazy when requested
        sum_data = result[0]["data"]
        if lazy:
            assert isinstance(sum_data, da.Array)
        else:
            assert isinstance(sum_data, np.ndarray)
        # Peak data: lazy when requested
        peak_data = result[1]["data"]
        if lazy:
            assert isinstance(peak_data, da.Array)
            np.testing.assert_array_equal(peak_data.shape, (NWRITES, NSEGS, NX, NPEAKS))
        else:
            assert isinstance(peak_data, np.ndarray)
            assert peak_data.shape == (NWRITES, NSEGS, NX, NPEAKS)


class TestRawFile:
    """
    Raw file default returns 1 signal (sum spectrum).
    """

    def test_returns_one_signal_by_default(self):
        assert len(file_reader(RAW_FILE)) == 1

    def test_sum_spectrum_shape(self):
        assert file_reader(RAW_FILE)[0]["data"].shape == (NSAMPLES,)

    def test_sum_spectrum_mass_axis(self):
        ax = file_reader(RAW_FILE)[0]["axes"][0]
        assert ax["name"] == "m/z"
        assert ax["units"] == "Da"
        assert "axis" in ax
        assert len(ax["axis"]) == NSAMPLES

    def test_file_type_flag(self):
        meta = file_reader(RAW_FILE)[0]["metadata"]
        assert meta["Acquisition_instrument"]["FIB_SIMS"]["file_type"] == "raw"

    def test_sum_spectrum_title(self):
        assert (
            "sum spectrum" in file_reader(RAW_FILE)[0]["metadata"]["General"]["title"]
        )

    @pytest.mark.parametrize("lazy", [True, False])
    def test_sum_spectrum_lazy(self, lazy):
        import dask.array as da

        data = file_reader(RAW_FILE, lazy=lazy)[0]["data"]
        if lazy:
            assert isinstance(data, da.Array)
        else:
            assert isinstance(data, np.ndarray)


class TestSignalPeakData:
    """
    Tests for signal="peak_data" on both raw and opened files.

    With ClockPeriod = SampleInterval in the fixture, clock_ratio = 1 and
    NbrWaveforms = 1, NActiveChannels = 1 → normalization = 1.  EventList
    values are direct ADC sample indices in [0, NSAMPLES).
    """

    def test_raw_default_returns_one_signal(self):
        assert len(file_reader(RAW_FILE)) == 1

    def test_raw_peak_data_returns_one_signal(self):
        assert len(file_reader(RAW_FILE, signal="peak_data")) == 1

    def test_peak_data_shape(self):
        sig = file_reader(RAW_FILE, signal="peak_data")[0]
        assert sig["data"].shape == (NWRITES, NSEGS, NX, NPEAKS)

    def test_peak_data_dtype(self):
        sig = file_reader(RAW_FILE, signal="peak_data")[0]
        assert sig["data"].dtype == np.float32

    def test_peak_data_title(self):
        sig = file_reader(RAW_FILE, signal="peak_data")[0]
        assert "peak data" in sig["metadata"]["General"]["title"]

    def test_peak_data_axes(self):
        axes = file_reader(RAW_FILE, signal="peak_data")[0]["axes"]
        assert len(axes) == 4
        assert [ax["name"] for ax in axes] == ["depth", "y", "x", "m/z"]

    def test_peak_data_mass_axis_values(self):
        ax = file_reader(RAW_FILE, signal="peak_data")[0]["axes"][3]
        np.testing.assert_allclose(ax["axis"], np.arange(1, NPEAKS + 1, dtype=float))

    def test_peak_data_non_negative(self):
        data = file_reader(RAW_FILE, signal="peak_data")[0]["data"]
        assert (data >= 0).all()

    def test_peak_data_values_match_algorithm(self):
        """Verify plumbing: reader output matches _compute_peak_data_from_file."""
        import h5py

        from rsciio.tofwerk._api import _compute_peak_data_from_file

        with h5py.File(RAW_FILE, "r") as f:
            expected = _compute_peak_data_from_file(f)
        result = file_reader(RAW_FILE, signal="peak_data")[0]["data"]
        np.testing.assert_array_equal(result, expected)

    def test_signal_peak_data_on_opened_file(self):
        """signal='peak_data' on opened file returns 1 signal (reads PeakData directly)."""
        sigs = file_reader(OPENED_FILE, signal="peak_data")
        assert len(sigs) == 1
        assert sigs[0]["data"].shape == (NWRITES, NSEGS, NX, NPEAKS)

    def test_lazy_raw_peak_data_raises(self):
        """lazy=True with signal='peak_data' on a raw file raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="lazy"):
            file_reader(RAW_FILE, signal="peak_data", lazy=True)

    def test_lazy_opened_peak_data_works(self):
        """lazy=True with signal='peak_data' on a pre-processed file works fine."""
        sigs = file_reader(OPENED_FILE, signal="peak_data", lazy=True)
        assert len(sigs) == 1
        import dask.array as da

        assert isinstance(sigs[0]["data"], da.Array)


class TestComputePeakDataNumpyFallback:
    """
    Covers the NumPy fallback path in ``compute_peak_data_from_eventlist``
    (the ``else`` branch reached when numba is not installed).

    numba is mocked out via ``sys.modules`` so the ImportError branch and the
    vectorised-NumPy loop are exercised even in environments that have numba.
    """

    @staticmethod
    def _run_no_numba(path):
        """Call compute_peak_data_from_eventlist with numba blocked."""
        import sys
        from unittest.mock import patch

        import h5py

        from rsciio.tofwerk._api import _compute_peak_data_from_file

        with h5py.File(path, "r") as f:
            with patch.dict(sys.modules, {"numba": None}):
                return _compute_peak_data_from_file(f)

    def test_numpy_fallback_shape(self, tmp_path):
        """NumPy fallback returns the correct (nwrites, nsegs, nx, npeaks) shape."""
        p = tmp_path / "fallback.h5"
        _make_minimal_tofdaq(p)
        result = self._run_no_numba(p)
        assert result.shape == (1, 2, 2, 2)  # nwrites=1, nsegs=2, nx=2, npeaks=2

    def test_numpy_fallback_dtype(self, tmp_path):
        """NumPy fallback returns float32."""
        p = tmp_path / "fallback_dtype.h5"
        _make_minimal_tofdaq(p)
        result = self._run_no_numba(p)
        assert result.dtype == np.float32

    def test_numpy_fallback_matches_numba(self, tmp_path):
        """NumPy fallback produces the same counts as the numba path."""

        import h5py

        from rsciio.tofwerk._api import _compute_peak_data_from_file

        p = tmp_path / "fallback_match.h5"
        _make_minimal_tofdaq(p)

        with h5py.File(p, "r") as f:
            numba_result = _compute_peak_data_from_file(f)

        numpy_result = self._run_no_numba(p)
        np.testing.assert_array_equal(numpy_result, numba_result)

    def test_numpy_fallback_empty_pixel(self, tmp_path):
        """Empty-pixel EventList entries are skipped without error (numpy path)."""
        p = tmp_path / "fallback_empty.h5"
        _make_minimal_tofdaq(p, empty_pixel=True)
        result = self._run_no_numba(p)
        assert result.shape[-1] == 2  # npeaks

    def test_numpy_fallback_out_of_range_events(self, tmp_path):
        """Out-of-range events are discarded without error (numpy path)."""
        p = tmp_path / "fallback_oor.h5"
        _make_minimal_tofdaq(p, out_of_range_pixel=True)
        result = self._run_no_numba(p)
        assert result.shape[-1] == 2  # npeaks


class TestSignalEventList:
    """Tests for signal='event_list' on raw files."""

    def test_returns_one_signal(self):
        assert len(file_reader(RAW_FILE, signal="event_list")) == 1

    def test_shape(self):
        sig = file_reader(RAW_FILE, signal="event_list")[0]
        assert sig["data"].shape == (NWRITES, NSEGS, NX)

    def test_dtype_object(self):
        sig = file_reader(RAW_FILE, signal="event_list")[0]
        assert sig["data"].dtype == object

    def test_elements_are_arrays(self):
        # Eager path: each element is a numpy array of TDC timestamps
        sig = file_reader(RAW_FILE, signal="event_list", lazy=False)[0]
        el = sig["data"]
        assert isinstance(el[0, 0, 0], np.ndarray)

    def test_three_axes(self):
        sig = file_reader(RAW_FILE, signal="event_list")[0]
        assert len(sig["axes"]) == 3

    def test_axis_names(self):
        axes = file_reader(RAW_FILE, signal="event_list")[0]["axes"]
        assert [ax["name"] for ax in axes] == ["depth", "y", "x"]

    def test_all_axes_navigate(self):
        axes = file_reader(RAW_FILE, signal="event_list")[0]["axes"]
        assert all(ax["navigate"] is True for ax in axes)

    def test_spatial_axes_units(self):
        axes = file_reader(RAW_FILE, signal="event_list")[0]["axes"]
        assert axes[1]["units"] == "µm"
        assert axes[2]["units"] == "µm"

    def test_spatial_axes_scale(self):
        axes = file_reader(RAW_FILE, signal="event_list")[0]["axes"]
        np.testing.assert_allclose(axes[1]["scale"], PIXEL_SIZE_UM)
        np.testing.assert_allclose(axes[2]["scale"], PIXEL_SIZE_UM)

    def test_title_contains_event_list(self):
        meta = file_reader(RAW_FILE, signal="event_list")[0]["metadata"]
        assert "event list" in meta["General"]["title"]

    def test_eager_data_is_ndarray(self):
        # Default (lazy=False): data is a numpy object array
        sig = file_reader(RAW_FILE, signal="event_list")[0]
        assert isinstance(sig["data"], np.ndarray)
        assert not sig.get("attributes", {}).get("_lazy", True)

    def test_lazy_data_is_dask_array(self):
        import dask.array as da

        sig = file_reader(RAW_FILE, signal="event_list", lazy=True)[0]
        assert isinstance(sig["data"], da.Array)
        assert sig["data"].dtype == object
        assert sig["data"].shape == (NWRITES, NSEGS, NX)

    def test_lazy_computed_element_is_array(self):
        # A single lazily computed pixel should be a numpy array of uint32

        sig = file_reader(RAW_FILE, signal="event_list", lazy=True)[0]
        val = sig["data"][0, 0, 0].compute()
        assert isinstance(val, np.ndarray)

    def test_not_available_on_opened_file(self):
        # Opened test file has no EventList — should raise ValueError
        with pytest.raises(ValueError, match="event_list|EventList"):
            file_reader(OPENED_FILE, signal="event_list")


class TestSignalAll:
    """Tests for signal='all'."""

    def test_opened_file_returns_three_signals(self):
        # Opened files have sum_spectrum + peak_data + fib_images (no EventList)
        assert len(file_reader(OPENED_FILE, signal="all")) == 3

    def test_raw_file_returns_four_signals(self):
        # Raw files have sum_spectrum + peak_data (reconstructed) + event_list + fib_images
        assert len(file_reader(RAW_FILE, signal="all")) == 4

    def test_sum_and_event_list_only(self):
        sigs = file_reader(RAW_FILE, signal=["sum_spectrum", "event_list"])
        assert len(sigs) == 2

    def test_all_opened_signal_order(self):
        sigs = file_reader(OPENED_FILE, signal="all")
        assert "sum spectrum" in sigs[0]["metadata"]["General"]["title"]
        assert "peak data" in sigs[1]["metadata"]["General"]["title"]
        assert "FIB SE images" in sigs[2]["metadata"]["General"]["title"]

    def test_all_raw_signal_order(self):
        sigs = file_reader(RAW_FILE, signal="all")
        assert "sum spectrum" in sigs[0]["metadata"]["General"]["title"]
        assert "peak data" in sigs[1]["metadata"]["General"]["title"]
        assert "event list" in sigs[2]["metadata"]["General"]["title"]
        assert "FIB SE images" in sigs[3]["metadata"]["General"]["title"]


class TestAvailableSignals:
    """Tests for the available_signals() helper."""

    def test_opened_file(self):
        sigs = available_signals(OPENED_FILE)
        assert sigs == ["sum_spectrum", "peak_data", "fib_images"]

    def test_raw_file(self):
        sigs = available_signals(RAW_FILE)
        assert sigs == ["sum_spectrum", "peak_data", "event_list", "fib_images"]

    def test_non_tofwerk_raises(self, tmp_path):
        p = tmp_path / "plain.h5"
        h5py.File(p, "w").close()
        with pytest.raises(IOError):
            available_signals(p)

    def test_no_peak_capability(self, tmp_path):
        """File with no EventList and no PeakData: 'peak_data' absent from result.

        Covers the False branch of:
          993->995  (has_peak_data or (has_event_list and has_peak_table) is False)
        """
        p = tmp_path / "no_peak_cap.h5"
        _make_minimal_tofdaq(p, include_eventlist=False)
        sigs = available_signals(p)
        assert "peak_data" not in sigs
        assert "sum_spectrum" in sigs

    def test_no_fib_images(self, tmp_path):
        """File without FIBImages group: 'fib_images' absent from result.

        Covers the False branch of:
          997->999  ("FIBImages" not in f or empty)
        """
        p = tmp_path / "no_fib.h5"
        _make_minimal_tofdaq(p, include_fibimages=False)
        sigs = available_signals(p)
        assert "fib_images" not in sigs


FIB_IMG_SIZE = 128  # test file has 128×128 SE images
N_FIB_IMAGES = 3  # test file has Image0000, Image0001, Image0002
# FIBParams.ViewField = 0.01 mm = 10 µm; FIB pixel size = 10 / 128
FIB_PIXEL_SIZE_UM = 10.0 / FIB_IMG_SIZE


class TestFIBImages:
    """Tests for signal='fib_images'."""

    def test_returns_one_signal(self):
        assert len(file_reader(OPENED_FILE, signal="fib_images")) == 1

    def test_data_shape(self):
        sig = file_reader(OPENED_FILE, signal="fib_images")[0]
        assert sig["data"].shape == (N_FIB_IMAGES, FIB_IMG_SIZE, FIB_IMG_SIZE)

    def test_axes(self):
        sig = file_reader(OPENED_FILE, signal="fib_images")[0]
        axes = sig["axes"]
        assert len(axes) == 3
        depth, y, x = axes
        assert depth["name"] == "depth"
        assert depth["size"] == N_FIB_IMAGES
        assert depth["navigate"] is True
        assert y["name"] == "y"
        assert y["size"] == FIB_IMG_SIZE
        assert y["navigate"] is False
        assert pytest.approx(y["scale"]) == FIB_PIXEL_SIZE_UM
        assert x["name"] == "x"
        assert x["size"] == FIB_IMG_SIZE
        assert x["navigate"] is False
        assert pytest.approx(x["scale"]) == FIB_PIXEL_SIZE_UM

    def test_title(self):
        sig = file_reader(OPENED_FILE, signal="fib_images")[0]
        assert "FIB SE images" in sig["metadata"]["General"]["title"]

    def test_also_works_on_raw_file(self):
        sig = file_reader(RAW_FILE, signal="fib_images")[0]
        assert sig["data"].shape == (N_FIB_IMAGES, FIB_IMG_SIZE, FIB_IMG_SIZE)

    def test_lazy(self):
        import dask.array as da

        sig = file_reader(OPENED_FILE, signal="fib_images", lazy=True)[0]
        assert isinstance(sig["data"], da.Array)
        assert sig["data"].shape == (N_FIB_IMAGES, FIB_IMG_SIZE, FIB_IMG_SIZE)

    def test_mixed_shape_images_are_filtered(self, tmp_path, caplog):
        """Images with a non-dominant shape (e.g. truncated last frame) are skipped."""
        import shutil

        p = tmp_path / "mixed_shape.h5"
        shutil.copy(OPENED_FILE, p)
        # Overwrite Image0002/Data with a smaller (truncated) image
        with h5py.File(p, "a") as f:
            del f["FIBImages/Image0002/Data"]
            f["FIBImages/Image0002"].create_dataset(
                "Data",
                data=np.zeros((FIB_IMG_SIZE - 2, FIB_IMG_SIZE), dtype=np.float64),
            )
        with caplog.at_level(logging.WARNING, logger="rsciio.tofwerk._api"):
            sig = file_reader(p, signal="fib_images")[0]
        # Only the 2 full-size images should be returned
        assert sig["data"].shape == (N_FIB_IMAGES - 1, FIB_IMG_SIZE, FIB_IMG_SIZE)
        assert any(
            "skipped" in r.message and "Image0002" in r.message for r in caplog.records
        )


class TestSignalValidation:
    """Tests for invalid signal parameter values."""

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Invalid signal"):
            file_reader(OPENED_FILE, signal="bad_value")

    def test_invalid_in_list_raises(self):
        with pytest.raises(ValueError, match="Invalid signal"):
            file_reader(OPENED_FILE, signal=["sum_spectrum", "bad_value"])

    def test_valid_enum_member_accepted(self):
        from rsciio.tofwerk._api import TofwerkSignal

        # Enum members are str subclasses and must work directly
        result = file_reader(OPENED_FILE, signal=TofwerkSignal.SUM_SPECTRUM)
        assert len(result) == 1


class TestDetection:
    """Tests for format detection and IOError on non-Tofwerk files."""

    def test_non_tofwerk_h5_raises(self, tmp_path):
        plain_h5 = tmp_path / "plain.h5"
        import h5py

        with h5py.File(plain_h5, "w") as f:
            f.create_group("SomeRandomGroup")
        with pytest.raises(IOError, match="not a Tofwerk"):
            file_reader(plain_h5)

    def test_opened_file_detected(self):
        meta = file_reader(OPENED_FILE, signal="peak_data")[0]["metadata"]
        assert (
            meta["Acquisition_instrument"]["FIB_SIMS"]["file_type"] == "pre-processed"
        )

    def test_raw_file_detected(self):
        meta = file_reader(RAW_FILE)[0]["metadata"]
        assert meta["Acquisition_instrument"]["FIB_SIMS"]["file_type"] == "raw"

    def test_plugin_discovered(self):
        from rsciio import IO_PLUGINS

        assert "Tofwerk" in [p["name"] for p in IO_PLUGINS]


# ---------------------------------------------------------------------------
# Helper: minimal valid TofDAQ HDF5 factory for edge-case tests
# ---------------------------------------------------------------------------


def _make_minimal_tofdaq(path, **kwargs):
    """
    Write a minimal TofDAQ HDF5 file at *path*.

    Keyword arguments
    -----------------
    timestring         : bytes – AcquisitionLog timestring (default: valid ISO-8601)
    hdf5_creation_time : bytes – root 'HDF5 File Creation Time' attr (omitted if absent)
    include_fibparams  : bool  – include FIBParams group (default True)
    include_viewfield  : bool  – include ViewField attr in FIBParams (default True)
    include_fibpressure: bool  – include FibParams/FibPressure (default True)
    include_satwarning : bool  – include FullSpectra/SaturationWarning (default True)
    include_eventlist  : bool  – include FullSpectra/EventList (default True)
    include_peakdata   : bool  – include PeakData/PeakData (default False)
    include_fibimages  : bool  – include FIBImages group (default True)
    mass_axis          : array – override MassAxis values
    mass_calib_p3      : float – if provided, add MassCalibration p3 attr
    clock_period       : float – if provided, add ClockPeriod attr to FullSpectra
    empty_pixel        : bool  – make pixel (0,0,0) have an empty EventList
    out_of_range_pixel : bool  – make pixel (0,0,1) have all out-of-range events
    """
    nwrites, nsegs, nx, nsamples, npeaks = 1, 2, 2, 8, 2
    sample_interval = 8.333e-10

    with h5py.File(path, "w") as f:
        # Root attributes
        f.attrs["TofDAQ Version"] = np.float32(1.99)
        f.attrs["NbrWrites"] = np.int32(nwrites)
        f.attrs["NbrSegments"] = np.int32(nsegs)
        f.attrs["NbrSamples"] = np.int32(nsamples)
        f.attrs["NbrPeaks"] = np.int32(npeaks)
        f.attrs["NbrWaveforms"] = np.int32(1)
        f.attrs["IonMode"] = b"positive"
        f.attrs["DAQ Hardware"] = b"TestDAQ"
        f.attrs["Computer ID"] = b"test"
        f.attrs["FiblysGUIVersion"] = b"1.0"
        f.attrs["Configuration File Contents"] = b"[TOFParameter]\nCh1Record=1\n"
        if "hdf5_creation_time" in kwargs:
            # Store as np.bytes_ (fixed-length) so h5py returns np.bytes_ on read,
            # exercising the isinstance branch in _parse_creation_time.
            f.attrs["HDF5 File Creation Time"] = np.bytes_(kwargs["hdf5_creation_time"])

        # AcquisitionLog
        log_dtype = np.dtype(
            [("timestamp", np.uint64), ("timestring", "S26"), ("logtext", "S256")]
        )
        timestring = kwargs.get("timestring", b"2025-06-01T08:00:00+00:00")
        log = np.array([(0, timestring, b"start")], dtype=log_dtype)
        f.create_group("AcquisitionLog").create_dataset("Log", data=log)

        # TimingData
        td = f.create_group("TimingData")
        td.attrs["TofPeriod"] = np.int32(9500)
        td.create_dataset("BufTimes", data=np.zeros((nwrites, nsegs), dtype=np.float64))

        # FullSpectra
        fs = f.create_group("FullSpectra")
        fs.attrs["MassCalibMode"] = np.int32(0)
        fs.attrs["MassCalibration p1"] = np.float64(812.2415)
        fs.attrs["MassCalibration p2"] = np.float64(222.0153)
        fs.attrs["SampleInterval"] = np.float64(sample_interval)
        fs.attrs["Single Ion Signal"] = np.float64(1.0)
        if "mass_calib_p3" in kwargs:
            fs.attrs["MassCalibration p3"] = np.float64(kwargs["mass_calib_p3"])
        if "clock_period" in kwargs:
            fs.attrs["ClockPeriod"] = np.float64(kwargs["clock_period"])

        mass_axis = kwargs.get(
            "mass_axis", np.linspace(0.0, 10.0, nsamples, dtype=np.float32)
        )
        fs.create_dataset("MassAxis", data=mass_axis)
        fs.create_dataset("SumSpectrum", data=np.ones(nsamples, dtype=np.float64))
        if kwargs.get("include_satwarning", True):
            fs.create_dataset(
                "SaturationWarning",
                data=np.zeros((nwrites, nsegs), dtype=np.uint8),
            )

        if kwargs.get("include_eventlist", True):
            vlen = h5py.vlen_dtype(np.uint16)
            el = fs.create_dataset("EventList", shape=(nwrites, nsegs, nx), dtype=vlen)
            rng = np.random.default_rng(99)
            for w in range(nwrites):
                for s in range(nsegs):
                    for x in range(nx):
                        if kwargs.get("empty_pixel") and (w, s, x) == (0, 0, 0):
                            el[w, s, x] = np.array([], dtype=np.uint16)
                        elif kwargs.get("out_of_range_pixel") and (w, s, x) == (
                            0,
                            0,
                            1,
                        ):
                            # All events well outside [0, nsamples)
                            el[w, s, x] = np.array([nsamples + 100], dtype=np.uint16)
                        else:
                            el[w, s, x] = rng.integers(0, nsamples, 3, dtype=np.uint16)

        # PeakData
        peak_dtype = np.dtype(
            [
                ("label", "S64"),
                ("mass", np.float32),
                ("lower integration limit", np.float32),
                ("upper integration limit", np.float32),
            ]
        )
        peaks = np.array(
            [(b"p0", 1.0, 0.5, 1.5), (b"p1", 2.0, 1.5, 2.5)], dtype=peak_dtype
        )
        pd_group = f.create_group("PeakData")
        pd_group.create_dataset("PeakTable", data=peaks)
        if kwargs.get("include_peakdata", False):
            pd_group.create_dataset(
                "PeakData",
                data=np.ones((nwrites, nsegs, nx, npeaks), dtype=np.float32),
            )

        # FIBImages + FIBParams
        if kwargs.get("include_fibimages", True):
            f.create_group("FIBImages")
        if kwargs.get("include_fibparams", True):
            fibparams = f.create_group("FIBParams")
            fibparams.attrs["FibHardware"] = b"TestFIB"
            fibparams.attrs["Voltage"] = np.float64(30000.0)
            fibparams.attrs["Current"] = np.float64(0.0)
            if kwargs.get("include_viewfield", True):
                fibparams.attrs["ViewField"] = np.float64(0.01)  # 10 µm = 0.01 mm

        # FibParams/FibPressure
        if kwargs.get("include_fibpressure", True):
            fp = f.create_group("FibParams/FibPressure")
            fp.create_dataset(
                "TwData", data=np.full((nwrites, 1), 1.7e-4, dtype=np.float64)
            )


# ---------------------------------------------------------------------------
# Tests: module __dir__
# ---------------------------------------------------------------------------


class TestModuleDir:
    def test_dir_contains_file_reader(self):
        import sys

        assert "file_reader" in dir(sys.modules["rsciio.tofwerk"])


# ---------------------------------------------------------------------------
# Tests: internal helpers
# ---------------------------------------------------------------------------


class TestInternalHelpers:
    def test_is_fib_sims_true(self):
        from rsciio.tofwerk._api import _is_fib_sims

        with h5py.File(OPENED_FILE, "r") as f:
            assert _is_fib_sims(f) is True

    def test_is_fib_sims_false_missing_fib_groups(self, tmp_path):
        from rsciio.tofwerk._api import _is_fib_sims

        p = tmp_path / "no_fib.h5"
        with h5py.File(p, "w") as f:
            f.attrs["TofDAQ Version"] = np.float32(1.0)
            f.create_group("FullSpectra")
            f.create_group("TimingData")
            f.create_group("AcquisitionLog")
            # No FIBImages or FIBParams
        with h5py.File(p, "r") as f:
            assert _is_fib_sims(f) is False

    def test_decode_str_passthrough(self):
        from rsciio.tofwerk._api import _decode

        assert _decode("already a string") == "already a string"

    def test_decode_attr_np_generic(self):
        from rsciio.tofwerk._api import _decode_attr

        result = _decode_attr(np.float64(3.14))
        assert isinstance(result, float)
        assert result == pytest.approx(3.14)

    def test_decode_attr_ndarray_string_dtype(self):
        from rsciio.tofwerk._api import _decode_attr

        result = _decode_attr(np.array([b"foo", b"bar"]))
        assert result == ["foo", "bar"]

    def test_decode_attr_ndarray_numeric(self):
        from rsciio.tofwerk._api import _decode_attr

        result = _decode_attr(np.array([1.0, 2.0]))
        assert result == [1.0, 2.0]

    def test_decode_attr_bytes(self):
        from rsciio.tofwerk._api import _decode_attr

        result = _decode_attr(np.bytes_(b"hello"))
        assert result == "hello"

    def test_parse_creation_time_str_timestring_no_timezone(self):
        """
        Cover two branches in _parse_creation_time that h5py/S26 cannot reach:
          84->86 : timestring already str (not bytes) → isinstance check is False
          89->107: datetime has no tzinfo → tz_str stays empty
        """
        from rsciio.tofwerk._api import _parse_creation_time

        class _MockLog:
            def __getitem__(self, idx):
                # Return as str, not bytes; no UTC offset
                return {"timestring": "2025-06-01T08:00:00"}

        class _MockFile:
            def __getitem__(self, key):
                return _MockLog()

            @property
            def attrs(self):
                return {}

        date_str, time_str, tz_str = _parse_creation_time(_MockFile())
        assert date_str == "2025-06-01"
        assert time_str == "08:00:00"
        assert tz_str == ""

    def test_parse_creation_time_hdf5_attr_as_str(self):
        """
        Cover branch 97->99: HDF5 File Creation Time attr returned as str by h5py
        (h5py 3.x default for plain Python bytes stored as attributes).
        """
        from rsciio.tofwerk._api import _parse_creation_time

        class _MockFile:
            def __getitem__(self, key):
                raise KeyError(key)  # AcquisitionLog missing → falls to except

            @property
            def attrs(self):
                # Return str (not bytes) as h5py 3.x would
                return {"HDF5 File Creation Time": "15.06.2024 09:30:00"}

        date_str, time_str, tz_str = _parse_creation_time(_MockFile())
        assert date_str == "2024-06-15"
        assert time_str == "09:30:00"
        assert tz_str == ""


# ---------------------------------------------------------------------------
# Tests: edge cases / fallback paths
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Coverage for optional/fallback code paths using minimal synthetic files."""

    def test_datetime_fallback_to_hdf5_creation_time(self, tmp_path):
        """AcquisitionLog timestring invalid → falls back to HDF5 File Creation Time."""
        p = tmp_path / "fallback_dt.h5"
        _make_minimal_tofdaq(
            p,
            timestring=b"not-a-valid-datetime!!",
            hdf5_creation_time=b"15.06.2024 09:30:00",
        )
        meta = file_reader(p)[0]["metadata"]["General"]
        assert meta["date"] == "2024-06-15"
        assert meta["time"] == "09:30:00"
        assert "time_zone" not in meta

    def test_datetime_both_sources_fail(self, tmp_path):
        """Both datetime sources invalid → warning logged, date/time fields absent."""
        p = tmp_path / "no_dt.h5"
        _make_minimal_tofdaq(
            p,
            timestring=b"bad",
            hdf5_creation_time=b"bad-date-format!!!!!",
        )
        general = file_reader(p)[0]["metadata"]["General"]
        assert "date" not in general
        assert "time" not in general

    def test_pixel_size_fallback_when_viewfield_absent(self, tmp_path):
        """ViewField attr missing → spatial axes are uncalibrated (no scale/units)."""
        p = tmp_path / "no_viewfield.h5"
        _make_minimal_tofdaq(p, include_viewfield=False)
        axes = file_reader(p, signal="event_list")[0]["axes"]
        # depth axis always has scale=1
        assert axes[0]["scale"] == pytest.approx(1.0)
        # spatial axes have no calibration when ViewField is absent
        assert "scale" not in axes[1]
        assert "scale" not in axes[2]

    def test_fibparams_absent(self, tmp_path):
        """FIBParams group absent → warning logged, no FIB key in metadata."""
        p = tmp_path / "no_fibparams.h5"
        _make_minimal_tofdaq(p, include_fibparams=False)
        meta = file_reader(p)[0]["metadata"]["Acquisition_instrument"]["FIB_SIMS"]
        assert "FIB" not in meta

    def test_mass_range_fallback_when_all_invalid(self, tmp_path):
        """All-zero mass axis → mass range defaults to [0.0, 0.0]."""
        p = tmp_path / "zero_mass.h5"
        _make_minimal_tofdaq(
            p, mass_axis=np.zeros(8, dtype=np.float32), include_eventlist=False
        )
        # Use an opened-style file so mass range comes from PeakTable (simpler path);
        # zero-mass axis test: raw file where valid mask is all-False
        _make_minimal_tofdaq(p, mass_axis=np.full(8, -1.0, dtype=np.float32))
        tof_meta = file_reader(p)[0]["metadata"]["Acquisition_instrument"]["FIB_SIMS"][
            "ToF"
        ]
        assert tof_meta["mass_range_Da"] == [0.0, 0.0]

    def test_chamber_pressure_absent(self, tmp_path):
        """FibPressure absent → chamber_pressure_Pa not in metadata."""
        p = tmp_path / "no_pressure.h5"
        _make_minimal_tofdaq(p, include_fibpressure=False)
        meta = file_reader(p)[0]["metadata"]["Acquisition_instrument"]["FIB_SIMS"]
        assert "chamber_pressure_Pa" not in meta

    def test_mass_calib_p3_included(self, tmp_path):
        """MassCalibration p3 present → included in ToF metadata."""
        p = tmp_path / "with_p3.h5"
        _make_minimal_tofdaq(p, mass_calib_p3=0.123)
        tof_meta = file_reader(p)[0]["metadata"]["Acquisition_instrument"]["FIB_SIMS"][
            "ToF"
        ]
        assert "mass_calib_p3" in tof_meta
        assert tof_meta["mass_calib_p3"] == pytest.approx(0.123)

    def test_saturation_warning_absent(self, tmp_path):
        """SaturationWarning dataset absent → not present in original_metadata."""
        p = tmp_path / "no_saturation.h5"
        _make_minimal_tofdaq(p, include_satwarning=False)
        orig = file_reader(p)[0]["original_metadata"]
        assert "SaturationWarning" not in orig

    def test_clock_ratio_fallback_when_clock_period_absent(self, tmp_path):
        """ClockPeriod attr absent → clock_ratio falls back to 1."""
        p = tmp_path / "no_clock.h5"
        # _make_minimal_tofdaq does not write ClockPeriod unless kwarg provided
        _make_minimal_tofdaq(p)
        result = file_reader(p, signal="peak_data")
        assert len(result) == 1
        assert result[0]["data"].shape[-1] == 2  # npeaks

    def test_empty_pixel_in_eventlist(self, tmp_path):
        """Pixel with empty EventList → skipped in peak_data reconstruction without error."""
        p = tmp_path / "empty_pixel.h5"
        _make_minimal_tofdaq(p, empty_pixel=True)
        result = file_reader(p, signal="peak_data")
        assert len(result) == 1

    def test_out_of_range_events_discarded(self, tmp_path):
        """Events outside [0, nsamples) → discarded in peak_data reconstruction."""
        p = tmp_path / "oor_events.h5"
        _make_minimal_tofdaq(p, out_of_range_pixel=True)
        result = file_reader(p, signal="peak_data")
        assert len(result) == 1

    def test_original_metadata_no_fibimages(self, tmp_path):
        """FIBImages group absent → not included in original_metadata."""
        p = tmp_path / "no_fibimages.h5"
        _make_minimal_tofdaq(p, include_fibimages=False, include_fibparams=False)
        orig = file_reader(p)[0]["original_metadata"]
        assert "FIBImages" not in orig

    def test_no_peak_table_in_metadata(self, tmp_path):
        """File without PeakData/PeakTable: peak_table absent from metadata and original_metadata.

        Covers the False branches of:
          443->446  (_build_metadata: 'PeakData/PeakTable' not in file)
          447->450  (_build_metadata: peak_table_list is None)
          480->484  (_build_original_metadata: 'PeakData/PeakTable' not in file)
        """
        p = tmp_path / "no_pt.h5"
        _make_minimal_tofdaq(p)
        with h5py.File(p, "a") as f:
            del f["PeakData/PeakTable"]
        result = file_reader(p)
        meta = result[0]["metadata"]
        assert "peak_table" not in meta.get("Signal", {})
        assert "PeakTable" not in result[0]["original_metadata"]

    def test_no_mass_axis_in_original_metadata(self, tmp_path):
        """File without FullSpectra/MassAxis: MassAxis absent from original_metadata.

        Covers the False branch of:
          494->496  (_build_original_metadata: 'FullSpectra/MassAxis' not in file)
        Calls _build_original_metadata directly because the public file_reader also
        reads MassAxis for the sum_spectrum signal, which would KeyError first.
        """
        from rsciio.tofwerk._api import _build_original_metadata

        p = tmp_path / "no_massaxis.h5"
        _make_minimal_tofdaq(p)
        with h5py.File(p, "r") as f:
            # Wrap the file so "FullSpectra/MassAxis" membership check returns False
            class _NoMassAxis:
                def __init__(self, real):
                    self._f = real

                @property
                def attrs(self):
                    return self._f.attrs

                def __contains__(self, key):
                    if key == "FullSpectra/MassAxis":
                        return False
                    return key in self._f

                def __getitem__(self, key):
                    return self._f[key]

            result = _build_original_metadata(_NoMassAxis(f))
        assert "MassAxis" not in result

    def test_no_fullspectra_timing_attrs_in_original_metadata(self, tmp_path):
        """FullSpectra has no SampleInterval or ClockPeriod: FullSpectra key absent.

        Covers the False branch of:
          503->506  (_build_original_metadata: fs_attrs is empty dict)
        """
        p = tmp_path / "no_timing.h5"
        _make_minimal_tofdaq(p)
        with h5py.File(p, "a") as f:
            del f["FullSpectra"].attrs["SampleInterval"]
        result = file_reader(p)
        assert "FullSpectra" not in result[0]["original_metadata"]

    def test_build_original_metadata_no_fullspectra(self):
        """_build_original_metadata with no FullSpectra group skips timing block.

        Covers the False branch of:
          496->506  (_build_original_metadata: 'FullSpectra' not in file)
        This path is unreachable via file_reader (blocked by _is_tofwerk_file),
        so the internal function is called directly with a mock.
        """
        from rsciio.tofwerk._api import _build_original_metadata

        class _MinimalFile:
            attrs = {}

            def __contains__(self, key):
                return False

        result = _build_original_metadata(_MinimalFile())
        assert "FullSpectra" not in result
        assert "MassAxis" not in result

    def test_nx_fallback_to_nsegs(self, tmp_path):
        """nx falls back to NbrSegments when neither PeakData/PeakData nor EventList exists."""
        p = tmp_path / "nx_fallback.h5"
        _make_minimal_tofdaq(p, include_eventlist=False)
        result = file_reader(p)
        assert len(result) == 1

    def test_want_peak_not_available_raises(self, tmp_path):
        """signal='peak_data' on file with no EventList and no PeakData raises ValueError."""
        p = tmp_path / "no_peak.h5"
        _make_minimal_tofdaq(p, include_eventlist=False)
        with pytest.raises(ValueError, match="peak_data"):
            file_reader(p, signal="peak_data")

    def test_want_fib_not_available_raises(self, tmp_path):
        """signal='fib_images' on file with empty FIBImages group raises ValueError."""
        p = tmp_path / "no_fib.h5"
        _make_minimal_tofdaq(p, include_fibimages=False)
        with pytest.raises(ValueError, match="fib_images|FIBImages"):
            file_reader(p, signal="fib_images")

    def test_available_signals_info_logged(self, caplog):
        """Loading only sum_spectrum from opened file logs available signals at INFO."""
        with caplog.at_level(logging.INFO, logger="rsciio.tofwerk._api"):
            file_reader(OPENED_FILE, signal="sum_spectrum")
        assert any(
            "peak_data" in r.message and r.levelname == "INFO" for r in caplog.records
        )


# ---------------------------------------------------------------------------
# Helper: minimal file with peaks in non-ascending mass order
# ---------------------------------------------------------------------------


def _make_unsorted_peaks_tofdaq(path, with_peak_data=True):
    """
    Write a minimal TofDAQ HDF5 file with peaks stored in reverse mass order.

    Peaks: [(b"p1", 2.0, 1.5, 2.5), (b"p0", 1.0, 0.5, 1.5)]  — mass 2.0 first.

    Parameters
    ----------
    path : pathlib.Path or str
    with_peak_data : bool
        If True, include PeakData/PeakData (pre-processed file).
        If False, include FullSpectra/EventList (raw file).
    """
    nwrites, nsegs, nx, nsamples, npeaks = 2, 2, 2, 8, 2
    sample_interval = 8.333e-10
    peak_dtype = np.dtype(
        [
            ("label", "S64"),
            ("mass", np.float32),
            ("lower integration limit", np.float32),
            ("upper integration limit", np.float32),
        ]
    )
    # Masses stored in DESCENDING order to trigger the sort path
    peaks = np.array([(b"p1", 2.0, 1.5, 2.5), (b"p0", 1.0, 0.5, 1.5)], dtype=peak_dtype)
    with h5py.File(path, "w") as f:
        f.attrs["TofDAQ Version"] = np.float32(1.99)
        f.attrs["NbrWrites"] = np.int32(nwrites)
        f.attrs["NbrSegments"] = np.int32(nsegs)
        f.attrs["NbrPeaks"] = np.int32(npeaks)
        f.attrs["NbrWaveforms"] = np.int32(1)
        f.attrs["IonMode"] = b"positive"
        f.attrs["DAQ Hardware"] = b"TestDAQ"
        f.attrs["Computer ID"] = b"test"
        f.attrs["FiblysGUIVersion"] = b"1.0"
        f.attrs["Configuration File Contents"] = b"[TOFParameter]\nCh1Record=1\n"
        log_dtype = np.dtype(
            [("timestamp", np.uint64), ("timestring", "S26"), ("logtext", "S256")]
        )
        log = np.array([(0, b"2025-06-01T08:00:00+00:00", b"start")], dtype=log_dtype)
        f.create_group("AcquisitionLog").create_dataset("Log", data=log)
        td = f.create_group("TimingData")
        td.attrs["TofPeriod"] = np.int32(9500)
        td.create_dataset("BufTimes", data=np.zeros((nwrites, nsegs), dtype=np.float64))
        fs = f.create_group("FullSpectra")
        fs.attrs["MassCalibMode"] = np.int32(0)
        fs.attrs["MassCalibration p1"] = np.float64(812.2415)
        fs.attrs["MassCalibration p2"] = np.float64(222.0153)
        fs.attrs["SampleInterval"] = np.float64(sample_interval)
        fs.attrs["Single Ion Signal"] = np.float64(1.0)
        mass_axis = np.linspace(0.0, 10.0, nsamples, dtype=np.float32)
        fs.create_dataset("MassAxis", data=mass_axis)
        fs.create_dataset("SumSpectrum", data=np.ones(nsamples, dtype=np.float64))
        if not with_peak_data:
            vlen = h5py.vlen_dtype(np.uint16)
            el = fs.create_dataset("EventList", shape=(nwrites, nsegs, nx), dtype=vlen)
            rng = np.random.default_rng(42)
            for w in range(nwrites):
                for s in range(nsegs):
                    for x in range(nx):
                        el[w, s, x] = rng.integers(0, nsamples, 3, dtype=np.uint16)
        pd_group = f.create_group("PeakData")
        pd_group.create_dataset("PeakTable", data=peaks)
        if with_peak_data:
            rng = np.random.default_rng(42)
            data = rng.random((nwrites, nsegs, nx, npeaks)).astype(np.float32)
            pd_group.create_dataset("PeakData", data=data)
        f.create_group("FIBImages")
        fibparams = f.create_group("FIBParams")
        fibparams.attrs["FibHardware"] = b"TestFIB"
        fibparams.attrs["Voltage"] = np.float64(30000.0)
        fibparams.attrs["Current"] = np.float64(0.0)
        fibparams.attrs["ViewField"] = np.float64(0.01)


# ---------------------------------------------------------------------------
# Tests: _mark_event_list_ragged
# ---------------------------------------------------------------------------


class TestMarkEventListRagged:
    def test_sets_ragged_and_returns_signal(self):
        from rsciio.tofwerk._api import _mark_event_list_ragged

        class _Stub:
            pass

        sig = _Stub()
        result = _mark_event_list_ragged(sig)
        assert result is sig
        assert sig.ragged is True


# ---------------------------------------------------------------------------
# Tests: mz_range parameter
# ---------------------------------------------------------------------------


class TestMzRange:
    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="mz_range"):
            file_reader(OPENED_FILE, signal="peak_data", mz_range=(5.0, 2.0))

    def test_empty_range_warns_and_no_signal(self, caplog):
        with caplog.at_level(logging.WARNING, logger="rsciio.tofwerk._api"):
            result = file_reader(
                OPENED_FILE, signal="peak_data", mz_range=(100.0, 200.0)
            )
        assert len(result) == 0
        assert any("mz_range" in r.message for r in caplog.records)

    def test_filters_peaks_opened_eager(self):
        result = file_reader(OPENED_FILE, signal="peak_data", mz_range=(1.0, 5.0))
        data = result[0]["data"]
        assert data.shape == (NWRITES, NSEGS, NX, 5)
        np.testing.assert_allclose(
            result[0]["axes"][3]["axis"], [1.0, 2.0, 3.0, 4.0, 5.0]
        )

    def test_filters_peaks_opened_lazy(self):
        import dask.array as da

        result = file_reader(
            OPENED_FILE, signal="peak_data", mz_range=(1.0, 5.0), lazy=True
        )
        data = result[0]["data"]
        assert isinstance(data, da.Array)
        assert data.shape == (NWRITES, NSEGS, NX, 5)

    def test_filters_peaks_raw(self):
        result = file_reader(RAW_FILE, signal="peak_data", mz_range=(1.0, 5.0))
        data = result[0]["data"]
        assert data.shape == (NWRITES, NSEGS, NX, 5)
        np.testing.assert_allclose(
            result[0]["axes"][3]["axis"], [1.0, 2.0, 3.0, 4.0, 5.0]
        )

    def test_unsorted_opened_selects_correct_peaks(self, tmp_path):
        """mz_range on unsorted pre-processed file: result in ascending mass order."""
        p = tmp_path / "unsorted_mz_opened.h5"
        _make_unsorted_peaks_tofdaq(p, with_peak_data=True)
        result = file_reader(p, signal="peak_data", mz_range=(1.0, 1.5))
        assert result[0]["data"].shape[-1] == 1
        np.testing.assert_allclose(result[0]["axes"][3]["axis"], [1.0])

    def test_unsorted_raw_selects_correct_peaks(self, tmp_path):
        """mz_range on unsorted raw file: reconstructs only selected peaks."""
        p = tmp_path / "unsorted_mz_raw.h5"
        _make_unsorted_peaks_tofdaq(p, with_peak_data=False)
        result = file_reader(p, signal="peak_data", mz_range=(1.0, 1.5))
        assert result[0]["data"].shape[-1] == 1
        np.testing.assert_allclose(result[0]["axes"][3]["axis"], [1.0])


# ---------------------------------------------------------------------------
# Tests: depth_range parameter
# ---------------------------------------------------------------------------


class TestDepthRange:
    def test_invalid_stop_raises(self):
        with pytest.raises(ValueError, match="depth_range"):
            file_reader(OPENED_FILE, depth_range=(0, NWRITES + 1))

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError, match="depth_range"):
            file_reader(OPENED_FILE, depth_range=(3, 2))

    def test_opened_eager(self):
        result = file_reader(OPENED_FILE, signal="peak_data", depth_range=(1, 3))
        data = result[0]["data"]
        assert data.shape == (2, NSEGS, NX, NPEAKS)
        assert result[0]["axes"][0]["size"] == 2

    def test_opened_lazy(self):
        import dask.array as da

        result = file_reader(
            OPENED_FILE, signal="peak_data", depth_range=(0, 2), lazy=True
        )
        assert isinstance(result[0]["data"], da.Array)
        assert result[0]["data"].shape == (2, NSEGS, NX, NPEAKS)

    def test_raw_peak_data(self):
        # Also exercises _compute_peak_data_from_file with depth_start/depth_stop (line 897)
        result = file_reader(RAW_FILE, signal="peak_data", depth_range=(1, 3))
        assert result[0]["data"].shape == (2, NSEGS, NX, NPEAKS)

    def test_event_list_eager(self):
        result = file_reader(RAW_FILE, signal="event_list", depth_range=(1, 3))
        assert result[0]["data"].shape == (2, NSEGS, NX)
        assert result[0]["axes"][0]["size"] == 2

    def test_event_list_lazy(self):
        import dask.array as da

        result = file_reader(
            RAW_FILE, signal="event_list", depth_range=(1, 3), lazy=True
        )
        assert isinstance(result[0]["data"], da.Array)
        assert result[0]["data"].shape == (2, NSEGS, NX)

    def test_fib_images(self):
        result = file_reader(OPENED_FILE, signal="fib_images", depth_range=(0, 2))
        assert result[0]["data"].shape == (2, FIB_IMG_SIZE, FIB_IMG_SIZE)
        assert result[0]["axes"][0]["size"] == 2


# ---------------------------------------------------------------------------
# Tests: dtype parameter
# ---------------------------------------------------------------------------


class TestDtype:
    def test_opened_eager(self):
        result = file_reader(OPENED_FILE, signal="peak_data", dtype=np.float16)
        assert result[0]["data"].dtype == np.float16

    def test_opened_lazy(self):
        result = file_reader(
            OPENED_FILE, signal="peak_data", dtype=np.float16, lazy=True
        )
        assert result[0]["data"].dtype == np.float16

    def test_raw(self):
        result = file_reader(RAW_FILE, signal="peak_data", dtype=np.float16)
        assert result[0]["data"].dtype == np.float16

    def test_string_dtype_accepted(self):
        result = file_reader(OPENED_FILE, signal="peak_data", dtype="float16")
        assert result[0]["data"].dtype == np.float16


# ---------------------------------------------------------------------------
# Tests: unsorted peak table (exercises batch-sort and lazy-sort paths)
# ---------------------------------------------------------------------------


class TestUnsortedPeaks:
    """
    Peaks stored in non-ascending mass order exercise the batch-sort and lazy-sort
    code paths (lines 1309, 1330–1337, 1373–1397).
    """

    def test_opened_eager_output_ascending(self, tmp_path):
        """Unsorted pre-processed file: reader still returns masses in ascending order."""
        p = tmp_path / "unsorted_opened.h5"
        _make_unsorted_peaks_tofdaq(p, with_peak_data=True)
        result = file_reader(p, signal="peak_data")
        ax = result[0]["axes"][3]
        np.testing.assert_allclose(ax["axis"], [1.0, 2.0])

    def test_opened_lazy_output_ascending(self, tmp_path):
        """Unsorted pre-processed file lazy: masses in ascending order after compute."""
        import dask.array as da

        p = tmp_path / "unsorted_opened_lazy.h5"
        _make_unsorted_peaks_tofdaq(p, with_peak_data=True)
        result = file_reader(p, signal="peak_data", lazy=True)
        assert isinstance(result[0]["data"], da.Array)
        np.testing.assert_allclose(result[0]["axes"][3]["axis"], [1.0, 2.0])

    def test_raw_output_ascending(self, tmp_path):
        """Unsorted raw file: reconstructed peak_data has masses in ascending order."""
        p = tmp_path / "unsorted_raw.h5"
        _make_unsorted_peaks_tofdaq(p, with_peak_data=False)
        result = file_reader(p, signal="peak_data")
        ax = result[0]["axes"][3]
        np.testing.assert_allclose(ax["axis"], [1.0, 2.0])

    def test_raw_with_dtype(self, tmp_path):
        """dtype cast applies after sort on raw unsorted file."""
        p = tmp_path / "unsorted_raw_dtype.h5"
        _make_unsorted_peaks_tofdaq(p, with_peak_data=False)
        result = file_reader(p, signal="peak_data", dtype=np.float16)
        assert result[0]["data"].dtype == np.float16


# ---------------------------------------------------------------------------
# Tests: tqdm progress bar branches
# ---------------------------------------------------------------------------


def _make_mock_tqdm():
    """Return (mock_tqdm_module, mock_pbar) with tqdm.auto.tqdm pre-wired."""
    from unittest.mock import MagicMock

    mock_pbar = MagicMock()
    mock_tqdm_auto = MagicMock()
    mock_tqdm_auto.tqdm = MagicMock(return_value=mock_pbar)
    return mock_tqdm_auto, mock_pbar


class TestTqdmPbar:
    """
    Verify that tqdm progress bars are created and updated when tqdm is available.

    The functions under test import tqdm lazily (``from tqdm.auto import tqdm``),
    so patching ``sys.modules["tqdm.auto"]`` before calling them is sufficient.
    """

    def test_pbar_numba_path(self, tmp_path):
        """tqdm pbar is created and closed in the numba reconstruction path."""
        import sys
        from unittest.mock import patch

        from rsciio.tofwerk._api import _compute_peak_data_from_file

        p = tmp_path / "tqdm_numba.h5"
        _make_minimal_tofdaq(p)
        mock_tqdm_auto, mock_pbar = _make_mock_tqdm()
        with h5py.File(p, "r") as f:
            with patch.dict(
                sys.modules, {"tqdm": mock_tqdm_auto, "tqdm.auto": mock_tqdm_auto}
            ):
                _compute_peak_data_from_file(f)
        # _make_pbar was called at least once (reconstruction loop)
        assert mock_tqdm_auto.tqdm.call_count >= 1
        assert mock_pbar.update.called
        assert mock_pbar.close.called

    def test_pbar_numpy_path(self, tmp_path):
        """tqdm pbar is created and closed in the NumPy fallback path."""
        import sys
        from unittest.mock import patch

        from rsciio.tofwerk._api import _compute_peak_data_from_file

        p = tmp_path / "tqdm_numpy.h5"
        _make_minimal_tofdaq(p)
        mock_tqdm_auto, mock_pbar = _make_mock_tqdm()
        with h5py.File(p, "r") as f:
            with patch.dict(
                sys.modules,
                {
                    "numba": None,
                    "tqdm": mock_tqdm_auto,
                    "tqdm.auto": mock_tqdm_auto,
                },
            ):
                _compute_peak_data_from_file(f)
        assert mock_tqdm_auto.tqdm.call_count >= 1
        assert mock_pbar.update.called
        assert mock_pbar.close.called

    def test_pbar_unsorted_raw_sort(self, tmp_path):
        """tqdm pbar is created during the post-reconstruction sort on raw unsorted files."""
        import sys
        from unittest.mock import patch

        p = tmp_path / "tqdm_sort.h5"
        _make_unsorted_peaks_tofdaq(p, with_peak_data=False)
        mock_tqdm_auto, mock_pbar = _make_mock_tqdm()
        with patch.dict(
            sys.modules, {"tqdm": mock_tqdm_auto, "tqdm.auto": mock_tqdm_auto}
        ):
            file_reader(p, signal="peak_data")
        # The sort pbar is a separate call; at least one tqdm call must have occurred
        assert mock_tqdm_auto.tqdm.called
        assert mock_pbar.update.called
        assert mock_pbar.close.called

    def test_pbar_event_list_eager(self, tmp_path):
        """tqdm pbar is created and closed in the EventList eager-load path."""
        import sys
        from unittest.mock import patch

        p = tmp_path / "tqdm_el.h5"
        _make_minimal_tofdaq(p)
        mock_tqdm_auto, mock_pbar = _make_mock_tqdm()
        with patch.dict(
            sys.modules, {"tqdm": mock_tqdm_auto, "tqdm.auto": mock_tqdm_auto}
        ):
            file_reader(p, signal="event_list", lazy=False)
        assert mock_tqdm_auto.tqdm.called
        assert mock_pbar.update.called
        assert mock_pbar.close.called

    def test_show_progressbar_false_suppresses_pbar(self, tmp_path):
        """show_progressbar=False prevents tqdm from being called."""
        import sys
        from unittest.mock import patch

        from rsciio.tofwerk._api import _compute_peak_data_from_file

        p = tmp_path / "tqdm_disabled.h5"
        _make_minimal_tofdaq(p)
        mock_tqdm_auto, mock_pbar = _make_mock_tqdm()
        with h5py.File(p, "r") as f:
            with patch.dict(
                sys.modules, {"tqdm": mock_tqdm_auto, "tqdm.auto": mock_tqdm_auto}
            ):
                _compute_peak_data_from_file(f, show_progressbar=False)
        assert mock_tqdm_auto.tqdm.call_count == 0

    def test_tqdm_absent_event_list_eager(self, tmp_path):
        """EventList eager load works when tqdm is not installed (pbar=None path)."""
        import sys
        from unittest.mock import patch

        p = tmp_path / "tqdm_absent_el.h5"
        _make_minimal_tofdaq(p)
        with patch.dict(sys.modules, {"tqdm": None, "tqdm.auto": None}):
            result = file_reader(p, signal="event_list", lazy=False)
        assert len(result) == 1

    def test_tqdm_absent_numpy_path(self, tmp_path):
        """EventList reconstruction works when tqdm is not installed (pbar=None path)."""
        import sys
        from unittest.mock import patch

        p = tmp_path / "tqdm_absent_numpy.h5"
        _make_minimal_tofdaq(p)
        with patch.dict(sys.modules, {"numba": None, "tqdm": None, "tqdm.auto": None}):
            result = file_reader(p, signal="peak_data")
        assert len(result) == 1
