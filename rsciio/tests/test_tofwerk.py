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

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py not installed")

from rsciio.tofwerk import file_reader  # noqa: E402

TEST_DATA = Path(__file__).parent / "data" / "tofwerk"
OPENED_FILE = TEST_DATA / "fib_sims_opened.h5"
RAW_FILE = TEST_DATA / "fib_sims_raw.h5"

# Fixture parameters (must match generate_fixtures.py)
NWRITES = 5
NSEGS = 16
NX = 16
NPEAKS = 10
NSAMPLES = 512
# FIBParams.ViewField = 1e-5 m = 10 µm; pixel_size = 10 µm / 16 = 0.625 µm
PIXEL_SIZE_UM = 10.0 / NX


class TestOpenedFile:
    """
    Opened file returns 3 signals:
      [0] sum spectrum  (1D, from FullSpectra/SumSpectrum)
      [1] TIC map       (2D, summed from PeakData over depth and mass)
      [2] peak data     (4D, depth × y × x × m/z)
    """

    def test_returns_three_signals(self):
        assert len(file_reader(OPENED_FILE)) == 3

    # ── Signal [0]: sum spectrum ──────────────────────────────────────────

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

    # ── Signal [1]: TIC map ───────────────────────────────────────────────

    def test_tic_map_shape(self):
        assert file_reader(OPENED_FILE)[1]["data"].shape == (NSEGS, NX)

    def test_tic_map_title(self):
        meta = file_reader(OPENED_FILE)[1]["metadata"]
        assert "TIC" in meta["General"]["title"]

    def test_tic_map_axes(self):
        axes = file_reader(OPENED_FILE)[1]["axes"]
        assert len(axes) == 2
        assert axes[0]["name"] == "y"
        assert axes[1]["name"] == "x"

    def test_tic_map_pixel_size(self):
        axes = file_reader(OPENED_FILE)[1]["axes"]
        np.testing.assert_allclose(axes[0]["scale"], PIXEL_SIZE_UM)
        np.testing.assert_allclose(axes[1]["scale"], PIXEL_SIZE_UM)

    # ── Signal [2]: 4D peak data ──────────────────────────────────────────

    def test_peak_data_shape(self):
        assert file_reader(OPENED_FILE)[2]["data"].shape == (NWRITES, NSEGS, NX, NPEAKS)

    def test_peak_data_dtype(self):
        assert file_reader(OPENED_FILE)[2]["data"].dtype == np.float32

    def test_peak_data_title(self):
        meta = file_reader(OPENED_FILE)[2]["metadata"]
        assert "peak data" in meta["General"]["title"]

    def test_four_axes(self):
        assert len(file_reader(OPENED_FILE)[2]["axes"]) == 4

    def test_axis_names(self):
        names = [ax["name"] for ax in file_reader(OPENED_FILE)[2]["axes"]]
        assert names == ["depth", "y", "x", "m/z"]

    def test_depth_axis(self):
        ax = file_reader(OPENED_FILE)[2]["axes"][0]
        assert ax["units"] == "slice"
        assert ax["navigate"] is True
        assert ax["size"] == NWRITES
        assert ax["scale"] == 1
        assert ax["offset"] == 0

    def test_spatial_axes_units(self):
        axes = file_reader(OPENED_FILE)[2]["axes"]
        assert axes[1]["units"] == "µm"
        assert axes[2]["units"] == "µm"

    def test_spatial_axes_navigate(self):
        axes = file_reader(OPENED_FILE)[2]["axes"]
        assert axes[1]["navigate"] is True
        assert axes[2]["navigate"] is True

    def test_pixel_size_from_viewfield(self):
        # FIBParams.ViewField = 1e-5 m = 10 µm; 10 / 16 = 0.625 µm/pixel
        axes = file_reader(OPENED_FILE)[2]["axes"]
        np.testing.assert_allclose(axes[1]["scale"], PIXEL_SIZE_UM)
        np.testing.assert_allclose(axes[2]["scale"], PIXEL_SIZE_UM)

    def test_mass_axis_units(self):
        ax = file_reader(OPENED_FILE)[2]["axes"][3]
        assert ax["units"] == "Da"

    def test_mass_axis_navigate(self):
        ax = file_reader(OPENED_FILE)[2]["axes"][3]
        assert ax["navigate"] is False

    def test_mass_axis_values(self):
        ax = file_reader(OPENED_FILE)[2]["axes"][3]
        assert "axis" in ax
        np.testing.assert_allclose(ax["axis"], np.arange(1, NPEAKS + 1, dtype=float))

    # ── Metadata (shared across all signals) ─────────────────────────────

    def test_signal_type(self):
        meta = file_reader(OPENED_FILE)[2]["metadata"]
        assert meta["Signal"]["signal_type"] == "FIB-SIMS"

    def test_binned_flag(self):
        meta = file_reader(OPENED_FILE)[2]["metadata"]
        assert meta["Signal"]["binned"] is True

    def test_file_type_flag(self):
        meta = file_reader(OPENED_FILE)[2]["metadata"]
        assert meta["Acquisition_instrument"]["FIB_SIMS"]["file_type"] == "opened"

    def test_voltage_kv(self):
        fib = file_reader(OPENED_FILE)[2]["metadata"]["Acquisition_instrument"][
            "FIB_SIMS"
        ]["FIB"]
        np.testing.assert_allclose(fib["voltage_kV"], 30.0)

    def test_fib_hardware(self):
        fib = file_reader(OPENED_FILE)[2]["metadata"]["Acquisition_instrument"][
            "FIB_SIMS"
        ]["FIB"]
        assert fib["hardware"] == "Tescan"

    def test_ion_mode(self):
        tof = file_reader(OPENED_FILE)[2]["metadata"]["Acquisition_instrument"][
            "FIB_SIMS"
        ]["ToF"]
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

        result = file_reader(OPENED_FILE, lazy=lazy)
        # Sum spectrum: lazy when requested
        sum_data = result[0]["data"]
        if lazy:
            assert isinstance(sum_data, da.Array)
        else:
            assert isinstance(sum_data, np.ndarray)
        # Peak data: lazy when requested
        peak_data = result[2]["data"]
        if lazy:
            assert isinstance(peak_data, da.Array)
            np.testing.assert_array_equal(peak_data.shape, (NWRITES, NSEGS, NX, NPEAKS))
        else:
            assert isinstance(peak_data, np.ndarray)
            assert peak_data.shape == (NWRITES, NSEGS, NX, NPEAKS)


class TestRawFile:
    """
    Raw file returns 2 signals:
      [0] sum spectrum  (1D)
      [1] TIC map       (2D, computed from EventList)
    """

    def test_returns_two_signals(self):
        assert len(file_reader(RAW_FILE)) == 2

    def test_sum_spectrum_shape(self):
        assert file_reader(RAW_FILE)[0]["data"].shape == (NSAMPLES,)

    def test_tic_map_shape(self):
        assert file_reader(RAW_FILE)[1]["data"].shape == (NSEGS, NX)

    def test_tic_map_dtype(self):
        assert file_reader(RAW_FILE)[1]["data"].dtype == np.int32

    def test_sum_spectrum_mass_axis(self):
        ax = file_reader(RAW_FILE)[0]["axes"][0]
        assert ax["name"] == "m/z"
        assert ax["units"] == "Da"
        assert "axis" in ax
        assert len(ax["axis"]) == NSAMPLES

    def test_tic_map_axes(self):
        axes = file_reader(RAW_FILE)[1]["axes"]
        assert len(axes) == 2
        assert axes[0]["name"] == "y"
        assert axes[1]["name"] == "x"

    def test_file_type_flag(self):
        meta = file_reader(RAW_FILE)[0]["metadata"]
        assert meta["Acquisition_instrument"]["FIB_SIMS"]["file_type"] == "raw"

    def test_sum_spectrum_title(self):
        assert (
            "sum spectrum" in file_reader(RAW_FILE)[0]["metadata"]["General"]["title"]
        )

    def test_tic_map_title(self):
        assert "TIC" in file_reader(RAW_FILE)[1]["metadata"]["General"]["title"]

    @pytest.mark.parametrize("lazy", [True, False])
    def test_sum_spectrum_lazy(self, lazy):
        import dask.array as da

        data = file_reader(RAW_FILE, lazy=lazy)[0]["data"]
        if lazy:
            assert isinstance(data, da.Array)
        else:
            assert isinstance(data, np.ndarray)


class TestComputePeakData:
    """
    Tests for the ``compute_peak_data=True`` option on raw files.

    With ClockPeriod = SampleInterval in the fixture, clock_ratio = 1 and
    NbrWaveforms = 1, NActiveChannels = 1 → normalization = 1.  EventList
    values are direct ADC sample indices in [0, NSAMPLES).
    """

    def test_raw_default_returns_two_signals(self):
        assert len(file_reader(RAW_FILE)) == 2

    def test_raw_compute_returns_three_signals(self):
        assert len(file_reader(RAW_FILE, compute_peak_data=True)) == 3

    def test_peak_data_shape(self):
        sig = file_reader(RAW_FILE, compute_peak_data=True)[2]
        assert sig["data"].shape == (NWRITES, NSEGS, NX, NPEAKS)

    def test_peak_data_dtype(self):
        sig = file_reader(RAW_FILE, compute_peak_data=True)[2]
        assert sig["data"].dtype == np.float32

    def test_peak_data_title(self):
        sig = file_reader(RAW_FILE, compute_peak_data=True)[2]
        assert "peak data" in sig["metadata"]["General"]["title"]

    def test_peak_data_axes(self):
        axes = file_reader(RAW_FILE, compute_peak_data=True)[2]["axes"]
        assert len(axes) == 4
        assert [ax["name"] for ax in axes] == ["depth", "y", "x", "m/z"]

    def test_peak_data_mass_axis_values(self):
        ax = file_reader(RAW_FILE, compute_peak_data=True)[2]["axes"][3]
        np.testing.assert_allclose(ax["axis"], np.arange(1, NPEAKS + 1, dtype=float))

    def test_peak_data_non_negative(self):
        data = file_reader(RAW_FILE, compute_peak_data=True)[2]["data"]
        assert (data >= 0).all()

    def test_peak_data_values_match_algorithm(self):
        """Verify plumbing: reader output matches _compute_peak_data_from_eventlist."""
        import h5py

        from rsciio.tofwerk._api import _compute_peak_data_from_eventlist

        with h5py.File(RAW_FILE, "r") as f:
            expected = _compute_peak_data_from_eventlist(f)
        result = file_reader(RAW_FILE, compute_peak_data=True)[2]["data"]
        np.testing.assert_array_equal(result, expected)

    def test_compute_peak_data_no_effect_on_opened_file(self):
        """compute_peak_data=True is a no-op on already-opened files."""
        sigs_default = file_reader(OPENED_FILE)
        sigs_flag = file_reader(OPENED_FILE, compute_peak_data=True)
        assert len(sigs_default) == len(sigs_flag) == 3
        np.testing.assert_array_equal(sigs_default[2]["data"], sigs_flag[2]["data"])


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
        meta = file_reader(OPENED_FILE)[2]["metadata"]
        assert meta["Acquisition_instrument"]["FIB_SIMS"]["file_type"] == "opened"

    def test_raw_file_detected(self):
        meta = file_reader(RAW_FILE)[0]["metadata"]
        assert meta["Acquisition_instrument"]["FIB_SIMS"]["file_type"] == "raw"

    def test_plugin_discovered(self):
        from rsciio import IO_PLUGINS

        assert "Tofwerk" in [p["name"] for p in IO_PLUGINS]
