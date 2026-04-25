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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

import struct
from pathlib import Path

import numpy as np
import pytest

from rsciio.quadstar._api import (
    _build_datetime,
    _decode_bytes,
    _find_first_data_position,
    _read_general_header,
    _read_trace_headers,
    file_reader,
)

TEST_DATA_PATH = Path(__file__).parent / "data" / "quadstar"
SBC_FILES = sorted(TEST_DATA_PATH.glob("*.sbc"))
FIRST_SBC_FILE = SBC_FILES[0]
SAC_FILE = TEST_DATA_PATH / "airdemo.sac"
EXPECTED_NUM_MASSES = {
    "1-200amu-2cycles.sbc": 200,
    "1-200amu-3cycles.sbc": 200,
    "1-400amu-2cycles.sbc": 400,
    "1-500amu-2cycles.sbc": 500,
    "1-50amu-2cycles.sbc": 50,
    "10-50amu-2cycles.sbc": 41,
}


class TestHelpers:
    def test_decode_bytes_normal(self):
        assert _decode_bytes(b"hello\x00\x00") == "hello"

    def test_decode_bytes_empty(self):
        assert _decode_bytes(b"\x00\x00") == ""

    def test_decode_bytes_non_bytes(self):
        assert _decode_bytes(42) == "42"

    def test_build_datetime_valid(self):
        header = {
            "year": 124,  # 2024
            "month": 3,
            "day": 15,
            "hour": 10,
            "minute": 30,
            "second": 45,
        }
        dt = _build_datetime(header)
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15
        assert dt.hour == 10
        assert dt.minute == 30
        assert dt.second == 45

    def test_build_datetime_invalid(self):
        header = {
            "year": 0,
            "month": 0,
            "day": 0,
            "hour": 0,
            "minute": 0,
            "second": 0,
        }
        dt = _build_datetime(header)
        assert dt is None

    def test_find_first_data_position(self):
        headers = [
            {"type": 0x00, "data_position": 100},
            {"type": 0x11, "data_position": 500},
            {"type": 0x11, "data_position": 900},
        ]
        assert _find_first_data_position(headers) == 500

    def test_find_first_data_position_none(self):
        headers = [
            {"type": 0x00, "data_position": 100},
        ]
        assert _find_first_data_position(headers) is None


def _read_sbc_header_values(path):
    buf = path.read_bytes()
    num_cycles = struct.unpack_from("<I", buf, 0x64)[0]
    width = struct.unpack_from("<H", buf, 0xCB)[0]
    num_masses = width + 1
    return len(buf), num_cycles, num_masses


class TestReadSbcFiles:
    @pytest.mark.parametrize("filename", SBC_FILES, ids=lambda p: p.name)
    def test_returns_single_signal_dict(self, filename):
        signals = file_reader(filename)
        assert isinstance(signals, list)
        assert len(signals) == 1
        sig = signals[0]
        assert "data" in sig
        assert "axes" in sig
        assert "metadata" in sig
        assert "original_metadata" in sig

    @pytest.mark.parametrize("filename", SBC_FILES, ids=lambda p: p.name)
    def test_shape_and_dtype_match_header(self, filename):
        file_size, num_cycles, num_masses = _read_sbc_header_values(filename)
        sig = file_reader(filename)[0]
        data = sig["data"]
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float32
        if num_cycles > 1:
            assert data.shape == (num_cycles, num_masses)
        else:
            assert data.shape == (num_masses,)

        sbc = sig["original_metadata"]["sbc_parameters"]
        assert sbc["num_masses"] == num_masses
        assert sbc["prefix_size"] == 13
        assert sbc["stride"] == num_masses * 8 + 13
        assert sbc["data_start"] == file_size - num_cycles * sbc["stride"]

    @pytest.mark.parametrize("name, expected", sorted(EXPECTED_NUM_MASSES.items()))
    def test_expected_num_masses_by_file(self, name, expected):
        sig = file_reader(TEST_DATA_PATH / name)[0]
        assert sig["original_metadata"]["sbc_parameters"]["num_masses"] == expected

    @pytest.mark.parametrize("filename", SBC_FILES, ids=lambda p: p.name)
    def test_mass_axis_and_masses_metadata(self, filename):
        _, num_cycles, num_masses = _read_sbc_header_values(filename)
        sig = file_reader(filename)[0]

        signal_axis = [a for a in sig["axes"] if not a["navigate"]]
        assert len(signal_axis) == 1
        ax = signal_axis[0]
        assert ax["name"] == "Mass-to-charge ratio"
        assert ax["units"] == "m/z"
        assert ax["size"] == num_masses
        assert ax["scale"] > 0

        masses = np.asarray(sig["original_metadata"]["masses"])
        assert masses.shape[0] == num_masses
        np.testing.assert_allclose(np.diff(masses), 1.0)

        if num_cycles > 1:
            assert ax["index_in_array"] == 1
        else:
            assert ax["index_in_array"] == 0

    @pytest.mark.parametrize("filename", SBC_FILES, ids=lambda p: p.name)
    def test_time_axis_and_timestamps(self, filename):
        _, num_cycles, _ = _read_sbc_header_values(filename)
        sig = file_reader(filename)[0]
        ts = sig["original_metadata"]["timestamps"]
        assert isinstance(ts, list)
        assert len(ts) == num_cycles

        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1]

        if num_cycles > 1:
            nav_axis = [a for a in sig["axes"] if a["navigate"]]
            assert len(nav_axis) == 1
            t = nav_axis[0]
            assert t["name"] == "Time"
            assert t["units"] == "s"
            assert t["size"] == num_cycles
            assert t["index_in_array"] == 0

    @pytest.mark.parametrize("filename", SBC_FILES, ids=lambda p: p.name)
    def test_time_axis_non_uniform_when_requested(self, filename):
        _, num_cycles, _ = _read_sbc_header_values(filename)
        if num_cycles <= 1:
            return

        sig = file_reader(filename, use_uniform_signal_axis=False)[0]
        ts = np.asarray(sig["original_metadata"]["timestamps"], dtype=np.float64)

        nav_axis = [a for a in sig["axes"] if a["navigate"]]
        assert len(nav_axis) == 1
        t = nav_axis[0]
        assert t["name"] == "Time"
        assert t["units"] == "s"
        assert t["size"] == num_cycles
        assert "axis" in t
        assert "offset" not in t
        assert "scale" not in t
        np.testing.assert_allclose(np.asarray(t["axis"], dtype=np.float64), ts - ts[0])


class TestReadSacFiles:
    def test_sac_returns_single_signal(self):
        signals = file_reader(SAC_FILE)
        assert isinstance(signals, list)
        assert len(signals) == 1
        sig = signals[0]
        assert "data" in sig
        assert "axes" in sig
        assert "metadata" in sig
        assert "original_metadata" in sig

    def test_sac_data_shape_and_dtype(self):
        sig = file_reader(SAC_FILE)[0]
        data = sig["data"]
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.float32
        # SAC should have at least 1D shape
        assert data.ndim >= 1

    def test_sac_has_mass_axis(self):
        sig = file_reader(SAC_FILE)[0]
        signal_axes = [a for a in sig["axes"] if not a["navigate"]]
        assert len(signal_axes) > 0
        # Check for mass-to-charge ratio axis
        mass_axes = [
            a
            for a in signal_axes
            if "mass" in a["name"].lower() or "m/z" in a["units"].lower()
        ]
        assert len(mass_axes) > 0
        ax = mass_axes[0]
        assert ax["size"] > 0

    def test_sac_metadata_structure(self):
        sig = file_reader(SAC_FILE)[0]
        original = sig["original_metadata"]

        assert "general_header" in original
        assert "trace_info" in original
        assert "timestamps" in original

        general = original["general_header"]
        trace_info = original["trace_info"]
        timestamps = original["timestamps"]
        np.testing.assert_allclose(
            timestamps,
            [
                7.63685197e08,
                7.63685258e08,
                7.63685320e08,
                7.63685381e08,
                7.63685442e08,
                7.63685504e08,
                7.63685565e08,
                7.63685627e08,
                7.63685688e08,
                7.63685749e08,
                7.63685810e08,
                7.63685872e08,
                7.63685933e08,
                7.63685994e08,
                7.63686056e08,
                7.63686117e08,
                7.63686178e08,
                7.63686239e08,
                7.63686301e08,
                7.63686362e08,
                7.63686423e08,
            ],
        )

        assert general["n_timesteps"] == 21
        assert general["n_traces"] == 3
        assert trace_info["scan_width"] == 60
        assert trace_info["values_per_mass"] == 64
        assert len(timestamps) == general["n_timesteps"]

    def test_sac_time_axis_non_uniform_when_requested(self):
        sig = file_reader(SAC_FILE, use_uniform_signal_axis=False)[0]
        timestamps = np.asarray(
            sig["original_metadata"]["timestamps"], dtype=np.float64
        )

        nav_axis = [a for a in sig["axes"] if a["navigate"]]
        assert len(nav_axis) == 1
        t = nav_axis[0]
        assert t["name"] == "Time"
        assert t["units"] == "s"
        assert "axis" in t
        assert "offset" not in t
        assert "scale" not in t
        np.testing.assert_allclose(
            np.asarray(t["axis"], dtype=np.float64), timestamps - timestamps[0]
        )


class TestReadGeneralHeader:
    def test_general_header(self):
        with open(FIRST_SBC_FILE, "rb") as f:
            buf = f.read()
        gen = _read_general_header(buf)
        assert isinstance(gen, dict)
        assert gen["n_timesteps"] > 0
        assert gen["n_traces"] >= 0
        assert gen["timestep_length"] >= 0

    def test_trace_headers(self):
        with open(FIRST_SBC_FILE, "rb") as f:
            buf = f.read()
        gen = _read_general_header(buf)
        headers = _read_trace_headers(buf, gen["n_traces"])
        assert len(headers) == gen["n_traces"]
        # SBC fixtures have no trace blocks.
        assert gen["n_traces"] == 0

    def test_sac_general_header(self):
        with open(SAC_FILE, "rb") as f:
            buf = f.read()
        gen = _read_general_header(buf)
        assert isinstance(gen, dict)
        assert gen["n_timesteps"] > 0
        assert gen["n_traces"] >= 0

    def test_sac_trace_headers(self):
        with open(SAC_FILE, "rb") as f:
            buf = f.read()
        gen = _read_general_header(buf)
        assert gen["n_traces"] > 0
        # SAC files have trace blocks, unlike SBC
        headers = _read_trace_headers(buf, gen["n_traces"])
        assert len(headers) == gen["n_traces"]
        # Verify trace headers have expected structure
        for header in headers:
            assert isinstance(header, dict)
            assert "type" in header


class TestLazyNotSupported:
    def test_lazy_raises_sbc(self):
        with pytest.raises(NotImplementedError, match="Lazy loading"):
            file_reader(FIRST_SBC_FILE, lazy=True)

    def test_lazy_raises_sac(self):
        with pytest.raises(NotImplementedError, match="Lazy loading"):
            file_reader(SAC_FILE, lazy=True)
