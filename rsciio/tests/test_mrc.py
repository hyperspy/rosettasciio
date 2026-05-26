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
from pathlib import Path

import importlib.util
import numpy as np
import pytest

from rsciio.exceptions import VisibleDeprecationWarning
from rsciio.mrc import file_reader
from rsciio.mrc._api import (
    MOVIE_RE,
    VIRTUAL_RE,
    find_related_de_files,
    get_data_type,
)

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
zarr_missing = importlib.util.find_spec("zarr") is None


TEST_DATA_DIR = Path(__file__).parent / "data" / "mrc"


def test_single_image():
    # Acquired from Velox
    s = hs.load(TEST_DATA_DIR / "HAADFscan.mrc")
    assert s.data.shape == (16, 16)
    assert s.axes_manager.signal_shape == (16, 16)
    assert s.axes_manager.navigation_shape == ()

    for axis in s.axes_manager.signal_axes:
        assert axis.scale == 5.679131317138672
        assert axis.offset == 0
        assert axis.units == "nm"


def test_4DSTEM_image():
    # Acquired from Velox
    s = hs.load(TEST_DATA_DIR / "4DSTEMscan.mrc")
    assert s.data.shape == (256, 256, 256)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (256,)


def test_4DSTEM_image_navigation_shape_16_16():
    # Acquired from Velox
    s = hs.load(
        TEST_DATA_DIR / "4DSTEMscan.mrc",
        navigation_shape=(16, 16),
    )
    assert s.data.shape == (16, 16, 256, 256)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (16, 16)


def test_4DSTEM_image_navigation_shape_8_32():
    s = hs.load(
        TEST_DATA_DIR / "4DSTEMscan.mrc",
        navigation_shape=(8, 32),
    )
    assert s.data.shape == (32, 8, 256, 256)
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == (8, 32)


@pytest.mark.parametrize("distributed", [True, False])
def test_distributed_deprecation_warning(distributed):
    with pytest.warns(VisibleDeprecationWarning):
        file_reader(
            str(TEST_DATA_DIR / "4DSTEMscan.mrc"),
            navigation_shape=(8, 32),
            distributed=distributed,
        )


def test_deprecated_mmap_mode():
    with pytest.warns(VisibleDeprecationWarning):
        file_reader(
            str(TEST_DATA_DIR / "4DSTEMscan.mrc"),
            navigation_shape=(8, 32),
            mmap_mode="r",
        )


def test_mrc_chunks_equal():
    s = hs.load(
        TEST_DATA_DIR / "4DSTEMscan.mrc",
        navigation_shape=(8, 32),
        chunks=(16, 4, 256, 256),
        lazy=True,
    )
    assert s.data.chunks == (
        (
            16,
            16,
        ),
        (
            4,
            4,
        ),
        (256,),
        (256,),
    )


@pytest.mark.parametrize("navigation_shape", [None, (256,), (8, 32), (8, 16, 2)])
def test_mrc_metadata(navigation_shape):
    s = hs.load(
        TEST_DATA_DIR / "4DSTEMscan.mrc",
        metadata_file=TEST_DATA_DIR / "info.txt",
        navigation_shape=navigation_shape,
    )
    if navigation_shape is None:
        navigation_shape = (8, 32)
    shape = navigation_shape[::-1] + (256, 256)
    assert s.data.shape == shape
    assert s.axes_manager.signal_shape == (256, 256)
    assert s.axes_manager.navigation_shape == navigation_shape
    assert s.metadata.Acquisition_instrument.TEM.detector == "CeleritasXS"
    assert s.metadata.Acquisition_instrument.TEM.magnification == "1000"
    assert s.metadata.Acquisition_instrument.TEM.frames_per_second == "40000"


def test_mrc_metadata_auto():
    s = hs.load(TEST_DATA_DIR / "20241021_00405_movie.mrc", lazy=True)
    navigation_shape = (8, 4)
    shape = navigation_shape[::-1] + (4, 8)
    assert s.data.shape == shape
    assert s.axes_manager.signal_shape == (8, 4)
    assert s.axes_manager.navigation_shape == navigation_shape
    assert s.metadata.Acquisition_instrument.TEM.detector == "DESim"
    assert s.metadata.Acquisition_instrument.TEM.magnification == "1000"
    assert s.metadata.Acquisition_instrument.TEM.frames_per_second == "700"
    assert len(s.metadata._HyperSpy.navigators.keys()) == 3
    assert isinstance(s.metadata._HyperSpy.navigators["Virt 0"], hs.signals.Signal2D)

    assert s.metadata._HyperSpy.navigator is not None

    shape = (
        s.axes_manager._navigation_shape_in_array
        + s.axes_manager._signal_shape_in_array
    )
    assert s.data.shape == shape


def test_mrc_metadata_auto_custom_shape():
    s = hs.load(
        TEST_DATA_DIR / "20241021_00405_movie.mrc", lazy=True, navigation_shape=(16, 2)
    )
    navigation_shape = (16, 2)
    shape = navigation_shape[::-1] + (4, 8)
    assert s.data.shape == shape
    assert s.axes_manager.signal_shape == (8, 4)
    assert s.axes_manager.navigation_shape == navigation_shape
    assert s.metadata.Acquisition_instrument.TEM.detector == "DESim"
    assert s.metadata.Acquisition_instrument.TEM.magnification == "1000"
    assert s.metadata.Acquisition_instrument.TEM.frames_per_second == "700"
    assert len(s.metadata._HyperSpy.navigators) == 3

    assert s.metadata._HyperSpy.navigator is not None
    assert s.metadata._HyperSpy.navigator.data.shape == navigation_shape[::-1]

    assert isinstance(s.metadata._HyperSpy.navigator, hs.signals.BaseSignal)

    shape = (
        s.axes_manager._navigation_shape_in_array
        + s.axes_manager._signal_shape_in_array
    )
    assert s.data.shape == shape


@pytest.mark.parametrize(
    "metadata_file",
    [
        TEST_DATA_DIR / "3DSTEM_scan_info.txt",
        TEST_DATA_DIR / "3DTEM_scan_info.txt",
        TEST_DATA_DIR / "3DTEMDiffracting_scan_info.txt",
    ],
)
def test_mrc_metadata_modes(metadata_file):
    s = hs.load(TEST_DATA_DIR / "20241021_00405_movie.mrc", metadata_file=metadata_file)
    diffracting = "STEM" in metadata_file.name or "Diffracting" in metadata_file.name
    s.axes_manager.navigation_axes[0].units = "sec"
    if diffracting:
        s.axes_manager.signal_axes[0].units = "nm^-1"
        s.axes_manager.signal_axes[1].units = "nm^-1"
    else:
        s.axes_manager.signal_axes[0].units = "nm"
        s.axes_manager.signal_axes[1].units = "nm"


def test_mrc_random_scan_pattern():
    s = hs.load(
        TEST_DATA_DIR / "ROI_Random_Scan_movie.mrc",
        metadata_file=TEST_DATA_DIR / "ROI_Random_Scan_info.txt",
        scan_file=TEST_DATA_DIR / "ROI_Random_Scan_scan_coordinates.csv",
    )
    assert s.data.shape == (29, 12, 16, 16)
    # check to make sure that the Sum image from DE Server matches the sum.
    sum_nav = hs.load(TEST_DATA_DIR / "ROI_Random_Scan_Sum.mrc")
    np.testing.assert_array_almost_equal(s.sum(axis=(2, 3)).data, sum_nav, decimal=-1)


def test_repeated_mrc_custom():
    s = hs.load(
        TEST_DATA_DIR / "Custom_movie.mrc",
        metadata_file=TEST_DATA_DIR / "Custom_info.txt",
        scan_file=TEST_DATA_DIR / "Custom_scan_coordinates.csv",
    )
    assert s.data.shape == (5, 5, 2, 16, 16)
    # make sure that the first and second dataset aren't equal
    assert not np.array_equal(s.data[:, :, 0], s.data[:, :, 1])
    np.testing.assert_array_equal(s.data[:, 1], 0)  # Skipped rows
    np.testing.assert_array_equal(s.data[:, 3], 0)  # Skipped rows


def test_repeated_mrc_custom_error():
    with pytest.raises(ValueError):
        hs.load(
            TEST_DATA_DIR / "Custom_movie.mrc",
            metadata_file=TEST_DATA_DIR / "Custom_info.txt",
            scan_file=TEST_DATA_DIR / "Custom_scan_coordinates.csv",
            chunks=(5, 5, 2, 2, 2),
        )


def test_repeated_mrc_custom_no_scan_file():
    with pytest.raises(ValueError):
        hs.load(
            TEST_DATA_DIR / "Custom_movie.mrc",
            metadata_file=TEST_DATA_DIR / "Custom_info.txt",
        )


# ---------------------------------------------------------------------------
# Tests for new PR code: regex patterns, find_related_de_files, uint8 mode 0
# ---------------------------------------------------------------------------


class TestMovieRegex:
    """Tests for the MOVIE_RE regex pattern."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            (
                "20241021_00405_movie.mrc",
                {
                    "timestamp": "20241021",
                    "acquisitionNumber": "00405",
                    "optionalSuffix": None,
                    "movieNum": None,
                },
            ),
            (
                "20241021_00405_suffix_movie.mrc",
                {
                    "timestamp": "20241021",
                    "acquisitionNumber": "00405",
                    "optionalSuffix": "suffix",
                    "movieNum": None,
                },
            ),
            (
                "20241021_00405_suffix_extra_movie.mrc",
                {
                    "timestamp": "20241021",
                    "acquisitionNumber": "00405",
                    "optionalSuffix": "suffix_extra",
                    "movieNum": None,
                },
            ),
        ],
    )
    def test_movie_re_matches(self, filename, expected):
        m = MOVIE_RE.fullmatch(filename)
        assert m is not None
        for key, val in expected.items():
            assert m.group(key) == val

    def test_movie_re_no_match(self):
        assert MOVIE_RE.fullmatch("20241021_00405_info.txt") is None
        assert MOVIE_RE.fullmatch("not_a_movie.mrc") is None
        assert MOVIE_RE.fullmatch("20241021_00405_0_Virt 0_sum.mrc") is None


class TestVirtualRegex:
    """Tests for the VIRTUAL_RE regex pattern."""

    @pytest.mark.parametrize(
        "filename, expected_name, expected_calc",
        [
            ("20241021_00405_0_Virt 0_sum.mrc", "Virt 0", "sum"),
            ("20241021_00405_1_Virt 1_sum.mrc", "Virt 1", "sum"),
            (
                "20251103_23389_3_CentroidAmp_centroid_amplitude.mrc",
                "CentroidAmp",
                "centroid_amplitude",
            ),
        ],
    )
    def test_virtual_re_matches(self, filename, expected_name, expected_calc):
        m = VIRTUAL_RE.fullmatch(filename)
        assert m is not None, f"VIRTUAL_RE should match {filename!r}"
        assert m.group("name") == expected_name
        assert m.group("calculationType") == expected_calc

    def test_virtual_re_no_match_movie(self):
        assert VIRTUAL_RE.fullmatch("20241021_00405_movie.mrc") is None

    def test_virtual_re_virt_prefix(self):
        # virtualImageNum with 'virt<digits>' prefix should also match
        m = VIRTUAL_RE.fullmatch("20241021_00405_virt0_BF_sum.mrc")
        assert m is not None
        assert m.group("virtualImageNum") == "virt0"
        assert m.group("name") == "BF"


class TestGetDataType:
    """Tests for the get_data_type function."""

    def test_mode_0_is_uint8(self):
        dtype = get_data_type(np.array([0]))
        assert dtype == np.dtype(np.uint8)

    def test_mode_1_is_int16(self):
        assert get_data_type(np.array([1])) == np.dtype(np.int16)

    def test_mode_2_is_float32(self):
        assert get_data_type(np.array([2])) == np.dtype(np.float32)

    def test_mode_6_is_uint16(self):
        assert get_data_type(np.array([6])) == np.dtype(np.uint16)

    def test_mode_12_is_float16(self):
        assert get_data_type(np.array([12])) == np.dtype(np.float16)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unrecognised mode"):
            get_data_type(np.array([99]))


class TestFindRelatedDeFiles:
    """Tests for find_related_de_files."""

    def test_find_related_de_files_basic(self):
        movie = str(TEST_DATA_DIR / "20241021_00405_movie.mrc")
        result = find_related_de_files(movie)

        assert result["info_file"] is not None
        assert "20241021_00405_info.txt" in result["info_file"]

        assert len(result["virtual_images"]) == 2
        assert len(result["virtual_image_names"]) == 2
        assert "Virt 0" in result["virtual_image_names"]
        assert "Virt 1" in result["virtual_image_names"]

        assert len(result["external_images"]) == 1
        assert len(result["external_image_names"]) == 1
        assert "Ext 1" in result["external_image_names"]

        # scan CSV is not present for this dataset
        assert result["scan_csv"] is None

    def test_find_related_de_files_non_movie_name(self):
        """Non-matching filenames should return empty result."""
        result = find_related_de_files("not_a_real_movie_file.mrc")
        assert result["info_file"] is None
        assert result["virtual_images"] == []
        assert result["virtual_image_names"] == []
        assert result["external_images"] == []
        assert result["scan_csv"] is None

    def test_find_related_de_files_virtual_images_sorted(self):
        """Virtual image list should be sorted."""
        movie = str(TEST_DATA_DIR / "20241021_00405_movie.mrc")
        result = find_related_de_files(movie)
        assert result["virtual_images"] == sorted(result["virtual_images"])

    def test_find_related_de_files_dedupe(self, tmp_path):
        """_dedupe should rename duplicate virtual image names."""
        import shutil

        # Create a minimal movie file and two virtual images with the same name
        src_movie = TEST_DATA_DIR / "20241021_00405_movie.mrc"
        ts = "20260101"
        acq = "99999"
        movie = tmp_path / f"{ts}_{acq}_movie.mrc"
        shutil.copy(src_movie, movie)
        v0 = tmp_path / f"{ts}_{acq}_0_BF_sum.mrc"
        v1 = tmp_path / f"{ts}_{acq}_1_BF_sum.mrc"
        shutil.copy(src_movie, v0)
        shutil.copy(src_movie, v1)
        # Create a dummy info file
        (tmp_path / f"{ts}_{acq}_info.txt").write_text("Camera Model = TestCamera\n")

        result = find_related_de_files(str(movie))
        names = result["virtual_image_names"]
        assert len(names) == 2
        # First occurrence keeps the name, second gets a suffix
        assert names[0] == "BF"
        assert names[1] != "BF"
        assert names[1].startswith("BF")


class TestMetadataVirtualImageNames:
    """Test that virtual and external image names appear correctly in loaded signal metadata."""

    def test_virtual_image_names_in_metadata(self):
        s = hs.load(TEST_DATA_DIR / "20241021_00405_movie.mrc", lazy=True)
        # HyperSpy stores _sig_* keys and exposes them without the prefix as signals
        keys = list(s.metadata._HyperSpy.navigators.as_dictionary().keys())
        assert "_sig_Virt 0" in keys
        assert "_sig_Virt 1" in keys
        # Accessible via HyperSpy attribute (prefix stripped)
        assert "Virt 0" in s.metadata._HyperSpy.navigators

    def test_external_image_names_in_metadata(self):
        s = hs.load(TEST_DATA_DIR / "20241021_00405_movie.mrc", lazy=True)
        nav_keys = list(s.metadata._HyperSpy.navigators.as_dictionary().keys())
        # 2 virtual + 1 external = 3 navigators total
        assert len(nav_keys) == 3
        # External key stored with _sig_ prefix
        assert "_sig_Ext 1" in nav_keys
        # Accessible via HyperSpy attribute (prefix stripped)
        assert "Ext 1" in s.metadata._HyperSpy.navigators

    def test_navigator_is_signal(self):
        s = hs.load(TEST_DATA_DIR / "20241021_00405_movie.mrc", lazy=True)
        assert isinstance(s.metadata._HyperSpy.navigator, hs.signals.BaseSignal)

    def test_virtual_image_is_signal2d(self):
        s = hs.load(TEST_DATA_DIR / "20241021_00405_movie.mrc", lazy=True)
        virt = s.metadata._HyperSpy.navigators["Virt 0"]
        assert isinstance(virt, hs.signals.Signal2D)


@pytest.mark.skipif(zarr_missing, reason="zarr not installed")
class TestLoadSave:
    @pytest.mark.parametrize(
        "metadata_file",
        [
            TEST_DATA_DIR / "3DSTEM_scan_info.txt",
            TEST_DATA_DIR / "3DTEM_scan_info.txt",
            TEST_DATA_DIR / "3DTEMDiffracting_scan_info.txt",
        ],
    )
    def test_mrc_metadata_save(self, metadata_file, tmp_path):
        """Saving an MRC-loaded signal to zspy should not raise a
        JSON-serialization error for numpy.void / bytes header fields."""
        s = hs.load(
            TEST_DATA_DIR / "20241021_00405_movie.mrc", metadata_file=metadata_file
        )
        diffracting = (
            "STEM" in metadata_file.name or "Diffracting" in metadata_file.name
        )
        s.axes_manager.navigation_axes[0].units = "sec"
        if diffracting:
            s.axes_manager.signal_axes[0].units = "nm^-1"
            s.axes_manager.signal_axes[1].units = "nm^-1"
        else:
            s.axes_manager.signal_axes[0].units = "nm"
            s.axes_manager.signal_axes[1].units = "nm"

        out = tmp_path / "out.zspy"
        # Must not raise TypeError: Object of type void is not JSON serializable
        s.save(str(out))

    def test_mrc_void_header_fields_are_hex_strings(self):
        """numpy.void and bytes fields in the MRC header (EXTRA, EXTRA2,
        CMAP, STAMP, LABELS) must be stored as plain hex strings, not raw
        numpy/bytes objects, so they can be JSON-serialized (e.g. for zspy)."""
        s = hs.load(TEST_DATA_DIR / "20241021_00405_movie.mrc")
        std_header = s.original_metadata.std_header
        for field in ("EXTRA", "EXTRA2", "CMAP", "STAMP", "LABELS"):
            value = std_header[field]
            assert isinstance(
                value, str
            ), f"Header field '{field}' should be a hex string, got {type(value)}"
            # Must be a valid hex string
            bytes.fromhex(value)

    def test_mrc_save_reload_zspy(self, tmp_path):
        """Round-trip: load MRC, save to zspy, reload – data and key metadata
        should be preserved, and no serialization errors should occur."""
        zarr = pytest.importorskip("zarr", reason="zarr not installed")  # noqa: F841
        s = hs.load(TEST_DATA_DIR / "20241021_00405_movie.mrc", lazy=True)
        out = tmp_path / "round_trip.zspy"
        s.save(str(out))
        s2 = hs.load(str(out))
        np.testing.assert_array_equal(s.data.compute(), s2.data)
