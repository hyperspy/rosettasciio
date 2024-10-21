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

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from rsciio import IO_PLUGINS

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")

from hyperspy.axes import DataAxis  # noqa: E402

TEST_DATA_PATH = Path(__file__).parent / "data"
FULLFILENAME = Path(__file__).parent / "test_io_overwriting.hspy"


class TestIOOverwriting:
    def setup_method(self, method):
        self.s = hs.signals.Signal1D(np.arange(10))
        self.new_s = hs.signals.Signal1D(np.ones(5))
        # make sure we start from a clean state
        self._clean_file()
        self.s.save(FULLFILENAME)
        self.s_file_hashed = self._hash_file(FULLFILENAME)

    def _hash_file(self, filename):
        with open(filename, "rb") as file:
            md5_hash = hashlib.md5(file.read())
            file_hashed = md5_hash.hexdigest()
        return file_hashed

    def _clean_file(self):
        if FULLFILENAME.exists():
            FULLFILENAME.unlink()

    def _check_file_is_written(self, filename):
        # Check that we have a different hash, in case the file have different
        # content from the original, the hash will be different.
        return not self.s_file_hashed == self._hash_file(filename)

    def test_io_overwriting_True(self):
        # Overwrite is True, when file exists we overwrite
        self.new_s.save(FULLFILENAME, overwrite=True)
        assert self._check_file_is_written(FULLFILENAME)

    def test_io_overwriting_False(self):
        # Overwrite if False, file exists we don't overwrite
        self.new_s.save(FULLFILENAME, overwrite=False)
        assert not self._check_file_is_written(FULLFILENAME)

    @pytest.mark.parametrize("overwrite", [None, True, False])
    def test_io_overwriting_no_existing_file(self, overwrite):
        self._clean_file()  # remove the file
        self.new_s.save(FULLFILENAME, overwrite=overwrite)
        assert self._check_file_is_written(FULLFILENAME)

    def test_io_overwriting_None_existing_file_y(self):
        # Overwrite is None, when file exists we ask, mock `y` here
        with patch("builtins.input", return_value="y"):
            self.new_s.save(FULLFILENAME)
            assert self._check_file_is_written(FULLFILENAME)

    def test_io_overwriting_None_existing_file_n(self):
        # Overwrite is None, when file exists we ask, mock `n` here
        with patch("builtins.input", return_value="n"):
            self.new_s.save(FULLFILENAME)
            assert not self._check_file_is_written(FULLFILENAME)

    def test_io_overwriting_invalid_parameter(self):
        with pytest.raises(ValueError, match="parameter can only be"):
            self.new_s.save(FULLFILENAME, overwrite="spam")

    def teardown_method(self, method):
        self._clean_file()


class TestNonUniformAxisCheck:
    def setup_method(self, method):
        axis = DataAxis(axis=1 / (np.arange(10) + 1), navigate=False)
        self.s = hs.signals.Signal1D(np.arange(10), axes=(axis.get_axis_dictionary(),))
        # make sure we start from a clean state

    def test_io_nonuniform(self, tmp_path):
        assert self.s.axes_manager[0].is_uniform is False
        self.s.save(tmp_path / "tmp.hspy")
        with pytest.raises(TypeError, match="not supported for non-uniform"):
            self.s.save(tmp_path / "tmp.msa")

    def test_nonuniform_writer_characteristic(self):
        for plugin in IO_PLUGINS:
            if "non_uniform_axis" not in plugin:
                print(
                    f"{plugin.name} IO-plugin is missing the "
                    "characteristic `non_uniform_axis`"
                )

    def test_nonuniform_error(self, tmp_path):
        assert self.s.axes_manager[0].is_uniform is False
        incompatible_writers = [
            plugin["file_extensions"][plugin["default_extension"]]
            for plugin in IO_PLUGINS
            if (
                plugin["writes"] is True
                or plugin["writes"] is not False
                and [1, 0] in plugin["writes"]
            )
            and not plugin["non_uniform_axis"]
        ]
        for ext in incompatible_writers:
            with pytest.raises(TypeError, match="not supported for non-uniform"):
                filename = "tmp." + ext
                self.s.save(tmp_path / filename, overwrite=True)


def test_glob_wildcards():
    s = hs.signals.Signal1D(np.arange(10))

    with tempfile.TemporaryDirectory() as dirpath:
        fnames = [os.path.join(dirpath, f"temp[1x{x}].hspy") for x in range(2)]

        for f in fnames:
            s.save(f)

        with pytest.raises(ValueError, match="No filename matches the pattern"):
            _ = hs.load(fnames[0])

        t = hs.load([fnames[0]])
        assert len(t) == 1

        t = hs.load(fnames)
        assert len(t) == 2

        t = hs.load(os.path.join(dirpath, "temp*.hspy"))
        assert len(t) == 2

        t = hs.load(
            os.path.join(dirpath, "temp[*].hspy"),
            escape_square_brackets=True,
        )
        assert len(t) == 2

        with pytest.raises(ValueError, match="No filename matches the pattern"):
            _ = hs.load(os.path.join(dirpath, "temp[*].hspy"))

        # Test pathlib.Path
        t = hs.load(Path(dirpath, "temp[1x0].hspy"))
        assert len(t) == 1

        t = hs.load([Path(dirpath, "temp[1x0].hspy"), Path(dirpath, "temp[1x1].hspy")])
        assert len(t) == 2

        t = hs.load(list(Path(dirpath).glob("temp*.hspy")))
        assert len(t) == 2

        t = hs.load(Path(dirpath).glob("temp*.hspy"))
        assert len(t) == 2


def test_file_not_found_error(tmp_path):
    temp_fname = tmp_path / "temp.hspy"

    if os.path.exists(temp_fname):
        os.remove(temp_fname)

    with pytest.raises(ValueError, match="No filename matches the pattern"):
        _ = hs.load(temp_fname)

    with pytest.raises(FileNotFoundError):
        _ = hs.load([temp_fname])


def test_file_reader_error(tmp_path):
    # Only None, str or objects with attr "file_reader" are supported
    s = hs.signals.Signal1D(np.arange(10))

    f = tmp_path / "temp.hspy"
    s.save(f)

    with pytest.raises(ValueError, match="reader"):
        _ = hs.load(f, reader=123)


def test_file_reader_warning(caplog, tmp_path):
    s = hs.signals.Signal1D(np.arange(10))

    f = tmp_path / "temp.hspy"
    s.save(f)
    try:
        with caplog.at_level(logging.WARNING):
            _ = hs.load(f, reader="some_unknown_file_extension")

        assert "Unable to infer file type from extension" in caplog.text
    except (ValueError, OSError):
        # Test fallback to Pillow imaging library
        pass


def test_file_reader_options():
    s = hs.signals.Signal1D(np.arange(10))

    with tempfile.TemporaryDirectory() as dirpath:
        f = os.path.join(dirpath, "temp.hspy")
        s.save(f)
        f2 = os.path.join(dirpath, "temp.emd")
        s.save(f2)

        # Test string reader
        t = hs.load(Path(dirpath, "temp.hspy"), reader="hspy")
        assert len(t) == 1
        np.testing.assert_allclose(t.data, np.arange(10))

        # Test string reader uppercase
        t = hs.load(Path(dirpath, "temp.hspy"), reader="HSpy")
        assert len(t) == 1
        np.testing.assert_allclose(t.data, np.arange(10))

        # Test string reader alias
        t = hs.load(Path(dirpath, "temp.hspy"), reader="hyperspy")
        assert len(t) == 1
        np.testing.assert_allclose(t.data, np.arange(10))

        # Test string reader name
        t = hs.load(Path(dirpath, "temp.emd"), reader="emd")
        assert len(t) == 1
        np.testing.assert_allclose(t.data, np.arange(10))

        # Test string reader aliases
        t = hs.load(Path(dirpath, "temp.emd"), reader="Electron Microscopy Data (EMD)")
        assert len(t) == 1
        np.testing.assert_allclose(t.data, np.arange(10))
        t = hs.load(Path(dirpath, "temp.emd"), reader="Electron Microscopy Data")
        assert len(t) == 1
        np.testing.assert_allclose(t.data, np.arange(10))

        # Test object reader
        from rsciio import hspy

        t = hs.load(Path(dirpath, "temp.hspy"), reader=hspy)
        assert len(t) == 1
        np.testing.assert_allclose(t.data, np.arange(10))


def test_save_default_format(tmp_path):
    s = hs.signals.Signal1D(np.arange(10))

    f = tmp_path / "temp"
    s.save(f)

    t = hs.load(tmp_path / "temp.hspy")
    assert len(t) == 1


def test_load_original_metadata(tmp_path):
    s = hs.signals.Signal1D(np.arange(10))
    s.original_metadata.a = 0

    f = tmp_path / "temp"
    s.save(f)
    assert s.original_metadata.as_dictionary() != {}

    t = hs.load(tmp_path / "temp.hspy")
    assert t.original_metadata.as_dictionary() == s.original_metadata.as_dictionary()

    t = hs.load(tmp_path / "temp.hspy", load_original_metadata=False)
    assert t.original_metadata.as_dictionary() == {}


def test_load_save_filereader_metadata():
    # tests that original FileReader metadata is correctly persisted and
    # appended through a save and load cycle
    s = hs.load(TEST_DATA_PATH / "msa" / "example1.msa")
    assert s.metadata.General.FileIO.Number_0.io_plugin == "rsciio.msa"
    assert s.metadata.General.FileIO.Number_0.operation == "load"
    assert s.metadata.General.FileIO.Number_0.hyperspy_version == hs.__version__

    with tempfile.TemporaryDirectory() as dirpath:
        f = os.path.join(dirpath, "temp")
        s.save(f)
        expected = {
            "0": {
                "io_plugin": "rsciio.msa",
                "operation": "load",
                "hyperspy_version": hs.__version__,
            },
            "1": {
                "io_plugin": "rsciio.hspy",
                "operation": "save",
                "hyperspy_version": hs.__version__,
            },
            "2": {
                "io_plugin": "rsciio.hspy",
                "operation": "load",
                "hyperspy_version": hs.__version__,
            },
        }
        del s.metadata.General.FileIO.Number_0.timestamp  # runtime dependent
        del s.metadata.General.FileIO.Number_1.timestamp  # runtime dependent
        assert s.metadata.General.FileIO.Number_0.as_dictionary() == expected["0"]
        assert s.metadata.General.FileIO.Number_1.as_dictionary() == expected["1"]

        t = hs.load(Path(dirpath, "temp.hspy"))
        del t.metadata.General.FileIO.Number_0.timestamp  # runtime dependent
        del t.metadata.General.FileIO.Number_1.timestamp  # runtime dependent
        del t.metadata.General.FileIO.Number_2.timestamp  # runtime dependent
        assert t.metadata.General.FileIO.as_dictionary() == expected
