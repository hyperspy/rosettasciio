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


from pathlib import Path

import numpy as np
import pytest

hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")


TEST_DATA_PATH = Path(__file__).parent / "data" / "dm5"


class TestDM5:
    @pytest.mark.parametrize("navigation_dimension", [0, 1, 2])
    @pytest.mark.parametrize("signal_dimension", [1, 2])
    @pytest.mark.parametrize(
        "dtype",
        [
            np.uint8,
            np.uint16,
            np.float32,
            np.float64,
            np.complex128,
            np.int8,
            np.int16,
            np.int32,
        ],
    )
    def test_save_load_files(
        self, navigation_dimension, signal_dimension, dtype, tmp_path
    ):
        fname = (
            tmp_path
            / f"test_save_files_nav{navigation_dimension}_sig{signal_dimension}_{dtype().dtype}.dm5"
        )

        dim = navigation_dimension + signal_dimension
        data_shape = [10, 11, 12, 13, 14][:dim]
        data = np.ones(data_shape, dtype=dtype)
        if signal_dimension == 1:
            signal = hs.signals.Signal1D(data)
        else:
            signal = hs.signals.Signal2D(data)
        names = ["a", "b", "c", "d", "e"]
        for i in range(dim):
            ax = signal.axes_manager[i]
            ax.name = names[i] + str(ax.size)
            ax.units = names[i] + " nm"
            ax.scale = 0.1
        original = [signal.axes_manager[i].name for i in range(dim)]
        signal.save(fname, overwrite=True)
        s = hs.load(fname)

        assert s.data.shape == data.shape
        assert s.data.dtype == data.dtype
        assert s.axes_manager.navigation_dimension == navigation_dimension
        assert s.axes_manager.signal_dimension == signal_dimension
        for i in range(dim):
            assert s.axes_manager[i].name == original[i]
            assert "nm" in s.axes_manager[i].units
            assert s.axes_manager[i].scale == 0.1
            assert s.axes_manager[i].size == int(original[i][-2:])
