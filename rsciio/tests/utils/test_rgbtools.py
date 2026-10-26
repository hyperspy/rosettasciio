# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

import dask.array as da
import numpy as np

import rsciio.utils.rgb as rt


class TestRGBTools:
    def setup_method(self, method):
        self.data_c = np.ones((2, 2, 3), dtype=np.uint8, order="C")
        self.data_f = np.ones((2, 2, 3), dtype=np.uint8, order="F")
        mask = [[[0, 1, 1], [1, 0, 1]], [[1, 1, 0], [0, 0, 1]]]
        self.data_masked = np.ma.masked_array(self.data_c, mask, hard_mask=True)

    def test_rgbx2regular_array_corder_from_c(self):
        d = rt.rgbx2regular_array(self.data_c)
        assert d.flags["C_CONTIGUOUS"]

    def test_rgbx2regular_array_corder_from_c_dask(self):
        data = da.from_array(self.data_c)
        d = rt.rgbx2regular_array(data)
        assert d.flags["C_CONTIGUOUS"]

    def test_rgbx2regular_array_corder_from_f(self):
        d = rt.rgbx2regular_array(self.data_f)
        assert d.flags["C_CONTIGUOUS"]

    def test_rgbx2regular_array_corder_from_c_slices(self):
        d = rt.rgbx2regular_array(self.data_c[0:1, ...])
        assert d.flags["C_CONTIGUOUS"]
        d = rt.rgbx2regular_array(self.data_c[:, 0:1, :])
        assert d.flags["C_CONTIGUOUS"]

    def test_rgbx2regular_array_cordermask_from_cmasked(self):
        d = rt.rgbx2regular_array(self.data_masked)
        assert isinstance(d, np.ma.MaskedArray)
        assert d.flags["C_CONTIGUOUS"]

    def test_rgbx2regular_array_cordermask_from_cmasked_slices(self):
        d = rt.rgbx2regular_array(self.data_masked[0:1, ...])
        assert d.flags["C_CONTIGUOUS"]
        assert isinstance(d, np.ma.MaskedArray)
        d = rt.rgbx2regular_array(self.data_masked[:, 0:1, :])
        assert d.flags["C_CONTIGUOUS"]
        assert isinstance(d, np.ma.MaskedArray)

    def test_regular_array2rgbx_corder_from_c(self):
        d = rt.regular_array2rgbx(self.data_c)
        assert d.flags["C_CONTIGUOUS"]

    def test_regular_array2rgbx_corder_from_f(self):
        d = rt.regular_array2rgbx(self.data_f)
        assert d.flags["C_CONTIGUOUS"]

    def test_regular_array2rgbx_corder_from_c_slices(self):
        d = rt.regular_array2rgbx(self.data_c[0:1, ...])
        assert d.flags["C_CONTIGUOUS"]
        d = rt.regular_array2rgbx(self.data_c[:, 0:1, :])
        assert d.flags["C_CONTIGUOUS"]

    def test_regular_array2rgbx_cordermask_from_cmasked(self):
        d = rt.regular_array2rgbx(self.data_masked)
        assert isinstance(d, np.ma.MaskedArray)
        assert d.flags["C_CONTIGUOUS"]

    def test_regular_array2rgbx_cordermask_from_cmasked_slices(self):
        d = rt.regular_array2rgbx(self.data_masked[0:1, ...])
        assert isinstance(d, np.ma.MaskedArray)
        assert d.flags["C_CONTIGUOUS"]
        d = rt.regular_array2rgbx(self.data_masked[:, 0:1, :])
        assert isinstance(d, np.ma.MaskedArray)
        assert d.flags["C_CONTIGUOUS"]
