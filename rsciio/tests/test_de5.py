# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.


# The EMD format is a hdf5 standard proposed at Lawrence Berkeley
# National Lab (see https://emdatasets.com/ for more information).
# NOT to be confused with the FEI EMD format which was developed later.

import numpy as np
import pytest

hs = pytest.importorskip("hyperspy.api")
pytest.importorskip("h5py")


def test_de5_write_load_cycle(tmp_path):
    fname = tmp_path / "test_file.de5"
    data = np.arange(20).reshape(2, 10)
    s = hs.signals.Signal1D(data)
    s.save(fname)
    assert fname.is_file()

    s2 = hs.load(fname)
    np.testing.assert_allclose(s2.data, s.data)
