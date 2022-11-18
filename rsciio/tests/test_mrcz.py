# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

import os
import tempfile
from time import perf_counter, sleep

import numpy as np
import numpy.testing as npt
import pytest
from datetime import datetime


hs = pytest.importorskip("hyperspy.api", reason="hyperspy not installed")
mrcz = pytest.importorskip("mrcz", reason="mrcz not installed")


from hyperspy.misc.test_utils import assert_deep_almost_equal

# ==============================================================================
# MRCZ Test
#
# Internal python-only test. Build a random image and save and re-load it.
# ==============================================================================
tmpDir = tempfile.gettempdir()

MAX_ASYNC_TIME = 10.0
dtype_list = ["float32", "int8", "int16", "uint16", "complex64"]


def _generate_parameters():
    parameters = []
    for dtype in dtype_list:
        for compressor in [None, "zstd", "lz4"]:
            for clevel in [1, 9]:
                for lazy in [True, False]:
                    parameters.append([dtype, compressor, clevel, lazy])
    return parameters


class TestPythonMrcz:
    def setup_method(self, method):
        pass

    def compareSaveLoad(
        self,
        testShape,
        dtype="int8",
        compressor=None,
        clevel=1,
        lazy=False,
        do_async=False,
        **kwargs
    ):

        # This is the main function which reads and writes from disk.
        mrcName = os.path.join(tmpDir, "testMage_{}_lazy_{}.mrcz".format(dtype, lazy))

        dtype = np.dtype(dtype)
        if dtype == "float32" or dtype == "float64":
            testData = np.random.normal(size=testShape).astype(dtype)
        elif dtype == "complex64" or dtype == "complex128":
            testData = np.random.normal(size=testShape).astype(
                dtype
            ) + 1.0j * np.random.normal(size=testShape).astype(dtype)
        else:  # integers
            testData = np.random.randint(10, size=testShape).astype(dtype)

        testSignal = hs.signals.Signal2D(testData)
        if lazy:
            testSignal = testSignal.as_lazy()

        # Add "File" metadata to testSignal
        testSignal.metadata.General.add_dictionary(
            {
                "FileIO": {
                    "0": {
                        "operation": "load",
                        "hyperspy_version": hs.__version__,
                        "io_plugin": "rsciio.mrcz",
                        "timestamp": datetime.now().astimezone().isoformat(),
                    }
                }
            }
        )

        # Unfortunately one cannot iterate over axes_manager in a Pythonic way
        # for axis in testSignal.axes_manager:
        testSignal.axes_manager[0].name = "z"
        testSignal.axes_manager[0].scale = np.random.uniform(low=0.0, high=1.0)
        testSignal.axes_manager[0].units = "nm"
        testSignal.axes_manager[1].name = "x"
        testSignal.axes_manager[1].scale = np.random.uniform(low=0.0, high=1.0)
        testSignal.axes_manager[1].units = "nm"
        testSignal.axes_manager[2].name = "y"
        testSignal.axes_manager[2].scale = np.random.uniform(low=0.0, high=1.0)
        testSignal.axes_manager[2].units = "nm"

        # Meta-data that goes into MRC fixed header
        testSignal.metadata.set_item("Acquisition_instrument.TEM.beam_energy", 300.0)
        # Meta-data that goes into JSON extended header
        testSignal.metadata.set_item("Acquisition_instrument.TEM.magnification", 25000)
        testSignal.metadata.set_item(
            "Signal.Noise_properties.Variance_linear_model.gain_factor", 1.0
        )

        testSignal.save(
            mrcName, compressor=compressor, clevel=clevel, do_async=do_async, **kwargs
        )
        if do_async:
            # Poll file on disk since we don't return the
            # concurrent.futures.Future
            t_stop = perf_counter() + MAX_ASYNC_TIME
            sleep(0.005)
            while perf_counter() < t_stop:
                try:
                    fh = open(mrcName, "a")
                    fh.close()
                    break
                except IOError:
                    sleep(0.001)
            print(
                "Time to save file: {} s".format(
                    perf_counter() - (t_stop - MAX_ASYNC_TIME)
                )
            )
            sleep(0.1)

        reSignal = hs.load(mrcName)
        try:
            os.remove(mrcName)
        except IOError:
            print("Warning: file {} left on disk".format(mrcName))

        # change file timestamp to make the metadata of both signals equal
        testSignal.metadata.General.FileIO.Number_0.timestamp = (
            reSignal.metadata.General.FileIO.Number_0.timestamp
        )

        npt.assert_array_almost_equal(testSignal.data.shape, reSignal.data.shape)
        npt.assert_array_almost_equal(testSignal.data, reSignal.data)
        npt.assert_almost_equal(
            testSignal.metadata.Acquisition_instrument.TEM.beam_energy,
            reSignal.metadata.Acquisition_instrument.TEM.beam_energy,
        )
        npt.assert_almost_equal(
            testSignal.metadata.Acquisition_instrument.TEM.magnification,
            reSignal.metadata.Acquisition_instrument.TEM.magnification,
        )

        for aName in ["x", "y", "z"]:
            npt.assert_equal(
                testSignal.axes_manager[aName].size, reSignal.axes_manager[aName].size
            )
            npt.assert_almost_equal(
                testSignal.axes_manager[aName].scale, reSignal.axes_manager[aName].scale
            )

        if dtype == "complex64":
            assert isinstance(reSignal, hs.signals.ComplexSignal2D)
        else:
            assert isinstance(reSignal, hs.signals.Signal2D)

        # delete last load operation from reSignal metadata so we can compare
        del reSignal.metadata.General.FileIO.Number_2
        assert_deep_almost_equal(
            testSignal.axes_manager.as_dictionary(),
            reSignal.axes_manager.as_dictionary(),
        )
        assert_deep_almost_equal(
            testSignal.metadata.as_dictionary(), reSignal.metadata.as_dictionary()
        )

        return reSignal

    @pytest.mark.parametrize(
        ("dtype", "compressor", "clevel", "lazy"), _generate_parameters()
    )
    def test_MRC(self, dtype, compressor, clevel, lazy):
        t_start = perf_counter()

        try:
            import blosc

            blosc_installed = True
        except BaseException:
            blosc_installed = False

        if not blosc_installed and compressor is not None:
            with pytest.raises(ImportError):
                return self.compareSaveLoad(
                    [2, 64, 32],
                    dtype=dtype,
                    compressor=compressor,
                    clevel=clevel,
                    lazy=lazy,
                )
        else:
            return self.compareSaveLoad(
                [2, 64, 32],
                dtype=dtype,
                compressor=compressor,
                clevel=clevel,
                lazy=lazy,
            )

    @pytest.mark.parametrize("dtype", dtype_list)
    def test_Async(self, dtype):
        _ = pytest.importorskip("blosc", reason="skipping test_async, requires blosc")
        t_start = perf_counter()
        self.compareSaveLoad(
            [2, 64, 32], dtype=dtype, compressor="zstd", clevel=1, do_async=True
        )
        print("MRCZ Asychronous test finished in {} s".format(perf_counter() - t_start))
