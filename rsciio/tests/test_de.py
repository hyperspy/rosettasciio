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

import dask.array
import pytest
import numpy as np
import os
from rsciio.de.api import SeqReader, CeleritasReader, file_reader

celeritas1_path = str(
    os.path.join(
        os.path.dirname(__file__), "de_data", "celeritas_data", "128x256_PRebuffer128"
    )
)
celeritas2_path = str(
    os.path.join(
        os.path.dirname(__file__), "de_data", "celeritas_data", "256x256_Prebuffer1"
    )
)
celeritas3_path = str(
    os.path.join(
        os.path.dirname(__file__), "de_data", "celeritas_data", "64x64_Prebuffer256"
    )
)

data_path = os.path.join(os.path.dirname(__file__), "de_data", "data")


class TestShared:
    @pytest.fixture
    def seq(self):
        return SeqReader(
            file=data_path + "/test.seq",
            dark=data_path + "/test.seq.dark.mrc",
            gain=data_path + "/test.seq.gain.mrc",
            metadata=data_path + "/test.seq.metadata",
            xml=data_path + "/test.seq.se.xml",
        )

    def test_parse_header(self, seq):
        header = seq._read_file_header()
        assert header["ImageWidth"] == 64
        assert header["ImageHeight"] == 64
        assert header["ImageBitDepthReal"] == 12
        assert header["NumFrames"] == 10
        assert header["TrueImageSize"] == 16384
        np.testing.assert_almost_equal(
            header["FPS"], 30, 1
        )  # Note this value wrong for Celeritas Camera
        # Read from the xml file... Factor of the frame buffer off

    def test_parse_metadata(self, seq):
        metadata = seq._read_metadata()
        print(metadata)

    def test_read_dark(self, seq):
        dark, gain = seq._read_dark_gain()
        assert dark.shape == (64, 64)
        assert gain.shape == (64, 64)

    def test_read_ref_none(self, seq):
        seq.gain = None
        dark, gain = seq._read_dark_gain()
        assert gain is None

    @pytest.mark.parametrize("nav_shape", [None, (5, 2), (5, 3)])
    @pytest.mark.parametrize("distributed", [True, False])
    @pytest.mark.parametrize("lazy", [True, False])
    def test_read(self, seq, nav_shape, distributed, lazy):
        data = seq.read_data(navigation_shape=nav_shape)
        data2 = seq.read_data(
            navigation_shape=nav_shape, distributed=distributed, lazy=lazy
        )
        if nav_shape is None:
            nav_shape = (10,)
        assert data["data"].shape == (*nav_shape, 64, 64)
        np.testing.assert_array_equal(data["data"], data2["data"])

    def test_file_reader(self, seq):
        file = data_path + "/test.seq"
        data_dict = file_reader(file)
        data_dict2 = seq.read_data()
        np.testing.assert_array_equal(data_dict[0]["data"], data_dict2["data"])


class TestLoadCeleritas:
    @pytest.fixture
    def seq(self):
        folder = celeritas1_path
        kws = {
            "file": folder + "/test.seq",
            "top": folder + "/test_Top_14-04-59.355.seq",
            "bottom": folder + "/test_Bottom_14-04-59.396.seq",
            "dark": folder + "/test.seq.dark.mrc",
            "gain": folder + "/test.seq.gain.mrc",
            "xml": folder + "/test.seq.Config.Metadata.xml",
            "metadata": folder + "/test_Top_14-04-59.355.seq.metadata",
        }
        return CeleritasReader(**kws)

    @pytest.fixture
    def seq2(self):
        folder = celeritas3_path
        kws = {
            "file": folder + "/test.seq",
            "top": folder + "/test_Top_14-13-42.780.seq",
            "bottom": folder + "/test_Bottom_14-13-42.822.seq",
            "dark": folder + "/test.seq.dark.mrc",
            "xml": folder + "/test.seq.Config.Metadata.xml",
            "metadata": folder + "/test_Top_14-13-42.780.seq",
        }
        return CeleritasReader(**kws)

    def test_parse_header(self, seq):
        print(seq.bottom)
        header = seq._read_file_header()
        assert header["ImageWidth"] == 256
        assert header["ImageHeight"] == 8192
        assert header["ImageBitDepthReal"] == 12
        assert header["NumFrames"] == 4  # this is wrong
        assert header["TrueImageSize"] == 4202496
        np.testing.assert_almost_equal(
            header["FPS"], 300, 1
        )  # This value is wrong for the celeritas camera

    def test_parse_metadata(self, seq):
        print(seq.metadata_file)
        header = seq._read_metadata()
        print(header)

    def test_parse_xml(self, seq):
        xml = seq._read_xml()
        assert xml["FileInfo"]["ImageSizeX"]["Value"] == 256
        assert xml["FileInfo"]["ImageSizeY"]["Value"] == 128
        assert xml["FileInfo"]["FrameRate"]["Value"] == 40000  # correct FPS
        assert xml["FileInfo"]["DarkRef"]["Value"] == "Yes"
        assert xml["FileInfo"]["GainRef"]["Value"] == "Yes"
        assert xml["FileInfo"]["SegmentPreBuffer"]["Value"] == 128
        assert not np.any(seq.metadata["Signal"]["BadPixels"])

    def test_bad_pixels_xml(self, seq):
        folder = celeritas1_path
        seq.xml = folder + "/test2.seq.Config.Metadata.xml"
        xml = seq._read_xml()
        assert xml["FileInfo"]["ImageSizeX"]["Value"] == 256
        assert xml["FileInfo"]["ImageSizeY"]["Value"] == 128
        assert xml["FileInfo"]["FrameRate"]["Value"] == 40000  # correct FPS
        assert xml["FileInfo"]["DarkRef"]["Value"] == "Yes"
        assert xml["FileInfo"]["GainRef"]["Value"] == "Yes"
        assert xml["FileInfo"]["SegmentPreBuffer"]["Value"] == 128
        data_dict = seq.read_data()
        data_dict["data"][:, seq.metadata["Signal"]["BadPixels"]] = 0
        assert np.any(seq.metadata["Signal"]["BadPixels"])

    def test_bad_pixel_failure(self, seq):
        with pytest.raises(KeyError):
            folder = celeritas1_path
            seq.xml = folder + "/test3.seq.Config.Metadata.xml"
            seq._read_xml()

    @pytest.mark.parametrize("nav_shape", [None, (5, 4), (5, 3)])
    @pytest.mark.parametrize("distributed", [True, False])
    @pytest.mark.parametrize("lazy", [True, False])
    def test_read(self, seq, nav_shape, distributed, lazy):
        data_dict = seq.read_data(
            navigation_shape=nav_shape,
        )

        data_dict2 = seq.read_data(
            navigation_shape=nav_shape, lazy=lazy, distributed=distributed
        )
        shape = (512, 128, 256)
        if nav_shape is not None:
            shape = nav_shape + shape[1:]
        assert data_dict["data"].shape == shape
        assert data_dict["axes"][-1]["size"] == data_dict["data"].shape[-1]
        assert data_dict["axes"][-2]["size"] == data_dict["data"].shape[-2]

        np.testing.assert_array_equal(data_dict["data"], data_dict2["data"])

    @pytest.mark.parametrize("chunks", [(2, 2, 128, 256), (4, 1, 128, 256)])
    @pytest.mark.parametrize("distributed", [True, False])
    def test_chunking(self, seq, chunks, distributed):
        data_dict = seq.read_data(
            navigation_shape=(4, 4),
            lazy=True,
            distributed=distributed,
            chunks=chunks,
        )
        assert isinstance(data_dict["data"], dask.array.Array)
        assert data_dict["data"].shape == (4, 4, 128, 256)
        chunk_sizes = tuple([c[0] for c in data_dict["data"].chunks[:2]])
        assert chunk_sizes == chunks[:2]

    @pytest.mark.parametrize(
        "kwargs",
        (
            {"filename": celeritas1_path + "/test_14-04-59.355.seq"},
            {"filename": None, "top": celeritas1_path + "/test_14-04-59.355.seq"},
        ),
    )
    def test_file_loader_failures(self, kwargs):
        with pytest.raises(ValueError):
            file_reader(
                **kwargs,
                celeritas=True,
            )

    def test_load_top_bottom(
        self,
    ):
        data_dict_top = file_reader(
            celeritas1_path + "/test_Top_14-04-59.355.seq",
            celeritas=True,
        )
        data_dict_bottom = file_reader(
            celeritas1_path + "/test_Bottom_14-04-59.396.seq",
            celeritas=True,
        )
        np.testing.assert_array_equal(
            data_dict_top[0]["data"], data_dict_bottom[0]["data"]
        )

    def test_read_data_no_xml(self, seq2):
        seq2.xml = None
        seq2.read_data()

    def test_load_file(self):
        data_dict = file_reader(
            celeritas1_path + "/test_Top_14-04-59.355.seq",
            celeritas=True,
        )
        assert data_dict[0]["data"].shape == (512, 128, 256)

    def test_load_file2(self):
        data_dict = file_reader(
            celeritas2_path + "/Movie_00785_Top_13-49-04.160.seq",
            celeritas=True,
        )
        assert data_dict[0]["data"].shape == (5, 256, 256)

    def test_load_file3(self):
        data_dict = file_reader(
            celeritas3_path + "/test_Bottom_14-13-42.822.seq",
            celeritas=True,
            lazy=True,
        )
        assert isinstance(data_dict[0]["data"], dask.array.Array)
        assert data_dict[0]["data"].shape == (512, 64, 64)

    def test_hyperspy(self):
        import hyperspy.api as hs

        hs.load(celeritas3_path + "/test_Bottom_14-13-42.822.seq", celeritas=True)
