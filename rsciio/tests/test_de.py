import matplotlib.pyplot as plt
import pytest
from hyperspy.io import load
import numpy as np

from rsciio.de.api import SeqReader,CeleritasReader


class TestShared:
    @pytest.fixture
    def seq(self):
        return SeqReader(file="de_data/data/test.seq",
                         dark="de_data/data/test.seq.dark.mrc",
                         gain="de_data/data/test.seq.gain.mrc",
                         metadata="de_data/data/test.seq.metadata",
                         xml="de_data/data/test.seq.se.xml")

    def test_parse_header(self, seq):
        header = seq._read_file_header()
        assert header["ImageWidth"] == 2048
        assert header["ImageHeight"] == 2048
        assert header["ImageBitDepthReal"] == 12
        assert header["NumFrames"] == 10
        assert header["TrueImageSize"] == 8396800
        np.testing.assert_almost_equal(header["FPS"], 11.111111, 1)  # This value is wrong for the celeritas camera

    def test_parse_metadata(self, seq):
        metadata = seq._read_metadata()

    def test_read_dark(self, seq):
        dark,gain = seq._read_dark_gain()
        assert gain.shape == dark.shape

    def test_read(self, seq):
        seq.read_data()


class TestLoadCeleritas:
    @pytest.fixture
    def seq(self):
        return CeleritasReader(file="de_data/celeritas_data/test_SS8.seq",
                               top="de_data/celeritas_data/test_SS8_Top_14-16-01.432.seq",
                               bottom="de_data/celeritas_data/test_SS8_Bottom_14-16-01.468.seq",
                               dark="de_data/celeritas_data/test_SS8.seq.dark.mrc",
                               gain="de_data/celeritas_data/test_SS8.seq.gain.mrc",
                               xml="de_data/celeritas_data/test_SS8.seq.Config.Metadata.xml",
                               metadata="de_data/celeritas_data/test_SS8_Bottom_14-16-01.468.seq.metadata")

    def test_parse_header(self, seq):
        header = seq._read_file_header()
        assert header["ImageWidth"] == 256
        assert header["ImageHeight"] == 8192
        assert header["ImageBitDepthReal"] == 12
        assert header["NumFrames"] == 14 #this is wrong
        assert header["TrueImageSize"] == 4202496
        np.testing.assert_almost_equal(header["FPS"], 309.5, 1)  # This value is wrong for the celeritas camera

    def test_parse_xml(self, seq):
        xml = seq._read_xml()
        assert xml["ImageSizeX"] == 256
        assert xml["ImageSizeY"] == 256
        assert xml["FrameRate"] == 20000
        assert xml["DarkRef"] == "Yes"
        assert xml["GainRef"] == "Yes"
        assert xml["SegmentPreBuffer"] == 64

    def test_read(self, seq):
        dict = seq.read_data()
