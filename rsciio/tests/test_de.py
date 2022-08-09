
from hyperspy.io import load
import numpy as np

from rsciio.de.api import parse_header, parse_metadata, parse_xml,read_full_seq


class TestShared:
    def test_parse_header(self):
        header = parse_header("de_data/celeritas_data/test_SS8_Bottom_14-16-01.468.seq")
        assert header["ImageWidth"] == 256
        assert header["ImageHeight"] == 8192
        assert header["ImageBitDepthReal"] == 12
        assert header["NumFrames"] == 14
        assert header["ImgBytes"] == 4202496
        np.testing.assert_almost_equal(header["FPS"], 309.52, 1)  # This value is wrong for the celeritas camera

    def test_parse_xml(self):
        xml = parse_xml("de_data/celeritas_data/test_SS8.seq.Config.Metadata.xml")
        assert xml["ImageSizeX"] == 256
        assert xml["ImageSizeY"] == 256
        assert xml["FrameRate"] == 20000
        assert xml["DarkRef"] == "Yes"
        assert xml["GainRef"] == "Yes"
        assert xml["SegmentPreBuffer"] == 64

    def test_parse_metadata(self):
        metadata = parse_metadata("de_data/celeritas_data/test_SS8_Bottom_14-16-01.468.seq.metadata")


class TestLoadFull:
    def test_load(self):
        header = parse_header("de_data/data/test.seq")
        data = read_full_seq("de_data/data/test.seq",
                             ImageWidth=header["ImageWidth"],
                             ImageHeight=header["ImageHeight"],
                             ImageBitDepthReal=header["ImageBitDepthReal"],
                             TrueImageSize=header["TrueImageSize"])
        assert data["Array"].shape, (10, 2048, 2048)

class TestLoadCeleritas:
    def test_load(self):
        header = parse_header("de_data/celeritas_data/test_SS8_Bottom_14-16-01.468.seq")
        data = read_full_seq("de_data/celeritas_data/test_SS8_Bottom_14-16-01.468.seq",
                             ImageWidth=header["ImageWidth"],
                             ImageHeight=header["ImageHeight"],
                             ImageBitDepthReal=header["ImageBitDepthReal"],
                             TrueImageSize=header["TrueImageSize"])
        assert data["Array"].shape, (10, 2048, 2048)