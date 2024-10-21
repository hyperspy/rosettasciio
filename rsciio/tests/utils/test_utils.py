from pathlib import Path

import numpy as np
import pytest
from dateutil import parser, tz

import rsciio.utils.date_time_tools as dtt
from rsciio.utils.distributed import get_chunk_slice
from rsciio.utils.tools import ET, DTBox, XmlToDict, dict2sarray, sanitize_msxml_float

dt = [("x", np.uint8), ("y", np.uint16), ("text", (bytes, 6))]


@pytest.fixture
def XML_TEST_NODE():
    MY_PATH = Path(__file__).parent
    TEST_XML_PATH = MY_PATH / ".." / "data" / "ToastedBreakFastSDD.xml"

    with open(TEST_XML_PATH, "r") as fn:
        weird_but_valid_xml_str = fn.read()

    yield ET.fromstring(weird_but_valid_xml_str)


# fmt: off
def test_msxml_sanitization():
    bad_msxml = b"""
    <main>
      <simpleFloat>0,2</simpleFloat>
      <scientificSmall>1,9E-5</scientificSmall>
      <scientificBig>0,2E10</scientificBig>
      <actuallyList>0,2,3</actuallyList>
    </main>
    """
    sanitized_xml_bytes = sanitize_msxml_float(bad_msxml)
    et = ET.fromstring(sanitized_xml_bytes)
    assert et[0].text == "0.2"
    assert et[1].text == "1.9E-5"
    assert et[2].text == "0.2E10"
    assert et[3].text == "0,2,3"  # is not float


def test_default_x2d(XML_TEST_NODE):
    """test of default XmlToDict translation with attributes prefixed with @,
    interchild_text_parsing set to 'first',
    no flattening tags set, and dub_text_str set to '#value'
    """
    x2d = XmlToDict()
    pynode = x2d.dictionarize(XML_TEST_NODE)
    assert (
        pynode["TestXML"]["Main"]["ClassInstance"]["Sample"]["Components"][
            "ComponentChildren"
        ]["Instance"][0]["name"]
        == 'Eggs'
    )
    t = "With one of these components"
    assert pynode["TestXML"]["Main"]["ClassInstance"]["Sample"]["#value"] == t


def test_skip_interchild_text_flatten(XML_TEST_NODE):
    """test of XmlToDict translation with interchild_text_parsing set to 'skip',
    three string containing list set to flattening tags. Other kwrds - default.
    """
    x2d = XmlToDict(
        interchild_text_parsing='skip',
        tags_to_flatten=["ClassInstance", "ComponentChildren", "Instance", "TestXML"],
    )
    pynode = x2d.dictionarize(XML_TEST_NODE)
    assert pynode["Main"]["Sample"]["Components"]["name"][0] == "Eggs"
    assert pynode["Main"]["Sample"].get("#value") is None


def test_concat_interchild_text_val_flatten(XML_TEST_NODE):
    """test of XmlToDict translator with interchild_text_parsing set to
    'cat' (concatenation), four flattening tags set, and dub_text_str set
    to '#text'
    """
    x2d = XmlToDict(
        dub_text_str="#text",
        interchild_text_parsing='cat',
        tags_to_flatten=["ClassInstance", "ComponentChildren", "Instance", "Main"],
    )
    pynode = x2d.dictionarize(XML_TEST_NODE.find("Main"))
    assert pynode["Instrument"]["Type"].get("#value") is None
    assert pynode["Instrument"]["Type"]["#text"] == "Toaster"
    assert pynode["Sample"].get("#value") is None
    assert pynode["Sample"].get("#text") is None
    t = "With one of these componentsSDD risks to be Toasted."
    assert pynode["Sample"]["#interchild_text"] == t


def test_list_interchild_text_val_flatten(XML_TEST_NODE):
    """test of XmlToDict translator interchild_text_parsing set to 'list'
    """
    x2d = XmlToDict(
        dub_text_str="#value",
        interchild_text_parsing='list',
        tags_to_flatten=["ClassInstance", "ComponentChildren", "Instance"]
    )
    pynode = x2d.dictionarize(XML_TEST_NODE.find("Main//Sample"))
    assert pynode["Sample"]["#interchild_text"] == [
        "With one of these components",
        "",
        "SDD risks to be Toasted.",
    ]


def x2d_subclass_for_custom_bool(XML_TEST_NODE):
    """test subclass of XmlToDict with updated eval function"""

    class CustomXmlToDict(XmlToDict):
        @staticmethod
        def eval(string):
            if string == "not today":
                return False
            if string == "affirmative":
                return True
            return XmlToDict.eval(string)

    x2d = CustomXmlToDict(
        dub_text_str="#value",
        tags_to_flatten=["ClassInstance", "ComponentChildren", "Instance"],
    )
    node = XML_TEST_NODE.find("Main//Instrument")
    pynode = x2d.dictionarize(node)
    assert pynode["IsToasted"] is False
    assert pynode["IsToasting"] is True


def test_wrong_type_x2d_initiation():
    with pytest.raises(ValueError):
        XmlToDict(dub_attr_pre_str=1)
    with pytest.raises(ValueError):
        XmlToDict(tags_to_flatten=0)
    with pytest.raises(ValueError):
        XmlToDict(interchild_text_parsing="simple")
    with pytest.raises(ValueError):
        XmlToDict(dub_text_str=2)


def _get_example(date, time, time_zone=None):
    md = DTBox({"General": {"date": date, "time": time}}, box_dots=True)
    if time_zone:
        md.set_item("General.time_zone", time_zone)
        dt = parser.parse("%sT%s" % (date, time))
        dt = dt.replace(tzinfo=tz.gettz(time_zone))
        iso = dt.isoformat()
    else:
        iso = "%sT%s" % (date, time)
        dt = parser.parse(iso)
    return md, dt, iso


md1, dt1, iso1 = _get_example("2014-12-27", "00:00:00", "UTC")
serial1 = 42000.00

md2, dt2, iso2 = _get_example("2124-03-25", "10:04:48", "EST")
serial2 = 81900.62833333334

md3, dt3, iso3 = _get_example("2016-07-12", "22:57:32")
serial3 = 42563.95662037037


def test_d2s_fail():
    d = dict(x=5, y=10, text="abcdef")
    with pytest.raises(ValueError):
        dict2sarray(d)


def test_d2s_dtype():
    d = dict(x=5, y=10, text="abcdef")
    ref = np.zeros((1,), dtype=dt)
    ref["x"] = 5
    ref["y"] = 10
    ref["text"] = "abcdef"

    assert ref == dict2sarray(d, dtype=dt)


def test_d2s_extra_dict_ok():
    d = dict(x=5, y=10, text="abcdef", other=55)
    ref = np.zeros((1,), dtype=dt)
    ref["x"] = 5
    ref["y"] = 10
    ref["text"] = "abcdef"

    assert ref == dict2sarray(d, dtype=dt)


def test_d2s_sarray():
    d = dict(x=5, y=10, text="abcdef")

    base = np.zeros((1,), dtype=dt)
    base["x"] = 65
    base["text"] = "gg"

    ref = np.zeros((1,), dtype=dt)
    ref["x"] = 5
    ref["y"] = 10
    ref["text"] = "abcdef"

    assert ref == dict2sarray(d, sarray=base)


def test_d2s_partial_sarray():
    d = dict(text="abcdef")

    base = np.zeros((1,), dtype=dt)
    base["x"] = 65
    base["text"] = "gg"

    ref = np.zeros((1,), dtype=dt)
    ref["x"] = 65
    ref["y"] = 0
    ref["text"] = "abcdef"

    assert ref == dict2sarray(d, sarray=base)


def test_d2s_type_cast_ok():
    d = dict(x="34", text=55)

    ref = np.zeros((1,), dtype=dt)
    ref["x"] = 34
    ref["y"] = 0
    ref["text"] = "55"

    assert ref == dict2sarray(d, dtype=dt)


def test_d2s_type_cast_invalid():
    d = dict(x="Test")
    with pytest.raises(ValueError):
        dict2sarray(d, dtype=dt)


def test_d2s_string_cut():
    d = dict(text="Testerstring")
    sa = dict2sarray(d, dtype=dt)
    assert sa["text"][0] == b"Tester"


def test_d2s_array1():
    dt2 = dt + [("z", (np.uint8, 4)), ("u", (np.uint16, 4))]
    d = dict(z=2, u=[1, 2, 3, 4])
    sa = dict2sarray(d, dtype=dt2)
    np.testing.assert_array_equal(sa["z"][0], [2, 2, 2, 2])
    np.testing.assert_array_equal(sa["u"][0], [1, 2, 3, 4])


def test_d2s_array2():
    d = dict(x=2, y=[1, 2, 3, 4])
    sa = np.zeros((4,), dtype=dt)
    sa = dict2sarray(d, sarray=sa)
    np.testing.assert_array_equal(sa["x"], [2, 2, 2, 2])
    np.testing.assert_array_equal(sa["y"], [1, 2, 3, 4])


def test_d2s_arrayX():
    dt2 = dt + [("z", (np.uint8, 4)), ("u", (np.uint16, 4))]
    d = dict(z=2, u=[1, 2, 3, 4])
    sa = np.zeros((4,), dtype=dt2)
    sa = dict2sarray(d, sarray=sa)
    np.testing.assert_array_equal(
        sa["z"],
        [
            [2, 2, 2, 2],
        ]
        * 4,
    )
    np.testing.assert_array_equal(
        sa["u"],
        [
            [1, 2, 3, 4],
        ]
        * 4,
    )


def test_serial_date_to_ISO_format():
    iso_1 = dtt.serial_date_to_ISO_format(serial1)
    dt1_local = dt1.astimezone(tz.tzlocal())
    assert iso_1[0] == dt1_local.date().isoformat()
    assert iso_1[1] == dt1_local.time().isoformat()
    assert iso_1[2] == dt1_local.tzname()

    iso_2 = dtt.serial_date_to_ISO_format(serial2)
    dt2_local = dt2.astimezone(tz.tzlocal())
    assert iso_2[0] == dt2_local.date().isoformat()
    # The below line will/can fail due to accuracy loss when converting to serial date:
    # We therefore truncate milli/micro seconds
    assert iso_2[1][:8] == dt2_local.time().isoformat()
    assert iso_2[2] == dt2_local.tzname()

    iso_3 = dtt.serial_date_to_ISO_format(serial3)
    dt3_aware = dt3.replace(tzinfo=tz.tzutc())
    dt3_local = dt3_aware.astimezone(tz.tzlocal())
    assert iso_3[0] == dt3_local.date().isoformat()
    assert iso_3[1] == dt3_local.time().isoformat()
    assert iso_3[2] == dt3_local.tzname()


def test_ISO_format_to_serial_date():
    res1 = dtt.ISO_format_to_serial_date(
        dt1.date().isoformat(), dt1.time().isoformat(), timezone=dt1.tzname()
    )
    np.testing.assert_allclose(res1, serial1, atol=1e-5)
    dt = dt2.astimezone(tz.tzlocal())
    res2 = dtt.ISO_format_to_serial_date(
        dt.date().isoformat(), dt.time().isoformat(), timezone=dt.tzname()
    )
    np.testing.assert_allclose(res2, serial2, atol=1e-5)
    res3 = dtt.ISO_format_to_serial_date(
        dt3.date().isoformat(), dt3.time().isoformat(), timezone=dt3.tzname()
    )
    np.testing.assert_allclose(res3, serial3, atol=1e-5)


def test_datetime_to_serial_date():
    np.testing.assert_allclose(dtt.datetime_to_serial_date(dt1), serial1, atol=1e-5)
    np.testing.assert_allclose(dtt.datetime_to_serial_date(dt2), serial2, atol=1e-5)
    np.testing.assert_allclose(dtt.datetime_to_serial_date(dt3), serial3, atol=1e-5)


def _get_example(date, time, time_zone=None):
    md = {"General": {"date": date, "time": time}}
    if time_zone:
        md["General"]["time_zone"] = time_zone
        dt = parser.parse(f"{date}T{time}")
        dt = dt.replace(tzinfo=tz.gettz(time_zone))
        iso = dt.isoformat()
    else:
        iso = f"{date}T{time}"
        dt = parser.parse(iso)
    return md, dt, iso


md1, dt1, iso1 = _get_example("2014-12-27", "00:00:00", "UTC")
serial1 = 42000.00

md2, dt2, iso2 = _get_example("2124-03-25", "10:04:48", "EST")
serial2 = 81900.62833333334

md3, dt3, iso3 = _get_example("2016-07-12", "22:57:32")
serial3 = 42563.95662037037


def test_get_date_time_from_metadata():
    assert dtt.get_date_time_from_metadata(md1) == "2014-12-27T00:00:00+00:00"
    assert (
        dtt.get_date_time_from_metadata(md1, formatting="ISO")
        == "2014-12-27T00:00:00+00:00"
    )
    assert dtt.get_date_time_from_metadata(
        md1, formatting="datetime64"
    ) == np.datetime64("2014-12-27T00:00:00.000000")
    assert dtt.get_date_time_from_metadata(md1, formatting="datetime") == dt1

    assert dtt.get_date_time_from_metadata(md2) == "2124-03-25T10:04:48-05:00"
    assert dtt.get_date_time_from_metadata(md2, formatting="datetime") == dt2
    assert dtt.get_date_time_from_metadata(
        md2, formatting="datetime64"
    ) == np.datetime64("2124-03-25T10:04:48")

    assert dtt.get_date_time_from_metadata(md3) == "2016-07-12T22:57:32"
    assert dtt.get_date_time_from_metadata(md3, formatting="datetime") == dt3
    assert dtt.get_date_time_from_metadata(
        md3, formatting="datetime64"
    ) == np.datetime64("2016-07-12T22:57:32.000000")

    assert dtt.get_date_time_from_metadata({"General": {}}) is None
    assert (
        dtt.get_date_time_from_metadata({"General": {"date": "2016-07-12"}})
        == "2016-07-12"
    )
    assert dtt.get_date_time_from_metadata({"General": {"time": "12:00"}}) == "12:00:00"
    assert (
        dtt.get_date_time_from_metadata(
            {"General": {"time": "12:00", "time_zone": "CET"}}
        )
        == "12:00:00"
    )


@pytest.mark.parametrize(
        "shape",
        ((10, 20, 30, 512, 512), (20, 30, 512, 512), (10, 512, 512), (512, 512))
)
def test_get_chunk_slice(shape):
    chunk_arr, chunk = get_chunk_slice(shape=shape, chunks=-1)  # 1 chunk
    assert chunk_arr.shape == (1,)*len(shape)+(len(shape), 2)
    assert chunk == tuple([(i,)for i in shape])

    chunks = (1,)*(len(shape)-2) +(-1,-1)
    # Eveything is 1 chunk
    chunk_arr, chunk = get_chunk_slice(shape=shape, chunks=chunks)
    assert chunk_arr.shape == shape[:-2]+(1, 1) + (len(shape), 2)
    assert chunk == (
        tuple([(1,)*i for i in shape[:-2]])+tuple([(i,) for i in shape[-2:]])
    )
