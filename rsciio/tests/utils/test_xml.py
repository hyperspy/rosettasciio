from pathlib import Path

import pytest

from rsciio.utils.xml import ET, XmlToDict, sanitize_msxml_float


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
