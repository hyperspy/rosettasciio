# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
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

import h5py
import numpy as np
import xml.etree.ElementTree as ET

from rsciio._docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC
from rsciio.utils.tools import _UREG, convert_xml_to_dict


old_app5e = h5py.File("/home/arg6/workspace/1750 1.app5", "r")
meta = old_app5e["18d9446f-22bf-4fb1-8d13-338174e75d20"]["Metadata"][
    ()
].decode()
root = ET.fromstring(meta)


def topspin_XML_to_dict(metadata_string: str, recursion_limit: int = 6):
    """
    Converts 'MetaData' strings into nested python dictionaries.

    Parameters
    ----------
    metadata_string
        text representation of app5 metadata. Can be generated
        from an app5 file opened with `h5py` as *f* using:
            metadata_string = f['path/to/Metadata'][()].decode()

    Returns
    -------
    dict
        Nested metadata dictionary.

    Notes
    -----
    MetaData strings typically have the same 18 Elements. 16 of these
    can be directly converted to dictionary key/item pairs. However, two
    of them ('ProcedureData' and 'HardwareSettings') contain nested
    XML Elements that follow the same pattern:

    <Item>
      <Name Serializer="String" Version="1">ScanRotation</Name>
      <Value Serializer="Double" Version="1">0</Value>
    </Item>

    which is equivalent to {'ScanRotation':float(0)}

    This function checks Value.attrib['Serializer'] for every name/value
    pair. If it's a python type, the pair are added to the dictionary.
    Otherwise, it is the start of a new Element leaf with one or more
    name/value pairs, and thus a new nested dictionary will be created.
    """

    def name_val_decode(element, recursion):
        recursion += 1
        if recursion > 6:
            return element.tag

        serializer = element.attrib["Serializer"]
        if serializer == "Boolean":
            out = {element.tag: bool(element.text)}
        elif serializer == "Double":
            out = {element.tag: float(element.text)}
        elif serializer == "String":
            out = {element.tag: str(element.text)}
        elif serializer in ["UInt32", "Int32"]:
            out = {element.tag: int(element.text)}
        else:
            out = {}
            for leaf in element:
                if len(leaf.attrib) > 0:
                    out.update(name_val_decode(leaf, recursion))
        return out

    root = ET.fromstring(metadata_string)
    metadata_dict = {}
    for branch in root:
        if branch.tag in ["ProcedureData", "HardwareSettings"]:
            branch_dict = {}
            for leaf in branch:
                serializer = leaf[1].attrib["Serializer"]
                if serializer == "Boolean":
                    branch_dict[leaf[0].text] = leaf[1].text
                elif serializer == "Double":
                    branch_dict[leaf[0].text] = float(leaf[1].text)
                elif serializer == "String":
                    branch_dict[leaf[0].text] = str(leaf[1].text)
                elif serializer in ["UInt32", "Int32"]:
                    branch_dict[leaf[0].text] = int(leaf[1].text)
                else:
                    # everything else follows the same nested pattern
                    try:
                        branch_dict[leaf[0].text] = name_val_decode(leaf[1], 0)
                    except:
                        raise Warning(
                            "rsciio was unable to read"
                            + "{} from {} in the app5 metadata".format(
                                leaf[0].text, branch.tag
                            )
                        )
            metadata_dict[branch.tag] = branch_dict
        else:
            metadata_dict[branch.tag] = branch.text
    return metadata_dict


test = topspin_XML_to_dict(meta)
