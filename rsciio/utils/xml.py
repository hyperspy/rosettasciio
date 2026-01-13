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
"""Utility functions for XML handling."""

import re
import xml.etree.ElementTree as ET
from ast import literal_eval
from collections import defaultdict

from rsciio.utils._dictionary import DTBox

__all__ = [
    "XmlToDict",
    "convert_xml_to_dict",
    "sanitize_msxml_float",
    "xml2dtb",
]


def __dir__():
    return sorted(__all__)


# MSXML sanitization ###
# re pattern with two capturing groups with comma in between;
# firstgroup looks for numeric value after <tag> (the '>' char) with or
# without minus sign, second group looks for numeric value with following
# closing <\tag> (the '<' char); '([Ee]-?\d*)' part (optionally a third group)
# checks for scientific notation (e.g. 8,843E-7 -> 'E-7');
# compiled pattern is binary, as raw xml string is binary string.:
_fix_dec_patterns = re.compile(b"(>-?\\d+),(\\d*([Ee]-?\\d*)?<)")


def sanitize_msxml_float(xml_b_string):
    """
    Replace comma with dot in floating point numbers in given xml
    raw string.

    Parameters
    ----------
    xml_b_string : str
        Raw binary string representing the xml to be parsed.

    Returns
    -------
    str
        Binary string with commas used as decimal marks replaced with dots
        to adhere to XML standard.

    Notes
    -----
    What, why, how? In case OEM software runs on MS Windows and directly
    uses system built-in MSXML lib, which does not comply with XML standards,
    and OS is set to locale of some country with weird and illogical
    preferences of using comma as decimal separation;
    Software in conjunction of these above listed conditions can produce
    not-interoperable XML, which leads to wrong interpretation of context.
    This sanitizer searches and corrects that - it should be used before
    being fed to .fromstring of element tree.
    """
    return _fix_dec_patterns.sub(b"\\1.\\2", xml_b_string)


class XmlToDict:
    """
    Customisable XML to python dict and list based Hierarchical tree
    translator.

    Parameters
    ----------
    dub_attr_pre_str : str
        String to be prepended to attribute name when creating
        dictionary tree if children element with same name is used.
        Default is "@".
    dub_text_str : str (default: "#text")
        String to use as key in case element contains text and children tag.
        Default "#text".
    tags_to_flatten : None, str or list of str
        Define tag names which should be flattened/skipped,
        placing children of such tag one level shallower in constructed
        python structure.
        It is useful when OEM generated XML are not human designed,
        but machine/programming language/framework generated
        and painfully verbose. See example below.
        Default is None, which means no tags are flattened.
    interchild_text_parsing : str
        Must be one of ("skip", "first", "cat", "list").
        This considers the  behaviour when both .text and children tags are
        presented under same element tree node:

        - "skip" - will not try to retrieve any .text values from such node.
        - "first" - only string under .text attribute will be returned.
        - "cat" - return concatenated string from .text of node and .tail's
          of children nodes.
        - "list" - similar to "cat", but return the result in list
          without concatenation.

        Default is "first", which is the most common case.

    Examples
    --------
    Consider such redundant tree structure:

    .. code-block::

        DetectorHeader
        |-ClassInstances
            |-ClassInstance
            |-Type
            |-Window
            ...

    It can be sanitized/simplified by setting tags_to_flatten keyword
    with ["ClassInstances", "ClassInstance"] to eliminate redundant
    levels of tree with such tag names:

    .. code-block::

        DetectorHeader
        |-Type
        |-Window
        ...

    Produced dict/list structures are then good enough to be
    returned as part of original metadata without making any more
    copies.

    Setup the parser:

    >>> from rsciio.utils.xml import XmlToDict
    >>> xml_to_dict = XmlToDict(
    ...     pre_str_dub_attr="XmlClass",
    ...     tags_to_flatten=[
    ...         "ClassInstance", "ChildrenClassInstance", "JustAnotherRedundantTag"
    ...     ]
    ... )

    Use parser:

    >>> pytree = xml_to_dict.dictionarize(etree_node)

    """

    def __init__(
        self,
        dub_attr_pre_str="@",
        dub_text_str="#value",
        tags_to_flatten=None,
        interchild_text_parsing="first",
    ):
        if tags_to_flatten is None:
            tags_to_flatten = []
        if type(tags_to_flatten) not in [str, list]:
            raise ValueError(
                "tags_to_flatten keyword accepts string or list of strings"
            )
        if not isinstance(dub_attr_pre_str, str):
            raise ValueError("dub_attr_pre_str should be of string type")
        if not isinstance(dub_text_str, str):
            raise ValueError("dub_text_str should be of string type")
        if isinstance(tags_to_flatten, str):
            tags_to_flatten = [tags_to_flatten]
        if interchild_text_parsing not in ("skip", "first", "cat", "list"):
            raise ValueError(
                "interleaved_text_parsing should be set to the one of: "
                "('skip', 'first', 'cat', 'list')"
            )
        self.tags_to_flatten = tags_to_flatten
        self.dub_attr_pre_str = dub_attr_pre_str
        self.dub_text_str = dub_text_str
        self.poor_text_mode = interchild_text_parsing

    @staticmethod
    def eval(string):
        """
        Interpret any string and return casted to appropriate
        dtype python object.

        Parameters
        ----------
        string : str
            String to be interpreted.

        Returns
        -------
        string
            Interpreted string.

        Notes
        -----
        If this does not return desired type, consider subclassing
        and reimplementing this method like this:

        .. code-block:: python

            class SubclassedXmlToDict(XmlToDict):
                @staticmethod
                def eval(string):
                    if condition check to catch the case
                    ...
                    elif
                    ...
                    else:
                        return XmlToDict.eval(string)
        """
        try:
            return literal_eval(string)
        except (ValueError, SyntaxError):
            # SyntaxError due to:
            # literal_eval have problems with strings like this '8842_80'
            return string

    def dictionarize(self, et_node):
        """
        Take etree XML node and return its conversion into
        pythonic dict/list representation of that XML tree
        with some sanitization.

        Parameters
        ----------
        et_node : xml.etree.ElementTree.Element
            XML node to be converted.

        Returns
        -------
        dict
            Dictionary representation of the XML node.
        """
        d_node = {et_node.tag: {} if et_node.attrib else None}
        children = list(et_node)
        if children:
            dd_node = defaultdict(list)
            for dc_node in map(self.dictionarize, children):
                for key, val in dc_node.items():
                    dd_node[key].append(val)
            d_node = {
                et_node.tag: {
                    key: self.eval(val[0]) if len(val) == 1 else val
                    for key, val in dd_node.items()
                }
            }
        if et_node.attrib:
            d_node[et_node.tag].update(
                (self.dub_attr_pre_str + key if children else key, self.eval(val))
                for key, val in et_node.attrib.items()
            )
        if et_node.text:
            text = et_node.text.strip()
            if text:
                if not et_node.attrib and not children:
                    d_node[et_node.tag] = self.eval(text)
                elif children:
                    if self.poor_text_mode == "first":
                        d_node[et_node.tag][self.dub_text_str] = self.eval(text)
                    elif self.poor_text_mode in ("cat", "list"):
                        tails = [
                            str(c.tail if c.tail is not None else "").strip()
                            for c in children
                        ]
                        if any(tails):
                            inter_pieces = [text]
                            inter_pieces.extend(tails)
                            if self.poor_text_mode == "cat":
                                inter_pieces = "".join(inter_pieces)
                            d_node[et_node.tag]["#interchild_text"] = inter_pieces
                        else:
                            d_node[et_node.tag][self.dub_text_str] = self.eval(text)
                elif et_node.attrib:
                    d_node[et_node.tag][self.dub_text_str] = self.eval(text)
        for tag in self.tags_to_flatten:
            if tag in d_node:
                return d_node[tag]
        return d_node


def xml2dtb(et, dictree):
    """
    Convert XML ElementTree node to DTBox object.
    This is a recursive function that traverses the XML tree
    and populates the DTBox object with the data from the XML node.

    Parameters
    ----------
    et : xml.etree.ElementTree.Element
        XML node to be converted.
    dictree : DTBox
        Box object to be populated.

    """
    if et.text:
        dictree.set_item(et.tag, et.text)
        return
    else:
        dictree.add_node(et.tag)
        if et.attrib:
            dictree[et.tag].merge_update(et.attrib)
        for child in et:
            xml2dtb(child, dictree[et.tag])


def convert_xml_to_dict(xml_object):
    """
    Convert XML object to a DTBox object.

    Parameters
    ----------
    xml_object : str or xml.etree.ElementTree.Element
        XML object to be converted. It can be a string or an ElementTree node.

    Returns
    -------
    DTBox
        A DTBox object containing the converted XML data.
    """
    if isinstance(xml_object, str):
        xml_object = ET.fromstring(xml_object)
    op = DTBox(box_dots=True)
    xml2dtb(xml_object, op)
    return op
