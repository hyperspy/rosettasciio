# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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


import importlib
import logging
import os
import re
import xml.etree.ElementTree as ET
from ast import literal_eval
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from box import Box
from pint import UnitRegistry

_UREG = UnitRegistry()


_logger = logging.getLogger(__name__)


@contextmanager
def dummy_context_manager(*args, **kwargs):
    yield


# MSXML sanitization ###
# re pattern with two capturing groups with comma in between;
# firstgroup looks for numeric value after <tag> (the '>' char) with or
# without minus sign, second group looks for numeric value with following
# closing <\tag> (the '<' char); '([Ee]-?\d*)' part (optionally a third group)
# checks for scientific notation (e.g. 8,843E-7 -> 'E-7');
# compiled pattern is binary, as raw xml string is binary string.:
_fix_dec_patterns = re.compile(b"(>-?\\d+),(\\d*([Ee]-?\\d*)?<)")


def sanitize_msxml_float(xml_b_string):
    """replace comma with dot in floatng point numbers in given xml
    raw string.

    Parameters
    ----------
    xml_b_string: raw binary string representing the xml to be parsed

    Returns
    ---------
    binary string with commas used as decimal marks replaced with dots
    to adhere to XML standard

    What, why, how?
    ----------------------------
    In case OEM software runs on MS Windows and directly uses system
    built-in MSXML lib, which does not comply with XML standards,
    and OS is set to locale of some country with weird and illogical
    preferences of using comma as decimal separation;
    Software in conjunction of these above listed conditions can produce
    not-interoperable XML, which leads to wrong interpretation of context.
    This sanitizer searches and corrects that - it should be used before
    being fed to .fromsting of element tree.
    """
    return _fix_dec_patterns.sub(b"\\1.\\2", xml_b_string)


def dump_dictionary(
    file, dic, string="root", node_separator=".", value_separator=" = "
):
    for key in list(dic.keys()):
        if isinstance(dic[key], dict):
            dump_dictionary(file, dic[key], string + node_separator + key)
        else:
            file.write(
                string + node_separator + key + value_separator + str(dic[key]) + "\n"
            )


def append2pathname(filename, to_append):
    """Append a string to a path name

    Parameters
    ----------
    filename : str
    to_append : str

    """
    p = Path(filename)
    return Path(p.parent, p.stem + to_append, p.suffix)


def incremental_filename(filename, i=1):
    """If a file with the same file name exists, returns a new filename that
    does not exists.

    The new file name is created by appending `-n` (where `n` is an integer)
    to path name

    Parameters
    ----------
    filename : str
    i : int
       The number to be appended.
    """
    filename = Path(filename)

    if filename.is_file():
        new_filename = append2pathname(filename, "-{i}")
        if new_filename.is_file():
            return incremental_filename(filename, i + 1)
        else:
            return new_filename
    else:
        return filename


def ensure_directory(path):
    """Check if the path exists and if it does not, creates the directory."""
    # If it's a file path, try the parent directory instead
    p = Path(path)
    p = p.parent if p.is_file() else p

    try:
        p.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        _logger.debug(f"Directory {p} already exists. Doing nothing.")


def overwrite(filename):
    """If file 'filename' exists, ask for overwriting and return True or False,
    else return True.

    Parameters
    ----------
    filename : str or pathlib.Path
        File to check for overwriting.

    Returns
    -------
    bool :
        Whether to overwrite file.

    """
    if Path(filename).is_file() or (
        Path(filename).is_dir() and os.path.splitext(filename)[1] == ".zspy"
    ):
        message = f"Overwrite '{filename}' (y/n)?\n"
        try:
            answer = input(message)
            answer = answer.lower()
            while (answer != "y") and (answer != "n"):
                print("Please answer y or n.")
                answer = input(message)
            if answer.lower() == "y":
                return True
            elif answer.lower() == "n":
                return False
            else:
                return True
        except Exception:
            # We are running in the IPython notebook that does not
            # support raw_input
            _logger.info(
                "Your terminal does not support raw input. "
                "Not overwriting. "
                "To overwrite the file use `overwrite=True`"
            )
            return False
    else:
        return True


class XmlToDict:
    """Customisable XML to python dict and list based Hierarchical tree
    translator.
    """

    def __init__(
        self,
        dub_attr_pre_str="@",
        dub_text_str="#value",
        tags_to_flatten=None,
        interchild_text_parsing="first",
    ):
        """
        Create translator for hierarchical XML etree node into
        dict/list conversion.

        Parameters
        ----------
        dub_attr_pre_str: string (default: "@"), which
            is going to be prepend to attribute name when creating
            dictionary tree if children element with same name is used
        dub_text_str: string (default: "#text"), which is going to be
            used for key in case element contains text and children tag
        interchild_text_parsing: string (default: "first") one from
            ("skip", "first", "cat", "list"). This considers the
            behaviour when both .text and children tags are presented
            under same element tree node:
            "skip" - will not try to retrieve any .text values from such node.
            "first" - only string under .text attribute will be returned
            "cat" - return concatenated string from .text of node and .tail's
            of children nodes.
            "list" - similar to "cat", but return the result in list
            without concatenation.
        tags_to_flatten: (default: None) None, string or list of strings
            with tag names which should be flattened/skipped,
            placing children of such tag one level shallower in constructed
            python structure.
            It is useful when OEM generated XML are not human designed,
            but machine/programming language/framework generated
            and painfully verboise. See example below:

        Examples
        --------
        Consider such redundant tree structure:

        DetectorHeader
        |-ClassInstances
            |-ClassInstance
            |-Type
            |-Window
            ...

        it can be sanitized/simplified by setting tags_to_flatten keyword
        with ["ClassInstances", "ClassInstance"] to eliminate redundant
        levels of tree with such tag names:

        DetectorHeader
        |-Type
        |-Window
        ...

        Produced dict/list structures are then good enought to be
        returned as part of original metadata without making any more
        copies.

        Usage
        -----
        in target format parser:

        from rsciio.utils.tools import XmlToDict

        #setup the parser:
        xml_to_dict = XmlToDict(pre_str_dub_attr="XmlClass",
                                tags_to_flatten=["ClassInstance",
                                                 "ChildrenClassInstance",
                                                 "JustAnotherRedundantTag"])
        # use parser:
        pytree = xml_to_dict.dictionarize(etree_node)
        """
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
        """interpret any string and return casted to appropriate
        dtype python object.
        If this does not return desired type, consider subclassing
        and reimplementing this method like this:

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
        """take etree XML node and return its conversion into
        pythonic dict/list representation of that XML tree
        with some sanitization"""
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
    if et.text:
        dictree.set_item(et.tag, et.text)
        return
    else:
        dictree.add_node(et.tag)
        if et.attrib:
            dictree[et.tag].merge_update(et.attrib)
        for child in et:
            xml2dtb(child, dictree[et.tag])


class DTBox(Box):
    """
    Subclass of Box to help migration from hyperspy `DictionaryTreeBrowser`
    to `Box` when splitting IO code from hyperspy to rosettasciio.

    When using `box_dots=True`, by default, period will be removed from keys.
    To support period containing keys, use `box_dots=False, default_box=True`.
    https://github.com/cdgriffith/Box/wiki/Types-of-Boxes#default-box
    """

    def add_node(self, path):
        keys = path.split(".")
        for key in keys:
            if self.get(key) is None:
                self[key] = {}
            self = self[key]

    def set_item(self, path, value):
        if self.get(path) is None:
            self.add_node(path)
        self[path] = value

    def has_item(self, path):
        return self.get(path) is not None


def convert_xml_to_dict(xml_object):
    if isinstance(xml_object, str):
        xml_object = ET.fromstring(xml_object)
    op = DTBox(box_dots=True)
    xml2dtb(xml_object, op)
    return op


def sarray2dict(sarray, dictionary=None):
    """Converts a struct array to an ordered dictionary

    Parameters
    ----------
    sarray: struct array
    dictionary: None or dict
        If dictionary is not None the content of sarray will be appended to the
        given dictonary

    Returns
    -------
    Ordered dictionary

    """
    if dictionary is None:
        dictionary = OrderedDict()
    for name in sarray.dtype.names:
        dictionary[name] = sarray[name][0] if len(sarray[name]) == 1 else sarray[name]
    return dictionary


def dict2sarray(dictionary, sarray=None, dtype=None):
    """Populates a struct array from a dictionary

    Parameters
    ----------
    dictionary: dict
    sarray: struct array or None
        Either sarray or dtype must be given. If sarray is given, it is
        populated from the dictionary.
    dtype: None, numpy dtype or dtype list
        If sarray is None, dtype must be given. If so, a new struct array
        is created according to the dtype, which is then populated.

    Returns
    -------
    Structure array

    """
    if sarray is None:
        if dtype is None:
            raise ValueError("Either sarray or dtype need to be specified.")
        sarray = np.zeros((1,), dtype=dtype)
    for name in set(sarray.dtype.names).intersection(set(dictionary.keys())):
        if len(sarray[name]) == 1:
            sarray[name][0] = dictionary[name]
        else:
            sarray[name] = dictionary[name]
    return sarray


def convert_units(value, units, to_units):
    return (value * _UREG(units)).to(to_units).magnitude


def get_object_package_info(obj):
    """Get info about object package

    Returns
    -------
    dic: dict
        Dictionary containing ``package`` and ``package_version`` (if available)
    """
    dic = {}
    # Note that the following can be "__main__" if the component was user
    # defined
    dic["package"] = obj.__module__.split(".")[0]
    if dic["package"] != "__main__":
        try:
            dic["package_version"] = importlib.import_module(dic["package"]).__version__
        except AttributeError:
            dic["package_version"] = ""
            _logger.warning(
                "The package {package} does not set its version in "
                + "{package}.__version__. Please report this issue to the "
                + "{package} developers.".format(package=dic["package"])
            )
    else:
        dic["package_version"] = ""
    return dic


def ensure_unicode(stuff, encoding="utf8", encoding2="latin-1"):
    if not isinstance(stuff, (bytes, np.bytes_)):
        return stuff
    else:
        string = stuff
    try:
        string = string.decode(encoding)
    except Exception:
        string = string.decode(encoding2, errors="ignore")
    return string


def get_file_handle(data, warn=True):
    """Return file handle of a dask array when possible; currently only hdf5 file are
    supported.
    """
    arrkey = None
    for key in data.dask.keys():
        # The if statement with both "array-original" and "original-array"
        # is due to dask changing the name of this key. After dask-2022.1.1
        # the key is "original-array", before it is "array-original"
        if ("array-original" in key) or ("original-array" in key):
            arrkey = key
            break
    if arrkey:
        try:
            return data.dask[arrkey].file
        except (AttributeError, ValueError):
            if warn:
                _logger.warning(
                    "Failed to retrieve file handle, either "
                    "the file is already closed or it is not "
                    "an hdf5 file."
                )
    return None


def jit_ifnumba(*decorator_args, **decorator_kwargs):
    try:
        import numba

        decorator_kwargs.setdefault("nopython", True)
        return numba.jit(*decorator_args, **decorator_kwargs)
    except ImportError:
        _logger.warning(
            "Falling back to slow pure python code, because `numba` is not installed."
        )

        def wrap(func):
            def wrapper_func(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper_func

        return wrap
