# -*- coding: utf-8 -*-
#
# Copyright 2007-2024 The HyperSpy developers
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with any project and source this library is coupled.
# If not, see <https://www.gnu.org/licenses/#GPL>.

import xml.etree.ElementTree as ET

from rsciio._docstrings import FILENAME_DOC
from rsciio.utils.tools import sanitize_msxml_float

from ._api import SFS_reader


def export_metadata(filename, output_filename=None):
    """
    Export the metadata from a ``.bcf`` file to an ``.xml`` file.

    Parameters
    ----------
    %s
    output_filename : str, pathlib.Path or None
        The filename of the exported ``.xml`` file.
        If ``None``, use "header.xml" as default.

    """
    sfs = SFS_reader(filename)
    # all file items in this singlefilesystem class instance is held inside
    # dictionary hierarchy, we fetch the header:
    header = sfs.vfs["EDSDatabase"]["HeaderData"]
    xml_str = sanitize_msxml_float(header.get_as_BytesIO_string().getvalue())
    xml = ET.ElementTree(ET.fromstring(xml_str))

    if output_filename is None:  # pragma: no cover
        output_filename = "header.xml"
    xml.write(output_filename)


export_metadata.__doc__ %= FILENAME_DOC
