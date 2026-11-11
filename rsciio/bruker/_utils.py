# -*- coding: utf-8 -*-
#
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

import xml.etree.ElementTree as ET

from rsciio._docstrings import FILENAME_DOC
from rsciio.utils.xml import sanitize_msxml_float

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
