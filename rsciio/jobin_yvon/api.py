# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy. If not, see <https://www.gnu.org/licenses/#GPL>.

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from copy import deepcopy

_logger = logging.getLogger(__name__)


class JobinYvonXMLReader:
    def __init__(self, file_path, use_uniform_signal_axis=False):
        self._file_path = file_path
    def parse_file(self):
        """First parse through file to extract data/metadata positions."""
        tree = ET.parse(self._file_path)
        root = tree.getroot()

        lsx_tree_list = root.findall("LSX_Tree")
        if len(lsx_tree_list) > 1:
            _logger.critical(
                "File contains multiple positions to read metadata from.\n"
                "The first location is choosen."
            )  # pragma: no cover
        elif len(lsx_tree_list) == 0:
            _logger.critical("No metadata found.")  # pragma: no cover
        lsx_tree = lsx_tree_list[0]

        lsx_matrix_list = root.findall("LSX_Matrix")
        if len(lsx_matrix_list) > 1:
            _logger.critical(
                "File contains multiple positions to read data from.\n"
                "The first location is choosen."
            )  # pragma: no cover
        elif len(lsx_matrix_list) == 0:
            _logger.critical("No data found.")  # pragma: no cover
        self._lsx_matrix = lsx_matrix_list[0]

        for child in lsx_tree:
            id = self._get_id(child)
            if id == "0x6C62D4D9":
                self._metadata_root = child
            if id == "0x6C7469D9":
                self._title = child.text
            if id == "0x6D707974":
                self._measurement_type = child.text
            if id == "0x6C676EC6":
                for child2 in child:
                    if self._get_id(child2) == "0x0":
                        self._angle = child2.text
            if id == "0x7A74D9D6":
                for child2 in child:
                    if self._get_id(child2) == "0x7B697861":
                        self._axis_root = child2

        if not hasattr(self, "_metadata_root"):
            _logger.critical("Could not extract metadata")  # pragma: no cover
        if not hasattr(self, "_axis_root"):
            _logger.critical("Could not extract axis")  # pragma: no cover
