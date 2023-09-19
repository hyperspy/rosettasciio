# -*- coding: utf-8 -*-
#
# Copyright 2022 The HyperSpy developers
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

FILENAME_DOC = """filename : str, pathlib.Path
        Filename of the file to read or corresponding `pathlib.Path`.
    """

SIGNAL_DOC = """signal : dict
        Dictionary containing the signal object.
        Should contain the following fields:

        - 'data' – multidimensional numpy array
        - 'axes' – list of dictionaries describing the axes
          containing the fields 'name', 'units', 'index_in_array', and
          either 'size', 'offset', and 'scale' or a numpy array 'axis'
          containing the full axes vector
        - 'metadata' – dictionary containing the metadata tree
    """

LAZY_DOC = """lazy : bool, Default=False
        Whether to open the file lazily or not.
    """

ENCODING_DOC = """encoding : str, Default="latin-1"
        The encoding used to read the content of the file. Different file
        encodings, such as ``"utf8"`` can be set via this argument.
    """

ENDIANESS_DOC = """endianess : str, Default="<"
        ``"<"`` or ``">"``, depending on how the bits are written to 
        the file.
    """

MMAP_DOC = """mmap_mode : {None, "r+", "r", "w+", "c"}, Default=None
        Argument passed to :py:func:`numpy.memmap`. If None (default), the
        value is ``"r"`` when ``lazy=True``, otherwise it is ``"c"``.
    """


RETURNS_DOC = """Returns
    -------

    list of dicts
        List of dictionaries containing the following fields:

        - 'data' – multidimensional numpy array
        - 'axes' – list of dictionaries describing the axes
          containing the fields 'name', 'units', 'index_in_array', and
          either 'size', 'offset', and 'scale' or a numpy array 'axis'
          containing the full axes vector
        - 'metadata' – dictionary containing the parsed metadata
        - 'original_metadata' – dictionary containing the full metadata tree from the input file
    """
