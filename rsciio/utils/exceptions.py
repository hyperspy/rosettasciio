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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.


class MountainsMapFileError(Exception):
    def __init__(self, msg="Corrupt Mountainsmap file"):
        self.error = msg

    def __str__(self):
        return repr(self.error)


class ByteOrderError(Exception):
    def __init__(self, order=""):
        self.byte_order = order

    def __str__(self):
        return repr(self.byte_order)


class DM3FileVersionError(Exception):
    def __init__(self, value=""):
        self.dm3_version = value

    def __str__(self):
        return repr(self.dm3_version)


class DM3TagError(Exception):
    def __init__(self, value=""):
        self.dm3_tag = value

    def __str__(self):
        return repr(self.dm3_tag)


class DM3DataTypeError(Exception):
    def __init__(self, value=""):
        self.dm3_dtype = value

    def __str__(self):
        return repr(self.dm3_dtype)


class DM3TagTypeError(Exception):
    def __init__(self, value=""):
        self.dm3_tagtype = value

    def __str__(self):
        return repr(self.dm3_tagtype)


class DM3TagIDError(Exception):
    def __init__(self, value=""):
        self.dm3_tagID = value

    def __str__(self):
        return repr(self.dm3_tagID)


class VisibleDeprecationWarning(UserWarning):
    """Visible deprecation warning.
    By default, python will not show deprecation warnings, so this class
    provides a visible one.

    """

    pass


class LazyCupyConversion(Exception):
    def __init__(self):
        self.error = (
            "Automatically converting data to cupy array is not supported "
            "for lazy signals. Read the corresponding section in the user "
            "guide for more information on how to use GPU with lazy signals."
        )

    def __str__(self):
        return repr(self.error)
