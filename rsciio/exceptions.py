# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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


__all__ = [
    "MountainsMapFileError",
    "ByteOrderError",
    "DM3FileVersionError",
    "DM3TagError",
    "DM3DataTypeError",
    "DM3TagTypeError",
    "DM3TagIDError",
    "VisibleDeprecationWarning",
    "LazyCupyConversion",
]


def __dir__():
    return sorted(__all__)


class MountainsMapFileError(Exception):
    """
    Raised when opening a MountainsMap file fails.

    Parameters
    ----------
    msg : str, optional
        The error message to display. Defaults to "Corrupt Mountainsmap file".
    """

    def __init__(self, msg="Corrupt Mountainsmap file"):
        self.error = msg

    def __str__(self):
        return repr(self.error)


class ByteOrderError(Exception):
    """
    Raised when the byte order is not recognized.

    Parameters
    ----------
    order : str, optional
        The byte order that was expected. Defaults to an empty string.
    """

    def __init__(self, order=""):
        self.byte_order = order

    def __str__(self):
        return repr(self.byte_order)


class DM3FileVersionError(Exception):
    """
    Raised when the DM3 file version is not recognized.

    Parameters
    ----------
    value : str, optional
        The DM3 file version that was not recognized. Defaults to an empty string.
    """

    def __init__(self, value=""):
        self.dm3_version = value

    def __str__(self):
        return repr(self.dm3_version)


class DM3TagError(Exception):
    """
    Raised when a DM3 tag is not recognized.

    Parameters
    ----------
    value : str, optional
        The DM3 tag that was not recognized. Defaults to an empty string.
    """

    def __init__(self, value=""):
        self.dm3_tag = value

    def __str__(self):
        return repr(self.dm3_tag)


class DM3DataTypeError(Exception):
    """
    Raised when a DM3 data type is not recognized.

    Parameters
    ----------
    value : str, optional
        The DM3 data type that was not recognized. Defaults to an empty string.
    """

    def __init__(self, value=""):
        self.dm3_dtype = value

    def __str__(self):
        return repr(self.dm3_dtype)


class DM3TagTypeError(Exception):
    """
    Raised when a DM3 tag type is not recognized.

    Parameters
    ----------
    value : str, optional
        The DM3 tag type that was not recognized. Defaults to an empty string.
    """

    def __init__(self, value=""):
        self.dm3_tagtype = value

    def __str__(self):
        return repr(self.dm3_tagtype)


class DM3TagIDError(Exception):
    """
    Raised when a DM3 tag ID is not recognized.

    Parameters
    ----------
    value : str, optional
        The DM3 tag ID that was not recognized. Defaults to an empty string.
    """

    def __init__(self, value=""):
        self.dm3_tagID = value

    def __str__(self):
        return repr(self.dm3_tagID)


class VisibleDeprecationWarning(UserWarning):
    """
    Visible deprecation warning.
    By default, python will not show deprecation warnings, so this class
    provides a visible one.
    """

    pass


class LazyCupyConversion(Exception):
    """Raised when trying to convert a lazy signal to cupy array."""

    def __init__(self):
        self.error = (
            "Automatically converting data to cupy array is not supported "
            "for lazy signals. Read the corresponding section in the user "
            "guide for more information on how to use GPU with lazy signals."
        )

    def __str__(self):
        return repr(self.error)
