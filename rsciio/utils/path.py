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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.
"""Utility functions for path handling."""

import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)


__all__ = [
    "append2pathname",
    "ensure_directory",
    "incremental_filename",
    "overwrite",
]


def __dir__():
    return sorted(__all__)


def append2pathname(filename, to_append):
    """
    Append a string to a path name.

    Parameters
    ----------
    filename : str
        The original file name.
    to_append : str
        The string to append to the file name.

    Returns
    -------
    pathlib.Path
        The new file name with the appended string.
    """
    p = Path(filename)
    return Path(p.parent, p.stem + to_append + p.suffix)


def incremental_filename(filename, i=1):
    """
    If a file with the same file name exists, returns a new filename that
    does not exists.

    The new file name is created by appending `-n` (where `n` is an integer)
    to path name

    Parameters
    ----------
    filename : str
        The original file name.
    i : int
       The number to be appended.

    Returns
    -------
    pathlib.Path
        The new file name with the appended number.
    """
    filename = Path(filename)

    if filename.is_file():
        new_filename = append2pathname(filename, f"-{i}")
        if new_filename.is_file():
            return incremental_filename(filename, i + 1)
        else:
            return new_filename
    else:
        return filename


def ensure_directory(path):
    """
    Check if the path exists and if it does not, creates the directory.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to check and create if it does not exist.
    """
    # If it's a file path, try the parent directory instead
    p = Path(path)
    p = p.parent if p.is_file() else p

    try:
        p.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        _logger.debug(f"Directory {p} already exists. Doing nothing.")


def overwrite(filename):
    """
    If file 'filename' exists, ask for overwriting and return True or False,
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
                print("Please answer y or n.")  # noqa: T201
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
