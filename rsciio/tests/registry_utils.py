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

from pathlib import Path
import pooch

from rsciio.tests.registry import POOCH

PATH = Path(__file__).parent


def update_registry():
    make_registry(
        PATH / "data",
        PATH / "registry.txt",
        recursive=True,
        exclude_pattern=[".git"],
    )


# TODO: see https://www.fatiando.org/pooch/v1.3.0/advanced.html to fix hash issue


def make_registry(directory, output, recursive=True, exclude_pattern=None):
    """
    Make a registry of files and hashes for the given directory.

    This is helpful if you have many files in your test dataset as it keeps you
    from needing to manually update the registry.

    Parameters
    ----------
    directory : str
        Directory of the test data to put in the registry. All file names in
        the registry will be relative to this directory.
    output : str
        Name of the output registry file.
    recursive : bool
        If True, will recursively look for files in subdirectories of
        *directory*.
    exclude_pattern : list or None
        List of pattern to exclude

    Notes
    -----
    Adapted from https://github.com/fatiando/pooch/blob/main/pooch/hashes.py
    BSD-3-Clause

    """
    directory = Path(directory)
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    if exclude_pattern is None:
        exclude_pattern = []

    files = set(
        str(path.relative_to(directory))
        for path in directory.glob(pattern)
        if path.is_file()
    )

    files_to_exclude = set(
        file for file in files if any(pattern in file for pattern in exclude_pattern)
    )

    files = sorted(files - files_to_exclude)

    hashes = [pooch.file_hash(str(directory / fname)) for fname in files]

    with open(output, "w") as outfile:
        for fname, fhash in zip(files, hashes):
            # Only use Unix separators for the registry so that we don't go
            # insane dealing with file paths.
            outfile.write("'{}' {}\n".format(fname.replace("\\", "/"), fhash))


def download_all(pooch_object=None, ignore_hash=False, progressbar=True):
    if pooch_object is None:
        pooch_object = POOCH
    if ignore_hash:
        for key in pooch_object.registry.keys():
            pooch_object.registry[key] = None
    for file in pooch_object.registry_files:
        pooch_object.fetch(file, progressbar=progressbar)
