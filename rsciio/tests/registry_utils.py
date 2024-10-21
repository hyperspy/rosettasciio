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

import sys
import warnings
from pathlib import Path

import pooch

PATH = Path(__file__).parent


def update_registry():
    """
    Update the ``rsciio.tests.registry.txt`` file, which is required after
    adding or updating test data files.

    Unix system only. This is not supported on windows, because the hash
    comparison will fail for non-binary file, because of difference in line
    ending.
    """
    if sys.platform == "win32":
        warnings.warn("Updating registry is not supported on Windows. Nothing done.")
        return

    make_registry(
        PATH / "data",
        PATH / "registry.txt",
        recursive=True,
        exclude_pattern=[".git"],
    )


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
        List of pattern to exclude.

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


def download_all(pooch_object=None, ignore_hash=None, show_progressbar=True):
    """
    Download all test data if they are not already locally available in
    ``rsciio.tests.data`` folder.

    Parameters
    ----------
    pooch_object : pooch.Pooch or None, default=None
        The registry to be used. If None, a RosettaSciIO registry will
        be used.
    ignore_hash : bool or None, default=None
        Don't compare the hash of the downloaded file with the corresponding
        hash in the registry. On windows, the hash comparison will fail for
        non-binary file, because of difference in line ending. If None, the
        comparision will only be used on unix system.
    show_progressbar : bool, default=True
        Whether to show the progressbar or not.
    """

    from rsciio.tests.registry import TEST_DATA_REGISTRY

    if pooch_object is None:
        pooch_object = TEST_DATA_REGISTRY
    if ignore_hash is None:
        ignore_hash = sys.platform == "win32"
    if ignore_hash:
        for key in pooch_object.registry.keys():
            pooch_object.registry[key] = None

    if show_progressbar:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(pooch_object.registry_files))
        except ImportError:
            print("Using progresbar requires the `tqdm` library.")
            show_progressbar = False

    for i, file in enumerate(pooch_object.registry_files):
        pooch_object.fetch(file, progressbar=False)
        if show_progressbar:
            pbar.update(i)

    if show_progressbar:
        pbar.close()


if __name__ == "__main__":
    # Used by the pre-commit hook
    update_registry()
