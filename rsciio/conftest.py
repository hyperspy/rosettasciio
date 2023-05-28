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

import os
from packaging.version import Version

from rsciio.tests.registry_utils import download_all

try:
    import hyperspy

    if Version(hyperspy.__version__) < Version("2.0.dev"):
        raise Exception(
            "To run the test suite using hyperspy, \
            hyperspy 2.0 or higher is required."
        )
except ImportError:
    pass


def pytest_configure(config):
    # Run in pytest_configure hook to avoid capturing stdout by pytest and
    # inform user that the test data are being downloaded

    # Workaround to avoid running it for each worker
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is None:
        print("Checking if test data need downloading...")
        download_all()
        print("All test data available.")
