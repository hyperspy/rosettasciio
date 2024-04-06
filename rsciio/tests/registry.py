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
from pathlib import Path

import pooch
from packaging.version import Version

import rsciio

version = rsciio.__version__


if Version(version).is_devrelease:
    version = "main"
else:
    version = f"v{version}"


TESTS_PATH = Path(__file__).parent


# This environment variable can be used to specify a base url other than the one
# from the hyperspy/rosettasciio repository.
# This is used in workflow when the test suite is run from the packages (test files
# not included), such as the "package and test" and "release" workflow on GitHub,
# other workflows use local files (available from the git repository)
BASE_URL = os.environ.get(
    "POOCH_BASE_URL",
    f"https://github.com/hyperspy/rosettasciio/raw/{version}/rsciio/tests/data/",
)


TEST_DATA_REGISTRY = pooch.create(
    path=TESTS_PATH / "data",
    base_url=BASE_URL,
    # We don't use the version functionality of pooch because we want to use the
    # local test folder (rsciio.tests.data)
    version=None,
    # We'll load it from a file later
    registry=None,
    allow_updates=False,
    retry_if_failed=3,
)

TEST_DATA_REGISTRY.load_registry(TESTS_PATH / "registry.txt")
