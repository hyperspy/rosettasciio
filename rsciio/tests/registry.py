# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

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
