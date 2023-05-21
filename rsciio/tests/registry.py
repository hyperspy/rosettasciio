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

from packaging.version import Version
from pathlib import Path

import pooch

import rsciio

version = rsciio.__version__

if not Version(version).release:
    version = "main"

TESTS_PATH = Path(__file__).parent


# We don't use the version functionality of pooch because we want to use the
# local test folder (rsciio.tests.data)
POOCH = pooch.create(
    path=TESTS_PATH / "data",
    # base_url=f"https://github.com/hyperspy/rosettasciio/raw/v{version}/rsciio/tests/data/",
    base_url="https://github.com/ericpre/rosettasciio/raw/pooch/rsciio/tests/data/",
    version=None,
    # We'll load it from a file later
    registry=None,
    allow_updates=False,
    retry_if_failed=3,
)

POOCH.load_registry(Path(__file__).parent / "registry.txt")
