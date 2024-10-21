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

import json

import pytest
from filelock import FileLock
from packaging.version import Version

try:
    import hyperspy

    if Version(hyperspy.__version__) < Version("2.0.dev"):
        raise Exception(
            "To run the test suite using hyperspy, \
            hyperspy 2.0 or higher is required."
        )
except ImportError:
    pass


# From https://pytest-xdist.readthedocs.io/en/latest/how-to.html#making-session-scoped-fixtures-execute-only-once
@pytest.fixture(scope="session", autouse=True)
def session_data(request, tmp_path_factory, worker_id):
    capmanager = request.config.pluginmanager.getplugin("capturemanager")

    def _download_test_data():
        from rsciio.tests.registry_utils import download_all

        with capmanager.global_and_fixture_disabled():
            print("Checking if test data need downloading...")
            download_all()
            print("All test data available.")

        return "Test data available"

    if worker_id == "master":
        # not executing in with multiple workers, just produce the data and let
        # pytest's fixture caching do its job
        return _download_test_data()

    # get the temp directory shared by all workers
    root_tmp_dir = tmp_path_factory.getbasetemp().parent

    fn = root_tmp_dir / "data.json"
    with FileLock(str(fn) + ".lock"):
        if fn.is_file():
            data = json.loads(fn.read_text())
        else:
            data = _download_test_data()
            fn.write_text(json.dumps(data))
    return data
