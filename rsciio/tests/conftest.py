# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

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
            print("Checking if test data need downloading...")  # noqa: T201
            download_all()
            print("All test data available.")  # noqa: T201

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
