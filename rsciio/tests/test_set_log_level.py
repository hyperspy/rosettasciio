# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

import pytest

import rsciio


@pytest.mark.parametrize(
    "level", ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET")
)
def test_set_log_level(level):
    rsciio.set_log_level(level)
