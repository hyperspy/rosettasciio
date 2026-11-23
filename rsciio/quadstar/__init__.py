# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

from ._api import file_reader

__all__ = [
    "file_reader",
]


def __dir__():
    return sorted(__all__)
