# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.


from pint import UnitRegistry

_UREG = UnitRegistry()


def convert_units(value, units, to_units):
    return (value * _UREG(units)).to(to_units).magnitude
