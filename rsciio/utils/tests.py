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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

import importlib

import numpy as np


def expected_is_binned():
    """
    Convenient function to infer binning. When exspy is installed
    some signal will be assigned to EDS or EELS class instead of
    Signal1D class and the binned attribute will change accordingly.
    """
    if importlib.util.find_spec("exspy") is None:
        binned = False
    else:
        binned = True

    return binned


# Adapted from:
# https://github.com/gem/oq-engine/blob/master/openquake/server/tests/helpers.py
def assert_deep_almost_equal(actual, expected, *args, **kwargs):
    """Assert that two complex structures have almost equal contents.
    Compares lists, dicts and tuples recursively. Checks numeric values
    using :func:`numpy.testing.assert_allclose` and
    checks all other values with :func:`numpy.testing.assert_equal`.
    Accepts additional positional and keyword arguments and pass those
    intact to assert_allclose() (that's how you specify comparison
    precision).

    Parameters
    ----------
    actual: list, dict or tuple
        Actual values to compare.
    expected: list, dict or tuple
        Expected values.
    *args :
        Arguments are passed to :func:`numpy.testing.assert_allclose` or
        :func:`assert_deep_almost_equal`.
    **kwargs :
        Keyword arguments are passed to
        :func:`numpy.testing.assert_allclose` or
        :func:`assert_deep_almost_equal`.
    """
    is_root = "__trace" not in kwargs
    trace = kwargs.pop("__trace", "ROOT")
    try:
        if isinstance(expected, (int, float, complex)):
            np.testing.assert_allclose(expected, actual, *args, **kwargs)
        elif isinstance(expected, (list, tuple, np.ndarray)):
            assert len(expected) == len(actual)
            for index in range(len(expected)):
                v1, v2 = expected[index], actual[index]
                assert_deep_almost_equal(v1, v2, __trace=repr(index), *args, **kwargs)
        elif isinstance(expected, dict):
            assert set(expected) == set(actual)
            for key in expected:
                assert_deep_almost_equal(
                    expected[key], actual[key], __trace=repr(key), *args, **kwargs
                )
        else:
            assert expected == actual
    except AssertionError as exc:
        exc.__dict__.setdefault("traces", []).append(trace)
        if is_root:
            trace = " -> ".join(reversed(exc.traces))
            exc = AssertionError("%s\nTRACE: %s" % (exc, trace))
        raise exc
