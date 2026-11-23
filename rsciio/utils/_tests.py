# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

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


def assert_deep_almost_equal(actual, expected, *args, **kwargs):
    """Recursively check that two nested structures are (almost) equal.

    Dictionaries are compared key by key, and lists, tuples and arrays are
    compared item by item. Python numbers are compared with
    :func:`numpy.testing.assert_allclose`; any other values are compared
    with ``==``. On failure, the error message gives the location of the
    first mismatch within the structure.

    Parameters
    ----------
    actual : dict, list, tuple, numpy.ndarray or scalar
        The structure to check.
    expected : dict, list, tuple, numpy.ndarray or scalar
        The reference structure.
    *args, **kwargs :
        Passed to :func:`numpy.testing.assert_allclose`, for example
        ``rtol`` or ``atol``.
    """
    _compare_nested(actual, expected, "ROOT", args, kwargs)


def _compare_nested(actual, expected, location, args, kwargs):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError(
                f"Keys differ at {location}: {sorted(map(repr, set(actual)))} "
                f"!= {sorted(map(repr, set(expected)))}"
            )
        for key, value in expected.items():
            _compare_nested(actual[key], value, f"{location} -> {key!r}", args, kwargs)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        if len(actual) != len(expected):
            raise AssertionError(
                f"Lengths differ at {location}: {len(actual)} != {len(expected)}"
            )
        for i, (a, e) in enumerate(zip(actual, expected)):
            _compare_nested(a, e, f"{location} -> {i}", args, kwargs)
    elif isinstance(expected, (int, float, complex)):
        try:
            np.testing.assert_allclose(actual, expected, *args, **kwargs)
        except AssertionError as exc:
            raise AssertionError(f"Values differ at {location}:\n{exc}") from None
    elif actual != expected:
        raise AssertionError(f"Values differ at {location}: {actual!r} != {expected!r}")


def normalize_micro(obj):
    """Normalize the micro prefix glyph to Greek small letter mu (U+03BC).

    pint >= 0.26 renders the micro prefix with the Greek mu (U+03BC),
    following the Unicode standard recommendation, while older versions
    render the micro sign. The two glyphs are visually identical, so
    tests comparing pint-formatted units strings normalize both sides
    with this helper to stay independent of the pint version.
    """
    if isinstance(obj, str):
        return obj.replace("\u00b5", "\u03bc")
    if isinstance(obj, tuple):
        return tuple(normalize_micro(o) for o in obj)
    return obj
