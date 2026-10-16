# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

import logging

_logger = logging.getLogger(__name__)


def jit_ifnumba(*decorator_args, **decorator_kwargs):
    try:
        import numba

        decorator_kwargs.setdefault("nopython", True)
        return numba.jit(*decorator_args, **decorator_kwargs)
    except ImportError:
        _logger.warning(
            "Falling back to slow pure python code, because `numba` is not installed."
        )

        def wrap(func):
            def wrapper_func(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper_func

        return wrap
