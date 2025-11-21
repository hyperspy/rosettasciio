# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
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
