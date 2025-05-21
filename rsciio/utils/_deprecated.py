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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.


import functools
import inspect
import warnings
from typing import Callable, Optional, Union

from rsciio.utils.exceptions import VisibleDeprecationWarning


class deprecated:
    """Decorator to mark deprecated functions with an informative
    warning.
    Adapted from
    `scikit-image
    <https://github.com/scikit-image/scikit-image/blob/main/skimage/_shared/utils.py>`_
    and `matplotlib
    <https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/_api/deprecation.py>`_.
    """

    def __init__(
        self,
        since: Union[str, int, float],
        alternative: Optional[str] = None,
        alternative_is_function: bool = True,
        removal: Union[str, int, float, None] = None,
    ):
        """
        Visible deprecation warning.

        Parameters
        ----------
        since
            The release at which this API became deprecated.
        alternative
            An alternative API that the user may use in place of the
            deprecated API.
        alternative_is_function
            Whether the alternative is a function. Default is ``True``.
        removal
            The expected removal version.
        """
        self.since = since
        self.alternative = alternative
        self.alternative_is_function = alternative_is_function
        self.removal = removal

    def __call__(self, func: Callable):
        # Wrap function to raise warning when called, and add warning to
        # docstring
        if self.alternative is not None:
            if self.alternative_is_function:
                alt_msg = f" Use `{self.alternative}()` instead."
            else:
                alt_msg = f" Use `{self.alternative}` instead."
        else:
            alt_msg = ""
        if self.removal is not None:
            rm_msg = f" and will be removed in version {self.removal}"
        else:
            rm_msg = ""
        msg = f"Function `{func.__name__}()` is deprecated{rm_msg}.{alt_msg}"

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.simplefilter(
                action="always",
                category=VisibleDeprecationWarning,
                append=True,
            )
            func_code = func.__code__
            warnings.warn_explicit(
                message=msg,
                category=VisibleDeprecationWarning,
                filename=func_code.co_filename,
                lineno=func_code.co_firstlineno + 1,
            )
            return func(*args, **kwargs)

        # Modify docstring to display deprecation warning
        old_doc = inspect.cleandoc(func.__doc__ or "").strip("\n")
        notes_header = "\nNotes\n-----"
        new_doc = (
            f"[*Deprecated*] {old_doc}\n"
            f"{notes_header if notes_header not in old_doc else ''}\n"
            f".. deprecated:: {self.since}\n"
            f"   {msg.strip()}"  # Matplotlib uses three spaces
        )
        wrapped.__doc__ = new_doc

        return wrapped


class deprecated_argument:
    """
    Decorator to remove an argument from a function or method's signature.
    Adapted from `scikit-image
    <https://github.com/scikit-image/scikit-image/blob/main/skimage/_shared/utils.py>`_.
    """

    def __init__(
        self, name, since, removal=None, alternative=None, additional_msg=None
    ):
        self.name = name
        self.since = since
        self.removal = removal
        self.alternative = alternative
        self.additional_msg = additional_msg

    def __call__(self, func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if self.name in kwargs.keys():
                msg = f"Argument `{self.name}` is deprecated"
                if self.removal is not None:
                    msg += f" and will be removed in version {self.removal}."
                else:
                    msg += " and has been removed. "
                    kwargs.pop(self.name)
                msg += f" To avoid this warning, please do not use `{self.name}`. "
                if self.alternative is not None:
                    msg += f"Use `{self.alternative}` instead. "
                    kwargs[self.alternative] = kwargs.pop(self.name)
                msg += f"See the documentation of `{func.__name__}()` for more details."
                if self.additional_msg is not None:
                    msg += f" {self.additional_msg}"
                warnings.simplefilter(
                    action="always", category=VisibleDeprecationWarning
                )
                func_code = func.__code__
                warnings.warn_explicit(
                    message=msg,
                    category=VisibleDeprecationWarning,
                    filename=func_code.co_filename,
                    lineno=func_code.co_firstlineno + 1,
                )
            return func(*args, **kwargs)

        return wrapped


def distributed_keyword_deprecation(func):
    return deprecated_argument(
        "distributed",
        "0.8.0",
        additional_msg=" Distributed memory mapping is now supported in the default implementation.",
    )(func)


def mmap_mode_keyword_deprecation(func):
    return deprecated_argument(
        "mmap_mode",
        "0.8.0",
        additional_msg=" `mmap_mode` is deprecated because it is not used anymore.",
    )(func)
