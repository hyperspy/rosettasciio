# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
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


from contextlib import contextmanager


@contextmanager
def dummy_context_manager(*args, **kwargs):
    yield


def get_progress_bar_context_manager(show_progressbar):
    """
    Wrapper to `dask.diagnostics.ProgressBar` context manager that
    defer dask import and add argument to enable/disable the progress bar.

    Parameters
    ----------
    show_progressbar : bool
        Whether to show the progress bar.

    Returns
    -------
    context manager
        A context manager that shows a progress bar if requested,
        otherwise it will return a dummy context manager.
    """
    if show_progressbar:
        from dask.diagnostics import ProgressBar

        return ProgressBar
    else:
        return dummy_context_manager
