# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.


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
