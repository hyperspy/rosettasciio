# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

from importlib.metadata import version
from pathlib import Path

__version__ = version("rosettasciio")

# For development version, `setuptools_scm` will be used at build time
# to get the dev version, in case of missing vcs information (git archive,
# shallow repository), the fallback version defined in pyproject.toml will
# be used

# if we have a editable install from a git repository try to use
# `setuptools_scm` to find a more accurate version:
# `importlib.metadata` will provide the version at installation
# time and for editable version this may be different

# we only do that if we have enough git history, e.g. not shallow checkout
_root = Path(__file__).resolve().parents[1]
if (_root / ".git").exists() and not (_root / ".git/shallow").exists():
    try:
        # setuptools_scm may not be installed
        from setuptools_scm import get_version

        __version__ = get_version(_root)
    except ImportError:  # pragma: no cover
        # setuptools_scm not install, we keep the existing __version__
        pass
