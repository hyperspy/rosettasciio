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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with RosettaSciIO. If not, see <https://www.gnu.org/licenses/#GPL>.

import logging
import yaml
import os

from rsciio._version import __version__


IO_PLUGINS = []
_logger = logging.getLogger(__name__)

for sub, _, _ in os.walk(os.path.abspath(os.path.dirname(__file__))):
    _specsf = os.path.join(sub, "specifications.yaml")
    if os.path.isfile(_specsf):
        with open(_specsf, "r") as stream:
            _specs = yaml.safe_load(stream)
            # for testing purposes
            _specs["api"] = "rsciio.%s" % os.path.split(sub)[1]
            IO_PLUGINS.append(_specs)

__all__ = [
    "__version__",
    "IO_PLUGINS",
]


def __dir__():
    return sorted(__all__)


if "dev" in __version__:
    # Copied from https://github.com/scikit-image/scikit-image
    # Append last commit date and hash to dev version information, if available

    import subprocess
    import os.path

    try:
        p = subprocess.Popen(
            ["git", "log", "-1", '--format="%h %aI"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(__file__),
        )
    except FileNotFoundError:
        pass
    else:
        out, err = p.communicate()
        if p.returncode == 0:
            git_hash, git_date = (
                out.decode("utf-8")
                .strip()
                .replace('"', "")
                .split("T")[0]
                .replace("-", "")
                .split()
            )

            __version__ = "+".join(
                [tag for tag in __version__.split("+") if not tag.startswith("git")]
            )
            __version__ += f"+git{git_date}.{git_hash}"
