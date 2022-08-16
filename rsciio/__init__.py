# -*- coding: utf-8 -*-
# Copyright 2007-2022 The HyperSpy developers
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

from rsciio.version import __version__

IO_PLUGINS = []
_logger = logging.getLogger(__name__)

here = os.path.abspath(os.path.dirname(__file__))

for sub, _, _ in os.walk(here):
    specsf = os.path.join(sub, "specifications.yaml")
    if os.path.isfile(specsf):
        with open(specsf, "r") as stream:
            specs = yaml.safe_load(stream)
            specs["api"] = "rsciio.%s.api" % os.path.split(sub)[1]
            IO_PLUGINS.append(specs)


__all__ = [
    "__version__",
    "IO_PLUGINS",
]


def __dir__():
    return sorted(__all__)
