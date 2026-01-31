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

import os

import yaml

__all__ = ["IO_PLUGINS"]


def __dir__():
    return sorted(__all__)


class IOPLUGINS(list):
    # Use a class to allow for docstring
    """
    List of available IO plugins.

    Each entry is a dictionary with the following keys:

    - ``'name'``: The name of the plugin.
    - ``'name_aliases'``: A list of alternative names for the plugin.
    - ``'description'``: A brief description of the plugin.
    - ``'full_support'``: A boolean indicating if the plugin has full support.
    - ``'default_extension'``: The default file extension for the plugin.
    - ``'writes'``: A boolean indicating if the plugin supports writing files.
    - ``'non_uniform_axis'``: A boolean indicating if the plugin supports non-uniform axes.
    - ``'api'``: The API module path as a string (e.g., 'rsciio.nexus').

    :meta hide-value:
    """


IO_PLUGINS = IOPLUGINS()

# libyaml C bindings may be missing
loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

for sub, _, _ in os.walk(os.path.abspath(os.path.dirname(__file__))):
    _specs_file = os.path.join(sub, "specifications.yaml")
    if os.path.isfile(_specs_file):
        with open(_specs_file, "r") as stream:
            _specs = yaml.load(stream, Loader=loader)
            # for testing purposes
            _specs["api"] = "rsciio.%s" % os.path.split(sub)[1]
            IO_PLUGINS.append(_specs)
