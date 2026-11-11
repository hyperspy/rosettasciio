# -*- coding: utf-8 -*-
# Copyright 2007-2025 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

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
