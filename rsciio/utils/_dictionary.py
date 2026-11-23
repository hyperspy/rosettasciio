# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

from box import Box


class DTBox(Box):
    """
    Subclass of Box to help migration from hyperspy `DictionaryTreeBrowser`
    to `Box` when splitting IO code from hyperspy to rosettasciio.

    When using `box_dots=True`, by default, period will be removed from keys.
    To support period containing keys, use `box_dots=False, default_box=True`.
    https://github.com/cdgriffith/Box/wiki/Types-of-Boxes#default-box
    """

    def add_node(self, path):
        keys = path.split(".")
        for key in keys:
            if self.get(key) is None:
                self[key] = {}
            self = self[key]

    def set_item(self, path, value):
        if self.get(path) is None:
            self.add_node(path)
        self[path] = value

    def has_item(self, path):
        return self.get(path) is not None


def dump_dictionary(
    file, dic, string="root", node_separator=".", value_separator=" = "
):
    for key in list(dic.keys()):
        if isinstance(dic[key], dict):
            dump_dictionary(file, dic[key], string + node_separator + key)
        else:
            file.write(
                string + node_separator + key + value_separator + str(dic[key]) + "\n"
            )
