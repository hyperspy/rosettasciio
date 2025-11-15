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


The following is a copy of the python code used to create small (200kb)
app5 files from larger (approx 6GB) app5 files.

The original files were collected at the Colorado School of Mines by
NIST's Quantitative Nanostructure Characterization Group. The code below
will not recreate the test data without the original files, but it is
provided regarless:

1) For transparency and validation.
2) So that future users can potientially use it to make their own test data.

=============================================================================

import numpy as np
import h5py
import xml.etree.ElementTree as ET

# %% Step 1: create from_file app5

old_app5 = h5py.File("nanomegas_topspin_test_file.app5", "r")
new_app5 = h5py.File("topspin_from_file.app5", "w")

# Image series (virtual STEM)
img_ser = np.asanyarray(
    old_app5["39a4bbcd-2551-4912-9093-0eaed678dcff"][10:12, 50:55]
)
new_app5.create_dataset(
    "thefirst-fake-guid-0for-4unittesting",
    shape=(2, 5),
    dtype="<f8",
    data=img_ser,
)

# Version
old_app5.copy(old_app5["Version"], new_app5, "Version")

# Data Series (4D stem datasets)
new_app5.create_group("0another-fake-guid-0for-4unittesting")
old_grp = old_app5["a0dba950-15bf-4a74-a81c-9cea8560c2c8"]
new_grp = new_app5["0another-fake-guid-0for-4unittesting"]
mask = np.zeros([50, 500], dtype=bool)
mask[10:12, 50:55] = True
keeps = mask.flatten() * np.arange(50 * 500)
keeps = keeps[keeps > 0]
for i in keeps:
    old_sub_grp = old_grp[str(i)]
    new_grp.create_group(str(i))
    data = np.asanyarray(old_sub_grp["Data"][::8, ::8]).copy()
    new_grp[str(i)].create_dataset(
        "Data", shape=(32, 32), dtype="<f4", data=data
    )
    old_sub_grp.copy(old_sub_grp["Metadata"], new_grp[str(i)], "Metadata")

# Metadata
root = ET.fromstring(old_app5["Metadata"][()].decode())
root[7].text = "Test data A for rsciio"
root[12].text = "Austin Gerlt and Edwin Supple"
root[16][18][1].text = "64"
root[16][54][1][0].text = "0another-fake-guid-0for-4unittesting"
root[16][54][1][1].text = "25"
root[16][55][1][0].text = "thefirst-fake-guid-0for-4unittesting"
txt = ET.tostring(root, xml_declaration=True)
new_app5.create_dataset_like("Metadata", old_app5["Metadata"], data=txt)

old_app5.close()
new_app5.close()

# %% Step 2: create from_export app5

old_app5e = h5py.File("1750 1.app5", "r")
new_app5e = h5py.File("topspin_from_export.app5", "w")

# Version
old_app5e.copy(old_app5e["Version"], new_app5e, "Version")

# session groups
new_app5e.create_group("18d9446f-22bf-4fb1-8d13-338174e75d20")
new_app5e.create_group("b38446c6-6b25-4fee-9c62-6d8e4fcbeb5c")
s1o = old_app5e["18d9446f-22bf-4fb1-8d13-338174e75d20"]
s2o = old_app5e["b38446c6-6b25-4fee-9c62-6d8e4fcbeb5c"]
s1n = new_app5e["18d9446f-22bf-4fb1-8d13-338174e75d20"]
s2n = new_app5e["b38446c6-6b25-4fee-9c62-6d8e4fcbeb5c"]


# this has two sessions, lets just do them each verbosely
s1_guids = [x for i, x in enumerate(s1o) if i in [0, 4, 5]]
s1n.create_group(s1_guids[0])
# session 1, guid1; a 4D stem map
s1g1o = s1o[s1_guids[0]]
s1g1n = s1n[s1_guids[0]]
mask = np.zeros([50, 50], dtype=bool)
mask[5:8, 4:11] = True
keeps = mask.flatten() * np.arange(50 * 50)
keeps = keeps[keeps > 0]
for i in keeps:
    s1g1n.create_group(str(i))
    data = np.asanyarray(s1g1o[str(i)]["Data"][::20, ::20]).copy()
    s1g1n[str(i)].create_dataset(
        "Data", shape=(29, 29), dtype="<f4", data=data
    )
    s1g1o.copy(s1g1o[str(i)]["Metadata"], s1g1n[str(i)], "Metadata")
# session 1, guid2; a virtual stem map
img_ser = np.asanyarray(s1o[s1_guids[1]][:11, :13])
s1n.create_dataset(s1_guids[1], shape=(11, 13), dtype="<f8", data=img_ser)
# session 1, guid3; a virtual stem map
img_ser = np.asanyarray(s1o[s1_guids[2]][:13, :17])
s1n.create_dataset(s1_guids[2], shape=(13, 17), dtype="<f8", data=img_ser)
# Version
s1o.copy(s1o["Version"], s1n, "Version")
# metadata
root1 = ET.fromstring(s1o["Metadata"][()].decode())
root1[7].text = "Test data B for rsciio"
root1[12].text = "Austin Gerlt and Edwin Supple"
root1[16][18][1].text = "58"
txt = ET.tostring(root1, xml_declaration=True)
s1n.create_dataset_like("Metadata", s1o["Metadata"], data=txt)


# session 2: a single 4D stem map
s2_guid = "5a203dd8-ddbd-4775-8786-78d456e8b877"
img_ser = np.asanyarray(s2o[s2_guid][:11, :13])
s2n.create_dataset(s2_guid, shape=(11, 13), dtype="<f8", data=img_ser)
s2o.copy(s2o["Version"], s2n, "Version")
# metadata
root2 = ET.fromstring(s2o["Metadata"][()].decode())
root2[7].text = "Test data C for rsciio"
root2[12].text = "Austin Gerlt and Edwin Supple"
txt = ET.tostring(root2, xml_declaration=True)
s2n.create_dataset_like("Metadata", s2o["Metadata"], data=txt)

old_app5e.close()
new_app5e.close()
