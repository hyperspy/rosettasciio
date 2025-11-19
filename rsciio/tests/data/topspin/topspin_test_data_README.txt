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

# %% Test file 1: Colorado School of Mines single-run file.


old_app5A = h5py.File("nanomegas_topspin_test_file.app5", "r")
new_app5A = h5py.File("topspin_test_A.app5", "w")

# Image series (virtual STEM)
img_ser = np.asanyarray(
    old_app5A["39a4bbcd-2551-4912-9093-0eaed678dcff"][10:12, 50:55]
)
new_app5A.create_dataset(
    "thefirst-fake-guid-0for-4unittesting",
    shape=(2, 5),
    dtype="<f8",
    data=img_ser,
)

# Version
old_app5A.copy(old_app5A["Version"], new_app5A, "Version")

# Data Series (4D stem datasets)
new_app5A.create_group("0another-fake-guid-0for-4unittesting")
old_grp = old_app5A["a0dba950-15bf-4a74-a81c-9cea8560c2c8"]
new_grp = new_app5A["0another-fake-guid-0for-4unittesting"]
mask = np.zeros([50, 500], dtype=bool)
mask[10:12, 50:55] = True
keeps = mask.flatten() * np.arange(50 * 500)
keeps = keeps[keeps > 0]
for new_i, i in enumerate(keeps):
    old_sub_grp = old_grp[str(i)]
    new_grp.create_group(str(new_i))
    data = np.asanyarray(old_sub_grp["Data"][::8, ::8]).copy()
    new_grp[str(new_i)].create_dataset(
        "Data", shape=(32, 32), dtype="<f4", data=data
    )
    old_sub_grp.copy(old_sub_grp["Metadata"], new_grp[str(new_i)], "Metadata")

# Metadata
root = ET.fromstring(old_app5A["Metadata"][()].decode())
root[7].text = "Test data A for rsciio"
root[12].text = "Edwin Supple"
root[16][18][1].text = "64"
root[16][54][1][0].text = "0another-fake-guid-0for-4unittesting"
root[16][54][1][1].text = "25"
root[16][55][1][0].text = "thefirst-fake-guid-0for-4unittesting"
txt = ET.tostring(root, xml_declaration=True)
new_app5A.create_dataset_like("Metadata", old_app5A["Metadata"], data=txt)

old_app5A.close()
new_app5A.close()

# %% Test file 2: Colorado School of Mines exported via Topspin

old_app5B = h5py.File("1750 1.app5", "r")
new_app5B = h5py.File("topspin_test_B.app5", "w")

# Version
old_app5B.copy(old_app5B["Version"], new_app5B, "Version")

# session subgroups
new_app5B.create_group("18d9446f-22bf-4fb1-8d13-338174e75d20")
new_app5B.create_group("b38446c6-6b25-4fee-9c62-6d8e4fcbeb5c")
s1o = old_app5B["18d9446f-22bf-4fb1-8d13-338174e75d20"]
s2o = old_app5B["b38446c6-6b25-4fee-9c62-6d8e4fcbeb5c"]
s1n = new_app5B["18d9446f-22bf-4fb1-8d13-338174e75d20"]
s2n = new_app5B["b38446c6-6b25-4fee-9c62-6d8e4fcbeb5c"]
s1_guids = [x for i, x in enumerate(s1o) if i in [0, 4, 5]]


# session 1, guid1; a 4D stem map
s1n.create_group(s1_guids[0])
s1g1o = s1o[s1_guids[0]]
s1g1n = s1n[s1_guids[0]]
mask = np.zeros([50, 50], dtype=bool)
mask[5:8, 4:11] = True
keeps = mask.flatten() * np.arange(50 * 50)
keeps = keeps[keeps > 0]
for new_i, i in enumerate(keeps):
    s1g1n.create_group(str(new_i))
    data = np.asanyarray(s1g1o[str(i)]["Data"][::20, ::20]).copy()
    s1g1n[str(new_i)].create_dataset(
        "Data", shape=(29, 29), dtype="<f4", data=data
    )
    s1g1o.copy(s1g1o[str(i)]["Metadata"], s1g1n[str(new_i)], "Metadata")

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
root1[12].text = "Edwin Supple"
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
root2[12].text = "Edwin Supple"
txt = ET.tostring(root2, xml_declaration=True)
s2n.create_dataset_like("Metadata", s2o["Metadata"], data=txt)

old_app5B.close()
new_app5B.close()

# %% Test file 3: Exported from Topspin at University of Glasgow

old_app5C = h5py.File("IM452.app5", "r")
new_app5C = h5py.File("topspin_test_C.app5", "w")

# Version
old_app5C.copy(old_app5C["Version"], new_app5C, "Version")

# session groups
new_app5C.create_group("030444c4-694d-4184-94be-4bb9ffeda552")
new_app5C.create_group("9ebd1693-3de8-4294-877a-62670150b9fa")
s1o = old_app5C["030444c4-694d-4184-94be-4bb9ffeda552"]
s2o = old_app5C["9ebd1693-3de8-4294-877a-62670150b9fa"]
s1n = new_app5C["030444c4-694d-4184-94be-4bb9ffeda552"]
s2n = new_app5C["9ebd1693-3de8-4294-877a-62670150b9fa"]
s1_guids = [x for i, x in enumerate(s1o) if i in [3, 4]]


# session 1, guid1; a virtual stem map
img_ser = np.asanyarray(s1o[s1_guids[0]][:3, :7])
s1n.create_dataset(s1_guids[0], shape=(3, 7), dtype="<f8", data=img_ser)

# session1, guid2; a 4D stem map
s1n.create_group(s1_guids[1])
s1g2o = s1o[s1_guids[1]]
s1g2n = s1n[s1_guids[1]]
mask = np.zeros([40, 200], dtype=bool)
mask[:3, :5] = True
keeps = mask.flatten() * np.arange(40 * 200)
keeps = keeps[keeps > 0]
for new_i, i in enumerate(keeps):
    s1g2n.create_group(str(new_i))
    data = np.asanyarray(s1g2o[str(i)]["Data"][::32, ::32]).copy()
    s1g2n[str(new_i)].create_dataset(
        "Data", shape=(8, 8), dtype="<f4", data=data
    )
    s1g2o.copy(s1g2o[str(i)]["Metadata"], s1g2n[str(new_i)], "Metadata")

# Version
s1o.copy(s1o["Version"], s1n, "Version")
# metadata
root1 = ET.fromstring(s1o["Metadata"][()].decode())
root1[7].text = "Test data C for rsciio"
root1[12].text = "Ian MacLaren"
root1[16][15][1].text = "8"  # Frame size
txt = ET.tostring(root1, xml_declaration=True)
s1n.create_dataset_like("Metadata", s1o["Metadata"], data=txt)


# session 2: a single 2D image
s2_guid = "8bc2d64f-ea0e-420c-9099-51a07131f6e1"
img_ser = np.asanyarray(s2o[s2_guid][:13, :17])
s2n.create_dataset(s2_guid, shape=(13, 17), dtype="<f8", data=img_ser)
s2o.copy(s2o["Version"], s2n, "Version")

# metadata
root2 = ET.fromstring(s2o["Metadata"][()].decode())
root2[7].text = "Test data C for rsciio"
root2[12].text = "Ian MacLaren"
txt = ET.tostring(root2, xml_declaration=True)
s2n.create_dataset_like("Metadata", s2o["Metadata"], data=txt)

old_app5C.close()
new_app5C.close()
