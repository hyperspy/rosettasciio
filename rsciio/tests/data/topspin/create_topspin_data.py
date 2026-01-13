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
#
#
# The following python code creates small .app5 files for unit testing by
# downsampling larger experimental datasets from various labs around the
# world (Details below).

# This file will not work as-written without the original files, but it is
# still provided for two reasons:

#     1) Transparency and validation.
#     2) To create additional test files if future versions of TopSpin
#        are incompatible with this reader.

# Additionally, while the datasets themselves have a consistent layout,
# their location within the .app5 files changes based on how the
# file was exported from TopSpin.


import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def downsample_4D_group(group, mask):
    """In-place reduction of hdf5 group representing 4D-STEM data."""
    pxls = mask.flatten() * np.arange(mask.size)
    pxls_dict = dict([str(x), str(i)] for i, x in enumerate(pxls[pxls > 0]))
    gnames = [str(x) for x in np.arange(len(group.keys()))]
    for gname in gnames:
        if gname in pxls_dict.keys():
            new_id = pxls_dict[gname]
            group.move(gname, new_id)
            data = group[new_id]["Data"][::19, ::23].copy()
            del group[new_id]["Data"]
            group[new_id].create_dataset("Data", dtype="<f4", data=data)
        else:
            del group[gname]


def downsample_2D_dataset(group, guid):
    """In-place reduction of 2D stem image."""
    img = group[guid][::7, ::9]
    del group[guid]
    group.create_dataset(guid, dtype="<f8", data=img)


# %% Test file 1: Colorado School of Mines single-run file.
# hdf5 layout:
# | cb76b58d-7906-423d-b681-3af9587e9e76.app5
# | - '39a4bbcd-2551-4912-9093-0eaed678dcff'  <-- STEM dataset
# | - 'Metadata'
# | - 'a0dba950-15bf-4a74-a81c-9cea8560c2c8' <-- 4DSTEM group
# | --- '1'
# | ----- 'Data'
# | ----- 'Metadata'
# | --- ....
# | - 'Version'

# Copy 6.6 GB app5 file
old_fname = "/home/arg6/data/app5/cb76b58d-7906-423d-b681-3af9587e9e76.app5"
temp_fname = "A.temp"
final_fname = "topspin_test_A.app5"
if os.path.exists(temp_fname):
    os.remove(temp_fname)
if os.path.exists(final_fname):
    os.remove(final_fname)
shutil.copyfile(old_fname, temp_fname)
app5_A = h5py.File(temp_fname, "a")

# downsample '39a4bbcd-2551-4912-9093-0eaed678dcff' (2D virtual STEM image).
downsample_2D_dataset(app5_A, "39a4bbcd-2551-4912-9093-0eaed678dcff")

# downsample 'a0dba950-15bf-4a74-a81c-9cea8560c2c8' (4D STEM group)
mask = np.zeros([50, 500], dtype=bool)
mask[10:12, 50:55] = True
downsample_4D_group(app5_A["a0dba950-15bf-4a74-a81c-9cea8560c2c8"], mask)

# Edit toplevel metadata to better reflect contributions.
# Done manually as different TopSpin setups store MetaData slightly different.
root = ET.fromstring(app5_A["Metadata"][()].decode())
root[7].text = "Test data A for rsciio"
root[12].text = "Edwin Supple"
root[13].text = (
    "Collected at the Colorado School of Mines for NIST's Quantitative"
    + " Nanostructure Characterization Group."
)
txt = ET.tostring(root, xml_declaration=True)
app5_A.move("Metadata", "temp")
app5_A.create_dataset_like("Metadata", app5_A["temp"], data=txt)
del app5_A["temp"]

# repackage to remove empty space
app5_A.close()
subprocess.run(["h5repack", temp_fname, final_fname])
subprocess.run(["rm", temp_fname])


# %% Test file 2: Colorado School of Mines exported via Topspin.
# hdf5 layout:
# | nanomegas_topspin_test_file.app5
# |
# | - '18d9446f-22bf-4fb1-8d13-338174e75d20'   <-- Session A
# | --- '3526f008-a687-41fb-a21e-c21362241492' <-- 4DSTEM group
# | ----- '1'
# | ------- 'Data'
# | ------- 'Metadata'   <-- Metadata for single pixel image
# | ----- ....
# | --- 'Metadata' <-- Metadata for session
# | --- 'Version'
# | --- 'b0212ada-90b1-4117-8ed5-355717e4910b' <-- STEM dataset
# | --- 'b0e00442-4f83-4585-b84c-b82d2537b14b' <-- STEM dataset
# |
# | - 'b38446c6-6b25-4fee-9c62-6d8e4fcbeb5c'   <-- Session B
# | --- ....
# |
# | - 'Version'


# Copy 1.7 GB app5 file.
old_fname = "/home/arg6/data/app5/1750 1.app5"
temp_fname = "B.app5"
final_fname = "topspin_test_B.app5"
if os.path.exists(temp_fname):
    os.remove(temp_fname)
if os.path.exists(final_fname):
    os.remove(final_fname)
shutil.copyfile(old_fname, temp_fname)
app5_B = h5py.File(temp_fname, "a")

# session 1
B1 = app5_B["18d9446f-22bf-4fb1-8d13-338174e75d20"]
# Downsample '3526f008-a687-41fb-a21e-c21362241492' (4D STEM group)
mask = np.zeros([50, 50], dtype=bool)
mask[5:8, 4:11] = True
downsample_4D_group(B1["3526f008-a687-41fb-a21e-c21362241492"], mask)
# Downsample both virtual stem images (2D virtual STEM image)
for guid in [
    "b0212ada-90b1-4117-8ed5-355717e4910b",
    "b0e00442-4f83-4585-b84c-b82d2537b14b",
]:
    downsample_2D_dataset(B1, guid)

# session 2
B2 = app5_B["b38446c6-6b25-4fee-9c62-6d8e4fcbeb5c"]
# Copy "5a203dd8-ddbd-4775-8786-78d456e8b877" (2D virtual STEM image)
downsample_2D_dataset(B2, "5a203dd8-ddbd-4775-8786-78d456e8b877")

# Edit toplevel metadata to better reflect contributions.
for grp in [B1, B2]:
    root = ET.fromstring(grp["Metadata"][()].decode())
    root[7].text = "Test data B for rsciio"
    root[12].text = "Edwin Supple"
    root[13].text = (
        "Collected at the Colorado School of Mines for NIST's Quantitative"
        + " Nanostructure Characterization Group."
    )
    txt = ET.tostring(root, xml_declaration=True)
    grp.move("Metadata", "temp")
    grp.create_dataset_like("Metadata", grp["temp"], data=txt)
    del grp["temp"]

# repackage to remove empty space
app5_B.close()
subprocess.run(["h5repack", temp_fname, final_fname])
subprocess.run(["rm", temp_fname])

# %% Test file 3: Exported from Topspin at University of Glasgow.
#
# hdf5 layout is identical to Test file 2, but Metadata is formatted
# differently.

# copy 2.1Gb app5 file.
old_fname = "/home/arg6/data/app5/IM452.app5"
temp_fname = "C.app5"
final_fname = "topspin_test_C.app5"
if os.path.exists(temp_fname):
    os.remove(temp_fname)
if os.path.exists(final_fname):
    os.remove(final_fname)
shutil.copyfile(old_fname, temp_fname)
app5_C = h5py.File(temp_fname, "a")

# session 1
C1 = app5_C["030444c4-694d-4184-94be-4bb9ffeda552"]
# Downsample 'e0c99515-f43c-44c3-8394-132dafbacc39' (4D STEM group)
mask = np.zeros([40, 200], dtype=bool)
mask[2:5, 3:8] = True
downsample_4D_group(C1["e0c99515-f43c-44c3-8394-132dafbacc39"], mask)
# Downsample 'b5b0b75e-a5ff-4cdc-95d6-cab450c49e09' (2D virtual STEM image)
downsample_2D_dataset(C1, "b5b0b75e-a5ff-4cdc-95d6-cab450c49e09")


# session 2
C2 = app5_C["9ebd1693-3de8-4294-877a-62670150b9fa"]
# Copy '8bc2d64f-ea0e-420c-9099-51a07131f6e1' (2D virtual STEM image)
downsample_2D_dataset(C2, "8bc2d64f-ea0e-420c-9099-51a07131f6e1")

# Edit toplevel metadata to better reflect contributions.
for grp in [C1, C2]:
    root = ET.fromstring(grp["Metadata"][()].decode())
    root[7].text = "Test data C for rsciio"
    root[12].text = "Nidhi Choudhary and Ian MacLaren"
    root[13].text = (
        "Collected at the University of Glasgow as part of the following"
        + " paper: 10.1063/5.0292737."
    )
    txt = ET.tostring(root, xml_declaration=True)
    grp.move("Metadata", "temp")
    grp.create_dataset_like("Metadata", grp["temp"], data=txt)
    del grp["temp"]

# repackage to remove empty space
app5_C.close()
subprocess.run(["h5repack", temp_fname, final_fname])
subprocess.run(["rm", temp_fname])
