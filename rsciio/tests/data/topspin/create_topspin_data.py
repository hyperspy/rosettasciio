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

"""
The following python code creates small .app5 files for unit testing by
downsampling larger experimental datasets from various labs around the
world (Details below).

This file will not work as-written without the original files, but it is
still provided for two reasons:

    1) Transparency and validation.
    2) So future users can create additional test data should a future
       Topspin version be incompatable with this reader.

Additionally, while there is a standard method for transferring the
datasets themselves within the h5 file, the layout of the groups and metadata
files changes based on the local install of TopSpin.
"""

import numpy as np
import h5py
import xml.etree.ElementTree as ET

# convenience function for consistently copying 4D STEM groups.


def copy_downsampled_4D_group(old_h5py, new_h5py, guid, mask):
    new_h5py.create_group(guid)
    old_h5py_grp = old_h5py[guid]
    new_h5py_grp = new_h5py[guid]
    pixels_to_keep = mask.flatten() * np.arange(mask.size)
    pixels_to_keep = pixels_to_keep[mask.flatten() > 0]
    for new_grp_id, old_grp_id in enumerate(pixels_to_keep):
        stem_pixel_grp = old_h5py_grp[str(old_grp_id)]
        new_h5py_grp.create_group(str(new_grp_id))
        data = np.asanyarray(stem_pixel_grp["Data"][::16, ::16]).copy()
        new_h5py_grp[str(new_grp_id)].create_dataset(
            "Data", dtype="<f4", data=data, shape=data.shape
        )
        old_h5py_grp.copy(
            stem_pixel_grp["Metadata"],
            new_h5py_grp[str(new_grp_id)],
            "Metadata",
        )


def copy_downsampled_2D_dataset(old_h5py, new_h5py, guid):
    img = old_h5py[guid][1:12, 1:14]
    new_h5py.create_dataset(
        guid,
        shape=(11, 13),
        dtype="<f8",
        data=img,
    )


# %% Test file 1: Colorado School of Mines single-run file.
# hdf5 layout:
# | nanomegas_topspin_test_file.app5
# | - '39a4bbcd-2551-4912-9093-0eaed678dcff'  <-- STEM dataset
# | - 'Metadata'
# | - 'a0dba950-15bf-4a74-a81c-9cea8560c2c8' <-- 4DSTEM group
# | --- '1'
# | ----- 'Data'
# | ----- 'Metadata'
# | --- ....
# | - 'Version'


old_app5A = h5py.File("nanomegas_topspin_test_file.app5", "r")
new_app5A = h5py.File("topspin_test_A.app5", "w")
old_app5A.copy(old_app5A["Version"], new_app5A, "Version")

# copy '39a4bbcd-2551-4912-9093-0eaed678dcff' (2D virtual STEM image)
copy_downsampled_2D_dataset(
    old_app5A, new_app5A, "39a4bbcd-2551-4912-9093-0eaed678dcff"
)

# copy 'a0dba950-15bf-4a74-a81c-9cea8560c2c8' (4D STEM group)
mask = np.zeros([50, 500], dtype=bool)
mask[10:12, 50:55] = True
copy_downsampled_4D_group(
    old_app5A, new_app5A, "a0dba950-15bf-4a74-a81c-9cea8560c2c8", mask
)

# Edit and copy Metadata (custom operation based on local hardware).
root = ET.fromstring(old_app5A["Metadata"][()].decode())
root[7].text = "Test data A for rsciio"
root[12].text = "Edwin Supple"
root[13].text = (
    "Collected at the Colorado School of Mines for NIST's Quantitative"
    + " Nanostructure Characterization Group."
)
txt = ET.tostring(root, xml_declaration=True)
new_app5A.create_dataset_like("Metadata", old_app5A["Metadata"], data=txt)

# Close files
old_app5A.close()
new_app5A.close()

# %% Test file 2: Colorado School of Mines exported via Topspin
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

old_app5B = h5py.File("1750 1.app5", "r")
new_app5B = h5py.File("topspin_test_B.app5", "w")
old_app5B.copy(old_app5B["Version"], new_app5B, "Version")


# session A
session_id = "18d9446f-22bf-4fb1-8d13-338174e75d20"
new_app5B.create_group(session_id)
old_app5B.copy(
    old_app5B[session_id]["Version"], new_app5B[session_id], "Version"
)
# Copy '3526f008-a687-41fb-a21e-c21362241492' (4D STEM group)
mask = np.zeros([50, 50], dtype=bool)
mask[5:8, 4:11] = True
copy_downsampled_4D_group(
    old_app5B,
    new_app5B,
    session_id + "/3526f008-a687-41fb-a21e-c21362241492",
    mask,
)
# copy both virtual stem images (2D virtual STEM image)
for guid in [
    "b0212ada-90b1-4117-8ed5-355717e4910b",
    "b0e00442-4f83-4585-b84c-b82d2537b14b",
]:
    full_address = session_id + "/" + guid
    copy_downsampled_2D_dataset(old_app5B, new_app5B, full_address)

# session B
session_id = "b38446c6-6b25-4fee-9c62-6d8e4fcbeb5c"
new_app5B.create_group(session_id)
old_app5B.copy(
    old_app5B[session_id]["Version"], new_app5B[session_id], "Version"
)
# Copy "5a203dd8-ddbd-4775-8786-78d456e8b877" (2D virtual STEM image)
copy_downsampled_2D_dataset(
    old_app5B, new_app5B, session_id + "/5a203dd8-ddbd-4775-8786-78d456e8b877"
)


# Both Session-level Metadata files
for grp in [
    "18d9446f-22bf-4fb1-8d13-338174e75d20",
    "b38446c6-6b25-4fee-9c62-6d8e4fcbeb5c",
]:
    root = ET.fromstring(old_app5B[grp]["Metadata"][()].decode())
    root[7].text = "Test data B for rsciio"
    root[12].text = "Edwin Supple"
    root[13].text = (
        "Collected at the Colorado School of Mines for NIST's Quantitative"
        + " Nanostructure Characterization Group."
    )
    txt = ET.tostring(root, xml_declaration=True)
    new_app5B[grp].create_dataset_like(
        "Metadata", old_app5B[grp]["Metadata"], data=txt
    )

old_app5B.close()
new_app5B.close()

# %% Test file 3: Exported from Topspin at University of Glasgow
#
# hdf5 layout is identical to Test file 2, but Metadata is formatted
# differently

old_app5C = h5py.File("IM452.app5", "r")
new_app5C = h5py.File("topspin_test_C.app5", "w")
old_app5C.copy(old_app5C["Version"], new_app5C, "Version")


# session A
session_id = "030444c4-694d-4184-94be-4bb9ffeda552"
new_app5C.create_group(session_id)
old_app5C.copy(
    old_app5C[session_id]["Version"], new_app5C[session_id], "Version"
)
# Copy 'b5b0b75e-a5ff-4cdc-95d6-cab450c49e09' (2D virtual STEM image)
copy_downsampled_2D_dataset(
    old_app5C, new_app5C, session_id + "/b5b0b75e-a5ff-4cdc-95d6-cab450c49e09"
)
# Copy 'e0c99515-f43c-44c3-8394-132dafbacc39' (4DSTEM group)
mask = np.zeros([40, 200], dtype=bool)
mask[2:5, 3:8] = True
copy_downsampled_4D_group(
    old_app5C,
    new_app5C,
    session_id + "/e0c99515-f43c-44c3-8394-132dafbacc39",
    mask,
)

# session B
session_id = "9ebd1693-3de8-4294-877a-62670150b9fa"
new_app5C.create_group(session_id)
old_app5C.copy(
    old_app5C[session_id]["Version"], new_app5C[session_id], "Version"
)
# Copy '8bc2d64f-ea0e-420c-9099-51a07131f6e1' (2D virtual STEM image)
copy_downsampled_2D_dataset(
    old_app5C, new_app5C, session_id + "/8bc2d64f-ea0e-420c-9099-51a07131f6e1"
)


# Both Session-level Metadata files
for grp in [
    "030444c4-694d-4184-94be-4bb9ffeda552",
    "9ebd1693-3de8-4294-877a-62670150b9fa",
]:
    root = ET.fromstring(old_app5C[grp]["Metadata"][()].decode())
    root[7].text = "Test data C for rsciio"
    root[12].text = "Nidhi Choudhary and Ian MacLaren"
    root[13].text = (
        "Collected at the University of Glasgow as part of the following"
        + " paper: 10.1063/5.0292737."
    )
    txt = ET.tostring(root, xml_declaration=True)
    new_app5C[grp].create_dataset_like(
        "Metadata", old_app5C[grp]["Metadata"], data=txt
    )

old_app5C.close()
new_app5C.close()
