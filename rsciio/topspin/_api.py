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


import datetime
import mmap
import os
import sys
import xml.etree.cElementTree as ET

import h5py
import numpy as np
from tqdm import tqdm

from rsciio._docstrings import FILENAME_DOC, RETURNS_DOC


def file_reader(filename, dataset_path=None, dryrun=False, show_progressbar=True):
    """
    Read .app5 file format both read in and exported by NanoMegas's
    Topspin software.

    .app5 files use the hdf5 file format, with metadata stored as
    binarized XML-style text strings. Each individual SPED (Scanning
    Precession Electron Diffraction) or STEM result is initially
    saved by Topspin as a simplified .app5, and groups of these files
    can then also be exported as a single .app5 file representing a
    larger experiment. Both methods can be read, with only the values
    of `dataset_path` being changed.

    Parameters
    ----------
    %s
    dataset_path: None, str, default=None
        If None, no absolute path is searched and every dataset in the
        .app5 file is returned. If a string is given, only the STEM or
        SPED dataset with the matching absolute path within the .app5
        file will be returned. For example,

        ```signals = rsciio.topspin.file_reader("fname.app5")```

        will import every SPED and STEM dataset recorded in the file
        `fname.app5`, whereas

        ```signals = rsciio.topspin.file_reader(`fname.app5`,
            ,dataset_path='18d9446f-22bf-4fb1-8d13-338174e75d20')

        would only import the exerimental data collected in experiment
        18d9446f-22bf-4fb1-8d13-338174e75d20. The `dryrun` variable
        can be used to list all allowable addresses without requiring
        loading from disk.

        ```rsciio.topspin.file_reader("fname.app5", dryrun=True)```

    dryrun : bool
        If True, the .app5 files are scanned without being loaded, and
        a summary is printed to the log. Default is False.

    show_progressbar: bool, default=True
        Whether to show the progressbar or not. If True, shows progress
        bar for each individual dataset loaded from the original file.
    %s

    Notes
    -----
    The hierarchy of the Metadata objects stored in app5 files changes
    based on Topspin Procedure and microscope setup. RosettaSciIO does
    not need this information to populate the 'metadata' or 'axes'
    results, but be aware the organization of hte 'original_metadata'
    can change if the experimental setup changes.
    """

    def looks_like_a_guid(name):
        """Checks if a text string matches the shape of a guid"""
        lengths = [len(x) for x in name.split("-")]
        return lengths == [8, 4, 4, 4, 12]

    # read dataset
    h5_file = h5py.File(filename, "r")
    load_single_file = False
    if dataset_path is not None:
        if "Metadata" in h5_file[dataset_path]:  # this is a session id
            h5_file = h5_file[dataset_path]
        else:  # This is a single file of interest
            load_single_file = True

    # Generate task list by searching .app5 for GUID's
    task_list = []
    group_names = [x for x in h5_file.keys()]
    if np.isin("Metadata", group_names):  # single experiment file.
        for name in h5_file.keys():
            if looks_like_a_guid(name):
                task_list.append([name, "Metadata"])
    else:  # Multi-experimental file.
        for top_name in h5_file.keys():
            if looks_like_a_guid(top_name):
                for name in h5_file[top_name].keys():
                    if looks_like_a_guid(name):
                        address = top_name + "/" + name
                        meta = top_name + "/" + "Metadata"
                        task_list.append([address, meta])
    # prune list for single-file query
    if load_single_file:
        task_list = [x for x in task_list if x[0] == dataset_path]

    # Parse each dataset's metadata
    datasets_list = []
    for address, meta_loc in task_list:
        name = address.split("/")[-1]
        xml_str = h5_file[meta_loc][()].decode()
        original_meta = _parse_app5_xml(xml_str)
        hspy_meta = _parse_hspy_meta(original_meta, filename)
        if isinstance(h5_file[address], h5py.Group):
            # 4D-STEM
            h5_grp = h5_file[address]
            axes = _get_4D_axes(h5_grp)
            hspy_meta["Signal"]["signal_type"] = "electron_diffraction"
        else:
            # 2D composite image
            axes = _get_2D_axes(xml_str, name)
            shape = h5_file[address].shape
            axes[0]["size"] = shape[0]
            axes[1]["size"] = shape[1]
            hspy_meta["Signal"]["signal_type"] = "STEM"
        datasets_list.append(
            {
                "axes": axes,
                "metadata": hspy_meta,
                "original_metadata": original_meta,
            }
        )

    if dryrun:
        filename = str(filename)
        message = "The following would be imported from " + filename + ":"
        for i, ds_dict in enumerate(datasets_list):
            shape = [x["size"] for x in ds_dict["axes"]]
            guid = task_list[i][0]
            name = ds_dict["original_metadata"]["Title"]
            message += "\n   - {}, {}\n      {}".format(name, shape, guid)
        sys.stdout.write(message)
        return []

    for i, ds_dict in enumerate(
        tqdm(
            datasets_list,
            desc="Loading Datasets...",
            disable=not show_progressbar,
            total=len(datasets_list),
        )
    ):
        shape = [x["size"] for x in ds_dict["axes"]]
        address = task_list[i][0]
        if len(shape) == 2:
            datasets_list[i]["data"] = np.asanyarray(h5_file[address])

        elif len(shape) == 4:
            # for 4D, data is loaded as ((x*y),ky,kz), then reshaped.
            first_key = [x for x in h5_file[address]][0]
            signal_dtype = h5_file[address][first_key]["Data"].dtype
            signal_shape = h5_file[address][first_key]["Data"].shape
            signal_size = np.prod(signal_shape)
            key_count = len(h5_file[address].keys())
            # develper note: it's possible to open/load/close every dataset
            # via h5py, but it's faster to just lookup the offsets and load
            # from memory with mmap and numpy.
            offsets = [
                h5_file[address][str(i)]["Data"].id.get_offset()
                for i in range(key_count)
            ]
            with open(filename, "rb") as f:
                fileno = f.fileno()
                mapping = mmap.mmap(fileno, 0, access=mmap.ACCESS_READ)
                data = np.stack(
                    [
                        np.frombuffer(
                            mapping, dtype=signal_dtype, count=signal_size, offset=i
                        ).reshape(signal_shape)
                        for i in offsets
                    ]
                )
            # check for length 1 navigation axes
            if shape[1] == 1:
                del shape[1]
                del datasets_list[i]["axes"][1]
                datasets_list[i]["axes"][-1]["index_in_array"] = 1
                datasets_list[i]["axes"][-2]["index_in_array"] = 2

            if shape[0] == 1:
                del shape[0]
                del datasets_list[i]["axes"][0]
                datasets_list[i]["axes"][0]["index_in_array"] = 0
                datasets_list[i]["axes"][1]["index_in_array"] = 1
                if len(shape) == 3:
                    datasets_list[i]["axes"][2]["index_in_array"] = 2

            datasets_list[i]["data"] = data.reshape(shape)

    return datasets_list


file_reader.__doc__ %= (FILENAME_DOC, RETURNS_DOC)


def _get_4D_axes(h5_grp):
    vuyx_axes = [
        {
            "name": "y",
            "units": "nm",
            "size": 0,
            "scale": 0,
            "offset": 0,
            "navigate": True,
            "index_in_array": 0,
        },
        {
            "name": "x",
            "units": "nm",
            "size": 0,
            "scale": 0,
            "offset": 0,
            "navigate": True,
            "index_in_array": 1,
        },
        {
            "name": "ky",
            "units": "mrads",
            "size": 0,
            "scale": 0,
            "offset": 0,
            "navigate": False,
            "index_in_array": 2,
        },
        {
            "name": "kx",
            "units": "mrads",
            "size": 0,
            "scale": 0,
            "offset": 0,
            "navigate": False,
            "index_in_array": 3,
        },
    ]

    # K-space axes
    k_space_id = [x for x in h5_grp.keys()][0]
    k_space = ET.fromstring(h5_grp[k_space_id]["Metadata"][()].decode())[0]
    shape = h5_grp[k_space_id]["Data"].shape
    vuyx_axes[2]["units"] = k_space[0][2][1].text
    vuyx_axes[2]["size"] = shape[0]
    vuyx_axes[2]["scale"] = float(k_space[0][1].text)
    vuyx_axes[2]["offset"] = float(k_space[0][0].text)
    vuyx_axes[3]["units"] = k_space[1][2][1].text
    vuyx_axes[3]["size"] = shape[1]
    vuyx_axes[3]["scale"] = float(k_space[1][1].text)
    vuyx_axes[3]["offset"] = float(k_space[1][0].text)

    # Image-space axes
    # Because the experiment-level metadata files sometimes change, it's
    # better to read information from dataset-level metadata files.
    # however, reading all is slow, so first try a lazy hack
    # that assumes a row-major scanning in the TEM.
    x = []
    y = []
    keys = np.sort([int(x) for x in h5_grp.keys()]).astype(str)
    for i in keys:
        txt = h5_grp[i]["Metadata"][()].decode()
        root = ET.fromstring(txt)[1][0][1]
        x.append(float(root[0].text))
        y.append(float(root[1].text))
        if y[-1] > y[0]:
            # new row in y, meaning we have dx, dy, and ly (theoretically)
            break
    ux = np.unique(x)
    dx = (ux.max() - ux.min()) / (ux.size - 1)
    lx = len(ux)
    uy = np.unique(y)
    dy = uy.max() - uy.min()
    ly = len(h5_grp.keys()) // len(ux)
    if len(h5_grp.keys()) % len(uy) != 0:
        # Something was wrong with the hack above, and the keys are not
        # numbered in the expected order. As a fallback, read every
        # voxel-leval metadata file (these are different than the main ones.)
        x = []
        y = []
        for i in h5_grp.keys():
            txt = h5_grp[i]["Metadata"][()].decode()
            root = ET.fromstring(txt)[1][0][1]
            x.append(float(root[0].text))
            y.append(float(root[1].text))
        ux = np.unique(x)
        uy = np.unique(y)
        dx = (ux.max() - ux.min()) / (ux.size - 1)
        dy = (uy.max() - uy.min()) / (uy.size - 1)
        lx = len(ux)
        ly = len(uy)

    vuyx_axes[0]["size"] = ly
    vuyx_axes[0]["scale"] = float(np.around(dy, 12) * 1e9)
    vuyx_axes[0]["offset"] = float(uy.min() * 1e9)
    vuyx_axes[1]["size"] = lx
    vuyx_axes[1]["scale"] = float(np.around(dx, 12) * 1e9)
    vuyx_axes[1]["offset"] = float(ux.min() * 1e9)
    return vuyx_axes


def _get_2D_axes(metadata_xml_string, name):
    root = ET.fromstring(metadata_xml_string)
    # populate with default values
    elem = None
    yx_axes = [
        {
            "name": "y",
            "units": "nm",
            "size": 0,
            "scale": 0,
            "offset": 0,
            "navigate": True,
            "index_in_array": 0,
        },
        {
            "name": "x",
            "units": "nm",
            "size": 0,
            "scale": 0,
            "offset": 0,
            "navigate": True,
            "index_in_array": 1,
        },
    ]
    for value in root.findall(".//Value"):
        if value.attrib["Serializer"] == "ImageDataSerializer":
            if value[0].text == name:
                elem = value.find("Calibration")
                break
    if elem is not None:
        yx_axes[0]["scale"] = float(elem.find("X/Scale").text) * 1e9
        yx_axes[0]["offset"] = float(elem.find("X/Offset").text) * 1e9
        yx_axes[1]["scale"] = float(elem.find("Y/Scale").text) * 1e9
        yx_axes[1]["offset"] = float(elem.find("Y/Offset").text) * 1e9
    return yx_axes


def _parse_app5_xml(metadata_string: str, f_name: str = ""):
    """
    Converts 'MetaData' strings into nested python dictionaries.

    Parameters
    ----------
    metadata_string
        text representation of app5 metadata. Can be generated
        from an app5 file opened with `h5py` as *f* using:
            metadata_string = f['path/to/Metadata'][()].decode()

    f_name
        Original filename. Used for populating the metadata.

    Returns
    -------
    dict
        Nested metadata dictionary.

    Notes
    -----
    Metadata strings in app5 files have 18 Elements. 16 of these can be
    directly converted to dictionary key/item pairs. The remaining two
    ('ProcedureData' and 'HardwareSettings') contain nested XML Elements
    that follow the pattern:

        <Item>
          <Name Serializer="String" Version="1">ScanRotation</Name>
          <Value Serializer="Double" Version="1">0</Value>
        </Item>

    which is equivalent to a python dictionary of the form:

        {'ScanRotation':float(0)}

    This function checks Value.attrib['Serializer'] for every name/value
    pair. If it's a python type, the pair are added to the dictionary.
    Otherwise, it is the start of a new Element leaf, and thus a new
    nested dictionary will be created.
    """

    def name_val_decode(element):
        serializer = element.attrib["Serializer"]
        if serializer == "Boolean":
            out = {element.tag: bool(element.text)}
        elif serializer == "Double":
            out = {element.tag: float(element.text)}
        elif serializer == "String":
            out = {element.tag: str(element.text)}
        elif serializer in ["UInt32", "Int32"]:
            out = {element.tag: int(element.text)}
        else:
            out = {}
            for leaf in element:
                if len(leaf.attrib) > 0:
                    out.update(name_val_decode(leaf))
        return out

    root = ET.fromstring(metadata_string)
    # Nested dictionary of ALL metadata organized identical to the original
    all_meta = {}
    for branch in root:
        if branch.tag in ["ProcedureData", "HardwareSettings"]:
            branch_dict = {}
            for leaf in branch:
                serializer = leaf[1].attrib["Serializer"]
                if serializer == "Boolean":
                    branch_dict[leaf[0].text] = leaf[1].text
                elif serializer == "Double":
                    branch_dict[leaf[0].text] = float(leaf[1].text)
                elif serializer == "String":
                    branch_dict[leaf[0].text] = str(leaf[1].text)
                elif serializer in ["UInt32", "Int32"]:
                    branch_dict[leaf[0].text] = int(leaf[1].text)
                else:
                    branch_dict[leaf[0].text] = name_val_decode(leaf[1])
            all_meta[branch.tag] = branch_dict
        else:
            all_meta[branch.tag] = branch.text
    return all_meta


def _parse_hspy_meta(all_meta, f_name):
    # Convert some of the useful user-defined metadata into a text string to
    # save as "General/notes".
    f_notes = ""
    for attr in [
        "ProcedureName",
        "FriendlyName",
        "Title",
        "SystemName",
        "Specimen",
        "Comments",
    ]:
        if attr in all_meta:
            f_notes += "{}:  {} \n".format(attr, str(all_meta[attr]))

    hspy_meta = {
        "General": {
            "FileIO": {
                "0": {
                    "operation": "load",
                    "io_plugin": "rsciio.topspin",
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            },
            "original_filename": os.path.split(f_name)[-1],
            "notes": f_notes,
            "title": all_meta["Id"],
        },
        "Sample": {},
        "Signal": {},
    }
    if "CreatedDateTime" in all_meta:
        dt = all_meta["CreatedDateTime"]
        date = "-".join(np.array(dt.split(" ")[0].split("/"))[(2, 0, 1),])
        time = dt.split(" ")[1]
        hspy_meta["General"]["date"] = date
        hspy_meta["General"]["time"] = time
    if "Specimen" in all_meta:
        hspy_meta["Sample"]["description"] = all_meta["Specimen"]

    return hspy_meta


# x = file_reader(
#     "/home/arg6/data/app5_reader_files/cb76b58d-7906-423d-b681-3af9587e9e76.app5"
# )
# x = file_reader(
#     "/home/arg6/GitHub/rosettasciio/rsciio/tests/data/topspin/topspin_test_A.app5"
# )
