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

import sys
import xml.etree.cElementTree as ET

import h5py
import numpy as np
from tqdm import tqdm

from rsciio._docstrings import FILENAME_DOC, RETURNS_DOC, SHOW_PROGRESSBAR_DOC


def file_reader(filename, subset=None, dryrun=False, show_progressbar=True):
    """
    Read .app5 file format used by NanoMegas's Topspin software.

    .app5 files use the hdf5 file format, with metadata stored as
    binarized XML-style text strings.


    Parameters
    ----------
    %s
    subset: str or None
        h5py-style address. If given, only the subset of experiments
        located at this address will be imported. If none, all datasets
        will be imported.

    dryrun : bool
        If True, the .app5 files are quickly scanned without being loaded,
        and a summary is printed to the log. Default is False.

    %s
    %s

    Notes
    -----
    The Metadata textstrings used in app5 files change based on the
    ProcedureName, Topspin version, and local microscope setup. Because
    of this, the RosettaSciIo Metadata parser can fail when used to
    decode newer Topspin proceedures it was not tested against.
    """

    def is_guid(name):
        """Checks if a text string matches the shape of a guid"""
        lengths = [len(x) for x in name.split("-")]
        return lengths == [8, 4, 4, 4, 12]

    datasets_list = []
    task_list = []
    h5_file = h5py.File(filename, "r")
    if subset is not None:
        h5_file = h5_file[subset]

    # Generate task list by searching .app5 for GUID's
    group_names = [x for x in h5_file.keys()]
    if np.isin("Metadata", group_names):  # single experiment file.
        for name in h5_file.keys():
            if is_guid(name):
                task_list.append([name, "Metadata"])
    else:  # Multi-experimental file.
        for top_name in h5_file.keys():
            if is_guid(top_name):
                for name in h5_file[top_name].keys():
                    if is_guid(name):
                        address = top_name + "/" + name
                        meta = top_name + "/" + "Metadata"
                        task_list.append([address, meta])

    # Parse each dataset's metadata
    for address, meta_loc in task_list:
        name = address.split("/")[-1]
        metadata_string = h5_file[meta_loc][()].decode()
        meta_dict = _parse_app5_xml(metadata_string)
        if isinstance(h5_file[address], h5py.Group):
            # 4D-STEM
            h5_grp = h5_file[address]
            axes = _get_4D_axes(h5_grp)
        else:
            # 2D composite image
            axes = _get_2D_axes(metadata_string, name)
            shape = h5_file[address].shape
            axes[0]["size"] = shape[0]
            axes[1]["size"] = shape[1]
        datasets_list.append(
            {
                "axes": axes,
                "metadata": meta_dict,
                "original_metadata": metadata_string,
            }
        )

    if dryrun:
        filename = str(filename)
        message = "The following would be imported from " + filename + ":"
        for i, ds_dict in enumerate(datasets_list):
            shape = [x["size"] for x in ds_dict["axes"]]
            guid = task_list[i][0]
            name = ds_dict["metadata"]["Title"]
            message += "\n   - {}, {}\n      {}".format(name, shape, guid)
        sys.stdout.write(message)
        return []

    for i, ds_dict in enumerate(datasets_list):
        shape = [x["size"] for x in ds_dict["axes"]]
        address = task_list[i][0]
        if len(shape) == 2:
            datasets_list[i]["data"] = np.asanyarray(h5_file[address])
        elif len(shape) == 4:
            data = np.zeros([np.prod(shape[:2]), shape[2], shape[3]], dtype=np.uint16)
            key_count = len(h5_file[address].keys())
            for key in tqdm(
                h5_file[address],
                desc="Loading {}: ".format(address),
                total=key_count,
                disable=not show_progressbar,
            ):
                data[int(key)] = h5_file[address][key]["Data"]
            datasets_list[i]["data"] = data.reshape(shape)

    return datasets_list


file_reader.__doc__ %= (FILENAME_DOC, SHOW_PROGRESSBAR_DOC, RETURNS_DOC)


def _get_4D_axes(h5_grp):
    vuyx_axes = [
        {
            "name": "y",
            "units": "nm",
            "size": 0,
            "scale": 0,
            "offset": 0,
        },
        {
            "name": "x",
            "units": "nm",
            "size": 0,
            "scale": 0,
            "offset": 0,
        },
        {
            "name": "ky",
            "units": "mrads",
            "size": 0,
            "scale": 0,
            "offset": 0,
        },
        {
            "name": "kx",
            "units": "mrads",
            "size": 0,
            "scale": 0,
            "offset": 0,
        },
    ]

    # K-space axes
    k_space_id = [x for x in h5_grp.keys()][0]
    k_space = ET.fromstring(h5_grp[k_space_id]["Metadata"][()].decode())[0]
    shape = h5_grp[k_space_id]["Data"].shape
    vuyx_axes[2]["units"] = k_space[0][2][1].text
    vuyx_axes[2]["size"] = shape[0]
    vuyx_axes[2]["scale"] = k_space[0][1]
    vuyx_axes[2]["offset"] = k_space[0][0]
    vuyx_axes[3]["units"] = k_space[1][2][1].text
    vuyx_axes[3]["size"] = shape[0]
    vuyx_axes[3]["scale"] = k_space[1][1]
    vuyx_axes[3]["offset"] = k_space[1][0]

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
        # metadata file.
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
    vuyx_axes[0]["scale"] = np.around(dy, 12) * 1e9
    vuyx_axes[0]["offset"] = uy.min() * 1e9
    vuyx_axes[1]["size"] = lx
    vuyx_axes[1]["scale"] = np.around(dx, 12) * 1e9
    vuyx_axes[1]["offset"] = ux.min() * 1e9
    return vuyx_axes


def _get_2D_axes(metadata_string, name):
    root = ET.fromstring(metadata_string)
    # populate with default values
    elem = None
    yx_axes = [
        {
            "name": "y",
            "units": "nm",
            "size": 0,
            "scale": 0,
            "offset": 0,
        },
        {
            "name": "x",
            "units": "nm",
            "size": 0,
            "scale": 0,
            "offset": 0,
        },
    ]
    for value in root.findall(".//Value"):
        if value.attrib["Serializer"] == "ImageDataSerializer":
            if value[0].text == name:
                elem = value.find("Calibration")
                break
    if elem is None:
        raise UserWarning("Unable to parse 'ImageDataSerializer' in Metadata")
        return yx_axes
    yx_axes[0]["scale"] = float(elem.find("Y/Scale").text)
    yx_axes[0]["offset"] = float(elem.find("Y/Offset").text)
    yx_axes[1]["scale"] = float(elem.find("Y/Scale").text)
    yx_axes[1]["offset"] = float(elem.find("Y/Offset").text)
    return yx_axes


def _parse_app5_xml(metadata_string: str, recursion_limit: int = 6):
    """
    Converts 'MetaData' strings into nested python dictionaries.

    Parameters
    ----------
    metadata_string
        text representation of app5 metadata. Can be generated
        from an app5 file opened with `h5py` as *f* using:
            metadata_string = f['path/to/Metadata'][()].decode()

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

    which is equivalent to a python dictonary of the form:

        {'ScanRotation':float(0)}

    This function checks Value.attrib['Serializer'] for every name/value
    pair. If it's a python type, the pair are added to the dictionary.
    Otherwise, it is the start of a new Element leaf, and thus a new
    nested dictionary will be created.
    """

    def name_val_decode(element, recursion):
        recursion += 1
        if recursion > 6:
            return element.tag

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
                    out.update(name_val_decode(leaf, recursion))
        return out

    root = ET.fromstring(metadata_string)
    metadata_dict = {}
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
                    # everything else follows the same nested pattern
                    try:
                        branch_dict[leaf[0].text] = name_val_decode(leaf[1], 0)
                    except:
                        raise Warning(
                            "rsciio was unable to read"
                            + "{} from {} in the app5 metadata".format(
                                leaf[0].text, branch.tag
                            )
                        )
            metadata_dict[branch.tag] = branch_dict
        else:
            metadata_dict[branch.tag] = branch.text
    return metadata_dict
