# -*- coding: utf-8 -*-
# Copyright 2007-2023 The HyperSpy developers
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

# The EMD format is a hdf5 standard proposed at Lawrence Berkeley
# National Lab (see https://emdatasets.com/ for more information).
# FEI later developed another EMD format, also based on the hdf5 standard. This
# reader first checked if the file have been saved by Velox (FEI EMD format)
# and use either the EMD class or the FEIEMDReader class to read the file.
# Writing file is only supported for EMD Berkeley file.

import json
import logging

import h5py

from rsciio._docstrings import (
    CHUNKS_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    RETURNS_DOC,
    SIGNAL_DOC,
)

from ._emd_ncem import read_emd_version

_logger = logging.getLogger(__name__)


def is_EMD_NCEM(file):
    """
    Parameters
    ----------
    file : h5py file handle
        DESCRIPTION.

    Returns
    -------
    bool
        DESCRIPTION.

    """

    def _is_EMD_NCEM(file):
        # the version can be defined in the root or the group
        if read_emd_version(file) != "":
            return True
        for key in file.keys():
            item = file[key]
            if isinstance(item, h5py.Group) and read_emd_version(item) != "":
                return True
        return False

    if isinstance(file, str):
        with h5py.File(file, "r") as f:
            return _is_EMD_NCEM(f)
    else:
        return _is_EMD_NCEM(file)


def is_EMD_Velox(file):
    """Function to check if the EMD file is an Velox file.

    Parameters
    ----------
    file : string or HDF5 file handle
        The name of the emd-file from which to load the signals. Standard
        file extension is 'emd'.

    Returns
    -------
    True if the file is a Velox file, otherwise False

    """

    def _is_EMD_velox(file):
        if "Version" in list(file.keys()):
            version = file.get("Version")
            v_dict = json.loads(version[0].decode("utf-8"))
            if v_dict["format"] in ["Velox", "DevelopersKit"]:
                return True
        return False

    if isinstance(file, str):
        with h5py.File(file, "r") as f:
            return _is_EMD_velox(f)
    else:
        return _is_EMD_velox(file)


def file_reader(
    filename,
    lazy=False,
    dataset_path=None,
    stack_group=None,
    select_type=None,
    first_frame=0,
    last_frame=None,
    sum_frames=True,
    sum_EDS_detectors=True,
    rebin_energy=1,
    SI_dtype=None,
    load_SI_image_stack=False,
):
    """
    Read EMD file, which can be an NCEM or a Velox variant of the EMD format.
    Also reads Direct Electron's DE5 format, which is read according to the
    NCEM specifications.

    Parameters
    ----------
    %s
    %s
    dataset_path : None, str or list of str, default=None
        NCEM only: Path of the dataset. If None, load all supported datasets,
        otherwise the specified dataset(s).
    stack_group : None, bool, default=None
        NCEM only: Stack datasets of groups with common path. Relevant for emd file
        version >= 0.5, where groups can be named ``group0000``, ``group0001``, etc.
    select_type : {None, 'image', 'single_spectrum', 'spectrum_image'}
        Velox only: specifies the type of data to load: if ``'image'`` is selected,
        only images (including EDS maps) are loaded, if ``'single_spectrum'`` is
        selected, only single spectra are loaded and if ``'spectrum_image'`` is
        selected, only the spectrum image will be loaded.
    first_frame : int, default=0
        Velox only: Select the start for the frame range of the EDS spectrum image
        to load.
    last_frame : int or None, default=None
        Velox only: Select the end for the frame range of the EDS spectrum image
        to load.
    sum_frames : bool, default=True
        Velox only: Load each individual EDS frame. The EDS spectrum image will
        be loaded with an extra navigation dimension corresponding to the frame
        index (time axis).
    sum_EDS_detectors : bool, default=True
        Velox only: Load the EDS signal as a sum over the signals from all EDS
        detectors (default) or, alternatively, load the signal of each individual
        EDS detector. In the latter case, a corresponding number of distinct
        EDS signals is returned.
    rebin_energy : int, default=1
        Velox only: Rebin the energy axis by given factor. Useful in combination
        with ``sum_frames=False`` to reduce the data size when reading the
        individual frames of the spectrum image.
    SI_dtype : numpy.dtype or None, default=None
        Velox only: Change the datatype of a spectrum image. Useful in combination
        with ``sum_frames=False`` to reduce the data size when reading the individual
        frames of the spectrum image. If ``None``, the dtype of the data in
        the emd file is used.
    load_SI_image_stack : bool, default=False
        Velox only: Allows loading the stack of STEM images acquired
        simultaneously with the EDS spectrum image. This option can be useful to
        monitor any specimen changes during the acquisition or to correct the
        spatial drift in the spectrum image by using the STEM images.

    %s
    """
    file = h5py.File(filename, "r")
    dictionaries = []
    try:
        if is_EMD_Velox(file):
            from ._emd_velox import FeiEMDReader

            _logger.debug("EMD file is a Velox variant.")
            emd_reader = FeiEMDReader(
                lazy=lazy,
                select_type=select_type,
                first_frame=first_frame,
                last_frame=last_frame,
                sum_frames=sum_frames,
                sum_EDS_detectors=sum_EDS_detectors,
                rebin_energy=rebin_energy,
                SI_dtype=SI_dtype,
                load_SI_image_stack=load_SI_image_stack,
            )
            emd_reader.read_file(file)
        elif is_EMD_NCEM(file):
            from ._emd_ncem import EMD_NCEM

            _logger.debug("EMD file is a Berkeley variant.")
            emd_reader = EMD_NCEM()
            emd_reader.read_file(
                file, lazy=lazy, dataset_path=dataset_path, stack_group=stack_group
            )
        else:
            raise IOError("The file is not a supported EMD file.")
    except Exception as e:
        raise e
    finally:
        if not lazy:
            file.close()

    dictionaries = emd_reader.dictionaries

    return dictionaries


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)


def file_writer(filename, signal, chunks=None, **kwds):
    """
    Write signal to EMD file. Only the specifications by the National Center
    for Electron Microscopy (NCEM) are supported.

    Parameters
    ----------
    %s
    %s
    %s
    **kwds : dict, optional
        Dictionary containing metadata, which will be written as attribute
        of the root group.
    """
    from ._emd_ncem import EMD_NCEM

    EMD_NCEM().write_file(filename, signal, chunks=chunks, **kwds)


file_writer.__doc__ %= (
    FILENAME_DOC.replace("read", "write to"),
    SIGNAL_DOC,
    CHUNKS_DOC,
)
