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

import logging

import mrcz as _mrcz
from packaging.version import Version

from rsciio._docstrings import (
    ENDIANESS_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    MMAP_DOC,
    RETURNS_DOC,
    SIGNAL_DOC,
)
from rsciio.utils.tools import DTBox

_logger = logging.getLogger(__name__)


_POP_FROM_HEADER = [
    "compressor",
    "MRCtype",
    "C3",
    "dimensions",
    "dtype",
    "extendedBytes",
    "gain",
    "maxImage",
    "minImage",
    "meanImage",
    "metaId",
    "packedBytes",
    "pixelsize",
    "pixelunits",
    "voltage",
]
# Hyperspy uses an unusual mixed Fortran- and C-ordering scheme
_READ_ORDER = [1, 2, 0]
_WRITE_ORDER = [0, 1, 2]


# API changes in mrcz 0.5
def _parse_metadata(metadata):
    if Version(_mrcz.__version__) < Version("0.5"):
        return metadata[0]
    else:
        return metadata


mapping = {
    "mrcz_header.voltage": ("Acquisition_instrument.TEM.beam_energy", _parse_metadata),
    "mrcz_header.gain": (
        "Signal.Noise_properties.Variance_linear_model.gain_factor",
        _parse_metadata,
    ),
    # There is no metadata field for spherical aberration
    #'mrcz_header.C3':
    # ("Acquisition_instrument.TEM.C3", lambda x: x),
}


def file_reader(filename, lazy=False, mmap_mode="c", endianess="<", **kwds):
    """
    File reader for the MRCZ format for tomographic data.

    Parameters
    ----------
    %s
    %s
    %s
    %s
    **kwds : dict, optional
        The keyword arguments are passed to :py:func:`mrcz.readMRC`.

    %s

    Examples
    --------
    >>> from rsciio.mrcz import file_reader
    >>> new_signal = file_reader('file.mrcz')
    """
    _logger.debug("Reading MRCZ file: %s" % filename)

    if mmap_mode != "c":
        # Note also that MRCZ does not support memory-mapping of compressed data.
        # Perhaps we could use the zarr package for that
        raise ValueError("MRCZ supports only C-ordering memory-maps")

    mrcz_endian = "le" if endianess == "<" else "be"
    data, mrcz_header = _mrcz.readMRC(
        filename, endian=mrcz_endian, useMemmap=lazy, pixelunits="nm", **kwds
    )

    # Create the axis objects for each axis
    names = ["y", "x", "z"]
    navigate = [False, False, True]
    axes = [
        {
            "size": data.shape[hsIndex],
            "index_in_array": hsIndex,
            "name": names[index],
            "scale": mrcz_header["pixelsize"][hsIndex],
            "offset": 0.0,
            "units": mrcz_header["pixelunits"],
            "navigate": nav,
        }
        for index, (hsIndex, nav) in enumerate(zip(_READ_ORDER, navigate))
    ]
    axes.insert(0, axes.pop(2))  # re-order the axes

    metadata = mrcz_header.copy()
    # Remove non-standard fields
    for popTarget in _POP_FROM_HEADER:
        metadata.pop(popTarget)

    dictionary = {
        "data": data,
        "axes": axes,
        "metadata": metadata,
        "original_metadata": {"mrcz_header": mrcz_header},
        "mapping": mapping,
    }

    return [
        dictionary,
    ]


file_reader.__doc__ %= (
    FILENAME_DOC,
    LAZY_DOC,
    MMAP_DOC.replace(
        "incompatible).",
        "incompatible). The MRCZ reader currently only supports C-ordering memory-maps.",
    ),
    ENDIANESS_DOC,
    RETURNS_DOC,
)


def file_writer(
    filename,
    signal,
    endianess="<",
    do_async=False,
    compressor=None,
    clevel=1,
    n_threads=None,
):
    """
    Write signal to MRCZ format.

    Parameters
    ----------
    %s
    %s
    %s
    do_async : bool, Default=False
        Currently supported within RosettaSciIO for writing only, this will
        save the file in a background thread and return immediately.
        Warning: there is no method currently implemented within RosettaSciIO
        to tell if an asychronous write has finished.
    compressor : {None, "zlib", "zstd", "lz4"}, Default=None
        The compression codec.
    clevel : int, Default=1
        The compression level, an ``int`` from 1 to 9.
    n_threads : int
        The number of threads to use for ``blosc`` compression. Defaults to
        the maximum number of virtual cores (including Intel Hyperthreading)
        on your system, which is recommended for best performance. If
        ``do_async = True`` you may wish to leave one thread free for the
        Python GIL.

    Notes
    -----
    The recommended compression codec is ``zstd`` (zStandard) with ``clevel=1`` for
    general use. If speed is critical, use ``lz4`` (LZ4) with ``clevel=9``. Integer data
    compresses more redably than floating-point data, and in general the histogram
    of values in the data reflects how compressible it is.

    To save files that are compatible with other programs that can use MRC such as
    GMS, IMOD, Relion, MotionCorr, etc. save with ``compressor=None``, extension ``.mrc``.
    JSON metadata will not be recognized by other MRC-supporting software but should
    not cause crashes.

    Examples
    --------
    >>> from rsciio.mrcz import file_writer
    >>> file_writer('file.mrcz', signal, do_async=True, compressor='zstd', clevel=1)
    """
    mrcz_endian = "le" if endianess == "<" else "be"

    md = DTBox(signal["metadata"], box_dots=True)

    # Get pixelsize and pixelunits from the axes
    pixelunits = signal["axes"][-1]["units"]
    pixelsize = [signal["axes"][I_]["scale"] for I_ in _WRITE_ORDER]

    # Strip out voltage from meta-data
    voltage = md.get("Acquisition_instrument.TEM.beam_energy")
    # There aren't hyperspy fields for spherical aberration or detector gain
    C3 = 0.0
    gain = md.get("Signal.Noise_properties.Variance_linear_model.gain_factor", 1.0)
    if do_async:
        _mrcz.asyncWriteMRC(
            signal["data"],
            filename,
            meta=md,
            endian=mrcz_endian,
            pixelsize=pixelsize,
            pixelunits=pixelunits,
            voltage=voltage,
            C3=C3,
            gain=gain,
            compressor=compressor,
            clevel=clevel,
            n_threads=n_threads,
        )
    else:
        _mrcz.writeMRC(
            signal["data"],
            filename,
            meta=md,
            endian=mrcz_endian,
            pixelsize=pixelsize,
            pixelunits=pixelunits,
            voltage=voltage,
            C3=C3,
            gain=gain,
            compressor=compressor,
            clevel=clevel,
            n_threads=n_threads,
        )


file_writer.__doc__ %= (
    FILENAME_DOC.replace("read", "write to"),
    SIGNAL_DOC,
    ENDIANESS_DOC,
)
