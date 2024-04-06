# -*- coding: utf-8 -*-
#
# Copyright 2022 The HyperSpy developers
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with any project and source this library is coupled.
# If not, see <https://www.gnu.org/licenses/#GPL>.

FILENAME_DOC = """filename : str, pathlib.Path
        Filename of the file to read or corresponding `pathlib.Path`.
    """


SIGNAL_DOC = """signal : dict
        Dictionary containing the signal object.
        Should contain the following fields:

        - 'data' – multidimensional numpy array
        - 'axes' – list of dictionaries describing the axes
          containing the fields 'name', 'units', 'index_in_array', and
          either 'size', 'offset', and 'scale' or a numpy array 'axis'
          containing the full axes vector
        - 'metadata' – dictionary containing the metadata tree
    """


LAZY_DOC = """lazy : bool, default=False
        Whether to open the file lazily or not.
    """


LAZY_UNSUPPORTED_DOC = """lazy : bool, default=False
        Lazy loading is not supported.
    """


CHUNKS_DOC = """chunks : tuple of int or None, default=None
        Define the chunking used for saving the dataset. If ``None``, calculates
        chunks for the signal, with preferably at least one chunk per signal
        space.
    """

CHUNKS_READ_DOC = """chunks : tuple of int or None, default="auto"
        The chunks used when reading the data lazily. This argument is passed
        to the ``chunks`` of the :py:func:`dask.array.from_array` function.
    """


SHOW_PROGRESSBAR_DOC = """show_progressbar : bool, default=True
        Whether to show the progressbar or not.
    """


ENCODING_DOC = """encoding : str, default="latin-1"
        The encoding used to read the content of the file. Different file
        encodings, such as ``"utf8"`` can be set via this argument.
    """


ENDIANESS_DOC = """endianess : str, default="<"
        ``"<"`` or ``">"``, depending on how the bits are written to
        the file.
    """


NAVIGATION_SHAPE = """navigation_shape : tuple or None, default=None
        Specify the shape of the navigation space. If ``None``, the navigation
        shape will be infer from metadata and if not possible, the data will be
        loaded as a stack with a navigation dimension equal to one.
    """


MMAP_DOC = """mmap_mode : {None, "r+", "r", "w+", "c"}, default=None
        Argument passed to :py:class:`numpy.memmap`. A memory-mapped array is
        stored on disk, and not directly loaded into memory.  However, it can be
        accessed and sliced like any ndarray.  Lazy loading does not support in-place writing (i.e lazy loading
        and the ``"r+"`` mode are incompatible).
        If ``None`` (default), the value is ``"r"`` when ``lazy=True``, otherwise
        it is ``"c"``.
    """

COMPRESSION_HDF5_DOC = """compression : None, 'gzip', 'szip', 'lzf', default='gzip'
        Compression can significantly increase the saving speed. If file size is not
        an issue, it can be disabled by setting ``compression=None``.
        RosettaSciIO uses h5py for reading and writing HDF5 files and, therefore,
        it supports all `compression filters supported by h5py
        <https://docs.h5py.org/en/stable/high/dataset.html#dataset-compression>`_.
        The default is ``'gzip'``. Also see notes below.
    """


COMPRESSION_HDF5_NOTES_DOC = """It is possible to enable other compression filters such as ``blosc`` by
    installing e.g. `hdf5plugin <https://github.com/silx-kit/hdf5plugin>`_.
    Similarly, the availability of ``'szip'`` depends on the HDF5 installation.
    If not available an error will be raised. Be aware that loading those
    files will require installing the package providing the compression filter
    and it may thus not be possible to load it on some platforms.
    Only ``compression=None`` and ``compression='gzip'`` are available on all
    platforms. For more details, see the `h5py documentation
    <https://docs.h5py.org/en/stable/faq.html#what-compression-processing-filters-are-supported>`_.
    """


DISTRIBUTED_DOC = """distributed : bool, default=False
        Whether to load the data using memory-mapping in a way that is
        compatible with dask-distributed.  This can sometimes improve
        performance when reading large files. And splitting the data
        loading/processing over multiple workers.
        """


RETURNS_DOC = """Returns
    -------

    list of dict
        List of dictionaries containing the following fields:

        - 'data' – multidimensional :py:class:`numpy.ndarray` or :py:class:`dask.array.Array`
        - 'axes' – list of dictionaries describing the axes
          containing the fields 'name', 'units', 'index_in_array', and
          either 'size', 'offset', and 'scale' or a numpy array 'axis'
          containing the full axes vector
        - 'metadata' – dictionary containing the parsed metadata
        - 'original_metadata' – dictionary containing the full metadata tree from the input file

        When the file contains several datasets, each dataset will be loaded as separate dictionary.
    """
