# -*- coding: utf-8 -*-
# Copyright 2007-2026 The HyperSpy developers
#
# This file is part of RosettaSciIO.
#
# SPDX-License-Identifier: BSD-3-Clause
# See the LICENSE file at the root of the repository for the full license text.

"""Reader and writer for Universal Spectroscopy and Imaging Data (h5USID).

The h5USID layout (https://pycroscopy.github.io/USID/) is implemented here
directly with h5py, without depending on pyUSID. A USID *Main* dataset is a
2D ``(positions, spectroscopic values)`` dataset with:

- ``quantity`` and ``units`` string attributes;
- ``Position_Indices``, ``Position_Values``, ``Spectroscopic_Indices`` and
  ``Spectroscopic_Values`` attributes, which are object references to the
  ancillary datasets describing each dimension.

The ancillary position datasets have one column per dimension and the
spectroscopic ones one row per dimension. Both carry ``labels`` and
``units`` attributes, with one entry per dimension.
"""

import logging
import os
import time
from collections.abc import MutableMapping
from warnings import warn

import dask.array as da
import h5py
import numpy as np

from rsciio._docstrings import (
    FILENAME_DOC,
    LAZY_DOC,
    RETURNS_DOC,
    SIGNAL_DOC,
)

_logger = logging.getLogger(__name__)

_ANCILLARY = (
    "Position_Indices",
    "Position_Values",
    "Spectroscopic_Indices",
    "Spectroscopic_Values",
)
# Placeholders used by the writer, since signals carry no quantity/units
# for the data values themselves.
_DATASET_NAME = "Raw_Data"
_UNKNOWN_QUANTITY = "Unknown Quantity"
_UNKNOWN_UNITS = "Unknown Units"
_DEFAULT_UNITS = "a. u."
# Relative tolerance on the step between values for a dimension to be
# considered uniform
_UNIFORM_RTOL = 1e-3


def _decode(value):
    """Convert bytes (or arrays of bytes) read from HDF5 attributes to str."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in "SO":
        return [_decode(v) for v in value.tolist()]
    return value


def _attributes(h5_object):
    return {key: _decode(value) for key, value in h5_object.attrs.items()}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _ancillary_datasets(h5_dataset):
    """Return the four ancillary datasets of a USID Main dataset, or None."""
    if h5_dataset.ndim != 2:
        return None
    attrs = h5_dataset.attrs
    if not all(isinstance(_decode(attrs.get(k)), str) for k in ("quantity", "units")):
        return None
    try:
        pos_inds, pos_vals, spec_inds, spec_vals = (
            h5_dataset.file[attrs[name]] for name in _ANCILLARY
        )
    except (KeyError, TypeError, ValueError):
        return None
    n_pos, n_spec = h5_dataset.shape
    if not (
        pos_inds.shape == pos_vals.shape
        and spec_inds.shape == spec_vals.shape
        and pos_inds.ndim == spec_inds.ndim == 2
        and pos_inds.shape[0] == n_pos
        and spec_inds.shape[1] == n_spec
    ):
        return None
    return pos_inds, pos_vals, spec_inds, spec_vals


def _find_main_datasets(h5_file):
    """Return all USID Main datasets of a file, in HDF5 traversal order."""
    found = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and _ancillary_datasets(obj) is not None:
            found.append(obj)

    h5_file.visititems(visit)
    return found


def _read_dimensions(h5_inds, h5_vals, spectroscopic):
    """Read the dimensions described by a pair of ancillary datasets.

    Returns a list of ``(label, units, values)`` ordered from the slowest to
    the fastest varying dimension, such that the corresponding axis of the
    Main dataset reshapes (in C order) to the sizes of these dimensions.
    """
    indices, values = h5_inds[()], h5_vals[()]
    if spectroscopic:
        indices, values = indices.T, values.T
    # rows are now points and columns are dimensions
    labels, units = (
        np.atleast_1d(_decode(h5_inds.attrs[name])).astype(str).tolist()
        for name in ("labels", "units")
    )

    # The fastest varying dimension is the one that changes most often
    n_changes = np.count_nonzero(np.diff(indices, axis=0), axis=0)
    order = np.argsort(n_changes, kind="stable")
    indices, values = indices[:, order], values[:, order]
    sizes = [int(column.max()) + 1 for column in indices.T]

    # Only a full grid of indices can be reshaped to N dimensions
    grid = np.indices(sizes).reshape(len(sizes), -1).T
    if indices.shape != grid.shape or not np.array_equal(indices, grid):
        raise ValueError(
            f"The dimensions of {h5_inds.parent.name} do not form a regular "
            "grid and cannot be reshaped to N dimensions."
        )

    dimensions = []
    for i, column in enumerate(order):
        # values of this dimension at the points where all other indices are 0
        points = np.arange(sizes[i]) * int(np.prod(sizes[i + 1 :]))
        dimensions.append((labels[column], units[column], values[points, i]))
    return dimensions


def _uniform_axis(label, units, values, ignore_non_uniform_dims):
    """Axis dictionary of a dimension, which must be uniformly spaced."""
    steps = np.diff(values.astype(float))
    scale = steps.mean()
    if not np.allclose(steps, scale, rtol=_UNIFORM_RTOL, atol=0):
        if not ignore_non_uniform_dims:
            raise ValueError(
                f"Cannot load provided dataset. Parameter: {label} was varied "
                "non-uniformly. Supply keyword argument "
                '"ignore_non_uniform_dims=True" to ignore this error'
            )
        warn(f"Ignoring non-uniformity of dimension: {label}")
        return {
            "size": values.size,
            "name": label,
            "units": units,
            "scale": 1,
            "offset": 0,
        }
    return {
        "size": values.size,
        "name": label,
        "units": units,
        "scale": float(scale),
        "offset": float(values[0]),
    }


def _split_descriptor(descriptor):
    """Split a descriptor of the form ``"quantity (units)"``.

    Square brackets are also accepted around the units. Returns the
    ``(quantity, units)`` tuple, with empty units if there are none.
    """
    descriptor = descriptor.strip()
    for opening in "([":
        index = descriptor.rfind(opening)
        if index >= 0:
            units = descriptor[index:]
            for bracket in "()[]":
                units = units.replace(bracket, "")
            return descriptor[:index].strip(), units
    return descriptor, ""


def _group_parameters(h5_dataset):
    """Return the attributes of the enclosing Channel and Measurement groups."""
    parameters = {}
    channel = h5_dataset.parent
    if "Channel" in channel.name.split("/")[-1]:
        parameters.update(_attributes(channel))
        measurement = channel.parent
        if "Measurement" in measurement.name.split("/")[-1]:
            parameters.update(_attributes(measurement))
    return parameters


def _read_main_dataset(h5_dataset, lazy, ignore_non_uniform_dims):
    """Convert a USID Main dataset into a list of signal dictionaries.

    A compound (structured) dataset gives one signal per field.
    """
    pos_inds, pos_vals, spec_inds, spec_vals = _ancillary_datasets(h5_dataset)
    dimensions = _read_dimensions(pos_inds, pos_vals, spectroscopic=False)
    dimensions += _read_dimensions(spec_inds, spec_vals, spectroscopic=True)

    # Dimensions with a single value are squeezed out of the data
    axes = [
        _uniform_axis(label, units, values, ignore_non_uniform_dims)
        for label, units, values in dimensions
        if values.size > 1
    ]
    shape = tuple(axis["size"] for axis in axes)
    if lazy:
        data = da.from_array(h5_dataset).reshape(shape)
    else:
        data = h5_dataset[()].reshape(shape)

    attrs = h5_dataset.attrs
    quantity = _decode(attrs["quantity"]).strip()
    units = _decode(attrs["units"]).strip()
    general = {
        "original_filename": h5_dataset.file.filename,
        "title": h5_dataset.name.split("/")[-1],
    }
    original_metadata = {
        "dataset_path": h5_dataset.name,
        "original_file_type": "USID HDF5",
        "parameters": _group_parameters(h5_dataset),
    }

    def signal_dict(array, quantity, units, signal_type):
        return {
            "data": array,
            "axes": [axis.copy() for axis in axes],
            "metadata": {
                "General": dict(general),
                "Signal": {"signal_type": signal_type},
            },
            "original_metadata": {
                "quantity": quantity,
                "units": units,
                **original_metadata,
            },
        }

    if data.dtype.names is None:
        return [signal_dict(data, quantity, units, "")]
    # Each field of a compound dataset becomes a signal; the quantity of the
    # whole dataset is kept as the signal type.
    return [
        signal_dict(data[field], *_split_descriptor(field), quantity)
        for field in data.dtype.names
    ]


def file_reader(filename, lazy=False, dataset_path=None, ignore_non_uniform_dims=True):
    """
    Read an h5USID file.

    Parameters
    ----------
    %s
    %s
    dataset_path : str, default=None
        Absolute path of the USID Main dataset to read, for example
        ``"/Measurement_000/Channel_000/Raw_Data"``. If None, all USID Main
        datasets in the file are read.
    ignore_non_uniform_dims : bool, default=True
        Non-uniform dimensions are not supported. If True, such dimensions
        are loaded with a scale of 1 and an offset of 0 and a warning is
        raised. If False, a ``ValueError`` is raised instead.

    %s

    Notes
    -----
    When ``lazy=True``, the file is kept open so that the data can be read
    later.
    """
    filename = os.fspath(filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"No file found at: {filename}")
    if dataset_path is not None and not isinstance(dataset_path, str):
        raise TypeError("'dataset_path' should be a string")

    h5_file = h5py.File(filename, mode="r")
    try:
        if dataset_path is None:
            h5_datasets = _find_main_datasets(h5_file)
        else:
            h5_dataset = h5_file[dataset_path]
            if not isinstance(h5_dataset, h5py.Dataset) or (
                _ancillary_datasets(h5_dataset) is None
            ):
                raise ValueError(f"{dataset_path} is not a USID Main dataset.")
            h5_datasets = [h5_dataset]
        signals = []
        for h5_dataset in h5_datasets:
            signals += _read_main_dataset(h5_dataset, lazy, ignore_non_uniform_dims)
    except Exception:
        h5_file.close()
        raise

    if not lazy:
        h5_file.close()
    return signals


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _flatten(mapping, prefix=""):
    """Flatten nested mappings, joining the keys with ``"-"``."""
    flat = {}
    for key, value in mapping.items():
        key = f"{prefix}-{key}" if prefix else key
        if isinstance(value, MutableMapping):
            flat.update(_flatten(value, key))
        else:
            flat[key] = value
    return flat


def _write_attributes(h5_object, attributes):
    """Write attributes, skipping those which cannot be stored in HDF5."""
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and any(isinstance(v, str) for v in value):
            value = np.array([str(v).encode("utf-8") for v in value], dtype="S")
        try:
            h5_object.attrs[key.strip()] = value
        except (TypeError, ValueError):
            _logger.warning(
                f"The metadata item {key!r} of type {type(value).__name__} "
                "cannot be written to an h5USID file and is skipped."
            )


def _write_timestamp(h5_object):
    h5_object.attrs["timestamp"] = time.strftime("%Y_%m_%d-%H_%M_%S")


def _create_indexed_group(h5_parent, base_name):
    """Create the group ``<base_name>_NNN`` following the existing ones."""
    prefix = f"{base_name}_"
    indices = [
        int(name[len(prefix) :])
        for name, obj in h5_parent.items()
        if isinstance(obj, h5py.Group)
        and name.startswith(prefix)
        and name[len(prefix) :].isdigit()
    ]
    group = h5_parent.create_group(f"{prefix}{max(indices, default=-1) + 1:03d}")
    _write_timestamp(group)
    return group


def _text_or_default(value, default):
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _dimensions_from_axes(axes, shape, default_name):
    """Return the ``(label, units, values)`` of each axis, slowest first.

    The sizes are taken from ``shape``, the matching part of the data shape.
    Unnamed dimensions are numbered from the fastest one.
    """
    if not axes:
        # USID always needs at least one dimension on each side
        return [("Arb", _DEFAULT_UNITS, np.zeros(1))]
    if len(axes) != len(shape):
        raise ValueError(
            f"The number of axes ({len(axes)}) does not match the number of "
            f"data dimensions ({len(shape)})."
        )
    n = len(axes)
    return [
        (
            _text_or_default(axis.get("name"), f"{default_name}_{n - 1 - i}"),
            _text_or_default(axis.get("units"), _DEFAULT_UNITS),
            np.arange(size) * axis["scale"] + axis["offset"],
        )
        for i, (axis, size) in enumerate(zip(axes, shape))
    ]


def _write_ancillary(h5_group, base_name, dimensions, spectroscopic):
    """Write the indices and values datasets describing ``dimensions``."""
    sizes = [len(values) for _, _, values in dimensions]
    # One row per dimension, the last (fastest) one varying fastest
    indices = np.indices(sizes).reshape(len(sizes), -1)
    values = np.stack(
        [
            dim_values[dim_indices]
            for (_, _, dim_values), dim_indices in zip(dimensions, indices)
        ]
    )
    if not spectroscopic:
        indices, values = indices.T, values.T

    attributes = {
        "labels": [label for label, _, _ in dimensions],
        "units": [units for _, units, _ in dimensions],
        "type": [0] * len(dimensions),
    }
    datasets = []
    for kind, array, dtype in (
        ("Indices", indices, np.uint32),
        ("Values", values, np.float32),
    ):
        dataset = h5_group.create_dataset(
            f"{base_name}_{kind}", data=array.astype(dtype)
        )
        _write_attributes(dataset, attributes)
        datasets.append(dataset)
    return datasets


def file_writer(filename, signal, **kwds):
    """
    Write a signal to an h5USID file.

    If the file already exists, the signal is added to it as a new
    ``Measurement`` group; the existing content is kept.

    Parameters
    ----------
    %s
    %s
    **kwds : dict, optional
        Passed to :meth:`h5py.Group.create_dataset` when creating the
        Main dataset, for example ``compression="gzip"``.

    Notes
    -----
    The metadata and original metadata are flattened and written as
    attributes of the ``Measurement`` group. Models and other objects linked
    to the signal are not written.
    """
    data = signal["data"]
    nav_axes = [axis for axis in signal["axes"] if axis["navigate"]]
    sig_axes = [axis for axis in signal["axes"] if not axis["navigate"]]
    nav_shape = data.shape[: len(nav_axes)]
    sig_shape = data.shape[len(nav_axes) :]

    pos_dims = _dimensions_from_axes(nav_axes, nav_shape, "Pos_Dim")
    spec_dims = _dimensions_from_axes(sig_axes, sig_shape, "Spec_Dim")
    # USID stores the data as a 2D (positions, spectroscopic values) matrix
    data_2d = data.reshape(int(np.prod(nav_shape)), int(np.prod(sig_shape)))

    parameters = _flatten(signal["metadata"])
    parameters.update(_flatten(signal["original_metadata"], "Original"))

    new_file = not os.path.exists(filename)
    with h5py.File(filename, mode="a") as h5_file:
        if new_file:
            h5_file.attrs["data_type"] = _DATASET_NAME
            _write_timestamp(h5_file)
        measurement = _create_indexed_group(h5_file, "Measurement")
        _write_attributes(measurement, parameters)
        channel = _create_indexed_group(measurement, "Channel")

        ancillary = _write_ancillary(channel, "Position", pos_dims, False)
        ancillary += _write_ancillary(channel, "Spectroscopic", spec_dims, True)

        main = channel.create_dataset(
            _DATASET_NAME, shape=data_2d.shape, dtype=data_2d.dtype, **kwds
        )
        if isinstance(data_2d, da.Array):
            da.store(data_2d, main)
        else:
            main[...] = data_2d
        main.attrs["quantity"] = _UNKNOWN_QUANTITY
        main.attrs["units"] = _UNKNOWN_UNITS
        _write_timestamp(main)
        for name, dataset in zip(_ANCILLARY, ancillary):
            main.attrs[name] = dataset.ref


file_writer.__doc__ %= (FILENAME_DOC.replace("read", "write to"), SIGNAL_DOC)
