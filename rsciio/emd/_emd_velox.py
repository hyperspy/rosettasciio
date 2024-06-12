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
import os
import time
from datetime import datetime

import dask.array as da
import numpy as np
from dateutil import tz

from rsciio.utils.elements import atomic_number2name
from rsciio.utils.hdf5 import (
    _get_keys_from_group,
    _parse_metadata,
    _parse_sub_data_group_metadata,
)
from rsciio.utils.tools import _UREG, convert_units

_logger = logging.getLogger(__name__)


def _parse_json(v, encoding="utf-8"):
    return json.loads(v.decode(encoding))


def _get_detector_metadata_dict(om, detector_name):
    detectors_dict = om["Detectors"]
    # find detector dict from the detector_name
    for key in detectors_dict:
        if detectors_dict[key]["DetectorName"] == detector_name:
            return detectors_dict[key]
    return None


PRUNE_WARNING = (
    "No spectrum stream is present in the file and the "
    "spectrum images are saved in a proprietary format, "
    "which is not supported by RosettaSciIO. This is "
    "because it has been 'pruned' or saved a different "
    "software than Velox, e.g. bcf to emd converter. "
    "If you want to open this data don't prune the "
    "file or read bcf file directly (in case the bcf "
    "to emd converter was used)."
)


class FeiEMDReader(object):
    """
    Class for reading FEI electron microscopy datasets.

    The :class:`~.FeiEMDReader` reads EMD files saved by the FEI Velox
    software package.

    Attributes
    ----------
    dictionaries: list
        List of dictionaries which are passed to the file_reader.
    im_type : string
        String specifying whether the data is an image, spectrum or
        spectrum image.

    """

    def __init__(
        self,
        filename=None,
        select_type=None,
        first_frame=0,
        last_frame=None,
        sum_frames=True,
        sum_EDS_detectors=True,
        rebin_energy=1,
        SI_dtype=None,
        load_SI_image_stack=False,
        lazy=False,
    ):
        # TODO: Finish lazy implementation using the `FrameLocationTable`
        # Parallelise streams reading
        self.filename = filename
        self.select_type = select_type
        self.dictionaries = []
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.sum_frames = sum_frames
        self.sum_EDS_detectors = sum_EDS_detectors
        self.rebin_energy = rebin_energy
        self.SI_data_dtype = SI_dtype
        self.load_SI_image_stack = load_SI_image_stack
        self.lazy = lazy
        self.detector_name = None
        self.original_metadata = {}
        # UUID: label mapping
        self._map_label_dict = {}

    def read_file(self, f):
        self.filename = f.filename
        self.version = _parse_json(f["Version"][0])["version"]
        _logger.info(f"EMD file version: {self.version}")
        self.d_grp = f.get("Data")
        self._check_im_type()
        for key in ["Displays", "Operations", "SharedProperties", "Features"]:
            # In Velox emd v11, the "Operation" group is removed:
            # 'operation settings' are moved to \SharedProperties
            # \Features link to \SharedProperties\DataReference
            # in version <11 \Features linked to \Operations
            if key in f.keys():
                self._parse_metadata_group(f.get(key), key)
        if self.im_type == "SpectrumStream":
            self._parse_image_display(f)
        self._read_data(self.select_type)

    def _read_data(self, select_type):
        self.load_images = self.load_SI = self.load_single_spectrum = True
        if select_type == "single_spectrum":
            self.load_images = self.load_SI = False
        elif select_type == "images":
            self.load_SI = self.load_single_spectrum = False
        elif select_type == "spectrum_image":
            self.load_images = self.load_single_spectrum = False
        elif select_type is None:
            pass
        else:
            raise ValueError(
                "`select_type` parameter takes only: `None`, "
                "'single_spectrum', 'images' or 'spectrum_image'."
            )

        if self.im_type == "Image":
            _logger.info("Reading the images.")
            self._read_images()
        elif self.im_type == "Spectrum":
            self._read_single_spectrum()
            self._read_images()
        elif self.im_type == "SpectrumStream":
            self._read_single_spectrum()
            _logger.info("Reading the spectrum image.")
            t0 = time.time()
            self._read_images()
            t1 = time.time()
            self._read_spectrum_stream()
            t2 = time.time()
            _logger.info("Time to load images: {} s.".format(t1 - t0))
            _logger.info("Time to load spectrum image: {} s.".format(t2 - t1))

    def _check_im_type(self):
        if "Image" in self.d_grp:
            if "SpectrumImage" in self.d_grp:
                self.im_type = "SpectrumStream"
            else:
                self.im_type = "Image"
        else:
            self.im_type = "Spectrum"

    def _read_single_spectrum(self):
        if not self.load_single_spectrum:
            return
        spectrum_grp = self.d_grp.get("Spectrum")
        if spectrum_grp is None:
            return  # No spectra in the file
        self.detector_name = "EDS"
        for spectrum_sub_group_key in _get_keys_from_group(spectrum_grp):
            self.dictionaries.append(
                self._read_spectrum(spectrum_grp, spectrum_sub_group_key)
            )

    def _read_spectrum(self, spectrum_group, spectrum_sub_group_key):
        spectrum_sub_group = spectrum_group[spectrum_sub_group_key]
        dataset = spectrum_sub_group["Data"]
        if self.lazy:
            data = da.from_array(dataset, chunks=dataset.chunks).T
        else:
            data = dataset[:].T
        original_metadata = _parse_metadata(spectrum_group, spectrum_sub_group_key)
        original_metadata.update(self.original_metadata)

        # Can be used in more recent version of velox emd files
        self.detector_information = self._get_detector_information(original_metadata)

        dispersion, offset, unit = self._get_dispersion_offset(original_metadata)
        axes = []
        if len(data.shape) == 2:
            if data.shape[0] == 1:
                # squeeze
                data = data[0, :]
            else:
                axes = [
                    {
                        "name": "Stack",
                        "offset": 0,
                        "scale": 1,
                        "size": data.shape[0],
                        "navigate": True,
                    }
                ]
        axes.append(
            {
                "name": "Energy",
                "offset": offset,
                "scale": dispersion,
                "size": data.shape[-1],
                "units": "keV",
                "navigate": False,
            },
        )

        md = self._get_metadata_dict(original_metadata)
        md["Signal"]["signal_type"] = "EDS_TEM"

        return {
            "data": data,
            "axes": axes,
            "metadata": md,
            "original_metadata": original_metadata,
            "mapping": self._get_mapping(),
        }

    def _read_images(self):
        # We need to read the image to get the shape of the spectrum image
        if not self.load_images and not self.load_SI:
            return
        # Get the image data group
        image_group = self.d_grp.get("Image")
        if image_group is None:
            return  # No images in the file
        # Get all the subgroup of the image data group and read the image for
        # each of them
        for image_sub_group_key in _get_keys_from_group(image_group):
            image = self._read_image(image_group, image_sub_group_key)
            if not self.load_images:
                # If we don't want to load the images, we stop here
                return
            self.dictionaries.append(image)

    def _read_image(self, image_group, image_sub_group_key):
        """Return a dictionary ready to parse of return to io module"""
        image_sub_group = image_group[image_sub_group_key]
        original_metadata = _parse_metadata(image_group, image_sub_group_key)
        original_metadata.update(self.original_metadata)

        # Can be used in more recent version of velox emd files
        self.detector_information = self._get_detector_information(original_metadata)
        try:
            self.detector_name = self._get_detector_name(image_sub_group_key)
        except KeyError:
            # File version >= 11 doesn't have the "Operations" group anymore
            if self.detector_information is not None:
                self.detector_name = self.detector_information["DetectorName"]

        read_stack = self.load_SI_image_stack or self.im_type == "Image"
        h5data = image_sub_group["Data"]
        # Get the scanning area shape of the SI from the images
        self.spatial_shape = h5data.shape[:-1]
        # For Velox FFT data, dtype must be specified and lazy is not
        # supported due to special dtype. The data is loaded as-is; to get
        # a traditional view the negative half must be created and the data
        # must be re-centered
        # Similar story for DPC signal
        fft_dtype = [
            [("realFloatHalfEven", "<f4"), ("imagFloatHalfEven", "<f4")],
            [("realFloatHalfOdd", "<f4"), ("imagFloatHalfOdd", "<f4")],
        ]
        dpc_dtype = [("realFloat", "<f4"), ("imagFloat", "<f4")]
        if h5data.dtype in fft_dtype or h5data.dtype == dpc_dtype:
            _logger.debug("Found an FFT or DPC, loading as Complex2DSignal")
            real = h5data.dtype.descr[0][0]
            imag = h5data.dtype.descr[1][0]
            if self.lazy:
                data = da.from_array(h5data, chunks=h5data.chunks)
                data = data[real] + 1j * data[imag]
                data = da.transpose(data, axes=[2, 0, 1])
            else:
                data = np.empty(h5data.shape, h5data.dtype)
                h5data.read_direct(data)
                data = data[real] + 1j * data[imag]
                # Set the axes in frame, y, x order
                data = np.rollaxis(data, axis=2)
        else:
            if self.lazy:
                data = da.transpose(
                    da.from_array(h5data, chunks=h5data.chunks), axes=[2, 0, 1]
                )
            else:
                # Workaround for a h5py bug https://github.com/h5py/h5py/issues/977
                # Change back to standard API once issue #977 is fixed.
                # Preallocate the numpy array and use read_direct method, which is
                # much faster in case of chunked data.
                # Do not specify dtype in np.empty, slows down substantially!
                data = np.empty(h5data.shape)
                h5data.read_direct(data)
                # Set the axes in frame, y, x order
                data = np.rollaxis(data, axis=2)

        pix_scale = original_metadata["BinaryResult"].get(
            "PixelSize", {"height": 1.0, "width": 1.0}
        )
        offsets = original_metadata["BinaryResult"].get("Offset", {"x": 0.0, "y": 0.0})
        original_units = original_metadata["BinaryResult"].get("PixelUnitX", "")

        axes = []
        # stack of images
        if not read_stack:
            data = data[0:1, ...]

        if data.shape[0] == 1:
            # Squeeze
            data = data[0, ...]
            i = 0
        else:
            if "FrameTime" in original_metadata["Scan"]:
                frame_time = original_metadata["Scan"]["FrameTime"]
            else:
                _logger.debug("No Frametime found, likely TEM image stack")
                det_ind = original_metadata["BinaryResult"]["DetectorIndex"]
                frame_time = original_metadata["Detectors"][f"Detector-{det_ind}"][
                    "ExposureTime"
                ]
            frame_time, time_unit = self._convert_scale_units(
                frame_time, "s", 2 * data.shape[0]
            )
            axes.append(
                {
                    "index_in_array": 0,
                    "name": "Time",
                    "offset": 0,
                    "scale": frame_time,
                    "size": data.shape[0],
                    "units": time_unit,
                    "navigate": True,
                }
            )
            i = 1
        scale_x, x_unit = self._convert_scale_units(
            pix_scale["width"], original_units, data.shape[i + 1]
        )
        # to avoid mismatching units between x and y axis, use the same unit as x
        # x is chosen as reference, because scalebar used (usually) the horizonal axis
        # and the units conversion is tuned to get decent scale bar
        scale_y = convert_units(float(pix_scale["height"]), original_units, x_unit)
        # Because "axes" only allows one common unit for offset and scale,
        # offset_x, offset_y is converted to the same unit as x_unit
        offset_x = convert_units(float(offsets["x"]), original_units, x_unit)
        offset_y = convert_units(float(offsets["y"]), original_units, x_unit)

        axes.extend(
            [
                {
                    "index_in_array": i,
                    "name": "y",
                    "offset": offset_y,
                    "scale": scale_y,
                    "size": data.shape[i],
                    "units": x_unit,
                    "navigate": False,
                },
                {
                    "index_in_array": i + 1,
                    "name": "x",
                    "offset": offset_x,
                    "scale": scale_x,
                    "size": data.shape[i + 1],
                    "units": x_unit,
                    "navigate": False,
                },
            ]
        )

        md = self._get_metadata_dict(original_metadata)
        if self.detector_name is not None:
            original_metadata["DetectorMetadata"] = _get_detector_metadata_dict(
                original_metadata, self.detector_name
            )
        if image_sub_group_key in self._map_label_dict:
            md["General"]["title"] = self._map_label_dict[image_sub_group_key]

        return {
            "data": data,
            "axes": axes,
            "metadata": md,
            "original_metadata": original_metadata,
            "mapping": self._get_mapping(
                map_selected_element=False, parse_individual_EDS_detector_metadata=False
            ),
        }

    def _get_detector_name(self, key):
        def iDPC_or_dDPC(metadata):
            return "iDPC" if metadata == "true" else "dDPC"

        om = self.original_metadata["Operations"]
        keys = [
            "CameraInputOperation",
            "StemInputOperation",
            "SurfaceReconstructionOperation",
            "MathematicsOperation",
            "DpcOperation",
            "IntegrationOperation",
            "FftOperation",
        ]

        for k in keys:
            if k in om.keys() and k == keys[0]:
                for metadata in om[k].items():
                    # Find the metadata group matching the key in the dataPath
                    if key in metadata[1]["dataPath"]:
                        return metadata[1]["cameraName"]
            if k in om.keys() and k == keys[1]:
                for metadata in om[k].items():
                    # Find the metadata group matching the key in the dataPath
                    if key in metadata[1]["dataPath"]:
                        return metadata[1]["detector"]
            if k in om.keys() and k == keys[2]:
                for metadata in om[k].items():
                    # Look first for the key in the unfilteredDataPath
                    if "unfilteredDataPath" in metadata[1].keys() and (
                        key in metadata[1]["unfilteredDataPath"]
                    ):
                        return iDPC_or_dDPC(metadata[1]["integrationMode"])
                    # Then look for the key in the DataPath
                    if key in metadata[1]["dataPath"]:
                        detector_name = iDPC_or_dDPC(metadata[1]["integrationMode"])
                        if metadata[1]["enableFilter"] == "true":
                            detector_name = "Filtered {}".format(detector_name)
                        return detector_name
            if k in om.keys() and k == keys[3]:
                for metadata in om[k].items():
                    if key in metadata[1]["dataPath"]:
                        if metadata[1]["outputs"][0]["inputIndex"] == "0":
                            return "A-C"
                        elif metadata[1]["outputs"][0]["inputIndex"] == "1":
                            return "B-D"
            if k in om.keys() and k == keys[4]:
                for metadata in om[k].items():
                    if key in metadata[1]["dataPath"]:
                        return "DPC"
            if k in om.keys() and k == keys[5]:
                for metadata in om[k].items():
                    if key in metadata[1]["dataPath"]:
                        return "DCFI"
            if k in om.keys() and k == keys[6]:
                for metadata in om[k].items():
                    if key in metadata[1]["imageOutputPath"]:
                        return "Half FFT"
        return "Unrecognized_image_signal"

    def _get_detector_information(self, om):
        # if the `BinaryResult/Detector` is not available, there should be only
        # one detector in `Detectors`:
        # e.g. original_metadata['Detectors']['Detector-0']
        if "BinaryResult" in om.keys():
            detector_index = om["BinaryResult"].get("DetectorIndex")
        else:
            detector_index = 0
        if detector_index is not None:
            return om["Detectors"]["Detector-{}".format(detector_index)]
        else:
            return None

    def _parse_frame_time(self, original_metadata, factor=1):
        try:
            frame_time = original_metadata["Scan"]["FrameTime"]
            time_unit = "s"
        except KeyError:
            frame_time, time_unit = None, None

        frame_time, time_unit = self._convert_scale_units(frame_time, time_unit, factor)
        return frame_time, time_unit

    def _parse_image_display(self, f):
        if int(self.version) >= 11:
            # - /Displays/ImageDisplay contains the list of all the image displays.
            #   A EDS Map is just an image display.
            # - These entries contain a json encoded dictionary that contains
            #   'data', 'id', 'settings' and 'title'.
            # - The 'id' is the name of the element. 'data' is pointing to the
            #   data reference in SharedProperties/ImageSeriesDataReference/<UUID>
            #   which in turn is pointing to the /Data/Image/<UUID> where the image
            #   data is located.
            om_image_display = self.original_metadata["Displays"]["ImageDisplay"]
            self._map_label_dict = {}
            for v in om_image_display.values():
                if "data" in v.keys():
                    data_key = _parse_json(f.get(v["data"])[0])["dataPath"]
                    self._map_label_dict[data_key.split("/")[-1]] = v["id"]

        else:
            image_display_group = f.get("Presentation/Displays/ImageDisplay")
            key_list = _get_keys_from_group(image_display_group)

            for key in key_list:
                v = _parse_json(image_display_group[key][0])
                data_key = v["dataPath"].split("/")[-1]  # key in data group
                self._map_label_dict[data_key] = v["display"]["label"]

    def _parse_metadata_group(self, group, group_name):
        d = {}
        try:
            for group_key in _get_keys_from_group(group):
                subgroup = group.get(group_key)
                if hasattr(subgroup, "keys"):
                    sub_dict = {}
                    for subgroup_key in _get_keys_from_group(subgroup):
                        v = _parse_json(subgroup[subgroup_key][0])
                        sub_dict[subgroup_key] = v
                else:
                    sub_dict = _parse_json(subgroup[0])
                d[group_key] = sub_dict
        except IndexError:
            _logger.warning("Some metadata can't be read.")
        self.original_metadata.update({group_name: d})

    def _read_spectrum_stream(self):
        if not self.load_SI:
            return
        self.detector_name = "EDS"
        # Try to read the number of frames from Data/SpectrumImage
        try:
            sig = self.d_grp["SpectrumImage"]
            self.number_of_frames = int(
                _parse_json(sig[next(iter(sig))]["SpectrumImageSettings"][0])[
                    "endFramePosition"
                ]
            )
        except Exception:
            _logger.exception(
                "Failed to read the number of frames from Data/SpectrumImage"
            )
            self.number_of_frames = None
        if self.last_frame is None:
            self.last_frame = self.number_of_frames
        elif self.number_of_frames and self.last_frame > self.number_of_frames:
            raise ValueError(
                "The `last_frame` cannot be greater than"
                " the number of frames, %i for this file." % self.number_of_frames
            )

        spectrum_stream_group = self.d_grp.get("SpectrumStream")
        if spectrum_stream_group is None:  # pragma: no cover
            # "Pruned" file, EDS SI data are in the
            # "SpectrumImage" group
            _logger.warning(PRUNE_WARNING)
            return

        subgroup_keys = _get_keys_from_group(spectrum_stream_group)
        if len(subgroup_keys) == 0:
            # "Pruned" file: in Velox emd v11, the "SpectrumStream"
            # group exists but it is empty
            _logger.warning(PRUNE_WARNING)
            return

        def _read_stream(key):
            stream = FeiSpectrumStream(spectrum_stream_group[key], self)
            return stream

        if self.sum_EDS_detectors:
            if len(subgroup_keys) == 1:
                _logger.warning("The file contains only one spectrum stream")
            # Read the first stream
            s0 = _read_stream(subgroup_keys[0])
            streams = [s0]
            # add other stream streams
            if len(subgroup_keys) > 1:
                for key in subgroup_keys[1:]:
                    stream_data = spectrum_stream_group[key]["Data"][:].T[0]
                    if self.lazy:
                        s0.spectrum_image = (
                            s0.spectrum_image
                            + s0.stream_to_sparse_array(stream_data=stream_data)
                        )
                    else:
                        s0.stream_to_array(
                            stream_data=stream_data, spectrum_image=s0.spectrum_image
                        )
        else:
            streams = [_read_stream(key) for key in subgroup_keys]
        if self.lazy:
            for stream in streams:
                sa = stream.spectrum_image.astype(self.SI_data_dtype)
                stream.spectrum_image = sa

        spectrum_image_shape = streams[0].shape
        original_metadata = streams[0].original_metadata
        original_metadata.update(self.original_metadata)

        # Can be used in more recent version of velox emd files
        self.detector_information = self._get_detector_information(original_metadata)

        pixel_size, offsets, original_units = streams[0].get_pixelsize_offset_unit()
        dispersion, offset, unit = self._get_dispersion_offset(original_metadata)

        scale_x, x_unit = self._convert_scale_units(
            pixel_size["width"], original_units, spectrum_image_shape[1]
        )
        # to avoid mismatching units between x and y axis, use the same unit as x
        # x is chosen as reference, because scalebar used (usually) the horizonal axis
        # and the units conversion is tuned to get decent scale bar
        scale_y = convert_units(float(pixel_size["height"]), original_units, x_unit)
        # Because "axes" only allows one common unit for offset and scale,
        # offset_x, offset_y is converted to the same unit as x_unit
        offset_x = convert_units(float(offsets["x"]), original_units, x_unit)
        offset_y = convert_units(float(offsets["y"]), original_units, x_unit)

        i = 0
        axes = []
        # add a supplementary axes when we import all frames individualy
        if not self.sum_frames:
            frame_time, time_unit = self._parse_frame_time(
                original_metadata, spectrum_image_shape[i]
            )
            axes.append(
                {
                    "index_in_array": i,
                    "name": "Time",
                    "offset": 0,
                    "scale": frame_time,
                    "size": spectrum_image_shape[i],
                    "units": time_unit,
                    "navigate": True,
                }
            )
            i = 1
        axes.extend(
            [
                {
                    "index_in_array": i,
                    "name": "y",
                    "offset": offset_y,
                    "scale": scale_y,
                    "size": spectrum_image_shape[i],
                    "units": x_unit,
                    "navigate": True,
                },
                {
                    "index_in_array": i + 1,
                    "name": "x",
                    "offset": offset_x,
                    "scale": scale_x,
                    "size": spectrum_image_shape[i + 1],
                    "units": x_unit,
                    "navigate": True,
                },
                {
                    "index_in_array": i + 2,
                    "name": "X-ray energy",
                    "offset": offset,
                    "scale": dispersion,
                    "size": spectrum_image_shape[i + 2],
                    "units": unit,
                    "navigate": False,
                },
            ]
        )

        md = self._get_metadata_dict(original_metadata)
        md["Signal"]["signal_type"] = "EDS_TEM"

        for stream in streams:
            original_metadata = stream.original_metadata
            original_metadata.update(self.original_metadata)
            self.dictionaries.append(
                {
                    "data": stream.spectrum_image,
                    "axes": axes,
                    "metadata": md,
                    "original_metadata": original_metadata,
                    "mapping": self._get_mapping(
                        parse_individual_EDS_detector_metadata=not self.sum_frames
                    ),
                }
            )

    def _get_dispersion_offset(self, original_metadata):
        try:
            for detectorname, detector in original_metadata["Detectors"].items():
                if (
                    original_metadata["BinaryResult"]["Detector"]
                    in detector["DetectorName"]
                ):
                    dispersion = (
                        float(detector["Dispersion"]) / 1000.0 * self.rebin_energy
                    )
                    offset = float(detector["OffsetEnergy"]) / 1000.0
                    return dispersion, offset, "keV"
        except KeyError:
            _logger.warning("The spectrum calibration can't be loaded.")

        return 1, 0, None

    def _convert_scale_units(self, value, units, factor=1):
        if units is None:
            return value, units
        factor /= 2
        v = float(value) * _UREG(units)
        converted_v = (factor * v).to_compact()
        converted_value = float(converted_v.magnitude / factor)
        converted_units = "{:~}".format(converted_v.units)
        return converted_value, converted_units

    def _get_metadata_dict(self, om):
        meta_gen = {}
        meta_gen["original_filename"] = os.path.split(self.filename)[1]
        if self.detector_name is not None:
            meta_gen["title"] = self.detector_name
        # We have only one entry in the original_metadata, so we can't use
        # the mapping of the original_metadata to set the date and time in
        # the metadata: need to set it manually here
        try:
            if "AcquisitionStartDatetime" in om["Acquisition"].keys():
                unix_time = om["Acquisition"]["AcquisitionStartDatetime"]["DateTime"]
            # Workaround when the 'AcquisitionStartDatetime' key is missing
            # This timestamp corresponds to when the data is stored
            elif (
                not isinstance(om["CustomProperties"], str)
                and "Detectors[BM-Ceta].TimeStamp" in om["CustomProperties"].keys()
            ):
                unix_time = (
                    float(
                        om["CustomProperties"]["Detectors[BM-Ceta].TimeStamp"]["value"]
                    )
                    / 1e6
                )
            date, time = self._convert_datetime(unix_time).split("T")
            meta_gen["date"] = date
            meta_gen["time"] = time
            meta_gen["time_zone"] = self._get_local_time_zone()
        except UnboundLocalError:
            # Error seems to come from h5py, covered in the test suite
            # Added in https://github.com/hyperspy/hyperspy/pull/1831
            pass

        meta_sig = {}
        meta_sig["signal_type"] = ""

        return {"General": meta_gen, "Signal": meta_sig}

    def _get_mapping(
        self, map_selected_element=True, parse_individual_EDS_detector_metadata=True
    ):
        mapping = {
            "Acquisition.AcquisitionStartDatetime.DateTime": (
                "General.time_zone",
                lambda x: self._get_local_time_zone(),
            ),
            "Optics.AccelerationVoltage": (
                "Acquisition_instrument.TEM.beam_energy",
                lambda x: float(x) / 1e3,
            ),
            "Optics.CameraLength": (
                "Acquisition_instrument.TEM.camera_length",
                lambda x: float(x) * 1e3,
            ),
            "CustomProperties.StemMagnification.value": (
                "Acquisition_instrument.TEM.magnification",
                float,
            ),
            "Instrument.InstrumentClass": (
                "Acquisition_instrument.TEM.microscope",
                None,
            ),
            "Stage.AlphaTilt": (
                "Acquisition_instrument.TEM.Stage.tilt_alpha",
                lambda x: round(np.degrees(float(x)), 3),
            ),
            "Stage.BetaTilt": (
                "Acquisition_instrument.TEM.Stage.tilt_beta",
                lambda x: round(np.degrees(float(x)), 3),
            ),
            "Stage.Position.x": (
                "Acquisition_instrument.TEM.Stage.x",
                lambda x: round(float(x), 6),
            ),
            "Stage.Position.y": (
                "Acquisition_instrument.TEM.Stage.y",
                lambda x: round(float(x), 6),
            ),
            "Stage.Position.z": (
                "Acquisition_instrument.TEM.Stage.z",
                lambda x: round(float(x), 6),
            ),
            "ImportedDataParameter.Number_of_frames": (
                "Acquisition_instrument.TEM.Detector.EDS.number_of_frames",
                None,
            ),
            "DetectorMetadata.ElevationAngle": (
                "Acquisition_instrument.TEM.Detector.EDS.elevation_angle",
                lambda x: round(float(x), 3),
            ),
            "DetectorMetadata.Gain": (
                "Signal.Noise_properties.Variance_linear_model.gain_factor",
                float,
            ),
            "DetectorMetadata.Offset": (
                "Signal.Noise_properties.Variance_linear_model.gain_offset",
                float,
            ),
        }

        # Parse individual metadata for each EDS detector
        if parse_individual_EDS_detector_metadata:
            mapping.update(
                {
                    "DetectorMetadata.AzimuthAngle": (
                        "Acquisition_instrument.TEM.Detector.EDS.azimuth_angle",
                        lambda x: "{:.3f}".format(np.degrees(float(x))),
                    ),
                    "DetectorMetadata.LiveTime": (
                        "Acquisition_instrument.TEM.Detector.EDS.live_time",
                        lambda x: "{:.6f}".format(float(x)),
                    ),
                    "DetectorMetadata.RealTime": (
                        "Acquisition_instrument.TEM.Detector.EDS.real_time",
                        lambda x: "{:.6f}".format(float(x)),
                    ),
                    "DetectorMetadata.DetectorName": ("General.title", None),
                }
            )

        # Add selected element
        if map_selected_element:
            if int(self.version) >= 11:
                key = "SharedProperties.EDSSpectrumQuantificationSettings"
            else:
                key = "Operations.ImageQuantificationOperation"
            mapping[key] = ("Sample.elements", self._convert_element_list)

        return mapping

    def _convert_element_list(self, d):
        atomic_number_list = d[d.keys()[0]]["elementSelection"]
        return [
            atomic_number2name[int(atomic_number)]
            for atomic_number in atomic_number_list
        ]

    def _convert_datetime(self, unix_time):
        # Since we don't know the actual time zone of where the data have been
        # acquired, we convert the datetime to the local time for convenience
        dt = datetime.fromtimestamp(float(unix_time), tz=tz.tzutc())
        return dt.astimezone(tz.tzlocal()).isoformat().split("+")[0]

    def _get_local_time_zone(self):
        return tz.tzlocal().tzname(datetime.today())


# Below some information we have got from FEI about the format of the stream:
#
# The SI data is stored as a spectrum stream, ‘65535’ means next pixel
# (these markers are also called `Gate pulse`), other numbers mean a spectrum
# count in that bin for that pixel.
# For the size of the spectrum image and dispersion you have to look in
# AcquisitionSettings.
# The spectrum image cube itself stored in a compressed format, that is
# not easy to decode.


class FeiSpectrumStream(object):
    """Read spectrum image stored in FEI's stream format

    Once initialized, the instance of this class supports numpy style
    indexing and slicing of the data stored in the stream format.
    """

    def __init__(self, stream_group, reader):
        self.reader = reader
        self.stream_group = stream_group
        # Parse acquisition settings to get bin_count and dtype
        acquisition_settings_group = stream_group["AcquisitionSettings"]
        acquisition_settings = _parse_json(acquisition_settings_group[0])
        self.bin_count = int(acquisition_settings["bincount"])
        if self.bin_count % self.reader.rebin_energy != 0:
            raise ValueError(
                "The `rebin_energy` needs to be a divisor of the",
                " total number of channels.",
            )
        if self.reader.SI_data_dtype is None:
            self.reader.SI_data_dtype = acquisition_settings["StreamEncoding"]
        # Parse the rest of the metadata for storage
        self.original_metadata = _parse_sub_data_group_metadata(stream_group)
        # If last_frame is None, compute it
        stream_data = self.stream_group["Data"][:].T[0]
        if self.reader.last_frame is None:
            # The information could not be retrieved from metadata
            # we compute, which involves iterating once over the whole stream.
            # This is required to support the `last_frame` feature without
            # duplicating the functions as currently numba does not support
            # parametetrization.
            spatial_shape = self.reader.spatial_shape
            last_frame = int(
                np.ceil(
                    (stream_data == 65535).sum() / (spatial_shape[0] * spatial_shape[1])
                )
            )
            self.reader.last_frame = last_frame
            self.reader.number_of_frames = last_frame
        self.original_metadata["ImportedDataParameter"] = {
            "First_frame": self.reader.first_frame,
            "Last_frame": self.reader.last_frame,
            "Number_of_frames": self.reader.number_of_frames,
            "Rebin_energy": self.reader.rebin_energy,
            "Number_of_channels": self.bin_count,
        }
        # Convert stream to spectrum image
        if self.reader.lazy:
            self.spectrum_image = self.stream_to_sparse_array(stream_data=stream_data)
        else:
            self.spectrum_image = self.stream_to_array(stream_data=stream_data)

    @property
    def shape(self):
        return self.spectrum_image.shape

    def get_pixelsize_offset_unit(self):
        om_br = self.original_metadata["BinaryResult"]
        return om_br["PixelSize"], om_br["Offset"], om_br["PixelUnitX"]

    def stream_to_sparse_array(self, stream_data):
        import rsciio.utils.fei_stream_readers as stream_readers

        """Convert stream in sparse array

        Parameters
        ----------
        stream_data: array

        """
        # Here we load the stream data into memory, which is fine is the
        # arrays are small. We could load them lazily when lazy.
        stream_data = self.stream_group["Data"][:].T[0]
        sparse_array = stream_readers.stream_to_sparse_COO_array(
            stream_data=stream_data,
            spatial_shape=self.reader.spatial_shape,
            first_frame=self.reader.first_frame,
            last_frame=self.reader.last_frame,
            channels=self.bin_count,
            sum_frames=self.reader.sum_frames,
            rebin_energy=self.reader.rebin_energy,
        )
        return sparse_array

    def stream_to_array(self, stream_data, spectrum_image=None):
        """Convert stream to array.

        Parameters
        ----------
        stream_data: array
        spectrum_image: array or None
            If array, the data from the stream are added to the array.
            Otherwise it creates a new array and returns it.

        """
        import rsciio.utils.fei_stream_readers as stream_readers

        spectrum_image = stream_readers.stream_to_array(
            stream=stream_data,
            spatial_shape=self.reader.spatial_shape,
            channels=self.bin_count,
            first_frame=self.reader.first_frame,
            last_frame=self.reader.last_frame,
            rebin_energy=self.reader.rebin_energy,
            sum_frames=self.reader.sum_frames,
            spectrum_image=spectrum_image,
            dtype=self.reader.SI_data_dtype,
        )
        return spectrum_image
