# -*- coding: utf-8 -*-
# Copyright 2007-2024 The HyperSpy developers
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

from PIL.ExifTags import TAGS

CustomTAGS = {
    **TAGS,
    # Customized EXIF TAGS from Renishaw
    0xFEA0: "FocalPlaneXYOrigins",  # 65184
    0xFEA1: "FieldOfViewXY",  # 65185
    0xFEA2: "Unknown",  # 65186, could it be magnification?
}


# from https://exiftool.org/TagNames/EXIF.html
# For tag 0x9210 (37392)
FocalPlaneResolutionUnit_mapping = {
    None: None,
    1: None,
    2: "inches",
    3: "cm",
    4: "mm",
    5: "µm",
}


def _parse_axes_from_metadata(exif_tags, sizes):
    # return of axes must not be empty, or dimensions are lost
    # if no exif_tags exist, axes are set to a scale of 1 per pixel,
    # unit is set to None, hyperspy will parse it as a traits.api.undefined value
    offsets = [0, 0]
    fields_of_views = [sizes[1], sizes[0]]
    unit = None
    if exif_tags is not None:
        # Fallback to default value when tag not available
        offsets = exif_tags.get("FocalPlaneXYOrigins", offsets)
        # jpg files made with Renishaw have this tag
        fields_of_views[0] = exif_tags.get("FocalPlaneXResolution", fields_of_views[0])
        fields_of_views[1] = exif_tags.get("FocalPlaneYResolution", fields_of_views[1])
        unit = FocalPlaneResolutionUnit_mapping[
            exif_tags.get("FocalPlaneResolutionUnit", unit)
        ]

    axes = [
        {
            "name": name,
            "units": unit,
            "size": size,
            "scale": fields_of_views[i] / size,
            "offset": offsets[i],
            "index_in_array": i,
        }
        for i, name, size in zip([1, 0], ["y", "x"], sizes)
    ]

    return axes


def _parse_exif_tags(im):
    """
    Parse exif tags from a pillow image

    Parameters
    ----------
    im : :class:`PIL.Image`
        The pillow image from which the exif tags will be parsed.

    Returns
    -------
    exif_dict : None or dict
        The dictionary of exif tags.

    """
    exif_dict = None
    try:
        # missing header keys when Pillow >= 8.2.0 -> does not flatten IFD anymore
        # see https://pillow.readthedocs.io/en/stable/releasenotes/8.2.0.html#image-getexif-exif-and-gps-ifd
        # Use fall-back _getexif method instead
        # Not all format plugin have the private method
        # prefer to use that method as it returns more items
        exif_dict = im._getexif()
    except AttributeError:
        exif_dict = im.getexif()
    if exif_dict is not None:
        exif_dict = {CustomTAGS.get(k, "unknown"): v for k, v in exif_dict.items()}

    return exif_dict
