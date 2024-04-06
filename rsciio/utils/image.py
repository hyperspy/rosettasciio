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
    0xFEA2: "Unknown",  # 65186
}


# from https://exiftool.org/TagNames/EXIF.html
# For tag 0x9210 (37392)
FocalPlaneResolutionUnit_mapping = {
    "": "",
    1: "",
    2: "inches",
    3: "cm",
    4: "mm",
    5: "µm",
}


def _parse_axes_from_metadata(exif_tags, sizes):
    if exif_tags is None:
        return []
    offsets = exif_tags.get("FocalPlaneXYOrigins", [0, 0])
    # jpg files made with Renishaw have this tag
    scales = exif_tags.get("FieldOfViewXY", [1, 1])

    unit = FocalPlaneResolutionUnit_mapping[
        exif_tags.get("FocalPlaneResolutionUnit", "")
    ]

    axes = [
        {
            "name": name,
            "units": unit,
            "size": size,
            "scale": scales[i] / size,
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
