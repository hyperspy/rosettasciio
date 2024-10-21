import h5py
import numpy as np

from rsciio._docstrings import FILENAME_DOC, LAZY_DOC, RETURNS_DOC


def file_reader(filename, lazy=False):
    """
    Read a Delmic hdf5 hyperspectral image.


    Parameters
    ----------
    %s
    %s

    %s
    """
    hdf = h5py.File(filename, "r")

    Acquisition2 = hdf.get("Acquisition2")
    Acquisition2_ImageData = Acquisition2.get("ImageData")
    Acquisition2_ImageData_Image = Acquisition2_ImageData.get("Image")

    Acquisition2_ImageData_DimensionScaleC = Acquisition2_ImageData.get(
        "DimensionScaleC"
    )
    Acquisition2_ImageData_DimensionScaleX = Acquisition2_ImageData.get(
        "DimensionScaleX"
    )
    Acquisition2_ImageData_DimensionScaleY = Acquisition2_ImageData.get(
        "DimensionScaleY"
    )

    DATA = Acquisition2_ImageData_Image[:, 0, 0, :, :]
    data = DATA.transpose()
    del DATA

    axes = [
        {
            "name": "X",
            "size": data.shape[0],
            "offset": 0,
            "scale": float(np.array(Acquisition2_ImageData_DimensionScaleX)),
            "units": "µm",
            "navigate": True,
        },
        {
            "name": "Y",
            "size": data.shape[1],
            "offset": 0,
            "scale": float(np.array(Acquisition2_ImageData_DimensionScaleY)),
            "units": "µm",
            "navigate": True,
        },
        {
            "name": "Wavelength",
            "axis": np.array(Acquisition2_ImageData_DimensionScaleC),
            "units": "nm",
            "navigate": False,
        },
    ]

    metadata = {"signal": {"signal_type": "", "quantity": "Intensity (counts)"}}

    original_metadata = dict(DimensionScaleX="182", DimensionScaleY="132")

    spim = {
        "data": data,
        "axes": axes,
        "metadata": metadata,
        "original_metadata": original_metadata,
    }

    return [
        spim,
    ]


file_reader.__doc__ %= (FILENAME_DOC, LAZY_DOC, RETURNS_DOC)
