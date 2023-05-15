import numpy as np
import h5py as hf

from rsciio.docstrings import FILENAME_DOC, RETURNS_DOC


def file_reader(filename):
    # add docstring
    hdf = hf.File(filename, "r")
    
    Acquisition2=hdf.get('Acquisition2')
    Acquisition2_ImageData=Acquisition2.get('ImageData')
    Acquisition2_ImageData_Image=Acquisition2_ImageData.get('Image')

    Acquisition2_ImageData_DimensionScaleC=Acquisition2_ImageData.get('DimensionScaleC')
    Acquisition2_ImageData_DimensionScaleX=Acquisition2_ImageData.get('DimensionScaleX')
    Acquisition2_ImageData_DimensionScaleY=Acquisition2_ImageData.get('DimensionScaleY')

    DATA=Acquisition2_ImageData_Image[:,0,0,:,:]
    data=DATA.transpose()
    del DATA
    
    
    axes = [
        {
            "name": "X",
            "size": data.shape[0],
            "offset": 0,
            "scale": float(np.array(Acquisition2_ImageData_DimensionScaleX)),
            "units": "µm",
        },
        {
            "name": "Y",
            "size": data.shape[1],
            "offset": 0,
            "scale": float(np.array(Acquisition2_ImageData_DimensionScaleY)),
            "units": "µm",
        },
        {
            "name": "Wavelength",
            "axis":  Acquisition2_ImageData_DimensionScaleC,
            "size":  len(Acquisition2_ImageData_DimensionScaleC),
            "units": "nm",
        },
    ]
        
    spim = {
        "data": data,
        "axes": axes,
    }

    return [
        spim,
    ]




file_reader.__doc__ %= (FILENAME_DOC, RETURNS_DOC)
