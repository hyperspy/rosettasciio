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

# NanoMegas app5 is a proprietary data format developed by APP5 for NanoMegas.
# It contains a lot of data that can be read as hdf5, but most of the
# internal indexing is binary and took some decoding.  The initial work
# was done by Dr Gary Paterson in the fpd library
# https://fpdpy.gitlab.io/fpd/index.html
# This has been recoded for use for hyperspy import by Dr Ian MacLaren with
# some simplification and redesign of the algorithm and some minor changes
# due to changes in xml functions used in python


import h5py
import xml.etree.cElementTree as ET
from tqdm import tqdm

from rsciio._docstrings import (
    CHUNKS_DOC,
    FILENAME_DOC,
    LAZY_DOC,
    RETURNS_DOC,
    SIGNAL_DOC,
)

_logger = logging.getLogger(__name__)


def import_app5(app5, which, imageflip):
    """

    Input:
    app5:
        A NanoMEGAS app5 file
    which: str
        What you wish for as an output:
        'survey': The overview survey image (note currently no box is imported to show the scan area)
        'virtual': The virtual image taken during acquisition of the dataset
        'SPED': The SPED 4DSTEM dataset
    imageflip: bool
        If False, then image exported as seen in NanoMEGAS Topspin
        If True, then image exported with a flip of horizontal axis (may be necessary with some detectors / scan systems)
    metadata: bool
        If True, a separate python dictionary of known metadata is also exported.  This is not written into the hyperspy file as most
        tags are not similar to standard tags and I don't know what to do with them in the hs ecosystem.  It also misses a bunch of
        useful information including date, operator, microscope, sample and more.  Further investigation needed there.
        Only applies for which='survey' or 'SPED', because only this part of the data has all these.

    Output:
        A suitable hyperspy object, either a BaseSignal object for survey or virtual, or a Signal2D object for SPED
    """

    h5file = h5py.File(app5, "r")
    groups = [v for k, v in h5file.items() if isinstance(v, h5py.Group)]
    metas = [g["Metadata"][()].decode() for g in groups]
    contexts = [ET.fromstring(mi).find("Context").text for mi in metas]

    # Sorts out details of survey image and checks that that the file actually contains only one scan
    # Finds the right context for the survey image
    Area = [c.endswith("Series Area") for c in contexts]
    # Checks that the file only has one dataset
    number_of_datasets = sum(Area)
    if number_of_datasets != 1:
        h5file.close()
        raise Exception(
            "Expected 1 'Survey Area' dataset, but there are %d. Conversion aborted."
            % (number_of_datasets)
        )
    # Gets the index in the context list
    seriesAreaIndex = Area.index(True)
    # Pulls the right group for the actual data
    seriesAreaGroup = groups[seriesAreaIndex]
    # Gets the name of the area
    Areaname = (
        ET.fromstring(metas[seriesAreaIndex]).find("ProcedureData/Item/Value/Id").text
    )

    # Sorts out details of virtual image and scan
    # Find if we have something called 'Series' in contexts, which will contain the scan data
    series = [c.endswith("Series") for c in contexts]
    # Get the index for the series in the short list
    seriesIndex = series.index(True)
    # Get the data group for this index
    seriesGroup = groups[seriesIndex]

    # Get metadata list for the Scan
    metas = ET.fromstring(metas[seriesIndex])
    # Get all the names for the metadata parameters
    names = [item.find("Name").text for item in metas.findall("ProcedureData/Item")]
    # Get the values for the metadata parameters as binary (needed for finding the virtual image)
    valuesbinary = [item.find("Value") for item in metas.findall("ProcedureData/Item")]
    # Get the values of the metadata parameters as texts
    valuestext = [
        item.find("Value").text for item in metas.findall("ProcedureData/Item")
    ]
    # Find the index in the names list for the Virtual Image
    VirtualImageListIndex = names.index("VirtualStemImageResult")
    # Get the binary index that actually finds the image in the seriesGroup
    VirtualImageIndexNumber = valuesbinary[VirtualImageListIndex].find("Id").text
    # Zip up a metadata dictionary
    SeriesMetaData = dict(zip(names, valuestext))

    # Now actually read the file data out

    if which == "survey":
        # Uses the name to get the image data
        SurveyAreaImage = seriesAreaGroup[Areaname][()]
        if imageflip == True:
            SurveyAreaImage = SurveyAreaImage[:, ::-1]

        # Pulls the metadata to get the scales for calibration
        surveyMD = ET.fromstring(seriesAreaGroup["Metadata"][()].decode())
        # determine scan calibrations from metadata, converted into microns
        xScale_surv = float(
            surveyMD.find("ProcedureData/Item/Value/Calibration/X/Scale").text
        ) / (1e-6)
        yScale_surv = float(
            surveyMD.find("ProcedureData/Item/Value/Calibration/Y/Scale").text
        ) / (1e-6)

        # Gets the image shape and makes the hs Survey Image
        surv_shape = SurveyAreaImage.shape
        dicty = {
            "size": surv_shape[0],
            "name": "y",
            "units": "µm",
            "scale": yScale_surv,
            "offset": 0,
        }
        dictx = {
            "size": surv_shape[1],
            "name": "x",
            "units": "µm",
            "scale": xScale_surv,
            "offset": 0,
        }
        axes = [dicty, dictx]
        data = SurveyAreaImage[()]

    elif which == "virtual" or "SPED":

        # Load the Virtual Image from its h5 dataset and gets shape
        virtualImage = seriesGroup[VirtualImageIndexNumber][()]
        shape = virtualImage[()].shape

        if which == "virtual":
            if imageflip == True:
                virtualImage = virtualImage[:, ::-1]
            # Pulls the metadata to get the scales for calibration
            series_calibration_metadata = ET.fromstring(
                seriesGroup["Metadata"][()].decode()
            )
            # determine scan calibrations from metadata, converted into nanometres
            xScale = float(
                series_calibration_metadata.find(
                    "ProcedureData/Item/Value/Calibration/X/Scale"
                ).text
            ) / (1e-9)
            yScale = float(
                series_calibration_metadata.find(
                    "ProcedureData/Item/Value/Calibration/Y/Scale"
                ).text
            ) / (1e-9)
            # Makes the hs Virtual Image
            dicty = {
                "size": shape[0],
                "name": "y",
                "units": "nm",
                "scale": yScale,
                "offset": 0,
            }
            dictx = {
                "size": shape[1],
                "name": "x",
                "units": "nm",
                "scale": xScale,
                "offset": 0,
            }
            axes = [dicty, dictx]
            data = virtualImage[()]

        if which == "SPED":
            # list out all the serializers
            serializers = [value.attrib["Serializer"] for value in valuesbinary]
            # Find the index for the right one
            ImageSeriesListIndex = serializers.index("ImageSeriesSerializer")
            # Recall the binary address
            serializerbinary = valuesbinary[ImageSeriesListIndex]
            # Now list IDs inside this binary address
            SerializerIDs = [
                ID.attrib["Serializer"] for ID in list(serializerbinary.iter())
            ]
            # Get the index in this list for the one that points to the data
            Guidindex = SerializerIDs.index("Guid")
            # Now get the binary address for the data itself
            Dataset_Guid = list(serializerbinary.iter())[Guidindex].text
            # Now get an hdf5 pointer to the data
            SPED_dataset = seriesGroup[Dataset_Guid]
            # Get the shapes set up
            yPoints, xPoints = shape
            DPshape = SPED_dataset["0"]["Data"][()].shape
            # Make an empty 3D array for the data
            data3D = np.empty(shape=(yPoints * xPoints, *DPshape))
            with tqdm(total=(yPoints * xPoints), unit="images") as pbar:
                for point in range(yPoints * xPoints):
                    data3D[point, :, :] = SPED_dataset[str(point)]["Data"][()]
                    pbar.update(point)
            data4D = data3D.reshape(yPoints, xPoints, *DPshape)[::-1]
            if imageflip == True:
                data4D = data4D[:, ::-1, :, :]

            SPED_calib_metadata = ET.fromstring(
                SPED_dataset["0"]["Metadata"][()].decode()
            )
            kxScale = float(SPED_calib_metadata.find("Calibration/X/Scale").text)
            kyScale = float(SPED_calib_metadata.find("Calibration/Y/Scale").text)
            kxOffset = float(SPED_calib_metadata.find("Calibration/X/Offset").text)
            kyOffset = float(SPED_calib_metadata.find("Calibration/Y/Offset").text)

            # Makes the hs Virtual Image
            dicty = {
                "size": shape[0],
                "name": "y",
                "units": "nm",
                "scale": yScale,
                "offset": 0,
            }
            dictx = {
                "size": shape[1],
                "name": "x",
                "units": "nm",
                "scale": xScale,
                "offset": 0,
            }
            dictky = {
                "size": DPshape[0],
                "name": "ky",
                "units": "unknown",
                "scale": kyScale,
                "offset": kyOffset,
            }
            dictkx = {
                "size": DPshape[1],
                "name": "kx",
                "units": "unknown",
                "scale": kxScale,
                "offset": kxOffset,
            }
            axes = [dicty, dictx, dictky, dictkx]
            data = data4D[()]

    return data, axes, SeriesMetaData


def file_reader(filename, which="survey", imageflip=False):
    data, axes, metadata = import_app5(filename, which, imageflip)
    imd = []
    imd.append(
        {
            "data": data,
            "axes": axes,
            "original_metadata": deepcopy(metadata),
        }
    )

    return imd
