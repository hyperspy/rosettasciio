from asyncore import write
from importlib.metadata import metadata

import h5py
import dask.array as da
import numpy as np
from rsciio.hspy._api import HyperspyWriter


# for some reason this is slightly different in dm4???
# Data Type 10: Integer?
#
data_types = {
    np.short().dtype: [2, 2], # 2 byte integer signed
    np.uint8().dtype: [6, 1], # 1 byte integer unsigned
    np.float32().dtype: [2, 4], # 4 byte real (IEEE 754)
    np.uint16().dtype: [10,2],  # 2 byte integer unsigned

    np.int32().dtype: [23, 4], # 4 byte integer signed
    np.uint32().dtype: [23, 4], # 4 byte integer signed
              }

class DM5:
    """
    The internal representation of a DM5 file.

    The basic structure of a DM5 file is:
    - ImageList: a list of images. This is the main data structure in a DM5 file.
    - Image Behavior: a list of behaviors that can be applied to images. (e.g. Image Shift, Image Rotation) (Not implemented)
    - Thumbnails: A slice of the image data that can be used to display the image as a thumbnail. (Not implemented)
    - Image Source List: A list of sources for the images. (Not implemented)



    """
    def __init__(self, file_path, mode='r'):
        self.file = h5py.File(file_path, mode=mode)

        if mode == 'r':
            self.image_list = self.file["ImageList"] # list of images
        else:
            self.image_list = self.file.create_group("ImageList")

        self.images = []

    def write_sources(self, signal_dimensions, navigation_dimensions):
        """
        I think that DM has a similar concept of a Navigator and a signal
        in hyperspy.
        """

        self.file.create_group("ImageSourceList")
        self.file["ImageSourceList"].create_group("[0]")

        if signal_dimensions==2 and navigation_dimensions ==2:
            # this is for 4D Data
            self.file["ImageSourceList"]["[0]"].attrs.update({"ClassName":"ImageSource:4DSummed", # Need the right ClassName
                                                              "Do Sum": 1, # Sum the image?
                                                              "ImageRef":0, # Image to take the sum of... (Similar to Hyperspy)
                                                              "LayerFstEnd": 0,
                                                              "LayerFstStart": 0,
                                                              "LayerSndEnd": 0,
                                                              "LayerSndStart": 0,
                                                              "Summed Fst Dimension": 0, # Sum 1st dimension
                                                              "Summed Snd Dimension": 1, # Sum 2nd dimension
                                                              })
            self.file["ImageSourceList"]["[0]"].create_group("Id")
            self.file["ImageSourceList"]["[0]"]["Id"].attrs.update({"[0]": 0})

        elif signal_dimensions ==2 and navigation_dimensions==1:
            #Insitu Imaging (Still need to check what the correct default behavior is...)
            self.file["ImageSourceList"]["[0]"].attrs.update(
                {"ClassName": "ImageSource:Summed",  # Need the right ImageSource!
                 "Do Sum": 1,  # Sum the image?
                 "ImageRef": 0,  # Image to take the sum of... (Similar to Hyperspy)
                 "LayerEnd": 0,
                 "LayerStart": 0,
                 "Summed Dimension": 0,  # Sum 1st dimension?
                 })
            self.file["ImageSourceList"]["[0]"].create_group("Id")
            self.file["ImageSourceList"]["[0]"]["Id"].attrs.update({"[0]": 0})
        elif signal_dimensions == 1 and navigation_dimensions == 1:



        # Do we need to write anything here?

    def write_document_objects(self):
        """
        DM uses Document Objects similar to markers/ROIs in hyperspy.
        """
        self.file.create_group("DocumentObjectList")
        self.file["DocumentObjectList"].create_group("[0]")
        self.file["DocumentObjectList"]["[0]"].attrs.update({"AnnotationType": 20,
                                                            'ImageSource': 0,
                                                            'ImageDisplayType': 1, # 1 = Image
                                                             'RangeAdjust': 1.0,
                                                             'SparseSurvey_GridSize': 32,
                                                             'SparseSurvey_NumberPixels': 64,
                                                             'SparseSurvey_UseNumberPixels': 1,
                                                             'SurveyTechnique': 2,
                                                             })
        self.file["DocumentObjectList"]["[0]"].create_group("ImageDisplayInfo")
        self.file["DocumentObjectList"]["[0]"]["ImageDisplayInfo"].attrs.update({"EstimatedMin":0,
                                                                "HiLimitContrastDeltaTriggerPercentage":0})

    def write_header_info(self):
        """
        Just write the minimum "header" info to open the file in DM.
        """
        self.file.attrs.update({"InImageMode": 1}) # The rest of the attributes are for defining window size...

    def write_image_behavior(self):
        """
        DM uses Image Behaviors to apply transformations to images.

        Write only the minimum required to open the file in DM.
        """
        self.file.create_group("Image Behavior")
        self.file["Image Behavior"].attrs.update({"ViewDisplayID": 8})

    def read_images(self):
        for image_group in self.image_list.values():
            self.images.append(Image(image_group))

    def write_image(self, data, axes_dicts=None, metadata =None, brightness=None):
        """
        Write an image to the DM5 file.
        """

        previous_images = [int(k.strip("[ ]")) for k in self.image_list.keys()]
        if len(previous_images) == 0:
            new_image_number = '[0]'
        else:
            new_image_number = f"[{max(previous_images) + 1}]"

        self.image_list.create_group(new_image_number)
        self.image_list[new_image_number].create_group("ImageData")
        self.image_list[new_image_number].create_group("ImageTags")
        self.image_list[new_image_number].create_group("UniqueID")

        image = Image(self.image_list[new_image_number])
        self.images.append(image)

        image.update_data(data)
        if axes_dicts is not None:
            for axis, axis_dict in axes_dicts.items():
                image.update_calibration(axis, **axis_dict)
                image.update_dimension(axis, axis_dict["size"])
        else:
            for axis in range(len(data.shape)):
                image.update_calibration(axis)
                image.update_dimension(axis)

        image.update_metadata(metadata)

        image.update_brightness(brightness)
        self.write_sources()
        self.write_document_objects()
        self.write_image_behavior()
        self.write_header_info()



class Image:
    """
    The internal representation of an image in a DM5 file.

    Each image has:
    - ImageData: the actual image data
        - Data: the actual image data
        - Calibrations: the calibration data
        - Dimension: the dimensions of the image
    - Tags: Arbitrary tags that can be used to store metadata about the image
    - UniqueID: a unique identifier for the image
    """

    def __init__(self, image_group, tags=None, unique_id=None):
        self.image_data = image_group['ImageData']
        self.image_tags = image_group['ImageTags']
        self.unique_id = image_group['UniqueID']

    def __str__(self):
        return f"Image: {self.image_data['Data'].shape}"


    def get_axis_dict(self, axis):
        """
        Get the calibration data for a given axis.

        Parameters
        ----------
        axis : int
            The axis to get the calibration data for (Starting from 0).

        Notes
        -----
            Axis is 0-indexed.
        """
        try:
            calibration_dict = dict(self.image_data['Calibrations']["Dimension"][f"[{axis}]"])
            axis_dict = {"name": calibration_dict.get("Label", ""),
                         "offset": calibration_dict.get("Origin", 0),
                         "scale": calibration_dict.get("Scale", 1),
                         "unit": calibration_dict.get("Unit", ""),
                         "size": self._get_dimension(axis)}

        except KeyError: # if calibration data is not present. Reads incomplete files...
            axis_dict = {"name": "",
                         "offset": 0,
                         "scale": 1,
                         "unit": "",
                         "size": self._get_dimension(axis)}
        return axis_dict

    def update_calibration(self, axis, label="", origin=0, scale=1, unit=""):
        """
        Add calibration data to the image for a given axis.
        """
        if not "Calibrations" in self.image_data:
            self.image_data.create_group("Calibrations")
        if not "Dimension" in self.image_data['Calibrations']:
            self.image_data['Calibrations'].create_group("Dimension")
        if not f"[{axis}]" in self.image_data['Calibrations']["Dimension"]:
            self.image_data['Calibrations']["Dimension"].create_group(f"[{axis}]")

        self.image_data['Calibrations']["Dimension"][f"[{axis}]"].attrs.update({
            "Label": label,
            "Origin": origin,
            "Scale": scale,
            "Unit": unit
        })
        self.image_data['Calibrations'].attrs.update({"DisplayCalibratedUnits": 1})


    def brightness(self):
        """
        Get the brightness of the image.
        """
        try:
            dict(self.image_data["Calibrations"]["Brightness"].attrs)
        except KeyError:
            return {}

    def update_brightness(self, brightness=None):
        """
        Update the brightness of the image.
        """
        if brightness is None:
            brightness = {"Label": "b",
                          "Origin": 0,
                          "Scale": 1,
                          "Units": "b"}
        if not "Calibrations" in self.image_data:
            self.image_data.create_group("Calibrations")
        if not "Brightness" in self.image_data["Calibrations"]:
            self.image_data["Calibrations"].create_group("Brightness")
        self.image_data["Calibrations"]["Brightness"].attrs.update(brightness)

    def update_dimension(self, axis, length=None):
        """
        Update the dimension of the image for a given axis.

        This is two places in the DM5 file???

        Under Calibrations and under Dimension. I think that only the Calibrations should be updated.

        """
        if not "Dimensions" in self.image_data:
            self.image_data.create_group("Dimensions")
        if length is None:
            length = self._get_dimension(axis)
        self.image_data['Dimensions'].attrs.update({f"[{axis}]": length})

    def _get_dimension(self, axis):
        try:
            return self.image_data['Dimensions'].attrs[f"[{axis}]"]
        except KeyError:
            shape = self.image_data['Data'].shape
            return shape[len(shape) - axis-1]

    def get_data(self, lazy=False):
        """
        Get the image data.
        """
        if lazy:
            return da.from_array(self.image_data['Data'])
        else:
            return np.array(self.image_data['Data'])

    def update_data(self, data):
        """
        Update the image data.
        """
        HyperspyWriter.overwrite_dataset(self.image_data,
                                                data,
                                                "Data")
        self.image_data.attrs.update({"DataType": data_types[data.dtype][0],
                                      "PixelDepth": data_types[data.dtype][1]})


    def get_metadata(self):
        """
        Get the metadata for the image.
        """
        original_metadata = _group2dict(self.image_tags)
        # translate to Hyperspy metadata format

        metadata = {}
        metadata["General"] = {}
        metadata["General"]["title"] = "" # DM uses the filename as the title


        metadata[""]
        if "Acquisition" in original_metadata:
            metadata["Acquisition"] = original_metadata["Acquisition"]

        # The tag structure of DMFiles is arbitrary for the most part. So we can
        # just copy the dict structure of the hyperspy metadata.
        if "Microscope Info" in original_metadata:
            metadata["Acquisition_instrument"] = {}
            metadata["Acquisition_instrument"]["TEM"] = {}
            metadata["Acquisition_instrument"]["TEM"]["beam_energy "] = (
                    original_metadata["Microscope Info"].get("Voltage", 0)/1000)
            metadata["Acquisition_instrument"]["TEM"]["acquisition_mode"] = (
                    original_metadata["Microscope Info"].get("Illumination Mode", "Unknown"))
            metadata["Acquisition_instrument"]["TEM"]["magnification"] = (
                    original_metadata["Microscope Info"].get("Indicated Magnification", 0))
            metadata["Acquisition_instrument"]["TEM"]["camera_length"] = (
                    original_metadata["Microscope Info"].get("STEM Camera Length", 0))
            metadata["Acquisition_instrument"] = original_metadata["Microscope Info"]
        metadata["Acquisition_instrument"] = {}
        return metadata, original_metadata


    def update_metadata(self, metadata=None):
        """
        Update the metadata for the image.


        """
        if metadata is None:
            metadata = {}
        formatted_metadata = {}

        formatted_metadata["Acquisition"] = {}

        # Test
        formatted_metadata["Meta Data"] = {}
        formatted_metadata["Meta Data"]["Format"] = "Diffraction image"
        formatted_metadata["Meta Data"]["Acquisition Mode"] = "Parallel imaging"
        formatted_metadata["Meta Data"]["Experiment keywords"] = {}
        formatted_metadata["Meta Data"]["Experiment keywords"]["[0]"] = "Label: Diffraction"
        # Data is reversed from usual...
        formatted_metadata["Meta Data"]["Data Order Swapped"] = "true"
        dict2group(formatted_metadata, self.image_tags)
        return

    def to_signal_dict(self):
        """
        Convert the image to a Hyperspy signal dictionary.
        """
        data = self.get_data()
        metadata, original_metadata = self.get_metadata()
        axes = []
        for axis in range(len(data.shape)):
            axes.append(self.get_axis_dict(axis + 1))
        return {"data": data, "metadata": metadata, "original_metadata": original_metadata, "axes": axes}




def dict2group(dictionary, group):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            dict2group(value, subgroup)
        else:
            group.attrs[key] = value

def _group2dict(group, dictionary=None):
    if dictionary is None:
        dictionary = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            try:
                value = value.decode()
            except UnicodeDecodeError:
                value = "Decoding error"
        if isinstance(value, (np.bytes_, str)):
            if value == "_None_":
                value = None
        elif isinstance(value, np.bool_):
            value = bool(value)
        elif isinstance(value, np.ndarray) and value.dtype.char == "S":
            # Convert strings to unicode
            value = value.astype("U")
            if value.dtype.str.endswith("U1"):
                value = value.tolist()
        else:
            dictionary[key] = value

    if not isinstance(group, h5py.Dataset):
        for key in group.keys():
            dictionary[key] = {}
            _group2dict(group[key], dictionary[key])
    return dictionary