.. _utils:

:mod:`rsciio.utils`
===================

RosettaSciIO provides certain utility functions that are applicable for multiple
formats, e.g. for the HDF5-format on which a number of plugins are based.

.. autosummary::
   rsciio.utils.file
   rsciio.utils.hdf5
   rsciio.utils.path
   rsciio.utils.rgb
   rsciio.utils.xml

.. automodule:: rsciio.utils
   :members:
   

.. _file-utils:

File
^^^^

.. autosummary::
   rsciio.utils.file.get_file_handle
   rsciio.utils.file.inspect_npy_bytes
   rsciio.utils.file.memmap_distributed

.. automodule:: rsciio.utils.file
   :members:


.. _hdf5-utils:

HDF5
^^^^

.. autosummary::
   rsciio.utils.hdf5.list_datasets_in_file
   rsciio.utils.hdf5.read_metadata_from_file

.. automodule:: rsciio.utils.hdf5
   :members:


.. _path-utils:

Path
^^^^

.. autosummary::
   rsciio.utils.path.append2pathname
   rsciio.utils.path.ensure_directory
   rsciio.utils.path.incremental_filename
   rsciio.utils.path.overwrite

.. automodule:: rsciio.utils.path
   :members:


.. _rgb-utils:

RGB
^^^

.. autosummary::
   rsciio.utils.rgb.is_rgb
   rsciio.utils.rgb.is_rgba
   rsciio.utils.rgb.is_rgbx
   rsciio.utils.rgb.regular_array2rgbx
   rsciio.utils.rgb.rgbx2regular_array
   rsciio.utils.rgb.RGB_DTYPES

.. automodule:: rsciio.utils.rgb
   :members:


.. _xml-utils:

XML
^^^

.. autosummary::
   rsciio.utils.xml.XmlToDict
   rsciio.utils.xml.convert_xml_to_dict
   rsciio.utils.xml.sanitize_msxml_float
   rsciio.utils.xml.xml2dtb

.. automodule:: rsciio.utils.xml
   :members:


Test
^^^^

.. automodule:: rsciio.tests.registry_utils
   :members: