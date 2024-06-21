.. _digitalsurf-format:

DigitalSurf format (SUR & PRO)
------------------------------

``.sur`` and ``.pro`` is format developed by digitalsurf to import/export data in their MountainsMap scientific 
analysis software. Target datasets originally result from (micro)-topography and imaging instruments: SEM, AFM, 
profilometer. RGB(A) images, multilayer surfaces and profiles are also supported. Even though it is essentially 
a surfaces format, 1D signals are supported for spectra and spectral maps. Specifically, this is the fileformat 
used by Attolight SA for its scanning electron microscope cathodoluminescence (SEM-CL) hyperspectral maps. This 
plugin was developed based on the MountainsMap software documentation.

Support for loading ``.sur`` and ``.pro`` datasets is complete, including parsing of user/customer-specific 
metadata, and opening of files containing multiple objects. Some rare specific objects (e.g. force curves) 
are not supported, due to no example data being available. Those can be added upon request and providing of 
example datasets. Heterogeneous data can be represented in ``.sur`` and ``.pro`` objects, for instance 
floating-point/topography and rgb data can coexist along the same navigation dimension. Those are casted to 
a homogeneous floating-point representation upon loading.

Support for data saving is partial as ``.sur`` and ``.pro`` do not support all features of hyperspy signals. 
First, they have limited dimensionality. Up to 3d data arrays with either 1d (series of images) or 2d 
(hyperspectral studiable) navigation space can be saved. Also, ``.sur`` and ``.pro`` do not support non-uniform 
axes and saving of models. Finally, ``.sur`` / ``.pro`` linearize intensities along a uniform axis to enforce 
an integer-representation of the data (with scaling and offset). This means that export from float-type hyperspy 
signals is inherently lossy.

Within these limitations, all features from ``.sur`` and ``.pro`` fileformats are supported, notably data 
compression and setting of custom metadata. The file writer splits a signal into the suitable digitalsurf
dataobject primarily by inspecting its dimensions and its datatype, ultimately how various axes and signal 
quantity are named. The criteria are listed here below:

+-----------------+---------------+------------------------------------------------------------------------------+
| Nav. dimension  | Sig dimension | Extension and MountainsMap subclass                                          |
+=================+===============+==============================================================================+
| 0               | 1             | ``.pro``: Spectrum (based on axes name), Profile (default)                   |
+-----------------+---------------+------------------------------------------------------------------------------+
| 0               | 2             | ``.sur``: BinaryImage (based on dtype), RGBImage (based on dtype),           |
|                 |               | Surface (default),                                                           |
+-----------------+---------------+------------------------------------------------------------------------------+
| 1               | 0             | ``.pro``: same as (1,0)                                                      |
+-----------------+---------------+------------------------------------------------------------------------------+
| 1               | 1             | ``.pro``: Spectrum Serie (based on axes name), Profile Serie (default)       |
+-----------------+---------------+------------------------------------------------------------------------------+
| 1               | 2             | ``.sur``: RGBImage Serie (based on dtype), Surface Series (default)          |
+-----------------+---------------+------------------------------------------------------------------------------+
| 2               | 0             | ``.sur``: same as (0,2)                                                      |
+-----------------+---------------+------------------------------------------------------------------------------+
| 2               | 1             | ``.sur``: hyperspectralMap (default)                                         |
+-----------------+---------------+------------------------------------------------------------------------------+

Axes named one of ``Wavelength``, ``Energy``, ``Energy Loss``, ``E``, are considered spectral, and quantities
named one of ``Height``, ``Altitude``, ``Elevation``, ``Depth``, ``Z`` are considered surface. The difference
between Surface and IntensitySurface stems from the AFM / profilometry origin of MountainsMap. "Surface" has
the proper meaning of an open boundary of 3d space, whereas "IntensitySurface" is a mere 2D mapping of an arbitrary
quantity.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.digitalsurf
   :members:
