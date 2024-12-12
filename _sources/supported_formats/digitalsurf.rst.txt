.. _digitalsurf-format:

DigitalSurf format (SUR & PRO)
------------------------------

``.sur`` and ``.pro`` is a format developed by digitalsurf to import/export data in the MountainsMap scientific 
analysis software. Target datasets are originally (micro)-topography maps and profile from imaging instruments: 
SEM, AFM, profilometery etc. RGB(A) images, multilayer surfaces and profiles are also supported. Even though it 
is essentially a surfaces format, 1D signals are supported for spectra and spectral maps. Specifically, this is 
the format used by Attolight for saving SEM-cathodoluminescence (SEM-CL) hyperspectral maps. This plugin was 
developed based on the MountainsMap software documentation.

Support for loading ``.sur`` and ``.pro`` files is complete, including parsing of custom metadata, and opening of 
files containing multiple objects. Some rare, deprecated object types (e.g. force curves) are not supported, due 
to no example data being available. Those can be added upon request to the module, if provided with example data
and a explanations. Unlike hyperspy.signal, ``.sur`` and ``.pro`` objects can be used to represent heterogeneous
data. For instance, float (topography) and int (rgb data) data can coexist along the same navigation dimension. 
Those are casted to a homogeneous floating-point representation upon loading.

Support for data saving is partial, as ``.sur`` and ``.pro`` do not support all features of hyperspy signals. Up 
to 3d data arrays with either 1d (series of images) or 2d (spectral maps) navigation space can be saved. ``.sur`` 
and ``.pro`` also do not support non-uniform axes and fitted models. Finally, MountainsMap maps intensities along 
an axis with constant spacing between numbers by enforcing an integer-representation of the data with scaling and 
offset. This means that export from float data is inherently lossy.

Within these limitations, all features from ``.sur`` and ``.pro`` fileformats are supported. Data compression and
custom metadata allows a good interoperability of hyperspy and Mountainsmap. The file writer splits a signal into 
the suitable digitalsurf dataobject. Primarily by inspecting its dimension and datatype. If ambiguity remains, it
inspects the names of signal axes and ``metadata.Signal.quantity``. The criteria are listed here below:

+-----------------+---------------+------------------------------------------------------------------------------+
| Nav. dimension  | Sig dimension | Extension and MountainsMap subclass                                          |
+=================+===============+==============================================================================+
| 0               | 1             | ``.pro``: Spectrum (based on axes name), Profile (default)                   |
+-----------------+---------------+------------------------------------------------------------------------------+
| 0               | 2             | ``.sur``: BinaryImage (based on dtype), RGBImage (based on dtype),           |
|                 |               | Surface (default)                                                            |
+-----------------+---------------+------------------------------------------------------------------------------+
| 1               | 0             | ``.pro``: same as (0,1)                                                      |
+-----------------+---------------+------------------------------------------------------------------------------+
| 1               | 1             | ``.pro``: Spectrum Serie (based on axes name), Profile Serie (default)       |
+-----------------+---------------+------------------------------------------------------------------------------+
| 1               | 2             | ``.sur``: RGBImage Serie (based on dtype), Surface Series (default)          |
+-----------------+---------------+------------------------------------------------------------------------------+
| 2               | 0             | ``.sur``: same as (0,2)                                                      |
+-----------------+---------------+------------------------------------------------------------------------------+
| 2               | 1             | ``.sur``: hyperspectralMap (default)                                         |
+-----------------+---------------+------------------------------------------------------------------------------+

Axes named one of ``Wavelength``, ``Energy``, ``Energy Loss`` or  ``E`` are considered spectral. A quantity named 
one of ``Height``, ``Altitude``, ``Elevation``, ``Depth`` or ``Z`` is  considered a surface. The difference between 
Surface and IntensitySurface stems from the AFM / profilometry origin of MountainsMap. "Surface" has its proper 
meaning of being a 2d-subset of 3d space, whereas "IntensitySurface" is a mere 2D mapping of an arbitrary quantity.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.digitalsurf
   :members:
