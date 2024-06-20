.. _digitalsurf-format:

DigitalSurf format (SUR & PRO)
------------------------------

The ``.sur`` and ``.pro`` files are a format developed by the digitalsurf company to handle 
various types of scientific data with their MountainsMap software, such as profilometer, SEM, 
AFM, RGB(A) images, multilayer surfaces and profiles. Even though it is essentially a surfaces 
format, 1D signals are supported for spectra and spectral maps. Specifically, this file format 
is used by Attolight SA for its scanning electron microscope cathodoluminescence (SEM-CL) 
hyperspectral maps. The plugin was developed based on the MountainsMap software documentation, 
which contains a description of the binary format.

Support for ``.sur`` and ``.pro`` datasets loading is complete, including parsing of user/customer
-specific metadata, and opening of files containing multiple objects. Some rare specific objects 
(e.g. force curves) are not supported, due to no example data being available. Those can be added
upon request and providing of example datasets. Heterogeneous data can be represented in ``.sur``
and ``.pro`` objects, for instance floating-point/topography and rgb data can coexist along the same 
navigation dimension. Those are casted to a homogeneous floating-point representation upon loading.

Support for data saving is partial as ``.sur`` and ``.pro`` can be fundamentally incompatible with
hyperspy signals. First, they have limited dimensionality. Up to 3d data arrays with 
either 1d (series of images) or 2d (hyperspectral studiable) navigation space can be saved. Also, 
``.sur`` and ``.pro`` do not support non-uniform axes and saving of models. Finally, ``.sur`` / ``.pro`` 
linearize intensities along a uniform axis to enforce an integer-representation of the data (with scaling and
offset). This means that export from float-type hyperspy signals is inherently lossy.

Within these limitations, all features from the fileformat are supported at export, notably data 
compression and setting of custom metadata.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.digitalsurf
   :members:
