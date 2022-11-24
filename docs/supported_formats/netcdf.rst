.. _netcdf-format:

NetCDF (EELSlab) format
-----------------------

The ``.nc`` format was the default format in HyperSpy's predecessor, EELSLab, but it has been
superseded by :ref:`hspy-format` in RosettaSciIO. We provide only reading capabilities
but we do not support writing to this format.

Note that only NetCDF files written by EELSLab are supported.

To use this format a python netcdf interface, such as `netcdf4
<http://unidata.github.io/netcdf4-python/>`_ must be installed manually because
it is not installed by default.


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.netcdf
   :members:
