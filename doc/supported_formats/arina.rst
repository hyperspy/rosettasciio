.. _arina-format:

Arina
-----

This is the file format used by the Dectris Arina detector. It stores 4D-STEM 
data. For each scan, the detector writes one master file, a series of
datafiles labeled with integers, and newer versions include an additional 
metadata file. When loading data, the ``filename`` should be the master file.
Note that this reader requires ``hdf5plugin``.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.arina
   :members:
