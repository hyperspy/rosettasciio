.. _numpy-format:

NumPy (NPY)
-----------

This reader supports reading and writing of numpy arrays in the ``.npy`` format,
which is a binary file format for storing numpy arrays. This format doesn't store
metadata like axes or units, but this reader supports reading large ``npy`` files
lazily and in a :ref:`distributed <lazy>` fashion.


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.numpy
   :members:
