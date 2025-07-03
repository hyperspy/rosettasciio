.. _numpy-format:

NumPy (NPY)
-----------

This reader supports reading and writing of numpy arrays in the ``.npy`` format,
which is a binary file format for storing numpy arrays. This reader supports reading
large ``npy`` files lazily and in a :ref:`distributed <lazy>` fashion.

.. note::

   This format doesn't support storing metadata like scale, units, etc.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.numpy
   :members:
