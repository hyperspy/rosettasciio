.. _quantumdetector-format:

Quantum Detector
----------------

The ``mib`` file format is the format from the Quantum Detector software to
acquired with the Quantum Detector Merlin camera. It is typically used to
store a series of diffraction patterns from scanning transmission electron
diffraction measurements. It supports reading data from camera with one or
four quadrants.

If a ``hdr`` file with the same file name was saved along the ``mib`` file,
it will be used to infer the navigation shape of the providing that the option
"line trigger" was used for the acquisition. Alternatively, the navigation
shape can be specified as an argument:

.. code-block:: python

    >>> from rsciio.quantumdetector import file_reader
    >>> s_dict = file_reader("file.mib", navigation_shape=(256, 256))


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.quantumdetector
   :members:
