.. _mrcz-format:

MRCZ format
-----------

.. note::
   To read this format, the optional dependencies ``blosc`` and ``mrcz`` are
   required.

The ``mrcz`` format is an extension of the CCP-EM MRC2014 file format.
`CCP-EM MRC2014 <https://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_ file format.
It uses the `blosc` meta-compression library to bitshuffle and compress files in
a blocked, multi-threaded environment. The supported data types are ``float32``,
``int8``, ``uint16``, ``int16`` and ``complex64``.

It supports arbitrary meta-data, which is serialized into JSON.

MRCZ also supports asynchronous reads and writes.

.. list-table:: More information on the ``mrcz`` format
   :widths: 25 75

   * - Repository
     - https://github.com/em-MRCZ
   * - PyPI
     - https://pypi.org/project/mrcz
   * - Citation
     - https://doi.org/10.1016/j.jsb.2017.11.012
   * - Preprint
     - https://www.biorxiv.org/content/10.1101/116533v1

Support for this format is not enabled by default. In order to enable it,
the `mrcz` library needs to be installed and optionally `blosc` to use
compression.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.mrcz
   :members:
