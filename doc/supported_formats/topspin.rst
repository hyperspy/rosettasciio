.. _app5-format:

TopSpin App5
---------

RosettaSciIO can read the app5 format used by NanoMegas in their TopSpin
software. These files use the hdf5 file format (and thus can be read with
h5py) to store scanning precession electron diffraction (SPED) measurements,
as well as standard STEM images. 

Data for 2D images is stored as HDF5 Datasets, 4D datasets are stored as HDF5
groups, and Metadata is stored as XML formatted binarized text strings.

Each individual data collection operation is assigned a unique 32-character
alphaneumeric ID, which is used for it's address within the App5 file.
Depending on how the file was exported from TopSpin, these can be further
grouped into 32-character session ID's, where all files within the session
correspond to the same collection area.

Additonally, as App5 files can be large, the ``file_reader`` can be
ran with ``dryrun=True`` to quickly scan the contents of an app5 file
without loading it in its entirety, or ``subset='id'`` to only load a single
experiment from the file. see the docstring for more details.


.. warning::

   While App5 is supported, it is a proprietary format, and future
   versions of the format might therefore not be readable. Additionally,
   the organization of MetaData files changes based on the local hardware
   setup. 


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.topspin
   :members:
