.. _renishaw-format:

Renishaw
--------

Reader for spectroscopy data saved using Renishaw's WiRE software.
Currently, RosettaSciIO can only read the ``.wdf`` format from Renishaw.
When reading spectral images, the white light image will be returned along the 
spectral images in the list of dictionaries. The position of the mapped area
is returned in the metadata dictionary of the white light image and this will
be displayed when plotting the image with HyperSpy.

If `LumiSpy <https://lumispy.org>`_ is installed, ``Luminescence`` will be
used as the ``signal_type``.

.. Note::

   There are many different options for the axes according to the format specifications.
   However, only a limited subset is tested: `Spectral` (Wavelength and Raman Shift) for
   the signal axes and `X`, `Y`, `Z`, `FocusTrackZ` and `Time` for navigation axes.
   Reading maps obtained in a serpentine path is not implemented.


This reader is based on the `py-wdf-reader <https://github.com/alchem0x2A/py-wdf-reader.git>`_,
which is inspired by the `matlab reader <https://doi.org/10.5281/zenodo.495477>`_ from Alex Henderson.
Moreover, inspiration is taken from `gwyddion's reader <http://gwyddion.net>`_.

API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.renishaw
   :members:
