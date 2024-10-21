File Specifications
===================

This section contains reference documentation for file specifications available
publicly.

.. toctree::
   :hidden:
   
   ripple-specs

.. _edax-file_specification:

EDAX
----

The file specifications for the :ref:`edax <edax-format>` file formats made
available publicly available from EDAX:

- :download:`spc <edax/SPECTRUM-V70.pdf>`
- :download:`spd <edax/SpcMap-spd.file.format.pdf>`
- :download:`ipr <edax/ImageIPR.pdf>`


Ripple (Lispix)
---------------

:ref:`Format description <ripple-file_specification>` for the :ref:`ripple
<ripple-format>` file format.


MRC (CCP-EM)
------------

The :ref:`MRC <mrc-format>` file format is a standard open file format for electron microscopy data and is
defined by  `Cheng et al <https://doi.org/10.1016/j.jsb.2015.04.002>`_. The file format is described in
detail following the link as well:  `MRC2014 <https://www.ccpem.ac.uk/mrc_format/mrc2014.php>`_.

Additionally Direct Electron saves a couple of files along with the ``.mrc`` file. In general, the file naming
scheme is CurrentDate_MovieNumber_suffix_movie.mrc with the metadata file named CurrentDate_MovieNumber_suffix_info.txt.
and different external detectors or virtual images are saved as CurrentDate_MovieNumber_suffix_ext#_extName.mrc or
CurrentDate_MovieNumber_suffix_#_virtualName.mrc respectively. The suffix is optional and can be any
string or left empty. By default Virtual image 0 is the sum at each navigation point and is equivalent to the
navigation image.

