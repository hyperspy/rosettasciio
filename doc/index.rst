.. RosettaSciIO documentation master file, created by
   sphinx-quickstart on Wed Jul 13 20:21:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##########################
RosettaSciIO documentation
##########################

The **Rosetta Scientific Input Output library** aims at providing easy reading and
writing capabilities in Python for a wide range of
:ref:`scientific data formats <supported-formats>`. Thus
providing an entry point to the wide ecosystem of python packages for scientific data
analysis and computation, as well as an interoperability between different file
formats. Just as the `Rosetta stone <https://en.wikipedia.org/wiki/Rosetta_Stone>`_
provided a translation between ancient Egyptian hieroglyphs and ancient Greek.
The RosettaSciIO library originates from the `HyperSpy <https://hyperspy.org>`_
project for multi-dimensional data analysis. As HyperSpy is rooted in the electron
microscopy community, data formats used by this community are still particularly
well represented.

RosettaSciIO provides the dataset, its axes and related metadata contained in a
file in a python dictionary that can be easily handled by other libraries.
Similarly, it takes a dictionary as input for file writers.

.. note::

   RosettaSciIO has recently been split out of the `HyperSpy repository
   <https://github.com/hyperspy/hyperspy>`_ and the new API is still under development.
   HyperSpy will use the RosettaSciIO IO-plugins from v2.0. It is already possible to import
   the readers directly from RosettaSciIO as follows:

   .. code::

      from rsciio import msa
      msa.file_reader("your_msa_file.msa")


Citing RosettaSciIO
===================

If RosettaSciIO has been significant to a project that leads to an academic
publication, please acknowledge that fact by citing it. The DOI in the
badge below is the `Concept DOI <https://help.zenodo.org/faq/#versioning>`_ --
it can be used to cite the project without referring to a specific
version. If you are citing RosettaSciIO because you have used it to process data,
please use the DOI of the specific version that you have employed. You can
find it by clicking on the DOI badge:

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.8011666.svg
   :target: https://doi.org/10.5281/zenodo.8011666


Credits
=======

RosettaSciIO is developed by `an active community of contributors
<https://github.com/hyperspy/rosettasciio/contributors>`_.


Table of contents
=================

.. toctree::
   :maxdepth: 2

   user_guide/index
   file_specification/index
   api/index
   contributing
   changes
