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


.. grid:: 2 3 3 3
  :gutter: 2

  .. grid-item-card::
    :link: user_guide/install
    :link-type: doc

    :octicon:`rocket;2em;sd-text-info` Getting Started
    ^^^

    New to RosettaSciIO or Python? Find information on how to
    install RosettaSciIO.

  .. grid-item-card::
    :link: supported_formats/index
    :link-type: doc

    :octicon:`checklist;2em;sd-text-info` Supported Formats
    ^^^

    Overview of the different file formats that RosettaSciIO can read
    from and write to, as well as information on specific options if applicable.

  .. grid-item-card::
    :link: user_guide/interoperability
    :link-type: doc

    :octicon:`arrow-switch;2em;sd-text-info` Interoperability
    ^^^

    Information on scripts and packages that allow opening data saved using
    RosettaSciIO with third-party programs.

  .. grid-item-card::
    :link: file_specification/index
    :link-type: doc

    :octicon:`checklist;2em;sd-text-info` File Specifications
    ^^^

    Links to or documentation of file specifications for
    some of the formats supported by RosettaSciIO.

  .. grid-item-card::
    :link: api/index
    :link-type: doc

    :octicon:`book;2em;sd-text-info` API Reference
    ^^^

    Documentation describing how to use RosettaSciIO with other
    libraries than HyperSpy or for custom implementations using the
    Application Programming Interface (API).

  .. grid-item-card::
    :link: contributing
    :link-type: doc

    :octicon:`code-square;2em;sd-text-info` Contributing
    ^^^

    Information on how to implement new file plugins or help improve existing ones.


RosettaSciIO provides the dataset, its axes and related metadata contained in a
file in a :ref:`python dictionary <interfacing-api>` that can be easily
handled by other libraries.

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
badge below is the `Concept DOI <https://support.zenodo.org/help/en-gb/1-upload-deposit/97-what-is-doi-versioning>`_ --
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
   supported_formats/index
   api/index
   contributing
   changes
