Changelog
*********

Changelog entries for the development version are available at
https://rosettasciio.readthedocs.io/en/latest/changes.html

.. towncrier-draft-entries:: |release| [UNRELEASED]

.. towncrier release notes start

API Removal
-----------

Following the deprecation cycle, the following keywords and attributes have been
removed:

- :ref:`Bruker composite file (BCF) <bcf-format>`: The 'spectrum' option for the
  `select_type` parameter was removed. Use 'spectrum_image' instead.
- :ref:`Electron Microscopy Dataset (EMD) NCEM <emd_ncem-format>`: Using the
  keyword 'dataset_name' was removed, use 'dataset_path' instead.
- :ref:`NeXus data format <nexus-format>`: The `dataset_keys`, `dataset_paths`
  and `metadata_keys` keywords were removed. Use `dataset_key`, `dataset_path`
  and `metadata_key` instead.

Initiation (2022-07-23)
=======================

- RosettaSciIO was split out of the `HyperSpy repository 
  <https://github.com/hyperspy/hyperspy>`_ on July 23, 2022. The IO-plugins
  and related functions so far developed in HyperSpy were moved to this
  new repository.
