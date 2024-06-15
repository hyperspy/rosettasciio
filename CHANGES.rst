Release Notes
*************

Changelog entries for the development version are available at
https://rosettasciio.readthedocs.io/en/latest/changes.html

.. towncrier-draft-entries:: |release| [UNRELEASED]

.. towncrier release notes start

0.5 (2024-06-15)
================

Enhancements
------------

- :ref:`emd_fei-format`: Enforce setting identical units for the ``x`` and ``y`` axes, as convenience to use the scalebar in HyperSpy. (`#243 <https://github.com/hyperspy/rosettasciio/issues/243>`_)
- :ref:`quantumdetector-format`: Add support for dask distributed scheduler. (`#267 <https://github.com/hyperspy/rosettasciio/issues/267>`_)


Bug Fixes
---------

- :ref:`emd_fei-format`: Fix conversion of offset units which can sometimes mismatch the scale units. (`#243 <https://github.com/hyperspy/rosettasciio/issues/243>`_)
- :ref:`ripple-format`: Fix typo and improve error message for unsupported ``dtype`` in writer. (`#251 <https://github.com/hyperspy/rosettasciio/issues/251>`_)
- :ref:`emd_fei-format`: Fix parsing elements from EDS data from velox emd file v11. (`#274 <https://github.com/hyperspy/rosettasciio/issues/274>`_)


Maintenance
-----------

- Use ``ruff`` for code formating and linting. (`#250 <https://github.com/hyperspy/rosettasciio/issues/250>`_)
- Fix ``tifffile`` deprecation. (`#262 <https://github.com/hyperspy/rosettasciio/issues/262>`_)
- Add support for ``python-box`` 7. (`#263 <https://github.com/hyperspy/rosettasciio/issues/263>`_)


0.4 (2024-04-02)
================

Enhancements
------------

- :ref:`Renishaw wdf <renishaw-format>`:

  - return survey image instead of saving it to the metadata and add marker of the mapping area on the survey image.
  - Add support for reading data with invariant axis, for example when the values of the Z axis doesn't change.
  - Parse calibration of ``jpg`` images saved with Renishaw Wire software. (`#227 <https://github.com/hyperspy/rosettasciio/issues/227>`_)
- Add support for reading :ref:`emd <emd_fei-format>` Velox version 11. (`#232 <https://github.com/hyperspy/rosettasciio/issues/232>`_)
- Add :ref:`making test data files <making_test_files>` section to contributing guide, explain characteristics of "good" test data files. (`#233 <https://github.com/hyperspy/rosettasciio/issues/233>`_)
- :ref:`Quantum Detector <quantumdetector-format>` reader: use timestamps to get navigation shape when the navigation shape is not available - for example, acquisition with pixel trigger or scan shape not in metadata. (`#235 <https://github.com/hyperspy/rosettasciio/issues/235>`_)
- Improve setting output size for an image. (`#244 <https://github.com/hyperspy/rosettasciio/issues/244>`_)


Bug Fixes
---------

- Fix saving ``hspy`` file with empty array (signal or metadata) and fix closing ``hspy`` file when a error occurs during reading or writing. (`#206 <https://github.com/hyperspy/rosettasciio/issues/206>`_)
- Fix saving ragged arrays of vectors from/to a chunked ``hspy`` and ``zspy`` store.  Greatly increases the speed of saving and loading ragged arrays from chunked datasets. (`#211 <https://github.com/hyperspy/rosettasciio/issues/211>`_)
- Fix saving ragged array of strings in ``hspy`` and ``zspy`` format. (`#217 <https://github.com/hyperspy/rosettasciio/issues/217>`_)
- Fix setting beam energy for XRF maps in ``bcf`` files. (`#231 <https://github.com/hyperspy/rosettasciio/issues/231>`_)
- :ref:`Quantum Detector <quantumdetector-format>` reader: fix setting chunks. (`#235 <https://github.com/hyperspy/rosettasciio/issues/235>`_)


Maintenance
-----------

- Add ``POOCH_BASE_URL`` to specify the base url used by pooch to download test data. This fixes the failure of the ``package_and_test.yml`` workflow in pull requests where test data are added or updated. (`#200 <https://github.com/hyperspy/rosettasciio/issues/200>`_)
- Fix documentation links following release of hyperspy 2.0. (`#210 <https://github.com/hyperspy/rosettasciio/issues/210>`_)
- Run test suite on osx arm64 on GitHub CI and speed running test suite using all available CPUs (3 or 4) instead of only 2. (`#222 <https://github.com/hyperspy/rosettasciio/issues/222>`_)
- Fix deprecation warnings introduced with numpy 1.25 ("Conversion of an array with ndim > 0 to a scalar is deprecated, ..."). (`#230 <https://github.com/hyperspy/rosettasciio/issues/230>`_)
- Fix numpy 2.0 removal (``np.product`` and ``np.string_``). (`#238 <https://github.com/hyperspy/rosettasciio/issues/238>`_)
- Fix download test data when using ``pytest --pyargs rsciio -n``. (`#245 <https://github.com/hyperspy/rosettasciio/issues/245>`_)


0.3 (2023-12-12)
================

New features
------------

- Add :func:`rsciio.set_log_level` to set the logging level of ``RosettaSciIO`` (`#69 <https://github.com/hyperspy/rosettasciio/issues/69>`_)
- Added the :func:`~rsciio.utils.distributed.memmap_distributed` function for loading a memmap file
  from multiple processes.

  - Added the arguments ``distributed`` and ``metadata_file`` to the .mrc file reader for loading metadata
    save from DirectElectron detectors.
  - Speed up to the .mrc file reader for large .mrc files by removing the need to reshape
    and transpose the data. (`#162 <https://github.com/hyperspy/rosettasciio/issues/162>`_)
- Add support for saving lazy ragged signals to the :ref:`zspy format<zspy-format>`. (`#193 <https://github.com/hyperspy/rosettasciio/pull/193>`_)


Bug Fixes
---------

- Fix error when reading :ref:`pantarhei-format` file with aperture ``"Out"`` (`#173 <https://github.com/hyperspy/rosettasciio/issues/173>`_)
- Improvement for installation without ``numba``:

  - Fix :ref:`tvips <tvips-format>` reader
  - Allow reading and writing :ref:`EMD NCEM <emd_ncem-format>` file
  - Fix running test suite without optional dependencies (`#182 <https://github.com/hyperspy/rosettasciio/issues/182>`_)
- Fix getting version on debian/ubuntu in system-wide install. Add support for installing from git archive and improve getting development version using setuptools `fallback_version <https://setuptools-scm.readthedocs.io/en/latest/config>`_ (`#187 <https://github.com/hyperspy/rosettasciio/issues/187>`_)
- Fix ``dwell_time`` reading in :ref:`QuantumDetectors <quantumdetector-format>` reader (``.mib`` file). The
  ``dwell_time`` is stored in milliseconds, not microseconds as the previous code
  assumed. (`#189 <https://github.com/hyperspy/rosettasciio/issues/189>`_)


Maintenance
-----------

- Remove usage of deprecated ``distutils`` (`#152 <https://github.com/hyperspy/rosettasciio/issues/152>`_)
- Fix installing exspy/hyperspy on GitHub CI and test failing without optional dependencies (`#186 <https://github.com/hyperspy/rosettasciio/issues/186>`_)
- Unpin pillow now that imageio supports pillow>=10.1.0 (`#188 <https://github.com/hyperspy/rosettasciio/issues/188>`_)
- Simplify GitHub CI workflows by using reusable workflow (`#190 <https://github.com/hyperspy/rosettasciio/issues/190>`_)


.. _changes_0.2:

0.2 (2023-11-09)
================

New features
------------

- Add support for reading the ``.img``-format from :ref:`Hamamatsu <hamamatsu-format>`. (`#87 <https://github.com/hyperspy/rosettasciio/issues/87>`_)
- Add support for reading the ``.mib``-format from :ref:`Quantum Detector Merlin <quantumdetector-format>` camera. (`#174 <https://github.com/hyperspy/rosettasciio/issues/174>`_)


Bug Fixes
---------

- Fix saving/reading ragged arrays with :ref:`hspy<hspy-format>`/:ref:`zspy<zspy-format>` plugins (`#164 <https://github.com/hyperspy/rosettasciio/issues/164>`_)
- Fixes slow loading of ragged :ref:`zspy<zspy-format>` arrays (#168) (`#169 <https://github.com/hyperspy/rosettasciio/issues/169>`_)


Improved Documentation
----------------------

- Improve docstrings, check API links when building documentation and set GitHub CI to fail when link is broken (`#142 <https://github.com/hyperspy/rosettasciio/issues/142>`_)
- Add zenodo doi to documentation (`#149 <https://github.com/hyperspy/rosettasciio/issues/149>`_)
- Update intersphinx mapping links of matplotlib/numpy. (`#150 <https://github.com/hyperspy/rosettasciio/issues/150>`_)


Enhancements
------------

- Add option to show progress bar when saving lazy signals to :ref:`hspy<hspy-format>`/:ref:`zspy<zspy-format>` files (`#170 <https://github.com/hyperspy/rosettasciio/issues/170>`_)
- Make ``numba`` and ``h5py`` optional dependencies to support RosettaSciIO on `pyodide <https://pyodide.org/>`_ and `PyPy <https://www.pypy.org/>`_ (`#180 <https://github.com/hyperspy/rosettasciio/issues/180>`_)


Maintenance
-----------

- Remove deprecated ``record_by`` attribute in :ref:`hspy <hspy-format>`/:ref:`zspy <zspy-format>`, (`#143 <https://github.com/hyperspy/rosettasciio/issues/143>`_)
- Add ``sidpy`` dependency and pin it to <0.12.1 as a workaround to fix ``pyusid`` import (`#155 <https://github.com/hyperspy/rosettasciio/issues/155>`_)
- Update :ref:`hspy<hspy-format>`/:ref:`zspy<zspy-format>` plugins to new markers API introduced in HyperSpy 2.0 (`#164 <https://github.com/hyperspy/rosettasciio/issues/164>`_)
- Pin pillow<10.1.0 until imageio supports newer pillow version - see https://github.com/imageio/imageio/issues/1044 (`#175 <https://github.com/hyperspy/rosettasciio/issues/175>`_)
- Update the test suite and the CI workflows to work with and without exspy installed (`#176 <https://github.com/hyperspy/rosettasciio/issues/176>`_)
- Add badges that became available after first release (`#177 <https://github.com/hyperspy/rosettasciio/issues/177>`_)

.. _changes_0.1:

0.1 (2023-06-06)
================

New features
------------

- Add support for reading the ``.xml``-format from Horiba :ref:`Jobin Yvon <jobinyvon-format>`'s LabSpec software. (`#25 <https://github.com/hyperspy/rosettasciio/issues/25>`_)
- Add support for reading the ``.tvf``-format from :ref:`TriVista <trivista-format>`. (`#27 <https://github.com/hyperspy/rosettasciio/issues/27>`_)
- Add support for reading the ``.wdf``-format from :ref:`Renishaw's WIRE <renishaw-format>` software. (`#55 <https://github.com/hyperspy/rosettasciio/issues/55>`_)
- Added subclassing of ``.sur`` files in CL signal type and updated metadata parsing (`#98 <https://github.com/hyperspy/rosettasciio/issues/98>`_)
- Add optional kwarg to tiff reader ``multipage_as_list`` which when set to True uses ``pages`` interface and returns list of signal for every page with full metadata. (`#104 <https://github.com/hyperspy/rosettasciio/issues/104>`_)
- Add file reader and writer for PRZ files generated by :ref:`CEOS PantaRhei <pantarhei-format>` (`HyperSpy #2896 <https://github.com/hyperspy/hyperspy/issues/2896>`_)


Bug Fixes
---------

- Ensure that the ``.msa`` plugin handles ``SIGNALTYPE`` values according to the official format specification. (`#39 <https://github.com/hyperspy/rosettasciio/issues/39>`_)
- Fix error when reading Velox file containing FFT with an odd number of pixels (`#49 <https://github.com/hyperspy/rosettasciio/issues/49>`_)
- Fix error when reading JEOL ``.pts`` file with un-ordered frame list or when length of ``frame_start_index`` is smaller than the sweep count (`#68 <https://github.com/hyperspy/rosettasciio/issues/68>`_)
- Fix exporting scalebar with reciprocal units containing space (`#90 <https://github.com/hyperspy/rosettasciio/issues/90>`_)
- Fix array indexing bug when loading a ``sur`` file format containing spectra series. (`#98 <https://github.com/hyperspy/rosettasciio/issues/98>`_)
- For more robust xml to dict conversion, ``convert_xml_to_dict`` is replaced by ``XmlToDict`` (introduced by PR #111). (`#101 <https://github.com/hyperspy/rosettasciio/issues/101>`_)
- Fix bugs with reading non-FEI and Velox ``mrc`` files, improve documentation of ``mrc`` and ``mrcz`` file format. Closes `#71 <https://github.com/hyperspy/rosettasciio/issues/71>`_, `#91 <https://github.com/hyperspy/rosettasciio/issues/91>`_, `#93 <https://github.com/hyperspy/rosettasciio/issues/93>`_, `#96 <https://github.com/hyperspy/rosettasciio/issues/96>`_, `#130 <https://github.com/hyperspy/rosettasciio/issues/130>`_. (`#131 <https://github.com/hyperspy/rosettasciio/issues/131>`_)


Improved Documentation
----------------------

- Consolidate docstrings and documentation for all plugins (see also `#47 <https://github.com/hyperspy/rosettasciio/pull/47>`_, `#59 <https://github.com/hyperspy/rosettasciio/pull/59>`_, `#64 <https://github.com/hyperspy/rosettasciio/pull/64>`_, `#72 <https://github.com/hyperspy/rosettasciio/pull/72>`_) (`#76 <https://github.com/hyperspy/rosettasciio/issues/76>`_)
- Remove persistent search field in left sidebar since this makes finding the sidebar on narrow screens difficult.
  Set maximal major version of Sphinx to 5. (`#84 <https://github.com/hyperspy/rosettasciio/issues/84>`_)


Deprecations
------------

- Remove deprecated ``record_by`` attribute from file readers where remaining (`#102 <https://github.com/hyperspy/rosettasciio/issues/102>`_)


Enhancements
------------

- Recognise both byte and string object for ``NXdata`` tag in NeXus reader (`#112 <https://github.com/hyperspy/rosettasciio/issues/112>`_)


API changes
-----------

- Move, enhance and share xml to dict/list translation and other tools (new api for devs) from ``Bruker._api`` to utils:
  ``utils.date_time_tools.msfiletime_to_unix`` function to convert the uint64 MSFILETIME to  datetime.datetime object.
  ``utils.tools.sanitize_msxml_float`` function to sanitize some MSXML generated xml where comma is used as float decimal separator.
  ``utils.tools.XmlToDict`` Xml to dict/list translator class with rich customization options as kwargs, and main method for translation ``dictionarize`` (`#111 <https://github.com/hyperspy/rosettasciio/issues/111>`_)


Maintenance
-----------

- Initiate GitHub actions for tests and documentation. (`#1 <https://github.com/hyperspy/rosettasciio/issues/1>`_)
- Initiate towncrier changelog and create templates for PRs and issues. (`#3 <https://github.com/hyperspy/rosettasciio/issues/3>`_)
- Add github CI workflow to check links, build docs and push to the ``gh-pages`` branch. Fix links and add EDAX reference file specification (`#4 <https://github.com/hyperspy/rosettasciio/issues/4>`_)
- Add azure pipelines CI to run test suite using conda-forge packages. Add pytest and coverage configuration in ``pyproject.toml`` (`#6 <https://github.com/hyperspy/rosettasciio/issues/6>`_)
- Fix minimum install, add corresponding tests build and tidy up leftover code (`#13 <https://github.com/hyperspy/rosettasciio/issues/13>`_)
- Fixes and code consistency improvements based on analysis provided by lgtm.org (`#23 <https://github.com/hyperspy/rosettasciio/issues/23>`_)
- Added github action for code scanning using the codeQL engine. (`#26 <https://github.com/hyperspy/rosettasciio/issues/26>`_)
- Following the deprecation cycle announced in `HyperSpy <https://hyperspy.org/hyperspy-doc/v2.0/changes.html>`_,
  the following keywords and attributes have been removed:

  - :ref:`Bruker composite file (BCF) <bruker-format>`: The ``'spectrum'`` option for the
    ``select_type`` parameter was removed. Use 'spectrum_image' instead.
  - :ref:`Electron Microscopy Dataset (EMD) NCEM <emd_ncem-format>`: Using the
    keyword ``'dataset_name'`` was removed, use ``'dataset_path'`` instead.
  - :ref:`NeXus data format <nexus-format>`: The ``dataset_keys``, ``dataset_paths``
    and ``metadata_keys`` keywords were removed. Use ``dataset_key``, ``dataset_path``
    and ``metadata_key`` instead. (`#30 <https://github.com/hyperspy/rosettasciio/issues/30>`_)
- Unify the ``format_name`` scheme of IO plugins using ``name`` instead and add ``name_aliases`` (list) for backwards compatibility. (`#35 <https://github.com/hyperspy/rosettasciio/issues/35>`_)
- Add drone CI to test on ``arm64``/``aarch64`` platform (`#42 <https://github.com/hyperspy/rosettasciio/issues/42>`_)
- Unify naming of folders/submodules to match documented format ``name`` (`#81 <https://github.com/hyperspy/rosettasciio/issues/81>`_)
- Add black as a development dependency.
  Add pre-commit configuration file with black code style check, which when installed will require changes to pass a style check before commiting. (`#86 <https://github.com/hyperspy/rosettasciio/issues/86>`_)
- Add support for python-box 7 (`#100 <https://github.com/hyperspy/rosettasciio/issues/100>`_)
- Migrate to API v3 of ``imageio.v3`` (`#106 <https://github.com/hyperspy/rosettasciio/issues/106>`_)
- Add explicit support for python 3.11 and drop support for python 3.6, 3.7 (`#109 <https://github.com/hyperspy/rosettasciio/issues/109>`_)
- Remove test data from packaging and download them when necessary (`#123 <https://github.com/hyperspy/rosettasciio/issues/123>`_)
- Define packaging in ``pyproject.toml`` and keep ``setup.py`` to handle compilation of C extension (`#125 <https://github.com/hyperspy/rosettasciio/issues/125>`_)
- Add release GitHub workflow to automate release process and add corresponding documentation in `releasing_guide.md <https://github.com/hyperspy/rosettasciio/blob/main/releasing_guide.md>`_ (`#126 <https://github.com/hyperspy/rosettasciio/issues/126>`_)
- Add pre-commit hook to update test data registry and pre-commit.ci to run from pull request (`#129 <https://github.com/hyperspy/rosettasciio/issues/129>`_)
- Tidy up ``rsciio`` namespace: privatise ``docstrings``, move ``conftest.py`` and ``exceptions`` to tests and utils folder, respectively (`#132 <https://github.com/hyperspy/rosettasciio/issues/132>`_)


Initiation (2022-07-23)
=======================

- RosettaSciIO was split out of the `HyperSpy repository 
  <https://github.com/hyperspy/hyperspy>`_ on July 23, 2022. The IO-plugins
  and related functions so far developed in HyperSpy were moved to this
  new repository.
