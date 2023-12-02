Contributing
************

RosettaSciIO is meant to be a community maintained project. We welcome contributions
in the form of bug reports, documentation, code (in particular :ref:`new io plugins
<defining-plugins>`), feature requests, and more. In the following we refer to some
resources to help you make useful contributions.

Issues
======

The `issue tracker <https://github.com/hyperspy/rosettasciio/issues>`_ can be used to
report bugs or propose new features. When reporting a bug, the following is
useful:

- give a minimal example demonstrating the bug,
- copy and paste the error traceback.

Pull Requests
=============

If you want to contribute to the RosettaSciIO source code, you can send us a
`pull request <https://github.com/hyperspy/rosettasciio/pulls>`_ against the ``main``
branch. Small bug fixes are corrections to the user guide are typically a good
starting point. But don't hesitate also for significant code contributions, such
as support for a new file format - if needed, we'll help you to get the code ready
to common standards.

Please refer to the
`HyperSpy developer guide <http://hyperspy.org/hyperspy-doc/current/dev_guide/intro.html>`_
in order to get started and for detailed contributing guidelines.

Lint
----
To keep the code style consistent (and more readable), `black <https://black.readthedocs.io/>`_
is used to check the code formatting. When the code doesn't comply with the expected formatting,
the `lint <https://github.com/hyperspy/rosettasciio/actions/workflows/black.yml>`_ will fail.
In practise, the code formatting can be fixed by installing ``black`` and running it on the
source code or by using :ref:`pre-commit hooks <pre-commit-hooks>`.


.. _adding-and-updating-test-data:

Adding and Updating Test Data
-----------------------------
The test data are located in the corresponding subfolder of the ``rsciio/tests/data`` folder.
To add or update test data:

#. use git as usual to add files to the repository.
#. Update ``rsciio.tests.registry.txt``.  The test data are not packaged in RosettaSciIO to
   keep the packages as small as possible in size. However, to be able to run the test suite
   of RosettaSciIO after installation or when packaging on conda-forge, pooch is used to
   download the data when necessary. It means that when adding and updating test files, it
   is necessary to update the registry ``rsciio.tests.registry.txt``, which can be done by
   running :py:func:`~.tests.registry_utils.update_registry` (Unix only):

   .. code-block:: python

      from rsciio.tests.registry_utils import update_registry

      update_registry()

   On windows, you can use :ref:`pre-commit.ci <pre-commit-hooks>` by adding a message to
   the pull request to update the registry.

Review
------

As quality assurance, to improve the code, and to ensure a generalized
functionality, pull requests need to be thoroughly reviewed by at least one
other member of the development team before being merged.

.. _pre-commit-hooks:

Pre-commit Hooks
----------------
Two pre-commit hooks are set up:

* Linting: run ``black``
* Update test data registry (Unix only)

These can be run locally by using `pre-commit <https://pre-commit.com>`__.
Alternatively, the comment ``pre-commit.ci autofix`` can be added to a PR to fix the formatting
using `pre-commit.ci <https://pre-commit.ci>`_.

.. _defining-plugins:

Defining new RosettaSciIO plugins
=================================

Each read/write plugin resides in a separate directory, e.g. ``spamandeggs`` the
name of which should be descriptive of the file type/manufacturer/software. This
directory should contain the following files:

* ``__init__.py`` -- Defines the exposed API functions, ``file_reader`` and optionally ``file_writer``

  .. code-block:: python

      from ._api import file_reader, file_writer


      __all__ = [
          "file_reader",
          "file_writer",
      ]


      def __dir__():
          return sorted(__all__)

* ``specifications.yaml`` -- The characteristics of the IO plugin in *yaml* format:

  .. code-block:: yaml

      name: <String> # unique, concise, no whitespace; corresponding to directory name (e.g. ``spamandeggs``)
      name_aliases: [<String>]  # List of strings, may contain whitespaces (empty if no alias defined)
      description: <String>
      full_support: <Bool>	# Whether all the Hyperspy features are supported
      file_extensions: <Tuple of string>  # Recognised file extension
      default_extension: <Int>	# Index of the extension that will be used by default
      writes: <Bool>/[Nested list]  # Writing capabilities
      # if only limited dimensions are supported, the supported combinations of signal
      # dimensions (sd) and navigation dimensions (nd) are given as list [[sd, nd], ...]
      non_uniform_axis: <Bool>  # Support for non-uniform axis

* ``_api.py`` -- Python file that implements the actual reader. The IO functionality
  should be interfaced with the following functions:

  * A function called ``file_reader`` with at least one attribute: ``filename``
    that returns the :ref:`standardized signal dictionary <interfacing-api>`.
  * (optional) A function called ``file_writer`` with at least two attributes:
    ``filename`` and ``signal`` (a python dictionary) in that order.

**Tests** covering the functionality of the plugin should be added to the
``tests`` directory with the naming ``test_spamandeggs.py`` corresponsing to
the plugin residing in the directory ``spamandeggs``. Data files for the tests
should be placed in a corresponding subdirectory - see the
:ref:`Adding and Updating Test Data <adding-and-updating-test-data>` section for more
information.

**Documentation** should be added both as **docstring**, as well as to the **user guide**,
for which a corresponding ``spamandeggs.rst`` file should be created in the directory
``doc/user_guide/supported_formats/`` and the format added to the lists in
``doc/user_guide/supported_formats/index.rst`` and ``doc/user_guide/supported_formats/supported_formats.rst``.

A few standard *docstring* components are provided by ``rsciio._docstrings.py`` and should
be used (see existing plugins).

The *docstrings* are automatically added in the *user guide* using the following lines

.. code-block:: rst

    API functions
    ^^^^^^^^^^^^^

    .. automodule:: rsciio.spamandeggs
       :members:

The *docstrings* follow `Numpy docstring style <https://numpydoc.readthedocs.io>`_. The
links to RosettaSciIO API and other Sphinx documented API are checked when building the documentation
and broken links will raise warnings. In order to identify potentially broken links during pull
request review, the `Documentation <https://github.com/hyperspy/rosettasciio/actions/workflows/Documentation.yml>`_
GitHub CI workflow is set to fail when the doc build raises warnings.

.. Note ::
    It is advisable to clone the files of an existing plugin when initiating a new
    plugin.


RosettaSciIO version
====================
The version of RosettaSciIO is defined by `setuptools_scm <https://setuptools-scm.readthedocs.io/>`_
and retrieve by ``importlib.metadata`` at runtime in case of user installation.

- Version at build time: the version is defined from the tag or the "distance from the tag".
- Version at runtime: use the version of the package (``sdist`` or ``wheel``), which would have been
  defined at build time. At runtime, the version is obtained using importlib.metadata as follow:

  .. code-block:: python
  
    from importlib.metadata import version
    __version__ = version("rosettasciio")

- Version at runtime for editable installation: the version is defined from the tag or "the distance from the tag".

.. note::

  To define the version in development installation or at build time, ``setuptools_scm`` uses
  the git history with all commits, and shallow checkout will provide incorrect version.
  For user installation in site-package, ``setuptools_scm`` is not used.


Dependencies
============
``RosettaSciIO`` strive to be easy to install with a minimum of dependencies and depends solely on
standard library modules, numpy and dask. Non-pure python (binaries) dependencies are optional for
the following reasons:

- provide maximum flexibility in usability and avoid forcing user to install library that they don't need:
  for user-cases, where only a file reader are necessary, it should be possible to install ``RosettaSciIO``
  without installing large or non-pure python dependencies, which are not always easy to install.
- Some binaries dependencies are not supported for all python implementation (``pypy`` or ``pyodide``)
  or for all platforms.

Maintenance
===========

Please refer to the
`HyperSpy developer guide <http://hyperspy.org/hyperspy-doc/current/dev_guide/intro.html>`_
for maintenance guidelines.
