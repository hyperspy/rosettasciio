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
``docs/supported_formats/`` and the format added to the lists in
``docs/supported_formats/index.rst`` and ``docs/supported_formats/supported_formats.rst``.

A few standard *docstring* components are provided by ``docstrings.py`` and should
be used (see existing plugins).

The *docstrings* are automatically added in the *user guide* using the following lines

.. code-block:: rst

    API functions
    ^^^^^^^^^^^^^

    .. automodule:: rsciio.spamandeggs
       :members:

.. Note ::
    It is advisable to clone the files of an existing plugin when initiating a new
    plugin.


Maintenance
===========

Please refer to the 
`HyperSpy developer guide <http://hyperspy.org/hyperspy-doc/current/dev_guide/intro.html>`_
for maintenance guidelines.
