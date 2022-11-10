
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
`pull request <https://github.com/hyperspy/rosettasciio/pulls>`_. Small bug fixes
are corrections to the user guide are typically a good starting point. But don't
hesitate also for significant code contributions, such as support for a new
file format - if needed, we'll help you to get the code ready to common standards.

Please refer to the 
`HyperSpy developer guide <http://hyperspy.org/hyperspy-doc/current/dev_guide/intro.html>`_
in order to get started and for detailed contributing guidelines.

Reviewing
---------

As quality assurance, to improve the code, and to ensure a generalized
functionality, pull requests need to be thoroughly reviewed by at least one
other member of the development team before being merged.


.. _defining-plugins:

Defining new RosettaSciIO plugins
=================================

Each read/write plugin resides in a separate directory, the name of which should
be descriptive of the file type/manufacturer/software. This directory should
contain the following files:

* ``__init__.py`` -- May usually be empty

* ``specifications.yaml`` -- The characteristics of the IO plugin in *yaml* format:

  .. code-block:: yaml

      name: <String> # unique, concise, no whitespace; usually corresponding to directory name
      name_aliases: [<String>]  # List of strings, may contain whitespaces (empty if no alias defined)
      description: <String>
      full_support: <Bool>	# Whether all the Hyperspy features are supported
      file_extensions: <Tuple of string>  # Recognised file extension
      default_extension: <Int>	# Index of the extension that will be used by default
      writes: <Bool>/[Nested list]  # Writing capabilities
      # if only limited dimensions are supported, the supported combinations of signal
      # dimensions (sd) and navigation dimensions (nd) are given as list [[sd, nd], ...]
      non_uniform_axis: <Bool>  # Support for non-uniform axis

* ``api.py`` -- Python file that implements the actual reader. The IO functionality
  should be interfaced with the following functions:

  * A function called ``file_reader`` with at least one attribute: ``filename``
  * (optional) A function called ``file_writer`` with at least two attributes: 
    ``filename`` and ``object2save`` (a python dictionary) in that order.

**Tests** covering the functionality of the plugin should be added to the
``tests`` directory with the naming ``test_spamandeggs.py`` corresponsing to
the plugin residing in the directory ``spamandeggs``. Data files for the tests
should be placed in a corresponding subdirectory [change for pooch].

**Documentation** should be added both as **docstring**, as well as to the **user guide**,
for which a corresponding ``.rst`` file should be created in the directory
``docs/supported_formats/`` and the format added to the lists in
``docs/supported_formats/index.rst`` and ``docs/supported_formats/supported_formats.rst``.

.. Note ::
    It is advisable to clone the files of an existing plugin when initiating a new
    plugin.


Maintenance
===========

Please refer to the 
`HyperSpy developer guide <http://hyperspy.org/hyperspy-doc/current/dev_guide/intro.html>`_
for maintenance guidelines.
