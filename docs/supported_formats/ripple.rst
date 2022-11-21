.. _ripple-format:

Ripple format
-------------

The ``.rpl`` format (short for "Raw Parameter List") is an *open standard format*
developed at NIST as native format for `Lispix
<https://www.nist.gov/services-resources/software/lispix>`_ and is widely used to
exchange multidimensional data. However, it only supports data of up to three
dimensions. See the :ref:`file specifications <ripple-file_specification>`
for a description of the file format and the parameter keys (some of which are
specific to `HyperSpy <https://hyperspy.org>`_). This format is often used in
EDS/EDX experiments.

The ``.rpl`` file lists the characteristics of the corresponding ``.raw`` file so
that it can be loaded without human intervention. Thus, the reader parses a
``.rpl`` file and reads the data from the corresponding
``.raw`` file, or directly from a ``.raw`` file if the dictionary ``rpl_info`` is
provided.

It can also be used to exchange data with Bruker and used in
combination with the :ref:`import-rpl` it is very useful for exporting data
to Gatan's Digital Micrograph.

.. note::

    This format may not provide information on the calibration.
    If so, you should add that after loading the file.


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.ripple
   :members:
