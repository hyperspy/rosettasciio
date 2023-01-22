.. _empad-format:

EMPAD format (XML & RAW)
------------------------

This is the file format used by the Electron Microscope Pixel Array
Detector (EMPAD). It is used to store a series of diffraction patterns from
scanning transmission electron diffraction measurements, with a limited set of
metadata. Similarly, to the :ref:`ripple format <ripple-format>`, the raw data
and metadata are saved in two different files and for the EMPAD reader, these
are saved in the ``.raw`` and ``.xml`` files, respectively. To read EMPAD data,
use the ``.xml`` file, which will automatically read the raw data from the ``.raw``
file too. The filename of the ``.raw`` file is defined in the ``.xml`` file, which
implies changing the file name of the ``.raw`` file will break reading the file.

.. Note::

    When using `HyperSpy <https://hyperspy.org>`_, you need to specify the correct
    ``reader`` plugin, as different ``.xml`` formats are supported:

    .. code-block:: python

        >>> import hyperspy.api as hs
        >>> sig = hs.load("file.xml", reader="EMPAD")


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.empad
   :members:
