.. _renishaw-format:

Renishaw
--------

Reader for spectroscopy data saved using Renishaw's WIRE software.
Currently, RosettaSciIO can only read the ``.wdf`` format from Renishaw.

When working with `HyperSpy <https://hyperspy.org>`_, a file can be read using
the following code:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> sig = hs.load("file.wdf")
