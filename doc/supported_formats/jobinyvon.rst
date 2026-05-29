.. _jobinyvon-format:

Horiba Jobin Yvon LabSpec
-------------------------

Reader for spectroscopy data saved using Horiba Jobin Yvon's LabSpec software.
Currently, RosettaSciIO can only read the ``.xml`` format from Jobin Yvon.
However, this format supports spectral maps and contains all relevant metadata.
Therefore, it is a good alternative to the binary ``.l5s`` or ``.l6s`` formats
in order to transfer data to python or other analysis software.

If `LumiSpy <https://lumispy.org>`_ is installed, ``Luminescence`` will be
used as the ``signal_type``.

When working with `HyperSpy <https://hyperspy.org>`_, a file can be read using
the following code:

.. code-block:: python

    >>> import hyperspy.api as hs
    >>> sig = hs.load("file.xml", reader="JobinYvon")

Specifying the reader is necessary as the :ref:`EMPAD format <empad-format>`
also uses the ``.xml`` file-extension.

The reader supports all signal axis units exported by LabSpec, i.e. wavelengths,
wavenumbers (absolute), Raman shift (relative wavenumbers),
as well as energy.

.. Note::

  The wavelength-to-energy conversion is not documented for LabSpec, i.e. it is
  not clear whether the refractive index of air is taken into account. When working
  with energy axes, it is therefore recommended to import the data using the
  wavelength axis and doing the `conversion using the LumiSpy package 
  <https://lumispy.readthedocs.io/en/latest/user_guide/signal_axis.html#the-energy-axis>`_.
  Additionally, to our knowledge, LabSpec does not have an option to take into
  account the `Jacobian transformation of the intensity
  <https://lumispy.readthedocs.io/en/latest/user_guide/signal_axis.html#jacobian-transformation>`_.

.. Note::

    From LabSpec 6.3, an alternative export to an open HDF5 data format is
    available. In the future, this plugin may be extended to support this format,
    as well as to writing data to the open Horiba Jobin Yvon formats.
    Contributions are welcome.


API functions
^^^^^^^^^^^^^

.. automodule:: rsciio.jobinyvon
   :members:
