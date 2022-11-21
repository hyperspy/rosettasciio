.. _ripple-file_specification:

Ripple (Lispix) format description
==================================

This page summarizes the specifications for the :ref:`ripple <ripple-format>`
("raw parameter list") file format developed at NIST as native format for `Lispix
<https://www.nist.gov/services-resources/software/lispix>`_.

Images and spectral images or data cubes that are written in the
(Lispix) raw file format, which contains just a continuous string of numbers.

Data cubes can be stored image by image, or spectrum by spectrum.
Single images are stored row by row, vector cubes are stored row by row
(each row spectrum by spectrum), image cubes are stored image by image.

All of the numbers are in the same format, such as 16 bit signed integer,
IEEE 8-byte real, 8-bit unsigned byte, etc.

The ``.raw`` file should be accompanied by an ASCII text, tab delimited file with
the same name and ``.rpl`` extension. This "raw parameter list" file lists the
characteristics of the ``.raw`` file.

RPL keywords
^^^^^^^^^^^^

Some of the following keys are specific to `HyperSpy <https://hyperspy.org>`_ and
will be ignored by other software.

.. code::

    ========================  =======  =================================================
    Key                       Type     Description
    ========================  =======  =================================================
    width                     int      pixels per row
    height                    int      number of rows
    depth                     int      number of images or spectral pts
    offset                    int      bytes to skip
    data-type                 str      'signed', 'unsigned', or 'float'
    data-length               str      bytes per pixel  '1', '2', '4', or '8'
    byte-order                str      'big-endian', 'little-endian', or 'dont-care'
    record-by                 str      'image', 'vector', or 'dont-care'
    # X-ray keys:
    ev-per-chan               int      optional, eV per channel
    detector-peak-width-ev    int      optional, FWHM for the Mn K-alpha line
    # HyperSpy-specific keys
    depth-origin              int      energy offset in pixels
    depth-scale               float    energy scaling (units per pixel)
    depth-units               str      energy units, usually eV
    depth-name                str      Name of the magnitude stored as depth
    width-origin              int      column offset in pixels
    width-scale               float    column scaling (units per pixel)
    width-units               str      column units, usually nm
    width-name                str      Name of the magnitude stored as width
    height-origin             int      row offset in pixels
    height-scale              float    row scaling (units per pixel)
    height-units              str      row units, usually nm
    height-name               str      Name of the magnitude stored as height
    signal                    str      Type of the signal stored, e.g. EDS_SEM
    convergence-angle         float    TEM convergence angle in mrad
    collection-angle          float    EELS spectrometer collection semi-angle in mrad
    beam-energy               float    TEM beam energy in keV
    elevation-angle           float    Elevation angle of the EDS detector
    azimuth-angle             float    Elevation angle of the EDS detector
    live-time                 float    Live time per spectrum
    energy-resolution         float    Resolution of the EDS (FHWM of MnKa)
    tilt-stage                float    The tilt of the stage
    date                      str      date in ISO 8601
    time                      str      time in ISO 8601
    title                     str      title of the signal to be stored
    ========================  =======  =================================================


Notes
^^^^^

When ``'data-length'`` is 1, the ``'byte order'`` is not relevant as there is only
one byte per datum, and ``'byte-order'`` should be ``'dont-care'``.

When ``'depth'`` is 1, the file has one image, ``'record-by'`` is not relevant and
should be ``'dont-care'``. For spectral images, ``'record-by'`` is 'vector'.
For stacks of images, ``'record-by'`` is ``'image'``.

Floating point numbers can be IEEE 4-byte, or IEEE 8-byte. Therefore if
data-type is float, data-length MUST be 4 or 8.

The ``.rpl`` file is read in a case-insensitive manner. However, when providing
a dictionary as input, the keys MUST be lowercase.

Comment lines, beginning with a semi-colon ``;`` are allowed anywhere.

The first non-comment in the ``.rpl`` file line MUST have two column names:
``'name_1'<TAB>'name_2'``; any name would do e.g. ``'key'<TAB>'value'``.

Parameters can be in ANY order.

In the ``.rpl`` file, the parameter name is followed by ONE tab (spaces are
ignored) e.g.: ``'data-length'<TAB>'2'``

In the ``.rpl`` file, other data and more tabs can follow the two items on
each row, and are ignored.

Other keys and values can be included and are ignored.

Any number of spaces can go along with each tab.
