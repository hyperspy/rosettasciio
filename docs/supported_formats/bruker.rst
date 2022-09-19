.. _bruker-format:

Bruker formats
--------------

RosettaSciIO can read ``.spx`` single spectrum and ``.bcf`` "hypermaps" file
formats saved with Bruker's Esprit v1.x or v2.x in ``.bcf``
hybrid (virtual file system/container with xml and binary data, optionally
compressed) format. Most ``.bcf`` import functionality is implemented. Both
high-resolution 16-bit SEM images and hyperspectral EDX data can be retrieved
simultaneously.

BCF can look as all inclusive format, however it does not save some key EDS
parameters: any of dead/live/real times, FWHM at Mn_Ka line. However, real time
for whole map is calculated from pixelAverage, lineAverage, pixelTime,
lineCounter and map height parameters.

Note that Bruker Esprit uses a similar format for EBSD data, but it is not
currently supported by RosettaSciIO.

The format contains extensive list of details and parameters of EDS analyses
which in `HyperSpy <https://hyperspy.org>`_ are mapped to ``metadata`` and
``original_metadata`` dictionaries.

Parameters
++++++++++

.. automodule:: rsciio.bruker
   :members:


Example
+++++++

Example of loading reduced (downsampled, and with energy range cropped)
"spectrum only" data from ``bcf`` (original shape: 80 keV EDS range (4096 channels),
100x75 pixels; SEM acceleration voltage: 20kV):

.. code-block:: python

    >>> hs.load("sample80kv.bcf", select_type='spectrum_image', downsample=2, cutoff_at_kV=10)
    <EDSSEMSpectrum, title: EDX, dimensions: (50, 38|595)>

load the same file with limiting array size to SEM acceleration voltage:

.. code-block:: python

    >>> hs.load("sample80kv.bcf", cutoff_at_kV='auto')
    [<Signal2D, title: BSE, dimensions: (|100, 75)>,
    <Signal2D, title: SE, dimensions: (|100, 75)>,
    <EDSSEMSpectrum, title: EDX, dimensions: (100, 75|1024)>]

The loaded array energy dimension can by forced to be larger than the data
recorded by setting the 'cutoff_at_kV' kwarg to higher value:

.. code-block:: python

    >>> hs.load("sample80kv.bcf", cutoff_at_kV=60)
    [<Signal2D, title: BSE, dimensions: (|100, 75)>,
    <Signal2D, title: SE, dimensions: (|100, 75)>,
    <EDSSEMSpectrum, title: EDX, dimensions: (100, 75|3072)>]

loading without setting ``cutoff_at_kV`` value would return data with all 4096
channels. Note that setting ``downsample`` higher than 1 currently locks out using SEM
images for navigation in the plotting.
