.. _de-format:

StreamPix .SEQ ()
-----------------

The .seq file format is a binary file format developed by
:ref:`Streampix` <https://www.norpix.com/products/streampix/streampix.php>. Streampix is
a high speed digital recording software.

Due to the flexible nature of the .seq file format we specifically support
the .seq file format associated with the DirectElectron-16 and DirectElectron-Celeritas cameras.

For other cameras support may not be fully realized but if an issue is raised
:ref: `here` <https://github.com/hyperspy/rosettasciio/issues> support can be
considered.

Specifically for the support of the DirectElectron-Celeritas camera there are two
possible ways to load some file.

1: Explicitly list files
.. code-block:: python
    >>> from rsciio.de import file_loader
    >>>
    >>> file_loader(None, top="de_Top.seq", bottom="de_Botom.seq"
    >>>             metadata="de_Top.metadata",gain="de.gain.mrc",
    >>>             dark="de.dark.mrc", xml="de.seq.Config.Metadata.xml",
    >>>             celeritas=True)

2: Automatically detect the files. In this case the program will automatically
   look for files with the same naming structure in the same folder.
.. code-block:: python
 >>> from rsciio.de import file_loader
    >>>
    >>> file_loader(top="de_Top.seq", celeritas=True)


All of the file loaders for the cameras also have a special `distributed` keyword which changes how the data is
loaded into memory or into a dask array.  This allows for the user to use
:ref: `dask-distributed` <https://distributed.dask.org/en/stable/> as a backend rather
than using the default scheduler.
