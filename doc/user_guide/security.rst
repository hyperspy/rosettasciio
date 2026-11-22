.. _security-untrusted-files:

Security and untrusted files
============================

RosettaSciIO readers process complex file formats whose metadata can control
memory allocation, decompression, recursion, and access to other files. Treat
shared or downloaded files as attacker-controlled unless you trust their source
and integrity. This applies both to direct RosettaSciIO readers and to loading
the same files through HyperSpy.

**Only open files you trust.** Validating a filename extension or successfully
opening a file does not make its contents safe.

Resource exhaustion
-------------------

Crafted files can declare excessive dimensions, element counts, or compressed
content. Reading or decompressing them can exhaust memory or CPU, terminate the
process, or make the system unresponsive. RosettaSciIO does not impose general
allocation or decompression limits. The following ten components have confirmed
resource-exhaustion risks:

* TIA SER: file-declared array sizes.
* EDAX SPD: file-declared array sizes.
* Hamamatsu IMG: file-declared array sizes.
* Semper UNF: file-declared array sizes.
* TOFWERK event lists: file-declared event counts and their processing.
* ZSPY shape metadata: file-declared array shapes.
* ZSPY ZipStore: decompression of archive members and data chunks.
* NumPy ``.npy``: file-declared array shapes.
* HSPY HDF5: dataset sizes and decompression.
* PantaRhei NPZ: array shapes and compressed archive data.

Lazy loading doesn't remove these risks. Metadata processing can allocate
resources while a file is opened, and data access or computation can trigger
further allocation and decompression later.

A crafted NeXus HDF5 file can also contain a hard-link cycle. Recursive link
traversal without cycle detection can repeatedly visit the same objects and
exhaust CPU or memory. The NeXus reader doesn't currently guard against this
cycle, so only open NeXus files you trust.

XML entity bombs
----------------

Systems using expat 2.8.x or later reject XML entity bombs. No additional
action is needed on current Python versions using such an expat release.

HDF5 external references
------------------------

HDF5 external raw storage, external links, and virtual-dataset sources remain
unaddressed in this release. This residual risk applies to **every HDF5-backed
reader and every public HDF5 inspection helper**, including
``rsciio.utils.hdf5.list_datasets_in_file`` and
``rsciio.utils.hdf5.read_metadata_from_file``.

Crafted HDF5 files can use these mechanisms to cause local file disclosure.
This finding doesn't demonstrate code execution.

Open only trusted HDF5 files, or use genuine OS-enforced filesystem sandboxing
that denies access outside an allowed directory. A subprocess alone isn't a
sandbox.

Self-contained HDF5 files remain valid for data distribution. The danger is
opening HDF5 from untrusted sources, not publishing HDF5 itself.

Zarr/ZSPY doesn't have this HDF5-specific external-reference mechanism. It
isn't universally safe because unsafe-codec and resource-exhaustion risks
remain.
