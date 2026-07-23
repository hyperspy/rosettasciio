The ``zspy`` writer now consolidates metadata by default using
``zarr.consolidate_metadata`` (or ``zarr.convenience.consolidate_metadata`` for zarr v2).
This bundles all individual ``.zattrs``, ``.zarray`` and ``.zgroup`` metadata files
(~20+ files for a typical signal) into a single ``.zmetadata`` entry, reducing the number
of HTTP requests needed to open a file from remote storage by an order of magnitude.
Set ``consolidate=False`` to disable this behaviour (e.g. for interactive workflows where
metadata is inspected manually).

(Rename this file to ``<ISSUE>_or_<PR>.enhancements.rst`` before merging.)
