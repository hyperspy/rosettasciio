Reject files whose ZSPY arrays select the Pickle codec. The check reads the
selected raw array metadata, including the N5 codec mapping that Zarr itself
builds, before any array or codec is constructed.
