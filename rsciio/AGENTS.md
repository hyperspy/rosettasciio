# rsciio package

## OVERVIEW
RosettaSciIO's importable package: shared hierarchical IO machinery plus filesystem-discovered file-format plugins.

**Location score: 10** = 0 file-count + 2 subdirectory + 2 code-ratio + 0 unique-config + 2 module-boundary + 2 symbol-density + 2 export-count + 0 centrality. Evidence excludes generated `AGENTS.md` files: 7 direct source files; 41 direct subdirectories, of which 39 contain `specifications.yaml` and the others are `utils/` and `tests/`; 100% code files; no own config; `__init__.py`; 54 direct definitions; 13 direct-module exports. Reference centrality is unmeasured; distinct package architecture.

## STRUCTURE
- `__init__.py`: lazy top-level facade for `__version__`, `IO_PLUGINS`, and `set_log_level`.
- `_io_plugins.py`: walks the package and loads every `specifications.yaml` into `IO_PLUGINS`.
- `_hierarchical.py`: shared HyperSpy HDF5/Zarr reader/writer base, ragged-array conversion, and chunk calculation.
- `_docstrings.py`: shared NumPy-style documentation fragments interpolated by plugin APIs.
- `exceptions.py`: domain-specific format exceptions and `VisibleDeprecationWarning`.
- `<format>/`: one directory per file format; `utils/`: shared helpers; `tests/`: tests and fixture registry.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add a format | `<format>/__init__.py`, `_api.py`, `specifications.yaml` | No central registry edit; discovery is by filesystem walk. |
| Change public imports | `__init__.py`, plugin `__init__.py` | `test_import.py` asserts exact `dir()` results. |
| Change HDF5/ZSpy storage | `_hierarchical.py`, `hspy/_api.py`, `zspy/_api.py` | Current hierarchical format version is `3.3`. |
| Change lazy binary IO | `utils/file.py`, relevant plugin `_api.py` | Lazy handles often must remain open after return. |
| Edit API docs | `_docstrings.py` | Plugins apply `%` interpolation at import time. |
| Optional compiled Bruker path | `bruker/unbcf_fast.pyx`, `bruker/_api.py` | Pure-Python fallback must remain functional. |

## CONVENTIONS
- Plugin `__init__.py` re-exports public functions from `_api.py`, defines `__all__`, and returns `sorted(__all__)` from `__dir__()`.
- Reader implementations return a list of signal dictionaries containing `data`, `axes`, `metadata`, and `original_metadata`; optional `mapping` and `post_process` entries are format-specific.
- `specifications.yaml` supplies registration metadata and capabilities; `default_extension` is an index, and `writes` is either a boolean or a dimension-pair matrix.
- Keep optional format dependencies out of package import time; readers probe or import them when needed.
- Use shared docstring constants rather than duplicating common parameter and return documentation.
- Implementation modules are underscore-prefixed; the package facade is the public compatibility boundary.

## ANTI-PATTERNS
- Do not add a plugin name to a static registry: discovery depends on its directory and YAML specification.
- Do not casually add exports to `__all__`; exact public directory listings are tested.
- Do not close file/store handles needed by returned lazy arrays.
- Do not assume all plugins support lazy reads or writes; enforce their declared capability and existing exception behavior.
- Do not reintroduce removed `record_by` reader attributes.
- Preserve the Bruker pure-Python fallback when changing the optional Cython acceleration.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
