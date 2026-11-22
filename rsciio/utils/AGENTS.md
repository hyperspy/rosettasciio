# rsciio utilities

## OVERVIEW
Shared binary IO, HDF5, path, RGB, XML, array, metadata, deprecation, and lazy/distributed helpers used across format plugins.

**Location score: 11** = 3 file-count + 0 subdirectory + 2 code-ratio + 0 unique-config + 2 module-boundary + 2 symbol-density + 2 export-count + 0 centrality. Evidence excludes generated `AGENTS.md` files: 24 direct Python files; no direct subdirectories; 100% code files; no own config; `__init__.py`; 101 direct definitions; 64 direct-module exports. Reference centrality is unmeasured; distinct cross-format utility domain.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| File handles and distributed memory maps | `file.py`, `_distributed.py` | `memmap_distributed` backs many lazy readers. |
| HDF5 inspection | `hdf5.py` | Dataset listing and metadata reading; intentional CLI-style printing. |
| Filename/path operations | `path.py` | Includes overwrite prompting and filename incrementing. |
| Structured arrays and dictionaries | `_array.py`, `_dictionary.py` | `DTBox`, `sarray2dict`, and `dict2sarray`. |
| XML parsing | `xml.py` | Current home of `XmlToDict`, `xml2dtb`, and related helpers. |
| FEI event streams | `_fei_stream_readers.py` | Dense/sparse COO conversion with optional numba acceleration. |
| Deprecating APIs or arguments | `_deprecated.py`, compatibility shims | Uses `VisibleDeprecationWarning`. |
| Test assertions shared by plugin tests | `_tests.py` | Includes `assert_deep_almost_equal` and `normalize_micro`. |

## CONVENTIONS
- `__init__.py` lazily exposes only `file`, `hdf5`, `path`, `rgb`, `tools`, and `xml`; implementation modules are otherwise underscore-prefixed.
- `tools.py`, `distributed.py`, `exceptions.py`, and `rgb_tools.py` are compatibility shims that redirect imports and warn of removal in version 1.0.
- Add moved names to a shim's `_import_mapping` when compatibility is required; do not duplicate their implementations.
- New deprecations use `deprecated` or `deprecated_argument`; `distributed`, `mmap_mode`, and the legacy endianness keyword already have dedicated argument decorators.
- Low-level binary readers live in `_readfile.py`; format plugins should reuse these primitives instead of creating another binary-read layer.
- Public utility module facades use PEP 562 `__getattr__` and sorted `__dir__()` to keep imports lazy.

## ANTI-PATTERNS
- Do not import new code from deprecated `tools`, `distributed`, `exceptions`, or `rgb_tools` shims; use their replacement modules.
- Do not remove compatibility warnings or aliases before their stated version-1.0 removal boundary.
- Do not make numba mandatory: `jit_ifnumba` preserves a non-numba path.
- Do not replace intentional deprecation warnings with silent aliases; tests assert warning behavior.
- Preserve lazy/distributed array semantics when changing file helpers; eager copies can break plugin lazy-loading contracts.
- `path.overwrite` is interactive by design; avoid calling it from non-interactive internals without an explicit overwrite decision.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
