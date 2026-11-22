# PROJECT KNOWLEDGE BASE

**Generated:** 2026-11-22T09:45:57.740Z
**Commit:** 405e1637
**Branch:** ai-policy-agents

## OVERVIEW
RosettaSciIO is a GPL-3.0-or-later Python scientific file-format IO library. Format plugins, shared utilities, hierarchical HDF5/Zarr support, and tests live under `rsciio/`.

## STRUCTURE
```
rosettasciio/
├── rsciio/              # Format packages, shared IO machinery, utilities, and tests
│   ├── utils/           # Cross-format utilities and compatibility facades
│   └── tests/           # Test suite, fixture registry, and format data
├── doc/                 # Sphinx sources and per-format documentation
├── upcoming_changes/    # Towncrier fragments
├── pyproject.toml       # Packaging, test, Ruff, coverage, and towncrier configuration
└── setup.py             # Optional Bruker Cython extension
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add or modify a format plugin | `rsciio/<format>/` | Usually `__init__.py`, `_api.py`, and `specifications.yaml`; larger formats add private modules |
| Understand plugin discovery | `rsciio/_io_plugins.py` | Walks packages and loads specifications into `IO_PLUGINS`; no static registry |
| Change hierarchical IO | `rsciio/_hierarchical.py` | Readers, writers, flattening, unflattening, and chunk selection |
| Find shared helpers | `rsciio/utils/` | Lazy facade plus file, HDF5, path, RGB, tools, and XML implementations |
| Change public utility exports | `rsciio/utils/__init__.py` | Lazy submodule facade and compatibility shims |
| Work with tests and fixtures | `rsciio/tests/` | Autouse session fixture, pooch registry, and per-format data |
| Update fixture tracking | `rsciio/tests/registry.txt` | Hash-tracks files under `rsciio/tests/data/<format>/` |
| Regenerate the test registry | `rsciio/tests/registry_utils.py` | Provides update, creation, and download helpers |
| Configure quality or releases | `pyproject.toml`, `upcoming_changes/` | Central tool configuration and towncrier fragments |
| Build documentation | `doc/` | Sphinx sources and per-format pages |

## CODE MAP
Reference centrality is unmeasured: LSP document symbols were unavailable, so observed imports do not provide reference counts.

| Symbol | Type | Location | Refs | Role |
|--------|------|----------|------|------|
| `__version__`, `IO_PLUGINS`, `set_log_level` | Exports | `rsciio` | Unmeasured | Lazy PEP 562 top-level facade |
| `HierarchicalReader`, `HierarchicalWriter` | Classes | `rsciio/_hierarchical.py` | Unmeasured | Hierarchical format IO; version `3.3` |
| `flatten_data`, `unflatten_data`, `get_signal_chunks` | Functions | `rsciio/_hierarchical.py` | Unmeasured | Hierarchical data conversion and chunking |
| `memmap_distributed`, `DTBox`, `dict2sarray`, `sarray2dict` | Utilities | `rsciio/utils/` | Unmeasured | Distributed arrays and structured-array conversion |
| `XmlToDict` | Utility | `rsciio/utils/` | Unmeasured | XML conversion |
| `jit_ifnumba`, `deprecated`, `deprecated_argument` | Utilities | `rsciio/utils/` | Unmeasured | Optional acceleration and compatibility decorators |
| `session_data` | Fixture | `rsciio/tests/conftest.py` | Unmeasured | Autouse, file-locked session setup for xdist |
| `TEST_DATA_REGISTRY` | Registry | `rsciio/tests/registry.py` | Unmeasured | Pooch test-data registry |
| `update_registry`, `make_registry`, `download_all` | Functions | `rsciio/tests/registry_utils.py` | Unmeasured | Fixture registry maintenance |
| `file_reader`, `file_writer` | Plugin exports | `rsciio/<format>/_api.py` | Unmeasured | Standard readers and, where supported, writers |

## CONVENTIONS
- Plugin `__init__.py` re-exports `_api.py`, declares `__all__`, and returns `sorted(__all__)` from `__dir__()`.
- Discovery injects `api = "rsciio.<directory>"` from package layout.
- Readers return lists of signal dictionaries containing `data`, `axes`, `metadata`, and `original_metadata`.
- Writer capabilities and lazy support are format-specific.
- Common docstrings use `%` interpolation from `rsciio._docstrings` at import time.
- Optional dependencies are guarded with `pytest.importorskip` before project imports and `# noqa: E402` afterward.
- `default_extension` in `specifications.yaml` is an index; `writes` is a boolean or dimension-pair matrix.
- Deprecated `tools`, `distributed`, `exceptions`, and `rgb_tools` shims emit `VisibleDeprecationWarning` for version-1.0 removal.
- Test data is pruned from distributions; deliberately malformed fixtures and case-sensitive vendor layouts are intentional.

## ANTI-PATTERNS (THIS PROJECT)
- Do not add static plugin registry entries or casually expand `__all__`; `test_import.py` checks exact public `dir()` results.
- Do not close handles needed by lazy arrays, advertise unsupported capabilities, or reintroduce removed `record_by` attributes.
- Do not import new code through deprecated utility shims or remove their warnings early.
- Never make optional numba/Cython acceleration the only implementation path.
- Do not change fixtures without updating `registry.txt`; keep fixtures small and preserve intentional malformed files.
- Do not ship `rsciio/tests/data`; `MANIFEST.in` deliberately excludes it.
- Do not import `rsciio/tests/create_topspin_data.py`; import executes generation side effects.
- Do not hand-edit the setuptools-scm version or `CHANGES.rst`.
- Library code must not use `print` without an intentional Ruff exemption.

## UNIQUE STYLES
- Filesystem-driven discovery replaces a central static plugin registry.
- Public package and utility surfaces use lazy facades.
- Test fixtures use a hash registry plus an autouse, file-locked session fixture compatible with xdist.
- Registry regeneration is unsupported on Windows.

## COMMANDS
```bash
pip install -e .
pip install -e '.[all,tests]'
pytest
pytest --pyargs rsciio --reruns 3 -n 2
python -m pytest rsciio/tests/test_<format>.py
python -m pytest rsciio/tests/utils
python rsciio/tests/registry_utils.py
python setup.py recythonize
ruff check --fix
ruff format
make -C doc html
```

## NOTES
- Pytest defaults enable xdist auto parallelism and `--dist loadfile`.
- `DISABLE_C_EXTENSIONS` disables the optional extension.
- `RSCIIO_TEST_DATA_SOURCE` copies local fixtures instead of downloading them.
- `POOCH_BASE_URL` overrides the test-data download source.
- Nested guidance exists for `rsciio/`, `rsciio/utils/`, and `rsciio/tests/`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->

### RSCIIO contributor notes
- For local development, run `python -m pip install -e '.[all,tests]'`. The documented user-facing all-extras installation in `doc/user_guide/install.rst` is `pip install rosettasciio[all]`.
- Keep optional dependencies and the `all` extra in `pyproject.toml` synchronized with the selectors and oldest-version `DEPENDENCIES` pins in `.github/workflows/tests.yml` when adding or changing an extra.
- Discover plugins, reader/writer entry points, and tests instead of maintaining a hard-coded inventory:
  ```bash
  find rsciio -mindepth 2 -maxdepth 2 -name specifications.yaml -print
  grep -RInE '^def (file_reader|file_writer)\b' rsciio --include='*.py'
  find rsciio/tests -type f -name 'test_*.py' -print
  python -m pytest --collect-only -q rsciio/tests
  ```
- When changing a reader family, enumerate every native-library open site in its modules and shared helpers, including public inspection helpers such as `list_datasets_in_file` and `read_metadata_from_file` in `rsciio/utils/hdf5.py`, not only `file_reader`; start with imports and open-like calls (replace `FORMAT`):
  ```bash
  format=FORMAT; grep -RInE '^(from|import) |\b(open|File|TiffFile)\s*\(|\.(open|imread|memmap)\s*\(' "rsciio/$format" rsciio/utils --include='*.py'
  ```
- Test native-library reads in eager and lazy modes, and force lazy data/materialization and metadata access after the reader returns; open-time checks alone do not cover read-time behavior.
- Install the relevant extras and inspect `pytest.importorskip` sites and skip summaries under `rsciio/tests`; investigate unexpected optional-dependency skips rather than accepting them as passing coverage.
<!-- hyperspy-ai-policy:begin -->
## HyperSpy AI policy (ecosystem contract)

Canonical source: https://github.com/hyperspy/.github/blob/main/AI-POLICY.md - do not reword
this block; the `check-agents-policy` CI script keeps it synchronized.

- Non-trivial AI-assisted changes need an accepted proposal in
  https://github.com/hyperspy/hyperspy-proposals BEFORE the implementation PR is reviewed.
  Trivial changes: PR review suffices.
- Every AI-assisted commit carries the trailer `Assisted-by: <tool>:<model>`
  (example: `Assisted-by: OpenCode:deepseek-4.0-pro`). Attribute the tool you interact with,
  not a wrapper.
- NEVER add `Co-authored-by:` trailers for AI tools; they are reserved for human co-authors.
  The `check-ai-co-author` commit-msg hook and the `compliance / ai-trailers` CI check reject
  them.
- Disable your tool's automatic co-author injection before committing: Claude Code
  `"includeCoAuthoredBy": false`; GitHub Copilot `github.copilot.chat.commitMessageGeneration`
  off; Cursor commit attribution off; OpenCode / oh-my-openagent: check skill and plugin configs.
- Run `pre-commit install` (installs the commit-msg stage) before your first commit; CI
  re-checks every PR commit regardless.
- Declare AI assistance in the pull-request template and link the accepted proposal when one
  is required.
<!-- hyperspy-ai-policy:end -->
