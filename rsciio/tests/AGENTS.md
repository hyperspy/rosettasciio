# rsciio tests

## OVERVIEW
Plugin contract tests, format fixtures, generated binary test inputs, shared utility tests, and an xdist-safe pooch fixture registry.

**Location score: 10** = 3 file-count + 0 subdirectory + 2 code-ratio + 1 unique-config + 2 module-boundary + 2 symbol-density + 0 export-count + 0 centrality. Evidence excludes generated `AGENTS.md` files: 58 direct files; 2 direct subdirectories; 57 code/config files (98.3%); `conftest.py`; `__init__.py`; 1426 direct definitions; no public export surface. Reference centrality is unmeasured; distinct test-infrastructure domain.

## STRUCTURE
- `test_<format>.py`: principal format test modules; `test_import.py` pins public APIs and YAML schemas.
- `utils/`: tests for shared utilities rather than a specific file format.
- `data/<format>/`: local fixtures, including intentionally malformed and vendor-specific layouts.
- `registry.py`, `registry_utils.py`, `registry.txt`: pooch configuration, hash maintenance, and fixture download/copy logic.
- `conftest.py`: autouse session fixture that makes test data available once across pytest-xdist workers.
- `generate_*`, `create_*`: fixture generators, some requiring external programs or original source data.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add a format test | `test_<format>.py`, `data/<format>/` | Exercise public plugin behavior and format-specific edge cases. |
| Change package exports | `test_import.py` | Exact `dir()` assertions are intentional API contracts. |
| Add/update fixture data | `data/`, `registry.txt` | Regenerate SHA-256 entries; keep fixtures minimal. |
| Change fixture acquisition | `conftest.py`, `registry.py`, `registry_utils.py` | Preserve the lock/sentinel used by xdist. |
| Test shared helpers | `utils/` | Run this directory independently for utility-only changes. |
| Test lazy rejection/support | `test_lazy_not_implemented.py`, format tests | Lazy rejection commonly occurs before opening the filename. |

## CONVENTIONS
- Put `pytest.importorskip(...)` before imports requiring an optional dependency; following imports carry `# noqa: E402`.
- Parametrize both lazy and eager behavior where supported, and often exercise integration through `hyperspy.api` as well as direct plugin calls.
- Fixture paths derive from `Path(__file__).parent / "data" / <format>`; generated files use temporary directories or explicit teardown.
- Deprecation tests assert `VisibleDeprecationWarning`; metadata tests commonly use `rsciio.utils._tests` helpers.
- ZSpy fixtures use Zarr v2 DirectoryStore, NestedDirectoryStore, and ZipStore layouts; do not silently upgrade their metadata.

## ANTI-PATTERNS
- Never "repair" intentionally invalid fixtures or normalize case-sensitive vendor extensions without updating the corresponding error-path tests.
- Do not add/update data without running `python rsciio/tests/registry_utils.py`; registry generation is unsupported on Windows.
- Do not bypass the registry with ad hoc downloads in new tests.
- Do not import `create_topspin_data.py`: it executes side-effecting generation code at module import.
- Do not remove optional-dependency guards merely because the full development environment installs every extra.
- Avoid new tests coupled to private `_api` internals unless binary-layout behavior cannot be checked through the public reader/writer contract.

## COMMANDS
```bash
python -m pytest rsciio/tests/test_<format>.py
python -m pytest rsciio/tests/utils
python -m pytest rsciio/tests/test_import.py rsciio/tests/test_lazy_not_implemented.py
python rsciio/tests/registry_utils.py
```
Set `RSCIIO_TEST_DATA_SOURCE` to copy fixtures from a local directory or `POOCH_BASE_URL` to override downloads.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
