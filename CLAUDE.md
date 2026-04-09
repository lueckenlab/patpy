# CLAUDE.md

## Project Overview

**patpy** is a Python toolbox for patient/sample-level representation from single-cell data. It provides interfaces to sample representation methods, analysis functions, and evaluation metrics. Source lives in `src/patpy/` with submodules: `tl/` (tools/analysis), `pp/` (preprocessing), `pl/` (plotting), `datasets/` (synthetic data).

## Build & Install

- **Build system:** Hatchling
- **Package manager:** uv
- **Python:** >=3.10

```bash
pip install -e ".[dev,test]"
pre-commit install
```

## Running Tests

```bash
pytest                          # Run all tests
pytest tests/test_specific.py   # Run a specific file
pytest -x                       # Stop on first failure
```

CI uses hatch: `uvx hatch run hatch-test:run-cov`

Test fixtures are in `tests/conftest.py` (session-scoped `synthetic_adata`, `pbmc3k_adata`, matrix fixtures).

Mark slow integration tests with `@pytest.mark.helical`.

## Linting & Formatting

Pre-commit hooks handle all checks. Run manually:

```bash
pre-commit run --all-files      # Run all hooks
ruff check src/ tests/          # Lint only
ruff format src/ tests/         # Format only
```

### Key lint rules (ruff)

- **Line length:** 120
- **Docstrings:** NumPy convention
- **Selected rules:** B, BLE, C4, D, E, F, I, RUF100, TID, UP, W
- **Ignored:** D100, D104, D105, D107, D203, D213, D400, D401, E501, E731, E741
- Tests (`tests/*`) are exempt from docstring rules (D)
- `__init__.py` files allow unused imports (F401)

## Code Style

- Use `from __future__ import annotations` at the top of modules
- NumPy-style docstrings (no period on first line, imperative mood not enforced)
- Type hints with modern syntax (`str | None`, not `Optional[str]`)
- Line length: 120 characters
- 4-space indentation (Python), 2-space (YAML)
- LF line endings

## Project Structure

```
src/patpy/
  tl/          # Tools: sample representation, supervised methods, evaluation
  pp/          # Preprocessing
  pl/          # Plotting
  datasets/    # Synthetic data generation
tests/         # pytest tests (mirrors src structure)
docs/          # Sphinx docs (MyST markdown)
```

## CI/CD

- **Tests:** GitHub Actions on push/PR to main (Python 3.10, 3.13 + pre-release)
- **Coverage:** Codecov
- **Releases:** Tag-triggered PyPI publishing via trusted OIDC
- **Pre-commit.ci:** Auto-fixes formatting on PRs
