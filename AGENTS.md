# WeatherCop Agent Guidelines

## Build / Lint / Test
- **Build**: `uv sync --group dev && python setup.py build_ext --inplace`
- **Lint**: `flake8 src/` and `black --check --line-length 79 src/`
- **Run all tests**: `pytest`
- **Run a single test file**: `pytest <path/to/test_file.py>`
- **Run a specific test function**: `pytest -k "<test_name>"` or `pytest <file>::<function>`

## Code Style
- Imports: standard library first, then third‑party, then local modules; use absolute imports.
- Formatting: 79‑char line length, `black` style, no trailing whitespace.
- Type hints: use `typing` annotations everywhere; avoid `Any` unless necessary.
- Naming: classes `CamelCase`, functions/variables `snake_case`; constants `UPPER_SNAKE_CASE`.
- Error handling: raise descriptive exceptions, use context managers for resources.
- Docstrings: follow NumPy style, include parameter and return types.

## Additional Rules
- No global mutable state; prefer dependency injection.
- Cython extensions must be rebuilt with `python setup.py build_ext --inplace` after changes.
- Keep tests deterministic; use fixtures from `src/weathercop/tests/conftest.py`.
