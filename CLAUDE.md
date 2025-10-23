# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Building the Project
- `uv sync --group dev` - Install dependencies and development tools
- `python setup.py build_ext --inplace` - Build Cython extensions in place for development
- `python setup.py install` - Full installation with Cython compilation

### Testing
- `pytest` - Run all tests (automatically drops into debugger on failure due to `--pdb` flag in pyproject.toml)
- `pytest src/weathercop/tests/test_vine.py` - Run specific test module
- `pytest -k "test_name"` - Run specific test by name
- `pytest -x` - Stop after first failure

### Code Quality
- `flake8 src/` - Run linting
- `black --line-length 79 src/` - Format code (note: line length is 79, not default 88)
- `black --check --line-length 79 src/` - Check formatting without making changes

### Documentation
- `make flowchart` - Regenerate flowchart PNG from LaTeX source (creates `img/weathercop_workflow.png`)
- `docs/regenerate_flowchart.sh` - Direct script for flowchart regeneration

## Architecture Overview

### Core Components

**Vine Copulas (`src/weathercop/vine.py`)**
- `CVine`: Canonical vine copulas with central node structure
- `RVine`: Regular vine copulas using minimum spanning trees
- `MultiStationVine`: Container for multiple vine models across weather stations
- Core functionality for tree construction, copula fitting, simulation, and visualization

**Copula Library (`src/weathercop/copulae.py`)**
- Bivariate copula implementations (Clayton, Gumbel, Joe, Plackett, etc.)
- Automatic code generation using SymPy for performance-critical functions
- Generated Cython extensions in `src/weathercop/ufuncs/` for fast computation
- Uses `ufuncify` to create fast NumPy universal functions from symbolic expressions

**Seasonal Copulas (`src/weathercop/seasonal_cop.py`)**
- `SeasonalCop` class wraps copulas with time-varying parameters
- Fits copula parameters using sliding windows over day-of-year
- Uses Fourier series approximation to smooth seasonal parameter variations
- Supports both manual copula specification and automatic selection via maximum likelihood

**Multisite Weather Generation (`src/weathercop/multisite.py`)**
- Weather generation workflows combining vine copulas with time series analysis
- Integration with VARWG library for managing marginal transformations
- Phase randomization methods (`varwg_ph`) for temporal dependence
- Parallel processing support for large ensemble generation

**Configuration (`src/weathercop/cop_conf.py`)**
- Global configuration settings for the package
- Debug flags (`PROFILE`, `DEBUG`), numerical tolerances, and computational parameters
- Path configurations for cache directories and temporary files
- Multiprocessing pool size (`n_nodes`) defaults to CPU count - 2

### Key Dependencies
- **VARWG Library**: Custom dependency for time series analysis and weather generation (`varwg = { git = "https://github.com/iskur/varwg" }`)
- **Cython**: Used for performance-critical numerical computations
- **SymPy**: Automatic generation of copula functions and derivatives
- **XArray/Pandas**: Data handling and analysis
- **Cartopy**: Geospatial visualization

### Build System
The project uses a hybrid build system:
- `pyproject.toml` defines modern Python packaging with uv as package manager
- `setup.py` handles complex Cython extension building with automatic code generation
- Extensions are built from both manually written `.pyx` files (e.g., `cvine.pyx`, `cinv_cdf.pyx`, `normal_conditional.pyx`) and auto-generated code in `ufuncs/`
- The `BuildExt` class defers Cython compilation until build time to avoid unnecessary rebuilds
- Auto-generated extensions combine `.pyx` wrapper files with SymPy-generated C code

### Testing Structure
- Tests are in `src/weathercop/tests/` following pytest conventions
- Test data generation utilities in `generate_test_data.py`
- Key test modules: `test_vine.py`, `test_copulae.py`, `test_seasonal_cop.py`, `test_multisite.py`
- Tests use both unittest-style (numpy.testing.TestCase) and pytest-style assertions

## Important Notes
- The codebase requires Python >=3.13
- Cython extensions must be built before running tests or using the package
- The `ufuncs/` directory contains auto-generated code - do not edit manually
- Configuration is imported as `cop_conf` throughout the codebase
- VARWG is installed from PyPI as a standard dependency
- When modifying copula implementations, regenerate Cython extensions with `python setup.py build_ext --inplace`

## Common Issues

### Pytest Triggering Recompilation
If pytest runs slowly or appears to hang, it's because the copulae module is auto-generating and compiling Cython extensions on first import. This happens when:
1. The required `.so` files don't exist in `src/weathercop/ufuncs/`
2. Missing modules trigger the fallback compilation using `sympy.autowrap.ufuncify()`

**Solutions:**
- Pre-build extensions: `python setup.py build_ext --inplace`
- Or wait for first-run compilation to complete (can take 5-10 minutes)
- The compilation results are cached for subsequent runs

### Debugging Import Issues
- Recompilation during pytests was fixed by importing "weathercop.ufunc.<name>" instead of "<name>"
- If imports fail, check that the `ufuncs/` directory exists and contains `__init__.py`
- The `copulae.py` module automatically creates the ufuncs directory if missing

### Performance Optimization
- Set `cop_conf.PROFILE = True` to disable multiprocessing for debugging with profilers
- Vine construction can be parallelized by setting `weights="likelihood"` (slower but more accurate)
- Default `weights="tau"` uses Kendall's tau for faster tree construction
- For large datasets, consider using `CVine` with a specified `central_node` instead of `RVine`

### Working with Vine Copulas
- Vine array `A` uses natural ordering - variables are reordered internally for efficient computation
- Original variable order is preserved in `varnames_old` attribute
- Use `simulate()` and `quantiles()` methods on vines - they handle variable reordering automatically
- Edge copulas are accessed via `vine[row, col]` indexing on the vine array
- The `name` property on vines is used in plot titles (set via `name` parameter or direct assignment)
- main entry point: @src/weathercop/multisite.py::Multisite
- VARWG is a single-site, WeatherCop is multisite. WeatherCop heavily depends on VARWG - it orchestrates VARWG instances and replaces their model via call-back functions. This lets WeatherCop deal with dependencies and VARWG fits distribution to the marginals and does the variable transform before and after WeatherCop simulation.