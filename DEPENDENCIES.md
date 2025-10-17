# Dependencies

This document describes the dependencies used in WeatherCop and their purpose.

## Python Version

WeatherCop requires **Python ≥ 3.13**.

## Core Dependencies

### Scientific Computing & Numerics

- **NumPy** (≥2.3.1) - Core numerical computing library for array operations
- **SciPy** (≥1.16.0) - Scientific computing functions (statistics, optimization, integration)
- **SymPy** (≥1.14.0) - Symbolic mathematics for automatic code generation of copula functions
- **numexpr** (≥2.11.0) - Fast numerical expression evaluator

### Data Handling & Analysis

- **pandas** (≥2.3.0) - Data manipulation and time series analysis
- **xarray** (≥2025.6.1) - Labeled multi-dimensional arrays for climate/weather data
- **NetCDF4** (≥1.7.2) - Reading/writing NetCDF files (standard meteorological data format)

### Visualization

- **matplotlib** (≥3.10.3) - Plotting library for copula visualizations and diagnostics
- **Cartopy** (≥0.24.1) - Geospatial data processing and map plotting

### Graph & Network Analysis

- **NetworkX** (≥3.5) - Graph data structures for vine copula tree construction

### Utilities

- **dill** (≥0.4.0) - Enhanced pickling for complex Python objects
- **tqdm** (≥4.67.1) - Progress bars for long-running computations
- **timezonefinder** (≥6.5.9) - Timezone lookup from coordinates

### Weather Generation

- **VG** (from GitHub: https://github.com/iskur/vg) - Custom library for weather generation and time series analysis
  - **IMPORTANT:** Currently installed from GitHub; PyPI publication pending
  - Required for temporal structure preservation and phase randomization
  - See README.md troubleshooting for details

## Build Dependencies

Required for building Cython extensions:

- **Cython** (≥3.1.2) - Compiling performance-critical numerical code
- **setuptools** (≥80.9.0) - Python packaging tools
- **NumPy** (build-time) - Required for Cython compilation

## Development Dependencies

For contributors and development work:

- **pytest** (≥8.4.1) - Testing framework
- **flake8** (≥7.3.0) - Code style linting
- **IPython** (≥9.5.0) - Enhanced interactive Python shell
- **pre-commit** (≥4.0.0) - Git hooks for automatic README generation
- **pyflakes** (≥3.4.0) - Python code checker

## Optional Dependencies

- **pyFFTW** (≥0.15.0) - Faster FFT operations (optional performance enhancement)

## Installation

Install all dependencies using uv:

```bash
# Core dependencies
uv sync

# With development tools
uv sync --group dev

# With optional dependencies
uv sync --group optional
```

## First Import Compilation

On first import, WeatherCop automatically compiles Cython extensions for copula functions.
This takes 5-10 minutes but only happens once. Compiled extensions are cached for subsequent imports.

To pre-compile extensions:

```bash
python setup.py build_ext --inplace
```

## Dependency Notes

### Why VG from GitHub?

The VG library is a companion project for weather time series analysis. It will be published to PyPI
in the future, which will allow WeatherCop to also be published to PyPI. Currently, WeatherCop
can only be installed from GitHub since PyPI does not support git-based dependencies.

### Why Python 3.13?

WeatherCop uses modern Python features and requires up-to-date versions of scientific libraries
that are optimized for Python 3.13+.

### Cython Compilation

WeatherCop uses Cython to generate high-performance C code for copula density and CDF computations.
SymPy automatically generates these functions from symbolic expressions, ensuring mathematical
correctness while maintaining performance.
