# What is WeatherCop?

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Managed with uv](https://img.shields.io/badge/managed%20with-uv-blue)](https://docs.astral.sh/uv/)

WeatherCop is a multisite weather generator based on vine copulas. It
was developed to generate synthetic weather scenarios that preserve both
spatial dependencies across weather stations and temporal structure
within each station. The package combines statistical copula theory with
time series analysis from the VarWG library to create realistic weather
ensembles for hydrodynamic and ecological modeling.

## Architecture Overview

![](./img/weathercop_workflow.png)

### How It Works

WeatherCop operates through a carefully choreographed workflow combining
two complementary systems:

1.  **VarWG Component** (marginal distributions):
    - Fits KDE/parametric distributions to each weather variable at each
      site, accounting for seasonal variations (F<sub>doy</sub>)
    - Transforms measurement data to standard-normal space via inverse
      CDF of day-of-year distributions: Φ⁻¹(F<sub>doy</sub>(X))
    - Handles dryness probability estimation for precipitation
    - Applies the inverse transformation to convert simulated data back
      to the measurement domain: X = F<sub>doy</sub>⁻¹(Φ(Y))
2.  **WeatherCop Component** (dependence structure):
    - Fits vine copulas to deseasonalized observations to capture
      inter-variate dependencies (relationships between different
      weather variables)
    - Decorrelates observations for phase randomization
    - Re-correlates synthetic data using the fitted copula structure
    - Preserves both temporal autocorrelation within each station and
      spatial cross-correlations between stations through phase
      randomization

This separation of concerns enables realistic what-if scenarios: by
conditioning on a guiding variable (e.g., temperature), changes
propagate to other variables through the vine copula's inter-variate
dependence structure, while VarWG ensures each variable maintains
realistic distributions and seasonal patterns.

# Installation

## Using uv (recommended)

WeatherCop uses [uv](https://docs.astral.sh/uv/) for dependency
management. To install:

1.  Install uv if you haven't already:

    ``` bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  Clone the repository and install:

    ``` bash
    git clone <repository-url>
    cd weathercop
    uv sync
    ```

3.  Build Cython extensions:

    ``` bash
    python setup.py build_ext --inplace
    ```

## Development installation

For development with additional tools:

``` bash
uv sync --group dev
python setup.py build_ext --inplace
```

**Note**: The first import may take 5-10 minutes as Cython extensions
compile. Pre-build them to avoid this delay.

**Important**: WeatherCop depends on the VarWG library, currently
installed from GitHub. For general availability, VarWG will be published
to PyPI first.

# Quick Start

Generate a weather ensemble in 10 lines of code:

``` python
import xarray as xr
from weathercop.multisite import Multisite, set_conf
from weathercop.configs import get_dwd_vg_config

# Load bundled DWD configuration
vg_conf = get_dwd_vg_config()
set_conf(vg_conf)
xds = xr.open_dataset("~/data/opendata_dwd/multisite_testdata.nc")
multisite = Multisite(xds, verbose=True)
ensemble = multisite.simulate_ensemble(n_realizations=5, name="example")
figs = multisite.plot_ensemble_meteogram_daily()
multisite.close()
```

**See [examples/quickstart.py](examples/quickstart.py) for a complete working example.**

For detailed tutorial with customization examples, see [examples/notebook_tutorial.ipynb](examples/notebook_tutorial.ipynb). To download real DWD weather data in the format WeatherCop expects, use the [dwd_opendata](https://github.com/iskur/dwd_opendata) package.

# Troubleshooting

## First Import Takes 5-10 Minutes

The first time you import weathercop, Cython extensions compile
automatically. This is normal. To avoid the wait:

``` bash
python setup.py build_ext --inplace
```

## Environment Variables

WeatherCop respects these environment variables:

- `WEATHERCOP_DIR`: Base directory for cache and data (default:
  `~/.weathercop`)
- `WEATHERCOP_ENSEMBLE_ROOT`: Directory for ensemble outputs (default:
  `$WEATHERCOP_DIR/ensembles`)

Example:

``` bash
export WEATHERCOP_DIR=/path/to/your/weathercop/data
python your_script.py
```

# Running Tests

To run the test suite:

``` bash
uv run pytest
```

Or install test dependencies and run:

``` bash
uv sync --group test
uv run pytest
```

## Memory-Efficient Testing

For running the full test suite on systems with limited memory, automated memory optimizations are enabled during pytest:

- Intermediate results are skipped by default during ensemble tests
- Fixtures prevent duplicate dataset loading
- Explicit garbage collection runs between tests

See [`docs/TESTING_MEMORY.md`](docs/TESTING_MEMORY.md) for detailed information and manual memory profiling.

# Key Features

- **Vine Copula Models**: Canonical (C-vine) and Regular (R-vine)
  implementations
- **Seasonal Variations**: Time-varying copula parameters with Fourier
  series smoothing
- **Multisite Generation**: Simultaneous weather generation across
  multiple stations
- **Comprehensive Copula Library**: Clayton, Gumbel, Joe, Plackett, and
  many more families
- **High Performance**: Cython-optimized computations with automatic
  SymPy code generation
- **Parallel Processing**: Built-in multiprocessing support for large
  ensembles

# Release Notes

## 0.1.0

- Initial release with vine copula implementations (CVine, RVine)
- Seasonal copula wrapper for time-varying parameters
- Integration with VarWG library for temporal structure preservation
- Automatic Cython code generation for copula functions
- Multisite weather generation workflows
- Migration to modern build system with pyproject.toml
- Dependency management with uv

**Requirements**: Python ≥ 3.13

# Web Sites

Code is hosted at: \<repository-url\>

Related project: [VarWG Weather
Generator](https://github.com/iskur/varwg)

# License Information

MIT License

Copyright (c) iskur \<32363199+iskur@users.noreply.github.com\>

See the file "LICENSE" for information on the history of this software,
terms & conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.
