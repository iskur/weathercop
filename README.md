# What is WeatherCop?

WeatherCop is a multisite weather generator based on vine copulas. It
was developed to generate synthetic weather scenarios that preserve both
spatial dependencies across weather stations and temporal structure
within each station. The package combines statistical copula theory with
time series analysis from the VG library to create realistic weather
ensembles for hydrodynamic and ecological modeling.

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

**Important**: WeatherCop depends on the VG library, currently installed
from GitHub. For general availability, VG will be published to PyPI
first.

# Quick Start

After installation, you can use WeatherCop to generate multisite
synthetic weather data:

``` python
import xarray as xr
from weathercop.multisite import Multisite, set_conf

# Configure VG (e.g., with your config module)
import your_vg_conf
set_conf(your_vg_conf)

# Load multisite weather data as xarray Dataset
# Expected dimensions: (time, station, variable)
xds = xr.open_dataset("path/to/multisite_data.nc")

# Initialize the multisite weather generator
wc = Multisite(
    xds,
    verbose=True,
)

# Generate a single realization
sim_result = wc.simulate()

# Generate an ensemble of 20 realizations
wc.simulate_ensemble(20)

# Visualize results
wc.plot_ensemble_stats()
wc.plot_ensemble_meteogram_daily()
wc.plot_ensemble_qq()
```

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
- Integration with VG library for temporal structure preservation
- Automatic Cython code generation for copula functions
- Multisite weather generation workflows
- Migration to modern build system with pyproject.toml
- Dependency management with uv

**Requirements**: Python ≥ 3.13

# Web Sites

Code is hosted at: \<repository-url\>

Related project: [VG Weather Generator](https://github.com/iskur/vg)

# License Information

MIT License

Copyright (c) iskur \<32363199+iskur@users.noreply.github.com\>

See the file "LICENSE" for information on the history of this software,
terms & conditions for usage, and a DISCLAIMER OF ALL WARRANTIES.
