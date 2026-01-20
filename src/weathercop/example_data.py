"""Public module exposing bundled example data and configuration for Quick Start examples.

This module provides convenient access to the multisite test dataset and DWD VARWG
configuration that are bundled with WeatherCop. These are useful for users following
the Quick Start guide or testing their installation.

Example:
    Load the bundled example dataset and DWD configuration::

        import xarray as xr
        from weathercop.example_data import get_example_dataset_path, get_dwd_config
        from weathercop.multisite import Multisite, set_conf

        # Get paths and configuration
        dataset_path = get_example_dataset_path()
        dwd_config = get_dwd_config()

        # Configure VARWG with DWD settings
        set_conf(dwd_config)

        # Load dataset
        xds = xr.open_dataset(dataset_path)

        # Initialize Multisite weather generator
        wc = Multisite(xds, verbose=True)
"""

from pathlib import Path
from typing import Union


def get_example_dataset_path() -> Path:
    """Get the path to the bundled multisite example dataset.

    Returns the path to `multisite_testdata.nc`, a NetCDF file containing
    multisite synthetic weather data for three stations. This file is bundled
    with WeatherCop and suitable for Quick Start examples and testing.

    Returns
    -------
    Path
        Absolute path to the example dataset (multisite_testdata.nc).

    Raises
    ------
    FileNotFoundError
        If the example dataset is not found in the expected location
        (indicates a broken installation).

    Examples
    --------
    >>> from weathercop.example_data import get_example_dataset_path
    >>> import xarray as xr
    >>> path = get_example_dataset_path()
    >>> xds = xr.open_dataset(path)
    >>> print(xds)
    """
    # Locate the fixtures directory relative to this module
    fixtures_dir = Path(__file__).parent / "tests" / "fixtures"
    dataset_file = fixtures_dir / "multisite_testdata.nc"

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Example dataset not found at {dataset_file}. "
            "This may indicate a broken installation. "
            "Verify that src/weathercop/tests/fixtures/multisite_testdata.nc exists."
        )

    return dataset_file


def get_dwd_config():
    """Get the DWD VARWG configuration for WeatherCop.

    Returns the pre-configured VARWG configuration object required for
    processing meteorological data from the German Weather Service (DWD).
    This configuration is essential when working with DWD data.

    Returns
    -------
    VarwgConfig
        VARWG configuration object with DWD-specific settings,
        including proper handling of relative humidity distributions.

    Examples
    --------
    >>> from weathercop.example_data import get_dwd_config
    >>> from weathercop.multisite import set_conf
    >>> config = get_dwd_config()
    >>> set_conf(config)

    Notes
    -----
    This is a re-export of :func:`weathercop.configs.get_dwd_vg_config`
    provided for convenience and discoverability in the public API.
    """
    from weathercop.configs import get_dwd_vg_config

    return get_dwd_vg_config()
