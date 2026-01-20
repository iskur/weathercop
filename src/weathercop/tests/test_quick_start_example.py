"""Integration test verifying Quick Start example code works end-to-end."""

import pytest
import xarray as xr
from collections import namedtuple
from weathercop.example_data import get_example_dataset_path, get_dwd_config
from weathercop.multisite import Multisite, set_conf


def test_quick_start_example_runs_end_to_end():
    """Verify the Quick Start example code from README.org actually works.

    This test ensures that users following the Quick Start guide can run
    the example code without modification or additional configuration.
    """
    # Configure VarWG with DWD settings
    set_conf(get_dwd_config())

    # Load example multisite weather data as xarray Dataset
    xds = xr.open_dataset(get_example_dataset_path())

    # Verify dataset structure
    assert "time" in xds.dims
    assert "station" in xds.dims
    assert "variable" in xds.dims

    # Initialize the multisite weather generator
    wc = Multisite(
        xds,
        verbose=False,  # Suppress output in test
    )

    # Generate a single realization
    sim_result = wc.simulate()

    # Verify simulation produced output
    assert sim_result is not None
    # simulate() returns a SimResult namedtuple with sim_sea, sim_trans, rphases
    assert hasattr(sim_result, 'sim_sea')
    assert hasattr(sim_result, 'sim_trans')

    # Verify ensemble generation works
    # (Note: Full ensemble may be slow in tests, so use small size)
    ensemble_result = wc.simulate_ensemble(2)
    assert ensemble_result is not None

    wc.close()


def test_quick_start_imports_work():
    """Verify all imports from Quick Start example are available."""
    # These should not raise ImportError
    from weathercop.example_data import (
        get_example_dataset_path,
        get_dwd_config,
    )
    from weathercop.multisite import Multisite, set_conf

    # Verify functions are callable
    assert callable(get_example_dataset_path)
    assert callable(get_dwd_config)
    assert callable(set_conf)
