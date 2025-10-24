"""Pytest configuration and fixtures for weathercop tests."""
import os
import pytest
from pathlib import Path
import xarray as xr
import gc
from weathercop.multisite import Multisite, set_conf
from weathercop import cop_conf
import opendata_vg_conf as vg_conf


# Enable memory-efficient mode for testing
os.environ.setdefault("WEATHERCOP_SKIP_INTERMEDIATE_RESULTS", "1")


@pytest.fixture(scope="session")
def vg_config():
    """Initialize and return VG configuration (session-scoped)."""
    set_conf(vg_conf)
    return vg_conf


@pytest.fixture(scope="session", autouse=True)
def configure_for_testing(vg_config):
    """Auto-configure for memory-efficient testing mode."""
    # Enable memory optimizations during test runs
    cop_conf.SKIP_INTERMEDIATE_RESULTS_TESTING = True
    cop_conf.AGGRESSIVE_CLEANUP = True
    yield
    # Reset after tests
    cop_conf.SKIP_INTERMEDIATE_RESULTS_TESTING = False
    cop_conf.AGGRESSIVE_CLEANUP = False


@pytest.fixture(scope="session")
def data_root():
    """Return path to test data (session-scoped)."""
    return Path().home() / "data/opendata_dwd"


@pytest.fixture(scope="function")
def test_dataset(data_root):
    """Load test dataset fresh for each test (function-scoped)."""
    xds = xr.open_dataset(data_root / "multisite_testdata.nc")
    yield xds
    # Cleanup after each test
    xds.close()


@pytest.fixture(scope="function")
def multisite_instance(test_dataset, vg_config):
    """Create a fresh Multisite instance for each test (function-scoped).

    Note: test_dataset is now function-scoped, so each test gets a fresh
    dataset and Multisite instance with independent memory.
    """
    wc = Multisite(
        test_dataset,
        verbose=False,
        refit=True,
        refit_vine=False,
        reinitialize_vgs=True,
        fit_kwds=dict(seasonal=True),
    )
    yield wc
    # Explicit cleanup after each test
    del wc
    gc.collect()


@pytest.fixture(scope="function", autouse=True)
def cleanup_after_test():
    """Auto-cleanup fixture to run after each test."""
    yield
    gc.collect()
