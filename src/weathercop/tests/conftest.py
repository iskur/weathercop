"""Pytest configuration and fixtures for weathercop tests."""

import os
import pytest
from pathlib import Path
import xarray as xr
import gc
from weathercop.multisite import Multisite, set_conf
from weathercop import cop_conf
from weathercop.tests.memory_diagnostics import get_memory_logger
from weathercop.tests.xarray_tracking import get_xarray_tracker
from weathercop.configs import get_dwd_vg_config

vg_conf = get_dwd_vg_config()


# Enable memory-efficient mode for testing
os.environ.setdefault("WEATHERCOP_SKIP_INTERMEDIATE_RESULTS", "1")

# Disable parallel xarray loading during testing to reduce memory fragmentation
os.environ.setdefault("WEATHERCOP_PARALLEL_LOADING", "0")

# Enable memory diagnostics logging (can be disabled with WEATHERCOP_DISABLE_DIAGNOSTICS=1)
os.environ.setdefault("WEATHERCOP_DISABLE_DIAGNOSTICS", "0")

# Custom log path for memory diagnostics (optional)
# os.environ.setdefault("WEATHERCOP_MEMORY_LOG", "/path/to/log")


def pytest_sessionstart(session):
    """Hook: Log session start and initialize diagnostics."""
    logger = get_memory_logger()
    logger.log_session_start()


def pytest_sessionfinish(session, exitstatus):
    """Hook: Log session end."""
    logger = get_memory_logger()
    logger.log_session_end()


@pytest.fixture(scope="session", autouse=True)
def setup_dwd_varwg_config():
    """Configure VarWG with DWD WeatherCop settings before any tests run.

    This must run at session scope and be autouse=True to ensure the config
    is set BEFORE VarWG modules are imported by test code.
    """
    from weathercop.configs import get_dwd_vg_config
    from weathercop.multisite import set_conf

    vg_conf = get_dwd_vg_config()
    set_conf(vg_conf)

    yield  # Tests run here
    # No cleanup needed - config persists for all tests in session


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
    # Cleanup after each test - use Multisite's close() method
    try:
        wc.close()
    except Exception as e:
        import warnings

        warnings.warn(f"Error closing Multisite instance: {e}")
    finally:
        del wc
        gc.collect()


@pytest.fixture(scope="function", autouse=True)
def log_test_memory(request):
    """Auto-log fixture for test memory tracking."""
    logger = get_memory_logger()
    logger.log_test_start(request.node.name)
    logger.start_peak_tracking()

    yield

    # Get peak memory and xarray count using tracker
    peak_mb = logger.get_peak_and_stop_tracking()
    tracker = get_xarray_tracker()
    xarray_count = tracker.count_active()

    logger.log_test_end(
        request.node.name, peak_memory_mb=peak_mb, xarray_count=xarray_count
    )


@pytest.fixture(scope="function", autouse=True)
def cleanup_after_test():
    """Auto-cleanup fixture to run after each test."""
    yield
    gc.collect()
