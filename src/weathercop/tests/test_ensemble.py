"""Tests for ensemble generation and memory efficiency."""

import pytest

# Mark entire module as memory-intensive
pytestmark = pytest.mark.memory_intensive
import tempfile
from pathlib import Path
import xarray as xr

from weathercop.tests.assertion_utils import assert_valid_ensemble_structure


@pytest.fixture
def ensemble_with_disk_output(multisite_instance):
    """Simulate ensemble once with disk output for multiple test assertions.

    This fixture runs the expensive clear_cache=True ensemble simulation
    once and makes the result available to multiple focused tests.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        multisite_instance.simulate_ensemble(
            n_realizations=1,
            name="test_ensemble",
            clear_cache=True,
            write_to_disk=True,
            ensemble_root=Path(tmpdir),
        )
        yield multisite_instance.ensemble, Path(tmpdir)


def test_ensemble_generation_creates_dataset(ensemble_with_disk_output):
    """Test that ensemble simulation creates a non-None dataset."""
    ensemble, _ = ensemble_with_disk_output
    assert ensemble is not None


def test_ensemble_is_valid_xarray(ensemble_with_disk_output):
    """Test that ensemble is a valid xarray Dataset with required dimensions."""
    ensemble, _ = ensemble_with_disk_output
    assert_valid_ensemble_structure(ensemble)


@pytest.mark.parametrize("dimension", ["station", "variable", "time"])
def test_ensemble_has_required_dimension(ensemble_with_disk_output, dimension):
    """Verify ensemble dataset has all required dimensions."""
    ensemble, _ = ensemble_with_disk_output
    assert dimension in ensemble.dims, \
        f"Missing dimension '{dimension}'. Available: {list(ensemble.dims)}"


def test_ensemble_writes_netcdf_files(ensemble_with_disk_output):
    """Test that ensemble simulation writes NetCDF files to disk."""
    _, tmpdir = ensemble_with_disk_output
    ensemble_dir = tmpdir / "test_ensemble"
    nc_files = list(ensemble_dir.glob("*.nc"))
    assert len(nc_files) > 0, "Ensemble files not written to disk"


def test_memory_optimization_flag_during_testing(configure_for_testing):
    """Verify memory optimization is active during test runs."""
    from weathercop import cop_conf

    assert cop_conf.SKIP_INTERMEDIATE_RESULTS_TESTING is True
