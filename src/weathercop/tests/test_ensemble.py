"""Tests for ensemble generation and memory efficiency."""
import pytest
import tempfile
from pathlib import Path
import xarray as xr
import gc


@pytest.mark.slow
def test_small_ensemble_generation(multisite_instance):
    """Test that small ensemble can be generated without excessive memory use."""
    with tempfile.TemporaryDirectory() as tmpdir:
        n_reals = 2
        multisite_instance.simulate_ensemble(
            n_realizations=n_reals,
            name="test_ensemble",
            clear_cache=True,
            write_to_disk=True,
            ensemble_root=Path(tmpdir),
        )

        # Verify ensemble was created
        assert multisite_instance.ensemble is not None

        # Check that files were written
        ensemble_dir = Path(tmpdir) / "test_ensemble"
        nc_files = list(ensemble_dir.glob("*.nc"))
        assert len(nc_files) > 0, "Ensemble files not written"


def test_ensemble_returns_valid_data(multisite_instance):
    """Test that ensemble returns valid xarray data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        multisite_instance.simulate_ensemble(
            n_realizations=1,
            name="validity_test",
            clear_cache=True,
            write_to_disk=False,
            ensemble_root=Path(tmpdir),
        )

        assert multisite_instance.ensemble is not None
        assert isinstance(multisite_instance.ensemble, (xr.Dataset, xr.DataArray))

        # Check required dimensions
        assert "station" in multisite_instance.ensemble.dims
        assert "variable" in multisite_instance.ensemble.dims
        assert "time" in multisite_instance.ensemble.dims


def test_memory_optimization_flag_during_testing(configure_for_testing):
    """Verify memory optimization is active during test runs."""
    from weathercop import cop_conf
    assert cop_conf.SKIP_INTERMEDIATE_RESULTS_TESTING is True
