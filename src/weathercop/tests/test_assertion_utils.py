"""Tests for assertion utility functions."""

import numpy as np
import pytest
import xarray as xr
from weathercop.tests.assertion_utils import (
    assert_valid_ensemble_structure,
    assert_mean_preservation,
    assert_correlation_preservation,
)


class TestAssertValidEnsembleStructure:
    """Test assert_valid_ensemble_structure function."""

    def test_valid_ensemble_passes(self):
        """Valid ensemble dataset should not raise."""
        ds = xr.Dataset(
            {
                "temp": (["station", "variable", "time"], np.random.randn(2, 3, 10)),
            }
        )
        # Should not raise
        assert_valid_ensemble_structure(ds)

    def test_non_xarray_raises(self):
        """Non-xarray object should raise AssertionError."""
        with pytest.raises(AssertionError, match="Expected xr.Dataset"):
            assert_valid_ensemble_structure({})

    def test_missing_dimension_raises(self):
        """Dataset missing required dimension should raise."""
        ds = xr.Dataset(
            {"temp": (["station", "time"], np.random.randn(2, 10))}
        )
        with pytest.raises(AssertionError, match="Missing dimensions"):
            assert_valid_ensemble_structure(ds)


class TestAssertMeanPreservation:
    """Test assert_mean_preservation function."""

    def test_identical_means_passes(self):
        """Identical means should not raise."""
        data = np.array([1.0, 2.0, 3.0])
        sim = xr.DataArray(data)
        obs = xr.DataArray(data)
        # Should not raise
        assert_mean_preservation(sim, obs, "test_var")

    def test_within_tolerance_passes(self):
        """Means within tolerance should not raise."""
        sim = xr.DataArray([1.0, 2.0, 3.0])  # mean = 2.0
        obs = xr.DataArray([1.0, 2.0, 2.9])  # mean = 1.967, ~1.6% diff
        # Should not raise with default 15% tolerance
        assert_mean_preservation(sim, obs, "test_var")

    def test_exceeds_tolerance_raises(self):
        """Means differing more than tolerance should raise."""
        sim = xr.DataArray([10.0, 20.0, 30.0])  # mean = 20.0
        obs = xr.DataArray([1.0, 2.0, 3.0])  # mean = 2.0, 900% diff
        with pytest.raises(AssertionError, match="differs from"):
            assert_mean_preservation(sim, obs, "test_var", tolerance=0.15)


class TestAssertCorrelationPreservation:
    """Test assert_correlation_preservation function."""

    def test_identical_correlations_passes(self):
        """Identical correlations should not raise."""
        # Create correlated data
        np.random.seed(42)
        x = np.random.randn(100)
        y = 2 * x + np.random.randn(100) * 0.1  # Highly correlated

        sim = xr.Dataset({
            "var1": (["time"], x),
            "var2": (["time"], y),
        })
        obs = xr.Dataset({
            "var1": (["time"], x),
            "var2": (["time"], y),
        })
        # Should not raise
        assert_correlation_preservation(sim, obs, ["var1", "var2"])

    def test_missing_variables_skipped(self):
        """Missing variables should be skipped without error."""
        sim = xr.Dataset({"var1": (["time"], [1, 2, 3])})
        obs = xr.Dataset({"var2": (["time"], [1, 2, 3])})
        # Should not raise
        assert_correlation_preservation(sim, obs, ["var1", "var2"])
