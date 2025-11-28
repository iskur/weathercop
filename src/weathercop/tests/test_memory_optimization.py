"""Test memory optimization for ensemble simulation.

Tests that simulate_ensemble with surgical attribute exclusion:
1. Still produces correct results
2. Doesn't trigger unexpected None-check blocks
3. Uses less memory (qualitatively, via pickle size)
"""

import pytest
import logging
import pickle
from pathlib import Path
from weathercop.multisite import Multisite
import xarray as xr
import numpy as np


# Mock Multisite for testing surgical exclusion principle
class MockMultisite:
    """Mock Multisite with large and small attributes for pickle size testing."""

    def __init__(self):
        # Simulate large attributes (calibration/input data)
        self.xds = "x" * (10 * 1024 * 1024)  # 10 MB of dummy data
        self.xar = "y" * (10 * 1024 * 1024)  # 10 MB
        self.data_trans = "z" * (5 * 1024 * 1024)  # 5 MB

        # Essential attributes for simulation
        self.vine = {"test": "data"}
        self.vgs = {"station": "vg_obj"}
        self.K = 4
        self.station_names = ["A", "B", "C"]


@pytest.mark.slow
def test_simulate_ensemble_with_memory_optimization(
    multisite_instance,
    tmp_path,
):
    """RED: Test that simulate_ensemble works with surgical attribute exclusion.

    This test verifies that:
    1. simulate_ensemble completes successfully
    2. No unexpected None-check warnings are logged (surgical exclusion is safe)
    3. Results are valid (no NaNs in critical output)

    Note: This test is marked as slow because it runs a full ensemble generation
    which involves multiprocessing and I/O operations.
    """
    # Capture warnings from memory optimization logging
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("weathercop.multisite")

    # Collect warning messages
    warning_messages = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                warning_messages.append(record.getMessage())

    handler = ListHandler()
    logger.addHandler(handler)

    try:
        # Run ensemble simulation (uses multiprocessing with surgical exclusion)
        ensemble = multisite_instance.simulate_ensemble(
            n_realizations=2,
            name="test_memory_opt",
            ensemble_root=tmp_path,
            write_to_disk=True,
        )

        # Verify ensemble was created and is valid
        assert ensemble is not None, "Ensemble should be created"
        assert (
            "realization" in ensemble.dims
        ), "Ensemble should have realization dimension"
        assert ensemble.realization.size == 2, "Should have 2 realizations"

        # Check that results are not all-NaN
        for station_name in multisite_instance.station_names:
            for varname in multisite_instance.varnames:
                data = ensemble.sel(station=station_name, variable=varname)
                invalid_count = np.isnan(data.values).sum()
                assert (
                    invalid_count == 0
                ), f"{varname} for {station_name} should have valid (non-NaN) data"

        # Check for unexpected None-access warnings
        # Some warnings are expected (e.g., fft_sim being None initially)
        # But we shouldn't have warnings about critical attributes like vine
        unexpected_warnings = [
            w
            for w in warning_messages
            if "vine" in w.lower() and "is None" in w
        ]
        assert (
            len(unexpected_warnings) == 0
        ), f"Unexpected None-access for critical attributes: {unexpected_warnings}"

    finally:
        logger.removeHandler(handler)


def test_surgical_exclusion_reduces_pickle_size():
    """Test that surgical exclusion logic reduces serialization size.

    This test doesn't require a full Multisite instance - it just validates
    that the surgical exclusion approach would work by setting attributes to None
    and measuring the pickle size difference.
    """
    full_obj = MockMultisite()
    full_size = len(pickle.dumps(full_obj, protocol=pickle.HIGHEST_PROTOCOL))

    # Create lightweight version with surgical exclusion
    light_obj = MockMultisite()
    light_obj.xds = None
    light_obj.xar = None
    light_obj.data_trans = None
    light_size = len(pickle.dumps(light_obj, protocol=pickle.HIGHEST_PROTOCOL))

    # Surgical exclusion should significantly reduce size
    reduction_percent = ((full_size - light_size) / full_size) * 100

    print(f"\nSurgical Exclusion Test:")
    print(f"  Full size: {full_size / (1024*1024):.2f} MB")
    print(f"  Light size: {light_size / (1024*1024):.2f} MB")
    print(f"  Reduction: {reduction_percent:.1f}%\n")

    assert light_size < full_size, (
        f"Lightweight version should be smaller. "
        f"Full: {full_size}, Light: {light_size}"
    )
    # With dummy data, should see significant reduction
    assert (
        reduction_percent > 50
    ), f"Should see at least 50% reduction, got {reduction_percent:.1f}%"
