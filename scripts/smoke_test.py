#!/usr/bin/env python3
"""Comprehensive smoke tests for WeatherCop installation validation.

This script provides progressive validation of a WeatherCop installation across
4 levels, designed to complete in <15 minutes per platform for CI/CD workflows.

Test Levels:
    1. Import Validation (~2 min): Verify all core modules import successfully
    2. Data File Validation (~1 min): Verify example dataset is bundled
    3. Core Functionality (~10 min): Test copulas, MultiSite, and simulation
    4. Quick Start Example (~5 min): Execute the bundled quick_start workflow

Usage:
    python smoke_test.py              # Run all 4 levels
    python smoke_test.py --level 2    # Run levels 1-2 only
    python smoke_test.py --level 3    # Run levels 1-3 only

Exit codes:
    0: All tests passed
    1: One or more tests failed
"""

import sys
import time
import argparse
from pathlib import Path


def print_header(text):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def print_success(text):
    """Print a success message."""
    print(f"✓ {text}")


def print_error(text):
    """Print an error message."""
    print(f"✗ {text}", file=sys.stderr)


def test_level_1_imports():
    """Level 1: Import Validation (~2 min).

    Validates that all core modules can be imported successfully.
    This also triggers Cython extension loading and ufuncify compilation.
    """
    print_header("Level 1: Import Validation")
    start_time = time.time()

    modules = [
        "weathercop.copulae",
        "weathercop.vine",
        "weathercop.multisite",
        "weathercop.seasonal_cop",
        "weathercop.example_data",
        "weathercop.configs",
        "weathercop.cop_conf",
    ]

    try:
        for module_name in modules:
            print(f"Importing {module_name}...", end=" ", flush=True)
            __import__(module_name)
            print("OK")

        elapsed = time.time() - start_time
        print_success(f"Level 1 passed in {elapsed:.1f}s")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print_error(f"Level 1 failed after {elapsed:.1f}s: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_level_2_data_file():
    """Level 2: Data File Validation (~1 min).

    Validates that the example dataset is bundled and accessible.
    Opens with xarray to verify file structure.
    """
    print_header("Level 2: Data File Validation")
    start_time = time.time()

    try:
        import xarray as xr
        from weathercop.example_data import get_example_dataset_path

        # Get path to bundled dataset
        print("Getting example dataset path...", end=" ", flush=True)
        dataset_path = get_example_dataset_path()
        print(f"OK ({dataset_path})")

        # Verify file exists
        print(f"Verifying file exists...", end=" ", flush=True)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        print(f"OK ({dataset_path.stat().st_size / (1024*1024):.1f} MB)")

        # Open with xarray to validate structure
        print("Opening dataset with xarray...", end=" ", flush=True)
        xds = xr.open_dataset(dataset_path)
        print(f"OK ({len(xds.data_vars)} variables, {len(xds.coords)} coords)")

        # Verify expected structure
        print("Verifying dataset structure...", end=" ", flush=True)
        required_dims = {"time", "station"}
        if not required_dims.issubset(set(xds.dims)):
            raise ValueError(
                f"Missing required dimensions. Found: {set(xds.dims)}"
            )
        print(f"OK (dims: {list(xds.dims.keys())})")

        xds.close()

        elapsed = time.time() - start_time
        print_success(f"Level 2 passed in {elapsed:.1f}s")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print_error(f"Level 2 failed after {elapsed:.1f}s: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_level_3_core_functionality():
    """Level 3: Core Functionality (~10 min).

    Tests copula objects, MultiSite initialization, and simulation.
    Uses sliced data (2 stations, 2 years) to reduce runtime.
    """
    print_header("Level 3: Core Functionality")
    start_time = time.time()

    try:
        import xarray as xr
        import numpy as np
        from weathercop.example_data import (
            get_example_dataset_path,
            get_dwd_config,
        )
        from weathercop.multisite import Multisite, set_conf
        from weathercop import copulae

        # Configure VARWG with DWD settings
        print("Configuring VARWG...", end=" ", flush=True)
        set_conf(get_dwd_config())
        print("OK")

        # Test copula creation and sampling (validates ufuncs)
        print("Testing copula sampling...", end=" ", flush=True)
        clayton = copulae.clayton
        sample = np.array(clayton.sample(100, 2.0))
        if sample.shape != (2, 100):
            raise ValueError(f"Unexpected sample shape: {sample.shape}")
        print(f"OK (sampled {sample.shape[1]} pairs)")

        # Load and slice test data (2 stations, 2 years = 730 days)
        print("Loading example dataset...", end=" ", flush=True)
        xds = xr.open_dataset(get_example_dataset_path())
        n_stations = min(2, len(xds.station))
        n_days = min(730, len(xds.time))  # 2 years
        xds_small = xds.isel(
            station=slice(0, n_stations), time=slice(0, n_days)
        )
        print(f"OK (sliced to {n_stations} stations, {n_days} days)")

        # Initialize MultiSite (heavy operation - vine construction)
        print(
            "Initializing MultiSite (this may take several minutes)...",
            flush=True,
        )
        init_start = time.time()
        wc = Multisite(
            xds_small,
            verbose=True,
            refit=True,
            refit_vine=True,
            reinitialize_vgs=True,  # No cache available in clean install
            infilling="vg",
            fit_kwds=dict(seasonal=True),
        )
        init_elapsed = time.time() - init_start
        print_success(f"MultiSite initialized in {init_elapsed:.1f}s")

        # Run simulation
        print("Running simulation (1 year)...", end=" ", flush=True)
        sim_start = time.time()
        sim_result = wc.simulate(T=365)
        sim_elapsed = time.time() - sim_start
        print(f"OK ({sim_elapsed:.1f}s)")

        # Verify simulation output
        print("Verifying simulation output...", end=" ", flush=True)
        if sim_result is None:
            raise ValueError("Simulation returned None")
        if not hasattr(sim_result, "dims"):
            raise ValueError("Simulation result is not an xarray object")
        print(f"OK (dims: {list(sim_result.dims.keys())})")

        # Cleanup
        wc.close()
        xds.close()

        elapsed = time.time() - start_time
        print_success(f"Level 3 passed in {elapsed:.1f}s")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print_error(f"Level 3 failed after {elapsed:.1f}s: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_level_4_quick_start():
    """Level 4: Quick Start Example (~5 min).

    Executes a simplified version of the Quick Start workflow.
    This is the ultimate smoke test - validates what users actually do.
    """
    print_header("Level 4: Quick Start Example")
    start_time = time.time()

    try:
        import xarray as xr
        from weathercop.example_data import (
            get_example_dataset_path,
            get_dwd_config,
        )
        from weathercop.multisite import Multisite, set_conf

        # Configure VARWG
        print("Setting up VARWG configuration...", end=" ", flush=True)
        set_conf(get_dwd_config())
        print("OK")

        # Load dataset
        print("Loading example dataset...", end=" ", flush=True)
        dataset_path = get_example_dataset_path()
        xds = xr.open_dataset(dataset_path)
        # Use small slice for faster execution
        xds_small = xds.isel(station=slice(0, 2), time=slice(0, 730))
        print(
            f"OK ({len(xds_small.station)} stations, {len(xds_small.time)} days)"
        )

        # Initialize weather generator
        print("Initializing weather generator...", flush=True)
        init_start = time.time()
        wc = Multisite(
            xds_small,
            verbose=False,
            refit=True,
            refit_vine=True,
            reinitialize_vgs=True,
        )
        init_elapsed = time.time() - init_start
        print_success(f"Weather generator ready in {init_elapsed:.1f}s")

        # Generate synthetic weather
        print("Generating 1-year synthetic weather ensemble...", flush=True)
        sim_start = time.time()
        synthetic = wc.simulate(T=365)
        sim_elapsed = time.time() - sim_start
        print_success(f"Synthetic weather generated in {sim_elapsed:.1f}s")

        # Basic validation
        print("Validating output...", end=" ", flush=True)
        if synthetic is None:
            raise ValueError("Simulation returned None")
        if "time" not in synthetic.dims:
            raise ValueError("Missing 'time' dimension in output")
        print(f"OK ({len(synthetic.time)} time steps)")

        # Cleanup
        wc.close()
        xds.close()

        elapsed = time.time() - start_time
        print_success(f"Level 4 passed in {elapsed:.1f}s")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print_error(f"Level 4 failed after {elapsed:.1f}s: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run smoke tests based on command-line arguments."""
    parser = argparse.ArgumentParser(
        description="WeatherCop installation smoke tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2, 3, 4],
        default=4,
        help="Maximum test level to run (default: 4 - run all tests)",
    )

    args = parser.parse_args()

    print_header(f"WeatherCop Smoke Tests (Levels 1-{args.level})")
    overall_start = time.time()

    # Test levels to run
    test_functions = [
        test_level_1_imports,
        test_level_2_data_file,
        test_level_3_core_functionality,
        test_level_4_quick_start,
    ]

    results = []
    for i, test_func in enumerate(test_functions[: args.level], 1):
        success = test_func()
        results.append(success)
        if not success:
            print_error(f"Stopping at level {i} due to failure")
            break

    # Summary
    overall_elapsed = time.time() - overall_start
    print_header("Summary")

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total} levels")
    print(f"Total time: {overall_elapsed:.1f}s")

    if all(results):
        print_success("All smoke tests passed!")
        return 0
    else:
        print_error("Some smoke tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
