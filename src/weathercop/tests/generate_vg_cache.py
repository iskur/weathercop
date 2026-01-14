"""Generate and serialize VARWG instances for faster testing.

This script fits VARWG instances once and serializes them using Multisite's
cache mechanism. Tests can then load these pre-fitted instances by using
reinitialize_vgs=False, dramatically reducing test runtime.

Usage:
    python generate_vg_cache.py          # Generate cache if not exists
    python generate_vg_cache.py --regenerate  # Force regenerate cache

The script will generate cache files in Multisite's vgs_cache_dir using the
standard structure:
    vgs_cache_dir / station_name / {station_name}_{identifier}.pkl
"""

import os
import sys
from pathlib import Path
import xarray as xr

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from weathercop.multisite import Multisite, set_conf
from weathercop.configs import get_dwd_vg_config


def generate_vg_cache(regenerate=False):
    """Generate and cache VARWG instances using Multisite's cache mechanism.

    Parameters
    ----------
    regenerate : bool
        If True, regenerate cache even if it exists.
    """
    # Setup paths
    tests_dir = Path(__file__).parent
    fixtures_dir = tests_dir / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)
    data_root = Path.home() / "data/opendata_dwd"

    # Check if cache marker exists
    cache_marker = fixtures_dir / "vg_cache_ready"
    if cache_marker.exists() and not regenerate:
        print("VG cache already generated. Use --regenerate to rebuild.")
        return True

    # Verify test data exists
    test_data_path = data_root / "multisite_testdata.nc"
    if not test_data_path.exists():
        print(f"Error: Test data not found at {test_data_path}")
        print(f"Expected location: {test_data_path}")
        return False

    print("Generating VARWG cache for testing...")
    print(f"Loading test data from {test_data_path}")

    # Load test data
    test_dataset = xr.open_dataset(test_data_path)

    # Setup DWD VARWG config
    vg_conf = get_dwd_vg_config()
    set_conf(vg_conf)

    # Create Multisite instance with fitting enabled
    # Cache will be stored in fixtures_dir
    print("Fitting VARWG instances (this may take 10-30 minutes)...")
    wc = Multisite(
        test_dataset,
        verbose=False,
        refit=True,
        refit_vine=False,
        reinitialize_vgs=True,
        infilling="vg",
        fit_kwds=dict(seasonal=True),
        vgs_cache_dir=fixtures_dir,  # Use fixtures dir as cache location
    )

    # Multisite automatically saves to cache during __init__
    # But we need to verify the cache files were created
    print("\nVerifying VARWG cache files...")
    cache_count = 0
    for station_name in wc.keys():
        cache_file = wc.cache_file(station_name)
        if cache_file.exists():
            cache_count += 1
            print(f"  ✓ {station_name}: {cache_file}")
        else:
            print(f"  ✗ {station_name}: Cache file not found at {cache_file}")

    # Cleanup
    test_dataset.close()
    wc.close()
    del wc

    if cache_count == 0:
        print("\nError: No cache files were created!")
        return False

    # Mark as ready
    cache_marker.touch()
    print(f"\nVG cache generation complete!")
    print(f"  Location: {fixtures_dir}")
    print(f"  Cache files: {cache_count}")
    return True


if __name__ == "__main__":
    regenerate = "--regenerate" in sys.argv
    success = generate_vg_cache(regenerate=regenerate)
    sys.exit(0 if success else 1)
