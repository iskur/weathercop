"""Debug script to identify NaN sources in rank calculation."""
import sys
import numpy as np
import xarray as xr
from weathercop.multisite import Multisite, set_conf
from weathercop.configs import get_dwd_vg_config
from pathlib import Path

def debug_ranks_initialization():
    """Instrument Multisite init to see where NaNs come from."""
    # Setup
    vg_conf = get_dwd_vg_config()
    set_conf(vg_conf)
    data_root = Path("src/weathercop/tests/data")
    xds = xr.open_dataset(data_root / "multisite_testdata.nc")

    # Create instance but catch before the assertion fails
    try:
        wc = Multisite(
            xds,
            verbose=False,
            refit=True,
            refit_vine=False,
            reinitialize_vgs=True,
            fit_kwds=dict(seasonal=True),
        )
    except AssertionError as e:
        print(f"AssertionError caught: {e}")
        # The error happens in _init_vgs at line 1136
        # We need to inspect the internal state before that point
        # This will require modifying multisite.py temporarily

    xds.close()

def test_empirical_cdf_with_nans():
    """Test how normal distribution CDFs handle NaN values.

    Note: WeatherCop uses normal distributions fitted to transformed data,
    not empirical distributions directly. The 'empirical' in config refers
    to VARWG's internal data transformation (e.g., using KDE).
    """
    import numpy as np
    from scipy import stats as dists

    print("\n=== Testing Normal CDF with NaN Values ===")

    # Create sample data and fit a normal distribution
    data_clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Clean data for fitting: {data_clean}")

    # Fit normal distribution (same as multisite.py line 1105-1107)
    dist = dists.norm(*dists.norm.fit(data_clean))
    print(f"Fitted distribution: N(μ={dist.mean():.2f}, σ={dist.std():.2f})")

    # Test CDF with NaN values
    test_values = np.array([1.5, np.nan, 3.5, np.nan])
    result = dist.cdf(test_values)

    print(f"\nInput to CDF: {test_values}")
    print(f"CDF result: {result}")
    print(f"Result contains NaN: {np.isnan(result).any()}")
    print(f"NaN positions in input:  {np.isnan(test_values)}")
    print(f"NaN positions in output: {np.isnan(result)}")
    print(f"NaN propagation correct: {np.array_equal(np.isnan(test_values), np.isnan(result))}")

    # Test edge cases
    print("\n--- Testing edge cases ---")
    edge_test = np.array([np.nan, -np.inf, np.inf, 0.0, np.nan])
    edge_result = dist.cdf(edge_test)
    print(f"Edge input:  {edge_test}")
    print(f"Edge result: {edge_result}")
    print(f"All NaN propagated: {np.isnan(edge_test).sum()} -> {np.isnan(edge_result).sum()}")

def check_data_trans_nans(data_trans, varnames):
    """Check for NaNs in transformed data per variable."""
    print("\n=== Checking data_trans for NaNs ===")
    for var in varnames:
        if var in data_trans.variable.values:
            var_data = data_trans.sel(variable=var).values
            nan_count = np.isnan(var_data).sum()
            if nan_count > 0:
                total = var_data.size
                pct = 100 * nan_count / total
                print(f"Variable '{var}': {nan_count} NaNs out of {total} values ({pct:.1f}%)")
            else:
                print(f"Variable '{var}': No NaNs")

if __name__ == "__main__":
    print("Starting debug script for NaN handling in ranks...")

    # Test 1: Understand empirical CDF behavior
    test_empirical_cdf_with_nans()

    # Test 2: Debug the actual initialization
    debug_ranks_initialization()
