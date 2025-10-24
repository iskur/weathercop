# Memory-Efficient Testing Guide

## Problem
Running the full test suite on the `test_multisite.py` tests can consume excessive memory due to:
1. Multiple copies of large datasets in memory during parallel ensemble generation
2. Storing intermediate transformed data for all realizations
3. Accumulating filepath lists and temporary xarray objects

## Solution
The project now includes several memory optimization mechanisms:

### 1. Configuration Flags
Two flags in `src/weathercop/cop_conf.py` control memory usage:

- `SKIP_INTERMEDIATE_RESULTS_TESTING`: When True (set automatically during pytest), skips storing transformed data
- `AGGRESSIVE_CLEANUP`: When True, enables explicit garbage collection between ensemble realizations

### 2. Pytest Fixtures
`src/weathercop/tests/conftest.py` provides:

- **Session-scoped fixtures** (`test_dataset`, `vg_config`): Shared across all tests to avoid duplicate loads
- **Function-scoped fixtures** (`multisite_instance`): Fresh instance per test with automatic cleanup
- **Auto-use fixtures** (`cleanup_after_test`): Explicit gc.collect() after each test

### 3. Memory Mode for Testing
When running pytest, memory optimization is **automatically enabled**:
```bash
pytest src/weathercop/tests/test_multisite.py
# Automatically sets SKIP_INTERMEDIATE_RESULTS_TESTING = True
```

### 4. Running Tests in Memory-Constrained Environments
To run all tests with maximum memory efficiency:

```bash
# Use a single process (no parallelization)
WEATHERCOP_SKIP_INTERMEDIATE_RESULTS=1 pytest src/weathercop/ -n0

# Or set the configuration directly in your test environment
```

### 5. Disabling Memory Optimizations for Debugging
If you need intermediate results for debugging:

```python
# In conftest.py or your test, set before running simulate_ensemble:
from weathercop import cop_conf
cop_conf.SKIP_INTERMEDIATE_RESULTS_TESTING = False
```

## Estimated Memory Savings
- **Per-realization saved**: ~500 MB - 2 GB (depending on dataset size)
- **Full ensemble**: 20+ realizations × 500 MB = 10+ GB saved
- **Peak memory**: Reduced from ~128+ GB to ~20-40 GB for typical runs

## Monitoring Memory Usage
To manually profile memory during tests:

```bash
pytest src/weathercop/tests/test_memory_profile.py -v -s -k memory_profile
```

This will print peak memory usage and help identify remaining issues.
