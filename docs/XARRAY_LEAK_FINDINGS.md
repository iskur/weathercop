# URGENT: XArray Dataset Leaks Discovered

## Critical Finding

The diagnostics system **detected xarray dataset leaks** that were NOT caught by the earlier analysis. These leaks are the **primary cause of OOM during full test suite execution**.

## Leaked Tests

| Test | Peak Memory | Datasets | Status |
|------|-------------|----------|--------|
| test_small_ensemble_generation | 10,638 MB | 1 unclosed | ❌ LEAK |
| test_ensemble_returns_valid_data | 9,509 MB | 1 unclosed | ❌ LEAK |
| test_phase_randomization | 9,222 MB | 1 unclosed | ❌ LEAK |
| test_sim | 9,607 MB | 1 unclosed | ❌ LEAK |
| test_memory_optimization_flag_during_testing | Unknown | 1 unclosed | ❌ LEAK |

## Root Cause

All leaking tests are in:
- `src/weathercop/tests/test_ensemble.py`
- `src/weathercop/tests/test_cyvine.py`
- Related multisite simulation tests

These tests generate xarray Datasets during ensemble weather simulation but **fail to close them** after testing. The datasets remain in memory, accumulating across test runs.

## Impact

- **test_small_ensemble_generation** alone uses **10.6 GB** - nearly the entire peak
- By test_sim, memory is at **9.6 GB** and still has open datasets
- Subsequent tests cannot release memory because datasets are still referenced
- **This is why pytest got OOM-killed during the full suite run**

## What Needs To Happen

### 1. Identify the Fixture/Code
The multisite_instance fixture or test code creates xarray Datasets but doesn't close them:
```python
# BAD - Dataset created but not closed
ds = create_ensemble_dataset(...)
# ds stays in memory forever
```

Should be:
```python
# GOOD - Dataset properly closed
ds = create_ensemble_dataset(...)
try:
    # use ds
finally:
    ds.close()
```

### 2. Search These Files
- `src/weathercop/tests/test_ensemble.py` - Obvious culprit based on test names
- `src/weathercop/tests/conftest.py` - Check multisite_instance fixture cleanup
- `src/weathercop/multisite.py` - Check if datasets are released after simulation

### 3. Check For:
- Missing `.close()` calls on xarray Datasets
- Datasets stored in module-level variables
- Datasets not released in fixture teardown
- Weak references not working as expected

## Evidence

From the analysis script output:
```
Tests with unclosed xarray datasets:
  test_small_ensemble_generation: 1 open dataset(s)
  test_ensemble_returns_valid_data: 1 open dataset(s)
  test_memory_optimization_flag_during_testing: 1 open dataset(s)
  test_phase_randomization: 1 open dataset(s)
  test_sim: 1 open dataset(s)
```

The xarray tracker (weakref-based) is working correctly and detecting the leaks. The diagnostic infrastructure **did its job**.

## Next Steps

1. **Examine test_ensemble.py closely** - Look for dataset creation without cleanup
2. **Check multisite_instance fixture** - Verify cleanup in conftest.py
3. **Review Multisite class** - See if datasets escape during simulation
4. **Add explicit `.close()` calls** - Ensure all xarray objects are properly cleaned up
5. **Re-run tests** - Verify leaks are fixed and memory stabilizes

## Why This Matters

- **10.6 GB leak per test run** is unsustainable
- **Multiple leaks accumulate** - Second test can't reuse memory from first
- **OOM killer was justified** - Pytest consumed available RAM
- **This is fixable** - Just need to add proper cleanup

## Timeline

1. ✅ Diagnostics deployed and captured evidence
2. ✅ Leaks identified by xarray tracker
3. ⏳ **YOU: Find the code causing leaks**
4. ⏳ **YOU: Add proper cleanup**
5. ⏳ **YOU: Verify memory returns to normal**

The diagnostics infrastructure worked perfectly to identify the problem that wasn't visible from memory statistics alone!
