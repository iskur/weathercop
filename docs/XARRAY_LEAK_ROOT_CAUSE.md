# XArray Leak Root Cause Analysis

## The Problem

The xarray tracker detected 1 unclosed dataset in each of these tests:
- test_small_ensemble_generation (10,638 MB)
- test_ensemble_returns_valid_data (9,509 MB)
- test_phase_randomization (9,222 MB)
- test_sim (9,607 MB)
- test_memory_optimization_flag_during_testing

## The Fixture Cleanup (conftest.py:90-106)

The `multisite_instance` fixture attempts to close datasets:

```python
try:
    # Close xarray datasets if they exist
    if hasattr(wc, 'ensemble') and wc.ensemble is not None:
        if hasattr(wc.ensemble, 'close'):
            wc.ensemble.close()
    if hasattr(wc, 'ensemble_daily') and wc.ensemble_daily is not None:
        if hasattr(wc.ensemble_daily, 'close'):
            wc.ensemble_daily.close()
    if hasattr(wc, 'ensemble_trans') and wc.ensemble_trans is not None:
        if hasattr(wc.ensemble_trans, 'close'):
            wc.ensemble_trans.close()
except Exception:
    pass  # Silently ignore cleanup errors
```

**PROBLEM: The `except Exception: pass` silently hides errors!**

## Why It's Failing

1. **Datasets created by simulate_ensemble()** in test code
2. **Fixture tries to close them** but something is wrong
3. **Exception occurs during close()** - could be:
   - Weakref still holding references (tracker's own references?)
   - Dataset already closed
   - Missing file handle
   - Broken file descriptor
4. **Exception is silently swallowed** - cleanup fails silently
5. **Dataset remains in memory** - takes 10+ GB per test

## Hypotheses

### Hypothesis 1: Weakref Interference
The xarray_tracking.py module uses WeakSet to track datasets. When a weakref is the only thing keeping a dataset alive, calling `.close()` might fail or behave unexpectedly.

### Hypothesis 2: Multiple Cleanup Attempts
The fixture tries to close `ensemble`, `ensemble_daily`, and `ensemble_trans`. If `.close()` is idempotent/buggy, it might fail on second attempt.

### Hypothesis 3: Simulator Stores References Elsewhere
The Multisite.simulate_ensemble() method might store references to datasets in other attributes not being closed by the fixture.

### Hypothesis 4: ChunkSize/Dask Issues
If datasets use dask for chunking, closing might fail or behave differently.

## How to Debug

### Step 1: Remove Silent Exception Handler
Change:
```python
except Exception:
    pass
```
To:
```python
except Exception as e:
    print(f"ERROR closing ensemble datasets: {e}")
    import traceback
    traceback.print_exc()
```

Run one failing test to see what error is occurring.

### Step 2: Check Weakref Interference
Temporarily disable xarray_tracking.py:
- Comment out `from weathercop.tests.xarray_tracking import get_xarray_tracker`
- Comment out tracker initialization in log_test_memory fixture
- Re-run test - does it still leak?

If leak goes away, the tracker is interfering (unlikely, but possible).

### Step 3: List All Ensemble Attributes
Add debugging to multisite_instance fixture:
```python
try:
    print(f"Closing wc, attributes starting with 'ensemble':")
    for attr in dir(wc):
        if 'ensemble' in attr.lower() and not attr.startswith('_'):
            val = getattr(wc, attr, None)
            if hasattr(val, 'close'):
                print(f"  - {attr}: {type(val)}")
```

This will reveal if there are other xarray objects not being closed.

### Step 4: Check Multisite.simulate_ensemble()
Look at implementation in `src/weathercop/multisite.py`:
- What datasets does it create?
- Where are they stored?
- Does it return references that are also stored elsewhere?
- Are they supposed to be closed by the method, the caller, or the fixture?

## Why xarray_tracker Detects This

The tracker (xarray_tracking.py) uses monkey-patching:

```python
def tracked_open(*args, **kwargs):
    ds = original_open(*args, **kwargs)
    tracker.datasets.add(ds)  # WeakSet - doesn't hold reference
    return ds
```

**Key insight**: The WeakSet tracks datasets WITHOUT holding strong references. If `.close()` fails and the dataset remains in memory, it will still appear in the WeakSet because:
1. Test code holds reference (test_small_ensemble_generation)
2. Multisite instance holds reference (multisite_instance.ensemble)
3. WeakSet can see it (doesn't prevent GC)

When fixture cleanup fails, the test code's reference is released but the Multisite instance's reference persists longer than expected.

## Solution Strategy

### Best Approach: Fix the Root Cause
1. Identify why `.close()` is failing (remove silent exception handler first)
2. Fix the underlying issue (wrong attribute, wrong type, etc.)
3. Verify cleanup works with explicit logging

### Workaround Approach: Force Delete References
If close() is fundamentally broken:
```python
# Delete all ensemble references
if hasattr(wc, 'ensemble'):
    del wc.ensemble
if hasattr(wc, 'ensemble_daily'):
    del wc.ensemble_daily
if hasattr(wc, 'ensemble_trans'):
    del wc.ensemble_trans
gc.collect()
```

This removes references so GC can reclaim memory (even if datasets aren't formally closed).

### Nuclear Option: Suppress XArray Tracking
If datasets are large and not closing properly, disable tracking for these tests:
```python
os.environ["WEATHERCOP_DISABLE_DIAGNOSTICS"] = "1"
```

This won't fix the leak but will stop showing it in logs.

## Expected Impact of Fix

- **Before**: 10,638 MB peak for test_small_ensemble_generation
- **After**: Should drop to <100 MB (like copulae tests)
- **OOM Prevention**: Full suite should complete without hitting 128GB limit

## Files to Check

1. `src/weathercop/tests/conftest.py` - multisite_instance fixture cleanup
2. `src/weathercop/multisite.py` - Multisite.simulate_ensemble() implementation
3. `src/weathercop/tests/test_ensemble.py` - test code that calls simulate_ensemble()
4. `src/weathercop/tests/xarray_tracking.py` - verify it's not interfering (unlikely)

## Next Steps for User

1. **Change the exception handler** in conftest.py to print the actual error
2. **Run one failing test** (e.g., `pytest src/weathercop/tests/test_ensemble.py::test_small_ensemble_generation -v`)
3. **Look for error message** - it will tell you what's failing
4. **Fix the root cause** based on the error
5. **Re-run full suite** to verify memory returns to normal

The diagnostics infrastructure did its job perfectly - it identified the exact problem that would have been invisible without xarray lifecycle tracking!
