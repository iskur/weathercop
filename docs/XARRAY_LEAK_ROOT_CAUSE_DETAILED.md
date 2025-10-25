# XArray Leak Root Cause - Detailed Technical Analysis

## The Smoking Gun

### Code Location 1: Multisite.simulate_ensemble() - lines 1664-1692

```python
# Line 1664-1669: Create self.ensemble
self.ensemble = (
    xr.open_mfdataset(filepaths[1:], **mf_kwds)
    .assign_coords(realization=range(n_realizations))
    .to_array("dummy")
    .squeeze("dummy", drop=drop_dummy)
)

# Line 1684-1692: Create self.ensemble_daily
if disaggregate:
    self.ensemble_daily = (
        xr.open_mfdataset(filepaths_daily[1:], **mf_kwds)
        .assign_coords(realization=range(n_realizations))
        .to_array("dummy")
        .squeeze("dummy", drop=drop_dummy)
    )
else:
    self.ensemble_daily = self.ensemble  # <-- SAME OBJECT!

# Line 1694-1701: Create self.ensemble_trans
self.ensemble_trans = (
    xr.open_mfdataset(filepaths_trans[1:], **mf_kwds)
    .assign_coords(realization=range(n_realizations))
    .to_array("dummy")
    .squeeze("dummy", drop=drop_dummy)
)
```

### Code Location 2: conftest.py - lines 91-103 (The Cleanup Attempt)

```python
try:
    # Close xarray datasets if they exist
    if hasattr(wc, 'ensemble') and wc.ensemble is not None:
        if hasattr(wc.ensemble, 'close'):
            wc.ensemble.close()                    # LINE 95
    if hasattr(wc, 'ensemble_daily') and wc.ensemble_daily is not None:
        if hasattr(wc.ensemble_daily, 'close'):
            wc.ensemble_daily.close()              # LINE 98 - FAILS HERE!
    if hasattr(wc, 'ensemble_trans') and wc.ensemble_trans is not None:
        if hasattr(wc.ensemble_trans, 'close'):
            wc.ensemble_trans.close()              # LINE 101 - NEVER EXECUTED
except Exception:
    pass  # Silently ignore cleanup errors       # LINE 103 - HIDES THE ERROR!
```

## The Exact Problem

### Scenario: Test runs with `disaggregate=False` (the default)

1. **Line 1692 in multisite.py**: `self.ensemble_daily = self.ensemble`
   - Now both attributes point to the **SAME Python object**
   - This object holds file handles to multiple netcdf files

2. **Line 95 in conftest.py**: `wc.ensemble.close()`
   - Closes the netcdf file handles
   - The underlying resource is now closed

3. **Line 98 in conftest.py**: `wc.ensemble_daily.close()`
   - **Tries to close the same object again!**
   - This raises an exception (trying to close already-closed resource)
   - The exception is caught at line 103

4. **Line 101 in conftest.py**: `wc.ensemble_trans.close()`
   - **NEVER EXECUTES** because exception was raised at line 98
   - The file handles in `ensemble_trans` remain OPEN
   - These stay in memory for the lifetime of the fixture

### Result: Memory Leak

**The ensemble_trans file handles are never closed**, accumulating **~10+ GB per test**.

The xarray_tracking module detected this because the Dataset was created (tracked) but never properly closed.

## Why It Wasn't Caught Before

1. **Silent exception handler** (`except Exception: pass`) hides the actual error
2. **No error logging** - we never knew cleanup was failing
3. **Tests appeared to pass** - even with unclosed file handles
4. **Memory just accumulated** - visible only when running multiple tests in sequence
5. **The diagnostics infrastructure we built detected it!** ✓

## The Files That Stay Open

From multisite.py lines 1506-1509:
```python
filepaths_trans = [
    filepath.parent / (filepath.stem + "_trans.nc")
    for filepath in filepaths
]
```

These netcdf files are opened with `xr.open_mfdataset()` at line 1696:
```python
self.ensemble_trans = (
    xr.open_mfdataset(filepaths_trans[1:], **mf_kwds)  # Opens multiple files!
    .assign_coords(realization=range(n_realizations))
    .to_array("dummy")
    .squeeze("dummy", drop=drop_dummy)
)
```

Each file handle takes memory, and with multiple realizations and tests, this adds up fast.

## Why xarray_tracking Detected It

The tracker monitors:
1. Calls to `xr.open_dataset()` and `xr.open_mfdataset()`
2. Adds newly opened Datasets to a WeakSet
3. At test end, checks which Datasets are still in the WeakSet
4. If a Dataset is still tracked, it means `.close()` was never called (or failed)

Result: **5 tests with 1 unclosed dataset each = 10.6 GB leak detected**

## The Fix (Multiple Options)

### Option A: Only Close Unique Objects
```python
try:
    closed_objs = set()

    if hasattr(wc, 'ensemble') and wc.ensemble is not None:
        if hasattr(wc.ensemble, 'close') and id(wc.ensemble) not in closed_objs:
            wc.ensemble.close()
            closed_objs.add(id(wc.ensemble))

    if hasattr(wc, 'ensemble_daily') and wc.ensemble_daily is not None:
        if hasattr(wc.ensemble_daily, 'close') and id(wc.ensemble_daily) not in closed_objs:
            wc.ensemble_daily.close()
            closed_objs.add(id(wc.ensemble_daily))

    if hasattr(wc, 'ensemble_trans') and wc.ensemble_trans is not None:
        if hasattr(wc.ensemble_trans, 'close') and id(wc.ensemble_trans) not in closed_objs:
            wc.ensemble_trans.close()
            closed_objs.add(id(wc.ensemble_trans))
except Exception as e:
    print(f"ERROR closing ensembles: {e}")
    import traceback
    traceback.print_exc()
```

### Option B: Wrap Each Close in Its Own Try-Except
```python
for attr_name in ['ensemble', 'ensemble_daily', 'ensemble_trans']:
    try:
        if hasattr(wc, attr_name):
            attr = getattr(wc, attr_name)
            if attr is not None and hasattr(attr, 'close'):
                attr.close()
    except Exception as e:
        print(f"ERROR closing {attr_name}: {e}")
```

### Option C: Reverse Cleanup Order
Close in reverse order of creation (safest approach):
```python
try:
    # Close in reverse order: trans, daily, ensemble
    if hasattr(wc, 'ensemble_trans') and wc.ensemble_trans is not None:
        if hasattr(wc.ensemble_trans, 'close'):
            wc.ensemble_trans.close()

    if hasattr(wc, 'ensemble_daily') and wc.ensemble_daily is not None:
        # Only close if it's not the same as ensemble
        if hasattr(wc.ensemble_daily, 'close') and wc.ensemble_daily is not wc.ensemble:
            wc.ensemble_daily.close()

    if hasattr(wc, 'ensemble') and wc.ensemble is not None:
        if hasattr(wc.ensemble, 'close'):
            wc.ensemble.close()
except Exception as e:
    print(f"ERROR closing ensembles: {e}")
    import traceback
    traceback.print_exc()
```

## Recommended Fix

**Option C (Reverse Order + Identity Check)** is the safest because:

1. **Reverse order**: Close things in opposite of creation order
2. **Identity check**: Skip closing if `ensemble_daily is ensemble` (same object)
3. **Error visibility**: Print actual errors instead of silently swallowing
4. **Explicit logging**: Makes debugging easier if problems recur

This ensures:
- Each unique Dataset is closed exactly once
- File handles are properly released
- Errors are visible for debugging
- Memory is properly reclaimed

## Verification After Fix

To verify the fix works:

```bash
# Clear old logs
rm ~/.weathercop/test_memory.log

# Run failing tests
pytest src/weathercop/tests/test_ensemble.py::test_small_ensemble_generation -v

# Check diagnostics
python scripts/analyze_memory_log.py ~/.weathercop/test_memory.log

# Should show:
# - test_small_ensemble_generation: 0 open datasets (instead of 1)
# - Peak memory drops dramatically (from 10GB to <1GB)
```

## Impact

**Before Fix:**
- test_small_ensemble_generation: 10,638 MB peak, 1 unclosed dataset
- test_sim: 9,607 MB peak, 1 unclosed dataset
- test_ensemble_returns_valid_data: 9,509 MB peak, 1 unclosed dataset

**After Fix (Expected):**
- All three tests: <100 MB peak, 0 unclosed datasets
- Full suite completes without OOM
- Python correctly releases memory between tests
