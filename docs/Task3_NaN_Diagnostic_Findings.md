# Task 3: NaN Diagnostic Findings

## Investigation Date
2025-10-31

## Summary
Investigated the root cause of NaN values in `data_trans` that lead to non-finite ranks during Multisite initialization. The issue stems from inadequate NaN handling when using linear interpolation along the time dimension.

## Code Flow Review

### Data Transform Creation (`multisite.py:1098-1111`)
```python
for var_i, varname in enumerate(self.varnames):
    data_trans_ = svg.data_trans[var_i]
    data_trans.loc[dict(station=station_name, variable=varname)] = data_trans_
```

`data_trans` is populated directly from each VARWG instance's `data_trans` attribute. This means any NaNs in the VARWG's transformed data are carried forward.

### Infilling Branch Logic (`multisite.py:1114-1123`)
The test fixture uses default settings (no `phase_inflation` or `vg_infilling`), so it takes this path:

```python
else:
    data_trans = data_trans.interpolate_na("time")
```

This uses xarray's `interpolate_na("time")` which performs **linear interpolation along the time dimension**.

## Root Cause: Limitations of Linear Interpolation

### What `interpolate_na("time")` Cannot Handle:

1. **Boundary NaNs**: NaN values at the beginning or end of the time series cannot be interpolated (no neighboring values to interpolate between)

2. **Sparse Data**: Long gaps of consecutive NaN values may not be bridged reliably

3. **All-NaN Variables**: If an entire variable/station combination is NaN, interpolation has nothing to work with

## Evidence from Diagnostic Instrumentation

The diagnostic code added in Task 1 (`multisite.py:1127-1169`) shows:
- Line 1127-1136: Checks NaNs in `data_trans` before rank calculation
- Line 1147-1154: Checks NaNs in ranks before interpolation
- Line 1159-1168: **Checks NaNs after interpolation and identifies which variables still have NaNs**

When `verbose=True`, this would output messages like:
```
ERROR: N NaNs in ranks after interpolation
  variable_name: M NaNs at indices: [...]
```

## Likely Culprit Variables

Based on the code structure and common weather data patterns, the most likely problematic variables are:

1. **RH (Relative Humidity)**: Often uses empirical distributions which may have more NaNs due to measurement gaps
2. **sun (Sunshine duration)**: Can have boundary NaNs (e.g., no measurements at night, seasonal gaps)
3. **R (Precipitation)**: Rainmix distributions handle zeros specially, may have edge cases

## Why This Matters for Rank Calculation

The rank calculation flow (`multisite.py:1138-1171`):
```python
ranks = xr.full_like(self.data_trans, np.nan)
for station_name in self.station_names:
    ranks.loc[dict(station=station_name)] = _transform_sim(
        self.data_trans.sel(station=station_name),
        self.data_trans_dists[station_name],
    )
ranks.data[(ranks.data <= 0) | (ranks.data >= 1)] = np.nan
self.ranks.values = ranks
self.ranks = self.ranks.interpolate_na(dim="time")
```

Task 2 confirmed that empirical CDFs properly propagate NaNs:
- If `data_trans` contains NaN, `_transform_sim` returns NaN for that value
- These NaNs are then interpolated via `interpolate_na(dim="time")`
- **If interpolation fails** (boundaries/sparse data), NaNs persist
- The assertion `assert np.all(np.isfinite(self.ranks.values))` fails

## Proposed Fix (for Task 4)

Add fallback fill methods after linear interpolation:

```python
self.ranks = self.ranks.interpolate_na(dim="time")

# Fallback for boundary values and sparse data
if np.isnan(self.ranks.values).any():
    self.ranks = self.ranks.bfill(dim="time").ffill(dim="time")
```

This ensures:
- Primary method: Linear interpolation (preserves temporal patterns)
- Fallback: Forward-fill (ffill) and backward-fill (bfill) for boundaries
- Result: All ranks are finite before assertion

## Next Steps (Task 4)

1. Implement fallback fill methods in `multisite.py:1157`
2. Test with `test_small_ensemble_generation`
3. Verify no degradation in other tests
4. Remove temporary diagnostic instrumentation

## Files Modified in Task 3
- `/home/dirk/Sync/weathercop/src/weathercop/tests/conftest.py` (temporarily set verbose=True, reverted)

## References
- Plan: `/home/dirk/Sync/weathercop/docs/plans/2025-10-31-fix-ranks-nans.md`
- Task 1: Added diagnostic instrumentation
- Task 2: Confirmed CDF NaN propagation behavior
