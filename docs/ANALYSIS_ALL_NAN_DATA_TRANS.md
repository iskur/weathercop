# Analysis: All-NaN Transformed Data (data_trans) from VarWG

## Problem Statement

The original tox error reported: "Ranks contain NaN values: 12785 NaNs" in the `Multisite.__init__` method. This indicated a serious data quality issue that needed proper diagnosis rather than masking.

## Root Cause Analysis

The issue stems from **VarWG's marginal transformation returning all-NaN values** for certain station/variable pairs. When `data_trans` (the transformed data from VarWG) is entirely NaN, the subsequent code path fails:

1. `data_trans_` is all-NaN
2. `dists.norm.fit(data_trans_)` on all-NaN data returns NaN parameters
3. The distribution object has NaN mu/sigma
4. `.cdf()` calls on the distribution return NaN for all values
5. The ranks remain NaN and the assertion fails

## Configuration Difference: Local vs. Tox

The original error appeared in tox but not in local testing. This is because:

**Local Testing:**
- Runs pytest directly: `pytest src/weathercop/tests/`
- Uses source code directly from `/src/weathercop/`
- Imports VarWG from local development checkout
- VarWG is on a development branch that may have fixes

**Tox Testing:**
- Builds a wheel: `uv build`
- Installs the wheel in an isolated environment: `pip install dist/*.whl`
- Installs published dependencies from PyPI
- Uses the **released/published version of VarWG**, not local development version

**Conclusion:** The error was triggered by an older version of VarWG that had the NaN issue. The local development version (on branch `maint/fix-weathercop-nans`) likely has a fix.

## Affected Variables

Based on the data patterns and VarWG's behavior, the likely problematic variables are:

1. **RH (Relative Humidity)**: Often uses empirical distributions; can have edge cases with gaps
2. **sun (Sunshine duration)**: Boundary NaNs (nighttime, seasonal gaps)
3. **R (Precipitation)**: Special rainmix distributions; zero-handling edge cases

## Solution: Diagnostic Checks

Rather than silently filling problematic values (which masks a real error), I've added defensive diagnostic code in `multisite.py` (lines 1127-1150) that:

1. **Detects** all station/variable pairs with completely NaN `data_trans`
2. **Collects** the complete list before raising an error
3. **Reports** each problematic pair with station_name and variable_name
4. **Raises RuntimeError** with actionable information pointing to VarWG

This ensures:
- The error is not silently masked
- Users get complete information about what went wrong
- The message directs them to fix the root cause in VarWG
- The diagnostic information helps identify which VarWG version has the issue

## Commit

Commit `06f6a4c` adds this diagnostic infrastructure.

## Next Steps

If/when the tox error is triggered again with a new VarWG version:

1. The diagnostic will identify exactly which station/variable pairs fail
2. This information can be used to:
   - Report the issue to VarWG maintainers
   - Identify if it's version-specific
   - Determine if a VarWG fix or update is needed
   - Understand if it's data-specific

## Technical Details

### Code Location
- **File**: `src/weathercop/multisite.py`
- **Lines**: 1127-1150
- **Called during**: `Multisite.__init__` after VarWG instances are created and before rank calculation

### Diagnostic Output Example
```
RuntimeError: Found all-NaN transformed data (data_trans) for the following
station/variable pairs. This indicates a VarWG marginal transformation
failure and needs investigation:
  - Station: Weißenburg-Emetzheim, Variable: rh
  - Station: Bad-Kissingen, Variable: sun

This is a serious error that requires fixing in VarWG, not masking in
WeatherCop. Please investigate the VarWG transformation for these variables.
```

## References

- Original error: `docs/tox-errors.log`
- Previous diagnostic work: `docs/Task3_NaN_Diagnostic_Findings.md`
- VarWG repository: `/home/dirk/workspace/python/vg`
- VarWG branch: `maint/fix-weathercop-nans`
