# API Improvements for Better Discoverability and Usability

## Summary

This document captures observations about where the WeatherCop API could be improved to make getting started easier for new users. These are proposed improvements, not currently implemented.

## Issue 1: No Native DWD Integration in WeatherCop

**Problem:**
- Users must install and understand dwd_opendata separately
- No integration between weathercop and dwd_opendata packages
- New users don't know dwd_opendata exists or how to use it

**Current Workflow:**
```python
from dwd_opendata import load_data, get_metadata
# User must know about dwd_opendata, install it, understand its API
xds = load_data(...)
```

**Proposed Solution:**

Add a convenience function in weathercop for DWD data discovery and download:

```python
from weathercop.data import download_dwd

# Simple interface for common use case
xds = download_dwd(
    stations=["berlin", "munich", "hamburg"],
    variables=["air_temperature", "precipitation", "relative_humidity"],
    start_date="2019-01-01",
    end_date="2023-12-31",
)

# Or discover stations first
stations_df = download_dwd.find_stations(
    region="central_germany",  # or lat/lon bounds
    variables=["air_temperature", "precipitation", "relative_humidity"],
)
print(stations_df)  # Shows available stations
```

**Benefits:**
- Single import point: `from weathercop.data import download_dwd`
- Hides dwd_opendata implementation details
- Consistent with weathercop naming and conventions
- Can add future data sources (ECMWF, NOAA, etc.) without breaking existing code

**Priority:** High | **Effort:** 1-2 hours

---

## Issue 2: Multisite Has 20+ Parameters With No Clear Starting Point

**Problem:**
- 20+ parameters in `Multisite.__init__()`, many with unclear purposes
- No clear "recommended for beginners" vs "advanced tuning" categories
- New users don't know which parameters to set or why
- No example parameter combinations for common use cases

**Current Situation:**
```python
class Multisite:
    def __init__(
        self,
        xds: xr.Dataset,
        *args,
        primary_var: str = "theta",
        discretization: str = "D",
        verbose: bool = False,
        infilling: Optional[str] = None,
        refit_vine: bool = False,
        station_vines: bool = False,
        # ... 13 more parameters
    ):
```

**Proposed Solution 1: Add Class Methods for Common Use Cases**

```python
class Multisite:
    @classmethod
    def quick_start(cls, xds: xr.Dataset, **kwds):
        """Initialize for quick learning (minimal configuration)."""
        return cls(
            xds,
            primary_var="theta",
            verbose=True,
            refit_vine=False,
            debias=True,
            # Sensible defaults for beginners
            **kwds  # Allow overrides
        )

    @classmethod
    def research(cls, xds: xr.Dataset, **kwds):
        """Initialize for research (maximum accuracy)."""
        return cls(
            xds,
            primary_var="theta",
            verbose=True,
            refit_vine=True,        # Refit for best fit
            weights="likelihood",   # Slower but more accurate
            debias=True,
            asymmetry=True,         # Model asymmetric dependencies
            **kwds
        )

# Usage:
multisite = Multisite.quick_start(xds)              # For learning
multisite = Multisite.research(xds)                 # For publication
multisite = Multisite(xds, verbose=True, ...)       # Custom
```

**Proposed Solution 2: Better Parameter Documentation**

Group parameters into categories in docstring:

```python
class Multisite:
    """Multisite ensemble weather generator.

    Parameters
    ----------
    xds : xr.Dataset
        Input data (required)

    --- BEGINNER PARAMETERS ---
    primary_var : str, default "theta"
        Variable for climate perturbations. Usually temperature.
    verbose : bool, default False
        Print progress messages. Set True to see what's happening.

    --- INTERMEDIATE PARAMETERS ---
    refit_vine : bool, default False
        Refit vine copulas. False=faster, True=more accurate.
    debias : bool, default True
        Remove systematic biases in generated ensemble.

    --- ADVANCED PARAMETERS ---
    infilling : str, optional
        How to handle missing data: "phase_inflation" or "vg"
    asymmetry : bool, default False
        Model asymmetric spatial dependencies (slower).
    ...
    """
```

**Benefits:**
- New users can focus on beginner parameters
- Clear upgrade path to intermediate/advanced
- Class methods encode best practices for different use cases
- Docstring categories guide parameter selection

**Priority:** High | **Effort:** 1-2 hours

---

## Issue 3: No "Recommended Visualizations" Pattern

**Problem:**
- 10+ plotting methods (`plot_ensemble_stats`, `plot_ensemble_meteogram_daily`, etc.)
- No guidance on which plots to create for common analyses
- New users create too few (missing key diagnostics) or too many (redundant plots)

**Current Pattern:**
```python
# User must know which plots to create
multisite.plot_ensemble_stats()
multisite.plot_ensemble_meteogram_daily()
multisite.plot_ensemble_qq()
multisite.plot_ensemble_violins()
# But which ones matter? When?
```

**Proposed Solution: Add "Validation Pipelines"**

```python
class Multisite:
    def validate_ensemble(self, output_dir: Path = None):
        """Create standard validation plots.

        Recommended for checking if ensemble is physically reasonable.
        Creates:
        - ensemble_stats.png (statistical properties)
        - ensemble_meteograms.png (time series spread)
        - ensemble_qq.png (distribution validation)
        - ensemble_violins.png (variable distributions)
        """
        plots = {}
        plots['stats'] = self.plot_ensemble_stats()
        plots['meteograms'] = self.plot_ensemble_meteogram_daily()
        plots['qq'] = self.plot_ensemble_qq()
        plots['violins'] = self.plot_ensemble_violins()

        if output_dir:
            for name, (fig, axs) in plots.items():
                fig.savefig(output_dir / f"{name}.png")

        return plots

    def diagnose_ensemble(self, obs_xds: xr.Dataset = None):
        """Create diagnostic plots for understanding biases.

        Compares ensemble to observations and highlights:
        - Systematic biases (mean differences)
        - Distribution mismatches (Q-Q plots)
        - Extreme value differences (exceedance plots)
        """
        # Implementation
        pass

# Usage:
multisite.validate_ensemble(Path("output"))
multisite.diagnose_ensemble(obs=xds)  # Compare to observations
```

**Benefits:**
- Clear "start here" plots for new users
- Encodes domain knowledge about validation
- Reduces decision paralysis
- Easier to document ("run validate_ensemble()")

**Priority:** Medium | **Effort:** 2 hours

---

## Issue 4: No Quick Way to Explore What Variables Are Available

**Problem:**
- Variables have cryptic names in dwd_opendata: "air_temperature", "precipitation", "relative_humidity"
- These map to weathercop names: "theta", "R", "rh"
- Users don't know what's available or how names map
- No way to check "what variables does my data have?"

**Current Situation:**
```python
xds = load_data(...)
print(xds.data_vars)  # User must understand xarray
# Output: ['theta', 'R', 'rh', ...]
```

**Proposed Solution:**

Add metadata display method:

```python
class Multisite:
    def describe(self):
        """Print human-readable description of data."""
        print(f"Stations: {self.station_names}")
        print(f"Variables: {self.varnames}")
        print(f"Time period: {self.data.time.values[0]} to {self.data.time.values[-1]}")
        print(f"Time resolution: {self.discretization}")
        print("\nVariable details:")
        for varname in self.varnames:
            var_data = self.data_daily[varname]
            print(f"  {varname}: {var_data.min():.2f} to {var_data.max():.2f} "
                  f"(mean: {var_data.mean():.2f})")

# Usage:
multisite = Multisite(xds)
multisite.describe()
# Output:
# Stations: ['Berlin', 'Munich', 'Hamburg']
# Variables: ['theta', 'R', 'rh']
# Time period: 2019-01-01 to 2023-12-31
# Time resolution: D (daily)
#
# Variable details:
#   theta: 263.42 to 297.15 K (mean: 283.21 K)
#   R: 0.00 to 45.32 mm (mean: 2.18 mm)
#   rh: 25.41 to 99.87 % (mean: 73.15 %)
```

**Benefits:**
- Quick sanity check of data
- Helps users understand their variables
- Detects data quality issues (e.g., all zeros, missing data)

**Priority:** Medium | **Effort:** 1 hour

---

## Issue 5: No Standard "Save & Share Ensemble" Workflow

**Problem:**
- Ensemble is complex xarray Dataset with multiple variables and coordinates
- Users don't know best practices for exporting results
- No clear format for sharing ensembles with collaborators

**Current Situation:**
```python
ensemble = multisite.simulate_ensemble(...)
# Now what? How do I save it?
ensemble.to_netcdf("myfile.nc")  # User must remember this
```

**Proposed Solution:**

Add convenience export methods:

```python
class Multisite:
    def export_ensemble(
        self,
        output_dir: Path,
        format: str = "netcdf",  # "netcdf", "csv", "parquet"
        include_metadata: bool = True,
    ):
        """Export ensemble to standard format.

        Parameters
        ----------
        format : str
            "netcdf" - Single NetCDF file (recommended for large ensembles)
            "csv" - CSV per station/variable (good for spreadsheets)
            "parquet" - Parquet per realization (good for big data tools)
        """
        if format == "netcdf":
            self.ensemble.to_netcdf(output_dir / "ensemble.nc")
        elif format == "csv":
            for station in self.station_names:
                for var in self.varnames:
                    self.ensemble.sel(station=station, variable=var).to_dataframe().to_csv(
                        output_dir / f"{station}_{var}.csv"
                    )
        # ... etc

# Usage:
multisite.export_ensemble(Path("./results"), format="netcdf")
```

**Benefits:**
- Clear, documented export options
- Reduces errors in manual export
- Users know which format to choose for their use case

**Priority:** Low | **Effort:** 1-2 hours

---

## Summary of Recommendations

| Issue | Current UX | Proposed Solution | Priority | Effort |
|-------|-----------|-------------------|----------|--------|
| No weathercop-native DWD integration | Must use separate dwd_opendata package | Add `weathercop.data.download_dwd()` wrapper | High | 1-2h |
| Too many Multisite parameters | 20+ parameters, no clear starting point | Add `Multisite.quick_start()` class method | High | 1-2h |
| Unclear which plots to create | 10+ plotting methods, no guidance | Add `Multisite.validate_ensemble()` method | Medium | 2h |
| No variable overview | Must manually inspect xarray | Add `Multisite.describe()` method | Medium | 1h |
| No standard export workflow | Users improvise | Add `Multisite.export_ensemble()` method | Low | 1-2h |

**Total Implementation Effort:** 6-9 hours for all improvements

**Recommended Approach:** Implement high-priority items first (DWD integration + quick_start class method). These would most significantly improve discoverability and reduce cognitive load for new users.

## Implementation Notes

- Each improvement should be backward compatible (only adds new methods/functions)
- Improvements should follow existing code style and patterns
- Consider adding docstring examples for each new method
- Include in release notes when implemented
