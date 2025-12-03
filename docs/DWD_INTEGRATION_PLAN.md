# DWD OpenData Integration Plan for WeatherCop

## Executive Summary

This document outlines the complete API specification of the `dwd_opendata_client` library and provides a detailed integration plan for incorporating it into the WeatherCop project. The `dwd_opendata` library provides a simple, user-friendly interface for downloading German Weather Service (DWD) meteorological data.

---

## Table of Contents

1. [API Overview](#api-overview)
2. [Core API Functions](#core-api-functions)
3. [Data Variables Reference](#data-variables-reference)
4. [Return Data Formats](#return-data-formats)
5. [Usage Examples](#usage-examples)
6. [Integration Strategy](#integration-strategy)
7. [Implementation Roadmap](#implementation-roadmap)

---

## API Overview

### Library Information
- **Package**: `dwd_opendata`
- **Python Version**: 3.13+
- **Main Module**: Single file implementation at `src/dwd_opendata/__init__.py` (~777 lines)
- **Data Source**: German Weather Service (DWD) FTP server at `opendata.dwd.de`
- **Default Cache Directory**: `~/.local/share/opendata_dwd` (respects `XDG_DATA_HOME`)

### Key Dependencies
- **xarray**: Multi-dimensional data arrays (primary output format)
- **pandas**: Data processing and CSV parsing
- **cartopy**: Geospatial visualization
- **tqdm**: Progress bars

---

## Core API Functions

### 1. **load_data()** - Main High-Level API

**Signature:**
```python
def load_data(
    station_names: Union[str, List[str]],
    variables: Union[str, List[str]],
    redownload: bool = False,
    time: str = "hourly",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    verbose: bool = False,
) -> xr.Dataset:
```

**Parameters:**
- `station_names` (str or List[str]): Weather station name(s), e.g., `"Konstanz"` or `["Konstanz", "Munich"]`
- `variables` (str or List[str]): Meteorological variable(s) to load
- `redownload` (bool): Force re-download of data files, ignoring cache. Default: `False`
- `time` (str): Time resolution - `"hourly"` (default), `"daily"`, or `"10_minutes"`
- `start_year` (int, optional): Filter to data from this year onward
- `end_year` (int, optional): Filter to data up to this year
- `verbose` (bool): Show progress bar with tqdm. Default: `False`

**Returns:**
- `xr.Dataset`: Combined meteorological data for all stations and variables
  - Dimensions: `(station, time, met_variable)`
  - Includes coordinate information for station locations

**Behavior:**
- Automatically downloads and caches data files from DWD FTP
- Handles multiple stations and variables transparently
- Gracefully skips stations with no data for requested variables
- Returns `None` if no data found for a station

**Example:**
```python
import dwd_opendata as dwd

data = dwd.load_data(
    station_names=["Konstanz", "Munich", "Berlin"],
    variables=["air_temperature", "precipitation", "wind"],
    time="hourly",
    start_year=2015,
    end_year=2020
)
# data is xr.Dataset with shape (3, time_steps, 3)
```

---

### 2. **load_station()** - Single Station, Multiple Variables

**Signature:**
```python
def load_station(
    station_name: str,
    variables: Union[str, List[str]],
    redownload: bool = False,
    era: Optional[str] = None,
    time: str = "hourly",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> xr.DataArray:
```

**Parameters:**
- `station_name` (str): Single station name, e.g., `"Konstanz"`
- `variables` (str or List[str]): Variable(s) to load for this station
- `era` (str, optional): Data era - `"historical"`, `"recent"`, or `None` (default)
- Other parameters same as `load_data()`

**Returns:**
- `xr.DataArray`: Data with dimensions `(time, met_variable)`
- `.name` attribute contains station name
- Returns `None` if no data available

**Example:**
```python
station_data = dwd.load_station(
    "Konstanz",
    variables=["air_temperature", "relative_humidity"],
    time="hourly",
    start_year=2010,
    end_year=2020
)
print(station_data.coords["met_variable"].values)
# Output: ['air_temperature', 'relative_humidity']
```

---

### 3. **get_metadata()** - Station Discovery

**Signature:**
```python
def get_metadata(
    variables: Union[str, List[str]],
    era: Optional[str] = None,
    time: str = "hourly",
    redownload: bool = False,
) -> pd.DataFrame:
```

**Parameters:**
- `variables` (str or List[str]): Variable(s) to search for
- `era` (str, optional): Data era to filter by
- `time` (str): Time resolution filter
- `redownload` (bool): Force re-download metadata

**Returns:**
- `pd.DataFrame`: Metadata for stations offering **all** requested variables
  - Index: `Stations_id` (integer)
  - Columns: `Stationsname`, `von_datum`, `bis_datum`, `Stationshoehe`, `geoBreite`, `geoLaenge`, `Bundesland`

**Key Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `Stationsname` | str | Station name (German) |
| `Stations_id` | int | DWD station ID (5 digits) |
| `von_datum` | datetime | Data availability start date |
| `bis_datum` | datetime | Data availability end date |
| `geoBreite` | float | Latitude (decimal degrees) |
| `geoLaenge` | float | Longitude (decimal degrees) |
| `Stationshoehe` | float | Station elevation (meters) |
| `Bundesland` | str | German state/region |

**Example:**
```python
# Find all stations with both wind and temperature data
metadata = dwd.get_metadata(["wind", "air_temperature"], time="hourly")
print(metadata.shape)  # (n_stations, 8 columns)
print(metadata[["Stationsname", "geoBreite", "geoLaenge", "Stationshoehe"]])
```

---

### 4. **filter_metadata()** - Spatial and Temporal Filtering

**Signature:**
```python
def filter_metadata(
    metadata: pd.DataFrame,
    lon_min: Optional[float] = None,
    lat_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_max: Optional[float] = None,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
) -> pd.DataFrame:
```

**Parameters:**
- `metadata`: DataFrame from `get_metadata()`
- `lon_min`, `lon_max`: Longitude bounds (degrees, west to east)
- `lat_min`, `lat_max`: Latitude bounds (degrees, south to north)
- `start`, `end`: Datetime objects for data availability filtering

**Returns:**
- Filtered `pd.DataFrame` with matching stations
- Returns empty DataFrame if no stations match criteria

**Example:**
```python
# Get stations in Bavaria with temperature data available 1980-2020
import datetime

all_meta = dwd.get_metadata("air_temperature")
bavarian = dwd.filter_metadata(
    all_meta,
    lon_min=8.5, lon_max=13.5,
    lat_min=47.3, lat_max=50.6,
    start=datetime.datetime(1980, 1, 1),
    end=datetime.datetime(2020, 12, 31)
)
```

---

### 5. **map_stations()** - Visualization

**Signature:**
```python
def map_stations(
    variables: Union[str, List[str]],
    lon_min: Optional[float] = None,
    lat_min: Optional[float] = None,
    lon_max: Optional[float] = None,
    lat_max: Optional[float] = None,
    start: Optional[datetime.datetime] = None,
    end: Optional[datetime.datetime] = None,
    **skwds,
) -> Tuple[plt.Figure, plt.Axes]:
```

**Returns:**
- Tuple of `(matplotlib.figure.Figure, matplotlib.axes.Axes)`
- Creates OSM map with station locations marked
- Shows station names and data availability years

---

### 6. **load_metadata()** - Internal Metadata Handling

**Signature:**
```python
def load_metadata(
    variable: str,
    era: Optional[str] = None,
    time: str = "hourly",
    redownload: bool = False,
) -> pd.DataFrame:
```

**Parameters:**
- `variable` (str): Single variable name
- Other parameters same as above

**Returns:**
- Raw metadata DataFrame for a single variable
- **Note**: Returns stations with **only this variable**, not intersection with other variables

**Usage Note:**
- Typically use `get_metadata()` instead, which handles multiple variables correctly
- `load_metadata()` is lower-level and requires manual station intersection logic

---

### 7. **_load_station_one_var()** - Low-Level Data Loading

**Signature:**
```python
def _load_station_one_var(
    station_name: str,
    variable: str,
    *,
    era: Optional[str] = None,
    time: str = "hourly",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    redownload: bool = False,
    download_pdf: bool = True,
) -> pd.DataFrame:
```

**Returns:**
- `pd.DataFrame` with columns being the requested variable
- Index: `time` (DatetimeIndex with name "time")
- `.name` attribute: station name

**Note:** This is a private/internal function. Use `load_station()` or `load_data()` instead.

---

## Data Variables Reference

### Hourly Time Resolution (`time="hourly"`)

Available variables and their column names in output data:

| Variable Name | Short Code | Column Name | Description | Units |
|---|---|---|---|---|
| `air_temperature` | TU | TT_TU | Temperature at 2m | °C |
| `relative_humidity` | TU | RF_TU | Relative humidity at 2m | % |
| `wind` | FF | F, D | Wind speed and direction | m/s, degrees |
| `wind_speed` | F | F | Wind speed at 10m | m/s |
| `wind_direction` | D | D | Wind direction | degrees (0-360) |
| `precipitation` | RR | R1 | Precipitation | mm |
| `pressure` | P0 | P0 | Air pressure at sea level | hPa |
| `cloudiness` | N | N | Cloud coverage | % |
| `soil_temperature` | EB | Multiple | Soil temperature at various depths | °C |
| `solar` | ST | Multiple | Solar radiation | J/cm² |
| `solar_diffuse` | DS | DS_10 | Diffuse solar radiation | J/cm² |
| `solar_global` | ST | FG_LBERG | Global solar radiation | J/cm² |
| `solar_duration` | SD | SD_10 | Sunshine duration | minutes |
| `sun` | SD | SD_SO | Sunshine duration | minutes |

**Note:** Some variables like `solar_*` may have multiple representations. The library handles automatic mapping.

### Daily Time Resolution (`time="daily"`)

| Variable Name | Short Code | Column Name | Description |
|---|---|---|---|
| `air_temperature` | KL | TT_TU | Mean temperature |
| `air_temperature_max` | KL | TXK | Maximum temperature |
| `air_temperature_min` | KL | TNK | Minimum temperature |
| `precipitation_daily` | KL | RSK | Daily precipitation |
| `wind` | FF | F, D | Wind speed and direction |
| `sun` | SD | SD_SO | Sunshine duration |
| `solar_in` | ST | FG_STRAHL | Solar radiation |

### 10-Minute Time Resolution (`time="10_minutes"`)

Primarily for solar radiation variables:

| Variable Name | Short Code | Description |
|---|---|---|
| `solar` | SOLAR | Solar radiation data |
| `solar_diffuse` | SOLAR | Diffuse solar radiation |
| `solar_global` | SOLAR | Global solar radiation |
| `solar_duration` | SD | Sunshine duration (10-min intervals) |

### Complete Variable Availability by Time Resolution

```python
# Hourly (most complete)
hourly_vars = [
    "cloudiness", "solar", "solar_diffuse", "solar_global", "solar_duration",
    "solar_long", "sun", "precipitation", "pressure", "air_temperature",
    "soil_temperature", "relative_humidity", "wind", "wind_speed", "wind_direction"
]

# Daily (subset)
daily_vars = [
    "precipitation_daily", "air_temperature", "air_temperature_max",
    "air_temperature_min", "solar_in", "solar", "solar_global", "wind", "sun"
]

# 10-minute (solar-focused)
ten_min_vars = [
    "solar", "solar_diffuse", "solar_global", "solar_duration", "solar_long"
]
```

---

## Return Data Formats

### Format 1: xarray.Dataset (from `load_data()`)

```python
data = dwd.load_data(
    ["Konstanz", "Munich"],
    ["air_temperature", "precipitation"],
    time="hourly"
)
# type(data) == xarray.Dataset

# Structure:
# <xarray.Dataset>
# Dimensions:  (station: 2, time: 8760, met_variable: 2)
# Coordinates:
#   * station        (station) object 'Konstanz' 'Munich'
#   * time           (time) datetime64[ns] 2020-01-01T00:00:00 ... 2020-12-31T23:00:00
#   * met_variable   (met_variable) object 'air_temperature' 'precipitation'
# Data variables:
#     (none - data is in coordinates structure)
```

**Accessing data:**
```python
# Get temperature for Konstanz
temp_konstanz = data.sel(station='Konstanz', met_variable='air_temperature')
# Result: (time: 8760) array

# Get all data for a variable
all_temp = data.sel(met_variable='air_temperature')
# Result: (station: 2, time: 8760) array

# Convert to DataFrame for easier manipulation
df = data.to_dataframe()
```

### Format 2: xarray.DataArray (from `load_station()`)

```python
station_data = dwd.load_station(
    "Konstanz",
    ["air_temperature", "precipitation"],
    time="hourly"
)
# type(station_data) == xarray.DataArray

# Structure:
# <xarray.DataArray (time: 8760, met_variable: 2)>
# Coordinates:
#   * time           (time) datetime64[ns] ...
#   * met_variable   (met_variable) object 'air_temperature' 'precipitation'
# Attributes:
#     name: Konstanz
```

**Accessing data:**
```python
# Get temperature values
temps = station_data.sel(met_variable='air_temperature').values
# Result: numpy array (8760,)

# Get metadata
station_name = station_data.name
# Result: 'Konstanz'
```

### Format 3: pandas.DataFrame (from `_load_station_one_var()` or internal)

```python
df = dwd._load_station_one_var("Konstanz", "air_temperature")
# type(df) == pandas.DataFrame

# Structure:
# Index: DatetimeIndex (time)
# Columns: ['air_temperature']
# Values: float64 (temperature in °C, NaN for missing data)

# Example:
#                      air_temperature
# time                              
# 2020-01-01 00:00:00          -2.5
# 2020-01-01 01:00:00          -3.0
# 2020-01-01 02:00:00          NaN
```

### Format 4: pandas.DataFrame (from `get_metadata()`)

```python
metadata = dwd.get_metadata(["air_temperature", "wind"])
# type(metadata) == pandas.DataFrame

# Structure:
# Index: Stations_id (int)
# Columns: [Stationsname, von_datum, bis_datum, Stationshoehe, geoBreite, geoLaenge, Bundesland]

# Example:
#          Stationsname von_datum  bis_datum  Stationshoehe  geoBreite  geoLaenge Bundesland
# Stations_id                                                                      
# 433      Konstanz     1990-01-01 2020-12-31 443.0         47.6779   9.1732    Baden-Wuerttemberg
# 434      Munich       1995-01-01 2020-12-31 500.0         48.0      9.5       Bayern
```

---

## Usage Examples

### Example 1: Download Temperature and Precipitation for Multiple Stations

```python
import dwd_opendata as dwd

# Load data for three stations
data = dwd.load_data(
    station_names=["Konstanz", "Munich", "Berlin"],
    variables=["air_temperature", "precipitation"],
    time="hourly",
    start_year=2015,
    end_year=2020,
    verbose=True  # Show progress bar
)

# Access temperature for Berlin
berlin_temp = data.sel(station='Berlin', met_variable='air_temperature')
print(berlin_temp.mean().values)  # Mean temperature in °C

# Convert to pandas for statistical analysis
df = data.to_dataframe().reset_index()
print(df.describe())
```

### Example 2: Find Stations in a Region with Required Data

```python
import dwd_opendata as dwd
import datetime

# Find all stations in southern Germany with wind and solar data
metadata = dwd.get_metadata(["wind", "solar"], time="hourly")

# Filter to geographic region (Bavaria/Baden-Wuerttemberg)
bavarian = dwd.filter_metadata(
    metadata,
    lon_min=8.5,   # West
    lon_max=13.5,  # East
    lat_min=47.3,  # South
    lat_max=50.6,  # North
    start=datetime.datetime(1990, 1, 1),
    end=datetime.datetime(2010, 12, 31)
)

print(f"Found {len(bavarian)} stations with both variables")
print(bavarian[["Stationsname", "geoBreite", "geoLaenge", "Stationshoehe"]])
```

### Example 3: Daily Data for Long-Term Analysis

```python
import dwd_opendata as dwd

# Load daily max/min temperatures for trend analysis
data = dwd.load_data(
    ["Konstanz", "Munich"],
    ["air_temperature_max", "air_temperature_min"],
    time="daily",
    start_year=1980,
    end_year=2020
)

# Calculate temperature range
tmax = data.sel(met_variable='air_temperature_max').values
tmin = data.sel(met_variable='air_temperature_min').values
daily_range = tmax - tmin

print(f"Mean daily temperature range: {daily_range.mean():.2f}°C")
```

### Example 4: 10-Minute Solar Data

```python
import dwd_opendata as dwd

# Get high-resolution solar radiation data for recent period
solar = dwd.load_station(
    "Konstanz",
    "solar_global",
    time="10_minutes",
    start_year=2019,
    end_year=2020
)

# Extract solar radiation values
solar_values = solar.sel(met_variable='solar_global').values
print(f"Shape: {solar_values.shape}")  # (52560, ) for 1 year at 10-min resolution
```

### Example 5: Filter by Data Availability

```python
import dwd_opendata as dwd
import pandas as pd

# Get all temperature data
metadata = dwd.get_metadata("air_temperature", time="hourly")

# Find stations with continuous data from 1990-2020
target_start = pd.Timestamp('1990-01-01')
target_end = pd.Timestamp('2020-12-31')

continuous = metadata[
    (metadata['von_datum'] <= target_start) &
    (metadata['bis_datum'] >= target_end)
]

print(f"Found {len(continuous)} stations with 30+ years of data")
station_names = continuous['Stationsname'].tolist()
print("Available stations:", station_names)
```

---

## Integration Strategy

### Integration Point: WeatherCop's Data Input Layer

The `dwd_opendata` library should be integrated into WeatherCop as a **data source adapter** that:

1. **Replaces or supplements** existing data loading mechanisms
2. **Provides a unified interface** for DWD data access
3. **Maintains compatibility** with existing WeatherCop workflows
4. **Enables multisite generation** from real-world German weather stations

### Integration Architecture

```
┌─────────────────────────────────────────┐
│      User Input (CLI/Config)            │
│  - Station names or coordinates         │
│  - Date range (start_year, end_year)    │
│  - Variables needed                     │
└────────────────┬────────────────────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ dwd_opendata.load_ │
        │ data() or          │
        │ load_station()     │
        └────────┬───────────┘
                 │
        ┌────────▼────────────────┐
        │  xarray.Dataset or      │
        │  xarray.DataArray       │
        └────────┬────────────────┘
                 │
                 ▼
        ┌────────────────────────────┐
        │ WeatherCop Data Adapter    │
        │ (NEW MODULE)               │
        │ - Validation               │
        │ - Preprocessing            │
        │ - Format conversion        │
        └────────┬───────────────────┘
                 │
        ┌────────▼──────────────────┐
        │ VARWG Integration         │
        │ - Marginal fitting        │
        │ - Variable transformation  │
        └────────┬──────────────────┘
                 │
        ┌────────▼──────────────────┐
        │ Vine Copula Generation    │
        │ (Existing WeatherCop)     │
        └──────────────────────────┘
```

### Key Integration Points

#### 1. **Configuration Module** (`src/weathercop/dwd_config.py` - NEW)

Create configuration for DWD-specific parameters:

```python
# Example configuration
DWD_CONFIG = {
    "default_time_resolution": "hourly",  # hourly, daily, 10_minutes
    "default_era": "historical",
    "cache_enabled": True,
    "cache_directory": None,  # Uses DWD default if None
    "max_stations": 10,
    "variables_map": {
        # Map WeatherCop variable names to DWD names
        "tas": "air_temperature",
        "tasmax": "air_temperature_max",
        "tasmin": "air_temperature_min",
        "pr": "precipitation",
        "rsds": "solar_global",
        "wind": "wind_speed",
        # ... more mappings
    }
}
```

#### 2. **Data Adapter Module** (`src/weathercop/dwd_adapter.py` - NEW)

Wrapper around `dwd_opendata` that:
- Validates inputs (station existence, variable availability)
- Converts xarray to WeatherCop-compatible formats
- Handles missing data
- Provides progress reporting

```python
class DWDAdapter:
    def __init__(self, config=None):
        self.config = config or DWD_CONFIG
        
    def load_stations(self, station_names, variables, start_year, end_year):
        """Download and validate DWD data."""
        # Calls dwd_opendata.load_data()
        # Validates data completeness
        # Returns preprocessed xarray Dataset
        pass
    
    def find_stations_in_region(self, bbox, variables, start_year, end_year):
        """Find available stations in geographic region."""
        # Calls dwd_opendata.get_metadata()
        # Filters by bbox
        # Returns station list with metadata
        pass
```

#### 3. **Multisite Integration** (`src/weathercop/multisite.py` - MODIFY)

Update the existing `Multisite` class to:
- Accept DWD data sources
- Auto-convert metadata to WeatherCop station objects
- Handle marginal preprocessing via VARWG

```python
class Multisite:
    @staticmethod
    def from_dwd(
        station_names: List[str],
        variables: List[str],
        start_year: int,
        end_year: int,
        **kwargs
    ) -> 'Multisite':
        """Create Multisite workflow from DWD data."""
        # Load from DWD
        # Setup VARWG for each station
        # Create vine copulas
        # Return configured Multisite instance
        pass
```

### Data Flow for Integration

1. **Input**: Station names + variable list + date range
2. **DWD Query**: `dwd_opendata.load_data()` → xarray Dataset
3. **Validation**: Check data completeness, flag missing values
4. **VARWG Processing**: Marginal normalization and transformation
5. **Vine Modeling**: Create copulas from normalized data
6. **Simulation**: Generate synthetic weather samples

---

## Implementation Roadmap

### Phase 1: Adapter Layer (Week 1)
- Create `dwd_adapter.py` with DWDAdapter class
- Implement input validation
- Add error handling and logging
- Write unit tests

**Deliverables:**
- `/src/weathercop/dwd_adapter.py`
- `/src/weathercop/tests/test_dwd_adapter.py`

### Phase 2: Multisite Integration (Week 2)
- Add `Multisite.from_dwd()` class method
- Integrate with VARWG preprocessing
- Handle multi-variable scenarios
- Create integration tests

**Deliverables:**
- Updated `/src/weathercop/multisite.py`
- `/src/weathercop/tests/test_multisite_dwd.py`

### Phase 3: Documentation & Examples (Week 3)
- Write user documentation
- Create Jupyter notebook examples
- Document variable mappings
- Performance profiling

**Deliverables:**
- `/docs/DWD_USAGE.md`
- `/examples/dwd_example.ipynb`
- Configuration guide

### Phase 4: CLI Enhancement (Week 4)
- Add DWD CLI commands
- Station discovery utilities
- Data preview functionality
- Integration with existing CLI

**Deliverables:**
- Updated CLI with DWD commands
- Example usage documentation

---

## Common Workflows

### Workflow 1: Generate Synthetic Weather for a Region

```python
from weathercop import Multisite
from weathercop.dwd_adapter import DWDAdapter

# Step 1: Find available stations
adapter = DWDAdapter()
stations = adapter.find_stations_in_region(
    bbox=(8.5, 47.3, 13.5, 50.6),  # Bavaria
    variables=["air_temperature", "precipitation", "wind"],
    start_year=1990,
    end_year=2020
)
print(f"Found {len(stations)} stations")

# Step 2: Create Multisite model
model = Multisite.from_dwd(
    station_names=stations,
    variables=["air_temperature", "precipitation"],
    start_year=1990,
    end_year=2020
)

# Step 3: Generate synthetic weather
synthetic_data = model.simulate(n_samples=100, n_years=30)
```

### Workflow 2: Climate Change Scenario Analysis

```python
# Load historical vs. recent period
historical = dwd.load_data(
    ["Konstanz", "Munich"],
    "air_temperature",
    start_year=1980,
    end_year=1999,
    time="daily"
)

recent = dwd.load_data(
    ["Konstanz", "Munich"],
    "air_temperature",
    start_year=2000,
    end_year=2020,
    time="daily"
)

# Create separate models
model_hist = Multisite.from_dwd_dataset(historical)
model_recent = Multisite.from_dwd_dataset(recent)

# Compare climate statistics
temp_change = model_recent.mean() - model_hist.mean()
print(f"Temperature change: {temp_change:.2f}°C")
```

### Workflow 3: Validation Against Observations

```python
# Load observed data
observed = dwd.load_data(
    ["Konstanz", "Munich", "Berlin"],
    ["air_temperature", "precipitation"],
    start_year=2015,
    end_year=2020
)

# Create model and generate synthetic data
model = Multisite.from_dwd(..., start_year=2010, end_year=2020)
synthetic = model.simulate(n_samples=100, n_years=5)

# Compare statistics
obs_mean = observed.mean(dim='time')
syn_mean = synthetic.mean(dim='time')
rmse = ((obs_mean - syn_mean) ** 2).mean() ** 0.5
```

---

## API Compatibility Notes

### What Works Well with WeatherCop
- ✅ xarray output format (native compatibility)
- ✅ Multi-station capability
- ✅ Multiple variable support
- ✅ Date range filtering
- ✅ Automatic data caching
- ✅ Missing data handling (NaN)

### Considerations
- ⚠️ DWD variable names differ from CMIP conventions (handle via mapping)
- ⚠️ Some variables only available at specific time resolutions
- ⚠️ Older stations have gaps (station dependence on era)
- ⚠️ Station network changed over time (check `von_datum`/`bis_datum`)

### Required Preprocessing
1. **Variable normalization**: Map DWD → WeatherCop variable names
2. **Unit conversion**: Ensure SI units (°C for temperature, mm for precip)
3. **Missing data handling**: Document treatment of NaN values
4. **QA/QC checks**: Flag suspicious values (e.g., temp < -50°C)

---

## Error Handling Reference

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `URLError: Could not download` | FTP connectivity issue | Check internet, retry with `redownload=True` |
| `No stations match requirements` | Geographic bounds too restrictive | Expand bbox or check coordinates |
| `No data for {station_name}` | Station doesn't provide variable | Use `get_metadata()` to verify availability |
| `BadZipFile` | Corrupted download | Use `redownload=True` to re-fetch |
| `KeyError: variable` | Variable name not recognized | Check available variables for time resolution |

### Best Practices

1. **Always validate station availability first:**
   ```python
   metadata = dwd.get_metadata(variables)
   assert station_name in metadata['Stationsname'].values
   ```

2. **Check data gaps:**
   ```python
   data = dwd.load_station(...)
   print(data.isnull().sum())  # Count missing values
   ```

3. **Cache management:**
   ```python
   # First run - downloads data
   data = dwd.load_data(..., redownload=False)
   
   # Subsequent runs - uses cache
   data = dwd.load_data(...)  # Much faster
   ```

---

## Conclusion

The `dwd_opendata` library provides a clean, Pythonic interface to real German weather data. Its xarray output format makes integration with WeatherCop straightforward. The main implementation effort involves:

1. Creating an adapter layer for data validation and preprocessing
2. Integrating with VARWG for marginal handling
3. Exposing convenient entry points via `Multisite.from_dwd()`

This enables WeatherCop users to directly use German weather stations for multisite weather generation and climate analysis.

---

## Appendix: Variable Availability Matrix

```python
# Quick reference for variable availability by time resolution

VARIABLE_AVAILABILITY = {
    "hourly": {
        "air_temperature": True,
        "relative_humidity": True,
        "wind": True,
        "precipitation": True,
        "pressure": True,
        "cloudiness": True,
        "solar": True,
        "soil_temperature": True,
    },
    "daily": {
        "air_temperature": True,
        "air_temperature_max": True,
        "air_temperature_min": True,
        "precipitation_daily": True,
        "wind": True,
        "sun": True,
    },
    "10_minutes": {
        "solar": True,
        "solar_diffuse": True,
        "solar_global": True,
        "solar_duration": True,
    }
}
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-26
**Status**: Reference Implementation Guide
