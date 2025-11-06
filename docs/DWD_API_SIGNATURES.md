# DWD OpenData - Exact API Signatures

This document provides the exact function signatures from the dwd_opendata library for integration planning.

## Module: `dwd_opendata`

### Main API Functions

#### 1. `load_data()`

**Full Signature:**
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

**Location:** `src/dwd_opendata/__init__.py:576`

**Docstring:**
```
Load meteorological data for multiple stations from the DWD open data platform.

Parameters
----------
station_names : Union[str, List[str]]
    Station name(s) to load. Can be a string or list of strings 
    (e.g., ("Berlin", "Munich"))
variables : Union[str, List[str]]
    Variable(s) to load. Can be a string or list of strings.
redownload : bool, optional
    Force redownload of data files. Default: False
time : str, optional
    Time resolution: "hourly", "daily", or "10_minutes". Default: "hourly"
start_year : int, optional
    Start year for data range. None (default) means from earliest available data.
end_year : int, optional
    End year for data range. None (default) means to latest available data.
verbose : bool, optional
    Show progress bar. Default: False

Returns
-------
xr.Dataset
    Combined data for all requested stations and variables
```

---

#### 2. `load_station()`

**Full Signature:**
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

**Location:** `src/dwd_opendata/__init__.py:516`

**Docstring:**
```
Load meteorological data for a single station from the DWD open data platform.

Parameters
----------
station_name : str
    Name of the weather station (e.g., "Berlin", "Munich")
variables : Union[str, List[str]]
    Variable(s) to load. Can be a string or list of strings.
redownload : bool, optional
    Force redownload of data files. Default: False
era : str, optional
    Era of data (e.g., "historical", "recent"). Default: None
time : str, optional
    Time resolution of data: "hourly", "daily", or "10_minutes". Default: "hourly"
start_year : int, optional
    Start year for data range. None (default) means from earliest available data.
end_year : int, optional
    End year for data range. None (default) means to latest available data.

Returns
-------
xr.DataArray
    Data for the requested station and variables
```

---

#### 3. `get_metadata()`

**Full Signature:**
```python
def get_metadata(
    variables: Union[str, List[str]],
    era: Optional[str] = None,
    time: str = "hourly",
    redownload: bool = False,
) -> pd.DataFrame:
```

**Location:** `src/dwd_opendata/__init__.py:671`

**Docstring:**
```
Get metadata for stations offering requested variables.

Returns stations that provide all requested variables with their location
and availability information.

Parameters
----------
variables : Union[str, List[str]]
    Variable(s) to search for
era : str, optional
    Era of data (e.g., "historical", "recent"). Default: None
time : str, optional
    Time resolution: "hourly", "daily", or "10_minutes". Default: "hourly"
redownload : bool, optional
    Force redownload of metadata. Default: False

Returns
-------
pd.DataFrame
    Metadata for matching stations including name, location, and date range
```

---

#### 4. `filter_metadata()`

**Full Signature:**
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

**Location:** `src/dwd_opendata/__init__.py:633`

**Parameters:**
- `metadata` (pd.DataFrame): DataFrame from get_metadata()
- `lon_min` (float, optional): Minimum longitude of bounding box
- `lat_min` (float, optional): Minimum latitude of bounding box
- `lon_max` (float, optional): Maximum longitude of bounding box
- `lat_max` (float, optional): Maximum latitude of bounding box
- `start` (datetime.datetime, optional): Start date for data availability filter
- `end` (datetime.datetime, optional): End date for data availability filter

**Returns:**
- `pd.DataFrame`: Filtered metadata

---

#### 5. `load_metadata()`

**Full Signature:**
```python
def load_metadata(
    variable: str,
    era: Optional[str] = None,
    time: str = "hourly",
    redownload: bool = False,
) -> pd.DataFrame:
```

**Location:** `src/dwd_opendata/__init__.py:310`

**Parameters:**
- `variable` (str): Single variable name
- `era` (Optional[str]): Era of data
- `time` (str): Time resolution. Default: "hourly"
- `redownload` (bool): Force re-download metadata. Default: False

**Returns:**
- `pd.DataFrame`: Raw metadata for single variable

**Warning:** Returns stations with **only this variable**, not intersection of multiple variables. Use `get_metadata()` instead for multiple variables.

---

#### 6. `map_stations()`

**Full Signature:**
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

**Location:** `src/dwd_opendata/__init__.py:718`

**Returns:**
- `Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]`: Figure and axes for further customization

---

#### 7. `_load_station_one_var()` (Private)

**Full Signature:**
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

**Location:** `src/dwd_opendata/__init__.py:365`

**Note:** Private function. Use `load_station()` or `load_data()` instead.

**Returns:**
- `pd.DataFrame`: Single variable data with DatetimeIndex

---

### URL/Path Functions (Internal)

```python
def get_ftp_root(*, era: str = "historical", time: str = "hourly") -> str:
    """Get FTP root path template."""
    
def get_ftp_variable_root(
    variable: str, *, era: str = "historical", time: str = "hourly"
) -> str:
    """Get FTP root for specific variable."""

def get_ftp_path(
    variable: str, *, time: str = "hourly", era: str = "historical"
) -> str:
    """Get FTP path for variable data."""

def get_data_url(variable: Optional[str] = None, **kwds) -> str:
    """Get data URL from FTP server."""

def get_url(variable, zip_filename, *, time="hourly", era="historical"):
    """Get full download URL for data."""

def get_zip_filename(variable: str, time: str = "hourly", **metadata) -> str:
    """Generate ZIP filename based on variable and station metadata."""

def get_zip_path(variable: str) -> Path:
    """Get local cache path for variable data."""

def get_meta_filename(variable_short: str, time: str = "hourly") -> str:
    """Generate metadata filename."""

def get_meta_url(
    variable: str, era: Optional[str] = None, time: str = "hourly"
) -> str:
    """Get metadata file URL."""

def get_description_url(
    variable: str, era: Optional[str] = None, time: str = "hourly"
) -> str:
    """Get PDF documentation URL."""
```

---

### Constants and Data Structures

#### Available Variables by Time Resolution

**Hourly:**
```python
variable_shorts_hourly = {
    "cloudiness": "N",
    "solar": "ST",
    "solar_diffuse": "DS",
    "solar_global": "ST",
    "solar_duration": "SD",
    "solar_long": "LS",
    "sun": "SD",
    "precipitation": "RR",
    "pressure": "P0",
    "air_temperature": "TU",
    "soil_temperature": "EB",
    "relative_humidity": "TU",
    "wind": "FF",
    "wind_speed": "F",
    "wind_direction": "D",
    "daily": "KL",
}
```

**Daily:**
```python
variable_shorts_daily = {
    "precipitation_daily": "KL",
    "air_temperature": "KL",
    "air_temperature_max": "KL",
    "air_temperature_min": "KL",
    "solar_in": "ST",
    "solar": "solar",
    "solar_global": "solar",
    "wind": "FF",
    "sun": "SD",
}
```

**10-Minute:**
```python
variable_shorts_10_minutes = {
    "solar": "SD",
    # (derived from daily with "SOLAR" for most solar variables)
}
```

#### Variable Columns

```python
variable_cols = {
    "wind_speed": ["F"],
    "wind_direction": ["D"],
    "wind": ["F", "D"],
    "air_temperature": ["TT_TU"],
    "air_temperature_max": ["TXK"],
    "air_temperature_min": ["TNK"],
    "precipitation_daily": ["RSK"],
    "relative_humidity": ["RF_TU"],
    "solar_diffuse": ["DS_10"],
    "solar_global_10_minutes": ["GS_10"],
    "solar_global_hourly": ["FG_LBERG"],
    "solar_duration": ["SD_10"],
    "solar_long": ["LS_10"],
    "solar_in": ["FG_STRAHL"],
    "sun": ["SD_SO"],
    "precipitation": ["R1"],
    "pressure": ["P0"],
}
```

#### Metadata Columns

```python
meta_header = [
    "Stations_id",
    "von_datum",
    "bis_datum",
    "Stationshoehe",
    "geoBreite",
    "geoLaenge",
    "Stationsname",
    "Bundesland",
]

header_dtypes = OrderedDict((
    ("Stations_id", int),
    ("von_datum", int),
    ("bis_datum", int),
    ("Stationshoehe", float),
    ("geoBreite", float),
    ("geoLaenge", float),
    ("Stationsname", str),
    ("Bundesland", str),
))
```

#### Date Formats

```python
date_formats = {
    "10_minutes": "%Y%m%d%H%M",
    "hourly": "%Y%m%d%H",
    "hourly_solar": "%Y%m%d%H:%M",
    "daily": "%Y%m%d",
}
```

---

### Module-Level Variables

```python
# FTP server configuration
ftp_url = "opendata.dwd.de"

# Data directory (respects XDG Base Directory spec)
data_dir = Path(os.getenv(
    "XDG_DATA_HOME", 
    os.path.join(Path.home(), ".local", "share")
)) / "opendata_dwd"

# Debug mode
DEBUG = False

# Version
__version__ = "0.1.0"
```

---

## Data Structure Examples

### xarray.Dataset Example (from load_data)

```python
<xarray.Dataset>
Dimensions:      (station: 2, time: 8760, met_variable: 2)
Coordinates:
  * station      (station) object 'Konstanz' 'Munich'
  * time         (time) datetime64[ns] 2020-01-01T00:00:00 ... 2020-12-31T23:00:00
  * met_variable (met_variable) object 'air_temperature' 'precipitation'
Data variables:
    (none)
```

### xarray.DataArray Example (from load_station)

```python
<xarray.DataArray (time: 8760, met_variable: 2)>
Coordinates:
  * time           (time) datetime64[ns] ...
  * met_variable   (met_variable) object 'air_temperature' 'precipitation'
Attributes:
    name: Konstanz
```

### pandas.DataFrame Example (metadata)

```
              Stationsname von_datum  bis_datum  Stationshoehe  geoBreite  geoLaenge Bundesland
Stations_id                                                                      
433          Konstanz     1990-01-01 2020-12-31 443.0         47.6779   9.1732    Baden-Wuerttemberg
434          Munich       1995-01-01 2020-12-31 500.0         48.0      9.5       Bayern
```

---

## Type Hints Used

```python
from typing import List, Optional, Union, Tuple
import datetime
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
```

---

## Decorator Functions

### `@shorten_compound_varnames`

```python
def shorten_compound_varnames(func):
    """Decorator that shortens 'solar_*' to 'solar'."""
    @functools.wraps(func)
    def wrapped(variable: str, *args, **kwds) -> str:
        if variable.startswith("solar"):
            variable = "solar"
        return func(variable, *args, **kwds)
    return wrapped
```

This decorator is applied to:
- `get_ftp_variable_root()`
- `get_ftp_path()`
- `get_url()`
- `get_zip_path()`

---

## Error Handling

### Exceptions Raised

```python
# URLError - Network/FTP issues
from urllib.error import URLError

# BadZipFile - Corrupted ZIP downloads
import zipfile  # BadZipFile exception

# ValueError - Invalid parameters
# KeyError - Missing variable/key
# AssertionError - Validation failures
```

### Custom Warnings

```python
from warnings import warn

# Used in:
warn(f"Bad zip file: {zip_filepath_existing}...")
warn(f"No {variable} for {station_name}")
warn(f"No stations match requirements.")
warn(f"No data for {station_name} ({variables})")
```

---

## Important Implementation Notes

1. **Variable Normalization**: The `@shorten_compound_varnames` decorator automatically maps `solar_*` variables to `solar`

2. **Station Intersection**: `get_metadata()` returns only stations that have **all** requested variables

3. **Data Caching**: All downloaded data cached in `data_dir`; use `redownload=True` to force fresh downloads

4. **Missing Values**: Represented as NaN; use `df.isnull()` to check

5. **Time Index**: All returned data has DatetimeIndex named "time"

6. **Station Names**: Case-sensitive; must match exact DWD naming

7. **Coordinates**: Latitude/Longitude in decimal degrees (EPSG:4326)

8. **FTP Structure**: Organized as:
   ```
   climate_environment/CDC/observations_germany/climate/{time_resolution}/{variable}/{era}/
   ```

---

## Source Code Location

**Repository**: `/home/dirk/Sync/dwd_opendata_client/`

**Main module**: `src/dwd_opendata/__init__.py` (777 lines)

**Tests**: `src/dwd_opendata/tests/`
- `test_dwd_opendata.py` - Unit tests
- `test_integration.py` - Integration tests with real DWD server

---

**API Signature Document Version**: 1.0
**dwd_opendata Version**: 0.1.0
**Last Updated**: 2025-10-26
