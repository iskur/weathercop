# DWD OpenData API Quick Reference

## Installation

```bash
pip install dwd_opendata
# or from source:
cd /path/to/dwd_opendata_client && uv sync
```

## Import

```python
import dwd_opendata as dwd
```

---

## Main Functions at a Glance

### 1. Load Data (Most Common)

```python
# Download data for multiple stations
data = dwd.load_data(
    station_names=["Konstanz", "Munich"],     # str or List[str]
    variables=["air_temperature", "precipitation"],  # str or List[str]
    time="hourly",                            # "hourly", "daily", "10_minutes"
    start_year=2015,                          # int, optional
    end_year=2020,                            # int, optional
    redownload=False,                         # bool
    verbose=False                             # bool
)
# Returns: xr.Dataset
```

### 2. Load Single Station

```python
data = dwd.load_station(
    station_name="Konstanz",                  # str, single station
    variables=["air_temperature", "precipitation"],
    time="hourly",
    era="historical",                         # "historical", "recent", None
    start_year=2015,
    end_year=2020
)
# Returns: xr.DataArray
```

### 3. Find Available Stations

```python
# Get metadata for stations with specific variables
metadata = dwd.get_metadata(
    variables=["air_temperature", "precipitation"],
    time="hourly",
    era="historical"
)
# Returns: pd.DataFrame with columns:
#   Stationsname, Stations_id, von_datum, bis_datum,
#   Stationshoehe, geoBreite, geoLaenge, Bundesland

# Filter by geographic region
filtered = dwd.filter_metadata(
    metadata,
    lon_min=8.5, lon_max=13.5,   # West to East (Germany ~5-16)
    lat_min=47.3, lat_max=50.6   # South to North (Germany ~47-55)
)
```

### 4. Visualize Stations

```python
fig, ax = dwd.map_stations(
    variables=["air_temperature", "precipitation"],
    lon_min=8.5, lon_max=13.5,
    lat_min=47.3, lat_max=50.6
)
fig.savefig("station_map.png")
```

---

## Available Variables

### Hourly (`time="hourly"`)

```
air_temperature        Temperature at 2m [°C]
relative_humidity      Relative humidity at 2m [%]
wind                   Wind speed (m/s) + direction (°)
wind_speed            Wind speed at 10m [m/s]
wind_direction        Wind direction [0-360°]
precipitation         Precipitation amount [mm]
pressure              Air pressure [hPa]
cloudiness            Cloud coverage [%]
soil_temperature      Soil temperature at various depths [°C]
solar                 Solar radiation [J/cm²]
solar_diffuse         Diffuse solar radiation [J/cm²]
solar_global          Global solar radiation [J/cm²]
solar_duration        Sunshine duration [minutes]
sun                   Sunshine duration [minutes]
```

### Daily (`time="daily"`)

```
air_temperature       Mean temperature [°C]
air_temperature_max   Maximum temperature [°C]
air_temperature_min   Minimum temperature [°C]
precipitation_daily   Daily precipitation [mm]
wind                  Wind speed + direction
sun                   Sunshine duration
solar_in              Solar radiation
```

### 10-Minute (`time="10_minutes"`)

```
solar                 Solar radiation
solar_diffuse         Diffuse solar radiation
solar_global          Global solar radiation
solar_duration        Sunshine duration
```

---

## Data Access Patterns

### Pattern 1: xarray Dataset (multiple stations)

```python
data = dwd.load_data(["Konstanz", "Munich"], "air_temperature")

# Access by station and variable
temp_konstanz = data.sel(station='Konstanz', met_variable='air_temperature')
# Returns: (time,) array

# Convert to pandas
df = data.to_dataframe()
```

### Pattern 2: xarray DataArray (single station)

```python
data = dwd.load_station("Konstanz", ["air_temperature", "precipitation"])

# Access by variable
temps = data.sel(met_variable='air_temperature').values  # numpy array
# Returns: (time,) array

# Access station name
name = data.name  # 'Konstanz'
```

### Pattern 3: pandas DataFrame (internal)

```python
df = dwd._load_station_one_var("Konstanz", "air_temperature")
# Index: DatetimeIndex (time)
# Columns: ['air_temperature']
```

---

## Commonly Used Stations

Known German weather stations:
- **Konstanz** (Lake Constance, Baden-Wuerttemberg)
- **Munich** (Bayern)
- **Berlin** (Berlin)
- **Hamburg** (Hamburg)
- **Cologne** (North Rhine-Westphalia)
- **Stuttgart** (Baden-Wuerttemberg)
- **Feldberg/Schwarzwald** (Black Forest)

To see all available stations:
```python
metadata = dwd.get_metadata("air_temperature")
print(metadata['Stationsname'].unique())
```

---

## Data Caching

Default cache location:
```
~/.local/share/opendata_dwd
```

Override with environment variable:
```bash
export XDG_DATA_HOME=/custom/path
```

Force re-download:
```python
data = dwd.load_data(..., redownload=True)
```

---

## Time Resolutions

| Resolution | Typical Data Coverage | Update Frequency |
|---|---|---|
| `hourly` | Most complete, varies by variable | Most frequent |
| `daily` | Longer history, basic variables | Daily |
| `10_minutes` | Mainly solar data, recent years | Most granular |

---

## Validation Checklist

Before loading data:

1. **Check station existence:**
   ```python
   metadata = dwd.get_metadata("air_temperature")
   assert "Konstanz" in metadata['Stationsname'].values
   ```

2. **Verify variable availability:**
   ```python
   # Get stations with BOTH variables
   meta = dwd.get_metadata(["air_temperature", "precipitation"])
   ```

3. **Check data availability:**
   ```python
   meta = dwd.get_metadata("air_temperature")
   station = meta[meta['Stationsname'] == 'Konstanz'].iloc[0]
   print(f"Data from {station['von_datum']} to {station['bis_datum']}")
   ```

4. **Filter by region:**
   ```python
   filtered = dwd.filter_metadata(
       metadata,
       lon_min=9.0, lon_max=10.0,
       lat_min=47.5, lat_max=48.5
   )
   ```

---

## Common Errors and Solutions

| Error | Solution |
|-------|----------|
| `URLError: Could not download` | Check internet, try `redownload=True` |
| `No data for {station_name}` | Use `get_metadata()` to verify station has variable |
| `No stations match requirements` | Expand geographic bounds |
| `BadZipFile` | Corrupted download, use `redownload=True` |
| `KeyError: variable` | Check correct variable name for time resolution |

---

## Integration with WeatherCop

Planned integration:

```python
from weathercop import Multisite

# Create multisite model from DWD data
model = Multisite.from_dwd(
    station_names=["Konstanz", "Munich"],
    variables=["air_temperature", "precipitation"],
    start_year=1990,
    end_year=2020
)

# Generate synthetic weather
synthetic = model.simulate(n_samples=100, n_years=30)
```

---

## Full Example

```python
import dwd_opendata as dwd
import datetime

# 1. Find stations in a region
metadata = dwd.get_metadata(
    variables=["air_temperature", "precipitation", "wind"],
    time="hourly"
)

bavarian = dwd.filter_metadata(
    metadata,
    lon_min=8.5, lon_max=13.5,
    lat_min=47.3, lat_max=50.6,
    start=datetime.datetime(1990, 1, 1),
    end=datetime.datetime(2020, 12, 31)
)

print(f"Found {len(bavarian)} stations")
print(bavarian[["Stationsname", "geoBreite", "geoLaenge"]])

# 2. Load data for multiple stations
station_names = bavarian['Stationsname'].tolist()[:5]  # Top 5
data = dwd.load_data(
    station_names,
    variables=["air_temperature", "precipitation"],
    time="hourly",
    start_year=2010,
    end_year=2020,
    verbose=True
)

# 3. Analyze data
print(data)
temp_mean = data.sel(met_variable='air_temperature').mean()
print(f"Mean temperature: {temp_mean.values:.2f}°C")

# 4. Export to pandas
df = data.to_dataframe().reset_index()
print(df.describe())
```

---

## Reference Links

- **DWD Website**: https://www.dwd.de/
- **Open Data Portal**: https://opendata.dwd.de/
- **FTP Server**: ftp://opendata.dwd.de/
- **GitHub Repository**: https://github.com/iskur/dwd_opendata

---

## Version Info

- **dwd_opendata version**: 0.1.0 (experimental)
- **Python requirement**: 3.13+
- **Last updated**: 2025-10-26
