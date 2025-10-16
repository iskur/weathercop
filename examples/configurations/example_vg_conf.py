"""
Example VG configuration for WeatherCop.

Copy this file and customize for your own weather data setup.
"""
from pathlib import Path

# Example paths - customize these for your setup
root = Path("/path/to/your/data")
stations_file = root / "stations.csv"
weather_data_dir = root / "weather_data"

# Add your VG-specific configuration here
# See https://github.com/iskur/vg for VG documentation
