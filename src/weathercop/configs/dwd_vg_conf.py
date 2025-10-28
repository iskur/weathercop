"""DWD OpenData VARWG Configuration for WeatherCop.

This module defines distribution families and seasonal parameters for German
Weather Service (DWD) meteorological variables. It is used with VARWG to
normalize marginal distributions before applying vine copulas.

Variables covered:
- theta: Temperature (°C)
- R: Precipitation (mm)
- sun: Sunshine duration (hours)
- rh: Relative humidity (%)
"""

from varwg.time_series_analysis import distributions
from varwg.time_series_analysis import seasonal_distributions as sd
from varwg.time_series_analysis import seasonal_kde as skde

# Station coordinates (None - using in-situ data, not location-dependent)
latitude = None
longitude = None

# Threshold for daily precipitation sum to count as rain
# Same unit as precipitation in the input file (mm)
threshold = 0.01 * 24  # 0.24 mm daily threshold

# Distribution families for each variable
# Maps variable names to VARWG distribution classes or "empirical" for KDE
dists = {
    "theta": distributions.norm,      # Temperature: normal distribution
    "R": (distributions.RainMix, distributions.kumaraswamy),  # Precip: mixed
    "sun": distributions.norm,         # Sunshine duration: normal
    "rh": "empirical",                 # Relative humidity: empirical (KDE)
}

# Seasonal distribution classes (fit parameters vs day-of-year)
seasonal_classes = dict.fromkeys(dists, sd.SlidingDist)
seasonal_classes["rh"] = skde.SeasonalKDE  # Use KDE for humidity

# Keyword arguments for seasonal distribution fitting
dists_kwds = {
    "R": dict(
        threshold=threshold,          # References top-level threshold
        q_thresh_lower=0.95,          # Lower quantile for rain transition
        q_thresh_upper=0.99,          # Upper quantile for extremes
        doy_width=10,                 # 10-day window for seasonal fitting
        fft_order=9,                  # Fourier series order for smoothing
        tabulate_cdf=True             # Pre-compute CDF for speed
    ),
    "theta": dict(
        doy_width=15,                 # 15-day window for temp seasonality
        fft_order=6,                  # Lower order for temperature smoothing
        tabulate_cdf=True
    ),
    "rh": dict(
        doy_width=10                  # 10-day window for humidity seasonality
    ),
    "sun": dict(
        doy_width=15,                 # 15-day window for sunshine seasonality
        q_thresh_lower=0.8,           # No sun on 80% of days
        q_thresh_upper=0.975,         # Maximum sun near equinoxes
        tabulate_cdf=True
    ),
}

# Known parameters fixed during fitting
par_known = dict.fromkeys(dists)
par_known.update({
    "R": {
        "l": lambda tt: 0 * tt,       # Lower bound: 0 (no negative rain)
        "u": lambda tt: 1.0 + 0 * tt,  # Upper bound: 1 (normalized)
    },
    "rh": {
        "lc": lambda tt: 0 * tt,      # Lower clipping: 0%
        "uc": lambda tt: 1.0 + 0 * tt,  # Upper clipping: 100%
    },
    "sun": {
        "l": lambda tt: 0 * tt,       # Lower bound: no negative sunshine
    },
})

# Human-readable variable names
long_var_names = {
    "theta": "Temperature",
    "R": "Precipitation",
    "sun": "Sunshine Duration",
    "rh": "Relative Humidity",
}

# Unit strings for plotting
units = {
    "theta": "[°C]",
    "R": "[mm]",
    "sun": "[h]",
    "rh": "[%]",
}

# Post-simulation conversions (none needed for DWD data)
conversions = []

# Output formatting
out_format = {
    "theta": "%.3f",
    "R": "%.6f",
    "sun": "%.3f",
    "rh": "%.3f",
}
