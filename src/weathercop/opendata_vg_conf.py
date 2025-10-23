"""VG configuration file to be used with DWD data and WeatherCop."""

from __future__ import print_function
from builtins import range
import collections
import numpy as np
from scipy import stats
from varwg.time_series_analysis import distributions
from varwg.time_series_analysis import seasonal_distributions as sd
from varwg.time_series_analysis import seasonal_kde as skde
from varwg.meteo import meteox2y_cy, meteox2y

# coordinates of measurement station
# latitude = 47.66
# longitude = 9.18
latitude = None
longitude = None

dists = {
    # "sun":             "empirical",
    # "sun":             distributions.kumaraswamy,
    # "sun":             distributions.beta,
    # "sun":             (distributions.Rain, distributions.kumaraswamy),
    "sun": (distributions.RainMix, distributions.kumaraswamy),
    # "sun":             (distributions.RainMix,
    #                     distributions.expon),
    # "rh":              distributions.truncnorm,
    "rh": "empirical",
    "ah": distributions.norm,
    "R": (distributions.RainMix, distributions.kumaraswamy),
    # "R":                (distributions.RainMix,
    #                      distributions.beta),
    # "R":               (distributions.RainMix, distributions.expon),
    # "R":               (distributions.RainMix, distributions.weibull),
    # "R":               (distributions.RainMix, distributions.gamma),
    # "R":               (distributions.RainMix, distributions.lognormal),
    "theta": distributions.norm,
    # "theta":           "empirical",
    # "theta":            stats.distributions.t,
    # "theta":             stats.distributions.exponnorm,
    # "theta": stats.distributions.johnsonsu,
    # "Cloud_Cover":     distributions.truncnorm,
    "U": distributions.lognormal,
    # "U": distributions.weibull,
    "Qsw": "empirical",
    # "theta_monthly":   distributions.norm,
    # "theta_max":       distributions.norm,
    # "theta_lars":      distributions.norm,
    # "e":               distributions.norm,
    "u": "empirical",
    "v": "empirical",
    # "ILWR":            distributions.norm,
    # "pressure":        distributions.norm,
    # "pressure":        stats.distributions.exponnorm,
    "pressure": "empirical",
}
seasonal_classes = dict.fromkeys(dists, sd.SlidingDist)
# seasonal_classes["sun"] = skde.SeasonalKDE
seasonal_classes["Qsw"] = skde.SeasonalKDE
seasonal_classes["rh"] = skde.SeasonalKDE
seasonal_classes["pressure"] = skde.SeasonalKDE
# seasonal_classes["R"] = skde.SeasonalKDE
seasonal_classes["u"] = seasonal_classes["v"] = skde.SeasonalKDE
# seasonal_classes["theta"] = skde.SeasonalKDE

# threshold for daily precipitation sum to count as such. same unit as
# precipitation in the input file
threshold = 0.0005
# keyword parameters passed to the seasonal distribution initializers
dists_kwds = {
    "R": dict(
        threshold=threshold,
        q_thresh_lower=0.8,
        q_thresh_upper=0.975,
        doy_width=15,
        fft_order=4,
    ),
    "rh": dict(
        doy_width=15,
        # fft_order=10
    ),
    "ah": dict(),
    "Qsw": dict(doy_width=10),
    "theta": dict(
        doy_width=15,
        # freibord=10 * 24,
        fft_order=3,
    ),
    # "theta": dict(doy_width=15,
    #               fft_order=4,
    #               # tabulate_cdf=True
    #               ),
    "sun": dict(
        doy_width=15,
        q_thresh_lower=0.8,
        q_thresh_upper=0.975,
    ),
    "u": dict(doy_width=35),
    "v": dict(doy_width=35),
    "pressure": dict(doy_width=35, freibord=10 * 24),
    "U": dict(doy_width=15, fft_order=3),
}


# functions that return known parameters to fix those values during fitting
def array_gen(scalar):
    # return lambda tt: scalar * np.ones_like(tt)
    return lambda tt: np.full_like(tt, scalar)


# @my.pickle_cache(os.path.join(cache_dir, "max_qsw_%s.pkl"), warn=False)
def max_qsw(doys, **kwds):
    """Use meteo.meteox2y to sum up hourly values of solar radiation for the
    specified days - of - year."""
    doys = np.atleast_1d(doys)
    qsw_2d = -1 * np.ones((22 - 4, len(doys)))
    for i, doy in enumerate(doys):
        date = ["%03d %02d" % (doy, hour) for hour in range(4, 22)]
        qsw_2d[:, i] = meteox2y_cy.pot_s_rad(
            date, longt=longitude, lat=latitude, in_format="%j %H", **kwds
        )
    return qsw_2d.sum(axis=0).ravel()


# read: hours per day
hpd = 24


def max_sunshine_hours(doys):
    from varwg import times

    dates = times.doy2datetime(doys)
    sun_hours = meteox2y.sunshine_hours(
        dates,
        longitude=longitude,
        latitude=latitude,
        # does not matter, because we
        # are not interested in when,
        # but in how long
        # tz_offset=0
        tz_offset=None,
    )
    return sun_hours


# distribution parameters that are fixed during optimization
par_known = dict.fromkeys(dists)
par_known.update(
    {
        "R": {"l": array_gen(0), "u": array_gen(1.0)},
        "sun": {
            "l": array_gen(0),
            "u": np.vectorize(max_sunshine_hours),
        },
        "rh": {
            # "lc": array_gen(-0.001),
            "lc": array_gen(0),
            # "uc": array_gen(hpd * 1.01),
            "uc": array_gen(hpd),
        },
        # "Cloud_Cover": {"l": array_gen(-.1),
        #                 "u": array_gen(hpd + .1)},
        "Qsw": {
            "l": array_gen(0),
            # the .725 is unfortunately a complicated
            # subject it seems as it should also be a
            # function of t
            "u": np.vectorize(
                lambda x: 0.8
                * max_qsw(x + 1 if x < 366 else 1)
                # lambda x: 0.9 * max_qsw(x + 1 if x < 366 else 1)
            ),
        },
        # "Qsw":         {"l": array_gen(1e-6),
        #                 # the .8 is unfortunately a complicated
        #                 # subject it seems as it should also be a
        #                 # function of t
        #                 "u": lambda t: .85 * max_qsw(t),
        #                 },
    }
)
# some of the variables have to be treated differently in the hourly
# discretization
par_known_hourly = {key: val for key, val in list(par_known.items())}
par_known_hourly.update(
    {
        "sun": {
            "lc": array_gen(-0.001),
            "uc": array_gen(60),
        },
        "rh": {
            "lc": array_gen(-0.001),
            "uc": array_gen(1.01),
        },
    }
)
in_vars = dict.fromkeys(dists)
long_var_names = {
    "R": "Precipitation",
    "U": "Wind velocity",
    "u": "Eastward wind speed",
    "v": "Northward wind speed",
    "ILWR": "Incident long-wave radiation",
    "Qsw": "Short-wave radiation",
    "Qsw_lars": "Short-wave radiation",
    "sun": "Sunshine duration",
    "Cloud_Cover": "Cloud cover",
    "theta": "Temperature",
    "theta_lars": "Temperature",
    "theta_monthly": "Monthly averaged Temperature",
    "e": "Vapour pressure",
    "rh": "Relative humidity",
    "pressure": "Atmospheric pressure",
    "nao": "North Atlantic Oscillation Index",
    "wtemp_bregenzerach": "Water temperature of Bregenzer Ach",
    "wtemp_rhein": "Water temperature of Rhine",
}
units = collections.defaultdict(lambda: r"-")
units.update(
    {
        "ILWR": "[$W / m ^ 2$]",
        "e": "[hPa]",
        "rh": "[-]",
        "Cloud_Cover": "[-]",
        "R": "[mm]",
        "Qsw": "[$W / m ^ 2$]",
        "Qsw_lars": "[$W / m ^ 2$]",
        "sun": "[min]",
        "theta": "[$^{\circ}C$]",
        "theta_lars": "[$^{\circ}C$]",
        "theta_monthly": "[$^ {\circ}C$]",
        "U": "[$m / s$]",
        "u": "[$m / s$]",
        "v": "[$m / s$]",
        "pressure": "[hPa]",
    }
)
# long variable names together with units
ylabels = [
    "%s %s" % (long_var_names[var_name], units[var_name])
    for var_name in long_var_names
]
ygreek = collections.defaultdict(lambda: r"-")
ygreek.update(
    {
        "R": r"$R$",
        "theta": r"$\theta$",
        "rh": r"$\phi$",
        "Qsw": r"$Q_{sw}$",
        "sun": r"sun",
        "ILWR": r"$Q_{lw(inc.)}$",
        "u": r"$u$",
        "v": r"$v$",
        "pressure": r"$atm$",
        "U": "U",
    }
)


def var_names_greek(var_names):
    return [ygreek[var_name] for var_name in var_names]


# -----------------------------------------------------------------------------#
# Here you can define conversions done after the simulation and before
# output of the data.
# The functions should accept a datetime array an array with the data and
# an iterable containing the variable names.
# -----------------------------------------------------------------------------#

conversions = []


# # for example: calculate long-wave radiation from air temperature
# def theta2ilwr(times_, data, var_names):
#     from scipy import constants
#     theta = data[var_names.index("theta")]
#     theta_kelvin = constants.C2K(theta)
#     e = 6.2 * np.exp(17.26 * theta / (theta_kelvin - 35.8))
#     boltz = 0.0000000567  # ask gideon
#     alpha, beta = .42, .065
#     ilwr = boltz * theta_kelvin ** 4 * (alpha + beta * np.sqrt(e))
#     if "ILWR" not in var_names:
#         data = np.vstack((data, ilwr))
#         var_names = tuple(list(var_names) + ["ILWR"])
#     else:
#         data[var_names.index("ILWR")] = ilwr
#     return times_, data, var_names
#
# conversions += [theta2ilwr]


# -----------------------------------------------------------------------------#
# Define the format the output should have
# -----------------------------------------------------------------------------#

# if not specified differently, output will have 3 decimal places
out_format = collections.defaultdict(lambda: "%.3f")
out_format.update(
    {
        "theta": "%.3f",
        "Qsw": "%.3f",
        "ILWR": "%.3f",
        "rh": "%.3f",
        "U": "%.3f",
        "wdir": "%.1f",
        "u": "%.3f",
        "v": "%.3f",
        "R": "%.6f",
    }
)
