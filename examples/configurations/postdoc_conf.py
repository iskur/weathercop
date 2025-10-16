"""Central file to keep the same configuration in all phd-related
code.

This has become an exercise in dont-repeat-yourself meta-programming.
"""
from collections import OrderedDict
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib as mpl
from weathercop.vine import vg_ph
from weathercop.tools import ADict

# from lhglib.contrib.dirks_globals import ADict

# plotting related
footnotesize = 8
scriptsize = 7
mpl.rcParams.update(
    {
        "text.usetex": False,
        "font.size": footnotesize,
        "figure.titlesize": footnotesize,
        "axes.titlesize": footnotesize,
        "axes.labelsize": scriptsize,
        "xtick.labelsize": scriptsize,
        "ytick.labelsize": scriptsize,
        "legend.fontsize": scriptsize,
        "lines.linewidth": 1,
    }
)


def gen_fig_width(width_in_inches):
    inches_per_pt = 1.0 / 72.27
    return width_in_inches * inches_per_pt


fig_width = gen_fig_width(441.01772)
fig_height_golden = fig_width / np.sqrt(2)

figsize = fig_width * 2 / 3, fig_height_golden * 2 / 3
figsize_wide = fig_width, fig_height_golden * 2 / 3
figsize_half = fig_width / 2, fig_height_golden / 2
figsize_full = fig_width, 10

s_kwds = dict(s=12, zorder=10)

root = Path("/home/dirk/Diss/thesis")
data_root = root / "data"
# data_root = Path("../../data/b_data_generation/d_scenario_data_500_real")
# data_root = Path("../data/b_data_generation/d_scenario_data_500_real")
models = "vg_seasonal", "vg_seasonal_fulldis", "resampler", "weathercop"
scenarios = "st", "ho", "sp", "hs"
scenario_labels = OrderedDict(
    (
        ("st", "Unchanged"),
        ("ho", "Incr. mean"),
        ("sp", "Incr. variability"),
        ("hs", "Incr. mean and var."),
    )
)
model_labels = OrderedDict(
    (("vgdis", "VG"), ("res", "KNN"), ("cop", "WeatherCop"))
)
neg_rain_doy_width = 30
neg_rain_fft_order = 2
svar_doy_width = 15
svar_fft_order = 2


# Multi-Site
ms_root = Path.home() / "data/PostDoc/"


class VGG:
    var_names = "theta", "R", "Qsw", "ILWR", "rh", "u", "v"
    init = ADict(
        var_names=var_names,
        verbose=True,
        dump_data=False,
        neg_rain_doy_width=neg_rain_doy_width,
        neg_rain_fft_order=neg_rain_fft_order,
    )
    # fit = ADict(p=3, seasonal=True, doy_width=30, fft_order=3)
    # fit = ADict(p=3, seasonal=True, doy_width=15, fft_order=3)
    fit = ADict(
        p=3, seasonal=True, doy_width=svar_doy_width, fft_order=svar_fft_order
    )
    # fit = ADict(p=3, seasonal=False)
    sim = ADict(phase_randomize=True)
    sim_nonphr = sim + dict(phase_randomize=False)
    dis = ADict(var_names_dis=("theta", "ILWR", "Qsw", "rh", "u", "v"))

    theta_incr = 4.0
    mean_arrival = 7.0
    disturbance_std = 5.0
    n_realizations = 500
    # n_realizations = 20

    cal_cache_dir = data_root / "BodenseeKlima" / "cache_cal"

    @property
    def stale_sim(self):
        return self.sim + dict(theta_incr=0.0)

    @property
    def hot_sim(self):
        return self.sim + dict(theta_incr=VGG.theta_incr)

    @property
    def spicy_sim(self):
        return self.sim + dict(
            mean_arrival=VGG.mean_arrival, disturbance_std=VGG.disturbance_std
        )

    @property
    def hotspicy_sim(self):
        return self.hot_sim + self.spicy_sim

    def __getattr__(self, name):
        if "filepath_" in name:
            name_orig, ext = name.split("filepath_")
            val_orig = getattr(self, name_orig + "filepath")
            return val_orig.with_name(
                f"{val_orig.stem}_{ext}{val_orig.suffix}"
            )
        raise AttributeError


vgg = VGG()


class MS:
    varnames = ("theta", "R", "rh", "U")
    nc_filepath = ms_root / "multisite_data.nc"
    nc_clean_filepath = ms_root / "multisite_data_clean.nc"
    nc_out_filepaths = {
        scen: ms_root / f"multisite_{scen}_simulation.nc" for scen in scenarios
    }
    n_realizations = 500
    # n_realizations = 10
    init = ADict(
        neg_rain_doy_width=neg_rain_doy_width,
        neg_rain_fft_order=neg_rain_fft_order,
        scop_kwds=dict(window_len=30, fft_order=3),
    )
    sim = dict(
        st=vgg.stale_sim,
        ho=vgg.hot_sim,
        sp=vgg.spicy_sim,
        hs=vgg.hotspicy_sim,
    )


class MSTest(MS):
    station_kwds = ADict(
        lon_min=8.6,
        lat_min=47.4,
        lon_max=10.0,
        lat_max=48.0,
    )
    nc_filepath_test = ms_root / "multisite_testdata.nc"


class Rich(MS):
    nc_out_filepath = ms_root / "richardson_simulation.nc"
