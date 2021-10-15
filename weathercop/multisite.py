from collections import Iterable, OrderedDict
import shutil
import inspect
from functools import wraps, partial
from itertools import repeat
import collections

# import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# from scipy.optimize import minimize_scalar
from scipy import stats, interpolate
import xarray as xr
import dill
from multiprocessing import Pool, cpu_count, Lock
from tqdm import tqdm, trange
from matplotlib.transforms import offset_copy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

import vg
from vg import helpers as my
from vg.time_series_analysis import (
    distributions as dists,
    time_series as ts,
    rain_stats,
)
from weathercop import cop_conf, plotting, tools, copulae as cops
from weathercop.vine import CVine

n_nodes = cpu_count() - 1
lock = Lock()


def set_conf(conf_obj, **kwds):
    objs = (vg, vg.vg_core, vg.vg_base, vg.vg_plotting)
    for obj in objs:
        obj.conf = conf_obj
        for key, value in kwds.items():
            setattr(obj.conf, key, value)


def suptitle_prepend(fig, name):
    if isinstance(fig, mpl.figure.Figure):
        try:
            suptitle = fig._suptitle.get_text()
        except AttributeError:
            suptitle = ""
    fig.suptitle(f"{name} {suptitle}")


def nan_corrcoef(data):
    ndim = data.shape[0]
    obs_corr = np.full(2 * (ndim,), np.nan)
    overlap = np.zeros_like(obs_corr)
    for row_i in range(ndim):
        vals1 = data[row_i]
        mask1 = np.isfinite(vals1)
        for col_i in range(row_i, ndim):
            vals2 = data[col_i]
            mask2 = np.isfinite(vals2)
            mask = mask1 & mask2
            overlap[row_i, col_i] = overlap[col_i, row_i] = np.mean(mask)
            if np.sum(mask) > 30:
                obs_corr[row_i, col_i] = np.corrcoef(vals1[mask], vals2[mask])[
                    0, 1
                ]
    nan_corrcoef.overlap = overlap
    return obs_corr


def sim_one(args):
    (
        real_i,
        total,
        wcop,
        filepath,
        sim_args,
        sim_kwds,
        dis_kwds,
        sim_times,
        csv,
        ensemble_dir,
        n_digits,
    ) = args
    np.random.seed(1000 * real_i)
    sim_sea, sim_trans = simulate(wcop, sim_times, *sim_args, **sim_kwds)
    if dis_kwds is not None:
        sim_sea_dis = wcop.disaggregate(**dis_kwds)
        sim_sea, sim_sea_dis = xr.align(sim_sea, sim_sea_dis, join="outer")
        sim_sea.loc[dict(variable=wcop.varnames)] = sim_sea_dis.sel(
            variable=wcop.varnames
        )
    sim_sea.to_netcdf(filepath)
    if csv:
        real_str = f"real_{real_i:0{n_digits}}"
        csv_path = ensemble_dir / "csv" / real_str
        wcop.to_csv(csv_path, sim_sea, filename_prefix=f"{real_str}_")
    # sim_trans.to_netcdf(filepath_trans)
    return real_i


def simulate(wcop, sim_times, *args, phase_randomize_vary_mean=True, **kwds):
    """this is like Multisite.simulate, but simplified for multiprocessing."""
    # allow for site-specific theta_incr
    theta_incrs = kwds.pop("theta_incr", None)
    sim_sea = sim_trans = None
    T_sim = len(sim_times)

    lock.acquire()
    vgs = wcop.vgs
    vg_obj = list(vgs.values())[0]
    T_data = vg_obj.T_summed
    K = wcop.K
    station_names = wcop.station_names
    n_stations = len(station_names)
    varnames = wcop.varnames
    phases = wcop.phases
    primary_var = wcop.primary_var
    lock.release()

    T_total = 0
    phases_stacked = []
    while T_total < T_sim:
        # phase randomization with same random phases in all
        # variables and stations
        phases_len = T_data // 2 - 1 + T_data % 2
        phases_pos = np.random.uniform(0, 2 * np.pi, phases_len)
        phases_pos = np.array(K * [phases_pos])
        phases_neg = -phases_pos[:, ::-1]
        nyquist = np.full(K, 0)[:, None]
        zero_phases = phases[:, 0, None]
        if T_data % 2 == 0:
            phases = np.hstack((zero_phases, phases_pos, nyquist, phases_neg))
        else:
            phases = np.hstack((zero_phases, phases_pos, phases_neg))
        phases_stacked += [phases]
        T_total += T_data

    vg_ph = partial(_vg_ph, rphases=phases_stacked, wcop=wcop)
    for station_name, svg in vgs.items():
        try:
            theta_incr = theta_incrs[station_name]
        except (KeyError, TypeError, IndexError):
            theta_incr = theta_incrs
        _, sim = svg.simulate(
            sim_func=vg_ph,
            primary_var=primary_var,
            theta_incr=theta_incr,
            *args,
            **kwds,
        )
        if sim_sea is None:
            sim_sea = xr.DataArray(
                np.empty((n_stations, K, T_sim)),
                coords=[station_names, varnames, sim_times],
                dims=["station", "variable", "time"],
            )
            sim_trans = xr.full_like(sim_sea, np.nan)
        sim_sea.loc[dict(station=station_name)] = sim
        sim_trans.loc[dict(station=station_name)] = svg.sim
    return sim_sea, sim_trans


def _vg_ph(
    vg_obj, sc_pars, wcop=None, rphases=None, phase_randomize_vary_mean=True
):
    """Call-back function for VG.simulate. Replaces a time_series_analysis
    model.

    Simlified version of Multisite._vg_ph for multiprocessing

    """
    lock.acquire()
    station_name = vg_obj.station_name
    var_names = vg_obj.var_names
    primary_var_ii = vg_obj.primary_var_ii
    T = vg_obj.T
    As = wcop.As.sel(station=station_name, drop=True)
    qq_std = wcop.qq_stds.sel(station=station_name).data
    qq_mean = wcop.qq_means.sel(station=station_name).data
    zero_phases = wcop.zero_phases
    vine = wcop.vine
    lock.release()

    # adjust zero-phase
    phases = []
    for phase_ in rphases:
        phase_[:, 0] = zero_phases[station_name]
        phases += [phase_]

    fft_sim = np.concatenate(
        [np.fft.ifft(As * np.exp(1j * phases_)).real for phases_ in phases],
        axis=1,
    )[:, :T]

    fft_sim *= (qq_std / fft_sim.std(axis=1))[:, None]
    fft_sim += (qq_mean - fft_sim.mean(axis=1))[:, None]
    if phase_randomize_vary_mean:
        # allow means to vary. let it flow from the central_node to
        # the others
        mean_eps = np.zeros(len(var_names))[:, None]
        mean_eps[0, primary_var_ii[0]] = 0.25 * np.random.randn()
        fft_sim += mean_eps

    # change in mean scenario
    prim_i = tuple(primary_var_ii)
    fft_sim[prim_i] += sc_pars.m[prim_i]
    fft_sim[prim_i] += sc_pars.m_t[prim_i]

    qq = dists.norm.cdf(fft_sim)
    ranks_sim = vine.simulate(T=np.arange(qq.shape[1]), randomness=qq)
    sim = dists.norm.ppf(ranks_sim)
    assert np.all(np.isfinite(sim))
    return sim


class ECDF:
    def __init__(self, data, data_min=None, data_max=None):
        self.data = data
        fin_mask = np.isfinite(data)
        data_fin = data[fin_mask]
        if data_min is None:
            data_min = data_fin.min()
        if data_max is None:
            data_max = data_fin.max()
        sort_ii = np.argsort(data_fin)
        self.ranks_rel = np.full(len(data), np.nan)
        self.ranks_rel[fin_mask] = (
            stats.rankdata(data_fin, "min") - 0.5
        ) / len(data_fin)
        self._data_sort_pad = np.concatenate(
            ([data_min], data_fin[sort_ii], [data_max])
        )
        self._ranks_sort_pad = np.concatenate(
            ([0], self.ranks_rel[fin_mask][sort_ii], [1])
        )
        self._cdf = interpolate.interp1d(
            self._data_sort_pad,
            self._ranks_sort_pad,
            bounds_error=False,
            fill_value=(0, 1),
        )
        self._ppf = interpolate.interp1d(
            self._ranks_sort_pad,
            self._data_sort_pad,
            bounds_error=False,
            fill_value=(data_min, data_max),
        )

    def cdf(self, x=None):
        if x is None:
            return self.ranks_rel
        return np.where(np.isfinite(x), self._cdf(x), np.nan)

    def ppf(self, p=None):
        if p is None:
            return self.data
        return np.where(np.isfinite(p), self._ppf(p), np.nan)

    def plot_cdf(self, fig=None, ax=None, *args, **kwds):
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        ax.plot(self._data_sort_pad, self._ranks_sort_pad, *args, **kwds)
        return fig, ax


class Multisite:
    def __init__(
        self,
        xds,
        *args,
        primary_var="theta",
        discretization="D",
        verbose=False,
        phase_inflation=False,
        refit_vine=False,
        asymmetry=False,
        rain_method="regression",
        cop_candidates=None,
        debias=True,
        scop_kwds=None,
        **kwds,
    ):
        """Multisite weather generation using vines and phase randomization.

        Parameters
        ----------
        xds : xr.Dataset
            Should contain the coordinates "time", "variable" and
            "station". Hourly discretization
        primary_var : str, one of var_names from __init__ or sequence of those,
                optional
            All disturbances (mean_arrival, distrubance_std, theta_incr,
            theta_grad and climate_signal) correspond to changes in this
            variable.
        phase_inflation : boolean
            Do phase inflation before fitting the vine. This might
            possibly alter the dependency structure, but fills in
            missing values.
        """
        self.varnames = list(xds.coords["variable"].data)
        # if "R" is included, put it second for better fitting!
        # if "R" in self.varnames:
        #     varnames_rest = [name for name in self.varnames
        #                      if name not in ("theta", "R")]
        #     self.varnames = ["theta", "R"] + varnames_rest
        #     xds = xds.sel(variable=self.varnames)
        self.station_names = list(xds.coords["station"].data)
        # self.station_names = sorted(list(self.xds.data_vars.keys()
        #                                  - {"longitude", "latitude"}))
        self.xds = xds
        try:
            self.longitudes = xds["longitude"]
            self.latitudes = xds["latitude"]
        except KeyError:
            print("No latitudes/longitudes in xds.")
            self.longitudes = self.latitudes = None
        if self.longitudes is not None and self.latitudes is not None:
            assert np.all(np.isfinite(self.longitudes))
            assert np.all(np.isfinite(self.latitudes))
            xds = xds.drop_vars(("latitude", "longitude"))
        self.xar = xds.to_array("station").transpose(
            "station", "variable", "time"
        )
        self.primary_var = primary_var
        self.discr = discretization
        self.verbose = verbose
        self.phase_inflation = phase_inflation
        self.asymmetry = asymmetry
        self.rain_method = rain_method
        if cop_candidates is None:
            cop_candidates = cops.all_cops
        self.cop_candidates = cop_candidates
        self.scop_kwds = scop_kwds
        # for pickling ease
        self.args, self.kwds = args, kwds

        self.refit_vine = refit_vine
        self.vgs = OrderedDict()
        self.K = self.n_variables = len(self.varnames)
        if self.discr == "D":
            self.sum_interval = 24
        elif self.discr == "H":
            self.sum_interval = 1
        else:
            print(f"Discretization '{self.discr}' not supported")
            return
        self.n_stations = len(self.station_names)
        self.times = self.xar.time
        self.varnames_refit = None
        start_dt, end_dt = pd.to_datetime(self.xar.time.data[[0, -1]])
        self.dtimes = pd.date_range(
            start_dt, end_dt, freq=self.discr
        ).to_pydatetime()

        if "refit" in kwds and kwds["refit"]:
            self.refit = True
            # this means we also need a new vine!
            # vine.clear_vine_cache()
            refits = kwds["refit"]
            if refits is not True:
                if isinstance(refits, str):
                    refits = kwds["refit"] = [refits]
                station_names_refit = [
                    name for name in refits if name in self.station_names
                ]
                self.varnames_refit = [
                    name for name in refits if name in self.varnames
                ]
            else:
                station_names_refit = None
            self._clear_vgs_cache(station_names_refit)
        else:
            self.refit = False
        self._init_vgs(*args, **kwds)

        # this is useful from time to time
        self.data_daily = self.xar.resample(time=self.discr).mean(
            dim="time",
            # skipna=True
        )

        nans = self.data_daily.isnull()
        if np.any(nans) and not phase_inflation:
            print("Nan sums:\n", nans.sum("time").to_pandas())
            raise ValueError("nans in xds. Remove or use phase_inflation.")

        # household variables for vines and phase randomization
        self.cop_quantiles = xr.full_like(self.data_trans, np.nan)
        self.fft_sim = None  # xr.full_like(self.data_trans, np.nan)
        # we will expand in frequency space in order to have variation
        # in mean
        self.As = xr.full_like(self.data_trans, np.nan)
        self.rphases = self.phases = self.vine = None

        # property caches
        self._obs_means = None

        # filled during simulation
        self.sim = None  # xr.full_like(self.data_trans, np.nan)
        self.ranks_sim = None  # xr.full_like(self.sim, np.nan)
        self.sim_sea = self.ensemble = None

    def __getstate__(self):
        dict_ = dict(self.__dict__)
        if "vgs" in dict_:
            del dict_["vgs"]
        return dict_

    def __setstate__(self, dict_):
        self.__dict__ = dict_
        self.vgs = OrderedDict()
        self._init_vgs(*self.args, **self.kwds)

    @property
    def obs_means(self):
        if self._obs_means is None:
            self._obs_means = self.data_daily.mean("time")
        return self._obs_means

    @property
    def station_names_short(self):
        names = {}
        for name in self.station_names:
            if "," in name:
                name_short = name.split(",")[0]
            elif "-" in name:
                name_short = name.split("-")[0]
            else:
                name_short = name
            names[name] = name_short
        return names

    def __getattr__(self, name):
        if name.startswith("plot_") and name not in dir(self):
            # call the plotting method on all vgs
            def meta_plot(*args, **kwds):
                returns = {}
                for station_name, svg in self.vgs.items():
                    longitude = float(
                        self.longitudes.sel(station=station_name)
                    )
                    latitude = float(self.latitudes.sel(station=station_name))
                    svg._conf_update(
                        dict(longitude=longitude, latitude=latitude)
                    )
                    fig, axs = getattr(svg, name)(*args, **kwds)
                    for fig_ in np.atleast_1d(fig):
                        suptitle_prepend(fig_, station_name)
                    returns[station_name] = fig, axs
                return returns

            return meta_plot
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has " f"no attribute '{name}'"
            )

    def __getitem__(self, name):
        if isinstance(name, int):
            return self.vgs[self.station_names[name]]
        try:
            return self.vgs[name]
        except KeyError:
            raise KeyError(f"Station {name} unknown.")

    def keys(self):
        return self.vgs.keys()

    def values(self):
        return self.vgs.values()

    def items(self):
        return self.vgs.items()

    @property
    def start_str(self):
        return self.dtimes[0].strftime("%Y-%m-%d")

    @property
    def end_str(self):
        return self.dtimes[-1].strftime("%Y-%m-%d")

    def _init_vgs(self, *args, **kwds):
        """Initiate VG instances to get transformed data."""
        # gather transformed data from all vg instances
        data_trans = (
            xr.full_like(self.xar, 0)
            .resample(time=self.discr)
            .mean(dim="time")
        )
        self.ranks = xr.full_like(data_trans, np.nan)
        if "R" in self.varnames:
            self.rain_mask = xr.full_like(
                self.ranks.sel(variable="R"), False, dtype=bool
            )
        refit = "refit" in kwds
        if refit and isinstance(kwds["refit"], collections.Iterable):
            # refit_vars = (set(kwds.get("refit", []))
            #               & set(self.varnames))
            refit_vars = self.varnames_refit
            refit_stations = set(kwds.get("refit", [])) & set(
                self.station_names
            )
        else:
            refit_vars = refit_stations = []
        for station_name in self.station_names:
            cache_file = self.cache_file(station_name)
            (cop_conf.vgs_cache_dir / station_name).mkdir(
                exist_ok=True, parents=True
            )
            seasonal_cache_file = cache_file.parent / cache_file.stem
            if self.longitudes is not None:
                longitude = float(self.longitudes.sel(station=station_name))
            else:
                longitude = None
            if self.latitudes is not None:
                latitude = float(self.latitudes.sel(station=station_name))
            else:
                latitude = None
            if (
                not refit_vars
                and station_name not in refit_stations
                and cache_file.exists()
            ):
                if self.verbose:
                    print(
                        f"Recovering VG instance of {station_name} from:\n{cache_file}"
                    )
                with cache_file.open("rb") as fi:
                    svg = dill.load(fi)
                    svg._conf_update(
                        dict(
                            longitude=longitude,
                            latitude=latitude,
                            seasonal_cache_file=seasonal_cache_file,
                        )
                    )
            else:
                if self.verbose:
                    print(
                        f"Fitting a VG instance on {station_name} "
                        f"saving to:\n{cache_file}"
                    )
                data = (
                    self.xar.sel(station=station_name, drop=True)
                    .to_dataset(dim="variable")
                    .to_dataframe()
                )
                cache_dir = str(cop_conf.vgs_cache_dir / station_name)
                kwds_ = {
                    key: value for key, value in kwds.items() if key != "refit"
                }
                if refit:
                    if kwds["refit"] is True:
                        kwds_["refit"] = True
                    elif station_name in refit_stations:
                        kwds_["refit"] = True
                    elif refit_vars:
                        kwds_["refit"] = list(refit_vars)
                conf_update = dict(
                    longitude=longitude,
                    latitude=latitude,
                    seasonal_cache_file=seasonal_cache_file,
                )
                svg = vg.VG(
                    self.varnames,
                    met_file=data,
                    cache_dir=cache_dir,
                    # we do not need it and don't want it to
                    # be read from the configuration file
                    data_dir="",
                    conf_update=conf_update,
                    dump_data=False,
                    sum_interval=self.sum_interval,
                    station_name=station_name,
                    rain_method=self.rain_method,
                    *args,
                    **kwds_,
                )
                with cache_file.open("wb") as fi:
                    dill.dump(svg, fi)
            self.vgs[station_name] = svg
            if "R" in self.varnames:
                self.rain_mask.loc[dict(station=station_name)] = svg.rain_mask
            for var_i, varname in enumerate(self.varnames):
                var_trans = svg.data_trans[var_i]
                data_trans.loc[
                    dict(station=station_name, variable=varname)
                ] = var_trans
        # this sets verbosity on all vg instances
        self.verbose = self.verbose
        if self.phase_inflation:
            if self.verbose:
                print("Phase inflation")
            data_trans = self._phase_inflation(data_trans)
        self.data_trans = data_trans.interpolate_na("time")
        self.ranks.values = dists.norm.cdf(self.data_trans)
        assert np.all(np.isfinite(self.ranks.values))
        # reorganize so that variable dependence does not consider
        # inter-site relationships
        self.ranks = self.ranks.stack(rank=("station", "time"))
        if "R" in self.varnames:
            self.rain_mask = self.rain_mask.stack(rank=("station", "time"))

    def cache_file(self, station_name):
        return (
            cop_conf.vgs_cache_dir
            / station_name
            / f"{station_name}_{self.discr}_"
            f"{self.rain_method}_"
            f"{self.start_str}-{self.end_str}.pkl"
        )

    def _clear_vgs_cache(self, station_names=None):
        if station_names is None:
            station_names = self.station_names
        for station_name in station_names:
            cache_file = self.cache_file(station_name)
            if cache_file.exists():
                cache_file.unlink()

    def _phase_inflation(self, data_ar):
        np.random.seed(0)
        if self.verbose:
            data = data_ar.values.copy()
        else:
            data = data_ar.values
        nan_mask = np.isnan(data)
        data[nan_mask] = 0
        T = data.shape[2]
        # n_missing = nan_mask.sum(axis=2)
        # data *= np.sqrt((T - 1) / (n_missing - 1))[..., None]
        std_obs = np.std(data[~nan_mask])[..., None]
        As = np.fft.rfft(data, n=T)
        # phases = np.random.uniform(0, 2 * np.pi, As.shape[2])

        phases = np.zeros(As.shape[2])
        As_stacked = np.abs(np.vstack(As))
        nan_mask_stacked = np.vstack(nan_mask)
        missing_mean = np.mean(nan_mask_stacked, axis=1)
        fullest_var_i = np.argmax(np.mean(nan_mask_stacked, axis=1))
        # find frequencies which are weak in the fullest and strong in
        # the emptiest variables
        n_missing_max = max(nan_mask_stacked.sum(axis=1))
        ii = np.argsort(
            np.sum(As_stacked * (1 - missing_mean[:, None]), axis=0)
            - As_stacked[fullest_var_i]
        )[: n_missing_max // 2]
        # ii = np.argsort(np.sum(As_stacked * (1 - missing_mean[:, None]),
        #                        axis=0)
        #                 - As_stacked[fullest_var_i])
        phases[ii] = np.random.uniform(0, np.pi, len(ii))

        # def opt_func(phase, phases, i):
        #     phases[i] = phase
        #     As_new = As * np.exp(1j * phases)
        #     data_new = np.fft.irfft(As_new, n=T)
        #     rsme = np.sqrt(np.nanmean((data[~nan_mask] -
        #                                data_new[~nan_mask]) ** 2))
        #     # standard deviation in data gap regions
        #     std = np.nanstd(data_new[nan_mask])
        #     return rsme + (1 - std) ** 2

        # for i in tqdm(ii):
        #     phases[i] = minimize_scalar(opt_func,
        #                                 bounds=(0, 2 * np.pi),
        #                                 args=(phases, i),
        #                                 ).x

        # if self.discr == "H":
        #     # if we are hourly discretized, do not change the phases
        #     # of the near-daily frequencies
        #     freq = np.fft.rfftfreq(T)
        #     daily_i = np.where(np.isclose(freq, 1 / 24))[0][0]
        #     freq_slice = slice(daily_i - 4, daily_i + 4)
        #     phases = np.broadcast_to(phases, As.shape).copy()
        #     phases[..., freq_slice] = np.angle(As)[..., freq_slice]

        As_new = As * np.exp(1j * phases)
        data = np.fft.irfft(As_new, n=T)
        data *= std_obs / np.nanstd(data, axis=2)[..., None]

        # if self.verbose:
        #     prop_cycle = plt.rcParams['axes.prop_cycle']
        #     edgecolor = prop_cycle.by_key()['color']
        #     inflated = xr.full_like(data_ar, np.nan)
        #     inflated.data = data
        #     fig, axs = self._plot_corr_scatter_var(data_ar,
        #                                            inflated,
        #                                            "inflated",
        #                                            facecolor=edgecolor[1],
        #                                            s=100)

        #     phases_full = np.random.uniform(0, 2 * np.pi, As.shape[2])
        #     As_full = As * np.exp(1j * phases_full)
        #     data_full = np.fft.irfft(As_full, n=T)
        #     data_full *= std_obs / np.nanstd(data_full, axis=2)[..., None]
        #     inflated_full = xr.full_like(data_ar, np.nan)
        #     inflated_full.data = data_full
        #     self._plot_corr_scatter_var(data_ar, inflated_full,
        #                                 "inflated", fig=fig, axs=axs,
        #                                 facecolor=edgecolor[2])

        #     # phases_rand = phases.copy()
        #     # phases_rand[ii] = np.random.uniform(0, np.pi, len(ii))
        #     # As_rand = As * np.exp(1j * phases_rand)
        #     # data_rand = np.fft.irfft(As_rand, n=T)
        #     # data_rand *= std_obs / np.nanstd(data_rand, axis=2)[..., None]
        #     # inflated_rand = xr.full_like(data_ar, np.nan)
        #     # inflated_rand.data = data_rand
        #     # self._plot_corr_scatter_var(data_ar,
        #     #                             inflated_rand,
        #     #                             "inflated", fig=fig, axs=axs,
        #     #                             facecolor=edgecolor[2])

        #     fig, axs = plt.subplots(nrows=self.n_stations,
        #                             ncols=self.n_variables,
        #                             sharex=True)
        #     axs = np.ravel(axs)
        #     for var_i, ax in enumerate(axs):
        #         time = data_ar.time
        #         obs = np.vstack(data_ar.values)[var_i]
        #         sim1 = np.vstack(inflated.values)[var_i]
        #         sim2 = np.vstack(inflated_full.values)[var_i]
        #         # sim2 = np.vstack(inflated_rand.values)[var_i]
        #         ax.plot(time, obs, label="observed", alpha=.5)
        #         ax.plot(time, sim1, label="light", alpha=.5)
        #         ax.plot(time, sim2, label="full", alpha=.5)
        #         # ax.plot(time, sim2, label="rand", alpha=.5)
        #     axs[0].draw_legend()

        #     self._plot_corr_scatter_stat(data_ar, inflated, "inflated")
        data_ar.data = data
        return data_ar

    @property
    def verbose(self):
        if not hasattr(self, "_verbose"):
            self._verbose = False
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value
        if hasattr(self, "vgs"):
            for svg in self.vgs.values():
                svg.verbose = value

    def simulate(
        self,
        *args,
        usevine=True,
        phase_randomize_vary_mean=True,
        return_trans=False,
        **kwds,
    ):
        self.usevine = usevine
        self.phase_randomize_vary_mean = phase_randomize_vary_mean
        # allow for site-specific theta_incr
        theta_incrs = kwds.pop("theta_incr", None)
        # self.sim_sea = xr.full_like(self.cop_quantiles, np.nan)
        # self.sim_trans = xr.full_like(self.sim_sea, np.nan)
        self.sim_sea = self.sim_trans = None
        # site-spe
        for station_name, svg in self.vgs.items():
            if self.verbose:
                print(f"Simulating {station_name}")
                svg.verbose = False
            try:
                theta_incr = theta_incrs[station_name]
            except (KeyError, TypeError, IndexError):
                theta_incr = theta_incrs
            sim_times, sim = svg.simulate(
                sim_func=self._vg_ph,
                primary_var=self.primary_var,
                theta_incr=theta_incr,
                *args,
                **kwds,
            )
            if self.sim_sea is None:
                self.sim_sea = xr.DataArray(
                    np.empty((self.n_stations, self.K, svg.T)),
                    coords=[self.station_names, self.varnames, sim_times],
                    dims=["station", "variable", "time"],
                )
                self.sim_trans = xr.full_like(self.sim_sea, np.nan)
            self.sim_sea.loc[dict(station=station_name)] = sim
            self.sim_trans.loc[dict(station=station_name)] = svg.sim
            if self.verbose:
                self.print_means(station_name)
                print()
        self.T = svg.T
        # make chosen phases available...
        self._rphases = self.rphases
        # ... but reset them also so we get new ones next time
        self.rphases = None
        if self.verbose:
            self.print_all_means()
        if return_trans:
            return self.sim_sea, self.sim_trans
        return self.sim_sea

    def disaggregate(self, *args, **kwds):
        self.sim_sea_dis = None
        for station_name, svg in self.vgs.items():
            if self.verbose:
                print(f"Disaggregating {station_name}")
            svg.verbose = False
            times_dis, sim_dis = svg.disaggregate(*args, **kwds)
            if self.sim_sea_dis is None:
                self.sim_sea_dis = xr.DataArray(
                    np.empty((self.n_stations, self.K, sim_dis.shape[1])),
                    coords=[self.station_names, self.varnames, times_dis],
                    dims=["station", "variable", "time"],
                )
            self.sim_sea_dis.loc[dict(station=station_name)] = sim_dis
        self.varnames_dis = self[0].var_names_dis
        return self.sim_sea_dis

    def simulate_ensemble(
        self,
        n_realizations,
        name="calibration",
        clear_cache=False,
        csv=False,
        dis_kwds=None,
        *args,
        **kwds,
    ):
        if dis_kwds is not None:
            name = f"{name}_disag"
        ensemble_dir = cop_conf.weathercop_dir / "ensembles" / name
        if clear_cache and ensemble_dir.exists():
            shutil.rmtree(ensemble_dir)
        ensemble_dir.mkdir(exist_ok=True)
        if self.verbose:
            print(f"Starting Ensemble Simulations for {name}")
            print(f"Saving nc-files here:\n{ensemble_dir}")
        # silence the vgs for the moment, so we can get a nice
        # progress bar
        verbose = self.verbose
        self.verbose = False
        n_digits = int(np.log10(n_realizations)) + 3
        filepaths = [
            ensemble_dir / f"real_{real_i:0{n_digits}}.nc"
            for real_i in range(n_realizations)
        ]
        # filepaths_trans = [filepath.parent / (filepath.stem + "_trans.nc")
        #                    for filepath in filepaths]
        if cop_conf.PROFILE:
            for real_i in tqdm(range(n_realizations)):
                filepath = filepaths[real_i]
                if filepath.exists():
                    continue
                # filepath_trans = filepaths_trans[real_i]
                np.random.seed(1000 * real_i)
                sim_sea = self.simulate(*args, **kwds)
                # sim_trans = self.sim_trans
                if dis_kwds is not None:
                    sim_sea_dis = self.disaggregate(**dis_kwds)
                    sim_sea, sim_sea_dis = xr.align(
                        sim_sea, sim_sea_dis, join="outer"
                    )
                    sim_sea.loc[
                        dict(variable=self.varnames)
                    ] = sim_sea_dis.sel(variable=self.varnames)
                sim_sea.to_netcdf(filepath)
                if csv:
                    real_str = f"real_{real_i:0{n_digits}}"
                    csv_path = ensemble_dir / "csv" / real_str
                    self.to_csv(
                        csv_path, sim_sea, filename_prefix=f"{real_str}_"
                    )
                # sim_trans.to_netcdf(filepath_trans)
        else:
            # this means we do parallel computation
            # do one simulation in the main loop to set up attributes
            np.random.seed(0)
            sim_sea = self.simulate(*args, **kwds)
            sim_times = sim_sea.coords["time"].data
            # sim_trans = self.sim_trans
            if dis_kwds is not None:
                sim_sea_dis = self.disaggregate(**dis_kwds)
                sim_sea, sim_sea_dis = xr.align(
                    sim_sea, sim_sea_dis, join="outer"
                )
                sim_sea.loc[dict(variable=self.varnames)] = sim_sea_dis.sel(
                    variable=self.varnames
                )
            sim_sea.to_netcdf(filepaths[0])
            if csv:
                real_i = 0
                real_str = f"real_{real_i:0{n_digits}}"
                csv_path = ensemble_dir / "csv" / real_str
                self.to_csv(csv_path, sim_sea, filename_prefix=f"{real_str}_")
            # sim_trans.to_netcdf(filepaths_trans[0])
            # filter realizations in advance according to output file
            # existance
            realizations = []
            filepaths_multi = []
            # filepaths_trans_multi = []
            for real_i in range(1, n_realizations):
                # real_missing = not (filepaths[real_i].exists() and
                #                   filepaths_trans[real_i].exists())
                real_missing = not filepaths[real_i].exists()
                if real_missing:
                    realizations += [real_i]
                    filepaths_multi += [filepaths[real_i]]
                    # filepaths_trans_multi += [filepaths_trans[real_i]]
            with Pool(n_nodes) as pool:
                completed_reals = list(
                    tqdm(
                        pool.imap(
                            sim_one,
                            zip(
                                realizations,
                                repeat(len(realizations)),
                                repeat(self),
                                filepaths_multi,
                                # filepaths_trans_multi,
                                repeat(args),
                                repeat(kwds),
                                repeat(dis_kwds),
                                repeat(sim_times),
                                repeat(csv),
                                repeat(ensemble_dir),
                                repeat(n_digits),
                            ),
                        ),
                        total=len(realizations),
                    )
                )
            assert len(completed_reals) == len(realizations)
        # expose the ensemble as a dask array
        drop_dummy = self.n_stations > 1
        self.ensemble = (
            xr.open_mfdataset(
                filepaths, concat_dim="realization", combine="nested"
            )
            .assign_coords(realization=range(n_realizations))
            .to_array("dummy")
            .squeeze("dummy", drop=drop_dummy)
            # .squeeze(drop=drop_dummy)
        )
        # self.ensemble_trans = (xr
        #                        .open_mfdataset(filepaths_trans,
        #                                        concat_dim="realization",
        #                                        combine="nested")
        #                        .assign_coords(realization=range(n_realizations))
        #                        .to_array("dummy")
        #                        .squeeze("dummy", drop=drop_dummy)
        #                        # .squeeze(drop=drop_dummy)
        #                        )

        # # make the first realization available for plotting methods
        # self.sim_sea = self.ensemble.sel(realization=0)
        # for station_name in self.station_names:
        #     svg = self.vgs[station_name]
        #     svg.sim_sea = self.sim_sea.sel(station=station_name).values

        # np.random.seed(0)
        # self.simulate(*args, **kwds)

        self.verbose = verbose
        if self.verbose:
            for station_name in self.station_names:
                print(f"\n{station_name}")
                self.print_means(station_name)
            self.print_ensemble_means()
        return self.ensemble

    def to_csv(self, csv_path, xar=None, filename_prefix=""):
        if xar is None:
            xar = self.sim_sea
        csv_path.mkdir(exist_ok=True, parents=True)
        for station_name in self.station_names:
            if self.varnames_dis is None:
                filename = f"{filename_prefix}{station_name}.csv"
                csv_df = xar.sel(station=station_name)
                csv_df.to_csv(csv_path / filename)
            else:
                filename = f"{filename_prefix}{station_name}_daily.csv"
                varnames_nondis = [
                    name
                    for name in self.varnames
                    if name not in self.varnames_dis
                ]
                csv_daily_df = (
                    xar.sel(station=station_name, variable=varnames_nondis)
                    .resample(time="D")
                    .mean()
                    .to_pandas()
                    .T
                )
                csv_daily_df.to_csv(csv_path / filename, float_format="%.3f")
                filename = f"{filename_prefix}{station_name}_hourly.csv"
                csv_hourly_df = (
                    xar.sel(station=station_name, variable=self.varnames_dis)
                    .to_pandas()
                    .T
                )
                csv_hourly_df.to_csv(csv_path / filename, float_format="%.3f")

    def print_means(self, station_name):
        obs = self.obs_means.sel(station=station_name, drop=True).to_dataframe(
            "obs"
        )
        if self.ensemble is not None:
            sim = (
                self.ensemble.sel(station=station_name, drop=True)
                .mean(["time", "realization"])
                .to_dataframe("sim")
            )
        else:
            sim = (
                self.sim_sea.sel(station=station_name, drop=True)
                .mean("time")
                .to_dataframe("sim")
            )
        # exclude possible varnames_ext
        sim = sim.loc[self.varnames]
        diff = pd.DataFrame(
            sim.values - obs.values, index=obs.index, columns=["diff"]
        )
        diff_perc = pd.DataFrame(
            100 * (sim.values - obs.values) / obs.values,
            index=obs.index,
            columns=["diff [%]"],
        )
        print(pd.concat([obs, sim, diff, diff_perc], axis=1).round(3))

    def print_all_means(self):
        obs = self.obs_means.mean("station").to_dataframe("obs")
        if self.ensemble is not None:
            sim = self.ensemble.mean(
                ["time", "realization", "station"]
            ).to_dataframe("sim")
        else:
            sim = self.sim_sea.mean(["time", "station"]).to_dataframe("sim")
        # exclude possible varnames_ext
        sim = sim.loc[self.varnames]
        diff = pd.DataFrame(
            sim.values - obs.values, index=obs.index, columns=["diff"]
        )
        diff_perc = pd.DataFrame(
            100 * (sim.values - obs.values) / obs.values,
            index=obs.index,
            columns=["diff [%]"],
        )
        print(pd.concat([obs, sim, diff, diff_perc], axis=1).round(3))

    def print_ensemble_means(self):
        obs = self.obs_means.mean("station").to_dataframe("obs")
        if self.ensemble is not None:
            sim = self.ensemble.mean(
                ["time", "station", "realization"]
            ).to_dataframe("sim")
        else:
            sim = self.sim_sea.mean(["time", "station"]).to_dataframe("sim")
        # exclude possible varnames_ext
        sim = sim.loc[self.varnames]
        diff = pd.DataFrame(
            sim.values - obs.values, index=obs.index, columns=["diff"]
        )
        diff_perc = pd.DataFrame(
            100 * (sim.values - obs.values) / obs.values,
            index=obs.index,
            columns=["diff [%]"],
        )
        print(pd.concat([obs, sim, diff, diff_perc], axis=1).round(3))

    def _vg_ph(self, vg_obj, sc_pars):
        """Call-back function for VG.simulate. Replaces a time_series_analysis
        model.

        """
        station_name = vg_obj.station_name
        weights = "tau"
        if self.vine is None and self.usevine:
            with tools.shelve_open(cop_conf.vine_cache) as sh:
                key = "_".join(
                    (
                        "_".join(sorted(vg_obj.var_names)),
                        "_".join(sorted(self.station_names)),
                        "_".join(str(i) for i in self.data_daily.data.shape),
                        weights,
                        vg_obj.primary_var[0],
                        self.discr,
                        self.rain_method,
                    )
                )
                if key in sh and not (self.refit or self.refit_vine):
                    vine = sh[key]
                else:
                    dtimes = np.tile(self.dtimes, self.n_stations)
                    vine = CVine(
                        self.ranks.data,
                        # varnames=vg_obj.var_names,
                        varnames=self.varnames,
                        dtimes=dtimes,
                        weights=weights,
                        central_node=vg_obj.primary_var[0],
                        verbose=False,
                        # verbose=self.verbose,
                        tau_min=0,
                        fit_mask=(
                            self.rain_mask.data
                            if "R" in self.varnames
                            else None
                        ),
                        asymmetry=self.asymmetry,
                        cop_candidates=self.cop_candidates,
                        scop_kwds=self.scop_kwds,
                    )
                    sh[key] = vine
            # these are not so interesting so far...
            vine.verbose = False
            self.vine = vine
            if self.verbose:
                print(vine)
            stacked = self.cop_quantiles.stack(stacked=("station", "time"))

            T_data = len(self.cop_quantiles.coords["time"])
            qq = self.vine.quantiles(T=(np.arange(stacked.shape[1]) % T_data))
            try:
                assert np.all(np.isfinite(qq))
            except AssertionError:
                qq[~np.isfinite(qq)] = np.nan
                qq = np.array([my.interp_nan(values) for values in qq])
            # normalize cop quantiles
            # qq = np.array([(stats.rankdata(values) - 0.5) / len(values)
            #                for values in qq])
            stacked.data = qq
            self.cop_quantiles = stacked.unstack(dim="stacked").transpose(
                *self.data_trans.dims
            )
        elif not self.usevine:
            self.qq_std = self.data_trans
        if self.phases is None:
            if self.usevine:

                def ppf_bounded(qq):
                    xx = dists.norm.ppf(qq)
                    xx[(~np.isfinite(qq)) | (np.abs(qq) > 6)] = np.nan
                    return xx

                qq_std = xr.apply_ufunc(ppf_bounded, self.cop_quantiles)
                self.qq_std = qq_std.interpolate_na("time")
            self.As.data = np.fft.fft(self.qq_std)
            self.phases = np.angle(self.As.sel(station=station_name))
            self.zero_phases = {
                station_name: np.angle(self.As.sel(station=station_name))[:, 0]
                for station_name in self.station_names
            }
            qq_means = self.qq_std.mean("time")
            qq_stds = self.qq_std.std("time")
            self.qq_means, self.qq_stds = qq_means, qq_stds

            # vine_bias = {}
            # for station_name in self.station_names:
            #     qq_station = qq_std.sel(station=station_name, drop=True)
            #     # quantiles_uni = np.full(qq_station.shape, 0.5)
            #     quantiles_uni = np.random.uniform(0, 1, qq_station.shape)
            #     self.sim_bias = np.zeros((self.K, 1))
            #     sim_uni = self.vine.simulate(randomness=quantiles_uni,
            #                                  T=np.arange(T_data))
            #     rank_bias = dists.norm.ppf(self.ranks.sel(station=station_name)).mean(axis=1)[:, None]
            #     sim_bias = dists.norm.ppf(sim_uni).mean(axis=1)[:, None]
            #     vine_bias[station_name] = sim_bias - rank_bias
            # self.vine_bias = vine_bias

        if self.rphases is None:
            T_sim = vg_obj.T
            T_data = vg_obj.T_summed
            T_total = 0
            phases_stacked = []
            while T_total < T_sim:
                # phase randomization with same random phases in all
                # variables and stations
                phases_len = T_data // 2 - 1 + T_data % 2
                phases_pos = np.random.uniform(0, 2 * np.pi, phases_len)
                # # do not touch phases that are close to the annual
                # # frequency
                # periods = np.fft.rfftfreq(phases_len * 2)[1:] ** -1
                # # mask = periods < 14
                # # mask = np.full_like(phases_pos, False, dtype=bool)
                # mask = (((periods > 7) & (periods < 28))
                #         | ((periods > 350) & (periods < 380))
                #         )
                # print(f"Holding {mask.mean() * 100:.3f} % "
                #       "of phases constant.")
                # # phases_pos[(periods > 250) & (periods < 400)] = 0
                # # phases_pos[periods > 30] = 0
                # phases_pos[mask] = 0
                phases_pos = np.array(self.K * [phases_pos])
                phases_neg = -phases_pos[:, ::-1]
                nyquist = np.full(self.K, 0)[:, None]
                zero_phases = self.phases[:, 0, None]
                if T_data % 2 == 0:
                    phases = np.hstack(
                        (zero_phases, phases_pos, nyquist, phases_neg)
                    )
                else:
                    phases = np.hstack((zero_phases, phases_pos, phases_neg))
                phases_stacked += [phases]
                T_total += T_data
            self.rphases = phases_stacked
            phases = phases_stacked
        else:
            # adjust zero-phase
            phases = []
            for phase_ in self.rphases:
                phase_[:, 0] = self.zero_phases[station_name]
                phases += [phase_]

        As = self.As.sel(station=station_name, drop=True)
        fft_sim = np.concatenate(
            [
                np.fft.ifft(As * np.exp(1j * phases_)).real
                for phases_ in phases
            ],
            axis=1,
        )[:, : vg_obj.T]

        fft_sim *= (
            self.qq_stds.sel(station=station_name).data / fft_sim.std(axis=1)
        )[:, None]
        fft_sim += (
            self.qq_means.sel(station=station_name).data - fft_sim.mean(axis=1)
        )[:, None]
        if self.phase_randomize_vary_mean:
            # allow means to vary. let it flow from the central_node to
            # the others
            mean_eps = np.zeros(self.K)[:, None]
            mean_eps[0, vg_obj.primary_var_ii[0]] = 0.25 * np.random.randn()
            fft_sim += mean_eps

        # change in mean scenario
        prim_i = tuple(vg_obj.primary_var_ii)
        fft_sim[prim_i] += sc_pars.m[prim_i]
        fft_sim[prim_i] += sc_pars.m_t[prim_i]

        if self.fft_sim is None:
            self.fft_sim = xr.DataArray(
                np.full((self.n_stations, self.K, vg_obj.T), 0.5),
                coords=[self.station_names, self.varnames, vg_obj.sim_times],
                dims=["station", "variable", "time"],
            )
        self.fft_sim.loc[dict(station=station_name)] = fft_sim

        if self.usevine:
            qq = dists.norm.cdf(fft_sim)
            ranks_sim = self.vine.simulate(
                T=np.arange(qq.shape[1]), randomness=qq
            )
            sim = dists.norm.ppf(ranks_sim)
            # sim -= self.vine_bias[station_name]
            # sim += (self.data_trans
            #         .sel(station=station_name)
            #         .mean("time")
            #         .data[:, None])
            assert np.all(np.isfinite(sim))
        else:
            sim = fft_sim
            ranks_sim = dists.norm.cdf(fft_sim)

        if self.ranks_sim is None:
            self.ranks_sim = xr.full_like(self.fft_sim, 0.5)
            self.sim = xr.full_like(self.fft_sim, 0)
        self.ranks_sim.loc[dict(station=station_name)] = ranks_sim
        # for being able to analyze later
        # self.sim.loc[dict(station=station_name)] = sim
        return sim

    def plot_map(self, resolution=10, terrain=True, *args, **kwds):
        if self.latitudes is None or self.longitudes is None:
            raise RuntimeError("Cant plot map without coordinates.")
        stamen_terrain = cimgt.StamenTerrain()
        crs = ccrs.PlateCarree()
        land_10m = cfeature.NaturalEarthFeature(
            "cultural",
            "admin_0_countries",
            "10m",
            edgecolor=(0, 0, 0, 0),
            facecolor=(0, 0, 0, 0),
        )
        fig = plt.figure(*args, **kwds)
        ax = fig.add_subplot(111, projection=stamen_terrain.crs)
        geodetic_transform = ccrs.Geodetic()._as_mpl_transform(ax)
        text_transform = offset_copy(geodetic_transform, units="dots", x=-25)
        for station_name in self.station_names:
            ax.text(
                self.longitudes.loc[station_name],
                self.latitudes.loc[station_name],
                station_name,
                fontsize=8,
                transform=text_transform,
                horizontalalignment="center",
                verticalalignment="bottom",
            )
        ax.scatter(self.longitudes, self.latitudes, transform=crs)
        ax.add_feature(land_10m, alpha=0.1)
        ax.add_feature(cfeature.STATES.with_scale("10m"))
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        if terrain:
            ax.add_image(stamen_terrain, resolution)
        return fig, ax

    def plot_seasonal(self, figsize=None):
        fig, axs = plt.subplots(
            nrows=self.n_stations,
            ncols=1,
            constrained_layout=True,
            figsize=figsize,
        )
        axs = np.ravel(axs)
        for stat_i, station_name in enumerate(self.station_names):
            ax = axs[stat_i]
            all_corrs = np.empty((12, self.K - 1), dtype=float)
            ranks_obs = self.ranks.unstack()
            months = ranks_obs.time.dt.month
            obs_ = ranks_obs.sel(station=station_name)
            for month_i, group in obs_.groupby(months):
                corrs = np.corrcoef(group)
                all_corrs[month_i - 1] = corrs[0, 1:]
            for corr, varname in zip(all_corrs.T, self.varnames[1:]):
                ax.plot(corr, label=varname)

            ax.set_prop_cycle(None)
            all_corrs = np.empty((12, self.K - 1), dtype=float)
            months = self.ranks_sim.time.dt.month
            obs_ = self.ranks_sim.sel(station=station_name)
            for month_i, group in obs_.groupby(months):
                corrs = np.corrcoef(group)
                all_corrs[month_i - 1] = corrs[0, 1:]
            for corr, varname in zip(all_corrs.T, self.varnames[1:]):
                ax.plot(corr, "--")
            ax.set_title(station_name)

        axs[0].legend(loc="best")
        return fig, axs

    def plot_ensemble_stats(
        self,
        stat_func=np.mean,
        obs=None,
        transformed=False,
        fig=None,
        axs=None,
    ):
        if obs is None:
            if transformed:
                obs = self.data_trans
            else:
                obs = self.data_daily
        if transformed:
            sim = self.ensemble_trans
        else:
            sim = self.ensemble

        # scipy.stats functions require wrapping
        if inspect.getmodule(stat_func) == stats.stats:

            @wraps(stat_func)
            def as_xarray(x):
                return xr.DataArray(stat_func(x, axis=None, nan_policy="omit"))

            stat_func_ = as_xarray
        else:
            stat_func_ = stat_func

        if fig is None and axs is None:
            fig, axs = plt.subplots(
                nrows=self.n_variables,
                ncols=1,
                sharex=True,
                figsize=(6, 6),
                constrained_layout=True,
            )
        xlocs = np.arange(self.n_stations)
        n_realizations = len(self.ensemble.coords["realization"])
        for ax, varname in zip(axs, self.varnames):
            ax.scatter(
                xlocs,
                (
                    obs.sel(variable=varname)
                    .groupby("station")
                    .apply(stat_func_, shortcut=False)
                ),
                facecolor="b",
                alpha=0.5,
            )
            var_stats = (
                sim.sel(variable=varname)
                .stack(stacked=["station", "realization"])
                .groupby("stacked")
                .apply(stat_func_, shortcut=False)
                .unstack("stacked")
            )
            try:
                # this is a known bug in xarray < 0.10.8
                var_stats = var_stats.rename(
                    dict(
                        stacked_level_0="station",
                        stacked_level_1="realization",
                    )
                )
            except ValueError:
                pass

            if n_realizations < 20:
                ax.scatter(
                    xlocs.repeat(n_realizations),
                    var_stats.values.ravel(),
                    s=4,
                    color="grey",
                )
            else:
                ax.violinplot(var_stats.values.T, xlocs, showmeans=True)
            ax.grid(True)
            ax.set_title(varname)
            ax.set_xticks(xlocs)
            ax.set_xticklabels(self.station_names, rotation=20)
        fig.suptitle(f"{stat_func.__name__} {transformed=}")
        return fig, axs

    def plot_ensemble_meteogram_hourly(self, *args, **kwds):
        dtimes = pd.to_datetime(self.ensemble.time.values)
        fig_axs = {}
        for station_name in self.station_names:
            svg = self.vgs[station_name]
            fig, axs = svg.plot_meteogram_hourly(
                plot_sim_sea=False, p_kwds=dict(linewidth=0.25), *args, **kwds
            )
            for ax, varname in zip(axs[:, 0], self.varnames):
                station_sims = self.ensemble.sel(
                    station=station_name, variable=varname
                ).load()
                mins = station_sims.min("realization")
                q10 = station_sims.quantile(0.10, "realization")
                q90 = station_sims.quantile(0.90, "realization")
                maxs = station_sims.max("realization")
                ax.fill_between(dtimes, mins, maxs, color="k", alpha=0.5)
                ax.fill_between(dtimes, q10, q90, color="red", alpha=0.5)
            for ax, varname in zip(axs[:, 1], self.varnames):
                station_sims = self.ensemble.sel(
                    station=station_name, variable=varname
                )
                ax.hist(
                    station_sims.values.ravel(),
                    40,
                    density=True,
                    orientation="horizontal",
                    histtype="step",
                    color="grey",
                )
            fig_axs[station_name] = fig, axs
        return fig_axs

    def plot_ensemble_meteogram_daily(self, *args, **kwds):
        dtimes = pd.to_datetime(self.ensemble.time.values)
        fig_axs = {}
        for station_name in self.station_names:
            svg = self.vgs[station_name]
            fig, axs = svg.plot_meteogram_daily(
                plot_sim_sea=False, p_kwds=dict(linewidth=0.25), *args, **kwds
            )
            for ax, varname in zip(axs[:, 0], self.varnames):
                station_sims = self.ensemble.sel(
                    station=station_name, variable=varname
                ).load()
                # if station_name == "Kinneret_station_A" and
                mins = station_sims.min("realization")
                q10 = station_sims.quantile(0.10, "realization")
                q90 = station_sims.quantile(0.90, "realization")
                maxs = station_sims.max("realization")
                ax.fill_between(dtimes, mins, maxs, color="k", alpha=0.5)
                ax.fill_between(dtimes, q10, q90, color="red", alpha=0.5)
            for ax, varname in zip(axs[:, 1], self.varnames):
                station_sims = self.ensemble.sel(
                    station=station_name, variable=varname
                )
                ax.hist(
                    station_sims.values.ravel(),
                    40,
                    density=True,
                    orientation="horizontal",
                    histtype="step",
                    color="grey",
                )
            fig_axs[station_name] = fig, axs
        return fig_axs

    def plot_ensemble_qq(
        self,
        obs=None,
        *args,
        lower_q=0.01,
        upper_q=0.99,
        figsize=None,
        fig_axs=None,
        color="b",
        **kwds,
    ):
        if obs is None:
            obs = self.data_daily
        if fig_axs is None:
            fig_axs = {}
        n_axes = len(self.varnames)
        n_cols = int(np.ceil(n_axes ** 0.5))
        n_rows = int(np.ceil(float(n_axes) / n_cols))
        alphas = np.linspace(0, 1, 200)
        bounds_kwds = dict(linestyle="--", color="gray", alpha=0.5)
        for station_name in self.station_names:
            if station_name in fig_axs:
                fig, axs = fig_axs[station_name]
            else:
                fig, axs = plt.subplots(
                    n_rows,
                    n_cols,
                    subplot_kw=dict(aspect="equal"),
                    constrained_layout=True,
                    figsize=figsize,
                )
            axs = np.ravel(axs)
            for ax_i, (ax, varname) in enumerate(zip(axs, self.varnames)):
                obs_ = obs.sel(station=station_name, variable=varname)
                sim_ = self.ensemble.sel(
                    station=station_name, variable=varname
                ).load()
                global_min = min(obs_.min(), sim_.min())
                global_max = max(obs_.max(), sim_.max())
                obs_qq = obs_.quantile(alphas, "time")
                sim_qq = sim_.quantile(alphas, "time")
                # calling quantile on the result of a quantile call
                # causes name collisions
                sim_qq = sim_qq.rename(dict(quantile="qq"))
                sim_lower = sim_qq.quantile(lower_q, "realization")
                sim_upper = sim_qq.quantile(upper_q, "realization")
                ax.plot(sim_qq.median("realization"), obs_qq, color=color)
                ax.plot(sim_qq.min("realization"), obs_qq, **bounds_kwds)
                ax.fill_betweenx(
                    obs_qq, sim_lower, sim_upper, color=color, alpha=0.5
                )
                ax.plot(sim_qq.max("realization"), obs_qq, **bounds_kwds)
                ax.plot(
                    [global_min, global_max],
                    [global_min, global_max],
                    "--",
                    color="k",
                )
                ax.grid(True)
                ax.set_ylabel("observed")
                if ax_i < (n_axes - n_cols):
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel("simulated")
                ax.set_title(varname)
            fig.suptitle(station_name)
            # delete axes that we did not use
            if len(axs) > n_axes:
                for ax in axs[n_axes:]:
                    fig.delaxes(ax)
                plt.draw()
            fig_axs[station_name] = fig, axs
        return fig_axs

    def plot_ensemble_exceedance_daily(
        self,
        *args,
        obs=None,
        figsize=None,
        thresh=None,
        name_figaxs=None,
        sim_color="blue",
        **kwds,
    ):
        assert "R" in self.varnames
        if thresh is None:
            thresh = vg.conf.dists_kwds["R"]["threshold"] / 24
        if name_figaxs is None:
            name_figaxs = self.plot_exceedance_daily(
                draw_scatter=False, figsize=figsize, thresh=thresh, alpha=0
            )
        if obs is None:
            obs = self.data_daily
        bounds_kwds = dict(linestyle="--", color="gray", alpha=0.5)
        # q_levels = np.linspace(0, 1, self.T)
        n_levels = self.T_sim / 2
        q_levels = (np.arange(n_levels) + 0.5) / n_levels
        for station_name, (fig, axs) in name_figaxs.items():
            obs_ = obs.sel(station=station_name, variable="R").load()
            sim_ = self.ensemble_daily.sel(
                station=station_name, variable="R"
            ).load()
            # depth part
            sim_qq = (
                sim_.where(sim_ > thresh)
                .quantile(q_levels, "time")
                .rename(dict(quantile="qq"))
            )
            obs_qq = (
                obs_.where(obs_ > thresh)
                .quantile(q_levels, "time")
                .rename(dict(quantile="qq"))
            )
            mins = sim_qq.min("realization")
            median = sim_qq.quantile(0.5, "realization")
            maxs = sim_qq.max("realization")
            ax = axs[0]
            q_levels = q_levels[::-1]
            ax.loglog(mins, q_levels, **bounds_kwds)
            ax.loglog(median, q_levels, color=sim_color)
            ax.loglog(maxs, q_levels, **bounds_kwds)
            ax.fill_betweenx(q_levels, mins, maxs, alpha=0.5, color=sim_color)
            ax.loglog(obs_qq, q_levels, color="k")

            dry_obs, wet_obs = rain_stats.spell_lengths(obs_, thresh=thresh)
            drys, wets = [], []
            dry_levels = 100 * np.linspace(0, 1, len(dry_obs))
            wet_levels = 100 * np.linspace(0, 1, len(wet_obs))
            for real_i, real in sim_.groupby("realization"):
                dry, wet = rain_stats.spell_lengths(real, thresh=thresh)
                drys += [np.percentile(dry, dry_levels)]
                wets += [np.percentile(wet, wet_levels)]
            episodes_sim = [drys, wets]
            episodes_obs = [
                np.percentile(dry_obs, dry_levels),
                np.percentile(wet_obs, wet_levels),
            ]
            levels = [dry_levels, wet_levels]
            for i, (ax, level) in enumerate(zip(axs[1:], levels)):
                episode_sim = episodes_sim[i]
                mins = np.min(episode_sim, axis=0)
                median = np.median(episode_sim, axis=0)
                maxs = np.max(episode_sim, axis=0)
                level = level[::-1] / 100
                ax.loglog(mins, level, **bounds_kwds)
                ax.loglog(median, level, color=sim_color)
                ax.loglog(maxs, level, **bounds_kwds)
                ax.loglog(episodes_obs[i], level, color="k")
                ax.fill_betweenx(level, mins, maxs, alpha=0.5, color=sim_color)

            for ax in axs:
                ax.grid(True)
            suptitle_prepend(fig, station_name)
        return name_figaxs

    def plot_ensemble_violins(
        self, figsize=None, time_period="m", *args, **kwds
    ):
        fig_axs = {}
        for station_name in self.station_names:
            fig, axs = plt.subplots(
                nrows=self.K, ncols=1, figsize=figsize, constrained_layout=True
            )
            for ax, varname in zip(axs, self.varnames):
                month = self.data_daily.time.dt.month
                obs_means = (
                    self.data_daily.sel(station=station_name, variable=varname)
                    .groupby(month)
                    .mean()
                )
                month = self.ensemble.time.dt.month
                sim_means = (
                    self.ensemble.sel(station=station_name, variable=varname)
                    .groupby(month)
                    .mean("time")
                )
                ax.violinplot(sim_means.T, showmeans=True)
                ax.scatter(np.arange(1, 13), obs_means, marker="_", color="k")
                ax.set_title(varname)
            suptitle_prepend(fig, station_name)
            fig_axs[station_name] = fig, axs
        return fig_axs

    def plot_exceedance_daily(self, draw_scatter=True, *args, **kwds):
        fig_axs = {}
        figsize = kwds.get("figsize", None)
        for station_name in self.station_names:
            fig, axs = plt.subplots(
                ncols=3, figsize=figsize, constrained_layout=True
            )
            svg = self.vgs[station_name]
            fig, axs = svg.plot_exceedance_daily(
                fig=fig, axs=axs, draw_scatter=draw_scatter, *args, **kwds
            )
            suptitle_prepend(fig, station_name)
            fig_axs[station_name] = fig, axs
        return fig_axs

    def plot_corr(self, ygreek=None, hourly=False, text=False, *args, **kwds):
        data = self.data_daily.stack(stacked=("station", "variable")).T.data
        fig, _ = ts.corr_img(
            data, 0, "Measured daily", text=text, *args, **kwds  # greek_short,
        )
        fig.name = "corr_measured_daily"
        fig = [fig]

        data = self.data_trans.stack(stacked=("station", "variable")).T.data
        fig += [
            ts.corr_img(
                data,
                0,
                "Measured daily transformed",  # greek_short,
                text=text,
                *args,
                **kwds,
            )
        ]

        if self.sim_sea is not None:
            data = self.sim.stack(stacked=("station", "variable")).T.data
            fig_ = ts.corr_img(
                data,
                0,
                "Simulated daily normal",  # greek_short,
                text=text,
                *args,
                **kwds,
            )
            fig_.name = "corr_sim_daily"
            fig += [fig_]

            data = self.sim_sea.stack(stacked=("station", "variable")).T.data
            fig_ = ts.corr_img(
                data,
                0,
                "Simulated daily",  # greek_short,
                text=text,
                *args,
                **kwds,
            )
            fig_.name = "corr_sim_daily"
            fig += [fig_]

        for fig_ in fig:
            ax = fig_.get_axes()[0]
            ax.set_xticklabels("")
            ax.set_yticklabels("")
            for i, station_name in enumerate(self.station_names):
                ax.axvline(i * self.K - 0.5, linewidth=0.5, color="k")
                ax.axhline(i * self.K - 0.5, linewidth=0.5, color="k")
                if text:
                    name = self.station_names_short[station_name]
                    ax.text(
                        (i + 0.5) * self.K - 0.5,
                        16,
                        name,
                        fontsize=6,
                        horizontalalignment="center",
                        verticalalignment="top",
                    )
                    ax.text(
                        16,
                        (i + 0.5) * self.K - 0.5,
                        name,
                        fontsize=6,
                        horizontalalignment="left",
                        verticalalignment="center",
                        rotation=90,
                    )
        return fig

    def plot_corr_scatter_var(
        self, transformed=False, hourly=False, fft=False, *args, **kwds
    ):
        (data_obs, data_sim, title_substring) = self._corr_scatter_data(
            transformed, hourly, fft
        )
        return self._plot_corr_scatter_var(
            data_obs, data_sim, title_substring=title_substring, *args, **kwds
        )

    def plot_corr_scatter_stat(
        self, transformed=False, hourly=False, fft=False, *args, **kwds
    ):
        (data_obs, data_sim, title_substring) = self._corr_scatter_data(
            transformed, hourly, fft
        )
        return self._plot_corr_scatter_stat(
            data_obs, data_sim, title_substring=title_substring, *args, **kwds
        )

    def _corr_scatter_data(self, transformed=False, hourly=False, fft=False):
        if transformed:
            data_obs = self.data_trans
            data_sim = self.sim_trans
            # data_obs = self.ranks.unstack()
            # data_sim = self.ranks_sim
            # data_obs = self.qq_std
            # qq_std = xr.zeros_like(self.cop_quantiles)
            # qq_std.data = self.qq_std
            # data_obs = qq_std
            # data_sim = self.fft_sim
            title_substring = "transformed "
        elif hourly:
            data_obs = self.xar.isel(time=slice(None, -24))
            data_sim = self.sim_sea_dis
            title_substring = "hourly"
        elif fft:
            data_obs = self.qq_std
            data_sim = self.fft_sim
            title_substring = "fft"
        else:
            data_obs = self.data_daily
            data_sim = self.sim_sea
            title_substring = ""
        return data_obs, data_sim, title_substring

    def _plot_corr_scatter_stat(
        self,
        obs_ar,
        sim_ar,
        *args,
        title_substring="",
        fig=None,
        axs=None,
        figsize=None,
        color=None,
        alpha=0.75,
        **kwds,
    ):
        if fig is None and axs is None:
            fig, axs = plt.subplots(
                nrows=self.n_variables - 1,
                ncols=self.n_variables - 1,
                figsize=figsize,
                subplot_kw=dict(aspect="equal"),
                constrained_layout=True,
            )
        if color is None:
            color = np.array(
                [
                    plt.get_cmap("viridis")(i)
                    for i in np.linspace(0, 1, self.n_stations)
                ]
            )
            color[:, -1] = alpha
            draw_legend = True
        else:
            color = self.n_stations * [color]
            draw_legend = False
        mins = np.full(2 * (self.n_variables,), np.inf)
        maxs = np.full_like(mins, -np.inf)
        for stat_i, station_name in enumerate(self.station_names):
            data_sim_ = sim_ar.sel(station=station_name).values
            sim_corr = np.corrcoef(data_sim_)
            ii, jj = np.triu_indices_from(sim_corr, 1)
            sim_corr = sim_corr[ii, jj]
            data_obs_ = obs_ar.sel(station=station_name).values
            obs_corr = nan_corrcoef(data_obs_)
            obs_corr = obs_corr[ii, jj]
            for s_corr, o_corr, i, j in zip(sim_corr, obs_corr, ii, jj):
                ax = axs[i, j - 1]
                ax.set_title(
                    f"{self.varnames[min(i,j)]} - "
                    f"{self.varnames[max(i,j)]}"
                )
                ax.scatter(
                    s_corr,
                    o_corr,
                    edgecolor=color[stat_i],
                    facecolor=(0, 0, 0, 0),
                    label=station_name,
                    *args,
                    **kwds,
                )
                mins[i, j] = min(mins[i, j], s_corr, o_corr)
                maxs[i, j] = max(maxs[i, j], s_corr, o_corr)
        for var_i in range(self.n_variables - 1):
            for var_j in range(self.n_variables - 1):
                ax = axs[var_i, var_j]
                if var_i > var_j:
                    ax.set_axis_off()
                    continue
                elif var_i == var_j:
                    ax.set_xlabel("simulated")
                    ax.set_ylabel("observed")
                ax.plot(
                    [mins[var_i, var_j + 1], maxs[var_i, var_j + 1]],
                    [mins[var_i, var_j + 1], maxs[var_i, var_j + 1]],
                    linestyle="--",
                    color="gray",
                )
                ax.grid(True)
        if draw_legend:
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, "lower left")
        fig.suptitle(f"Intra-site correlations {title_substring}")
        fig.tight_layout()
        return fig, axs

    # def _plot_corr_scatter_stat(self, obs_ar, sim_ar, *args, color="b",
    #                           title_substring="", figsize=None,
    #                           fig=None, axs=None, alpha=.75, **kwds):
    #     if fig is None and axs is None:
    #         fig, axs = plt.subplots(nrows=self.n_variables-1,
    #                                 ncols=self.n_variables-1,
    #                                 figsize=figsize,
    #                                 # subplot_kw=dict(aspect="equal"),
    #                                 constrained_layout=True
    #                                 )
    #     edgecolor = kwds.pop("edgecolor", None)
    #     if edgecolor is None:
    #         edgecolor = np.array([plt.get_cmap("viridis")(i)
    #                               for i in np.linspace(0, 1, self.n_stations)])
    #         edgecolor[:, -1] = alpha
    #         draw_legend = True
    #     else:
    #         edgecolor = self.n_stations * [edgecolor]
    #         draw_legend = False
    #     mins = np.full(2 * (self.n_variables,), np.inf)
    #     maxs = np.full_like(mins, -np.inf)
    #     for stat_i, station_name in enumerate(self.station_names):
    #         data_sim_ = (sim_ar
    #                      .sel(station=station_name)
    #                      .values)
    #         sim_corr = np.corrcoef(data_sim_)
    #         ii, jj = np.triu_indices_from(sim_corr, 1)
    #         sim_corr = sim_corr[ii, jj]
    #         data_obs_ = (obs_ar
    #                      .sel(station=station_name)
    #                      .values)
    #         obs_corr = nan_corrcoef(data_obs_)
    #         obs_corr = obs_corr[ii, jj]
    #         for s_corr, o_corr, i, j in zip(sim_corr, obs_corr, ii, jj):
    #             ax = axs[i, j - 1]
    #             ax.set_title(f"{self.varnames[min(i,j)]} - "
    #                          f"{self.varnames[max(i,j)]}")
    #             ax.scatter(s_corr, o_corr,
    #                        edgecolor=edgecolor[stat_i],
    #                        facecolor=(0, 0, 0, 0),
    #                        label=station_name, *args, **kwds)
    #             mins[i, j] = min(mins[i, j], s_corr, o_corr)
    #             maxs[i, j] = max(maxs[i, j], s_corr, o_corr)
    #     for var_i in range(self.n_variables - 1):
    #         for var_j in range(self.n_variables - 1):
    #             ax = axs[var_i, var_j]
    #             if var_i > var_j:
    #                 ax.set_axis_off()
    #                 continue
    #             elif var_i == var_j:
    #                 ax.set_xlabel("simulated")
    #                 ax.set_ylabel("observed")
    #             ax.plot([mins[var_i, var_j + 1],
    #                      maxs[var_i, var_j + 1]],
    #                     [mins[var_i, var_j + 1],
    #                      maxs[var_i, var_j + 1]],
    #                     linestyle="--", color="gray")
    #             ax.grid(True)
    #     if draw_legend:
    #         handles, labels = axs[0, 0].get_legend_handles_labels()
    #         fig.legend(handles, labels, "lower left")
    #     fig.suptitle(f"Intra-site correlations {title_substring}")
    #     # fig.tight_layout()
    #     return fig, axs

    def _plot_ensemble_corr_scatter_var(
        self,
        obs_ar,
        sim_ar,
        title_substring="",
        figsize=None,
        varnames=None,
        alpha=0.75,
        *args,
        **kwds,
    ):
        if varnames is None:
            varnames = self.varnames
        K = len(varnames)
        nrows = int(np.sqrt(K))
        ncols = int(np.ceil(K / nrows))
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            subplot_kw=dict(aspect="equal"),
            constrained_layout=True,
            figsize=figsize,
        )
        axs = np.ravel(axs)
        edgecolor = kwds.pop("edgecolor", None)
        if edgecolor is None:
            edgecolor = np.array(
                [
                    plt.get_cmap("viridis")(i)
                    for i in np.linspace(0, 1, self.n_stations)
                ]
            )
            edgecolor[:, -1] = alpha
            draw_legend = True
        else:
            edgecolor = self.n_stations * [edgecolor]
            draw_legend = False
        mins = np.full(2 * (self.n_variables,), np.inf)
        maxs = np.full_like(mins, -np.inf)
        for var_i, varname in enumerate(self.varnames):
            data_sim_ = sim_ar.sel(variable=varname).values
            sim_corrs = []
            for values in data_sim_:
                sim_corr = np.corrcoef(values)
                ii, jj = np.triu_indices_from(sim_corr, 1)
                sim_corrs += [sim_corr[ii, jj]]
            sim_corrs = np.array(sim_corrs).T
            data_obs_ = obs_ar.sel(variable=varname).values
            obs_corr = nan_corrcoef(data_obs_)
            obs_corr = obs_corr[ii, jj]
            ax = axs[var_i]
            for s_corrs, o_corr in zip(sim_corrs, obs_corr):
                ax.plot(
                    [s_corrs.min(), s_corrs.max()],
                    [o_corr, o_corr],
                    linestyle="-",
                    marker="|",
                    markersize=2.5,
                    color=(0, 0, 1, alpha),
                )
            min_corr = min(sim_corr.min(), obs_corr.min())
            ax.plot(
                [min_corr, 1], [min_corr, 1], "k", linestyle="--", zorder=99
            )
            ax.set_xlabel("simulated")
            ax.set_ylabel("observed")
            ax.grid(True)
            ax.set_title(varname)
        fig.suptitle(f"Inter-site correlations {title_substring}")
        return fig, ax

    def _plot_ensemble_corr_scatter_stat(
        self,
        obs_ar,
        sim_ar,
        title_substring="",
        figsize=None,
        alpha=0.75,
        *args,
        **kwds,
    ):
        fig, axs = plt.subplots(
            nrows=self.n_variables - 1,
            ncols=self.n_variables - 1,
            figsize=figsize,
            # subplot_kw=dict(aspect="equal"),
            constrained_layout=True,
        )
        edgecolor = kwds.pop("edgecolor", None)
        if edgecolor is None:
            edgecolor = np.array(
                [
                    plt.get_cmap("viridis")(i)
                    for i in np.linspace(0, 1, self.n_stations)
                ]
            )
            edgecolor[:, -1] = alpha
            draw_legend = True
        else:
            edgecolor = self.n_stations * [edgecolor]
            draw_legend = False
        mins = np.full(2 * (self.n_variables,), np.inf)
        maxs = np.full_like(mins, -np.inf)
        for stat_i, station_name in enumerate(self.station_names):
            data_sim_ = sim_ar.sel(station=station_name).values
            sim_corrs = []
            for values in data_sim_:
                sim_corr = np.corrcoef(values)
                ii, jj = np.triu_indices_from(sim_corr, 1)
                sim_corrs += [sim_corr[ii, jj]]
            sim_corrs = np.array(sim_corrs).T
            data_obs_ = obs_ar.sel(station=station_name).values
            obs_corr = nan_corrcoef(data_obs_)
            obs_corr = obs_corr[ii, jj]
            for s_corrs, o_corr, i, j in zip(sim_corrs, obs_corr, ii, jj):
                ax = axs[i, j - 1]
                ax.set_title(
                    f"{self.varnames[min(i,j)]} - "
                    f"{self.varnames[max(i,j)]}"
                )
                # parts = ax.violinplot(s_corrs, [o_corr], vert=False,
                #                       # showmedians=True,
                #                       widths=0.01)
                # for pc in parts["bodies"]:
                #     pc.set_facecolor("b")
                #     pc.set_edgecolor("b")
                ax.plot(
                    [s_corrs.min(), s_corrs.max()],
                    [o_corr, o_corr],
                    linestyle="-",
                    marker="|",
                    markersize=2.5,
                    color="b",
                )
                mins[i, j] = min(mins[i, j], np.min(s_corrs), o_corr)
                maxs[i, j] = max(maxs[i, j], np.max(s_corrs), o_corr)
        for var_i in range(self.n_variables - 1):
            for var_j in range(self.n_variables - 1):
                ax = axs[var_i, var_j]
                if var_i > var_j:
                    ax.set_axis_off()
                    continue
                ax.plot(
                    [mins[var_i, var_j + 1], maxs[var_i, var_j + 1]],
                    [mins[var_i, var_j + 1], maxs[var_i, var_j + 1]],
                    linestyle="--",
                    color="gray",
                )
                ax.set_xlabel("simulated")
                ax.set_ylabel("observed")
                ax.grid(True)
        if draw_legend:
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, "lower left")
        fig.suptitle(f"Intra-site correlations {title_substring}")
        fig.tight_layout()
        return fig, ax

    def _plot_corr_scatter_var(
        self,
        obs_ar,
        sim_ar,
        title_substring="",
        fig=None,
        axs=None,
        figsize=None,
        varnames=None,
        *args,
        **kwds,
    ):
        if varnames is None:
            varnames = self.varnames
        K = len(varnames)
        if fig is None and axs is None:
            nrows = int(np.sqrt(K))
            ncols = int(np.ceil(K / nrows))
            fig, axs = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                subplot_kw=dict(aspect="equal"),
                constrained_layout=True,
                figsize=figsize,
            )
            axs = np.ravel(axs)
        s = kwds.pop("s", None)
        for var_i, varname in enumerate(varnames):
            ax = axs[var_i]
            data_sim_ = sim_ar.sel(variable=varname).values
            sim_corr = nan_corrcoef(data_sim_)
            sim_corr = sim_corr[np.triu_indices_from(sim_corr, 1)]
            data_obs_ = obs_ar.sel(variable=varname).values
            obs_corr = nan_corrcoef(data_obs_)
            obs_corr = obs_corr[np.triu_indices_from(obs_corr, 1)]
            ax.scatter(
                sim_corr,
                obs_corr,
                s=s,
                # s=((50 * nan_corrcoef.overlap)
                #    if s is None else s),
                # facecolors="None", edgecolors="b",
                # alpha=.75,
                *args,
                **kwds,
            )
            min_corr = min(sim_corr.min(), obs_corr.min())
            ax.plot(
                [min_corr, 1], [min_corr, 1], "k", linestyle="--", zorder=99
            )
            ax.set_xlabel("simulated")
            ax.set_ylabel("observed")
            ax.grid(True)
            ax.set_title(varname)
        fig.suptitle(f"Inter-site correlations {title_substring}")
        return fig, axs

    def plot_cross_corr_var(
        self,
        *,
        fig=None,
        axs=None,
        varname=None,
        max_lags=7,
        transformed=False,
        figsize=None,
    ):
        if varname is None:
            varname = self.vgs[self.station_names[0]].primary_var[0]
        if transformed:
            data_obs = self.data_trans
            data_sim = self.sim
            title_substring = "transformed "
        else:
            data_obs = self.data_daily
            data_sim = self.sim_sea
            title_substring = ""
        data_obs = (
            data_obs.sel(variable=varname).transpose("station", "time").data
        )
        kwds = dict(
            var_names=self.station_names, max_lags=max_lags, figsize=figsize
        )
        n_axes = len(self.station_names)
        n_cols = int(np.ceil(n_axes ** 0.5))
        n_rows = int(np.ceil(float(n_axes) / n_cols))
        if fig is None and axs is None:
            figsize = kwds.pop("figsize")
            fig, axs = plt.subplots(
                nrows=n_rows, ncols=n_cols, figsize=figsize
            )
            axs = np.ravel(axs)
        fig, axs = ts.plot_cross_corr(data_obs, fig=fig, axs=axs, **kwds)
        fig.suptitle("Crosscorrelations " f"{title_substring}{varname}")

        if self.sim is not None:
            data_sim = (
                data_sim.sel(variable=varname)
                .transpose("station", "time")
                .data
            )
            fig, axs = ts.plot_cross_corr(
                data_sim, linestyle="--", fig=fig, axs=axs, **kwds
            )
        return fig, axs

    def plot_cross_corr_stat(
        self, *, station_name=None, max_lags=7, transformed=False
    ):
        if station_name is None:
            station_name = self.station_names[0]
        if transformed:
            data_obs = self.data_trans
            data_sim = self.sim
            title_substring = "transformed "
        else:
            data_obs = self.data_daily
            data_sim = self.sim_sea
            title_substring = ""
        data_obs = (
            data_obs.sel(station=station_name)
            .transpose("variable", "time")
            .data
        )
        kwds = dict(var_names=self.varnames, max_lags=max_lags)
        fig, axs = ts.plot_cross_corr(data_obs, **kwds)
        fig.suptitle("Crosscorrelations " f"{title_substring}{station_name}")

        if self.sim is not None:
            data_sim = (
                data_sim.sel(station=station_name)
                .transpose("variable", "time")
                .data
            )
            fig, axs = ts.plot_cross_corr(
                data_sim, linestyle="--", fig=fig, axs=axs, **kwds
            )
        return fig, axs

    def plot_meteogram_trans_stat(self, *args, **kwds):
        vg_first = self.vgs[self.station_names[0]]
        figs, axss = vg_first.plot_meteogram_trans(
            station_name=None, *args, **kwds
        )
        if not isinstance(figs, Iterable):
            figs = [figs]
            axss = [axss]
        for station_name in self.station_names[1:]:
            svg = self.vgs[station_name]
            svg.plot_meteogram_trans(figs=figs, axss=axss, *args, **kwds)

        for fig in figs:
            fig.subplots_adjust(right=0.75, hspace=0.25)
            fig.legend(
                axss[0][0][0].lines, self.station_names, loc="center right"
            )
        return figs, axss

        figs, axss = vg_first.plot_meteogram_daily(
            station_name=None, *args, **kwds
        )
        return figs, axss

    def plot_meteogram_daily_stat(self, *args, **kwds):
        vg_first = self.vgs[self.station_names[0]]
        figs, axss = vg_first.plot_meteogram_daily(
            station_name=None, *args, **kwds
        )
        if not isinstance(figs, Iterable):
            figs = [figs]
            axss = [axss]
        for station_name in self.station_names[1:]:
            svg = self.vgs[station_name]
            svg.plot_meteogram_daily(
                station_name=None, figs=figs, axss=axss, *args, **kwds
            )
        for fig in figs:
            fig.subplots_adjust(right=0.75, hspace=0.25)
            fig.legend(
                axss[0][0][0].lines, self.station_names, loc="center right"
            )
        return figs, axss

    def plot_meteogram_daily_decorr(self, varnames=None, *args, **kwds):
        if varnames is None:
            varnames = self.varnames
        station_name = self.station_names[0]
        vg_first = self.vgs[station_name]
        obs = self.qq_std.sel(station=station_name).values
        sim = self.fft_sim.sel(station=station_name).values
        figs, axss = vg_first.plot_meteogram_daily(
            obs=obs,
            sim=sim,
            var_names=varnames,
            station_name=None,
            plot_daily_bounds=False,
            *args,
            **kwds,
        )
        if not isinstance(figs, Iterable):
            figs = [figs]
            axss = [axss]
        for station_name in self.station_names[1:]:
            svg = self.vgs[station_name]
            obs = self.qq_std.sel(station=station_name).values
            sim = self.fft_sim.sel(station=station_name).values
            svg.plot_meteogram_daily(
                obs=obs,
                sim=sim,
                var_names=varnames,
                station_name=None,
                plot_daily_bounds=False,
                figs=figs,
                axss=axss,
                *args,
                **kwds,
            )
        for fig in figs:
            fig.subplots_adjust(right=0.75, hspace=0.25)
            fig.legend(
                axss[0][0][0].lines, self.station_names, loc="center right"
            )
        return figs, axss

    def plot_ccplom(self, masked=False, *args, **kwds):
        """Cross-Copula-plot matrix of input and output."""
        if masked and "R" in self.varnames:
            obs_all = self.ranks[:, self.rain_mask.data]
            thresh = vg.conf.threshold
            sim_all = self.ranks_sim.where(
                self.sim_sea.sel(variable="R") >= thresh
            ).stack(rank=("time", "station"))
        else:
            obs_all = self.ranks
            sim_all = self.ranks_sim.stack(rank=("time", "station"))
        fig_in, axs_in = plotting.ccplom(
            obs_all.data, varnames=self.varnames, *args, **kwds
        )
        fig_in.suptitle("Input")
        fig_out, axs_out = plotting.ccplom(
            sim_all.data, varnames=self.varnames, *args, **kwds
        )
        fig_out.suptitle("Output")
        return (fig_in, fig_out), (axs_in, axs_out)

    def plot_ccplom_seasonal(self, masked=False, *args, **kwds):
        """Cross-Copula-plot matrix of input and output."""
        figs = {}
        if masked and "R" in self.varnames:
            obs_all = self.ranks[:, self.rain_mask.data].unstack("rank")
            thresh = vg.conf.threshold
            sim_all = self.ranks_sim.where(
                self.sim_sea.sel(variable="R") >= thresh
            )
        else:
            obs_all = self.ranks.unstack("rank")
            sim_all = self.ranks_sim
        for season, obs in obs_all.groupby("time.season"):
            obs = obs.stack(rank=("time", "station")).dropna("rank")
            fig, axs = plotting.ccplom(
                obs.data, varnames=self.varnames, *args, **kwds
            )
            fig.suptitle(f"Input {season}")
            figs[f"in_{season}"] = fig, axs
        for season, sim in sim_all.groupby("time.season"):
            sim = sim.stack(rank=("time", "station")).dropna("rank")
            fig, axs = plotting.ccplom(
                sim.data, varnames=self.varnames, *args, **kwds
            )
            fig.suptitle(f"Output {season}")
            figs[f"out_{season}"] = fig, axs
        return figs


if __name__ == "__main__":
    import opendata_vg_conf as vg_conf

    set_conf(vg_conf)
    xds = xr.open_dataset(
        "/home/dirk/data/opendata_dwd/" "multisite_testdata.nc"
    )
    # station_names = list(xar.station.values)
    # station_names.remove("Sigmarszell-Zeisertsweiler")
    # xar = xar.sel(station=station_names)

    # import warnings
    # warnings.simplefilter("error", category=RuntimeWarning)
    wc = Multisite(
        xds,
        verbose=True,
        # refit=True,
        # refit_vine=True,
        # refit=True,
        # refit="R",
        # refit=("R", "sun"),
        # refit="sun",
        rain_method="regression",
        # rain_method="distance",
        # debias=True,
        # cop_candidates=dict(gaussian=cops.gaussian),
        scop_kwds=dict(window_len=30, fft_order=3),
    )

    np.random.seed(0)
    # sim = wc.simulate(usevine=False)
    sim = wc.simulate(usevine=True)
    sim = wc.simulate_ensemble(
        25,
        "test_vine",
        clear_cache=True,
        # usevine=False
    )
    wc.plot_corr_scatter_var(transformed=True)
    wc.plot_seasonal()
    wc.plot_ensemble_stats()
    wc.plot_ensemble_stats(transformed=True)
    # wc.vine.plot()
    print(wc.vine)
    plt.show()

    # for varname in wc.varnames:
    #     wc.plot_cross_corr_var(varname=varname)

    # wc.plot_meteogram_trans_stations()
    # wc.plot_daily_fit("R")

    # for station_name in wc.station_names:
    #     rain_dist, solution = wc.vgs[station_name].dist_sol["R"]
    #     fig, axs = rain_dist.plot_fourier_fit()
    #     fig.suptitle(station_name)
    #     fig, axs = rain_dist.plot_monthly_fit()
    #     fig.suptitle(station_name)
    # plt.show()

    # for station_name in wc.station_names:
    #     sun_dist, solution = wc.vgs[station_name].dist_sol["sun"]
    #     fig, axs = sun_dist.scatter_pdf(solution)
    #     fig.suptitle(station_name)
    #     # fig, axs = sun_dist.plot_monthly_fit()
    #     # fig.suptitle(station_name)
    # plt.show()

    # sim = wc.simulate_mc(2)
    # sim = wc.simulate(theta_incr=4, disturbance_std=3, mean_arrival=7)

    # wc.plot_meteogram_daily_stations()
    # wc.plot_meteogram_trans_stations()
    # wc.plot_qq()
    # # wc.plot_meteogram_daily()
    # # wc.plot_meteogram_trans()
    # # wc.plot_meteogram_hourly()
    # wc.plot_doy_scatter()
    # wc.plot_exceedance_daily()
    # for station_name in wc.station_names:
    #     wc.plot_cross_corr_stat(station_name=station_name)
    # wc.ccplom()
    # wc.vine.plot(edge_labels="copulas")
    # # wc.vine.plot_seasonal()
    # wc.vine.plot_qqplom()
    # # doesn't work with seasonal copulas
    # # wc.vine.plot_tplom()
    # plt.show()
