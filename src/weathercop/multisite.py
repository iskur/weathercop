from collections.abc import Iterable
from collections import OrderedDict, namedtuple
import os
import shutil
import inspect
from functools import wraps, partial
from itertools import repeat
from pathlib import Path
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, interpolate
import xarray as xr
import dill
from multiprocessing import Pool
import copy
import threading
from multiprocessing import current_process  # For legacy debug output
from tqdm import tqdm

# Configure logging for memory optimization diagnostics
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
from matplotlib.transforms import offset_copy
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature

import varwg
from varwg.core.core import seasonal_back
from varwg import core as vg_core
from varwg import base as vg_base
from varwg import plotting as vg_plotting
from varwg.time_series_analysis import (
    distributions as dists,
    time_series as ts,
    rain_stats,
)
from varwg.time_series_analysis.phase_randomization import _random_phases
from weathercop import cop_conf, plotting as wplt, tools, copulae as cops
from weathercop.vine import CVine, MultiStationVine

DEBUG = cop_conf.DEBUG

# Locks for multiprocessing worker I/O synchronization
_netcdf_lock = threading.Lock()  # Protects HDF5/NetCDF4 I/O (not thread-safe)


def _log_none_access(attr_name, wcop_desc="Multisite"):
    """Log when a None-check block is entered for a surgical-exclusion attribute.

    This helps identify if excluded attributes are unexpectedly needed during simulation.
    """
    logger.debug(
        f"{wcop_desc} attribute '{attr_name}' is None. "
        f"This was excluded during worker serialization for memory optimization. "
        f"If this breaks simulation, the attribute must be retained."
    )


# from dask.distributed import Client

# client = Client(
#     n_workers=cop_conf.n_nodes,
#     threads_per_worker=2,
#     memory_limit=cop_conf.memory_limit,
# )
# import webbrowser

# webbrowser.open(client.cluster.dashboard_link)

SimResult = namedtuple("SimResult", ["sim_sea", "sim_trans", "rphases"])

# Disable parallel loading by default to avoid dask dependency
# Set WEATHERCOP_PARALLEL_LOADING=1 to enable (requires dask)
parallel_loading = os.environ.get("WEATHERCOP_PARALLEL_LOADING", "0") != "0"

# Build mf_kwds, only including chunks if parallel loading enabled (dask required)
mf_kwds = dict(
    concat_dim="realization",
    combine="nested",
    # data_vars="minimal",
    # coords="minimal",
    # compat="override",
    # see https://github.com/pydata/xarray/issues/7079
    parallel=parallel_loading,
)
# Only add chunks when parallel loading is enabled
if parallel_loading:
    mf_kwds["chunks"] = dict(realization=5)


def set_conf(conf_obj, **kwds):
    """Apply DWD configuration to VarWG modules.

    Validates that the configuration is properly applied to all VarWG
    modules that cache it. This is critical for tox and other contexts
    where import order may vary.
    """
    objs = (varwg, vg_core, vg_base, vg_plotting)
    for obj in objs:
        obj.conf = conf_obj
        for key, value in kwds.items():
            setattr(obj.conf, key, value)

    # Defensive check: verify config was actually set
    if varwg.conf is not conf_obj:
        warnings.warn(
            f"VarWG configuration may not be properly applied. "
            f"Expected {conf_obj}, got {varwg.conf}"
        )


def pickle_filepath(xds, rain_method="simulation", warn=True):
    start_str_data = str(xds.time[0].dt.strftime("%Y-%m-%d").data)
    stop_str_data = str(xds.time[-1].dt.strftime("%Y-%m-%d").data)
    station_names = station_names = xds.coords["station"].values
    station_names_str = "_".join(sorted(station_names))
    ms_filename = (
        f"{station_names_str}_D_{rain_method}_"
        f"{start_str_data}_{stop_str_data}.pkl"
    )
    ms_filepath = cop_conf.cache_dir / ms_filename
    if warn and not ms_filepath.exists():
        warnings.warn(f"{ms_filepath} does not exist")
    return ms_filepath


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
        filepath_daily,
        filepath_trans,
        filepath_rphases,
        filepath_rphases_src,
        sim_args,
        sim_kwds,
        dis_kwds,
        sim_times,
        csv,
        ensemble_dir,
        n_digits,
        write_to_disk,
        verbose,
        conversions,
    ) = args
    varwg.reseed((1000 * real_i))
    # sim_sea, sim_trans = simulate(wcop, sim_times, *sim_args, **sim_kwds)
    if filepath_rphases_src:
        rphases = np.load(filepath_rphases_src.with_suffix(".npy"))
    else:
        rphases = None
    return_trans = filepath_trans is not None
    sim_result = simulate(
        wcop,
        sim_times,
        rphases=rphases,
        vgs=wcop.vgs,
        *sim_args,
        **sim_kwds,
    )
    sim_sea = sim_result.sim_sea
    if dis_kwds is not None:
        if write_to_disk:
            with _netcdf_lock:
                sim_result.sim_sea.to_netcdf(filepath_daily)
        varwg.reseed((1000 * real_i))
        if DEBUG:
            print(current_process().name + " in sim_one. before disagg")
        sim_sea_dis = wcop.disaggregate(**dis_kwds)
        if DEBUG:
            print(current_process().name + " in sim_one. after disagg")
        sim_sea, sim_sea_dis = xr.align(
            sim_result.sim_sea, sim_sea_dis, join="outer"
        )
        sim_sea.loc[dict(variable=wcop.varnames)] = sim_sea_dis.sel(
            variable=wcop.varnames
        )
    if write_to_disk:
        if DEBUG:
            print(current_process().name + " in sim_one. before to_netcdf")
        with _netcdf_lock:
            sim_sea.to_netcdf(filepath)
            if return_trans and sim_result.sim_trans is not None:
                sim_result.sim_trans.to_netcdf(filepath_trans)
        if DEBUG:
            print(current_process().name + " in sim_one. after to_netcdf")
        if csv:
            real_str = f"real_{real_i:0{n_digits}}"
            csv_path = ensemble_dir / "csv" / real_str
            wcop.to_csv(
                csv_path, sim_result.sim_sea, filename_prefix=f"{real_str}_"
            )
        # Use rphases from sim_result instead of reading from shared wcop._rphases
        np.save(filepath_rphases, sim_result.rphases)
    return real_i


def _adjust_fft_sim(
    fft_sim,
    # qq_mean,
    # qq_std,
    sc_pars,
    primary_var_ii,
    phase_randomize_vary_mean,
    adjust_prim=True,
    usevine=False,
):

    # fft_sim *= (qq_std / fft_sim.std(axis=1))[:, None]
    # fft_sim += (qq_mean - fft_sim.mean(axis=1))[:, None]

    # fft_sim /= fft_sim.std(axis=1)[:, None]
    # fft_sim -= fft_sim.mean(axis=1)[:, None]

    if phase_randomize_vary_mean:
        # allow means to vary. let it flow from the central_node to
        # the others
        K = fft_sim.shape[0]
        mean_eps = np.zeros(K)[:, None]
        mean_eps[primary_var_ii[0], 0] = (
            phase_randomize_vary_mean * varwg.get_rng().normal()
        )
        fft_sim += mean_eps

    if adjust_prim:
        # change in mean scenario
        prim_i = tuple(primary_var_ii)
        fft_sim[prim_i] += sc_pars.m[prim_i]
        fft_sim[prim_i] += sc_pars.m_t[prim_i]
        T = fft_sim.shape[1]
        fft_sim[prim_i] += (
            np.arange(T, dtype=float) / T * sc_pars.m_trend[prim_i]
        )
    return fft_sim


def _debias_fft_sim(sim, fft_dist, qq_dist):
    return np.array(
        [
            qq_dist[varname].ppf(fft_dist[varname].cdf(sim))
            for varname, sim in zip(fft_dist.keys(), sim)
        ]
    )


def _debias_ranks_sim(sim, sim_dist):
    return np.array(
        [
            sim_dist[varname].cdf(sim)
            # sim_dist[varname].ppf(sim)
            for varname, sim in zip(sim_dist.keys(), sim)
        ]
    )


def _transform_sim(data_trans, data_trans_dist):
    return np.array(
        [
            data_trans_dist[varname].cdf(data)
            for varname, data in zip(data_trans_dist.keys(), data_trans)
        ]
    )


def _retransform_sim(ranks_sim, data_trans_dist):
    return np.array(
        [
            data_trans_dist[varname].ppf(ranks)
            for varname, ranks in zip(data_trans_dist.keys(), ranks_sim)
        ]
    )


def _rename_by_conversions(conversions, varnames_old, sim_sea, data_trans):
    # need some dumvg.helpers data
    times_ = sim_sea.time[:2]
    data = sim_sea.isel(station=0, time=slice(2))
    for conversion in conversions:
        # first, try whether the function exposes the renaming
        if name_conv := getattr(conversion, "name_conv", False):
            varnames_new = [
                (name_conv[name_old] if name_old in name_conv else name_old)
                for name_old in varnames_old
            ]
        else:
            varnames_new = conversion(times_, data, varnames_old)[-1]
        nonequal_elements = [
            1
            for varname_old, varname_new in zip(varnames_old, varnames_new)
            if varname_old != varname_new
        ]
        if len(nonequal_elements):
            sim_sea = sim_sea.assign_coords(
                new_coords := {"variable": varnames_new}
            )
            data_trans = data_trans.assign_coords(new_coords)
    return sim_sea, data_trans


def simulate(
    wcop,
    sim_times,
    *args,
    phase_randomize=True,
    phase_randomize_vary_mean=0.25,
    stop_at=None,
    mask_fun=None,
    rphases=None,
    return_rphases=False,
    return_trans=False,
    heat_wave_kwds=None,
    usevg=False,
    usevine=True,
    conversions=None,
    vgs=None,
    **kwds,
):
    """this is like Multisite.simulate, but simplified for multiprocessing.

    Parameters
    ----------
    vgs : dict, optional
        Thread-local VG objects. If None, uses wcop.vgs (default behavior).
        When called from worker threads, this should be the thread-local copy
        to avoid shared state issues.
    """
    # allow for site-specific theta_incr
    theta_incrs = kwds.pop("theta_incr", None)
    sim_sea = sim_trans = None
    # sim_sea = None
    T_sim = len(sim_times)

    # Use passed-in vgs if available, otherwise fall back to wcop.vgs
    if vgs is None:
        vgs = wcop.vgs

    vg_obj = list(vgs.values())[0]
    T_data = vg_obj.T_summed
    K = wcop.K
    station_names = wcop.station_names
    n_stations = len(station_names)
    varnames = wcop.varnames
    primary_var = wcop.primary_var
    # Don't write to shared wcop.usevine - pass as parameter instead to avoid race condition
    # with _wcop_lock:
    #     wcop.usevine = usevine

    primary_var_sim = kwds.pop("primary_var", primary_var)
    if rphases is None:
        rphases = _random_phases(K, T_sim, T_data, mask_fun=mask_fun)

    # fft_sim = xr.DataArray(
    #     np.full((n_stations, K, T_sim), 0.5),
    #     coords=[
    #         station_names,
    #         varnames,
    #         vg_obj.sim_times,
    #     ],
    #     dims=["station", "variable", "time"],
    # )
    # for station_name_ in station_names:
    #     As = As.sel(station=station_name_, drop=True)
    #     fft_sim = np.concatenate(
    #         [
    #             np.fft.ifft(As * np.exp(1j * rphases_)).real
    #             for rphases_ in rphases
    #         ],
    #         axis=1,
    #     )[:, : vg_obj.T_sim]
    #     fft_sim = _adjust_fft_sim(
    #         fft_sim,
    #         self.qq_means.sel(station=station_name_).data,
    #         self.qq_stds.sel(station=station_name_).data,
    #         sc_pars,
    #         vg_obj.primary_var_ii,
    #         self.phase_randomize_vary_mean,
    #         adjust_prim=(primary_var_sim == self.primary_var),
    #     )
    #     self.fft_sim.loc[dict(station=station_name_)] = fft_sim
    # if heat_wave_kwds is not None:
    #     self.heat_waves = wplt.HeatWavesN(
    #         self.fft_sim.sel(variable=primary_var_sim).mean("station"),
    #         **heat_wave_kwds,
    #     )

    # Capture usevine in partial to avoid race condition
    vg_ph = partial(
        _vg_ph,
        rphases=rphases,
        wcop=wcop,
        usevine=usevine,  # Pass usevine as parameter, not via shared wcop
        # phase_randomize=phase_randomize,
        phase_randomize_vary_mean=phase_randomize_vary_mean,
    )
    for station_i, (station_name, svg) in enumerate(vgs.items()):
        try:
            theta_incr = theta_incrs[station_name]
        except (KeyError, TypeError, IndexError):
            theta_incr = theta_incrs
        if DEBUG:
            print("in sim_one ", heat_wave_kwds)
        # When usevine=True, theta_incr is applied directly by VarWG after callback,
        # so we pass it to VarWG but don't apply it through sc_pars.m in _adjust_fft_sim
        sim_returned = svg.simulate(
            sim_func=None if usevg else vg_ph,
            primary_var=primary_var,
            theta_incr=theta_incr,
            phase_randomize=phase_randomize,
            phase_randomize_vary_mean=phase_randomize_vary_mean,
            sim_func_kwds=dict(
                stop_at=stop_at,
                wcop=wcop,
                return_rphases=return_rphases,
                primary_var_sim=primary_var_sim,
                heat_wave_kwds=heat_wave_kwds,
            ),
            *args,
            **kwds,
        )
        if return_rphases:
            _, sim, ranks_sim, rphases = sim_returned
        elif usevg:
            _, sim = sim_returned
        else:
            _, sim, ranks_sim = sim_returned
        if station_i == 0:
            sim_sea = xr.DataArray(
                np.empty((n_stations, K, T_sim)),
                coords=[station_names, varnames, sim_times],
                dims=["station", "variable", "time"],
            )
            sim_trans = xr.full_like(sim_sea, np.nan)
        sim_sea.loc[dict(station=station_name)] = sim
        sim_trans.loc[dict(station=station_name)] = svg.sim
    if conversions:
        sim_sea, sim_trans = _rename_by_conversions(
            conversions, varnames, sim_sea, sim_trans
        )
    return SimResult(sim_sea, rphases=rphases, sim_trans=sim_trans)


def _vg_ph(
    vg_obj,
    sc_pars,
    wcop=None,
    rphases=None,
    phase_randomize_vary_mean=0.25,
    stop_at=None,
    return_rphases=False,
    primary_var_sim=None,
    heat_wave_kwds=None,
    usevine=True,  # Pass as parameter to avoid race condition
):
    """Call-back function for VG.simulate. Replaces a time_series_analysis
    model.

    Simlified version of Multisite._vg_ph for multiprocessing

    Parameters
    ----------
    usevine : bool
        Whether to use vine copulas. This is passed as a parameter (captured
        in the partial function) instead of reading from wcop.usevine to
        avoid race conditions in threaded execution.
    """
    # Each thread has its own VG copy via initializer, no lock needed
    station_name = vg_obj.station_name
    primary_var = vg_obj.primary_var
    primary_var_ii = vg_obj.primary_var_ii
    T_sim = vg_obj.T_sim
    sim_times = vg_obj.sim_times
    # As = wcop.As.sel(station=station_name, drop=True)
    As = wcop.As
    fft_dist = wcop.fft_dists[station_name]
    qq_dist = wcop.qq_dists[station_name]
    data_trans_dist = wcop.data_trans_dists[station_name]
    # Use parameter instead of reading from shared wcop to avoid race condition
    # usevine = wcop.usevine
    varnames_wcop = (
        wcop.varnames
    )  # Read wcop attributes at start to minimize shared access
    if usevine:
        varnames_vine = wcop.vine.varnames
        sim_dist = wcop.sim_dists[station_name]
    zero_phases = wcop.zero_phases
    vine = wcop.vine
    heat_waves = wcop.heat_waves
    As = As.sel(station=station_name, drop=True)
    if primary_var_sim is None:
        primary_var_sim = primary_var

    # adjust zero-phase
    phases = []
    for phase_ in rphases:
        phase_[:, 0] = zero_phases[station_name]
        phases += [phase_]
    rphases = phases

    fft_sim = np.concatenate(
        [np.fft.ifft(As * np.exp(1j * phases_)).real for phases_ in rphases],
        axis=1,
    )[:, :T_sim]

    # debias fft sim
    fft_sim = _debias_fft_sim(fft_sim, fft_dist, qq_dist)

    # add scenario perturbations
    fft_sim = _adjust_fft_sim(
        fft_sim,
        # qq_mean,
        # qq_std,
        sc_pars,
        primary_var_ii,
        phase_randomize_vary_mean,
        adjust_prim=(primary_var_sim == primary_var),
        usevine=usevine,
    )

    if usevine:
        # qq = dists.norm.cdf(fft_sim)
        qq = np.array(
            [
                fft_dist[varname].cdf(sim)
                for varname, sim in zip(fft_dist.keys(), fft_sim)
            ]
        )
        if primary_var_sim != primary_var:
            primary_var_sim_i = varnames_wcop.index(primary_var_sim)

            # heat wave experimentation
            # get heat waves and make them worse
            if DEBUG:
                print(current_process().name + " in _vg_ph. before heatwave")
            if heat_wave_kwds is not None:
                sim_prim_xr = xr.DataArray(
                    fft_sim[primary_var_sim_i], coords=dict(time=sim_times)
                )
                # heat_waves = wplt.heatwaves_kuglitsch(sim_prim_xr)
                # heat_waves = wplt.HeatWavesN(sim_prim_xr, n_per_summer=3)
                if DEBUG:
                    print(
                        f"Adding heat waves "
                        f"(n_days={heat_waves.mask.sum().values}, "
                        f"diff={heat_wave_kwds['diff']})"
                    )
                sim_prim_xr.values[heat_waves.mask] += heat_wave_kwds["diff"]
                print(qq[primary_var_sim_i, heat_waves.mask].mean())
                qq[primary_var_sim_i] = dists.norm.cdf(sim_prim_xr.values)
                print(qq[primary_var_sim_i, heat_waves.mask].mean())
            if DEBUG:
                print(current_process().name + " in _vg_ph. after heatwave")

            # get a baseline U
            T = qq.shape[1]
            trend = (
                np.arange(T, dtype=float)
                / T
                * sc_pars.m_trend[primary_var_sim_i]
            )

            # ranks_sim = vine.simulate(
            #     T=np.arange(T),
            #     randomness=qq,
            #     stop_at=varnames_vine.index(primary_var_sim) + 1,
            # )
            # # ranks_sim = _debias_ranks_sim(ranks_sim, sim_dist)
            # # U_i = dists.norm.cdf(
            # #     dists.norm.ppf(ranks_sim[primary_var_sim_i])
            # #     + (sc_pars.m_t + sc_pars.m)[primary_var_sim_i]
            # #     + trend
            # # )
            # U_i = prim_dist.cdf(
            #     prim_dist.ppf(ranks_sim[primary_var_sim_i])
            #     + (sc_pars.m_t + sc_pars.m)[primary_var_sim_i]
            #     + trend
            # )
            # qq[primary_var_sim_i] = U_i
            prim_dist = sim_dist[primary_var]
            qq[primary_var_sim_i] = prim_dist.cdf(
                prim_dist.ppf(qq[primary_var_sim_i])
                + (sc_pars.m_t + sc_pars.m)[primary_var_sim_i]
                + trend
            )
            ranks_sim = vine.simulate(
                T=np.arange(T),
                randomness=qq,
                stop_at=stop_at,
                primary_var=primary_var_sim,
                # U_i=U_i,
            )
            ranks_sim = _debias_ranks_sim(ranks_sim, sim_dist)
        else:
            # U_i = None
            ranks_sim = vine[station_name].simulate(
                T=np.arange(qq.shape[1]),
                randomness=qq,
                stop_at=stop_at,
            )
            ranks_sim = _debias_ranks_sim(ranks_sim, sim_dist)

        # sim = dists.norm.ppf(ranks_sim)
        # sim_trans = _retransform_sim(ranks_sim, data_trans_dist)
        sim = _retransform_sim(ranks_sim, data_trans_dist)

        if stop_at is None:
            assert np.all(np.isfinite(sim))
        else:
            var_ii = [
                varnames_wcop.index(varname)
                for varname in varnames_vine[:stop_at]
            ]
            assert np.all(np.isfinite(sim[var_ii]))
    else:
        sim = fft_sim
        ranks_sim = dists.norm.cdf(fft_sim)

    if DEBUG:
        print(current_process().name + " in _vh_ph. done.")
    if return_rphases:
        return sim, ranks_sim, rphases
    return sim, ranks_sim


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
            # bounds_error=False,
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
        infilling=None,
        refit_vine=False,
        station_vines=False,
        asymmetry=False,
        rain_method="simulation",
        cop_candidates=None,
        debias=True,
        scop_kwds=None,
        vgs_cache_dir=None,
        vine_cache=None,
        pad_input=True,
        reinitialize_vgs=False,
        conversions=None,
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
        infilling : None or "phase_inflation" or "vg"
            phase_inflation
                Do phase inflation before fitting the vine. This might
                possibly alter the dependency structure, but fills in
                missing values.
            vg
                Use vg's VAR process for infilling.
        pad_hourly : boolean
            Pad missing hours at the beginning of the input.
        """
        self.vgs_cache_dir = (
            cop_conf.vgs_cache_dir if vgs_cache_dir is None else vgs_cache_dir
        )
        self.vine_cache = (
            cop_conf.vine_cache if vine_cache is None else vine_cache
        )
        self.varnames = [
            str(varname) for varname in xds.coords["variable"].data
        ]
        # if "R" is included, put it second for better fitting!
        # if "R" in self.varnames:
        #     varnames_rest = [name for name in self.varnames
        #                      if name not in ("theta", "R")]
        #     self.varnames = ["theta", "R"] + varnames_rest
        #     xds = xds.sel(variable=self.varnames)
        self.station_names = list(xds.coords["station"].data)
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
        self.xar = (
            xds.to_array("station")
            .transpose("station", "variable", "time")
            .sel(station=self.station_names)
        )
        if pad_input:
            self.xar = self.pad_input()
        if isinstance(primary_var, str):
            primary_var = (primary_var,)
        self.primary_var = primary_var
        self.discr = discretization
        self.verbose = verbose
        self.verbose_old = None
        self.phase_inflation = False
        self.vg_infilling = False
        if infilling == "phase_inflation":
            self.phase_inflation = True
        elif infilling == "vg":
            self.vg_infilling = True
        self.asymmetry = asymmetry
        self.rain_method = rain_method
        if cop_candidates is None:
            cop_candidates = cops.all_cops
        self.cop_candidates = cop_candidates
        self.scop_kwds = scop_kwds
        self.conversions = conversions
        # for pickling ease
        self.args, self.kwds = args, kwds

        self.refit_vine = refit_vine
        self.station_vines = station_vines
        self.vgs = OrderedDict()
        self.K = self.n_variables = len(self.varnames)
        # match self.discr:
        #     case "D":
        #         self.sum_interval = 24
        #     case "H":
        #         self.sum_interval = 1
        #     case _:
        #         raise RuntimeError(
        #             f"Discretization '{self.discr}' not supported"
        #         )
        self.n_stations = len(self.station_names)
        self.times = self.xar.time
        self.varnames_refit = self.varnames_dis = self.sim_sea_dis = None
        start_dt, end_dt = pd.to_datetime(self.xar.time.data[[0, -1]])
        self.times_daily = pd.date_range(start_dt, end_dt, freq=self.discr)
        self.dtimes = self.times_daily.to_pydatetime()

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
        self._init_vgs(reinitialize_vgs, *args, **kwds)

        # this is useful from time to time
        # self.data_daily = self.xar.resample(time=self.discr).mean(
        #     dim="time",
        #     # skipna=True
        # )
        self.T_daily = len(self.data_daily.coords["time"])

        nans = self.data_daily.isnull()
        if np.any(nans) and infilling is None:
            print("Nan sums:\n", nans.sum("time").to_pandas())
            raise ValueError("nans in xds. Remove or use an infilling method.")

        # household variables for vines and phase randomization
        self.cop_quantiles = xr.full_like(self.data_trans, np.nan)
        self.fft_sim = None  # xr.full_like(self.data_trans, np.nan)
        self.As = xr.full_like(self.data_trans, np.nan)
        self.rphases = self._rphases = self.zero_phases = self.vine = None
        self.qq_dists, self.fft_dists, self.sim_dists = {}, {}, {}

        # property caches
        self._obs_means = None

        # filled during simulation
        self.sim = None  # xr.full_like(self.data_trans, np.nan)
        self.ranks_sim = None  # xr.full_like(self.sim, np.nan)
        self.sim_sea = self._sim_sea = self._sim_sea_dis = self.ensemble = None
        self.ensemble_dir = self.sc_pars_sum = None
        # for saving expensive heatwave calculation intermediate results
        self._heatwave_bounds = self.heat_waves = None

    def __getstate__(self):
        dict_ = dict(self.__dict__)
        if "vgs" in dict_:
            del dict_["vgs"]

        # If this instance is marked for worker serialization, exclude large attributes
        # to reduce memory footprint when forking processes (avoids deadlock issues)
        exclude_flag = getattr(self, "_exclude_large_attrs_for_workers", False)
        if exclude_flag:
            import sys

            # Only exclude attributes that are NOT used during simulation in workers.
            # Key insight: Workers need fitting results (As, vine, zero_phases, etc.)
            # but not the raw training data or ensemble results.
            excluded_attrs = [
                "xds",  # Original input dataset - not needed by workers
                "data_trans",  # Transformed training data - not needed after fitting
                "ranks",  # Rank data - not needed after fitting
                "data_daily",  # Daily aggregations - used for fitting, not simulation
                "times_daily",  # Daily times - not needed by workers
                "ensemble",  # Ensemble results from realization 0 - not needed
                "ensemble_trans",  # Transformed ensemble results - not needed
                "ensemble_daily",  # Daily ensemble results - not needed
                "sim_sea",  # Simulation results from realization 0 - not needed
                "sim_trans",  # Transformed simulation from realization 0 - not needed
                "rain_mask",  # Intermediate mask - not needed by workers
            ]
            if DEBUG:
                excluded_count = 0
                for attr in excluded_attrs:
                    if attr in dict_:
                        dict_.pop(attr)
                        excluded_count += 1
                print(
                    f"DEBUG: __getstate__ excluded {excluded_count} attributes",
                    file=sys.stderr,
                    flush=True,
                )

        return dict_

    def __setstate__(self, dict_):
        self.__dict__ = dict_
        self.vgs = OrderedDict()
        self.vgs_cache_dir = cop_conf.vgs_cache_dir
        self.vine_cache = cop_conf.vine_cache
        self._init_vgs(*self.args, **self.kwds)

    def __enter__(self):
        """Context manager entry: return self for use in with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close all ensemble datasets."""
        self.close()
        return False  # Don't suppress exceptions

    def close(self):
        """Close all xarray ensemble datasets to release file handles and memory.

        Safely handles cases where ensemble_daily and ensemble are the same
        object (when disaggregate=False).
        """
        closed_ids = set()

        for attr_name in ["ensemble_trans", "ensemble_daily", "ensemble"]:
            if (obj := getattr(self, attr_name, None)) is None:
                continue

            obj_id = id(obj)
            if obj_id in closed_ids or not hasattr(obj, "close"):
                continue

            try:
                obj.close()
                closed_ids.add(obj_id)
            except Exception as e:
                import warnings

                warnings.warn(f"Failed to close {attr_name}: {e}")

    def __eq__(self, other):
        if not hasattr(other, "__getitem__"):
            return False
        for name, svg in self.items():
            svg_other = other.get(name)
            if svg != svg_other:
                return False

    def save(self):
        filename = (
            "_".join(sorted(self.station_names) + [self.identifier]) + ".pkl"
        )
        filepath = cop_conf.cache_dir / filename
        if self.verbose:
            print(f"Pickling {self} to:\n{filepath}")
        with open(filepath, "wb") as fi:
            dill.dump(self, fi)
        return filepath

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as fi:
            return dill.load(fi)

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
            # call the wplt method on all vgs
            def meta_plot(*args, **kwds):
                returns = {}
                station_names = kwds.pop("stations", self.vgs.keys())
                fig_axs = kwds.pop("fig_axs", None)
                if isinstance(station_names, str):
                    station_names = (station_names,)
                stations = {name: self.vgs[name] for name in station_names}
                for station_name, svg in stations.items():
                    longitude = float(
                        self.longitudes.sel(station=station_name)
                    )
                    latitude = float(self.latitudes.sel(station=station_name))
                    svg._conf_update(
                        dict(longitude=longitude, latitude=latitude)
                    )
                    if fig_axs is not None:
                        fig, axs = fig_axs[station_name]
                    else:
                        fig, axs = None, None
                    fig, axs = getattr(svg, name)(
                        figs=fig, axss=axs, *args, **kwds
                    )
                    for fig_ in np.atleast_1d(fig):
                        wplt.suptitle_prepend(fig_, station_name)
                    returns[station_name] = fig, axs
                return returns

            return meta_plot
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
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

    def pad_input(self):
        start_hour = int(self.xar.coords["time"][0].dt.hour)
        if start_hour:
            xar_pad = self.xar.isel(time=slice(24 - start_hour, 24))
            xar_pad.coords["time"] = xar_pad.time - pd.Timedelta(days=1)
            return xr.concat(
                (xar_pad, self.xar),
                dim="time",
            )
        else:
            if self.verbose:
                print("No need to pad input")
            return self.xar

    def _init_vgs(self, reinitialize_vgs=False, *args, **kwds):
        """Initiate VG instances to get transformed data."""
        # gather transformed data from all vg instances
        data_trans = (
            xr.full_like(self.xar, 0)
            .resample(time=self.discr)
            .mean(dim="time")
        )
        self.data_trans_dists = {}
        data_daily = xr.full_like(data_trans, np.nan)
        self.ranks = xr.full_like(data_trans, np.nan)
        if "R" in self.varnames:
            self.rain_mask = xr.full_like(
                self.ranks.sel(variable="R"), False, dtype=bool
            )
        refit = "refit" in kwds
        if refit and isinstance(kwds["refit"], Iterable):
            # refit_vars = (set(kwds.get("refit", []))
            #               & set(self.varnames))
            refit_vars = self.varnames_refit
            refit_stations = set(kwds.get("refit", [])) & set(
                self.station_names
            )
        else:
            refit_vars = refit_stations = []
        for station_name in self.station_names:
            self.data_trans_dists[station_name] = {}
            cache_file = self.cache_file(station_name)
            (self.vgs_cache_dir / station_name).mkdir(
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
                and not reinitialize_vgs
            ):
                if self.verbose:
                    print(
                        f"Recovering VG instance of {station_name} "
                        f"from:\n{cache_file}"
                    )
                with cache_file.open("rb") as fi:
                    svg = dill.load(fi)
                    svg._conf_update(
                        dict(
                            longitude=longitude,
                            latitude=latitude,
                            seasonal_cache_file=seasonal_cache_file,
                            cache_dir=cache_file.parent,
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
                cache_dir = str(self.vgs_cache_dir / station_name)
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
                if self.conversions:
                    for forward, _ in self.conversions:
                        _times, data, varnames = forward(
                            data.index, data.values, list(data.columns)
                        )
                    data = pd.DataFrame(
                        index=_times, data=data, columns=varnames
                    )
                svg = varwg.VG(
                    list(data.columns),
                    met_file=data,
                    cache_dir=cache_dir,
                    # we do not need it and don't want it to
                    # be read from the configuration file
                    data_dir="",
                    conf_update=conf_update,
                    dump_data=False,
                    # sum_interval=self.sum_interval,
                    station_name=station_name,
                    rain_method=self.rain_method,
                    infill=self.vg_infilling,
                    *args,
                    **kwds_,
                )
                with cache_file.open("wb") as fi:
                    dill.dump(svg, fi)
            self.vgs[station_name] = svg
            if "R" in self.varnames:
                self.rain_mask.loc[dict(station=station_name)] = svg.rain_mask
            for var_i, varname in enumerate(self.varnames):
                data_trans_ = svg.data_trans[var_i]
                data_trans.loc[
                    dict(station=station_name, variable=varname)
                ] = data_trans_
                # self.data_trans_dists[station_name][varname] = ECDF(
                #     data_trans_, data_min=-9, data_max=9
                # )
                self.data_trans_dists[station_name][varname] = dists.norm(
                    *dists.norm.fit(data_trans_)
                )
                # also replace the observations with the infilled data
                data_daily.loc[
                    dict(station=station_name, variable=varname)
                ] = (svg.data_raw[var_i] / svg.sum_interval[var_i, 0])
        # this sets verbosity on all vg instances
        self.verbose = self.verbose
        if self.phase_inflation:
            if self.verbose:
                print("Phase inflation")
            data_trans = self._phase_inflation(data_trans)
        elif self.vg_infilling:
            if self.verbose:
                print("Infilling nans with vg's VAR process.")
            data_trans = self._vg_infilling(data_trans)
        else:
            data_trans = data_trans.interpolate_na("time")
        self.data_trans = data_trans
        self.data_daily = data_daily

        # Check for all-NaN station-variable pairs (indicator of VarWG failures)
        all_nan_pairs = []
        for station_name in self.station_names:
            for var_i, varname in enumerate(self.varnames):
                data = self.data_trans.sel(
                    station=station_name, variable=varname
                ).values
                if np.isnan(data).all():
                    all_nan_pairs.append((station_name, varname))

        if all_nan_pairs:
            error_msg = (
                "Found all-NaN transformed data (data_trans) for the following "
                "station/variable pairs. This indicates a VarWG marginal "
                "transformation failure and needs investigation:\n"
            )
            for station_name, varname in all_nan_pairs:
                error_msg += (
                    f"  - Station: {station_name}, Variable: {varname}\n"
                )
            error_msg += (
                "\nThis is a serious error that requires fixing in VarWG, not "
                "masking in WeatherCop. Please investigate the VarWG "
                "transformation for these variables."
            )
            raise RuntimeError(error_msg)

        # ranks = dists.norm.cdf(self.data_trans)
        ranks = xr.full_like(self.data_trans, np.nan)
        for station_name in self.station_names:
            ranks.loc[dict(station=station_name)] = _transform_sim(
                self.data_trans.sel(station=station_name),
                self.data_trans_dists[station_name],
            )
        ranks.data[(ranks.data <= 0) | (ranks.data >= 1)] = np.nan

        self.ranks.values = ranks
        self.ranks = self.ranks.interpolate_na(dim="time")

        # Fallback for boundary/sparse NaNs
        if np.isnan(self.ranks.values).any():
            self.ranks = self.ranks.bfill(dim="time").ffill(dim="time")

        assert np.all(
            np.isfinite(self.ranks.values)
        ), f"Ranks contain NaN values: {np.isnan(self.ranks.values).sum()} NaNs"
        if not self.station_vines:
            # reorganize so that variable dependence does not consider
            # inter-site relationships
            self.ranks = self.ranks.stack(rank=("station", "time"))
            if "R" in self.varnames:
                self.rain_mask = self.rain_mask.stack(rank=("station", "time"))

    @property
    def identifier(self):
        return "_".join(
            (self.discr, self.rain_method, self.start_str, self.end_str)
        )

    def cache_file(self, station_name):
        return (
            self.vgs_cache_dir
            / station_name
            / f"{station_name}_{self.identifier}.pkl"
        )

    def _clear_vgs_cache(self, station_names=None):
        if station_names is None:
            station_names = self.station_names
        for station_name in station_names:
            cache_file = self.cache_file(station_name)
            if cache_file.exists():
                cache_file.unlink()

    def _phase_inflation(self, data_ar):
        varwg.reseed((0))
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
        phases[ii] = varwg.get_rng().uniform(0, np.pi, len(ii))

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

    def _vg_infilling(self, data_trans):
        for station_name, svg in self.vgs.items():
            if self.verbose:
                print(station_name)
            data_trans.loc[dict(station=station_name)] = (
                svg.infill_trans_nans()
            )
        return data_trans

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
        usevg=False,
        phase_randomize=True,
        phase_randomize_vary_mean=0.25,
        write_to_disk=True,
        return_trans=False,
        stop_at=None,
        mask_fun=None,
        rphases=None,
        return_rphases=False,
        heat_wave_kwds=None,
        **kwds,
    ):
        self.usevine = usevine
        self.phase_randomize_vary_mean = phase_randomize_vary_mean
        self.write_to_disk = write_to_disk
        # allow for site-specific theta_incr
        theta_incrs = kwds.pop("theta_incr", None)
        primary_var_sim = kwds.pop("primary_var", self.primary_var)
        # self.sim_trans = xr.full_like(self.sim_sea, np.nan)
        self.sim_sea = self.sim_trans = None
        for station_i, (station_name, svg) in enumerate(self.vgs.items()):
            if self.verbose:
                print(f"Simulating {station_name}{' (vg)' if usevg else ''}")
                svg.verbose = False
            if theta_incrs is None:
                theta_incr = None
            elif isinstance(theta_incrs, dict):
                theta_incr = theta_incrs[station_name]
            else:
                theta_incr = theta_incrs
            # When usevine=True, theta_incr is applied directly by VarWG after callback,
            # so we pass it to VarWG but don't apply it through sc_pars.m in _adjust_fft_sim

            sim_returned = svg.simulate(
                sim_func=None if usevg else self._vg_ph,
                primary_var=self.primary_var,
                # primary_var=primary_var_sim,
                theta_incr=theta_incr,
                phase_randomize=phase_randomize,
                phase_randomize_vary_mean=phase_randomize_vary_mean,
                sim_func_kwds=dict(
                    stop_at=stop_at,
                    mask_fun=mask_fun,
                    rphases=rphases,
                    wcop=self,
                    return_rphases=return_rphases,
                    primary_var_sim=primary_var_sim,
                    heat_wave_kwds=heat_wave_kwds,
                ),
                *args,
                **kwds,
            )
            if return_rphases:
                sim_times, sim_sea, ranks_sim, rphases = sim_returned
            elif usevg:
                sim_times, sim_sea = sim_returned
            else:
                sim_times, sim_sea, ranks_sim = sim_returned
            if self.conversions:
                for _, backward in self.conversions:
                    sim_sea = backward(None, sim_sea, svg.var_names)[1]
            if station_i == 0:
                self.sim_sea = xr.DataArray(
                    np.empty((self.n_stations, self.K, svg.T_sim)),
                    coords=[self.station_names, self.varnames, sim_times],
                    dims=["station", "variable", "time"],
                )
            self.sim_sea.loc[dict(station=station_name)] = sim_sea
            if self.verbose:
                self.print_means(station_name)
                print()
            if station_i == 0:
                self.sim_trans = xr.full_like(self.sim_sea, np.nan)
            self.sim_trans.loc[dict(station=station_name)] = svg.sim
            if station_i == 0:
                self.ranks_sim = xr.full_like(self.sim_sea, 0.5)
            if usevg:
                self.ranks_sim.loc[dict(station=station_name)] = (
                    dists.norm.cdf(svg.sim)
                )
                continue
            self.ranks_sim.loc[dict(station=station_name)] = ranks_sim
        # # in case we made conversions on the data, make sure to rename
        # # variables if necessary
        # if conversions := kwds.get("conversions", False):
        #     self.sim_sea, self.data_trans = _rename_by_conversions(
        #         conversions, self.varnames, self.sim_sea, self.data_trans
        #     )
        self.T_sim = svg.T_sim
        # the following two lines ensure that MultisiteConditional works!
        # make chosen phases available...
        self._rphases = self.rphases
        # ... but reset them also so we get new ones next time
        self.rphases = None
        if self.verbose:
            self.print_all_means()
        return SimResult(
            self.sim_sea, rphases=rphases, sim_trans=self.sim_trans
        )

    @property
    def sim_sea(self):
        return self._sim_sea

    @sim_sea.setter
    def sim_sea(self, xar):
        if xar is None:
            return
        self._sim_sea = xar
        times = pd.to_datetime(xar.time)
        for svg in self:
            svg.sim_times = times
            svg.sim_sea = xar.sel(station=svg.station_name).load().values

    @property
    def sim_sea_dis(self):
        return self._sim_sea_dis

    @sim_sea_dis.setter
    def sim_sea_dis(self, xar):
        if xar is None:
            return
        self._sim_sea_dis = xar
        times = pd.to_datetime(xar.time)
        for svg in self:
            svg.dis_sim_times = times
            svg.sim_sea_dis = xar.sel(station=svg.station_name).load().values

    def reset_sim(self):
        self.fft_sim = None

    def disaggregate(self, *args, **kwds):
        self.sim_sea_dis = None
        for station_i, (station_name, svg) in enumerate(self.vgs.items()):
            if self.verbose:
                print(f"Disaggregating {station_name}")
            svg.verbose = False
            longitude = float(self.longitudes.sel(station=station_name))
            latitude = float(self.latitudes.sel(station=station_name))
            svg._conf_update(
                dict(
                    longitude=longitude,
                    latitude=latitude,
                    station_name=station_name,
                )
            )
            times_dis, sim_dis = svg.disaggregate(
                latitude=latitude, longitude=longitude, *args, **kwds
            )
            if station_i == 0:
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
        name_derived=None,
        clear_cache=False,
        csv=False,
        dis_kwds=None,
        write_to_disk=True,
        ensemble_root=None,
        *args,
        **kwds,
    ):
        # # If parallel loading is disabled (no dask), default to keeping ensemble in memory
        # # because reading chunked NetCDF files requires dask
        # if not parallel_loading and write_to_disk is True:
        #     write_to_disk = False

        # TODO: the first realization is weird, so omit it for now
        n_realizations += 1
        if dis_kwds is not None:
            disaggregate = True
            name = f"{name}_disag"
        else:
            disaggregate = False
        # ensure those are written. they might be needed for derived scenarios
        # Only return transformed data if not in test/memory-constrained mode
        if not cop_conf.SKIP_INTERMEDIATE_RESULTS_TESTING:
            kwds.update(return_trans=True)
        ensemble_root = (
            cop_conf.ensemble_root if ensemble_root is None else ensemble_root
        )
        self.ensemble_dir = ensemble_dir = ensemble_root / name
        if clear_cache and ensemble_dir.exists():
            shutil.rmtree(ensemble_dir)
        ensemble_dir.mkdir(exist_ok=True, parents=True)
        if self.verbose:
            print(f"Starting Ensemble Simulations for {name}")
            print(f"Will save nc-files here:\n{ensemble_dir}")
        # silence the vgs for the moment, so we can get a nice
        # uninterrupted progress bar
        self.verbose_old = self.verbose
        self.write_to_disk = write_to_disk
        self.verbose = False
        filepaths = [
            ensemble_dir / f"{name}_real_{real_i:0{cop_conf.n_digits}}.nc"
            for real_i in range(n_realizations)
        ]
        # filepath_rphases = filepaths[0].parent / (
        #     filepaths[0].stem + "_phases"
        # )
        filepaths_rphases = [
            filepath.parent / (filepath.stem + "_phases.npy")
            for filepath in filepaths
        ]
        if name_derived:
            ensemble_dir_derived = ensemble_root / name_derived
            if not ensemble_dir_derived.exists():
                name_derived = (
                    f"{name_derived}_disag"
                    if not name_derived.endswith("_disag")
                    else name_derived
                )
                ensemble_dir_derived = ensemble_root / name_derived
            filepaths_rphases_src = [
                ensemble_dir_derived
                / (name_derived + filepath.stem[len(name) :] + "_phases.npy")
                for filepath in filepaths
            ]
            if not filepaths_rphases_src[0].exists():
                raise RuntimeError(
                    f"Could not find phases here: {filepaths_rphases_src[0]}"
                )
        if disaggregate:
            # keep the daily values as well!
            filepaths_daily = [
                ensemble_dir
                / f"{name}_real_daily_{real_i:0{cop_conf.n_digits}}.nc"
                for real_i in range(n_realizations)
            ]
        else:
            filepaths_daily = filepaths

        filepaths_trans = [
            filepath.parent / (filepath.stem + "_trans.nc")
            for filepath in filepaths
        ]
        if cop_conf.PROFILE:
            for real_i in tqdm(range(n_realizations)):
                filepath = filepaths[real_i]
                if filepath.exists() and filepath.stat().st_size:
                    continue
                filepath_trans = filepaths_trans[real_i]
                varwg.reseed((1000 * real_i))
                if name_derived:
                    rphases = np.load(filepaths_rphases_src[real_i])
                else:
                    rphases = None
                sim_result = self.simulate(rphases=rphases, *args, **kwds)
                # don't recalibrate the resampler after the first simulation
                if "res_kwds" in kwds:
                    if "recalibrate" in kwds:
                        kwds["res_kwds"]["recalibrate"] = False
                sim_sea = sim_result.sim_sea
                assert np.all(np.isfinite(sim_sea))
                # sim_trans = self.sim_trans
                if disaggregate:
                    filepath_daily = filepaths_daily[real_i]
                    if write_to_disk:
                        sim_sea.to_netcdf(filepath_daily)
                    sim_sea_dis = self.disaggregate(**dis_kwds)
                    sim_sea, sim_sea_dis = xr.align(
                        sim_sea, sim_sea_dis, join="outer"
                    )
                    sim_sea.loc[dict(variable=self.varnames)] = (
                        sim_sea_dis.sel(variable=self.varnames)
                    )
                if write_to_disk:
                    sim_sea.to_netcdf(filepath)
                    if csv:
                        real_str = f"real_{real_i:0{cop_conf.n_digits}}"
                        csv_path = ensemble_dir / "csv" / real_str
                        self.to_csv(
                            csv_path, sim_sea, filename_prefix=f"{real_str}_"
                        )
                    np.save(filepaths_rphases[real_i], self.rphases)
                    if (
                        sim_result.sim_trans is not None
                        and not cop_conf.SKIP_INTERMEDIATE_RESULTS_TESTING
                    ):
                        sim_result.sim_trans.to_netcdf(filepath_trans)
        else:
            # this means we do parallel computation
            # do one simulation in the main loop to set up attributes
            varwg.reseed((0))
            if (
                name_derived
                and filepaths_rphases_src[0].with_suffix(".npy").exists()
            ):
                rphases = np.load(filepaths_rphases_src[0].with_suffix(".npy"))
            else:
                rphases = None
            sim_result = self.simulate(rphases=rphases, *args, **kwds)
            # don't recalibrate the resampler after the first simulation
            if "res_kwds" in kwds:
                if "recalibrate" in kwds:
                    kwds["res_kwds"]["recalibrate"] = False
            sim_sea = sim_result.sim_sea
            np.save(filepaths_rphases[0], self._rphases)
            sim_times = sim_sea.coords["time"].data
            if disaggregate:
                filepath_daily = filepaths_daily[0]
                sim_sea.to_netcdf(filepath_daily)
                sim_sea_dis = self.disaggregate(**dis_kwds)
                sim_sea, sim_sea_dis = xr.align(
                    sim_sea, sim_sea_dis, join="outer"
                )
                sim_sea.loc[dict(variable=self.varnames)] = sim_sea_dis.sel(
                    variable=self.varnames
                )
            if write_to_disk:
                sim_sea.to_netcdf(filepaths[0])
                if csv:
                    real_i = 0
                    real_str = f"real_{real_i:0{cop_conf.n_digits}}"
                    csv_path = ensemble_dir / "csv" / real_str
                    self.to_csv(
                        csv_path, sim_sea, filename_prefix=f"{real_str}_"
                    )
                if (
                    write_to_disk
                    and not cop_conf.SKIP_INTERMEDIATE_RESULTS_TESTING
                ):
                    if sim_result.sim_trans is not None:
                        sim_result.sim_trans.to_netcdf(filepaths_trans[0])
            # filter realizations in advance according to output file
            # existance
            realizations = []
            filepaths_multi = []
            filepaths_multi_daily = []
            filepaths_trans_multi = []
            filepaths_multi_rphases = []
            filepaths_multi_rphases_src = []
            for real_i, filepath, filepath_daily in zip(
                range(1, n_realizations), filepaths[1:], filepaths_daily[1:]
            ):
                realization_missing = not (
                    filepath.exists()
                    and filepath.stat().st_size
                    and filepath_daily.exists()
                    and filepath_daily.stat().st_size
                )
                if realization_missing:
                    realizations += [real_i]
                    filepaths_multi += [filepath]
                    filepaths_multi_daily += [filepath_daily]
                    filepaths_trans_multi += [filepaths_trans[real_i]]
                    filepaths_multi_rphases += [filepaths_rphases[real_i]]
                    filepaths_multi_rphases_src += [
                        filepaths_rphases_src[real_i] if name_derived else None
                    ]
            self.varnames_refit = []
            # Set a flag that the instance should exclude large attributes during pickling
            # This reduces memory footprint when forking processes
            self._exclude_large_attrs_for_workers = True

            # Use multiprocessing for worker isolation (avoids C-library race conditions)
            with Pool(cop_conf.n_nodes) as pool:
                completed_reals = list(
                    tqdm(
                        pool.imap(
                            sim_one,
                            zip(
                                realizations,
                                repeat(len(realizations)),
                                repeat(self),
                                filepaths_multi,
                                filepaths_multi_daily,
                                filepaths_trans_multi,
                                filepaths_multi_rphases,
                                filepaths_multi_rphases_src,
                                repeat(args),
                                repeat(kwds),
                                repeat(dis_kwds),
                                repeat(sim_times),
                                repeat(csv),
                                repeat(ensemble_dir),
                                repeat(cop_conf.n_digits),
                                repeat(write_to_disk),
                                repeat(self.verbose),
                                repeat(kwds.get("conversions", None)),
                            ),
                            chunksize=max(
                                1, len(realizations) // cop_conf.n_nodes
                            ),
                        ),
                        total=len(realizations),
                    )
                )
            assert len(completed_reals) == len(realizations)
        # now go back to the original number of realizations
        n_realizations -= 1
        if "conversions" in kwds:
            self.varnames = [
                str(varname)
                for varname in self.sim_sea.coords["variable"].data
            ]
        # expose the ensemble as a dask array (or in-memory if write_to_disk=False)
        drop_dummy = self.n_stations > 1

        if write_to_disk:
            # Load ensemble from disk files
            try:
                self.ensemble = (
                    xr.open_mfdataset(filepaths[1:], **mf_kwds)
                    .assign_coords(realization=range(n_realizations))
                    .to_array("dummy")
                    .squeeze("dummy", drop=drop_dummy)
                )
            except KeyError as exc:
                print(f"KeyError loading ensemble: {exc}")
                raise
            except OSError as exc:
                print(f"OSError loading ensemble: {exc}")
                raise
            except RuntimeError as exc:
                if exc.__str__() == "NetCDF: Not a valid ID":
                    nc_file = exc.__context__.args[0][1][0]
                    print(f"Removing offending nc-file ({nc_file})")
                    Path(nc_file).unlink()
                raise
        else:
            # Use in-memory ensemble from self.sim_sea
            # Store the current simulation as the ensemble
            self.ensemble = self.sim_sea.expand_dims("realization")

        if disaggregate:
            self.ensemble_daily = (
                # xr.open_mfdataset(filepaths_daily, **mf_kwds)
                xr.open_mfdataset(filepaths_daily[1:], **mf_kwds)
                .assign_coords(realization=range(n_realizations))
                .to_array("dummy")
                .squeeze("dummy", drop=drop_dummy)
            )
        else:
            self.ensemble_daily = self.ensemble
        try:
            self.ensemble_trans = (
                # xr.open_mfdataset(filepaths_trans, **mf_kwds)
                xr.open_mfdataset(filepaths_trans[1:], **mf_kwds)
                .assign_coords(realization=range(n_realizations))
                .to_array("dummy")
                .squeeze("dummy", drop=drop_dummy)
                # .squeeze(drop=drop_dummy)
            )
        except FileNotFoundError:
            print("Could not load transformed ensemble data.")
        except RuntimeError as exc:
            if exc.__str__() == "NetCDF: Not a valid ID":
                nc_file = exc.__context__.args[0][1][0]
                print(f"Removing offending nc-file ({nc_file})")
                Path(nc_file).unlink()
            raise

        self.ensemble.name = name
        self.verbose = self.verbose_old
        if self.verbose:
            for station_name in self.station_names:
                theta_incr = kwds.get("theta_incr", None)
                if theta_incr:
                    theta_incr = (
                        theta_incr
                        if np.isscalar(theta_incr)
                        else theta_incr[station_name]
                    )
                print(
                    f"\n{station_name}"
                    + (f" (theta_incr={theta_incr:.3f})" if theta_incr else "")
                )
                self.print_means(station_name)
            self.print_ensemble_means()
        return self.ensemble

    @classmethod
    def load_ensemble(self, name, n_realizations, **mf_dict):
        # this is because we omit the first realization due to
        # unexplored weirdness
        n_realizations += 1
        self.ensemble_dir = cop_conf.ensemble_root / name
        filepaths = [
            self.ensemble_dir / f"{name}_real_{real_i:0{cop_conf.n_digits}}.nc"
            for real_i in range(n_realizations)
        ]
        # this is because we omit the first realization due to
        # unexplored weirdness
        return (
            # xr.open_mfdataset(filepaths, **mf_dict)
            xr.open_mfdataset(filepaths[1:], **mf_dict)
            .assign_coords(realization=range(n_realizations))
            .to_array("dummy")
            .squeeze("dummy", drop=True)
        )

    def to_csv(self, csv_path, xar=None, filename_prefix=""):
        if xar is None:
            xar = self.sim_sea
        csv_path.mkdir(exist_ok=True, parents=True)
        for station_name in self.station_names:
            if self.varnames_dis is None:
                filename = f"{filename_prefix}{station_name}.csv"
                csv_df = xar.sel(station=station_name).to_pandas()
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

    def print_means(self, station_name, n_eval=100):
        obs = self.obs_means.sel(station=station_name, drop=True).to_dataframe(
            "obs"
        )
        if self.ensemble is not None:
            sim = (
                self.ensemble.sel(station=station_name, drop=True)
                .isel(realization=slice(None, n_eval))
                .mean(["time", "realization"])
                .to_dataframe("sim")
            )
        else:
            sim_sea = (
                self.sim_sea if self.sim_sea_dis is None else self.sim_sea_dis
            )
            sim = (
                sim_sea.sel(station=station_name, drop=True)
                .mean("time")
                .to_dataframe("sim")
            )
        # exclude possible varnames_ext
        # sim = sim.loc[self.varnames]
        diff = pd.DataFrame(
            sim.values - obs.values, index=obs.index, columns=["diff"]
        )
        diff_perc = pd.DataFrame(
            100 * (sim.values - obs.values) / obs.values,
            index=obs.index,
            columns=["diff [%]"],
        )
        print(pd.concat([obs, sim, diff, diff_perc], axis=1).round(3))

    def print_all_means(self, n_eval=100):
        obs = self.obs_means.mean("station").to_dataframe("obs")
        if self.ensemble is not None:
            sim = (
                self.ensemble.isel(realization=slice(None, n_eval))
                .mean(["time", "realization", "station"])
                .to_dataframe("sim")
            )
        else:
            sim_sea = (
                self.sim_sea if self.sim_sea_dis is None else self.sim_sea_dis
            )
            sim = sim_sea.mean(["time", "station"]).to_dataframe("sim")
        # exclude possible varnames_ext
        # sim = sim.loc[self.varnames]
        diff = pd.DataFrame(
            sim.values - obs.values, index=obs.index, columns=["diff"]
        )
        diff_perc = pd.DataFrame(
            100 * (sim.values - obs.values) / obs.values,
            index=obs.index,
            columns=["diff [%]"],
        )
        print(pd.concat([obs, sim, diff, diff_perc], axis=1).round(3))

    def print_ensemble_means(self, n_eval=100):
        obs = self.obs_means.mean("station").to_dataframe("obs")
        if self.ensemble is not None:
            sim = (
                self.ensemble.isel(realization=slice(None, n_eval))
                .mean(["time", "station", "realization"])
                .to_dataframe("sim")
            )
        else:
            sim = self.sim_sea.mean(["time", "station"]).to_dataframe("sim")
        # exclude possible varnames_ext
        # sim = sim.loc[self.varnames]
        diff = pd.DataFrame(
            sim.values - obs.values, index=obs.index, columns=["diff"]
        )
        diff_perc = pd.DataFrame(
            100 * (sim.values - obs.values) / obs.values,
            index=obs.index,
            columns=["diff [%]"],
        )
        print(pd.concat([obs, sim, diff, diff_perc], axis=1).round(3))

    def _gen_vine_key(self, vg_obj, weights, station_names):
        if isinstance(station_names, str):
            station_str = station_names
        else:
            station_str = "_".join(sorted(station_names))
        return "_".join(
            (
                "_".join(sorted(vg_obj.var_names)),
                station_str,
                "_".join(str(i) for i in self.data_daily.data.shape),
                weights,
                # vg_obj.primary_var[0],
                # self.primary_var,
                (
                    "None"
                    if self.primary_var is None
                    else "_".join(sorted(self.primary_var))
                ),
                self.discr,
                self.rain_method,
            )
        )

    def _fit_vine_all(self, vg_obj, weights):
        with tools.shelve_open(self.vine_cache) as sh:
            key = self._gen_vine_key(vg_obj, weights, self.station_names)
            vine = None
            if key in sh and not (self.refit or self.refit_vine):
                try:
                    vine = sh[key]
                except EOFError:
                    # we have to refit
                    pass
                except dill.UnpicklingError:
                    pass
            if vine is None:
                dtimes = np.tile(self.dtimes, self.n_stations)
                vine = CVine(
                    self.ranks.data,
                    varnames=self.varnames,
                    dtimes=dtimes,
                    weights=weights,
                    central_node=vg_obj.primary_var[0],
                    # central_nodes=vg_obj.primary_var,
                    verbose=False,
                    tau_min=0,
                    fit_mask=(
                        self.rain_mask.data if "R" in self.varnames else None
                    ),
                    asymmetry=self.asymmetry,
                    cop_candidates=self.cop_candidates,
                    scop_kwds=self.scop_kwds,
                )
                sh[key] = vine
        return vine

    def _fit_vine(self, vg_obj, weights):
        if not self.station_vines:
            return self._fit_vine_all(vg_obj, weights)
        vines = dict()
        for station_name in self.station_names:
            if self.verbose:
                print(f"Fitting vine on {station_name}")
            key = self._gen_vine_key(vg_obj, weights, station_name)
            vine = None
            with tools.shelve_open(self.vine_cache) as sh:
                if key in sh and not (self.refit or self.refit_vine):
                    try:
                        vine = sh[key]
                    except EOFError:
                        # we have to refit
                        pass
                    except dill.UnpicklingError:
                        pass
                if vine is None:
                    ranks_data = (
                        self.ranks.unstack().sel(station=station_name).data
                    )
                    vine = CVine(
                        ranks_data,
                        varnames=self.varnames,
                        dtimes=self.dtimes,
                        weights=weights,
                        central_node=vg_obj.primary_var[0],
                        # central_nodes=vg_obj.primary_var,
                        verbose=False,
                        tau_min=0,
                        fit_mask=(
                            self.rain_mask.sel(station=station_name).data
                            if "R" in self.varnames
                            else None
                        ),
                        asymmetry=self.asymmetry,
                        cop_candidates=self.cop_candidates,
                        scop_kwds=self.scop_kwds,
                    )
                    sh[key] = vine
            vines[station_name] = vine
        return MultiStationVine(vines)

    def _vine_bias_fitting(self):
        T_data = len(self.cop_quantiles.coords["time"])
        T_sim = 10 * T_data
        rphases = _random_phases(
            self.K,
            T_sim,
            T_data,
            # source_phases=np.angle(self.As.sel(station=station_name)),
        )

        progress = tqdm if self.verbose else lambda x: x
        for station_name in progress(self.station_names):
            self.qq_dists[station_name] = qq_dist = {
                varname: ECDF(
                    self.qq_std.sel(
                        station=station_name, variable=varname
                    ).data
                )
                for varname in self.varnames
            }

            # adjust zero-phase
            phases = []
            for phase_ in rphases:
                phase_[:, 0] = self.zero_phases[station_name]
                phases += [phase_]
            rphases = phases

            fft_sim = np.concatenate(
                [
                    np.fft.ifft(
                        self.As.sel(station=station_name)
                        * np.exp(1j * phases_)
                    ).real
                    for phases_ in rphases
                ],
                axis=1,
            )

            self.fft_dists[station_name] = fft_dist = {
                varname: ECDF(
                    fft_sim[var_i],
                    data_min=-np.inf,
                    data_max=np.inf,
                )
                for var_i, varname in enumerate(self.varnames)
            }
            fft_sim = _debias_fft_sim(fft_sim, fft_dist, qq_dist)
            if self.usevine:
                qq = dists.norm.cdf(fft_sim)
                sim = self.vine[station_name].simulate(
                    randomness=qq, T=np.arange(T_sim)
                )
                self.sim_dists[station_name] = {
                    varname: ECDF(
                        sim[var_i],
                        # data_min=1e-12,
                        # data_max=1 - 1e-12,
                        data_min=0,
                        data_max=1,
                    )
                    for var_i, varname in enumerate(self.varnames)
                }

            # fig, axs = plt.subplots(
            #     nrows=self.K, ncols=2, sharex="col", figsize=(self.K, 5)
            # )
            # for (ax0, ax1), varname in zip(axs, self.varnames):
            #     self.qq_dists[station_name][varname].plot_cdf(
            #         fig=fig, ax=ax0, label="qq_dist"
            #     )
            #     self.fft_dists[station_name][varname].plot_cdf(
            #         fig=fig, ax=ax0, label="fft_dists"
            #     )
            #     ax0.legend(loc="best")
            #     ax0.set_title(varname)
            #     self.sim_dists[station_name][varname].plot_cdf(
            #         fig=fig, ax=ax1, label="sim_dists"
            #     )
            #     ax1.axline((0, 0), slope=1, color="k", alpha=0.5)
            #     ax1.set_aspect("equal")
            # for ax in np.ravel(axs):
            #     ax.grid(True)
            # fig.suptitle(station_name)
            # plt.show()

    def _vg_ph(
        self,
        vg_obj,
        sc_pars,
        stop_at=None,
        mask_fun=None,
        rphases=None,
        return_rphases=False,
        primary_var_sim=None,
        heat_wave_kwds=None,
        **kwds,
    ):
        """Call-back function for VG.simulate. Replaces a time_series_analysis
        model.

        """
        station_name = vg_obj.station_name
        weights = "tau"
        # rphases_before = None
        if rphases is not None:
            # if self._rphases is not None and not np.all(
            #     np.isclose(self._rphases[0], rphases[0])
            # ):
            #     rphases_before = self._rphases
            self.rphases = self._rphases = rphases
        if primary_var_sim is None:
            primary_var_sim = self.primary_var
        if self.vine is None and self.usevine:
            vine = self._fit_vine(vg_obj, weights)
            # these are not so interesting so far...
            vine.verbose = False
            self.vine = vine
            if self.verbose:
                print(vine)
            stacked = self.cop_quantiles.stack(stacked=("station", "time"))
            # stacked = self.cop_quantiles.stack(stacked=("time", "station"))
            T_data = len(self.cop_quantiles.coords["time"])
            if self.station_vines:
                T = T_data
            else:
                T = np.arange(stacked.shape[1]) % T_data
            qq = self.vine.quantiles(T=T)
            finite_mask = np.isfinite(qq)
            try:
                assert np.all(finite_mask)
            except AssertionError:
                qq[~finite_mask] = np.nan
                qq = np.array(
                    [vg.helpers.interp_nonfin(values) for values in qq]
                )
            stacked.data = qq
            self.cop_quantiles = (
                stacked.unstack(dim="stacked")
                .sel(station=self.station_names)
                .transpose(*self.data_trans.dims)
            )

        elif not self.usevine:
            self.qq_std = self.data_trans
        if self.zero_phases is None:
            if self.usevine:

                def ppf_bounded(qq):
                    xx = dists.norm.ppf(qq)
                    xx[(~np.isfinite(qq)) | (np.abs(qq) > 6)] = np.nan
                    return xx

                qq_std = xr.apply_ufunc(ppf_bounded, self.cop_quantiles)
                self.qq_std = qq_std.interpolate_na(
                    "time", fill_value="extrapolate"
                )
                nonfin_mask = ~np.isfinite(self.qq_std).data
                if np.any(nonfin_mask):
                    self.qq_std.data[nonfin_mask] = 0.5

            self.As.data = np.fft.fft(self.qq_std)
            self.zero_phases = {
                station_name: np.angle(self.As.sel(station=station_name))[:, 0]
                for station_name in self.station_names
            }
            # self.qq_means = self.qq_std.mean("time")
            # self.qq_stds = self.qq_std.std("time")
            # print(f"{self.qq_means.values.round(2)=}")
            # print(f"{self.qq_stds.values.round(2)=}")

            if self.verbose:
                print("Fitting distributions for vine-fft bias correction.")
            self._vine_bias_fitting()

        if self.rphases is None:
            # this means we are the first station to be simulated this cycle!
            T_sim = vg_obj.T_sim
            T_data = vg_obj.T_summed
            if rphases is None:
                rphases = _random_phases(
                    self.K,
                    T_sim,
                    T_data,
                    verbose=self.verbose,
                    mask_fun=mask_fun,
                )
            self.rphases = rphases
            if self.verbose > 1:
                mask = _random_phases.mask
                if np.any(mask):
                    self._plot_fixed_harmonics(mask)
        else:
            rphases = []
            for phase_ in self.rphases:
                # adjust zero-phase
                phase_[:, 0] = self.zero_phases[station_name]
                rphases += [phase_]

        # # calculate the phase randomized decorrelated time series in
        # # advance for all stations, so that we can determine heat
        # # waves regionally
        # # if any of the sc_pars have changed, we need to rerun the next block
        # sc_pars_sum = sum(par.sum() for par in sc_pars)
        # if self.sc_pars_sum is None:
        #     self.sc_pars_sum = sc_pars_sum
        # elif self.sc_pars_sum != sc_pars_sum:
        #     self.sc_pars_sum = sc_pars_sum
        #     self.fft_sim = None

        if self.fft_sim is None or self.fft_sim.time.size != vg_obj.T_sim:
            if self.fft_sim is None:
                _log_none_access("fft_sim")
            self.fft_sim = xr.DataArray(
                np.full((self.n_stations, self.K, vg_obj.T_sim), 0.5),
                coords=[
                    self.station_names,
                    self.varnames,
                    vg_obj.sim_times,
                ],
                dims=["station", "variable", "time"],
            )
            # this means we also need to recalculate ranks_sim
            self.ranks_sim = None
            for station_name_ in self.station_names:
                As = self.As.sel(station=station_name_, drop=True)
                fft_sim = np.concatenate(
                    [
                        np.fft.ifft(As * np.exp(1j * rphases_)).real
                        for rphases_ in rphases
                    ],
                    axis=1,
                )[:, : vg_obj.T_sim]
                # debias fft sim
                fft_sim = _debias_fft_sim(
                    fft_sim,
                    self.fft_dists[station_name_],
                    self.qq_dists[station_name_],
                )
                fft_sim = _adjust_fft_sim(
                    fft_sim,
                    # self.qq_means.sel(station=station_name_).data,
                    # self.qq_stds.sel(station=station_name_).data,
                    sc_pars,
                    vg_obj.primary_var_ii,
                    self.phase_randomize_vary_mean,
                    adjust_prim=(primary_var_sim == self.primary_var),
                    usevine=self.usevine,
                )
                self.fft_sim.loc[dict(station=station_name_)] = fft_sim
            if heat_wave_kwds is not None:
                self.heat_waves = wplt.HeatWavesN(
                    self.fft_sim.sel(variable=primary_var_sim).mean("station"),
                    **heat_wave_kwds,
                )
        # else:
        #     fft_sim = self.fft_sim.sel(station=station_name)

        fft_sim = self.fft_sim.sel(station=station_name)

        if self.usevine:
            # qq = dists.norm.cdf(fft_sim)
            fft_dist = self.fft_dists[station_name]
            qq = np.array(
                [
                    fft_dist[varname].cdf(sim)
                    for varname, sim in zip(fft_dist.keys(), fft_sim)
                ]
            )
            if primary_var_sim != self.primary_var:
                primary_var_sim_i = self.varnames.index(primary_var_sim)

                # heat wave experimentation
                # get heat waves and make them worse
                if heat_wave_kwds is not None:
                    sim_prim_xr = xr.DataArray(
                        fft_sim[primary_var_sim_i],
                        coords=dict(time=vg_obj.sim_times),
                    )
                    # heat_waves = wplt.HeatWavesN(sim_prim_xr, n_per_summer=3)
                    sim_prim_xr.values[self.heat_waves.mask] += heat_wave_kwds[
                        "diff"
                    ]
                    qq[primary_var_sim_i] = dists.norm.cdf(sim_prim_xr.values)

                # get a baseline U
                T = qq.shape[1]
                trend = (
                    np.arange(T, dtype=float)
                    / T
                    * sc_pars.m_trend[primary_var_sim_i]
                )
                # ranks_sim = self.vine[station_name].simulate(
                #     T=np.arange(T),
                #     randomness=qq,
                #     stop_at=self.vine.varnames.index(primary_var_sim) + 1,
                # )
                # ranks_sim = _debias_ranks_sim(
                #     ranks_sim, self.sim_dists[station_name]
                # )
                # U_i = dists.norm.cdf(
                #     dists.norm.ppf(ranks_sim[primary_var_sim_i])
                #     + (sc_pars.m_t + sc_pars.m)[primary_var_sim_i]
                #     + trend
                # )

                # qq[primary_var_sim_i] = U_i
                prim_dist = self.sim_dists[station_name][self.primary_var[0]]
                qq[primary_var_sim_i] = prim_dist.cdf(
                    prim_dist.ppf(qq[primary_var_sim_i])
                    + (sc_pars.m_t + sc_pars.m)[primary_var_sim_i]
                    + trend
                )
                ranks_sim = self.vine[station_name].simulate(
                    T=np.arange(T),
                    randomness=qq,
                    stop_at=stop_at,
                    primary_var=primary_var_sim,
                    # U_i=U_i,
                )
                ranks_sim = _debias_ranks_sim(
                    ranks_sim, self.sim_dists[station_name]
                )

            else:
                # U_i = None
                ranks_sim = self.vine[station_name].simulate(
                    T=np.arange(qq.shape[1]),
                    randomness=qq,
                    stop_at=stop_at,
                )
                ranks_sim = _debias_ranks_sim(
                    ranks_sim, self.sim_dists[station_name]
                )

            # sim = dists.norm.ppf(ranks_sim)
            sim = _retransform_sim(
                ranks_sim, self.data_trans_dists[station_name]
            )

            # for varname, dist in self.data_trans_dists[station_name].items():
            #     fig, ax = dist.plot_cdf()
            #     dist_norm = dists.norm(*dists.norm.fit(dist.data))
            #     ax.plot(
            #         dist._ranks_sort_pad,
            #         dist_norm.cdf(dist._ranks_sort_pad),
            #         color="k",
            #         alpha=0.5,
            #     )
            #     fig.suptitle(varname)
            # plt.show()

            if stop_at is None:
                assert np.all(np.isfinite(sim))
            else:
                var_ii = [
                    self.varnames.index(varname)
                    for varname in self.vine.varnames[:stop_at]
                ]
                assert np.all(np.isfinite(sim[var_ii]))
        else:
            sim = fft_sim
            ranks_sim = dists.norm.cdf(fft_sim)
        if return_rphases:
            return sim, ranks_sim, rphases
        return sim, ranks_sim

    def _plot_fixed_harmonics(self, mask=None, figsize=None):
        if mask is None:
            mask = _random_phases.mask
        fig_axs = {}
        for station_name in self.station_names:
            fig, axs = plt.subplots(
                nrows=self.K,
                ncols=1,
                constrained_layout=True,
                sharex=True,
                figsize=figsize,
            )
            for varname, ax in zip(self.varnames, axs):
                A = self.As.sel(
                    station=station_name, variable=varname, drop=True
                ).data
                var = np.fft.ifft(A)
                A[~mask] = 0
                var_fixed = np.fft.ifft(A)
                ax.plot(self.dtimes, var_fixed, label="fixed")
                ax.plot(
                    self.dtimes,
                    var,
                    color="k",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.5,
                    label="raw",
                )
                ax.set_title(varname)
            fig.suptitle(
                station_name
                + f" n_fixed:{int(mask.sum() / 2)}"
                + f" ({mask.mean() * 100:.2f} %)"
            )
            fig_axs[station_name] = fig, axs
        return fig_axs

    def plot_map(self, resolution=10, terrain=True, *args, **kwds):
        if self.latitudes is None or self.longitudes is None:
            raise RuntimeError("Cant plot map without coordinates.")
        stamen_terrain = cimgt.Stamen("terrain-background")
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
            ranks_obs = self.ranks.unstack().sel(station=self.station_names)
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
        color=None,
        alpha=None,
    ):
        if obs is None:
            if transformed:
                obs = self.data_trans
            else:
                obs = self.data_daily
        if transformed:
            sim = self.ensemble_trans
            # sim = self.ensemble_trans.isel(realization=slice(1, None))
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
                figsize=(4, 3 * self.K),
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
                    .sel(station=self.station_names)  # reorder necessary :|
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
                .sel(station=self.station_names)  # reorder necessary :|
            )

            # try:
            #     # this is a known bug in xarray < 0.10.8
            #     var_stats = var_stats.rename(
            #         dict(
            #             stacked_level_0="station",
            #             stacked_level_1="realization",
            #         )
            #     )
            # except ValueError:
            #     pass

            if n_realizations < 20:
                ax.scatter(
                    xlocs.repeat(n_realizations),
                    var_stats.values.ravel(),
                    s=4,
                    color="grey",
                )
            else:
                parts = ax.violinplot(
                    var_stats.values.T, xlocs, showmeans=True
                )
                for pc in parts["bodies"]:
                    if color:
                        pc.set_facecolor(color)
                        pc.set_edgecolor(color)
                    if alpha:
                        pc.set_alpha(alpha)

            ax.grid(True)
            ax.set_title(varname)
        ax.set_xticks(xlocs)
        ax.set_xticklabels(self.station_names, rotation=30, ha="right")
        fig.suptitle(f"{stat_func.__name__} {transformed=}")
        return fig, axs

    def plot_ensemble_meteogram_hourly(
        self, station_names=None, *args, **kwds
    ):
        if station_names is None:
            station_names = self.station_names
        elif isinstance(station_names, str):
            station_names = (station_names,)
        dtimes = pd.to_datetime(self.ensemble.time.values)
        fig_axs = {}
        for station_name in station_names:
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
            wplt.suptitle_prepend(fig, station_name)
        return fig_axs

    def plot_ensemble_meteogram_daily(self, station_names=None, *args, **kwds):
        if station_names is None:
            station_names = self.station_names
        elif isinstance(station_names, str):
            station_names = (station_names,)
        dtimes = pd.to_datetime(self.ensemble_daily.time.values)
        fig_axs = {}
        for station_name in station_names:
            svg = self.vgs[station_name]
            fig, axs = svg.plot_meteogram_daily(
                plot_sim_sea=False, p_kwds=dict(linewidth=0.25), *args, **kwds
            )
            for ax, varname in zip(axs[:, 0], self.varnames):
                station_sims = self.ensemble_daily.sel(
                    station=station_name, variable=varname
                ).load()
                mins = station_sims.min("realization")
                q10 = station_sims.quantile(0.10, "realization")
                q90 = station_sims.quantile(0.90, "realization")
                maxs = station_sims.max("realization")
                ax.fill_between(dtimes, mins, maxs, color="k", alpha=0.5)
                ax.fill_between(dtimes, q10, q90, color="red", alpha=0.5)
            for ax, varname in zip(axs[:, 1], self.varnames):
                station_sims = self.ensemble_daily.sel(
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
            wplt.suptitle_prepend(fig, station_name)
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
        n_cols = int(np.ceil(n_axes**0.5))
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
                sim_ = self.ensemble_daily.sel(
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
            wplt.suptitle_prepend(fig, station_name)
        return fig_axs

    def plot_ensemble_exceedance_daily(
        self,
        *args,
        obs=None,
        figsize=None,
        thresh=None,
        name_figaxs=None,
        sim_color="blue",
        loglog=True,
        **kwds,
    ):
        assert "R" in self.varnames
        if thresh is None:
            thresh = varwg.conf.dists_kwds["R"]["threshold"] / 24
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
            # ax.loglog(mins, q_levels, **bounds_kwds)
            # ax.loglog(median, q_levels, color=sim_color)
            # ax.loglog(maxs, q_levels, **bounds_kwds)
            plot_func = ax.loglog if loglog else ax.plot
            plot_func(mins, q_levels, **bounds_kwds)
            plot_func(median, q_levels, color=sim_color)
            plot_func(maxs, q_levels, **bounds_kwds)
            ax.fill_betweenx(q_levels, mins, maxs, alpha=0.5, color=sim_color)
            plot_func(obs_qq, q_levels, color="k")

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
                # ax.loglog(mins, level, **bounds_kwds)
                # ax.loglog(median, level, color=sim_color)
                # ax.loglog(maxs, level, **bounds_kwds)
                # ax.loglog(episodes_obs[i], level, color="k")
                plot_func = ax.loglog if loglog else ax.plot
                plot_func(mins, level, **bounds_kwds)
                plot_func(median, level, color=sim_color)
                plot_func(maxs, level, **bounds_kwds)
                plot_func(episodes_obs[i], level, color="k")
                ax.fill_betweenx(level, mins, maxs, alpha=0.5, color=sim_color)

            for ax in axs:
                ax.grid(True)
            wplt.suptitle_prepend(fig, station_name)
        return name_figaxs

    def plot_ensemble_violins(
        self,
        figsize=None,
        time_period="m",
        nrows=None,
        ncols=None,
        *args,
        **kwds,
    ):
        if figsize is None:
            figsize = (6, 4 * self.K)
        if nrows is None and ncols is None:
            nrows, ncols = self.K, 1
        fig_axs = {}
        for station_name in self.station_names:
            fig, axs = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                constrained_layout=True,
            )
            for ax, varname in zip(np.ravel(axs), self.varnames):
                month = self.data_daily.time.dt.month
                obs_means = (
                    self.data_daily.sel(station=station_name, variable=varname)
                    .groupby(month)
                    .mean()
                )
                month = self.ensemble_daily.time.dt.month
                sim_means = (
                    self.ensemble_daily.sel(
                        station=station_name, variable=varname
                    )
                    .groupby(month)
                    .mean("time")
                )
                ax.violinplot(sim_means, showmeans=True)
                ax.scatter(np.arange(1, 13), obs_means, marker="_", color="k")
                ax.grid(True)
                ax.set_title(varname)
            fig_axs[station_name] = fig, axs
            wplt.suptitle_prepend(fig, station_name)
        return fig_axs

    def plot_ensemble_hydyearsum_cdfs(
        self,
        *args,
        figsize=None,
        lower_q=0.01,
        upper_q=0.99,
        color="b",
        **kwds,
    ):
        if figsize is None:
            figsize = (6, 4 * self.n_stations)
        alphas = np.linspace(0, 1, 200)
        bounds_kwds = dict(linestyle="--", color="gray", alpha=0.5)
        fig, axs = plt.subplots(
            nrows=self.n_stations,
            ncols=1,
            sharex=True,  # sharey=True
            figsize=figsize,
            constrained_layout=True,
        )
        for station_name, ax in zip(self.station_names, axs):
            obs = self.data_daily.sel(station=station_name, variable="R")
            obs = np.sort(rain_stats.hyd_year_sums(obs, full_years=True))
            sim = self.ensemble_daily.sel(
                station=station_name, variable="R"
            ).load()
            sim = rain_stats.hyd_year_sums(sim, full_years=True)
            sim_qq = sim.quantile(alphas, "hydyear")
            sim_qq = sim_qq.rename(dict(quantile="qq"))
            sim_lower = sim_qq.quantile(lower_q, "realization")
            sim_upper = sim_qq.quantile(upper_q, "realization")
            ax.step(
                obs,
                vg.helpers.rel_ranks(len(obs)),
                where="mid",
                label="obs",
                color="k",
            )
            ax.plot(sim_qq.median("realization"), alphas, color=color)
            ax.plot(sim_qq.min("realization"), alphas, **bounds_kwds)
            ax.plot(sim_qq.max("realization"), alphas, **bounds_kwds)
            ax.fill_betweenx(
                alphas, sim_lower, sim_upper, color=color, alpha=0.5
            )
            ax.grid(True)
            ax.set_ylabel("cdf")
            ax.set_xlabel("precipitation")
            ax.set_title(station_name)
        title_str = "Hydrological year sums (cdf)"
        fig.suptitle(title_str)
        # fig.canvas.set_window_title(
        #     "%s (%d)" % (title_str, fig.canvas.manager.num)
        # )
        try:
            set_window_title = (
                fig.canvas.set_window_title or fig.canvas.setWindowTitle
            )
        except AttributeError:
            pass
        else:
            set_window_title("%s (%d)" % (title_str, fig.canvas.manager.num))
        return fig, axs

    def plot_ensemble_member_heatwaves(
        self,
        realization=0,
        n_realizations=100,
        station_names=None,
        reference=None,
        drop_variables=tuple("R"),
        individual_wave_filepath=None,
        definition="kuglitsch",
        *args,
        **kwds,
    ):
        if isinstance(station_names, str):
            station_names = (station_names,)
        if station_names is None:
            station_names = self.station_names
        if reference is None:
            reference = self.ensemble
        elif isinstance(reference, str):
            reference = self.load_ensemble(
                reference, n_realizations=n_realizations
            )
        match definition:
            case "kuglitsch":
                HeatWaves = wplt.HeatwavesKuglitsch
            case "zitis":
                HeatWaves = wplt.HeatWavesZitis
            case _:
                raise RuntimeError(
                    f"{definition=} not understood. "
                    f" Must be 'kuglitsch' or 'zitis'"
                )
        if self._heatwave_bounds is None:
            heatwave_bounds = {}
            for station_name in station_names:
                station_data = self.ensemble.sel(
                    station=station_name, variable="theta"
                )
                station_reference = reference.sel(
                    station=station_name, variable="theta"
                )
                # heatwaves = wplt.heatwaves_kuglitsch(
                #     station_data.sel(realization=0).load(),
                #     station_reference.isel(
                #         realization=slice(None, n_realizations)
                #     ).load(),
                # )
                # heatwave_bounds[station_name] = (
                #     heatwaves.bound_day,
                #     heatwaves.bound_night,
                # )
                heatwaves = HeatWaves(
                    station_data.sel(realization=realization).load(),
                    station_reference.isel(
                        realization=slice(None, n_realizations)
                    ).load(),
                )
                heatwave_bounds[station_name] = heatwaves.bound
            self._heatwave_bounds = heatwave_bounds
        else:
            heatwave_bounds = self._heatwave_bounds
        varnames = self.varnames
        if drop_variables:
            varnames = [
                varname
                for varname in varnames
                if varname not in drop_variables
            ]
        fig_axs = {}
        for station_name in station_names:
            station_data = self.ensemble.sel(
                station=station_name, variable=varnames
            )
            fig, axs = wplt.plot_heatwaves(
                station_data.sel(realization=realization).load(),
                # station_reference.isel(
                #     realization=slice(None, n_realizations)
                # ).load(),
                bound_day=heatwave_bounds[station_name][0],
                bound_night=heatwave_bounds[station_name][1],
                individual_wave_filepath=individual_wave_filepath,
            )
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
            wplt.suptitle_prepend(fig, station_name)
            fig_axs[station_name] = fig, axs
            wplt.suptitle_prepend(fig, station_name)
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
            )[0]
        ]

        if self.sim_sea is not None:
            data = self.sim_trans.stack(stacked=("station", "variable")).T.data
            fig_, ax = ts.corr_img(
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
            fig_, ax = ts.corr_img(
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
                    f"{self.varnames[min(i, j)]} - "
                    f"{self.varnames[max(i, j)]}"
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
            # draw_legend = True
        else:
            edgecolor = self.n_stations * [edgecolor]
            # draw_legend = False
        # mins = np.full(2 * (self.n_variables,), np.inf)
        # maxs = np.full_like(mins, -np.inf)
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
                    f"{self.varnames[min(i, j)]} - "
                    f"{self.varnames[max(i, j)]}"
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
        n_cols = int(np.ceil(n_axes**0.5))
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

    def plot_ccplom(self, masked=False, alpha=0.1, *args, **kwds):
        """Cross-Copula-plot matrix of input and output."""
        if masked and "R" in self.varnames:
            obs_all = self.ranks[:, self.rain_mask.data]
            thresh = varwg.conf.threshold
            sim_all = self.ranks_sim.where(
                self.sim_sea.sel(variable="R") >= thresh
            ).stack(rank=("time", "station"))
        else:
            obs_all = self.ranks
            sim_all = self.ranks_sim.stack(rank=("time", "station"))
        fig_in, axs_in = wplt.ccplom(
            obs_all.data, varnames=self.varnames, alpha=alpha, *args, **kwds
        )
        fig_in.suptitle("Input")
        fig_out, axs_out = wplt.ccplom(
            sim_all.data, varnames=self.varnames, alpha=alpha, *args, **kwds
        )
        fig_out.suptitle("Output")
        return (fig_in, fig_out), (axs_in, axs_out)

    def plot_ccplom_stations(
        self, station_names=None, masked=False, alpha=0.1, *args, **kwds
    ):
        """Cross-Copula-plot matrix of input and output."""
        if isinstance(station_names, str):
            station_names = (station_names,)
        if station_names is None:
            station_names = self.station_names
        if masked and "R" in self.varnames:
            obs_all = self.ranks[:, self.rain_mask.data]
            thresh = varwg.conf.threshold
            sim_all = self.ranks_sim.where(
                self.sim_sea.sel(variable="R") >= thresh
            )
        else:
            obs_all = self.ranks
            sim_all = self.ranks_sim
        figs = dict()
        for station_name in station_names:
            fig_in, axs_in = wplt.ccplom(
                obs_all.sel(station=station_name).data,
                varnames=self.varnames,
                alpha=alpha,
                *args,
                **kwds,
            )
            fig_in.suptitle(f"Input {station_name}")
            figs[f"{station_name}_in"] = fig_in, axs_in
            fig_out, axs_out = wplt.ccplom(
                sim_all.sel(station=station_name).data,
                varnames=self.varnames,
                alpha=alpha,
                *args,
                **kwds,
            )
            fig_out.suptitle(f"Output {station_name}")
            figs[f"{station_name}_out"] = fig_out, axs_out
        return figs

    def plot_ccplom_seasonal(self, masked=False, alpha=0.2, *args, **kwds):
        """Cross-Copula-plot matrix of input and output."""
        figs = {}
        if masked and "R" in self.varnames:
            obs_all = self.ranks[:, self.rain_mask.data].unstack("rank")
            thresh = varwg.conf.threshold
            sim_all = self.ranks_sim.where(
                self.sim_sea.sel(variable="R") >= thresh
            )
        else:
            obs_all = self.ranks.unstack("rank")
            sim_all = self.ranks_sim
        for season, obs in obs_all.groupby("time.season"):
            obs = obs.stack(rank=("time", "station")).dropna("rank")
            fig, axs = wplt.ccplom(
                obs.data, varnames=self.varnames, alpha=alpha, *args, **kwds
            )
            fig.suptitle(f"Input {season}")
            figs[f"in_{season}"] = fig, axs
        for season, sim in sim_all.groupby("time.season"):
            sim = sim.stack(rank=("time", "station")).dropna("rank")
            fig, axs = wplt.ccplom(
                sim.data, varnames=self.varnames, alpha=alpha, *args, **kwds
            )
            fig.suptitle(f"Output {season}")
            figs[f"out_{season}"] = fig, axs
        return figs


if __name__ == "__main__":
    from weathercop.configs import get_dwd_vg_config

    vg_conf = get_dwd_vg_config()
    set_conf(vg_conf)
    # Example usage - replace with your own data path:
    # xds = xr.open_dataset("path/to/your/multisite_testdata.nc")
    xds = xr.open_dataset("/home/dirk/data/opendata_dwd/multisite_testdata.nc")
    # station_names = list(xar.station.values)
    # station_names.remove("Sigmarszell-Zeisertsweiler")
    # xar = xar.sel(station=station_names)

    # import warnings
    # warnings.simplefilter("error", category=RuntimeWarning)
    wc = Multisite(
        xds,
        verbose=True,
        refit=True,
        # reinitialize_vgs=True,
        # refit_vine=True,
        # refit=True,
        # refit="R",
        # refit=("R", "sun"),
        # refit="rh",
        # refit="sun",
        # rain_method="regression",
        # rain_method="distance",
        rain_method="simulation",
        infilling="vg",
        # debias=True,
        # cop_candidates=dict(gaussian=cops.gaussian),
        scop_kwds=dict(window_len=30, fft_order=3),
    )

    # wc.save()
    # ms_filepath = pickle_filepath(xds)
    # Multisite.load(ms_filepath)

    # varwg.reseed(0)
    # sim = wc.simulate(usevine=False)
    # sim = wc.simulate(usevine=True)
    sim = wc.simulate_ensemble(
        50,
        "test_vine",
        # theta_incr=3,
        clear_cache=True,
        # usevine=False,
        phase_randomize_vary_mean=False,
    )
    # wc.plot_corr_scatter_var()
    # wc.plot_corr_scatter_var(transformed=True)
    wc.plot_seasonal()
    wc.plot_ensemble_stats()
    wc.plot_ensemble_stats(transformed=True)
    # wc.plot_ccplom_seasonal(alpha=0.01)
    # wc.plot_meteogram_trans()
    # wc.vine.plot()
    # wc["Weinbiet"].plot_daily_fit("rh")
    # wc["Weinbiet"].plot_monthly_hists("rh")
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
