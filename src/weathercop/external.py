import shelve
import numpy as np
import xarray as xr
import pandas as pd

from weathercop import multisite, cop_conf
from varwg.time_series_analysis import (seasonal_distributions as sdists,
                                        seasonal_kde as skde,
                                        distributions as dists)

class MultisiteExternal(multisite.Multisite):
    def __init__(self, xds, external_ds, seas_class, *args,
               dist_family=None, refit=False, sdist_kwds=None, **kwds):
        """Multisite generation with additional phase randomized external
        variables.

        Parameters
        ----------
        external_ds : xr.Dataset
            External variables in hourly discretization.
        seas_class : vg.time_series_analysis.seasonal_distributions or
                     vg.time_series_analysis.seasonal_kde
        dist_family : vg.time_series_analysis.distributions.Dist type or None
        sdists_kwds : dict or None
            Additional keyword arguments for seas_class.
        """
        super().__init__(xds, *args, refit=refit, **kwds)
        self.ext_ds = external_ds
        self.ext_ar = external_ds.to_array("station")
        self.ext_trans = xr.full_like(self.ext_ds, np.nan)
        self.sim_ext = xr.full_like(self.ext_ds, np.nan)
        self.varnames_ext = list(self.ext_ar.coords["variable"].data)
        self.station_names_ext = list(self.ext_ds.data_vars.keys())
        self.dist_sol = {}
        self._external_as = None
        if sdist_kwds is None:
            sdist_kwds = {}
        self.fit_externals(seas_class, dist_family, refit=refit,
                           **sdist_kwds)

    def fit_externals(self, seas_class, dist_family, *, refit=False,
                    **sdist_kwds):
        if dist_family is not None:
            sdist_kwds["dist"] = dist_family
        shelve_filepath = cop_conf.cache_dir / "external_dists.she"
        for station_name, data_ar in self.ext_ds.data_vars.items():
            for varname in self.varnames_ext:
                key = f"{station_name}_{varname}"
                data = data_ar.sel(variable=varname)
                with shelve.open(str(shelve_filepath), "c") as sh:
                    if not refit and key in sh:
                        if self.verbose:
                            print(f"Recovering distribution of {key}")
                        seas_class, sol = sh[key]
                        dist = seas_class(data=data,
                                          datetimes=self.dtimes,
                                          solution=sol,
                                          **sdist_kwds)
                    else:
                        if self.verbose:
                            print(f"Fitting a distribution to {key}")
                        dist = seas_class(data=data,
                                          datetimes=self.dtimes,
                                          **sdist_kwds)
                        sol = dist.fit()
                        sh[key] = seas_class, sol
                self.dist_sol[key] = dist, sol
                data_trans = dists.norm.ppf(dist.cdf())
                (self.ext_trans[station_name]
                 .loc[dict(variable=varname)]) = data_trans

    @property
    def external_As(self):
        if self._external_as is None:
            self._external_as = {}
            for station_name, data_ar in self.ext_trans.data_vars.items():
                for varname in self.varnames_ext:
                    key = f"{station_name}_{varname}"
                    data = data_ar.sel(variable=varname)
                    self._external_as[key] = np.fft.fft(data)
        return self._external_as

    def simulate(self, *args, **kwds):
        sim_sea = super().simulate(*args, **kwds)
        rphases = self._rphases[0]
        for key, A in self.external_As.items():
            station_name, varname = key.split("_")
            sim_ext_norm = np.fft.ifft(A * np.exp(1j * rphases)).real
            dist, sol = self.dist_sol[key]
            quantiles = dists.norm.cdf(sim_ext_norm)
            (self.sim_ext[station_name]
             .loc[dict(variable=varname)]) = dist.ppf(sol, quantiles)
        sim_sea = xr.merge([sim_sea.to_dataset("station"),
                            self.sim_ext]
                           ).to_array("station")
        self.sim_sea = sim_sea
        return sim_sea

    def plot_ensemble_external_meteogramm(self, *args, **kwds):
        dtimes = pd.to_datetime(self.ensemble.time.values)
        fig_axs = {}
        svg = self.vgs[self.station_names[0]]
        for station_name in self.station_names_ext:
            obs = self.ext_ds[station_name].values.T[None, :]
            fig, axs = svg.plot_meteogramm_daily(plot_sim_sea=False,
                                                 p_kwds=dict(linewidth=.25),
                                                 obs=obs,
                                                 var_names=station_name,
                                                 *args, **kwds)
            for ax, varname in zip(axs[:, 0], self.varnames):
                station_sims = self.ensemble.sel(station=station_name,
                                                 variable=varname).load()
                mins = station_sims.min("realization")
                q10 = station_sims.quantile(.10, "realization")
                q90 = station_sims.quantile(.90, "realization")
                maxs = station_sims.max("realization")
                ax.fill_between(dtimes, mins, maxs,
                                color="k", alpha=.5)
                ax.fill_between(dtimes, q10, q90,
                                color="red", alpha=.5)
            for ax, varname in zip(axs[:, 1], self.varnames):
                station_sims = self.ensemble.sel(station=station_name,
                                                 variable=varname).load()
                ax.hist(station_sims.values.ravel().dropna(),
                        40, density=True,
                        orientation="horizontal", histtype="step",
                        color="grey")
            fig_axs[station_name] = fig, axs
        return fig_axs

    def plot_corr_scatter_var(self, transformed=False):
        data_obs = xr.merge((self.data_daily.to_dataset("station"),
                             self.ext_ds)
                            ).to_array("station")
        data_sim = self.sim_sea
        return self._plot_corr_scatter_var(data_obs, data_sim,
                                      "")

