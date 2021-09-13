from functools import partial
from pathlib import Path

import numpy as np
import numpy.testing as npt
import xarray as xr
import xarray.testing as xrt
from matplotlib import pyplot as plt

# from lhglib.contrib.meteo import dwd_opendata as dwd
from vg.time_series_analysis import time_series as ts
from weathercop.multisite import Multisite, set_conf, nan_corrcoef
import opendata_vg_conf as vg_conf

set_conf(vg_conf)
data_root = Path("/media/data/opendata_dwd")


class Test(npt.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.verbose = True
        self.refit = False
        # self.xar = xr.open_dataarray(str(data_root /
        #                                  "multisite_testdata.nc"))
        # self.xds = self.xar.to_dataset("station")
        self.xds = xr.open_dataset(data_root /
                                   "multisite_testdata.nc")
        self.wc = Multisite(self.xds, verbose=self.verbose,
                            refit=self.refit,
                            refit_vine=True,
                            )
        self.sim_sea = self.wc.simulate()
        # self.wc.plot_cross_corr()

    def tearDown(self):
        pass

    def test_seasonality(self):
        quantiles = self.wc.vine.quantiles()
        qq_xr = quantiles.sel(station="Konstanz")
        fig, ax = plt.subplots(nrows=1, ncols=1)
        coeffs = [np.corrcoef(group)[0, 1:]
                  for month_i, group in
                  qq_xr.groupby(qq_xr.time.dt.month)]
        ax.plot(coeffs)
        # ax.set_prop_cycle(None)
        # coeffs = [np.corrcoef(group)[0, 1:]
        #           for month_i, group in
        #           ranks_xr.groupby(ranks_xr.time.dt.month)]
        # ax.plot(coeffs, "--")
        plt.show()

    def test_phase_randomization(self):
        ranks_obs = self.wc.ranks
        ranks_sim = self.wc.ranks_sim
        for var_i, varname in enumerate(self.wc.varnames):
            data_sim_ = (ranks_sim
                         .sel(variable=varname)
                         .values)
            sim_corr = nan_corrcoef(data_sim_)
            sim_corr = sim_corr[np.triu_indices_from(sim_corr, 1)]
            data_obs_ = (ranks_obs
                         .sel(variable=varname)
                         .values)
            obs_corr = nan_corrcoef(data_obs_)
            obs_corr = obs_corr[np.triu_indices_from(obs_corr, 1)]
            npt.assert_almost_equal(sim_corr, obs_corr)

    def test_sim(self):
        sim_sea = self.sim_sea
        # for station_name, svg in self.wc.vgs.items():
        #     fig, axs = svg.plot_meteogramm()
        # plt.show()
        sim_stacked = sim_sea.stack(stacked=("station", "variable")).T
        obs_stacked = (self.wc.data_daily
                       .stack(stacked=("station", "variable"))
                       .T)
        corr_sim = np.corrcoef(sim_stacked)
        corr_obs = np.corrcoef(obs_stacked)
        try:
            npt.assert_almost_equal(corr_sim, corr_obs, decimal=1)
        except AssertionError:
            self.wc.plot_corr()
            plt.show()
            # raise

        sim_means = sim_sea.mean("time")
        obs_means = self.wc.data_daily.mean("time")
        xrt.assert_allclose(sim_means, obs_means)
        # # test auto- and cross-correlation
        # import ipdb; ipdb.set_trace()
        # cross_obs = (self.wc.data_daily
        #              .to_dataset(dim="station")
        #              .apply(partial(ts.cross_corr, k=range(7))))

    def test_sim_mean_increase(self):
        theta_incr = 4
        sim = self.wc.simulate(theta_incr=theta_incr)
        sim = (sim
               .sel(variable="theta", drop=True)
               .mean(dim="time"))
        obs = (self.wc.data_daily
               .sel(variable="theta", drop=True)
               .mean(dim="time"))
        npt.assert_almost_equal(sim.data, (obs + theta_incr).data,
                                decimal=2)


if __name__ == "__main__":
    # npt.run_module_suite()
    xds = xr.open_dataset(data_root /
                          "multisite_testdata.nc")
    from weathercop import multisite as ms
    from importlib import reload
    reload(ms)
    # xds = xds.sel(time=slice("2000", "2005"))
    wc = ms.Multisite(xds,
                      verbose=True,
                      # primary_var="R",
                      # refit=True,
                      # refit=("R", "sun"),
                      # refit="Mühldorf",
                      refit_vine=True
                      )

    wc.simulate()
    np.random.seed(0)
    wc.simulate_ensemble(20,
                         clear_cache=True,
                         # usevine=False,
                         )
    print(wc.vine)

    wc.plot_ensemble_stats()
    wc.plot_ensemble_exceedance_daily()
    wc.plot_ensemble_qq()
    # wc["Kempten"].plot_daily_fit("R")
    # wc["Kempten"].plot_monthly_hists("R")

    # qq_xr = wc.cop_quantiles.sel(station="Regensburg")
    # fig, ax = plt.subplots(nrows=1, ncols=1)
    # coeffs = [np.corrcoef(group)[0, 1:]
    #           for month_i, group in
    #           qq_xr.groupby(qq_xr.time.dt.month)]
    # ax.plot(coeffs)
    # # ax.set_prop_cycle(None)
    # # coeffs = [np.corrcoef(group)[0, 1:]
    # #           for month_i, group in
    # #           ranks_xr.groupby(ranks_xr.time.dt.month)]
    # # ax.plot(coeffs, "--")

    # wc.plot_corr_scatter_var(transformed=True)
    # wc.plot_corr_scatter_var()

    # wc.plot_corr_scatter_stat(transformed=True)
    # wc.plot_corr_scatter_stat()

    plt.show()
