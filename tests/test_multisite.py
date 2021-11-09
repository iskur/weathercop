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
data_root = Path().home() / "data/opendata_dwd"


class Test(npt.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.verbose = True
        self.refit = False
        # self.xar = xr.open_dataarray(str(data_root /
        #                                  "multisite_testdata.nc"))
        # self.xds = self.xar.to_dataset("station")
        self.xds = xr.open_dataset(data_root / "multisite_testdata.nc")
        self.wc = Multisite(
            self.xds,
            verbose=self.verbose,
            refit=self.refit,
            refit_vine=self.refit,
        )
        self.sim_sea = self.wc.simulate(phase_randomize_vary_mean=False)
        # self.wc.plot_cross_corr()

    def tearDown(self):
        pass

    # def test_seasonality(self):
    #     station_name = self.wc.station_names[0]
    #     qq_xr = self.wc.cop_quantiles.sel(station=station_name)
    #     fig, ax = plt.subplots(nrows=1, ncols=1)
    #     coeffs = [
    #         np.corrcoef(group)[0, 1:]
    #         for month_i, group in qq_xr.groupby(qq_xr.time.dt.month)
    #     ]
    #     ax.plot(coeffs)
    #     # ax.set_prop_cycle(None)
    #     # coeffs = [np.corrcoef(group)[0, 1:]
    #     #           for month_i, group in
    #     #           ranks_xr.groupby(ranks_xr.time.dt.month)]
    #     # ax.plot(coeffs, "--")
    #     plt.show()

    def test_phase_randomization(self):
        qq_std = self.wc.qq_std
        fft_sim = self.wc.fft_sim
        for var_i, varname in enumerate(self.wc.varnames):
            data_sim_ = fft_sim.sel(variable=varname).values
            sim_corr = nan_corrcoef(data_sim_)
            sim_corr = sim_corr[np.triu_indices_from(sim_corr, 1)]
            data_obs_ = qq_std.sel(variable=varname).values
            obs_corr = nan_corrcoef(data_obs_)
            obs_corr = obs_corr[np.triu_indices_from(obs_corr, 1)]
            try:
                npt.assert_almost_equal(sim_corr, obs_corr, decimal=4)
            except AssertionError:
                fig, ax = plt.subplots(
                    nrows=1, ncols=1, subplot_kw=dict(aspect="equal")
                )
                ax.scatter(obs_corr, sim_corr, marker="x")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                plt.show()
                raise

    def test_sim_rphases(self):
        rphases = rphases_before = self.wc._rphases
        sim_sea_new = self.wc.simulate(
            rphases=rphases, phase_randomize_vary_mean=False
        )
        rphases_after = self.wc._rphases
        npt.assert_almost_equal(rphases_after, rphases_before)
        sim_sea_new2 = self.wc.simulate(
            rphases=rphases, phase_randomize_vary_mean=False
        )
        npt.assert_almost_equal(sim_sea_new2.values, sim_sea_new.values)
        for station_i, station_name in enumerate(self.wc.station_names):
            print(f"{station_name=}")
            actual = self.sim_sea.sel(station=station_name)
            expected = sim_sea_new.sel(station=station_name)
            try:
                npt.assert_almost_equal(actual.values, expected.values)
            except AssertionError:
                fig, axs = plt.subplots(
                    nrows=self.wc.K,
                    ncols=1,
                    sharex=True,
                    constrained_layout=True,
                )
                for ax, varname in zip(axs, self.wc.varnames):
                    ax.plot(
                        expected.time,
                        expected.sel(variable=varname).values,
                        label="sim1",
                        color="k",
                    )
                    ax.plot(
                        actual.time,
                        actual.sel(variable=varname).values,
                        label="sim2",
                        color="b",
                        linestyle="--",
                    )
                    ax.set_title(varname)
                axs[0].legend(loc="best")
                fig.suptitle(f"{station_i=}: {station_name}")

                fig, axs = plt.subplots(
                    nrows=1,
                    ncols=self.wc.K,
                    subplot_kw=dict(aspect="equal"),
                    constrained_layout=True,
                    figsize=(self.wc.K * 4, 4),
                )
                for ax, varname in zip(axs, self.wc.varnames):
                    expected_var = expected.sel(variable=varname).values
                    actual_var = actual.sel(variable=varname).values
                    ax.scatter(
                        expected_var, actual_var, s=1, marker="x", color="b"
                    )
                    min_ = min(expected_var.min(), actual_var.min())
                    max_ = max(expected_var.max(), actual_var.max())
                    ax.plot(
                        [min_, max_],
                        [min_, max_],
                        linestyle="--",
                        color="k",
                        linewidth=1,
                    )
                    ax.grid(True)
                    ax.set_title(varname)

                plt.show()
                raise

    def test_sim(self):
        sim_sea = self.sim_sea
        # for station_name, svg in self.wc.vgs.items():
        #     fig, axs = svg.plot_meteogramm()
        # plt.show()
        # sim_stacked = sim_sea.stack(stacked=("station", "variable")).T
        # obs_stacked = self.wc.data_daily.stack(
        #     stacked=("station", "variable")
        # ).T
        sim_stacked = sim_sea.stack(stacked=("variable", "time")).T
        obs_stacked = self.wc.data_daily.stack(stacked=("variable", "time")).T
        # corr_sim = np.corrcoef(sim_stacked)
        # corr_obs = np.corrcoef(obs_stacked)
        corr_sim = nan_corrcoef(sim_stacked.T)
        corr_obs = nan_corrcoef(obs_stacked.T)
        try:
            npt.assert_almost_equal(corr_sim, corr_obs, decimal=1)
        except AssertionError:
            self.wc.plot_corr()
            fig, ax = plt.subplots(
                nrows=1, ncols=1, subplot_kw=dict(aspect="equal")
            )
            ax.scatter(corr_obs, corr_sim, marker="x")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.show()
            raise

        sim_means = sim_sea.mean("time")
        obs_means = self.wc.data_daily.mean("time")
        npt.assert_almost_equal(
            sim_means.mean(), obs_means.data.mean(), decimal=2
        )
        # xrt.assert_allclose(sim_means, obs_means, atol=0.1)
        # # test auto- and cross-correlation
        # import ipdb; ipdb.set_trace()
        # cross_obs = (self.wc.data_daily
        #              .to_dataset(dim="station")
        #              .apply(partial(ts.cross_corr, k=range(7))))

    def test_sim_mean_increase(self):
        theta_incr = 4
        sim = self.wc.simulate(
            theta_incr=theta_incr, phase_randomize_vary_mean=False
        )
        # sim = sim.sel(variable="theta", drop=True).mean(dim="time")
        # obs = self.wc.data_daily.sel(variable="theta", drop=True).mean(
        #     dim="time"
        # )
        sim = sim.sel(variable="theta", drop=True).mean()
        obs = self.wc.data_daily.sel(variable="theta", drop=True).mean()
        npt.assert_almost_equal(sim - obs, theta_incr, decimal=2)


if __name__ == "__main__":
    # npt.run_module_suite()
    xds = xr.open_dataset(data_root / "multisite_testdata.nc")
    from weathercop import multisite as ms
    from importlib import reload

    reload(ms)
    # xds = xds.sel(time=slice("2000", "2005"))
    wc = ms.Multisite(
        xds,
        verbose=True,
        # primary_var="R",
        # refit=True,
        # refit=("R", "sun"),
        # refit="Mühldorf",
        refit_vine=True,
    )

    wc.simulate()
    np.random.seed(0)
    wc.simulate_ensemble(
        20,
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
