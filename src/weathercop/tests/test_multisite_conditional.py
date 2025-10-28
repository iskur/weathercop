import numpy as np
import numpy.testing as npt
from weathercop import multisite_conditional as msc, multisite as ms
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
from weathercop import copulae
from weathercop.configs import get_dwd_vg_config
import varwg

vg_conf = get_dwd_vg_config()

ms.set_conf(vg_conf)
data_root = Path().home() / "data/opendata_dwd"


class Test(npt.TestCase):
    def setUp(self):
        varwg.reseed(0)
        self.verbose = True
        self.refit = True
        self.refit_vine = True
        xds = xr.open_dataset(data_root / "multisite_testdata.nc")
        self.weinbiet_bounds = msc.YearlyRainBounds(
            lower=500 / 24,
            upper=790 / 24,
            station_name="Weinbiet",
            lower_period=int(0.5 * 365),
            upper_period=int(2 * 365),
        )

        self.wc = msc.MultisiteConditional(
            self.weinbiet_bounds,
            xds,
            # refit_vine=True,
            primary_var="R",
            refit=self.refit,
            rain_method="distance",
            # cop_candidates=cop_candidates,
            verbose=2,
        )

        # self.wc = ms.Multisite(
        #     xds,
        #     refit_vine=self.refit_vine,
        #     primary_var="R",
        #     refit=self.refit,
        #     rain_method="distance",
        #     # cop_candidates=cop_candidates,
        #     verbose=2,
        # )

        self.sim_sea = self.wc.simulate(phase_randomize_vary_mean=False)
        # self.sim_sea = super(ms.Multisite, self.wc).simulate(
        #     rphases=self.wc._rphases, phase_randomize_vary_mean=False
        # )

    def tearDown(self):
        pass

    def test_bounds(self):
        try:
            npt.assert_almost_equal(self.weinbiet_bounds(self.sim_sea), 0)
        except AssertionError:
            self.weinbiet_bounds.plot(self.sim_sea)
            plt.show()
            raise

    def test_phase_randomization(self):
        qq_std = self.wc.qq_std
        fft_sim = self.wc.fft_sim
        npt.assert_almost_equal(
            qq_std.mean("time").values, fft_sim.mean("time").values
        )
        for var_i, varname in enumerate(self.wc.varnames):
            data_sim_ = fft_sim.sel(variable=varname).values
            sim_corr = ms.nan_corrcoef(data_sim_)
            sim_corr = sim_corr[np.triu_indices_from(sim_corr, 1)]
            data_obs_ = qq_std.sel(variable=varname).values
            obs_corr = ms.nan_corrcoef(data_obs_)
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
                ax.grid(True)

                # fig, axs = plt.subplots(
                #     nrows=self.wc.n_stations,
                #     ncols=1,
                #     figsize=(self.wc.n_stations, 5),
                # )
                # for ax, station_name in zip(axs, self.wc.station_names):
                #     ax.plot(
                #         fft_sim.sel(
                #             variable=varname, station=station_name
                #         ).values,
                #         "b",
                #     )
                #     ax.plot(
                #         qq_std.sel(variable=varname, station=station_name),
                #         "b",
                #         linestyle="--",
                #     )
                #     ax.set_title(station_name)

                plt.show()
                raise
        # self.wc.plot_cross_corr_var(transformed=True)
        # self.wc.plot_cross_corr_stat(transformed=True)
        # plt.show()


if __name__ == "__main__":
    npt.run_module_suite()
