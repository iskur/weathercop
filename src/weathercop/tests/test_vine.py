import time
from collections import OrderedDict

import numpy as np
import xarray as xr
import dill
import matplotlib.pyplot as plt
import numpy.testing as npt
import pandas as pd
from scipy import stats as spstats

from vg import helpers as my
import vg
from vg.time_series_analysis import distributions as dists
from weathercop import plotting, copulae
from weathercop.vine import CVine, RVine, vg_ph


class Test(npt.TestCase):
    def setUp(self):
        vg.reseed(0)
        self.verbose = True
        # self.cov = np.array([[ 1.3  , -0.045,  0.597,  0.669,  0.568,  0.507],
        #                      [-0.045,  0.927,  0.576, -0.248,  0.072,  0.136],
        #                      [ 0.597,  0.576,  0.963,  0.283,  0.288,  0.282],
        #                      [ 0.669, -0.248,  0.283,  0.988,  0.129,  0.116],
        #                      [ 0.568,  0.072,  0.288,  0.129,  0.91 ,  0.686],
        #                      [ 0.507,  0.136,  0.282,  0.116,  0.686,  0.892]])
        # self.cov = np.array([[ 1.3  , -0.045,  0.568,  0.597,  0.669,  0.507],
        #                      [-0.045,  0.927,  0.072,  0.576, -0.248,  0.136],
        #                      [ 0.568,  0.072,  0.91 ,  0.288,  0.129,  0.686],
        #                      [ 0.597,  0.576,  0.288,  0.963,  0.283,  0.282],
        #                      [ 0.669, -0.248,  0.129,  0.283,  0.988,  0.116],
        #                      [ 0.507,  0.136,  0.686,  0.282,  0.116,  0.892]])
        self.cov = np.array(
            [
                [1.3, 0.568, 0.597, 0.507, -0.045, 0.669],
                [0.568, 0.91, 0.288, 0.686, 0.072, 0.129],
                [0.597, 0.288, 0.963, 0.282, 0.576, 0.283],
                [0.507, 0.686, 0.282, 0.892, 0.136, 0.116],
                [-0.045, 0.072, 0.576, 0.136, 0.927, -0.248],
                [0.669, 0.129, 0.283, 0.116, -0.248, 0.988],
            ]
        )

        self.K = len(self.cov)
        self.K -= 1
        T = 1000
        self.data_normal = np.random.multivariate_normal(
            len(self.cov) * [0], self.cov, T
        ).T
        self.data_normal = self.data_normal[: self.K]
        self.data_ranks = np.array(
            [
                dists.norm.cdf(row, mu=row.mean(), sigma=row.std())
                for row in self.data_normal
            ]
        )
        import string

        self.varnames = list(string.ascii_lowercase[: self.K])
        # self.varnames = list("".join("%d" % i for i in range(self.K)))
        # self.varnames = "R", "theta", "ILWR", "u", "v"
        # self.varnames = "R", "theta", "ILWR", "rh", "u", "v"
        # self.varnames = 'R', 'theta', 'u', 'ILWR', 'rh', 'v'

        # self.varnames = 'R', 'u', 'ILWR', 'v', 'theta', 'rh'
        # weights = "likelihood"
        weights = "tau"

        # self.rvine = RVine(self.data_ranks, varnames=self.varnames,
        #                    weights=weights, verbose=self.verbose,
        #                    debug=self.verbose)
        # self.rsim = self.rvine.simulate(T=3 * T)
        # self.rsim_normal = np.array([dists.norm.ppf(values,
        #                                             mu=source.mean(),
        #                                             sigma=source.std())
        #                              for values, source
        #                              in zip(self.rsim, self.data_normal)])
        # self.rquantiles = self.rvine.quantiles()

        # dtimes = None
        dtimes = pd.date_range("2000-01-01", periods=T, freq="D")
        # cop_candidates = dict(gaussian=copulae.gaussian)
        cop_candidates = None
        self.cvine = CVine(
            self.data_ranks,
            varnames=self.varnames,
            dtimes=dtimes,
            weights=weights,
            verbose=self.verbose,
            debug=self.verbose,
            tau_min=0,
            cop_candidates=cop_candidates,
        )
        self.csim = self.cvine.simulate()  # (T=3 * T)
        self.csim_normal = np.array(
            [
                dists.norm.ppf(values, mu=source.mean(), sigma=source.std())
                for values, source in zip(self.csim, self.data_normal)
            ]
        )
        self.cquantiles = self.cvine.quantiles()

        self.vines = (self.cvine,)

    def test_serialize(self):
        for vine in self.vines:
            dill_str = dill.dumps(vine)
            vine_recovered = dill.loads(dill_str)

    # def test_likelihood_tree(self):
    #     if self.verbose:
    #         print("Testing tree construction with likelihood as weight")
    #     rvine = RVine(self.data_ranks, varnames=self.varnames,
    #                   verbose=self.verbose, weights="likelihood")
    #     fig, ax = rvine.plot(edge_labels="copulas")
    #     fig.suptitle("likelihood")
    #     fig, ax = rvine.plot_tplom()
    #     fig.suptitle("likelihood")
    #     fig, ax = rvine.plot_qqplom()
    #     fig.suptitle("likelihood")

    #     rvine = RVine(self.data_ranks, varnames=self.varnames,
    #                   verbose=self.verbose, weights="tau")
    #     fig, ax = self.rvine.plot(edge_labels="copulas")
    #     fig.suptitle("tau")
    #     fig, ax = self.rvine.plot_tplom()
    #     fig.suptitle("tau")
    #     fig, ax = self.rvine.plot_qqplom()
    #     fig.suptitle("tau")
    #     plt.show()

    # def test_rsimulate(self):
    #     if self.verbose:
    #         print("Covariance matrix of RVine simulation")
    #     try:
    #         npt.assert_almost_equal(np.cov(self.rsim_normal),
    #                                 np.cov(self.data_normal),
    #                                 decimal=1)
    #     except AssertionError:
    #         plotting.ccplom(self.rsim, k=0, kind="img",
    #                         title="simulated from RVine")
    #         plotting.ccplom(self.data_ranks, k=0, kind="img", title="observed")
    #         plt.show()
    #         raise

    def test_csimulate(self):
        if self.verbose:
            print("Stats of CVine simulation")
        npt.assert_almost_equal(
            self.csim_normal.mean(axis=1),
            self.data_normal.mean(axis=1),
            decimal=1,
        )
        try:
            npt.assert_almost_equal(
                np.cov(self.csim_normal), np.cov(self.data_normal), decimal=1
            )
        except AssertionError:
            print("Obs \n", np.corrcoef(self.data_ranks).round(3))
            print("Sim \n", np.corrcoef(self.cquantiles).round(3))
            # plotting.ccplom(self.rsim, k=0, kind="img",
            #                 title="simulated from CVine")
            # plotting.ccplom(self.data_ranks, k=0, kind="img", title="observed")
            # plt.show()
            raise

    # def test_rquantiles(self):
    #     if self.verbose:
    #         print("Quantile simulation roundtrip of RVine")
    #     # does it matter if we specify the ranks explicitly?
    #     quantiles_self = self.rvine.quantiles(ranks=self.rvine.ranks)
    #     try:
    #         npt.assert_almost_equal(quantiles_self, self.rquantiles)
    #     except AssertionError:
    #         fig, axs = plt.subplots(nrows=self.K, sharex=True)
    #         for i, ax in enumerate(axs):
    #             ax.plot(self.rquantiles[i])
    #             ax.plot(quantiles_self[i], "--")
    #         # plt.show()
    #         # raise

    #     sim = self.rvine.simulate(randomness=self.rquantiles)
    #     try:
    #         npt.assert_almost_equal(sim, self.data_ranks, decimal=2)
    #     except AssertionError:
    #         fig, axs = plt.subplots(nrows=self.K, sharex=True)
    #         for i, ax in enumerate(axs):
    #             ax.plot(self.data_ranks[i])
    #             ax.plot(sim[i], "--")
    #             ax.set_ylabel(self.varnames[i])

    #         fig, axs = plt.subplots(nrows=int(np.sqrt(self.K)) + 1,
    #                                 ncols=int(np.sqrt(self.K)),
    #                                 sharex=True, sharey=True,
    #                                 subplot_kw=dict(aspect="equal"))
    #         axs = np.ravel(axs)
    #         for i, ax in enumerate(axs[:self.K]):
    #             ax.scatter(self.data_ranks[i], sim[i],
    #                        marker="x", s=2)
    #             ax.plot([0, 1], [0, 1], color="black")
    #             ax.grid(True)
    #             ax.set_title(self.varnames[i])
    #         for ax in axs[self.K:]:
    #             ax.set_axis_off()
    #         self.rvine.plot(edge_labels="copulas")
    #         plt.show()
    #         raise

    # def test_rquantiles_corr(self):
    #     if self.verbose:
    #         print("Zero correlation matrix of RVine quantiles")
    #     corr_exp = np.zeros_like(self.cov)
    #     corr_exp.ravel()[::self.K+1] = 1
    #     try:
    #         npt.assert_almost_equal(np.corrcoef(self.rquantiles),
    #                                 corr_exp,
    #                                 decimal=1)
    #     except AssertionError:
    #         plotting.ccplom(self.rquantiles, k=0, kind="img",
    #                         title="rquantiles")
    #         plotting.ccplom(self.data_ranks, k=0, kind="img",
    #                         title="copula_input")
    #         self.rvine.plot_qqplom()
    #         self.rvine.plot(edge_labels="copulas")
    #         plt.show()
    #         raise

    def test_cquantiles_hist(self):
        if self.verbose:
            print("Uniformity of quantiles' marginals")
        for i, q in enumerate(self.cquantiles):
            _, p_value = spstats.kstest(q, spstats.uniform.cdf)
            try:
                assert p_value > 0.25
            except AssertionError:
                label = self.varnames[i]
                uni = np.random.random(size=q.size)
                fig, ax = my.hist([q, uni], 20, dist=spstats.uniform)
                fig.suptitle("%s p-value: %.3f" % (label, p_value))
                plt.show()

    def test_cquantiles(self):
        if self.verbose:
            print("Quantile simulation roundtrip of CVine")
        corr_exp = np.zeros_like(self.cov)
        corr_exp.ravel()[:: self.K + 1] = 1
        sim = self.cvine.simulate(randomness=self.cquantiles)
        # corr_act = np.corrcoef(self.cquantiles)
        # try:
        #     npt.assert_almost_equal(corr_act, corr_exp, decimal=2)
        # except AssertionError:
        #     print("Obs \n", np.corrcoef(self.data_ranks).round(3))
        #     print("Sim \n", np.corrcoef(sim).round(3))
        #     # cc_kwds = dict(k=0, kind="img", varnames=self.varnames)
        #     # plotting.ccplom(self.cquantiles, title="cquantiles",
        #     #                 **cc_kwds)
        #     # plotting.ccplom(self.data_ranks, title="copula_input",
        #     #                 **cc_kwds)
        #     # plotting.ccplom(sim, title="copula output", **cc_kwds)
        #     # self.cvine.plot_qqplom()
        #     # plt.show()
        #     raise

        # # does it matter if we specify the ranks explicitly?
        # quantiles_self = self.cvine.quantiles(ranks=self.cvine.ranks)
        # try:
        #     npt.assert_almost_equal(quantiles_self, self.cquantiles)
        # except AssertionError:
        #     fig, axs = plt.subplots(nrows=self.K, sharex=True)
        #     for i, ax in enumerate(axs):
        #         ax.plot(self.cquantiles[i])
        #         ax.plot(quantiles_self[i], "--")
        #     plt.show()
        #     raise

        print(self.cvine)
        print(self.cvine.varnames)
        print(self.cvine.varnames_old)
        try:
            npt.assert_almost_equal(sim, self.data_ranks, decimal=2)
        except AssertionError:
            obs_corr = pd.DataFrame(
                np.corrcoef(self.data_ranks).round(3),
                index=self.varnames,
                columns=self.varnames,
            )
            sim_corr = pd.DataFrame(
                np.corrcoef(sim).round(3),
                index=self.varnames,
                columns=self.varnames,
            )
            print("\nObs\n", obs_corr)
            print("\nSim\n", sim_corr)
            print("\nDiff\n", obs_corr - sim_corr)
            fig, axs = plt.subplots(nrows=self.K, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(self.data_ranks[i])
                ax.plot(sim[i], "--")
                ax.set_title(self.varnames[i])

            fig, axs = plt.subplots(
                nrows=int(np.sqrt(self.K) + 1),
                ncols=int(np.sqrt(self.K)),
                sharex=True,
                sharey=True,
                subplot_kw=dict(aspect="equal"),
            )
            axs = np.ravel(axs)
            for i, ax in enumerate(axs[: self.K]):
                ax.scatter(self.data_ranks[i], sim[i], marker="x", s=2)
                ax.plot([0, 1], [0, 1], color="black")
                ax.grid(True)
                ax.set_title(self.varnames[i])
            for ax in axs[self.K :]:
                ax.set_axis_off()

            self.cvine.plot(edge_labels="copulas")
            plt.show()
            raise

    def test_cquantiles_corr(self):
        if self.verbose:
            print("Zero correlation matrix of CVine quantiles")
        corr_exp = np.zeros((self.K, self.K), dtype=float)
        corr_exp.ravel()[:: self.K + 1] = 1
        corr_act = np.corrcoef(self.cquantiles)
        try:
            npt.assert_almost_equal(corr_act, corr_exp, decimal=1)
        except AssertionError:
            print("Quantiles corr \n", corr_act.round(3))
            if self.verbose:
                plotting.ccplom(
                    self.cquantiles, k=0, kind="img", title="cquantiles"
                )
                plotting.ccplom(
                    self.data_ranks, k=0, kind="img", title="copula_input"
                )
                self.cvine.plot_qqplom()
                fig, axs = plt.subplots(nrows=self.K, ncols=1, sharex=True)
                plt.show()
            raise

    # def test_seasonal_yearly(self):
    #     if self.verbose:
    #         print("Seasonal Vine with singe year VG data")
    #         import vg
    #         from vg import vg_plotting, vg_base
    #         import config_konstanz_disag as conf
    #     vg.conf = vg_plotting.conf = vg_base.conf = conf
    #     # met_vg = vg.VG(("theta", "ILWR", "rh", "R"), verbose=True)
    #     met_vg = vg.VG(("theta", "Qsw", "ILWR", "rh", "u", "v"), verbose=True)
    #     ranks = np.array([spstats.norm.cdf(values)
    #                       for values in met_vg.data_trans])
    #     T = ranks.shape[1]
    #     for ranks_ in (ranks[:, :T//2], ranks[:, T//2:]):
    #         cvine = RVine(ranks_,
    #                       varnames=met_vg.var_names,
    #                       # dtimes=met_vg.times[year_mask],
    #                       # weights="likelihood"
    #                       )
    #     # year_first = met_vg.times[0].year
    #     # year_last = met_vg.times[-1].year
    #     # years = np.array([dtime.year for dtime in met_vg.times])
    #     # ranks = np.array([spstats.norm.cdf(values)
    #     #                   for values in met_vg.data_trans])
    #     # for year in range(year_first, year_last + 1):
    #     #     year_mask = years == year
    #     #     cvine = RVine(ranks[:, year_mask],
    #     #                   varnames=met_vg.var_names,
    #     #                   # dtimes=met_vg.times[year_mask],
    #     #                   # weights="likelihood"
    #     #                   )
    #         fig, axs = cvine.plot(edge_labels="copulas")
    #     plt.show()

    def test_seasonal(self):
        if self.verbose:
            print("Seasonal Vine with VG data")
            import vg
            from vg import vg_plotting, vg_base
            import config_konstanz as conf
        vg.conf = vg_plotting.conf = vg_base.conf = conf
        # met_vg = vg.VG(("theta", "ILWR", "rh", "R"), verbose=True)
        met_vg = vg.VG(("theta", "ILWR", "rh"), verbose=True)
        ranks = np.array(
            [spstats.norm.cdf(values) for values in met_vg.data_trans]
        )
        cvine = CVine(
            ranks,
            varnames=met_vg.var_names,
            dtimes=met_vg.times,
            # weights="likelihood"
        )
        quantiles = cvine.quantiles()
        qq_xr = xr.DataArray(
            quantiles,
            dims=("variable", "time"),
            coords=dict(variable=met_vg.var_names, time=met_vg.times),
        )
        ranks_xr = xr.zeros_like(qq_xr)
        ranks_xr.data = ranks

        fig, ax = plt.subplots(nrows=1, ncols=1)
        coeffs = [
            np.corrcoef(group)[0, 1:]
            for month_i, group in qq_xr.groupby(qq_xr.time.dt.month)
        ]
        ax.plot(coeffs)
        ax.set_prop_cycle(None)
        coeffs = [
            np.corrcoef(group)[0, 1:]
            for month_i, group in ranks_xr.groupby(ranks_xr.time.dt.month)
        ]
        ax.plot(coeffs, "--")
        plt.show()

        assert np.all(np.isfinite(quantiles))
        sim = cvine.simulate(randomness=quantiles)
        try:
            npt.assert_almost_equal(sim, ranks, decimal=3)
        except AssertionError:
            fig, axs = plt.subplots(nrows=met_vg.K, ncols=1, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(ranks[i])
                ax.plot(sim[i], "--")
            fig, axs = plt.subplots(
                nrows=met_vg.K, ncols=1, subplot_kw=dict(aspect="equal")
            )
            for i, ax in enumerate(axs):
                ax.scatter(ranks[i], sim[i])
            plt.show()
            raise

    # def test_seasonal_ph(self):
    #     if self.verbose:
    #         print("VG with phase randomization")
    #     import vg
    #     import config_konstanz as conf

    #     vg.conf = vg.vg_base.conf = vg.vg_plotting.conf = conf
    #     met_vg = vg.VG(
    #         (
    #             # 'R',
    #             "theta",
    #             "ILWR",
    #             "rh",
    #             "u",
    #             "v",
    #         ),
    #         refit=True,
    #         verbose=True,
    #     )
    #     met_vg.fit(p=3)
    #     theta_incr = 0.1
    #     vg.reseed(0)
    #     n_realizations = 4
    #     means_norm0, means0, means1 = [], [], []
    #     before = time.perf_counter()
    #     for _ in range(n_realizations):
    #         simt0, sim0 = met_vg.simulate(sim_func=vg_ph)  #  theta_incr=0.,
    #         means_norm0 += [met_vg.sim.mean(axis=1)]
    #         simt1, sim1 = met_vg.simulate(
    #             theta_incr=theta_incr, sim_func=vg_ph
    #         )
    #         prim_i = met_vg.primary_var_ii
    #         means0 += [sim0.mean(axis=1)]
    #         # means1 += [sim1[prim_i].mean()]
    #         # means0 += [sim0[prin_i].mean()]
    #         means1 += [sim1.mean(axis=1)]
    #     print(time.perf_counter() - before)
    #     means0 = np.mean(means0, axis=0)
    #     means1 = np.mean(means1, axis=0)
    #     # print(means0[prim_i] + theta_incr, means1[prim_i])
    #     # print(means0, means1)
    #     np.set_printoptions(suppress=True)
    #     import pandas as pd

    #     data_dict = OrderedDict(
    #         (
    #             ("data_norm", met_vg.data_trans.mean(axis=1)),
    #             ("sim_norm", np.mean(means_norm0, axis=0)),
    #             ("data", (met_vg.data_raw / met_vg.sum_interval).mean(axis=1)),
    #             ("sim", means0),
    #         )
    #     )
    #     means_df = pd.DataFrame(data_dict, index=met_vg.var_names)
    #     print(means_df)
    #     npt.assert_almost_equal(
    #         met_vg.data_trans.mean(axis=1),
    #         np.mean(means_norm0, axis=0),
    #         decimal=2,
    #     )
    #     means0[prim_i] += theta_incr
    #     means_obs = met_vg.data_raw.mean(axis=1) / 24
    #     means_obs[prim_i] += theta_incr
    #     npt.assert_almost_equal(means_obs[prim_i], means1[prim_i], decimal=2)
    #     npt.assert_almost_equal(means0[prim_i], means1[prim_i], decimal=2)

    #     # vg_ph.clear_cache()
    #     # for _ in range(10):
    #     #     simt, sim = met_vg.simulate(
    #     #         theta_incr=4, mean_arrival=7,
    #     #         disturbance_std=5,
    #     #         sim_func=vg_ph)
    #     # vg_ph.vine.plot(edge_labels="copulas")
    #     # # vg_ph.vine.plot_tplom()
    #     # vg_ph.vine.plot_qqplom()
    #     # vg_ph.vine.plot_seasonal()
    #     # # met_vg.plot_all()
    #     # plt.show()


if __name__ == "__main__":
    npt.run_module_suite()
