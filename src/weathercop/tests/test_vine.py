import time
from collections import OrderedDict
import string

import numpy as np
import xarray as xr
import dill
import matplotlib.pyplot as plt
import numpy.testing as npt
import pandas as pd
import pytest
from scipy import stats as spstats

from varwg import helpers as my
import varwg
from varwg.time_series_analysis import distributions as dists
from weathercop import plotting, copulae
from weathercop.vine import CVine, RVine, vg_ph
from weathercop.multisite import set_conf


@pytest.fixture(scope="class")
def vine_test_data():
    """Generate synthetic data for vine tests (class-scoped for speed).

    Creates covariance matrix, synthetic normal data, rank transformations,
    and metadata for vine copula testing.
    """
    varwg.reseed(0)

    cov = np.array(
        [
            [1.3, 0.568, 0.597, 0.507, -0.045, 0.669],
            [0.568, 0.91, 0.288, 0.686, 0.072, 0.129],
            [0.597, 0.288, 0.963, 0.282, 0.576, 0.283],
            [0.507, 0.686, 0.282, 0.892, 0.136, 0.116],
            [-0.045, 0.072, 0.576, 0.136, 0.927, -0.248],
            [0.669, 0.129, 0.283, 0.116, -0.248, 0.988],
        ]
    )

    K = len(cov) - 1
    T = 1000
    data_normal = np.random.multivariate_normal(
        len(cov) * [0], cov, T
    ).T[:K]

    data_ranks = np.array(
        [
            dists.norm.cdf(row, mu=row.mean(), sigma=row.std())
            for row in data_normal
        ]
    )

    varnames = list(string.ascii_lowercase[:K])
    dtimes = pd.date_range("2000-01-01", periods=T, freq="D")

    return {
        "cov": cov,
        "K": K,
        "T": T,
        "data_normal": data_normal,
        "data_ranks": data_ranks,
        "varnames": varnames,
        "dtimes": dtimes,
    }


@pytest.fixture(scope="class")
def cvine_fitted(vine_test_data):
    """Fitted CVine instance (class-scoped, expensive to build).

    Constructs a canonical vine copula from synthetic rank data with
    tau-weighted tree construction.
    """
    data = vine_test_data
    verbose = False  # Set to False for automated tests

    cvine = CVine(
        data["data_ranks"],
        varnames=data["varnames"],
        dtimes=data["dtimes"],
        weights="tau",
        verbose=verbose,
        debug=verbose,
        tau_min=0,
        cop_candidates=None,
    )
    return cvine


@pytest.fixture(scope="class")
def cvine_simulation(cvine_fitted, vine_test_data):
    """Simulation results from CVine (expensive, class-scoped).

    Generates simulated data and quantiles from the fitted vine copula.
    Returns both rank-space and normal-space simulations.
    """
    data_normal = vine_test_data["data_normal"]

    # Simulate from vine
    csim = cvine_fitted.simulate()

    # Transform to normal space for comparison with original data
    csim_normal = np.array(
        [
            dists.norm.ppf(values, mu=source.mean(), sigma=source.std())
            for values, source in zip(csim, data_normal)
        ]
    )

    # Compute quantiles for roundtrip testing
    cquantiles = cvine_fitted.quantiles()

    return {
        "csim": csim,
        "csim_normal": csim_normal,
        "cquantiles": cquantiles,
    }


@pytest.mark.usefixtures("vine_test_data", "cvine_fitted", "cvine_simulation")
class Test:
    """Test suite for vine copula functionality using pytest fixtures."""

    def test_serialize(self, cvine_fitted):
        """Test that vine copulas can be serialized and deserialized with dill."""
        dill_str = dill.dumps(cvine_fitted)
        vine_recovered = dill.loads(dill_str)
        assert vine_recovered is not None

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

    def test_csimulate(self, vine_test_data, cvine_simulation):
        """Test that CVine simulation preserves mean and covariance structure."""
        data_normal = vine_test_data["data_normal"]
        data_ranks = vine_test_data["data_ranks"]
        csim_normal = cvine_simulation["csim_normal"]
        cquantiles = cvine_simulation["cquantiles"]

        npt.assert_almost_equal(
            csim_normal.mean(axis=1),
            data_normal.mean(axis=1),
            decimal=1,
        )
        try:
            npt.assert_almost_equal(
                np.cov(csim_normal), np.cov(data_normal), decimal=1
            )
        except AssertionError:
            print("Obs \n", np.corrcoef(data_ranks).round(3))
            print("Sim \n", np.corrcoef(cquantiles).round(3))
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

    def test_cquantiles_hist(self, vine_test_data, cvine_simulation):
        """Test that quantile marginals are uniformly distributed."""
        cquantiles = cvine_simulation["cquantiles"]
        varnames = vine_test_data["varnames"]

        for i, q in enumerate(cquantiles):
            _, p_value = spstats.kstest(q, spstats.uniform.cdf)
            try:
                assert p_value > 0.25
            except AssertionError:
                label = varnames[i]
                uni = np.random.random(size=q.size)
                fig, ax = my.hist([q, uni], 20, dist=spstats.uniform)
                fig.suptitle("%s p-value: %.3f" % (label, p_value))
                plt.show()
                raise

    def test_cquantiles(self, vine_test_data, cvine_fitted, cvine_simulation):
        """Test quantile simulation roundtrip (quantiles → simulate → should match original ranks)."""
        cov = vine_test_data["cov"]
        K = vine_test_data["K"]
        data_ranks = vine_test_data["data_ranks"]
        varnames = vine_test_data["varnames"]
        cquantiles = cvine_simulation["cquantiles"]

        corr_exp = np.zeros_like(cov)
        corr_exp.ravel()[:: K + 1] = 1
        sim = cvine_fitted.simulate(randomness=cquantiles)
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

        print(cvine_fitted)
        print(cvine_fitted.varnames)
        print(cvine_fitted.varnames_old)
        try:
            npt.assert_almost_equal(sim, data_ranks, decimal=2)
        except AssertionError:
            obs_corr = pd.DataFrame(
                np.corrcoef(data_ranks).round(3),
                index=varnames,
                columns=varnames,
            )
            sim_corr = pd.DataFrame(
                np.corrcoef(sim).round(3),
                index=varnames,
                columns=varnames,
            )
            print("\nObs\n", obs_corr)
            print("\nSim\n", sim_corr)
            print("\nDiff\n", obs_corr - sim_corr)
            fig, axs = plt.subplots(nrows=K, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(data_ranks[i])
                ax.plot(sim[i], "--")
                ax.set_title(varnames[i])

            fig, axs = plt.subplots(
                nrows=int(np.sqrt(K) + 1),
                ncols=int(np.sqrt(K)),
                sharex=True,
                sharey=True,
                subplot_kw=dict(aspect="equal"),
            )
            axs = np.ravel(axs)
            for i, ax in enumerate(axs[:K]):
                ax.scatter(data_ranks[i], sim[i], marker="x", s=2)
                ax.plot([0, 1], [0, 1], color="black")
                ax.grid(True)
                ax.set_title(varnames[i])
            for ax in axs[K:]:
                ax.set_axis_off()

            cvine_fitted.plot(edge_labels="copulas")
            plt.show()
            raise

    def test_cquantiles_corr(self, vine_test_data, cvine_simulation):
        """Test that quantiles have zero correlation (are independent)."""
        K = vine_test_data["K"]
        cquantiles = cvine_simulation["cquantiles"]

        corr_exp = np.zeros((K, K), dtype=float)
        corr_exp.ravel()[:: K + 1] = 1
        corr_act = np.corrcoef(cquantiles)
        try:
            npt.assert_almost_equal(corr_act, corr_exp, decimal=1)
        except AssertionError:
            print("Quantiles corr \n", corr_act.round(3))
            raise

    # def test_seasonal_yearly(self):
    #     if self.verbose:
    #         print("Seasonal Vine with singe year VG data")
    #         import varwg
    #         from vg import varwg_plotting, vg_base
    #         import config_konstanz_disag as conf
    #     varwg.conf = vg_plotting.conf = vg_base.conf = conf
    #     # met_vg = varwg.VG(("theta", "ILWR", "rh", "R"), verbose=True)
    #     met_vg = varwg.VG(("theta", "Qsw", "ILWR", "rh", "u", "v"), verbose=True)
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
        """Test seasonal vine with real VG data (skipped if dependencies unavailable)."""
        # Try to import config_konstanz, skip if not available
        try:
            import config_konstanz as conf
            import varwg
            from varwg import plotting as vg_plotting, base as vg_base
            varwg.conf = vg_plotting.conf = vg_base.conf = conf
        except (ImportError, ModuleNotFoundError):
            pytest.skip("config_konstanz not available for test_seasonal")

        # Try to load VG data, skip if it fails (e.g., due to module import issues)
        try:
            met_vg = varwg.VG(("theta", "ILWR", "rh"), verbose=False)
        except (ModuleNotFoundError, ImportError) as e:
            pytest.skip(f"Could not load VG data: {e}")
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
    #     import varwg
    #     import config_konstanz as conf

    #     varwg.conf = varwg.vg_base.conf = varwg.vg_plotting.conf = conf
    #     met_vg = varwg.VG(
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
    #     varwg.reseed(0)
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
