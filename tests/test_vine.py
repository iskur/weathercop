import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from scipy import stats as spstats

from weathercop.vine import RVine, CVine, stats, vg_ph
from weathercop import plotting
from lhglib.contrib.time_series_analysis import distributions as dists
from lhglib.contrib import dirks_globals as my


class Test(npt.TestCase):

    def setUp(self):
        np.random.seed(0)
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
        self.cov = np.array([[ 1.3  ,  0.568,  0.597,  0.507, -0.045,  0.669],
                             [ 0.568,  0.91 ,  0.288,  0.686,  0.072,  0.129],
                             [ 0.597,  0.288,  0.963,  0.282,  0.576,  0.283],
                             [ 0.507,  0.686,  0.282,  0.892,  0.136,  0.116],
                             [-0.045,  0.072,  0.576,  0.136,  0.927, -0.248],
                             [ 0.669,  0.129,  0.283,  0.116, -0.248,  0.988]])

        self.K = len(self.cov)
        T = 1000
        self.data_normal = np.random.multivariate_normal(len(self.cov) * [0],
                                                         self.cov, T).T
        self.data_ranks = np.array([dists.norm.cdf(row,
                                                   mu=row.mean(),
                                                   sigma=row.std())
                                    for row in self.data_normal])
        # varnames = list("".join("%d" % i for i in range(self.K)))
        self.varnames = "R", "theta", "ILWR", "rh", "u", "v"
        # self.varnames = 'R', 'theta', 'u', 'ILWR', 'rh', 'v'

        # self.varnames = 'R', 'u', 'ILWR', 'v', 'theta', 'rh'
        # weights = "likelihood"
        weights = "tau"

        self.rvine = RVine(self.data_ranks, varnames=self.varnames,
                           weights=weights, verbose=True)
        self.rsim = self.rvine.simulate(T=3 * T)
        self.rsim_normal = np.array([dists.norm.ppf(values,
                                                    mu=source.mean(),
                                                    sigma=source.std())
                                     for values, source
                                     in zip(self.rsim, self.data_normal)])
        self.rquantiles = self.rvine.quantiles()
 
        self.cvine = CVine(self.data_ranks, varnames=self.varnames,
                           weights=weights, verbose=True)
        self.csim = self.cvine.simulate(T=3 * T)
        self.csim_normal = np.array([dists.norm.ppf(values,
                                                    mu=source.mean(),
                                                    sigma=source.std())
                                    for values, source
                                    in zip(self.csim, self.data_normal)])
        self.cquantiles = self.cvine.quantiles()

    def test_likelihood_tree(self):
        if self.verbose:
            print("Testing tree construction with likelihood as weight")
        rvine = RVine(self.data_ranks, varnames=self.varnames,
                      verbose=self.verbose, weights="likelihood")
        fig, ax = rvine.plot(edge_labels="copulas")
        fig.suptitle("likelihood")
        fig, ax = rvine.plot_tplom()
        fig.suptitle("likelihood")
        fig, ax = rvine.plot_qqplom()
        fig.suptitle("likelihood")

        rvine = RVine(self.data_ranks, varnames=self.varnames,
                      verbose=self.verbose, weights="tau")
        fig, ax = self.rvine.plot(edge_labels="copulas")
        fig.suptitle("tau")
        fig, ax = self.rvine.plot_tplom()
        fig.suptitle("tau")
        fig, ax = self.rvine.plot_qqplom()
        fig.suptitle("tau")
        plt.show()
        
    def test_rsimulate(self):
        if self.verbose:
            print("Covariance matrix of RVine simulation")
        try:
            npt.assert_almost_equal(np.cov(self.rsim_normal),
                                    np.cov(self.data_normal),
                                    decimal=1)
        except AssertionError:
            plotting.ccplom(self.rsim, k=0, kind="img",
                            title="simulated from RVine")
            plotting.ccplom(self.data_ranks, k=0, kind="img", title="observed")
            plt.show()
            raise

    def test_csimulate(self):
        if self.verbose:
            print("Covariance matrix of CVine simulation")
        try:
            npt.assert_almost_equal(np.cov(self.csim_normal),
                                    np.cov(self.data_normal),
                                    decimal=1)
        except AssertionError:
            plotting.ccplom(self.rsim, k=0, kind="img",
                            title="simulated from CVine")
            plotting.ccplom(self.data_ranks, k=0, kind="img", title="observed")
            plt.show()
            raise

    def test_rquantiles(self):
        if self.verbose:
            print("Quantile simulation roundtrip of RVine")
        # does it matter if we specify the ranks explicitly?
        quantiles_self = self.rvine.quantiles(ranks=self.rvine.ranks)
        try:
            npt.assert_almost_equal(quantiles_self, self.rquantiles)
        except AssertionError:
            fig, axs = plt.subplots(nrows=self.K, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(self.rquantiles[i])
                ax.plot(quantiles_self[i], "--")
            # plt.show()
            # raise

        sim = self.rvine.simulate(randomness=self.rquantiles)
        try:
            npt.assert_almost_equal(sim, self.data_ranks, decimal=2)
        except AssertionError:
            fig, axs = plt.subplots(nrows=self.K, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(self.data_ranks[i])
                ax.plot(sim[i], "--")
                ax.set_ylabel(self.varnames[i])

            fig, axs = plt.subplots(nrows=int(np.sqrt(self.K)) + 1,
                                    ncols=int(np.sqrt(self.K)),
                                    sharex=True, sharey=True,
                                    subplot_kw=dict(aspect="equal"))
            axs = np.ravel(axs)
            for i, ax in enumerate(axs[:self.K]):
                ax.scatter(self.data_ranks[i], sim[i],
                           marker="x", s=2)
                ax.plot([0, 1], [0, 1], color="black")
                ax.grid(True)
                ax.set_title(self.varnames[i])
            for ax in axs[self.K:]:
                ax.set_axis_off()
            self.rvine.plot(edge_labels="copulas")
            plt.show()
            raise

    def test_rquantiles_corr(self):
        if self.verbose:
            print("Zero correlation matrix of RVine quantiles")
        corr_exp = np.zeros_like(self.cov)
        corr_exp.ravel()[::self.K+1] = 1
        try:
            npt.assert_almost_equal(np.corrcoef(self.rquantiles),
                                    corr_exp,
                                    decimal=1)
        except AssertionError:
            plotting.ccplom(self.rquantiles, k=0, kind="img",
                            title="rquantiles")
            plotting.ccplom(self.data_ranks, k=0, kind="img",
                            title="copula_input")
            self.rvine.plot_qqplom()
            self.rvine.plot(edge_labels="copulas")
            plt.show()
            raise

    def test_cquantiles_hist(self):
        for i, q in enumerate(self.cquantiles):
            _, p_value = spstats.kstest(q, spstats.uniform.cdf)
            try:
                assert p_value > .25
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
        corr_exp.ravel()[::self.K+1] = 1
        try:
            npt.assert_almost_equal(np.corrcoef(self.cquantiles),
                                    corr_exp,
                                    decimal=2)
        except AssertionError:
            plotting.ccplom(self.cquantiles, k=0, kind="img",
                            title="cquantiles")
            plotting.ccplom(self.data_ranks, k=0, kind="img",
                            title="copula_input")
            self.cvine.plot_qqplom()
            # plt.show()
            # raise

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

        sim = self.cvine.simulate(randomness=self.cquantiles)
        try:
            npt.assert_almost_equal(sim, self.data_ranks, decimal=2)
        except AssertionError:
            fig, axs = plt.subplots(nrows=self.K, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(self.data_ranks[i])
                ax.plot(sim[i], "--")

            fig, axs = plt.subplots(nrows=int(np.sqrt(self.K) + 1),
                                    ncols=int(np.sqrt(self.K)),
                                    sharex=True, sharey=True,
                                    subplot_kw=dict(aspect="equal"))
            axs = np.ravel(axs)
            for i, ax in enumerate(axs[:self.K]):
                ax.scatter(self.data_ranks[i], sim[i],
                           marker="x", s=2)
                ax.plot([0, 1], [0, 1], color="black")
                ax.grid(True)
                ax.set_title(self.varnames[i])
            for ax in axs[self.K:]:
                ax.set_axis_off()

            self.cvine.plot(edge_labels="copulas")
            plt.show()
            raise

    def test_cquantiles_corr(self):
        if self.verbose:
            print("Zero correlation matrix of RVine quantiles")
        corr_exp = np.zeros_like(self.cov)
        corr_exp.ravel()[::self.K+1] = 1
        try:
            npt.assert_almost_equal(np.corrcoef(self.cquantiles),
                                    corr_exp,
                                    decimal=1)
        except AssertionError:
            plotting.ccplom(self.cquantiles, k=0, kind="img",
                            title="cquantiles")
            plotting.ccplom(self.data_ranks, k=0, kind="img",
                            title="copula_input")
            self.cvine.plot_qqplom()
            fig, axs = plt.subplots(nrows=self.K, ncols=1, sharex=True)
            plt.show()
            raise

    def test_seasonal(self):
        if self.verbose:
            print("Seasonal Vine with VG data")
        from lhglib.contrib.veathergenerator import vg, vg_plotting, vg_base
        from lhglib.contrib.veathergenerator import config_konstanz_disag as conf
        vg.conf = vg_plotting.conf = vg_base.conf = conf
        # met_vg = vg.VG(("theta", "ILWR", "rh", "R"), verbose=True)
        met_vg = vg.VG(("theta", "ILWR"), verbose=True)
        ranks = np.array([spstats.norm.cdf(values)
                          for values in met_vg.data_trans])
        cvine = CVine(ranks, varnames=met_vg.var_names,
                      dtimes=met_vg.times,
                      weights="likelihood"
                      )
        quantiles = cvine.quantiles()
        assert np.all(np.isfinite(quantiles))
        sim = cvine.simulate(randomness=quantiles)
        try:
            npt.assert_almost_equal(sim, ranks, decimal=3)
        except AssertionError:
            fig, axs = plt.subplots(nrows=met_vg.K, ncols=1, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(ranks[i])
                ax.plot(sim[i], "--")
            fig, axs = plt.subplots(nrows=met_vg.K, ncols=1,
                                    subplot_kw=dict(aspect="equal"))
            for i, ax in enumerate(axs):
                ax.scatter(ranks[i], sim[i])
            plt.show()
            raise

    def test_seasonal_ph(self):
        if self.verbose:
            print("VG with phase randomization")
        from lhglib.contrib.veathergenerator import vg
        from lhglib.contrib.veathergenerator import config_konstanz_disag as conf
        from lhglib.contrib.veathergenerator import vg_plotting
        vg.conf = vg.vg_base.conf = vg_plotting.conf = conf
        met_vg = vg.VG((
                        # 'R',
                        'theta', 'ILWR', 'rh',
                        # 'u', 'v'
                       ),
                       # refit=True,
                       verbose=True)
        met_vg.fit(p=3)
        vg_ph.clear_cache()
        simt, sim = met_vg.simulate(
            # theta_incr=4, mean_arrival=7,
            # disturbance_std=5,
            sim_func=vg_ph)
        # met_vg.plot_all()
        # plt.show()


if __name__ == "__main__":
    npt.run_module_suite()
