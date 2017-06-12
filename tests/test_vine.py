import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from weathercop.vine import RVine, CVine, stats
from weathercop import plotting
from lhglib.contrib.time_series_analysis import distributions


class Test(npt.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.cov = [[1.5, 1., -1., 2.],
                    [1., 1., 0.1, .5],
                    [-1., 0.1, 2., -.75],
                    [2., .5, -.75, 1.5]]
        self.K = len(self.cov)
        T = 1000
        self.data_normal = np.random.multivariate_normal(len(self.cov) * [0],
                                                         self.cov, T).T
        self.data_ranks = np.array([stats.rel_ranks(row)
                                    for row in self.data_normal])
        varnames = "".join("%d" % i for i in range(len(self.cov)))
        self.rvine = RVine(self.data_ranks, varnames=varnames, verbose=True)
        self.cvine = CVine(self.data_ranks, varnames=varnames, verbose=True)
        self.rsim = self.rvine.simulate(T=3 * T)
        self.csim = self.cvine.simulate(T=3 * T)
        self.rsim_normal = np.array([distributions.norm.ppf(values,
                                                            mu=source.mean(),
                                                            sigma=source.std())
                                    for values, source
                                    in zip(self.rsim, self.data_normal)])
        self.csim_normal = np.array([distributions.norm.ppf(values,
                                                            mu=source.mean(),
                                                            sigma=source.std())
                                    for values, source
                                    in zip(self.csim, self.data_normal)])
        self.rquantiles = self.rvine.quantiles()
        self.cquantiles = self.cvine.quantiles()

    def test_rsimulate(self):
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
        # does it matter if we specify the ranks explicitly?
        quantiles_self = self.rvine.quantiles(ranks=self.rvine.ranks)
        try:
            npt.assert_almost_equal(quantiles_self, self.rquantiles)
        except AssertionError:
            fig, axs = plt.subplots(nrows=self.K, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(self.rquantiles[i])
                ax.plot(quantiles_self[i], "--")
            plt.show()

        sim = self.rvine.simulate(randomness=self.rquantiles)
        try:
            npt.assert_almost_equal(sim, self.data_ranks, decimal=2)
        except AssertionError:
            fig, axs = plt.subplots(nrows=self.K, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(self.data_ranks[i])
                ax.plot(sim[i], "--")

            fig, axs = plt.subplots(nrows=int(np.sqrt(self.K)),
                                    ncols=int(np.sqrt(self.K)),
                                    sharex=True, sharey=True,
                                    subplot_kw=dict(aspect="equal"))
            for i, ax in enumerate(np.ravel(axs)):
                ax.scatter(self.data_ranks[i], sim[i],
                           marker="x", s=2)
                ax.plot([0, 1], [0, 1])
                ax.grid(True)
            plt.show()
            raise

    def test_rquantiles_corr(self):
        corr_exp = np.zeros_like(self.cov)
        corr_exp.ravel()[::self.K+1] = 1
        try:
            npt.assert_almost_equal(np.corrcoef(self.rquantiles),
                                    corr_exp,
                                    decimal=2)
        except AssertionError:
            plotting.ccplom(self.rquantiles, k=0, kind="img",
                            title="rquantiles")
            plotting.ccplom(self.data_ranks, k=0, kind="img",
                            title="copula_input")
            plt.show()
            raise

    def test_cquantiles(self):
        # does it matter if we specify the ranks explicitly?
        quantiles_self = self.cvine.quantiles(ranks=self.cvine.ranks)
        try:
            npt.assert_almost_equal(quantiles_self, self.cquantiles)
        except AssertionError:
            fig, axs = plt.subplots(nrows=self.K, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(self.cquantiles[i])
                ax.plot(quantiles_self[i], "--")
            plt.show()

        sim = self.cvine.simulate(randomness=self.cquantiles)
        try:
            npt.assert_almost_equal(sim, self.data_ranks, decimal=2)
        except AssertionError:
            fig, axs = plt.subplots(nrows=self.K, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(self.data_ranks[i])
                ax.plot(sim[i], "--")

            fig, axs = plt.subplots(nrows=np.sqrt(self.K),
                                    ncols=np.sqrt(self.K),
                                    sharex=True, sharey=True,
                                    subplot_kw=dict(aspect="equal"))
            for i, ax in enumerate(np.ravel(axs)):
                ax.scatter(self.data_ranks[i], sim[i],
                           marker="x", s=2)
                ax.plot([0, 1], [0, 1])
                ax.grid(True)
            plt.show()
            raise

    def test_cquantiles_corr(self):
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
            plt.show()
            raise


if __name__ == "__main__":
    npt.run_module_suite()
