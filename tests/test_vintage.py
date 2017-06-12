import os

import numpy as np
import numpy.testing as npt
from matplotlib import pyplot as plt

from weathercop import plotting, vintage, cop_conf, stats


class Test(npt.TestCase):

    def setUp(self):
        data_filepath = os.path.join(cop_conf.weathercop_dir, "code",
                                     "vg_data.npz")
        varnames = "theta ILWR Qsw rh u v".split()
        self.K = len(varnames)
        data_varnames = "R theta Qsw ILWR rh u v".split()
        with np.load(data_filepath, encoding="bytes") as saved:
            data_all = saved["all"]
            dtimes = saved["dtimes"]

        self.data = np.array([data_all[data_varnames.index(varname)]
                              for varname in varnames])
        self.data_ranks = np.array([stats.rel_ranks(row)
                                    for row in self.data])
        self.vint = vintage.Vintage(self.data_ranks, dtimes,
                                    varnames=varnames, window_len=30)
        self.vint.setup()
        self.quantiles = self.vint.quantiles()
        self.sim_ranks = self.vint.simulate()

    def test_quantiles_corr(self):
        corr_exp = np.zeros((self.K, self.K))
        corr_exp.ravel()[::self.K+1] = 1
        try:
            
            npt.assert_almost_equal(np.corrcoef(self.quantiles),
                                    corr_exp,
                                    decimal=2)
        except AssertionError:
            plotting.ccplom(self.quantiles, k=0, kind="img",
                            title="rquantiles")
            plotting.ccplom(self.data_ranks, k=0, kind="img",
                            title="copula_input")
            plt.show()
            raise

    def test_quantiles(self):
        # does it matter if we specify the ranks explicitly?
        quantiles_self = self.vint.quantiles(ranks=self.vint.ranks)
        try:
            npt.assert_almost_equal(quantiles_self, self.quantiles)
        except AssertionError:
            fig, axs = plt.subplots(nrows=self.K, sharex=True)
            for i, ax in enumerate(axs):
                ax.plot(self.quantiles[i])
                ax.plot(quantiles_self[i], "--")
            plt.show()

        sim = self.vint.simulate(randomness=self.quantiles)
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

    def test_simulate(self):
        try:
            npt.assert_almost_equal(np.cov(self.sim_ranks),
                                    np.cov(self.data_ranks),
                                    decimal=1)
        except AssertionError:
            plotting.ccplom(self.sim_ranks, k=0, kind="img",
                            title="simulated from RVine")
            plotting.ccplom(self.data_ranks, k=0, kind="img", title="observed")
            plt.show()
            raise


if __name__ == "__main__":
    import warnings
    warnings.simplefilter('ignore', category=DeprecationWarning)
    npt.run_module_suite("test_vintage.py")
