import time
import multiprocessing
from itertools import repeat
import numpy as np
import numpy.testing as npt
import varwg
from varwg.time_series_analysis import distributions as dists
from weathercop.vine import CVine, cquant_py, csim_py
from weathercop.cvine import csim as csim_cy, cquant as cquant_cy

n_nodes = multiprocessing.cpu_count() - 1


class Test(npt.TestCase):
    def setUp(self):
        varwg.reseed(0)
        self.verbose = True
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
        T = 1000
        self.data_normal = np.random.multivariate_normal(
            len(self.cov) * [0], self.cov, T
        ).T
        self.data_ranks = np.array(
            [
                dists.norm.cdf(row, mu=row.mean(), sigma=row.std())
                for row in self.data_normal
            ]
        )
        # varnames = list("".join("%d" % i for i in range(self.K)))
        self.varnames = "R", "theta", "ILWR", "rh", "u", "v"
        # self.varnames = 'R', 'theta', 'u', 'ILWR', 'rh', 'v'

        # self.varnames = 'R', 'u', 'ILWR', 'v', 'theta', 'rh'
        # weights = "likelihood"
        weights = "tau"
        self.cvine = CVine(
            self.data_ranks,
            varnames=self.varnames,
            weights=weights,
            verbose=self.verbose,
            debug=self.verbose,
            tau_min=0,
        )
        self.csim = self.cvine.simulate(T=3 * T)
        self.csim_normal = np.array(
            [
                dists.norm.ppf(values, mu=source.mean(), sigma=source.std())
                for values, source in zip(self.csim, self.data_normal)
            ]
        )
        self.cquantiles = self.cvine.quantiles()

    def tearDown(self):
        pass

    def test_csim_one(self):
        Ps = self.csim
        T = Ps.shape[1]
        tt = np.arange(T)
        stop_at = self.K
        zero = 1e-5
        one = 1 - zero
        time0 = time.time()
        print("python")
        Us_py = csim_py((Ps, self.cvine, zero, one, tt, stop_at))
        time_py = time.time()
        print("cython")
        Us_cy = csim_cy((Ps, self.cvine, zero, one, tt, stop_at))
        time_cy = time.time()
        print("multiprocessing")
        # with multiprocessing.Pool(n_nodes) as pool:
        #     Us_mp = pool.map(csim_one_cy,
        #                      zip(range(T),
        #                          Ps.T,
        #                          repeat(self.cvine),
        #                          repeat(zero),
        #                          repeat(one)),
        #                      chunksize=(T // n_nodes))
        # Us_mp = np.array(Us_mp).T
        # time_mp = time.time() - time_cy
        time_cy = time_cy - time_py
        time_py = time_py - time0
        print("Time python: ", time_py * 1e6)
        print("Time cython: ", time_cy * 1e6)
        # print("Time multip: ", time_mp * 1e6)
        try:
            npt.assert_almost_equal(Us_py, Us_cy, decimal=3)
        except AssertionError:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(
                nrows=Ps.shape[0], ncols=1, subplot_kw=dict(aspect="equal")
            )
            for ax, py, cy in zip(axs, Us_py, Us_cy):
                ax.scatter(py, cy)
            plt.show()
            raise

    def test_cquant_one(self):
        Us = self.data_ranks
        T = Us.shape[1]
        tt = np.arange(T)
        zero = 1e-5
        one = 1 - zero
        time0 = time.time()
        print("python")
        Ps_py = cquant_py((Us, self.cvine, zero, one, tt))
        time_py = time.time()
        print("cython")
        Ps_cy = cquant_cy((Us, self.cvine, zero, one, tt))
        time_cy = time.time()
        # print("multiprocessing")
        # with multiprocessing.Pool(n_nodes) as pool:
        #     Ps_mp = pool.map(cquant_one_cy,
        #                      zip(range(T),
        #                          Us.T,
        #                          repeat(self.cvine),
        #                          repeat(zero),
        #                          repeat(one)),
        #                      chunksize=(T // n_nodes))
        # Ps_mp = np.array(Ps_mp).T
        # time_mp = time.time() - time_cy
        time_cy = time_cy - time_py
        time_py = time_py - time0
        print("Time python: ", time_py * 1e6)
        print("Time cython: ", time_cy * 1e6)
        # print("Time multip: ", time_mp * 1e6)
        try:
            npt.assert_almost_equal(Ps_py, Ps_cy, decimal=4)
        except AssertionError:
            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(
                nrows=Us.shape[0], ncols=1, subplot_kw=dict(aspect="equal")
            )
            for ax, py, cy in zip(axs, Ps_py, Ps_cy):
                ax.scatter(py, cy)
            plt.show()
            raise


if __name__ == "__main__":
    npt.run_module_suite()
