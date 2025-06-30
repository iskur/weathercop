import time
import numpy as np
import numpy.testing as npt
from scipy.special import erf, erfinv
import matplotlib.pyplot as plt
from weathercop import normal_conditional
from weathercop import copulae as cop


class Test(npt.TestCase):
    def setUp(self):
        self.verbose = True

    def test_inverse_conditional(self):
        uu = np.linspace(0, 1, 100).repeat(100)
        qq = np.concatenate(100 * [np.linspace(0, 1, 100)])
        theta = np.full_like(uu, 0.75)
        time0 = time.perf_counter()
        vv_act = np.empty_like(uu)
        normal_conditional.norm_inv_cdf_given_u(uu, qq, theta, vv_act)
        time_cy = time.perf_counter()
        vv_exp = cop.gaussian.inv_cdf_given_u(uu, qq, theta)
        time_py = time.perf_counter() - time_cy
        time_cy -= time0
        if self.verbose:
            print("Inverse conditional")
            print("Time python: ", time_py * 1e6)
            print("Time cython: ", time_cy * 1e6)
        npt.assert_almost_equal(vv_act, vv_exp)

    def test_conditional(self):
        n_points = 250
        uu = np.linspace(0, 1, n_points).repeat(n_points)
        vv = np.concatenate(n_points * [np.linspace(0, 1, n_points)])
        theta = np.full_like(uu, 0.75)
        time0 = time.perf_counter()
        qq_act = np.empty_like(uu)
        normal_conditional.norm_cdf_given_u(uu, vv, theta, qq_act)
        time_cy = time.perf_counter()
        qq_exp = cop.gaussian.cdf_given_u(uu, vv, theta)
        time_py = time.perf_counter() - time_cy
        time_cy -= time0
        if self.verbose:
            print("Conditional")
            print("Time python: ", time_py * 1e6)
            print("Time cython: ", time_cy * 1e6)
        try:
            npt.assert_almost_equal(qq_act, qq_exp)
        except AssertionError:
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(12, 12),
                subplot_kw=dict(aspect="equal"),
            )
            ax.scatter(
                qq_exp,
                qq_act,
                s=4,
                facecolor=(0, 0, 0, 0.8),
                edgecolor=(0, 0, 0, 0),
            )
            ax.set_title(
                "2 Implementations of the conditional gaussian copula cdf (one has a bug!)"
            )
            plt.show()
            raise

    def test_erf(self):
        xx = np.linspace(-2, 2, 10000)
        q_act = np.empty_like(xx)
        time0 = time.perf_counter()
        normal_conditional.erf(xx, q_act)
        time_cy = time.perf_counter()
        q_exp = erf(xx)
        time_py = time.perf_counter() - time_cy
        time_cy -= time0
        if self.verbose:
            print("erf")
            print("Time scipy: ", time_py * 1e6)
            print("Time mkl: ", time_cy * 1e6)
        npt.assert_almost_equal(q_act, q_exp)

    def test_erfinv(self):
        xx = np.linspace(-1, 1, 10000)
        q_act = np.empty_like(xx)
        time0 = time.perf_counter()
        normal_conditional.erfinv(xx, q_act)
        time_cy = time.perf_counter()
        q_exp = erfinv(xx)
        time_py = time.perf_counter() - time_cy
        time_cy -= time0
        if self.verbose:
            print("erfinv")
            print("Time scipy: ", time_py * 1e6)
            print("Time mkl: ", time_cy * 1e6)
        npt.assert_almost_equal(q_act, q_exp)

    def tearDown(self):
        pass


if __name__ == "__main__":
    npt.run_module_suite()
