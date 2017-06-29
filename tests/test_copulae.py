import warnings
from scipy import integrate
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from weathercop import copulae as cop


class Test(npt.TestCase):

    def setUp(self):
        self.verbose = True
        self.eps = 1e-6

    def tearDown(self):
        pass

    def test_cop(self):
        """A few very basic copula requirements."""
        if self.verbose:
            print("\nTesting copula function")
        for name, frozen_cop in cop.frozen_cops:
            if self.verbose:
                print(name)
            zero = frozen_cop.copula_func(self.eps, self.eps)
            one = frozen_cop.copula_func(1 - self.eps, 1 - self.eps)
            try:
                npt.assert_almost_equal(zero, self.eps, decimal=4)
                npt.assert_almost_equal(one, 1 - self.eps, decimal=4)
            except AssertionError:
                frozen_cop.plot_copula()  # theta=frozen_cop.theta)
                plt.show()
                raise

    def test_cdf_given_u(self):
        if self.verbose:
            print("\nTesting u-conditional cdfs")
        for name, copulas in cop.frozen_cops:
            if self.verbose:
                print(name)
            zero = copulas.cdf_given_u(1 - self.eps, self.eps)
            one = copulas.cdf_given_u(self.eps, 1 - self.eps)
            try:
                npt.assert_almost_equal(zero, self.eps, decimal=4)
                npt.assert_almost_equal(one, 1 - self.eps, decimal=4)
            except AssertionError:
                print(zero, self.eps)
                print(one, 1 - self.eps)
                warnings.warn(name.upper())

    def test_cdf_given_v(self):
        if self.verbose:
            print("\nTesting v-conditional cdfs by roundtrip")
        for name, copulas in cop.frozen_cops:
            if self.verbose:
                print(name)
            zero = copulas.cdf_given_v(self.eps, 1 - self.eps)
            one = copulas.cdf_given_v(1 - self.eps, self.eps)
            try:
                npt.assert_almost_equal(zero, self.eps, decimal=4)
                npt.assert_almost_equal(one, 1 - self.eps, decimal=4)
            except AssertionError:
                print(zero, self.eps)
                print(one, 1 - self.eps)
                warnings.warn(name.upper())

    def test_inv_cdf_given_u(self):
        if self.verbose:
            print("\nTesting inverse u-conditional cdfs by roundtrip")
        for name, copulas in cop.frozen_cops:
            if self.verbose:
                print(name)
            uu = np.linspace(self.eps, 1 - self.eps, 100)
            vv_exp = np.copy(uu)
            qq = copulas.cdf_given_u(uu, vv_exp)
            vv_actual = copulas.inv_cdf_given_u(uu, np.squeeze(qq))
            try:
                npt.assert_almost_equal(vv_actual, vv_exp, decimal=2)
            except AssertionError:
                if not self.verbose:
                    raise
                fig, ax = plt.subplots()
                ax.scatter(vv_actual, vv_exp)
                ax.set_xlabel("vv_actual")
                ax.set_ylabel("vv_exp")
                ax.set_title("%s inv_cdf_given_u" % name)
                plt.show()

    def test_inv_cdf_given_v(self):
        if self.verbose:
            print("\nTesting inverse v-conditional cdfs")
        for name, copulas in cop.frozen_cops:
            if self.verbose:
                print(name)
            vv = np.linspace(self.eps, 1 - self.eps, 100)
            uu_exp = np.copy(vv)
            qq = copulas.cdf_given_v(uu_exp, vv)
            uu_actual = copulas.inv_cdf_given_v(vv, np.squeeze(qq))
            try:
                npt.assert_almost_equal(uu_actual, uu_exp, decimal=2)
            except AssertionError:
                if not self.verbose:
                    raise
                fig, ax = plt.subplots()
                ax.scatter(uu_actual, uu_exp)
                ax.set_xlabel("uu_actual")
                ax.set_ylabel("uu_exp")
                ax.set_title("%s inv_cdf_given_v" % name)
                plt.show()

    def test_density(self):
        """Does the density integrate to 1?"""
        if self.verbose:
            print("\nTesting density by numerical integration")
        for name, frozen_cop in cop.frozen_cops:
            if self.verbose:
                print(name)
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    one = integrate.nquad(frozen_cop.density,
                                          ([self.eps, 1 - self.eps],
                                           [self.eps, 1 - self.eps]))[0]
                except integrate.IntegrationWarning:
                    print("Numerical integration of %s is problematic" % name)
                else:
                    npt.assert_almost_equal(one, 1., decimal=5)

    def test_fit(self):
        """Is fit able to reproduce parameters of a self-generated sample?
        """
        if self.verbose:
            print("\nTesting fit")
        np.random.seed(1)
        for name, copulas in cop.all_cops.items():
            if self.verbose:
                print(name)
            sample_x, sample_y = copulas.sample(10000, copulas.theta_start)
            try:
                fitted_theta = copulas.fit(sample_x, sample_y)
            except cop.NoConvergence:
                print("... fitting did not converge.")
            else:
                if copulas.theta_start[0] is None and fitted_theta is None:
                    # for the independence copula
                    continue
                try:
                    npt.assert_almost_equal(fitted_theta, copulas.theta_start,
                                            decimal=1)
                except AssertionError:
                    if not self.verbose:
                        raise
                    ax = copulas.plot_density()
                    ax.scatter(sample_x, sample_y, marker="x",
                               facecolor=(0, 0, 1, .1))
                    plt.show()
                    raise


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    npt.run_module_suite()
