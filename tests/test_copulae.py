import warnings
from scipy import integrate
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from weathercop import copulae as cop


class Test(npt.TestCase):

    def setUp(self):
        self.verbose = False

    def tearDown(self):
        pass

    def test_cop(self):
        """A few very basic copula requirements."""
        if self.verbose:
            print("\nTesting copula function")
        for name, frozen_cop in cop.frozen_cops:
            if self.verbose:
                print(name)
            zero = frozen_cop.copula_func(1e-9, 1e-9)
            one = frozen_cop.copula_func(1 - 1e-9, 1 - 1e-9)
            try:
                npt.assert_almost_equal(zero, 0, decimal=4)
                npt.assert_almost_equal(one, 1, decimal=4)
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
            zero = copulas.cdf_given_u(1 - 1e-9, 1e-9)
            one = copulas.cdf_given_u(1e-9, 1 - 1e-9)
            try:
                npt.assert_almost_equal(zero, 0, decimal=4)
                npt.assert_almost_equal(one, 1, decimal=4)
            except AssertionError:
                print(zero, 0)
                print(one, 1)
                warnings.warn(name.upper())

    def test_cdf_given_v(self):
        if self.verbose:
            print("\nTesting v-conditional cdfs by roundtrip")
        for name, copulas in cop.frozen_cops:
            print(name)
            zero = copulas.cdf_given_v(1e-9, 1 - 1e-9)
            one = copulas.cdf_given_v(1 - 1e-9, 1e-9)
            try:
                npt.assert_almost_equal(zero, 0, decimal=4)
                npt.assert_almost_equal(one, 1, decimal=4)
            except AssertionError:
                print(zero, 0)
                print(one, 1)
                warnings.warn(name.upper())

    def test_inv_cdf_given_u(self):
        if self.verbose:
            print("\nTesting inverse u-conditional cdfs by roundtrip")
        for name, copulas in cop.frozen_cops:
            if self.verbose:
                print(name)
            uu = np.linspace(1e-9, 1 - 1e-9, 100)
            vv_exp = np.copy(uu)
            qq = copulas.cdf_given_u(uu, vv_exp)
            vv_actual = copulas.inv_cdf_given_u(uu, np.squeeze(qq))
            npt.assert_almost_equal(vv_actual, vv_exp, decimal=2)

    def test_inv_cdf_given_v(self):
        if self.verbose:
            print("\nTesting inverse v-conditional cdfs")
        for name, copulas in cop.frozen_cops:
            if self.verbose:
                print(name)
            vv = np.linspace(1e-9, 1 - 1e-9, 100)
            uu_exp = np.copy(vv)
            qq = copulas.cdf_given_v(uu_exp, vv)
            uu_actual = copulas.inv_cdf_given_v(vv, np.squeeze(qq))
            npt.assert_almost_equal(uu_actual, uu_exp, decimal=2)

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
                                          ([1e-9, 1 - 1e-9],
                                           [1e-9, 1 - 1e-9]))[0]
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
                npt.assert_almost_equal(fitted_theta, copulas.theta_start,
                                        decimal=1)


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    npt.run_module_suite()
