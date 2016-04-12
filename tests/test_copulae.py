import warnings
from scipy import integrate
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from weathercop import copulae as cop

frozen_cops = tuple((name, copulas(copulas.theta_start))
                    for name, copulas in sorted(cop.all_cops.items()))


class Test(npt.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_cop(self):
        """A few very basic copula requirements."""
        print("\nTesting copula function")
        for name, frozen_cop in frozen_cops:
            print(name)
            zero = frozen_cop.copula_func(1e-9, 1e-9)
            one = frozen_cop.copula_func(1 - 1e-9, 1 - 1e-9)
            try:
                npt.assert_almost_equal(zero, 0)
                npt.assert_almost_equal(one, 1)
            except AssertionError:
                frozen_cop.plot_copula(theta=frozen_cop.theta)
                plt.show()

    def test_cdf_given_u(self):
        print("\nTesting u-conditional cdfs")
        for name, copulas in frozen_cops:
            print(name)
            zero = copulas.cdf_given_u(.5, 1e-9)
            one = copulas.cdf_given_u(.5, 1 - 1e-9)
            npt.assert_almost_equal(zero, 0)
            npt.assert_almost_equal(one, 1)

    def test_cdf_given_v(self):
        print("\nTesting v-conditional cdfs")
        for name, copulas in frozen_cops:
            print(name)
            zero = copulas.cdf_given_v(1e-9, .5)
            one = copulas.cdf_given_v(1 - 1e-9, .5)
            try:
                npt.assert_almost_equal(zero, 0)
                npt.assert_almost_equal(one, 1)
            except AssertionError:
                warnings.warn(name)

    def test_density(self):
        """Does the density integrate to 1?"""
        print("\nTesting density")
        for name, frozen_cop in frozen_cops:
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
        print("\nTesting fit")
        np.random.seed(1)
        for name, copulas in cop.all_cops.items():
            if name == "joe":
                continue
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
