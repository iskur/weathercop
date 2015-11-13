from scipy import integrate
import numpy.testing as npt
import copulae as cop

imported_copulas = {name: obj for name, obj
                    in cop.__dict__.items()
                    if isinstance(obj, cop.Copulae)}
frozen_cops = {name: copulas(*copulas.theta_start)
               for name, copulas
               in imported_copulas.items()}


class Test(npt.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_density(self):
        """Does the density integrate to 1?"""
        for name, frozen_cop in frozen_cops.items():
            print name
            one = integrate.nquad(frozen_cop.density,
                                  ([0, 1], [0, 1]))[0]
            npt.assert_almost_equal(one, 1.)
    
    def test_fit_ml(self):
        """Is fit_ml able to reproduce parameters of a self-generated
        sample?"""
        for name, copulas in imported_copulas.items():
            print name
            sample_x, sample_y = copulas.sample(10000)
            fitted_theta = copulas.fit_ml(sample_x, sample_y)
            npt.assert_almost_equal(fitted_theta, copulas.theta_start,
                                    decimal=1)
        

if __name__ == "__main__":
    npt.run_module_suite()
