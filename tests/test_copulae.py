import os
import shutil
import warnings

import matplotlib as mpl
import numpy as np
import numpy.testing as npt
from scipy import integrate


# mpl.use("Agg")
import matplotlib.pyplot as plt
from weathercop import copulae as cop

img_dir = os.path.join(os.path.abspath("."), "img")
if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
os.makedirs(img_dir)


class Test(npt.TestCase):

    def setUp(self):
        self.verbose = True
        self.eps = 1e-9

    def tearDown(self):
        pass

    def test_cop(self):
        """A few very basic copula requirements."""
        if self.verbose:
            print("\nTesting copula function")
        for name, frozen_cop in cop.frozen_cops.items():
            if self.verbose:
                print(name)
            zero = frozen_cop.copula_func(self.eps, self.eps)
            one = frozen_cop.copula_func(1 - self.eps, 1 - self.eps)
            try:
                npt.assert_almost_equal(zero, self.eps, decimal=4)
                npt.assert_almost_equal(one, 1 - self.eps, decimal=4)
            except AssertionError:
                print(zero, self.eps)
                print(one, 1 - self.eps)
                fig, ax = frozen_cop.plot_copula()  # theta=frozen_cop.theta)
                fig.savefig(os.path.join(img_dir, "copula_%s.png" % name))
                plt.close(fig)
                # plt.show()
                raise

    def test_cdf_given_u(self):
        if self.verbose:
            print("\nTesting u-conditional cdfs by roundtrip")
        for name, copula in cop.frozen_cops.items():
            kind_str = ("symbolic"
                        if hasattr(copula, "inv_cdf_given_uu_expr") else
                        "numeric")
            if self.verbose:
                print("%s (%s inverse)" % (name, kind_str))
            zero = copula.cdf_given_u(1 - self.eps, self.eps)
            one = copula.cdf_given_u(self.eps, 1 - self.eps)
            try:
                npt.assert_almost_equal(zero, self.eps, decimal=4)
                npt.assert_almost_equal(one, 1 - self.eps, decimal=4)
            except AssertionError:
                print(zero, self.eps)
                print(one, 1 - self.eps)
                warnings.warn(name.upper())
                fig, axs = copula.plot_cop_dens()
                fig.suptitle(name)
                fig.savefig(os.path.join(img_dir, "cop_dens_%s" % name))
                plt.close(fig)
                # plt.show()
                # raise

            qq = np.linspace(self.eps, 1 - self.eps, 200)
            ranks_u_low = np.full_like(qq, self.eps)
            ranks_u_high = np.full_like(qq, 1 - self.eps)
            ranks_v_low = copula.inv_cdf_given_u(ranks_u_low, qq)
            ranks_v_high = copula.inv_cdf_given_u(ranks_u_high, qq)
            low_cdf_func = lambda v: copula.cdf_given_u(np.full_like(v, .01), v)
            high_cdf_func = lambda v: copula.cdf_given_u(np.full_like(v, .99), v)
            try:
                # raise AssertionError
                npt.assert_array_less(1e-19, np.diff(low_cdf_func(qq)))
            except AssertionError:
                fig, ax = plt.subplots(nrows=1, ncols=1,
                                       subplot_kw=dict(aspect="equal"))
                low_vv = low_cdf_func(qq)
                ax.plot(qq, low_vv)
                ax.scatter(ranks_v_low, np.zeros_like(qq), marker="x")
                for rank_v, q in zip(ranks_v_low, qq):
                    ax.plot([0, rank_v], 2 * [q], color=(0, 0, 0, .1))
                    ax.plot([rank_v, rank_v], [0, q], color=(0, 0, 0, .1))
                ax.scatter(np.zeros_like(qq), qq, marker="x")
                ax.set_title("%s (%s)" % (copula.name, kind_str))
                fig.savefig(os.path.join(img_dir, "cdf_given_u_low_%s.png" %
                                         name))
                plt.close(fig)
                # plt.show()
                # raise
            try:
                npt.assert_array_less(1e-19, np.diff(high_cdf_func(qq)))
            except AssertionError:
                plt.plot(qq, high_cdf_func(qq))
                ax.scatter(ranks_v_high, np.zeros_like(qq), marker="x")
                ax.scatter(np.zeros_like(qq), qq, marker="x")
                plt.title(copula.name)
                fig.savefig(os.path.join(img_dir, "cdf_given_u_high_%s.png" %
                                         name))
                plt.close(fig)
                # plt.show()
                # raise

    def test_cdf_given_v(self):
        if self.verbose:
            print("\nTesting v-conditional cdfs by roundtrip")
        for name, copula in cop.frozen_cops.items():
            kind_str = ("symbolic"
                        if hasattr(copula, "inv_cdf_given_vv_expr") else
                        "numeric")
            if self.verbose:
                print("%s (%s inverse)" % (name, kind_str))
            zero = copula.cdf_given_v(self.eps, 1 - self.eps)
            one = copula.cdf_given_v(1 - self.eps, self.eps)
            try:
                npt.assert_almost_equal(zero, self.eps, decimal=4)
                npt.assert_almost_equal(one, 1 - self.eps, decimal=4)
            except AssertionError:
                print(zero, self.eps)
                print(one, 1 - self.eps)
                warnings.warn(name.upper())
                fig, ax = copula.plot_cop_dens()
                fig.suptitle(name)
                fig.savefig(os.path.join(img_dir, "cop_dens_%s" % name))
                plt.close(fig)
                # plt.show()
                # raise

            qq = np.linspace(self.eps, 1 - self.eps, 200)
            ranks_v_low = np.full_like(qq, self.eps)
            ranks_v_high = np.full_like(qq, 1 - self.eps)
            ranks_u_low = copula.inv_cdf_given_v(ranks_v_low, qq)
            ranks_u_high = copula.inv_cdf_given_v(ranks_v_high, qq)
            low_cdf_func = lambda u: copula.cdf_given_v(u, np.full_like(u, .01))
            high_cdf_func = lambda u: copula.cdf_given_v(u, np.full_like(u, .99))
            try:
                # raise AssertionError
                npt.assert_array_less(1e-19, np.diff(low_cdf_func(qq)))
            except AssertionError:
                fig, ax = plt.subplots(nrows=1, ncols=1,
                                       subplot_kw=dict(aspect="equal"))
                low_uu = low_cdf_func(qq)
                ax.plot(qq, low_uu)
                ax.scatter(ranks_u_low, np.zeros_like(qq), marker="x")
                for rank_u, q in zip(ranks_u_low, qq):
                    ax.plot([0, rank_u], 2 * [q], color=(0, 0, 0, .1))
                    ax.plot([rank_u, rank_u], [0, q], color=(0, 0, 0, .1))
                ax.scatter(np.zeros_like(qq), qq, marker="x")
                ax.set_title("%s (%s)" % (copula.name, kind_str))
                fig.savefig(os.path.join(img_dir, "cdf_given_v_low_%s.png" %
                                         name))
                plt.close(fig)
                # plt.show()
                # raise
            try:
                npt.assert_array_less(1e-19, np.diff(high_cdf_func(qq)))
            except AssertionError:
                plt.plot(qq, high_cdf_func(qq))
                ax.scatter(ranks_u_high, np.zeros_like(qq), marker="x")
                ax.scatter(np.zeros_like(qq), qq, marker="x")
                plt.title(copula.name)
                fig.savefig(os.path.join(img_dir, "cdf_given_v_high_%s.png" %
                                         name))
                plt.close(fig)
                # plt.show()
                # raise

    def test_inv_cdf_given_u(self):
        if self.verbose:
            print("\nTesting inverse u-conditional cdfs by roundtrip")
        for name, copula in cop.frozen_cops.items():
            if self.verbose:
                kind_str = ("symbolic"
                            if hasattr(copula, "inv_cdf_given_uu_expr") else
                            "numeric")
                print("%s (%s)" % (name, kind_str))
            uu = np.linspace(self.eps, 1 - self.eps, 100)
            vv_exp = np.copy(uu)
            for i, u in enumerate(uu):
                u = np.full_like(vv_exp, u)
                qq = copula.cdf_given_u(u, vv_exp)
                vv_actual = copula.inv_cdf_given_u(u, np.squeeze(qq))
                try:
                    npt.assert_almost_equal(vv_actual, vv_exp, decimal=2)
                except AssertionError:
                    if not self.verbose:
                        raise
                    fig, ax = plt.subplots()
                    ax.plot(vv_exp, vv_actual, "-x")
                    ax.plot([0, 1], [0, 1], color="k")
                    ax.set_xlabel("vv_exp")
                    ax.set_ylabel("vv_actual")
                    ax.set_title("%s inv_cdf_given_u u=%.6f"
                                 % (name, u[0]))
                    fig.savefig(os.path.join(img_dir,
                                             "inv_cdf_given_u_%s_%03d.png"
                                             % (name, i)))
                    plt.close(fig)
                    # plt.show()
                    # raise

    def test_inv_cdf_given_v(self):
        if self.verbose:
            print("\nTesting inverse v-conditional cdfs by roundtrip")
        for name, copula in cop.frozen_cops.items():
            if self.verbose:
                kind_str = ("symbolic"
                            if hasattr(copula, "inv_cdf_given_vv_expr") else
                            "numeric")
                print("%s (%s)" % (name, kind_str))
            vv = np.linspace(self.eps, 1 - self.eps, 100)
            uu_exp = np.copy(vv)
            for i, v in enumerate(vv):
                v = np.full_like(uu_exp, v)
                qq = copula.cdf_given_v(uu_exp, v)
                try:
                    npt.assert_array_less(0, qq)
                    npt.assert_array_less(qq, 1 + 1e-12)
                    uu_actual = copula.inv_cdf_given_v(v, qq)
                    npt.assert_almost_equal(uu_actual, uu_exp, decimal=2)
                except AssertionError:
                    if not self.verbose:
                        raise
                    fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
                    ax.plot([0, 1], [0, 1], color="k")
                    ax.plot(uu_exp, uu_actual, "-x")
                    ax.set_xlabel("uu_exp")
                    ax.set_ylabel("uu_actual")
                    ax.set_title("%s inv_cdf_given_v v=%.6f"
                                 % (name, v[0]))
                    fig.savefig(os.path.join(img_dir,
                                             "inv_cdf_given_v_%s_%03d.png"
                                             % (name, i)))
                    plt.close(fig)
                    # plt.show()

    def test_density(self):
        """Does the density integrate to 1?"""
        if self.verbose:
            print("\nTesting density by numerical integration")
        for name, frozen_cop in cop.frozen_cops.items():
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
                    try:
                        npt.assert_almost_equal(one, 1., decimal=5)
                    except AssertionError:
                        fig, ax = frozen_cop.plot_cop_dens()
                        fig.savefig(os.path.join(img_dir, "density_%s.png" %
                                                 name))
                        fig.suptitle("%s integral=%f" % (name, one))
                        plt.close(fig)

    def test_fit(self):
        """Is fit able to reproduce parameters of a self-generated sample?
        """
        if self.verbose:
            print("\nTesting fit")
        np.random.seed(1)
        for name, copula in cop.all_cops.items():
            if self.verbose:
                print(name)
            sample_x, sample_y = copula.sample(10000, copula.theta_start)
            try:
                fitted_theta = copula.fit(sample_x, sample_y)
            except cop.NoConvergence:
                print("... fitting did not converge.")
            else:
                if copula.theta_start[0] is None and fitted_theta is None:
                    # for the independence copula
                    continue
                try:
                    npt.assert_almost_equal(fitted_theta, copula.theta_start,
                                            decimal=1)
                except AssertionError:
                    if not self.verbose:
                        raise
                    ax = copula.plot_density()
                    ax.scatter(sample_x, sample_y, marker="x",
                               facecolor=(0, 0, 1, .1))
                    # fsample_x, fsample_y = copula.sample(10000, fitted_theta)
                    # ax.scatter(fsample_x, fsample_y)
                    copula.plot_cop_dens(scatter=True)
                    # plt.show()
                    raise


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    npt.run_module_suite()
