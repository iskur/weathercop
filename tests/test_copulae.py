import shutil
import warnings
from pathlib import Path
from collections import OrderedDict

# import matplotlib as mpl
import numpy as np
import numpy.testing as npt
from scipy import integrate

# mpl.use("Agg")
import matplotlib.pyplot as plt
from weathercop import copulae as cop

img_dir = Path("./img").absolute()
if img_dir.exists():
    shutil.rmtree(img_dir)
img_dir.mkdir(parents=True, exist_ok=True)


class Test(npt.TestCase):

    def setUp(self):
        self.verbose = True
        self.eps = np.array([1e-5])
        test_cop = "all"
        # test_cop = "nelsen15"
        if test_cop != "all":
            cop.all_cops = OrderedDict((name, obj)
                                       for name, obj in cop.all_cops.items()
                                       if test_cop in name)
            cop.frozen_cops = OrderedDict((name, obj)
                                          for name, obj in cop.frozen_cops.items()
                                          if test_cop in name)
            print(f"Testing only {(name for name in cop.all_cops.keys())}")

    def tearDown(self):
        pass

    # def test_erf(self):
    #     from scipy.special import erf as sp_erf
    #     xx = np.linspace(-2, 2, 100)
    #     sp_erf_vals = sp_erf(xx)
    #     erf_vals = cop.erf(xx)
    #     npt.assert_almost_equal(erf_vals, sp_erf_vals, decimal=4)

    # def test_erf_inv(self):
    #     from scipy.special import erfinv as sp_erfinv
    #     xx = np.linspace(-1, 1, 100)
    #     sp_erfinv_vals = sp_erfinv(xx)
    #     erfinv_vals = cop.erfinv(xx)
    #     npt.assert_almost_equal(erfinv_vals, sp_erfinv_vals,
    #                             decimal=3)

    def test_bounds(self):
        """Can we evaluate densities on theta bounds?"""
        if self.verbose:
            print("\nTesting theta bounds")
        # uu = np.linspace(self.eps, 1 - self.eps, 100).repeat(100)
        uu = (np.linspace(self.eps, 1 - self.eps, 100)
              .repeat(100)
              .reshape(100, 100))
        vv = np.tile(np.linspace(self.eps, 1 - self.eps, 100), 100)
        for name, copula in cop.all_cops.items():
            if self.verbose:
                print(name)
            for theta in np.array(copula.theta_bounds).T:
                frozen_cop = copula(theta)
                dens = frozen_cop.density(uu, vv)
                assert np.all(np.isfinite(dens))

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
                fig.savefig(img_dir / f"copula_{name}.png")
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
                print(f"{name} ({kind_str} inverse)")
            # zero = copula.cdf_given_u(1 - self.eps, self.eps)
            # one = copula.cdf_given_u(self.eps, 1 - self.eps)
            eps = np.array([1e-9])
            zero = copula.cdf_given_u(1 - self.eps, eps)
            while zero > self.eps and eps > 1e-17:
                eps *= .1
                zero = copula.cdf_given_u(1 - self.eps, eps)
                print(eps, zero)

            eps = np.array([1e-9])
            one = copula.cdf_given_u(self.eps, 1 - eps)
            while one < 1 - self.eps and eps > 1e-17:
                eps *= .1
                one = copula.cdf_given_u(self.eps, 1 - eps)
                print(eps, one)

            try:
                npt.assert_almost_equal(zero, self.eps, decimal=4)
                npt.assert_almost_equal(one, 1 - self.eps, decimal=4)
            except AssertionError:
                print(zero, self.eps)
                print(one, 1 - self.eps)
                warnings.warn(name.upper())
                fig, axs = copula.plot_cop_dens()
                fig.suptitle(name)
                fig.savefig(img_dir / f"cop_dens_{name}")
                plt.close(fig)
                # plt.show()
                raise

            eps = self.eps[0]
            qq = np.linspace(eps, 1 - eps, 200)
            ranks_u_low = np.full_like(qq, eps)
            ranks_u_high = np.full_like(qq, 1 - eps)
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
                ax.set_title(f"{copula.name} ({kind_str})")
                fig.savefig(img_dir / f"cdf_given_u_low_{name}.png")
                plt.close(fig)
                # plt.show()
                raise
            try:
                npt.assert_array_less(1e-19, np.diff(high_cdf_func(qq)))
            except AssertionError:
                plt.plot(qq, high_cdf_func(qq))
                ax.scatter(ranks_v_high, np.zeros_like(qq), marker="x")
                ax.scatter(np.zeros_like(qq), qq, marker="x")
                plt.title(copula.name)
                fig.savefig(img_dir / f"cdf_given_u_high_{name}.png")
                plt.close(fig)
                # plt.show()
                raise

    def test_cdf_given_v(self):
        if self.verbose:
            print("\nTesting v-conditional cdfs by roundtrip")
        for name, copula in cop.frozen_cops.items():
            kind_str = ("symbolic"
                        if hasattr(copula, "inv_cdf_given_vv_expr") else
                        "numeric")
            if self.verbose:
                print(f"{name} ({kind_str} inverse)")
            eps = np.array([1e-9])
            zero = copula.cdf_given_v(eps, 1 - self.eps)
            while zero > self.eps and eps > 1e-17:
                eps *= .1
                zero = copula.cdf_given_v(eps, 1 - self.eps)
                print(eps, zero)

            eps = np.array([1e-9])
            one = copula.cdf_given_v(1 - eps, self.eps)
            while one < 1 - self.eps and eps > 1e-17:
                eps *= .1
                one = copula.cdf_given_v(1 - eps, self.eps)
                print(eps, one)

            try:
                # npt.assert_almost_equal(zero, self.eps, decimal=4)
                # npt.assert_almost_equal(one, 1 - self.eps, decimal=4)
                npt.assert_array_less(zero, self.eps)
                npt.assert_array_less(1 - self.eps, one)
            except AssertionError:
                print(zero, self.eps)
                print(one, 1 - self.eps)
                warnings.warn(name.upper())
                fig, ax = copula.plot_cop_dens()
                fig.suptitle(name)
                fig.savefig(img_dir / f"cop_dens_{name}")
                plt.close(fig)
                plt.show()
                raise

            eps = self.eps[0]
            qq = np.linspace(eps, 1 - eps, 200)
            ranks_v_low = np.full_like(qq, eps)
            ranks_v_high = np.full_like(qq, 1 - eps)
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
                ax.set_title(f"{copula.name} ({kind_str})")
                fig.savefig(img_dir / f"cdf_given_v_low_{name}.png")
                plt.close(fig)
                # plt.show()
                raise
            try:
                npt.assert_array_less(1e-19, np.diff(high_cdf_func(qq)))
            except AssertionError:
                plt.plot(qq, high_cdf_func(qq))
                ax.scatter(ranks_u_high, np.zeros_like(qq), marker="x")
                ax.scatter(np.zeros_like(qq), qq, marker="x")
                plt.title(copula.name)
                fig.savefig(img_dir / f"cdf_given_v_high_{name}.png")
                plt.close(fig)
                # plt.show()
                # print("test failed!!!")
                raise

    def test_inv_cdf_given_u(self):
        if self.verbose:
            print("\nTesting inverse u-conditional cdfs by roundtrip")
        for name, copula in cop.frozen_cops.items():
            if self.verbose:
                kind_str = ("symbolic"
                            if hasattr(copula, "inv_cdf_given_uu_expr") else
                            "numeric")
                print(f"{name} ({kind_str})")
            uu = np.squeeze(np.linspace(self.eps, 1 - self.eps, 100))
            vv_exp = np.copy(uu)
            for i, u in enumerate(uu):
                u = np.full_like(vv_exp, u)
                qq = copula.cdf_given_u(u, vv_exp)
                # print(i, qq.max())
                # qq[qq > 1 - 1e-7] = 1 - 1e-7
                # qq = np.full_like(u, qq)
                vv_actual = copula.inv_cdf_given_u(u, qq)
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
                    ax.set_title(f"{name} inv_cdf_given_u "
                                 f"u={np.squeeze(u)[0]:.6f}")
                    fig.savefig(img_dir /
                                f"inv_cdf_given_u_{name}_{i:03d}.png")
                    plt.close(fig)
                    print("test failed!!!")
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
                print(f"{name} ({kind_str})")
            vv = np.squeeze(np.linspace(self.eps, 1 - self.eps, 100))
            uu_exp = np.copy(vv)
            for i, v in enumerate(vv):
                v = np.full_like(uu_exp, v)
                qq = copula.cdf_given_v(uu_exp, v)
                try:
                    npt.assert_array_less(0, qq + 1e-12)
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
                    ax.set_title(f"{name} inv_cdf_given_v "
                                 f"v={np.squeeze(v)[0]:.6f}")
                    fig.savefig(img_dir /
                                f"inv_cdf_given_v_{name}_{i:03d}.png")
                    plt.close(fig)
                    # plt.show()
                    # raise

    def test_density(self):
        """Does the density integrate to 1?"""
        if self.verbose:
            print("\nTesting density by numerical integration")
        for name, frozen_cop in cop.frozen_cops.items():
            if self.verbose:
                print(name)
            # with warnings.catch_warnings():
            #     warnings.filterwarnings("error")
            try:
                # if isinstance(frozen_cop.copula, cop.Gaussian):
                #     bins_per_dim = 50
                #     bins = np.linspace(self.eps, 1 - self.eps,
                #                        bins_per_dim)
                #     uu = bins.repeat(bins_per_dim)
                #     vv = np.tile(bins, bins_per_dim)
                #     # bins = np.concatenate(([0], bins, [1]))
                #     densities = frozen_cop.density(uu, vv)
                #     one = np.sum(densities) / (len(uu) * len(vv))
                #     import ipdb; ipdb.set_trace()
                # else:
                def density(x, y):
                    return frozen_cop.density(np.array([x]),
                                              np.array([y]))
                one = integrate.nquad(density,
                                      ([self.eps, 1 - self.eps],
                                       [self.eps, 1 - self.eps]))[0]
            except integrate.IntegrationWarning:
                print(f"Numerical integration of {name} is problematic")
            else:
                try:
                    npt.assert_almost_equal(one, 1., decimal=4)
                except AssertionError:
                    fig, ax = frozen_cop.plot_cop_dens()
                    fig.savefig(img_dir / f"density_{name}.png")
                    fig.suptitle(f"{name} integral={one}")
                    plt.close(fig)
                    raise

    def test_fit(self):
        """Is fit able to reproduce parameters of a self-generated sample?
        """
        if self.verbose:
            print("\nTesting fit")
        np.random.seed(1)
        for name, copula in cop.all_cops.items():
            if self.verbose:
                print(name)
            # if isinstance(copula, cop.Clayton):
            #     import ipdb; ipdb.set_trace()
            sample_x, sample_y = copula.sample(10000, *copula.theta_start)
            try:
                fitted_theta = copula.fit(sample_x, sample_y)
            except cop.NoConvergence:
                print("... fitting did not converge.")
            else:
                if isinstance(copula, cop.Independence):
                    continue
                try:
                    npt.assert_almost_equal(fitted_theta, copula.theta_start,
                                            decimal=1)
                except AssertionError:
                    if not self.verbose:
                        raise
                    fig, ax = copula.plot_density()
                    ax.scatter(sample_x, sample_y, marker="x",
                               facecolor=(0, 0, 1, .1))
                    # fsample_x, fsample_y = copula.sample(10000, fitted_theta)
                    # ax.scatter(fsample_x, fsample_y)
                    fig, ax = copula.plot_cop_dens(scatter=True)
                    fig.savefig(img_dir / f"cop_dens_{name}.png")
                    raise

    # def test_likelihood(self):
    #     """Do we have roughly the same likelihood for self-generated samples?
    #     """
    #     name_likelihoods = {}
    #     for name, copula in cop.all_cops.items():
    #         if self.verbose:
    #             print(name)
    #         sample_x, sample_y = copula.sample(1000, copula.theta_start)
    #         fitted = cop.Fitted(copula, sample_x, sample_y,
    #                             copula.theta_start)
    #         name_likelihoods[name] = fitted.likelihood
    #     names = [name for name in name_likelihoods.keys()
    #              if name != "independence"]
    #     likelihoods = [name_likelihoods[name] for name in names]
    #     fig, ax = plt.subplots()
    #     ax.bar(1 + np.arange(len(names)), likelihoods, tick_label=names)
    #     ax.tick_params(axis='x', labelrotation=30)
    #     plt.show()


if __name__ == "__main__":
    # warnings.filterwarnings("error")
    npt.run_module_suite()
