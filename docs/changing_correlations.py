from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, linalg, optimize, integrate
from weathercop import stats as wc_stats

zero, one = 1e-12, 1 - 1e-12


def asymmetry(uu, vv):
    uu = uu - 0.5
    vv = vv - 0.5
    return np.sum(uu**2 * vv + uu * vv**2)


def nonexceedance(xx):
    K, N = xx.shape
    nep = np.empty(N, dtype=float)
    for i, x_i in enumerate(xx.T):
        nep[i] = np.sum(np.prod(xx < x_i[:, None], axis=0)) / N
    return nep


class Independence:
    def simulate(
        self, sigma0, sigma1, n_sim, U=None, antithetic=True, return_U=False
    ):
        K = sigma0.shape[0]
        if U is None:
            if antithetic:
                U = np.random.uniform(
                    zero, one, size=(K, n_sim // 2 + n_sim % 2)
                )
                U = np.concatenate((U, -U), axis=1)[:, :n_sim]
            else:
                U = np.random.uniform(zero, one, size=(K, n_sim))
        taus = np.prod(U, axis=0)[:, None, None]
        taus = wc_stats.rel_ranks(taus)[:, None, None]
        # taus = stats.norm.ppf(taus)
        sigma0_sqrt = linalg.sqrtm(sigma0)[None, ...]
        sigma1_sqrt = linalg.sqrtm(sigma1)[None, ...]
        sigmas = taus * sigma1_sqrt + (1 - taus) * sigma0_sqrt
        Z = np.array([sigma.dot(u) for sigma, u in zip(sigmas, U.T)])
        stds = (sigmas @ sigmas)[:, np.arange(K), np.arange(K)]
        for k in range(K):
            for t in range(n_sim):
                Z[t, k] = stats.norm.cdf(Z[t, k], scale=stds[t, k])
        Z = Z.T
        # Z = stats.norm.cdf(Z.T)
        if return_U:
            return Z, U
        return Z


class Diag:
    # def simulate(self, scale0, scale1, n_sim, U=None, return_U=False):
    #     K = len(scale0)
    #     if U is None:
    #         U = np.random.uniform(-one, one, size=(K, n_sim))
    #     diag_z = np.random.uniform(zero, one, n_sim)
    #     # diag_z = np.random.triangular(zero, 0.5, one, size=n_sim)
    #     scale0 = scale0[:, None]
    #     scale1 = scale1[:, None]
    #     scales = diag_z[None, :] * scale1 + (1 - diag_z[None, :]) * scale0
    #     max_dist = np.where(diag_z < 0.5, diag_z, 1 - diag_z)
    #     # diag_z *= np.sqrt(K)
    #     # Z = np.zeros((K, n_sim)) + diag_z[None, :]
    #     # Z += scales * U * max_dist
    #     Z = diag_z[None, :] + scales * U * max_dist
    #     if return_U:
    #         return Z, U
    #     return Z

    def simulate(self, scale0, scale1, n_sim, U=None, return_U=False):
        K = len(scale0)
        if U is None:
            U = np.random.uniform(zero, one, size=(K, n_sim))
        # diag_z =

    def decorrelate(self, Z, scale0, scale1):
        pass


class TwoCorr:
    def __init__(self, n_tau=1000):
        self.n_tau = n_tau

    def simulate_approx(
        self,
        sigma0,
        sigma1,
        n_sim,
        seed=None,
        U=None,
        antithetic=True,
        return_U=False,
    ):
        K = sigma0.shape[0]
        sigma0_sqrt = linalg.sqrtm(sigma0)[None, ...]
        sigma1_sqrt = linalg.sqrtm(sigma1)[None, ...]
        taus = np.linspace(zero, one, self.n_tau)[:, None, None]
        sigmas = taus * sigma1_sqrt + (1 - taus) * sigma0_sqrt
        # sigmas = sigmas @ sigmas
        # sigmas[:] /= sigmas[:, 0, 0][:, None, None]
        # __import__("pdb").set_trace()
        # sigmas = np.array([linalg.sqrtm(sigma) for sigma in sigmas])
        if seed is not None:
            np.random.seed(seed)
        if U is None:
            if antithetic:
                U = np.random.randn(K, n_sim // 2 + n_sim % 2)
                U = np.concatenate((U, -U), axis=1)[:, :n_sim]
            else:
                U = np.random.randn(K, n_sim)
        sigma_U = np.array([sigma.dot(U) for sigma in sigmas])
        phi_taus = stats.norm.ppf(taus)
        ii = np.argmin(
            np.abs(phi_taus - sigma_U),
            axis=0,
        )
        Z = np.squeeze(taus[ii])
        if return_U:
            return Z, U
        return Z

    def decorrelate(self, Z, sigma0, sigma1):
        """Inverse of simulate_approx."""
        K, n_sim = Z.shape
        sigma0_sqrt = linalg.sqrtm(sigma0)[None, ...]
        sigma1_sqrt = linalg.sqrtm(sigma1)[None, ...]
        Z_broad = Z.T[:, :, None]
        sigmas = Z_broad * sigma1_sqrt + (1 - Z_broad) * sigma0_sqrt
        sigmas_I = np.linalg.inv(sigmas)
        Z_std = stats.norm.ppf(Z)
        return np.array([sigma @ Z_ for sigma, Z_ in zip(sigmas_I, Z_std.T)]).T

    def copula_approx(self, Z, sigma0, sigma1, return_U=False):
        U = self.decorrelate(Z, sigma0, sigma1)
        cop = np.prod(stats.norm.cdf(U), axis=0)
        if return_U:
            return cop, U
        return cop

    def density(self, Z, sigma0, sigma1):
        K, n_sim = Z.shape
        sigma0_sqrt = linalg.sqrtm(sigma0)[None, ...]
        sigma1_sqrt = linalg.sqrtm(sigma1)[None, ...]
        Z_broad = Z.T[:, :, None]
        sigmas = Z_broad * sigma1_sqrt + (1 - Z_broad) * sigma0_sqrt
        sigmas = sigmas @ sigmas
        # sigmas[:] /= sigmas[:, 0, 0][:, None, None]
        sigma_dets = np.array([linalg.det(sigma) for sigma in sigmas])
        singular_mask = np.isclose(sigma_dets, 0)
        sigmas_I = np.linalg.inv(sigmas)
        dens = (
            1
            / np.sqrt(sigma_dets)
            * np.exp(
                np.squeeze(
                    -0.5
                    * (Z.T[:, None, :] @ (sigmas_I - np.identity(K)))
                    @ Z.T[:, :, None]
                )
            )
        )
        return np.where(singular_mask, np.nan, dens)

    def fit(self, data, n_sim=10000, verbose=False):
        K = data.shape[0]
        data_norm = np.array([stats.norm.ppf(row) for row in data])
        pearson_corrs = np.corrcoef(data_norm)
        Sigma0 = np.ones((K, K))
        Sigma1 = np.ones((K, K))
        for i, j in zip(*np.triu_indices(K, 1)):
            asy = asymmetry(data[i], data[j])
            corr = pearson_corrs[i, j]
            if verbose:
                print(f"{i=}, {j=}, {corr=:.3f}, {asy=:.3f}")

            def obj(sol):
                sigma0_, sigma1_ = sol
                sigma0 = np.array([[1, sigma0_], [sigma0_, 1]])
                sigma1 = np.array([[1, sigma1_], [sigma1_, 1]])
                Z = self.simulate_approx(
                    sigma0, sigma1, n_sim, seed=0, antithetic=True
                )
                asy_sim = asymmetry(Z[0], Z[1])
                corr_sim = np.corrcoef(stats.norm.ppf(Z))[0, 1]
                if verbose:
                    print(
                        f"{sigma0_=:.3f}, {sigma1_=:.3f}, "
                        f"{corr_sim=:.3f}, {asy_sim=:.3f}"
                    )
                return (asy - asy_sim) ** 2 + (corr - corr_sim) ** 2

            def likelihood(sol):
                sigma0_, sigma1_ = sol
                sigma0 = np.array([[1, sigma0_], [sigma0_, 1]])
                sigma1 = np.array([[1, sigma1_], [sigma1_, 1]])
                density = self.density(
                    np.array([data[i], data[j]]), sigma0, sigma1
                )
                density[np.isnan(density)] = 1e-19
                # density = density[np.isfinite(density)]
                return -np.prod(density)

            result = optimize.minimize(
                obj,
                # likelihood,
                [corr, corr],
                # [0.5, 0.5],
                # [corr**2, 1 - corr**2][:: int(np.sign(asy))],
                bounds=[(-1, 1), (-1, 1)],
                method="Nelder-Mead",
            )
            Sigma0[i, j], Sigma1[i, j] = Sigma0[j, i], Sigma1[j, i] = result.x

        return Sigma0, Sigma1

    def plot_copula(self, sigma0, sigma1, n_grid=100, figsize=None, **f_kwds):
        K = sigma0.shape[0]
        ii, jj = np.triu_indices(K, 1)
        if figsize is None:
            figsize = (3 * K, 3)
        fig, axs = plt.subplots(
            nrows=2,
            ncols=len(ii),
            subplot_kw=dict(aspect="equal"),
            constrained_layout=True,
            **f_kwds,
        )
        axs = np.atleast_2d(axs.T)
        Z = self.simulate_approx(sigma0, sigma1, n_sim=1000)
        Z_1dim = np.linspace(zero, one, n_grid, endpoint=False)
        Z_grid = np.array(
            np.broadcast_arrays(Z_1dim[:, None], Z_1dim[None, :])
        ).reshape(2, -1)
        for ax, i, j in zip(axs, ii, jj):
            cop = self.copula_approx(
                Z_grid,
                np.array([[1, sigma0[i, j]], [sigma0[i, j], 1]]),
                np.array([[1, sigma1[i, j]], [sigma1[i, j], 1]]),
            )
            ax[0].scatter(
                Z[i],
                Z[j],
                color=(0, 0, 0, 0),
                edgecolors=(0, 0, 0, 0.1),
                zorder=99,
            )
            ax[0].contourf(
                Z_grid[0].reshape(n_grid, n_grid),
                Z_grid[1].reshape(n_grid, n_grid),
                cop.reshape((n_grid, n_grid)),
                30,
            )
            density = self.density(
                Z_grid,
                np.array([[1, sigma0[i, j]], [sigma0[i, j], 1]]),
                np.array([[1, sigma1[i, j]], [sigma1[i, j], 1]]),
            )
            ax[1].contourf(
                Z_grid[0].reshape(n_grid, n_grid),
                Z_grid[1].reshape(n_grid, n_grid),
                density.reshape((n_grid, n_grid)),
                30,
            )
        plt.show()
        return fig, axs


if __name__ == "__main__":
    np.random.seed(0)
    n_tau = 1000
    n_sim = 5000

    # sigma0 = np.array([[1, 0.2], [0.2, 1]])
    # sigma1 = np.array([[1, 0.9], [0.9, 1]])

    # sigma0 = np.array([[1, 0.6], [0.6, 1]])
    # sigma1 = np.array([[1, 0.6], [0.6, 1]])

    # sigma0 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    # sigma1 = np.array([[1, 0, 0], [0, 1, 1], [0, 1, 1]])

    scale0 = np.array([0.5, 0.5])
    scale1 = np.array([1, 1])

    diag = Diag()
    Z = diag.simulate(scale0, scale1, n_sim)
    ii, jj = np.triu_indices(scale0.shape[0], 1)
    pearson = np.corrcoef(Z)

    for i, j in zip(ii, jj):
        fig, axs = plt.subplots(
            2,
            2,
            width_ratios=(0.8, 0.2),
            height_ratios=(0.8, 0.2),
            sharex="col",
            sharey="row",
            figsize=(6, 6),
        )
        axs[0, 0].scatter(
            Z[i], Z[j], color=(0, 0, 0, 0), edgecolors=(0, 0, 0, 0.5)
        )
        axs[0, 0].grid(True)
        axs[0, 1].hist(
            Z[j], 30, orientation="horizontal", color="k", histtype="step"
        )
        axs[1, 0].hist(Z[i], 30, color="k", histtype="step")
        axs[1, 1].set_axis_off()
        fig.suptitle(rf"$rho_{{i, j}}$ = {pearson[i, j]:.3f}")

    fig, axs = plt.subplots(
        nrows=1, ncols=len(ii), subplot_kw=dict(aspect="equal")
    )
    axs = np.atleast_1d(axs)
    for ax, i, j in zip(axs, ii, jj):
        ax.scatter(Z[i], Z[j], color=(0, 0, 0, 0), edgecolors=(0, 0, 0, 0.5))
        ax.set_title(rf"$rho_{{i, j}}$ = {pearson[i, j]:.3f}")
        ax.grid(True)
    plt.show()

    # indep = Independence()
    # Z = indep.simulate(sigma0, sigma1, n_sim, antithetic=False)
    # ii, jj = np.triu_indices(sigma0.shape[0], 1)
    # pearson = np.corrcoef(Z)
    # fig, axs = plt.subplots(
    #     nrows=1, ncols=len(ii), subplot_kw=dict(aspect="equal")
    # )
    # axs = np.atleast_1d(axs)
    # for ax, i, j in zip(axs, ii, jj):
    #     ax.scatter(Z[i], Z[j], color=(0, 0, 0, 0), edgecolors=(0, 0, 0, 0.5))
    #     ax.set_title(rf"$rho_{{i, j}}$ = {pearson[i, j]:.3f}")
    #     ax.grid(True)
    # plt.show()

    # twocorr = TwoCorr(n_tau=n_tau)
    # fig, axs = twocorr.plot_copula(sigma0, sigma1)
    # fig.savefig("copula_approximation.png")

    # def density(*x):
    #     return twocorr.density(np.array(x)[:, None], sigma0, sigma1)

    # dens_integral = integrate.nquad(
    #     density,
    #     np.stack([[zero, one] for i in range(sigma0.shape[0])]),
    # )
    # print(f"{dens_integral=}")

    # Z, U0 = twocorr.simulate_approx(
    #     sigma0, sigma1, n_sim, antithetic=True, return_U=True
    # )
    # # zz1 = twocorr.simulate_approx(sigma0, sigma1, n_sim, U=U0)

    # cop, U = twocorr.copula_approx(Z, sigma0, sigma1, return_U=True)
    # Z_roundtrip = twocorr.simulate_approx(sigma0, sigma1, n_sim, U=U0)

    # import numpy.testing as npt

    # try:
    #     npt.assert_allclose(U, U0)
    # except AssertionError:
    #     fig, axs = plt.subplots(
    #         nrows=1, ncols=sigma0.shape[0], subplot_kw=dict(aspect="equal")
    #     )
    #     for row_i, ax in enumerate(axs):
    #         ax.scatter(
    #             U0[row_i],
    #             U[row_i],
    #             color=(0, 0, 0, 0),
    #             edgecolors=(0, 0, 0, 0.5),
    #         )
    #         ax.axline((0, 0), slope=1)
    #         ax.grid(True)
    #         ax.set_xlabel("original")
    #         ax.set_ylabel("recovered")
    #     axs[0].legend(loc="best")
    #     fig.suptitle("Random numbers")

    # try:
    #     npt.assert_allclose(Z, Z_roundtrip)
    # except AssertionError:
    #     fig, axs = plt.subplots(
    #         nrows=1, ncols=sigma0.shape[0], subplot_kw=dict(aspect="equal")
    #     )
    #     for row_i, ax in enumerate(axs):
    #         ax.scatter(
    #             Z[row_i],
    #             Z_roundtrip[row_i],
    #             color=(0, 0, 0, 0),
    #             edgecolors=(0, 0, 0, 0.5),
    #         )
    #         ax.axline((0, 0), slope=1)
    #         ax.grid(True)
    #     axs[0].legend(loc="best")

    # fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw=dict(aspect="equal"))
    # ax.scatter(
    #     cop,
    #     nonexceedance(Z),
    #     color=(0, 0, 0, 0),
    #     edgecolors=(0, 0, 0, 0.5),
    # )
    # ax.axline((0, 0), slope=1)
    # ax.set_xlabel("copula approximation")
    # ax.set_ylabel("empirical")
    # ax.grid(True)
    # fig.savefig("approximation_vs_empirical.png")
    # plt.show()

    # sigma0, sigma1 = twocorr.fit(Z, n_sim, verbose=True)
    # Z_fit = twocorr.simulate_approx(sigma0, sigma1, n_sim)
    # print(sigma0)
    # print(sigma1)

    # ii, jj = np.triu_indices(sigma0.shape[0], 1)
    # pearson = np.corrcoef(Z)
    # fig, axs = plt.subplots(
    #     nrows=1, ncols=len(ii), subplot_kw=dict(aspect="equal")
    # )
    # axs = np.atleast_1d(axs)
    # for ax, i, j in zip(axs, ii, jj):
    #     ax.scatter(
    #         Z_fit[i], Z_fit[j], color=(0, 0, 0, 0), edgecolors=(0, 0, 0, 0.5)
    #     )
    #     ax.set_title(rf"$rho_{{i, j}}$ = {pearson[i, j]:.3f}")
    #     ax.grid(True)
    # plt.show()
