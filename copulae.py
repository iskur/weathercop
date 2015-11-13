"""Bivariate Copulas intendet for Vine Copulas."""
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy import stats, integrate, interpolate


class Copulae(object):

    """Base of all copula implementations, defining what a copula
    must implement to be a copula."""
    __metaclass__ = ABCMeta

    theta_bounds = None

    @abstractmethod
    def density(self):
        """Copula density"""
        pass

    @abstractproperty
    def theta_start(self):
        """Starting solution for parameter estimation."""
        pass

    @abstractproperty
    def name(self):
        pass

    def __call__(self, *theta):
        return Frozen(self, *theta)

    def _ppf_given_u(self, uu, theta=None, sample_size=1000):
        if theta is None:
            theta = self.theta_start
        vv = np.linspace(0., 1, sample_size)
        density_along_uu = self.density(uu, vv, *theta)
        cdf_along_uu = integrate.cumtrapz(density_along_uu, vv,
                                          initial=0.)
        cdf_along_uu[-1] = 1.
        ppf_interpol = interpolate.interp1d(cdf_along_uu, vv)
        return ppf_interpol

    def sample(self, size, theta=None):
        uu = np.random.rand(size)
        vv = np.random.rand(size)
        vv = np.array([self._ppf_given_u(u_, theta=theta)(v_)
                       for u_, v_ in zip(uu, vv)])
        return 1 - uu, 1 - vv

    def fit_ml(self, ranks_u, ranks_v, method=None, verbose=False):
        """Maximum likelihood estimate."""
        def neg_log_likelihood(theta):
            dens = self.density(ranks_u, ranks_v, theta)
            return -np.sum(np.log(dens))

        result = minimize(neg_log_likelihood, self.theta_start,
                          bounds=self.theta_bounds,
                          method=method,
                          options=(dict(disp=True) if verbose else None))
        return result.x

    def plot_density(self, theta=None, scatter=True, ax=None, opacity=.1):
        if theta is None:
            theta = self.theta_start
        uu = vv = rel_ranks(np.arange(1000))
        density = self.density(uu[None, :], vv[:, None], *theta)
        # get rid of large values for visualizations sake
        density[density >= np.sort(density.ravel())[-100]] = np.nan
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        ax.contourf(uu, vv, density[::-1, ::-1])
        if scatter:
            u_sample, v_sample = self.sample(size=10000, theta=theta)
            ax.scatter(u_sample, v_sample,
                       marker="o",
                       facecolor=(0, 0, 0, 0),
                       edgecolor=(0, 0, 0, opacity))
        return ax


class Frozen(object):

    def __init__(self, copula, *theta):
        """Copula with frozen parameters."""
        self.copula = copula
        self.theta = theta

    def density(self, ranks_x, ranks_y):
        return self.copula.density(ranks_x, ranks_y, *self.theta)

    def sample(self, size):
        return self.copula.sample(size, self.theta)


class Clayton(Copulae):
    name = "Clayton"
    theta_start = 20,
    theta_bounds = [(1e-9, np.inf)]

    # def density(self, uu, vv, theta):
    #     uu = 1 - uu
    #     vv = 1 - vv
    #     return ((-1 + uu ** -theta + vv ** -theta) ** (-2 - 1 / theta) *
    #             (uu ** (-theta - 1) *
    #              (vv ** (-theta - 1) * (theta + 1))))

    def density(self, uu, vv, theta):
        return ((theta + 1) * (uu * vv) ** (-(theta + 1)) *
                (uu ** -theta + vv ** -theta - 1) **
                (-(2 * theta + 1) / theta))

    # def density(self, uu, vv, theta):
    #     dens = np.array((theta + 1) * uu ** (theta - 1) * vv ** (theta - 1) *
    #                     (uu ** -theta + vv ** -theta - 1) ** (-1 / theta) /
    #                     (uu ** theta * vv ** theta -
    #                      uu ** theta - vv ** theta) ** 2)
    #     dens[dens < 0] = 0
    #     return np.squeeze(dens)

clayton = Clayton()


class Frank(Copulae):
    name = "Frank"
    theta_start = 5,
    theta_bounds = [(1e-9, np.inf)]

    def density(self, uu, vv, theta):
        eta = 1 - np.exp(-theta)
        return ((theta * eta * np.exp(-theta * (uu + vv))) /
                (eta -
                 (1 - np.exp(-theta * uu)) *
                 (1 - np.exp(-theta * vv))) ** 2)
frank = Frank()


class Joe(Copulae):
    name = "Joe"
    theta_start = 4,
    theta_bounds = [(1 + 1e-9, np.inf)]

    def density(self, uu, vv, theta):
        # derivative obtained by software (wolfram alpha), as
        # suggested by Joe (2004)
        return (-(1 - uu) ** (theta - 1) *
                (1 - vv) ** (theta - 1) *
                ((1 - vv) ** theta -
                 (1 - uu) ** theta *
                 ((1 - vv) ** theta - 1)) ** (1 / theta - 2) *
                ((1 - uu) ** theta * ((1 - vv) ** theta - 1) -
                 (1 - vv) ** theta - theta + 1))
joe = Joe()


class Joe180(Copulae):
    name = "Joe180"
    theta_start = 2,
    theta_bounds = [(1 + 1e-9, np.inf)]

    def density(self, uu, vv, theta):
        return joe.density(1 - uu, 1 - vv, theta)
joe180 = Joe180()


def rel_ranks(data, method="average"):
    return (stats.rankdata(data, method) - .5) / len(data)


# class StudentT(Copulae):
#     """"""


# class Tawn(Copulae):
#     """"""

if __name__ == '__main__':
    from lhglib.contrib import dirks_globals as my

    with np.load("vg_data.npz") as saved:
        data_summer = saved["summer"]
    ranks_u_tm1 = rel_ranks(data_summer[5, :-1])
    ranks_rh = rel_ranks(data_summer[4, 1:])

    for copula in (clayton, frank, joe, joe180):
        theta = copula.fit_ml(ranks_u_tm1, ranks_rh,
                              method="TNC",
                              verbose=True)

        fig, axs = plt.subplots(ncols=2, subplot_kw=dict(aspect="equal"))
        fig.suptitle(copula.name)
        opacity = .1
        copula.plot_density(theta=theta, ax=axs[0], opacity=opacity,
                            scatter=True)
        my.hist2d(ranks_u_tm1, ranks_rh, ax=axs[1],
                  # kind="contourf",
                  scatter=False)
        axs[1].scatter(ranks_u_tm1, ranks_rh,
                       marker="o", facecolors=(0, 0, 0, 0),
                       edgecolors=(0, 0, 0, opacity))
    plt.show()
