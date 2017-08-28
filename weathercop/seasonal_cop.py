import functools
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pathos
from tqdm import tqdm

from lhglib.contrib import dirks_globals as my, times
from lhglib.contrib.time_series_analysis import (distributions as dists,
                                                 spectral)
from weathercop import copulae as cops, stats

n_nodes = pathos.multiprocessing.cpu_count() - 2


@functools.total_ordering
class SeasonalCop:

    def __init__(self, copula, dtimes, ranks_u, ranks_v, *,
                 window_len=15, fft_order=3, verbose=False):
        """Seasonally adapting copula.

        Parameter
        ---------
        copula : weathercop.Copula instance or None
            If None, the copula from weathercop.copulae with the
            highest likelihood is chosen.
        dtimes : (T,) array of datetime objects
        ranks_u : (T,) float array
        ranks_v : (T,) float array
        window_len : int, optional
        fft_order : int, optional
        verbose : boolean, optional
        """
        if copula is None:
            my_cops = [cop for cop in cops.all_cops.values()
                       if not isinstance(cop, cops.Independence)]

            def expand_ar(ar):
                if np.ndim(ar) == 2:
                    return ar
                n_cops = len(my_cops)
                return np.tile(ar, (n_cops, 1))

            dtimes, ranks_u, ranks_v = map(expand_ar,
                                           (dtimes, ranks_u, ranks_v))
            # pool = pathos.pools.ProcessPool(nodes=n_nodes)
            pool = pathos.pools.ThreadPool()
            scops = pool.imap(SeasonalCop, my_cops, dtimes, ranks_u,
                              ranks_v, window_len=window_len,
                              verbose=verbose)

            # scops = [SeasonalCop(cop,
            #                      dtimes,
            #                      ranks_u,
            #                      ranks_v,
            #                      window_len=window_len,
            #                      verbose=verbose)
            #          for cop in tqdm(my_cops)
            #          if not isinstance(cop, cops.Independence)]

            # become the copula with the best likelihood
            self.__dict__ = max(scops).__dict__
        else:
            self.pool = pathos.pools.ProcessPool(nodes=n_nodes)
            self.copula = copula
            self.dtimes = dtimes
            self.ranks_u = ranks_u
            self.ranks_v = ranks_v
            self.window_len = window_len
            self.fft_order = fft_order
            self.verbose = verbose

            self.name = "seasonal " + copula.name
            self.T = len(ranks_u)
            self.len_theta = len(self.copula.theta_start)
            self.doys = times.datetime2doy(dtimes)
            timestep = ((self.dtimes[1] -
                         self.dtimes[0]).total_seconds() //
                        (60 ** 2 * 24))
            self.doys_unique = np.unique(my.round_to_float(self.doys,
                                                           timestep))
            self.n_doys = len(self.doys_unique)
            self._doy_mask = self._sliding_theta = self._solution = None

    def _call_copula_func(self, method_name, conditioned, condition, t=None):
        if t is None:
            theta = self.thetas
        else:
            theta = self.thetas[t % self.n_doys]
        method = getattr(self.copula, method_name)
        return method(conditioned, condition, np.squeeze(theta))

    def cdf_given_u(self, t=None, *, conditioned, condition):
        return self._call_copula_func("cdf_given_u", condition, conditioned, t)

    def cdf_given_v(self, t=None, *, conditioned, condition):
        return self._call_copula_func("cdf_given_v", condition, conditioned, t)

    def inv_cdf_given_u(self, t=None, *, conditioned, condition):
        return self._call_copula_func("inv_cdf_given_u", condition,
                                      conditioned, t)

    def inv_cdf_given_v(self, t=None, *, conditioned, condition):
        return self._call_copula_func("inv_cdf_given_v", condition,
                                      conditioned, t)
    
    @property
    def doy_mask(self):
        """Returns a (n_unique_doys, T) ndarray"""
        if self._doy_mask is None:
            window_len, doys = self.window_len, self.doys
            self._doy_mask = \
                np.empty((len(self.doys_unique), self.T), dtype=bool)
            for doy_i, doy in enumerate(self.doys_unique):
                ii = (doys > doy - window_len) & (doys <= doy + window_len)
                if (doy - window_len) < 0:
                    ii |= doys > (365. - window_len + doy)
                if (doy + window_len) > 365:
                    ii |= doys < (doy + window_len - 365.)
                self._doy_mask[doy_i] = ii
        return self._doy_mask

    @property
    def sliding_theta(self):
        if self._sliding_theta is None:
            self._sliding_theta = np.ones((self.n_doys, self.len_theta))
            for doy_ii in tqdm(range(self.n_doys), disable=(not self.verbose)):
                ranks_u = self.ranks_u[self.doy_mask[doy_ii]]
                ranks_v = self.ranks_v[self.doy_mask[doy_ii]]
                if doy_ii == 0:
                    try:
                        theta = self.copula.fit(ranks_u, ranks_v)
                    except cops.NoConvergence:
                        theta = self.copula.theta_start
                else:
                    x0 = self._sliding_theta[doy_ii - 1]
                    try:
                        theta = self.copula.fit(ranks_u, ranks_v, x0=x0)
                    except cops.NoConvergence:
                        try:
                            theta = self.copula.fit(ranks_u, ranks_v)
                        except cops.NoConvergence:
                            warnings.warn("No Convergence reached in %s."
                                          % self.copula.name)
                            theta = np.nan
                self._sliding_theta[doy_ii] = theta

        # try to interpolate over bad fittings
        thetas = self._sliding_theta
        for theta_i, theta in enumerate(thetas.T):
            if np.any(np.isnan(theta)):
                half = len(theta) // 2
                theta_pad = np.concatenate((theta[-half:],
                                            theta,
                                            theta[:half]))
                interp = my.interp_nan(theta_pad)[half:-half]
                self._sliding_theta[:, theta_i] = interp
        return self._sliding_theta.T

    @property
    def solution(self):
        if self._solution is None:
            trans = np.fft.rfft(self.sliding_theta)
            self._solution = np.array(trans)
            self._T = self.doys * 2 * np.pi / 365
            self.thetas = self.trig2thetas(self._solution, self._T)
        return self._solution

    @property
    def solution_mll(self):
        A = np.zeros(self.T, dtype=complex)
        fft_order = self.fft_order

        def mml(A_parts):
            real, imag = A_parts.reshape(2, -1)
            A[:fft_order].real = real
            A[:fft_order].imag = imag
            thetas = self.trig2thetas(A)
            density = self.density(thetas=thetas)
            return np.nansum(ne.evaluate("""-(log(density))"""))

        if self._solution is None:
            trig_theta0 = np.fft.rfft(self.sliding_theta)[0, :fft_order]
            x0 = trig_theta0.real, trig_theta0.imag
            self._solution = minimize(mml, x0,
                                      options=dict(disp=True)
                                      ).x
        self.thetas = self.trig2thetas(self._solution, self._T)
        return self._solution

    def fourier_approx(self, fft_order=4, trig_theta=None):
        if trig_theta is None:
            trig_theta = self.solution

        _fourier_approx = \
            np.empty((len(self.copula.theta_start), self.n_doys))
        approx = np.fft.irfft(trig_theta[:fft_order + 1], self.n_doys)
        lower_bound, upper_bound = self.copula.theta_bounds[0]
        approx[approx < lower_bound] = lower_bound
        approx[approx > upper_bound] = upper_bound
        _fourier_approx[0] = approx
        return _fourier_approx[0]

    def fit(self, ranks_u=None, ranks_v=None, **kwds):
        if ranks_u is not None:
            self.ranks_u = ranks_u
        if ranks_v is not None:
            self.ranks_v = ranks_v
        return self.solution

    def trig2thetas(self, trig_theta, _T=None, fft_order=None):
        fft_order = self.fft_order if fft_order is None else fft_order
        if _T is None:
            try:
                _T = self._T
            except AttributeError:
                _T = self._T = (2 * np.pi / 365 * self.doys)[np.newaxis, :]
        doys = np.atleast_1d(365 * np.squeeze(_T) / (2 * np.pi))
        doys_ii = np.where(np.isclose(self.doys_unique, doys[:, None]))[1]
        if len(doys_ii) < len(doys):
            doys_ii = [my.val2ind(self.doys_unique, doy) for doy in doys]
        fourier_thetas = self.fourier_approx(fft_order, trig_theta)
        thetas = np.array([fourier_thetas[doy_i] for doy_i in doys_ii])
        return np.squeeze(thetas.T)

    def density(self, dtimes=None, ranks_u=None, ranks_v=None, thetas=None):
        if dtimes is None:
            dtimes = self.dtimes
            doys = self.doys
        else:
            doys = times.datetime2doy(dtimes)
        if ranks_u is None:
            ranks_u = self.ranks_u
        if ranks_v is None:
            ranks_v = self.ranks_v
        _T = doys * 2 * np.pi / 365
        if thetas is None:
            thetas = self.trig2thetas(self.solution, _T)
        return self.pool.map(self.copula.density, ranks_u, ranks_v, thetas)

    def quantiles(self, dtimes=None, ranks_u=None, ranks_v=None):
        if dtimes is None:
            dtimes = self.dtimes
            doys = self.doys
        else:
            doys = times.datetime2doy(dtimes)
        if ranks_u is None:
            ranks_u = self.ranks_u
        if ranks_v is None:
            ranks_v = self.ranks_v
        _T = doys * 2 * np.pi / 365
        thetas = self.trig2thetas(self.solution, _T)
        qq_u = ranks_u
        qq_v = self.copula.cdf_given_u(qq_u, ranks_v, thetas)
        return qq_u, qq_v

    def sample(self, dtimes=None, qq_u=None, qq_v=None):
        if dtimes is None:
            dtimes = self.dtimes
            doys = self.doys
        else:
            doys = times.datetime2doy(dtimes)
        if qq_u is None:
            qq_u = cops.random_sample(len(doys))
        if qq_v is None:
            qq_v = cops.random_sample(len(doys))
        _T = doys * 2 * np.pi / 365
        thetas = self.trig2thetas(self.solution, _T)
        uu = qq_u
        vv = self.copula.inv_cdf_given_u(uu, qq_v, thetas)
        return uu, vv

    @property
    def likelihood(self):
        if self._likelihood is None:
            density = self.density()
            self._likelihood = np.sum(ne.evaluate("""log(density)"""))
            if self.verbose:
                print("\t%s: %.3f" % (self.copula.name, self._likelihood))
        return self._likelihood

    def __lt__(self, other):
        return self.likelihood < other

    def plot_fourier_fit(self, fft_order=None):
        """Plots the Fourier approximation of all theta elements."""
        fft_order = self.fft_order if fft_order is None else fft_order
        fig, axs = plt.subplots(self.len_theta, sharex=True, squeeze=True)
        if self.len_theta == 1:
            axs = axs,
        thetas = self.fourier_approx_new(fft_order)
        if thetas.shape[1] > 1:
            for theta_i in range(self.len_theta):
                axs[theta_i].plot(self.doys_unique,
                                  self.sliding_theta[theta_i])
                axs[theta_i].plot(self.doys_unique,
                                  self.fourier_approx_new(fft_order)[theta_i],
                                  label="new")
                axs[theta_i].plot(self.doys_unique,
                                  self.fourier_approx(fft_order)[theta_i],
                                  label="old")
                axs[theta_i].grid(True)
                axs[theta_i].set_title("%s\nFourier fft_order: %d"
                                       % (self.copula.name, fft_order))
        plt.legend(loc="best")
        return fig, axs

    def plot_corr(self, sample=None):
        """Plots correlation over doy.

        Notes
        -----
        Fitted correlations are based on random sample, not on
        theory.
        """
        if sample is None:
            sample = self.sample()
        sample = np.array(sample)
        fig, ax = plt.subplots()
        corrs_emp = np.empty(self.n_doys)
        corrs_fit = np.empty(self.n_doys)
        for doy_i in range(self.n_doys):
            doy_mask = self.doy_mask[doy_i]
            corrs_emp[doy_i] = np.corrcoef(self.ranks_u[doy_mask],
                                           self.ranks_v[doy_mask]
                                           )[0, 1]
            corrs_fit[doy_i] = np.corrcoef(sample[0, doy_mask],
                                           sample[1, doy_mask]
                                           )[0, 1]
        ax.plot(self.doys_unique, corrs_emp, label="observed")
        ax.plot(self.doys_unique, corrs_fit, label="fitted")
        ax.set_title("Correlations (%s)" % self.name)
        ax.legend(loc="best")
        ax.grid(True)
        return fig, ax

    def plot_seasonal_densities(self, opacity=.1, *args, **kwds):
        fig, axs = plt.subplots(nrows=2, ncols=2,
                                subplot_kw=dict(aspect="equal"))
        axs = np.ravel(axs)
        doys = np.linspace(0, 366, 5)[:-1]
        for ax, doy in zip(axs, doys):
            doy = int(round(doy))
            theta = self._sliding_theta[doy]
            self.copula.theta = theta
            self.copula.plot_density(ax=ax, scatter=False, *args, **kwds)
            ranks_u = self.ranks_u[self._doy_mask[doy]]
            ranks_v = self.ranks_v[self._doy_mask[doy]]
            ax.scatter(ranks_u, ranks_v,
                       marker="o",
                       facecolor=(0, 0, 0, 0),
                       edgecolor=(0, 0, 0, opacity))
            ax.set_title(r"doy: %d, $\theta=%.3f$" % (doy, theta))
        fig.suptitle("Seasonal copula densities (%s)" % self.name)
        return fig, axs


@my.cache("scop", "As", "phases", "cop_quantiles", "qq_std")
def vg_ph(vg_obj, sc_pars):
    assert vg_obj.K == 2, "Can only handle 2 variables"
    if vg_ph.scop is None:
        ranks_u, ranks_v = np.array([stats.rel_ranks(values)
                                     for values in vg_obj.data_trans])
        vg_ph.scop = SeasonalCop(None, vg_obj.times, ranks_u,
                                 ranks_v)
        vg_ph.cop_quantiles = np.array(vg_ph.scop.quantiles())
        # attach SeasonalCop instance to vg so that does not get lost.
        vg_obj.scop = vg_ph.scop
    if vg_ph.phases is None:
        vg_ph.qq_std = np.array([dists.norm.ppf(q)
                                 for q in vg_ph.cop_quantiles])
        vg_ph.As = np.fft.fft(vg_ph.qq_std)
        vg_ph.phases = np.angle(vg_ph.As)
    T = vg_obj.T
    # phase randomization with same random phases in both variables
    phases_lh = np.random.uniform(0, 2 * np.pi,
                                  T // 2 if T % 2 == 1 else T // 2 - 1)
    phases_lh = np.array([phases_lh, phases_lh])
    phases_rh = -phases_lh[:, ::-1]
    if T % 2 == 0:
        phases = np.hstack((vg_ph.phases[:, 0, None],
                            phases_lh,
                            vg_ph.phases[:, vg_ph.phases.shape[1] // 2, None],
                            phases_rh))
    else:
        phases = np.hstack((vg_ph.phases[:, 0, None],
                            phases_lh,
                            phases_rh))

    try:
        A_new = vg_ph.As * np.exp(1j * phases)
        adjust_variance = False
    except ValueError:
        vg_ph.As = np.fft.fft(vg_ph.qq_std, n=T)
        vg_ph.phases = np.angle(vg_ph.As)
        A_new = vg_ph.As * np.exp(1j * phases)
        adjust_variance = True

    fft_sim = (np.fft.ifft(A_new)).real
    if adjust_variance:
        fft_sim /= fft_sim.std(axis=1)[:, None]

    # from lhglib.contrib.time_series_analysis import time_series as ts
    # fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    # for i, ax in enumerate(axs):
    #     freqs = np.fft.fftfreq(T)
    #     ax.bar(freqs, phases[i], width=.5 / T, label="surrogate")
    #     ax.bar(freqs, vg_ph.phases[i], width=.5 / T, label="data")
    # axs[0].legend(loc="best")

    # my.hist(vg_ph.phases.T, 20)

    # fig, axs = ts.plot_cross_corr(vg_ph.qq_std)
    # fig, axs = ts.plot_cross_corr(fft_sim, linestyle="--", axs=axs, fig=fig)
    # fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
    # for i, ax in enumerate(axs):
    #     ax.plot(vg_ph.qq_std[i])
    #     ax.plot(fft_sim[i], linestyle="--")

    # # fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True)
    # # for i, ax in enumerate(axs):
    # #     ax.plot(phases_lh_signal[i], label="signal phases")
    # #     ax.plot(phases_lh[i], label="random phases")
    # # axs[0].legend(loc="best")

    # plt.show()

    # change in mean scenario
    prim_i = vg_obj.primary_var_ii
    fft_sim[prim_i] += sc_pars.m[prim_i]
    fft_sim[prim_i] += sc_pars.m_t[prim_i]
    qq_u, qq_v = np.array([dists.norm.cdf(values) for values in fft_sim])
    ranks_u_sim, ranks_v_sim = vg_ph.scop.sample(dtimes=vg_obj.sim_times,
                                                 qq_u=qq_u, qq_v=qq_v)
    return np.array([dists.norm.ppf(ranks)
                     for ranks in (ranks_u_sim, ranks_v_sim)])


@my.cache("scop", "spec", "cop_quantiles", "qq_std")
def vg_sim(vg_obj, sc_pars):
    assert vg_obj.K == 2, "Can only handle 2 variables"
    if vg_sim.scop is None:
        ranks_u, ranks_v = np.array([stats.rel_ranks(values)
                                     for values in vg_obj.data_trans])
        vg_sim.scop = SeasonalCop(None, vg_obj.times, ranks_u,
                                  ranks_v)
        vg_sim.cop_quantiles = np.array(vg_sim.scop.quantiles())
        # attach SeasonalCop instance to vg so that does not get lost.
        vg_obj.scop = vg_sim.scop
    if vg_sim.spec is None:
        vg_sim.qq_std = np.array([dists.norm.ppf(q)
                                  for q in vg_sim.cop_quantiles])
        vg_sim.spec = spectral.MultiSpectral(vg_sim.qq_std, vg_sim.qq_std,
                                             T=vg_obj.T, pool_size=100)
    spec_sim = vg_sim.spec.sim
    # change in mean scenario
    spec_sim[vg_obj.primary_var_ii] += sc_pars.m[vg_obj.primary_var_ii]

    # from lhglib.contrib.time_series_analysis import time_series as ts
    # fig, axs = ts.plot_cross_corr(vg_sim.qq_std)
    # fig, axs = ts.plot_cross_corr(spec_sim, linestyle="--", axs=axs, fig=fig)
    # plt.show()

    qq_u, qq_v = np.array([dists.norm.cdf(values)
                           for values in spec_sim])
    ranks_u_sim, ranks_v_sim = vg_sim.scop.sample(dtimes=vg_obj.sim_times,
                                                  qq_u=qq_u, qq_v=qq_v)
    return np.array([dists.norm.ppf(ranks)
                     for ranks in (ranks_u_sim, ranks_v_sim)])


# def res_ranks(doys, data, means, sigmas):
#     ranks_u, ranks_v = np.full((2, len(dtimes)), .5)
#     for doy_i, doy in enumerate(np.unique(doys)):
#         mean = means[:, doy_i]
#         sigma = sigmas[..., doy_i] ** .5
#         doy_mask = doys == doy
#         ranks_u[doy_mask] = (dists.norm.cdf((data[0, doy_mask] -
#                                              mean[0]) /
#                                             sigma[0, 0]))
#         ranks_v[doy_mask] = (dists.norm.cdf((data[1, doy_mask] -
#                                              mean[1]) /
#                                             sigma[1, 1]))
#     # ranks_u[np.isclose(ranks_u, 1)] = np.nan
#     # ranks_v[np.isclose(ranks_v, 1)] = np.nan
#     return ranks_u, ranks_v


# def res_back(doys, ranks, means, sigmas):
#     data1, data2 = np.empty((2, len(doys)))
#     for doy_i, doy in enumerate(np.unique(doys)):
#         mean = means[:, doy_i]
#         sigma = sigmas[..., doy_i] ** .5
#         doy_mask = doys == doy
#         data1[doy_mask] = (dists.norm.ppf(ranks[0, doy_mask]) *
#                            sigma[0, 0] + mean[0])
#         data2[doy_mask] = (dists.norm.ppf(ranks[1, doy_mask]) *
#                            sigma[1, 1] + mean[1])
#     return data1, data2


if __name__ == '__main__':
    # import pandas as pd
    from lhglib.contrib.veathergenerator import vg, vg_base, vg_plotting
    from lhglib.contrib.veathergenerator import config_konstanz_disag as conf
    vg.conf = vg_base.conf = vg_plotting.conf = conf
    from lhglib.contrib.time_series_analysis import distributions as dists
    from weathercop import stats, copulae as cops
    # varnames = "theta", "ILWR"
    # varnames = "theta", "R"
    varnames = "theta", "u"
    met_vg = vg.VG(varnames,
                   refit=True,
                   verbose=True
                   )
    met_vg.fit()

    simt, sim = met_vg.simulate(theta_incr=0, sim_func=vg_ph)
    mean_before = np.nanmean(met_vg.data_raw[1] / 24)
    mean_after = np.nanmean(sim[1])
    print("\t"
          "%s mean (data): %.3f, %s mean (sim): %.3f, diff: %.3f" %
          (varnames[1], mean_before,
           varnames[1], mean_after,
           mean_after - mean_before))
    met_vg.plot_all()
    plt.show()
    vg_ph.clear_cache()

