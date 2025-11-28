import functools
from itertools import repeat
import multiprocessing
import time

import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from varwg import helpers as my, times
from varwg.time_series_analysis import distributions as dists, spectral
from weathercop import copulae as cops, stats
from weathercop import cop_conf


@functools.total_ordering
class SeasonalCop:
    def __init__(
        self,
        copula,
        dtimes,
        ranks_u,
        ranks_v,
        *,
        window_len=15,
        fft_order=4,
        cop_candidates=None,
        verbose=True,
        asymmetry=False,
        fit_mask=None,
    ):
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
        asymmetry : boolean, optional
            Assign weights in likelihood estimation in order to favour
            asymmetrical copulas.
        fit_mask : boolean ndarray or None, optional
            Use only the corresponding variables for fitting.
        """
        if copula is None:
            if cop_candidates is None:
                cop_candidates = [
                    cop
                    for cop in cops.all_cops.values()
                    if not isinstance(cop, cops.Independence)
                ]

            if cop_conf.PROFILE:
                scops = [
                    SeasonalCop(
                        cop,
                        dtimes,
                        ranks_u,
                        ranks_v,
                        window_len=window_len,
                        verbose=verbose,
                        asymmetry=asymmetry,
                        fit_mask=fit_mask,
                    )
                    for cop in cop_candidates
                ]
            else:
                with multiprocessing.Pool(cop_conf.n_nodes) as pool:
                    scops = pool.map(
                        SeasonalCop._unpack,
                        zip(
                            cop_candidates,
                            repeat(dtimes),
                            repeat(ranks_u),
                            repeat(ranks_v),
                            repeat(window_len),
                            repeat(verbose),
                            repeat(asymmetry),
                            repeat(fit_mask),
                        ),
                    )
            # become the copula with the best likelihood
            self.__dict__ = np.nanmax(scops).__dict__
        else:
            self.copula = copula
            self.dtimes = dtimes
            self.ranks_u = ranks_u
            self.ranks_v = ranks_v
            self.window_len = window_len
            self.fft_order = fft_order
            self.asymmetry = asymmetry
            self.fit_mask = fit_mask
            self.verbose = verbose

            self.name = f"seasonal {copula.name}"
            self.T = len(ranks_u)
            self.n_par = self.copula.n_par
            self.doys = times.datetime2doy(dtimes)
            # fmt: off
            timestep = ((self.dtimes[1] -
                         self.dtimes[0]).total_seconds() /
                        (60 ** 2 * 24))
            # fmt: on
            self.doys_unique = np.unique(
                my.round_to_float(self.doys, timestep)
            )
            self.n_doys = len(self.doys_unique)
            self._doy_mask = self._sliding_theta = self._solution = None
            self._likelihood = None
            self.convergence = True
            # we want this to happen in parallel (in the
            # multiprocessing.pool), not later in np.nanmax(scops)!
            # the likelihood value is cached and will not be
            # calculated again.
            self.likelihood

    def __str__(self):
        return self.name

    @staticmethod
    def _unpack(args):
        copula, dtimes, ranks_u, ranks_v = args[:4]
        window_len, verbose, asymmetry, fit_mask = args[4:]
        return SeasonalCop(
            copula,
            dtimes,
            ranks_u,
            ranks_v,
            window_len=window_len,
            verbose=verbose,
            asymmetry=asymmetry,
            fit_mask=fit_mask,
        )

    def _call_cdf_func(self, method_name, t=None, *, conditioned, condition):
        if t is None:
            theta = self.thetas
        else:
            theta = self.thetas[t % self.n_doys]
        method = getattr(self.copula.__class__, method_name)
        try:
            return method(self.copula, conditioned, condition, theta)
        except TypeError:
            return method(conditioned, condition, theta)

    def cdf_given_u(self, t=None, *, conditioned, condition):
        return self._call_cdf_func(
            "cdf_given_u", t, conditioned=condition, condition=conditioned
        )

    def cdf_given_v(self, t=None, *, conditioned, condition):
        return self._call_cdf_func(
            "cdf_given_v", t, conditioned=conditioned, condition=condition
        )

    def inv_cdf_given_u(self, t=None, *, conditioned, condition):
        return self._call_cdf_func(
            "inv_cdf_given_u", t, conditioned=condition, condition=conditioned
        )

    def inv_cdf_given_v(self, t=None, *, conditioned, condition):
        return self._call_cdf_func(
            "inv_cdf_given_v", t, conditioned=conditioned, condition=condition
        )

    @property
    def doy_mask(self):
        """Returns a (n_unique_doys, T) ndarray"""
        if self._doy_mask is None:
            window_len, doys = self.window_len, self.doys
            self._doy_mask = np.empty(
                (len(self.doys_unique), self.T), dtype=bool
            )
            for doy_i, doy in enumerate(self.doys_unique):
                ii = (doys > doy - window_len) & (doys <= doy + window_len)
                if (doy - window_len) < 0:
                    ii |= doys > (365.0 - window_len + doy)
                if (doy + window_len) > 365:
                    ii |= doys < (doy + window_len - 365.0)
                self._doy_mask[doy_i] = ii
        return self._doy_mask

    @property
    def sliding_theta(self):
        if self._sliding_theta is None:
            self._sliding_theta = np.ones((self.n_doys, self.n_par))
            for doy_ii in tqdm(
                range(self.n_doys),
                # disable=(not self.verbose)
                disable=True,
            ):
                doy_mask = self.doy_mask[doy_ii]
                ranks_u = self.ranks_u[doy_mask]
                ranks_v = self.ranks_v[doy_mask]
                if self.fit_mask is not None:
                    fit_mask_doy = self.fit_mask[doy_mask]
                    f_kwds = dict(fit_mask=fit_mask_doy)
                    if fit_mask_doy.mean() < 0.05:
                        self._sliding_theta[doy_ii] = np.nan
                        continue
                else:
                    f_kwds = dict()

                # try:
                #     theta = self.copula.fit(ranks_u, ranks_v, **f_kwds)
                # except cops.NoConvergence:
                #     theta = self.copula.theta_start

                if doy_ii == 0:
                    try:
                        theta = self.copula.fit(ranks_u, ranks_v, **f_kwds)
                    except cops.NoConvergence:
                        theta = self.copula.theta_start
                else:
                    x0 = self._sliding_theta[doy_ii - 1]
                    if not np.isfinite(x0):
                        x0 = self.copula.theta_start
                    try:
                        theta = self.copula.fit(
                            ranks_u, ranks_v, x0=x0, **f_kwds
                        )
                    except cops.NoConvergence:
                        # theta = np.nan
                        try:
                            # theta = self.copula.fit(ranks_u, ranks_v, **f_kwds)
                            theta = self.copula.fit(
                                ranks_u,
                                ranks_v,
                                x0=self.copula.theta_start,
                                **f_kwds,
                            )
                        except cops.NoConvergence:
                            # warnings.warn("No Convergence reached in "
                            #               f"{self.copula.name}.")
                            theta = np.nan
                            # __import__("pdb").set_trace()
                            # theta = self.copula.fit(ranks_u, ranks_v, **f_kwds)

                # I don't trust fittings that are very close to the
                # parameter bounds!
                theta_lower, theta_upper = self.copula.theta_bounds[0]
                if isinstance(theta, tuple):
                    theta = theta[0]
                if theta < theta_lower or theta > theta_upper:
                    theta = np.nan
                if np.isclose(theta, theta_lower) or np.isclose(
                    theta, theta_upper
                ):
                    theta = np.nan

                self._sliding_theta[doy_ii] = theta

            # try to interpolate over bad fittings
            thetas = self._sliding_theta
            # self.theta_nan_mask = [np.isnan(thet) for thet in thetas]
            self.theta_nan_mask = np.isnan(thetas)
            self._sliding_theta = self._interp_thetas(thetas)
            if np.any(np.isnan(self._sliding_theta)):
                self.convergence = False
        return self._sliding_theta.T

    def _interp_thetas(self, thetas):
        theta_filled = np.empty_like(thetas)
        for theta_i, theta in enumerate(thetas.T):
            nan_mask = np.isnan(theta)
            if nan_mask.mean() > 0.5:
                if self.verbose:
                    print(
                        f"\tRejected seasonal {self.copula.name} "
                        f"({100 * (~nan_mask).mean():.1f}% convergences)"
                    )
                self.convergence = False
                break
            if np.any(nan_mask):
                half = len(theta) // 2
                theta_pad = np.concatenate(
                    (theta[-half:], theta, theta[:half])
                )
                # interp = my.interp_nan(theta_pad, max_interp=5)[half:-half]
                interp = my.interp_nonfin(theta_pad)[half:-half]
                theta_filled[:, theta_i] = interp
            else:
                theta_filled[:, theta_i] = thetas[:, theta_i]
        # assert np.all(np.isfinite(theta_filled))
        return theta_filled

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
        # this is very slow, as it optimizes fft parameters!
        fft_len = len(np.fft.rfftfreq(self.n_doys))
        A = np.zeros((1, fft_len), dtype=complex)
        fft_order = self.fft_order
        ranks_u = self.ranks_u
        ranks_v = self.ranks_v
        if self.fit_mask is None:
            fit_mask = slice(None)
            censor = False
        else:
            fit_mask = self.fit_mask
            n_dry = np.sum(~fit_mask)
            ranks_u = ranks_u[fit_mask]
            ranks_v = ranks_v[fit_mask]
            u_max = np.array([np.max(self.ranks_u[~fit_mask])])
            v_max = np.array([np.max(self.ranks_v[~fit_mask])])
            censor = True

        def fill_A(A_parts):
            real, imag = A_parts.reshape(2, -1)
            A[:, :fft_order].real = real
            A[:, :fft_order].imag = imag

        def mml(A_parts):
            fill_A(A_parts)
            thetas = self.trig2thetas(A)
            dens = self.density(
                ranks_u=ranks_u, ranks_v=ranks_v, thetas=thetas[fit_mask]
            )
            loglike = np.nansum(ne.evaluate("""-(log(dens))"""))
            if censor:
                one = 1.0 - 1e-12
                if u_max > v_max:
                    lower_int = self.copula.cdf(
                        np.full(n_dry, one),
                        np.full(n_dry, v_max),
                        thetas[~fit_mask],
                    )
                else:
                    lower_int = self.copula.cdf(
                        np.full(n_dry, u_max),
                        np.full(n_dry, one),
                        thetas[~fit_mask],
                    )
                # loglike -= np.sum(np.log(lower_int))
                loglike -= np.nansum(ne.evaluate("""log(lower_int)"""))
            return loglike

        if self._solution is None:
            trig_theta0 = np.fft.rfft(self.sliding_theta)[0, :fft_order]
            x0 = trig_theta0.real, trig_theta0.imag
            _solution = minimize(
                mml,
                x0,
                # options=dict(disp=True)
            ).x
            fill_A(_solution)
            self._solution = A

        self.thetas = self.trig2thetas(self._solution, self._T)
        return self._solution

    def fourier_approx(self, fft_order=4, trig_theta=None):
        if trig_theta is None:
            trig_theta = self.solution

        _fourier_approx = np.empty((self.copula.n_par, self.n_doys))
        approx = np.fft.irfft(trig_theta[:, : fft_order + 1], self.n_doys)
        lower_bound, upper_bound = self.copula.theta_bounds[0]
        if not np.any(np.isnan(approx)):
            approx[approx < lower_bound] = lower_bound
            approx[approx > upper_bound] = upper_bound
        _fourier_approx[:] = approx
        return _fourier_approx

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
        # # no crazy waves where the fitting failed
        # fourier_thetas[self.theta_nan_mask.T] = np.nan
        # fourier_thetas = self._interp_thetas(fourier_thetas)
        thetas = np.array([fourier_thetas[:, doy_i] for doy_i in doys_ii])
        # assert all(np.isfinite(thetas))
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
        dens = self.copula.density(ranks_u, ranks_v, thetas)
        if self.asymmetry:
            ranks_u, ranks_v = self.ranks_u, self.ranks_v
            mask = np.full_like(ranks_u, False, dtype=bool)
            asy1 = stats.asymmetry1(ranks_u, ranks_v)
            asy2 = stats.asymmetry2(ranks_u, ranks_v)
            lower, upper, thresh = 0.25, 0.75, 0
            if asy1 > thresh:
                mask |= (ranks_u > upper) & (ranks_v > upper)
            elif asy1 < -thresh:
                mask |= (ranks_u < lower) & (ranks_v < lower)
            if asy2 > thresh:
                mask |= (ranks_u > upper) & (ranks_v < lower)
            elif asy2 < -thresh:
                mask |= (ranks_u < lower) & (ranks_v > upper)
            # dens[mask] *= 100
            dens[~mask] *= 0
        return dens

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
            if not self.convergence:
                self._likelihood = -np.inf
            else:
                before = time.time()
                density = self.density()
                mask = np.isfinite(density)
                density[~mask] = 1e-15
                density[density <= 0] = 1e-15
                # self._likelihood = float(ne.evaluate("""sum(log(density))"""))
                # if self.verbose:
                #     print(f"\t{self.copula.name}: {self.likelihood:.3f} (symm)")

                self._likelihood = float(ne.evaluate("""sum(log(density))"""))
                if self.verbose:
                    duration = time.time() - before
                    print(
                        f"\t{self.copula.name}: {self.likelihood:.3f}"
                        f" ({duration:.3f}s)"
                    )
                # if self.verbose:
                #     print(f"\t{self.copula.name}: {self.likelihood:.3f} (asymm)")
        return self._likelihood

    def __lt__(self, other):
        return self.likelihood < other

    def plot_fourier_fit(self, fft_order=None, fig=None, ax=None, title=None):
        """Plots the Fourier approximation of all theta elements."""
        fft_order = self.fft_order if fft_order is None else fft_order
        if fig is None or ax is None:
            fig, axs = plt.subplots(self.n_par, sharex=True, squeeze=True)
            if self.n_par == 1:
                axs = (axs,)
        else:
            fig = plt.gcf()
            axs = (ax,)
        for theta_i in range(self.n_par):
            ax = axs[theta_i]
            ax.plot(self.doys_unique, self.sliding_theta[theta_i])
            trig_theta0 = np.fft.rfft(self.sliding_theta)
            trig_theta0[:, fft_order + 1 :] = 0
            theta0 = self.trig2thetas(trig_theta0, self._T)
            like0 = np.sum(np.log(self.density(thetas=theta0)))
            fourier_appr = self.fourier_approx(
                fft_order, trig_theta=trig_theta0
            )
            ax.plot(self.doys_unique, np.squeeze(fourier_appr), label="raw")
            trig_theta0[:, fft_order + 1 :] = 0
            approx = np.squeeze(
                self.fourier_approx(fft_order, trig_theta=trig_theta0)
            )
            ax.plot(self.doys_unique, approx, label=f"prelim {like0:.2f}")
            # solution_prelim = self._solution
            # self._solution = None
            # self.solution_mll
            ax.plot(
                self.doys_unique,
                np.squeeze(self.fourier_approx(fft_order)),
                label=f"mll {self.likelihood:.2f}",
            )
            # self._solution = solution_prelim
            ax.grid(True)
            ax.set_title(
                (f"{self.copula.name}\n" f"Fourier fft_order: {fft_order}")
                if title is None
                else title
            )
            ax.legend(loc="best")
        return fig, axs

    def plot_corr(self, sample=None, fig=None, ax=None, title=None):
        """Plots correlation over doy.

        Notes
        -----
        Fitted correlations are based on random sample, not on
        theory.
        """
        if sample is None:
            # draw a large sample, so this is closer to asymptopia
            # sample = self.sample(np.concatenate(10 * [self.dtimes]))
            sample = self.sample(self.dtimes)
        sample = np.array(sample)
        if fig is None or ax is None:
            fig, ax = plt.subplots()
        corrs_emp = np.empty(self.n_doys)
        corrs_fit = np.empty(self.n_doys)
        for doy_i in range(self.n_doys):
            doy_mask = self.doy_mask[doy_i]
            corrs_emp[doy_i] = np.corrcoef(
                self.ranks_u[doy_mask], self.ranks_v[doy_mask]
            )[0, 1]
            corrs_fit[doy_i] = np.corrcoef(
                sample[0, doy_mask], sample[1, doy_mask]
            )[0, 1]
        ax.plot(self.doys_unique, corrs_emp, label="observed")
        ax.plot(self.doys_unique, corrs_fit, label="simulated")
        ax.set_title(
            (f"Correlations ({self.name})") if title is None else title
        )
        ax.legend(loc="best")
        ax.grid(True)
        return fig, ax

    def plot_qq(
        self,
        fig=None,
        ax=None,
        opacity=0.25,
        s_kwds=None,
        title=None,
        *args,
        **kwds,
    ):
        if s_kwds is None:
            s_kwds = dict(
                marker="o",
                s=1,
                facecolors=(0, 0, 0, 0),
                edgecolors=(0, 0, 0, opacity),
            )

        empirical = cops.bipit(self.ranks_u, self.ranks_v)
        theoretical = self.copula.copula_func(
            self.ranks_u, self.ranks_v, self.thetas
        )
        if ax is None:
            fig, ax = plt.subplots(
                # subplot_kw=dict(aspect="equal")
            )
        if fig is None:
            fig = plt.gcf()
        ax.scatter(empirical, theoretical - empirical, **s_kwds)
        if title is None:
            title = "qq " + self.name[len("fitted ") :]
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax

    def plot_seasonal_densities(
        self,
        fig=None,
        axs=None,
        plot_doys=None,
        opacity=0.1,
        skwds=None,
        *args,
        **kwds,
    ):
        if skwds is None:
            skwds = {}
        if fig is None or axs is None:
            fig, axs = plt.subplots(
                nrows=2, ncols=2, subplot_kw=dict(aspect="equal")
            )
        else:
            for ax in np.ravel(axs):
                ax.set_aspect("equal")
        axs = np.ravel(axs)
        if plot_doys is None:
            plot_doys = np.linspace(0, 366, 6)[1:-1]
        for ax, doy in zip(axs, plot_doys):
            doy = int(round(doy))
            theta = self.thetas[doy]
            # self.copula.theta = theta
            self.copula.plot_density(
                theta=(np.array([theta]),),
                fig=fig,
                ax=ax,
                scatter=False,
                *args,
                **kwds,
            )
            ranks_u = self.ranks_u[self._doy_mask[doy]]
            ranks_v = self.ranks_v[self._doy_mask[doy]]
            ax.scatter(
                ranks_u,
                ranks_v,
                marker="o",
                facecolor=(0, 0, 0, 0),
                edgecolor=(0, 0, 0, opacity),
                **skwds,
            )
            ax.set_title(rf"doy: {doy}, $\theta={theta:.3f}$")
        fig.suptitle(f"Seasonal copula densities ({self.name})")
        return fig, axs

    def plot_density(self, fig=None, ax=None, *args, **kwds):
        """Plots density with mean theta!"""
        if fig is None or ax is None:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, subplot_kw=dict(aspect="equal")
            )
        self.copula.plot_density(
            theta=(np.array([self.thetas.mean()]),),
            fig=fig,
            ax=ax,
            *args,
            **kwds,
        )
        return fig, ax


def phase_rand(*, ranks=None, data_stdn=None, T=None):
    if ranks is not None:
        transform = True
        data_stdn = np.array([dists.norm.ppf(ranks_) for ranks_ in ranks])
    elif data_stdn is None:
        raise RuntimeError("Must supply ranks or data_stdn.")
    else:
        transform = False

    if T is None:
        T = data_stdn.shape[1]
        adjust_variance = False
    else:
        adjust_variance = True
    K = data_stdn.shape[0]

    As = np.fft.fft(data_stdn, n=T)
    middle_i = As.shape[1] // 2
    phase_first, phase_middle = np.angle(As[:, [0, middle_i]])
    # phase randomization with same random phases in both variables
    phases_lh = varwg.get_rng().uniform(
        0, 2 * np.pi, T // 2 if T % 2 == 1 else T // 2 - 1
    )
    phases_lh = np.array(K * [phases_lh])
    phases_rh = -phases_lh[:, ::-1]
    if T % 2 == 0:
        phases = np.hstack(
            (phase_first[:, None], phases_lh, phase_middle[:, None], phases_rh)
        )
    else:
        phases = np.hstack((phase_first[:, None], phases_lh, phases_rh))

    A_new = As * np.exp(1j * phases)
    fft_sim = (np.fft.ifft(A_new)).real
    if adjust_variance:
        fft_sim /= fft_sim.std(axis=1)[:, None]

    if transform:
        # we were given ranks, so we give ranks back
        fft_sim = np.array([dists.norm.cdf(data) for data in fft_sim])

    return fft_sim


@my.cache("scop", "As", "phases", "cop_quantiles", "qq_std")
def vg_ph(vg_obj, sc_pars):
    assert vg_obj.K == 2, "Can only handle 2 variables"
    if vg_ph.scop is None:
        ranks_u, ranks_v = np.array(
            [stats.rel_ranks(values) for values in vg_obj.data_trans]
        )
        vg_ph.scop = SeasonalCop(None, vg_obj.times, ranks_u, ranks_v)
        vg_ph.cop_quantiles = np.array(vg_ph.scop.quantiles())
        # attach SeasonalCop instance to vg so that does not get lost.
        vg_obj.scop = vg_ph.scop
    if vg_ph.phases is None:
        vg_ph.qq_std = np.array(
            [dists.norm.ppf(q) for q in vg_ph.cop_quantiles]
        )
    T = vg_obj.T
    fft_sim = phase_rand(data_stdn=vg_ph.qq_std, T=T)

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
    ranks_u_sim, ranks_v_sim = vg_ph.scop.sample(
        dtimes=vg_obj.sim_times, qq_u=qq_u, qq_v=qq_v
    )
    return np.array(
        [dists.norm.ppf(ranks) for ranks in (ranks_u_sim, ranks_v_sim)]
    )


@my.cache("scop", "spec", "cop_quantiles", "qq_std")
def vg_sim(vg_obj, sc_pars):
    assert vg_obj.K == 2, "Can only handle 2 variables"
    if vg_sim.scop is None:
        ranks_u, ranks_v = np.array(
            [stats.rel_ranks(values) for values in vg_obj.data_trans]
        )
        vg_sim.scop = SeasonalCop(None, vg_obj.times, ranks_u, ranks_v)
        vg_sim.cop_quantiles = np.array(vg_sim.scop.quantiles())
        # attach SeasonalCop instance to vg so that does not get lost.
        vg_obj.scop = vg_sim.scop
    if vg_sim.spec is None:
        vg_sim.qq_std = np.array(
            [dists.norm.ppf(q) for q in vg_sim.cop_quantiles]
        )
        vg_sim.spec = spectral.MultiSpectral(
            vg_sim.qq_std, vg_sim.qq_std, T=vg_obj.T_sim, pool_size=100
        )
    spec_sim = vg_sim.spec.sim
    # change in mean scenario
    spec_sim[vg_obj.primary_var_ii] += sc_pars.m[vg_obj.primary_var_ii]

    # from lhglib.contrib.time_series_analysis import time_series as ts
    # fig, axs = ts.plot_cross_corr(vg_sim.qq_std)
    # fig, axs = ts.plot_cross_corr(spec_sim, linestyle="--", axs=axs, fig=fig)
    # plt.show()

    qq_u, qq_v = np.array([dists.norm.cdf(values) for values in spec_sim])
    ranks_u_sim, ranks_v_sim = vg_sim.scop.sample(
        dtimes=vg_obj.sim_times, qq_u=qq_u, qq_v=qq_v
    )
    return np.array(
        [dists.norm.ppf(ranks) for ranks in (ranks_u_sim, ranks_v_sim)]
    )


if __name__ == "__main__":
    # import pandas as pd
    import varwg
    from varwg import base as vg_base, plotting as vg_plotting
    import config_konstanz_disag as conf

    varwg.conf = vg_base.conf = vg_plotting.conf = conf
    from varwg.time_series_analysis import distributions as dists
    from weathercop import stats, copulae as cops

    # varnames = "theta", "ILWR"
    # varnames = "theta", "R"
    varnames = "theta", "u"
    met_vg = varwg.VG(varnames, refit=True, verbose=True)
    met_vg.fit()

    simt, sim = met_vg.simulate(theta_incr=0, sim_func=vg_ph)
    mean_before = np.nanmean(met_vg.data_raw[1] / 24)
    mean_after = np.nanmean(sim[1])
    print(
        "\t"
        "%s mean (data): %.3f, %s mean (sim): %.3f, diff: %.3f"
        % (
            varnames[1],
            mean_before,
            varnames[1],
            mean_after,
            mean_after - mean_before,
        )
    )
    met_vg.plot_all()
    plt.show()
    vg_ph.clear_cache()
