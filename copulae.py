"""Bivariate Copulas intended for Vine Copulas."""
from abc import ABCMeta, abstractproperty
import functools
import warnings
import os
import importlib
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sympy
from sympy.utilities import autowrap
from sympy import ln, exp  # , asin
# from sympy import stats as sstats
# from sympy.functions.special import error_functions
from mpmath import mp
from scipy.optimize import minimize
from scipy.special import erfinv, erf

from weathercop import cop_conf as conf
from weathercop import tools, stats

# used wherever theta can be inf in principle
theta_large = 1e3


def ufuncify(cls, name, uargs, expr, *args, **kwds):
    expr_hash = tools.hash_cop(expr)
    module_name = "%s_%s_%s" % (cls.name, name, expr_hash)
    try:
        with tools.chdir(conf.ufunc_tmp_dir):
            ufunc = importlib.import_module("%s_0" % module_name).autofunc_c
    except ImportError:
        warnings.warn("Compiling %s" % repr(expr))
        _filename_orig = autowrap.CodeWrapper._filename
        _module_basename_orig = autowrap.CodeWrapper._module_basename
        _module_counter_orig = autowrap.CodeWrapper._module_counter
        autowrap.CodeWrapper._filename = "%s_code" % module_name
        autowrap.CodeWrapper._module_basename = module_name
        autowrap.CodeWrapper._module_counter = 0
        ufunc = autowrap.ufuncify(uargs, expr, *args, **kwds)
        autowrap.CodeWrapper._module_basename = _module_basename_orig
        autowrap.CodeWrapper._module_counter = _module_counter_orig
        autowrap.CodeWrapper._filename = _filename_orig
    return ufunc


def random_sample(size, bound=1e-12):
    """Sample in the closed interval (0, 1)."""
    return (1 - 2 * bound) * np.random.random_sample(size) + bound


def positive(func):
    @functools.wraps(func)
    def inner(*args, **kwds):
        if isinstance(args[0], Copulae):
            args = args[1:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(*args, **kwds)
            result[(result < 0) | (~np.isfinite(result))] = 1e-15
        return result
    return inner


def broadcast_2d(func):
    @functools.wraps(func)
    def inner(*args, **kwds):
        if isinstance(args[0], Copulae):
            args = args[1:]
        args = [np.atleast_2d(arg) for arg in args]
        shape_broad = np.array([arg.shape for arg in args]).max(axis=0)
        args_broad_raveled = []
        for array in args:
            array_broad = np.empty(shape_broad)
            array_broad[:] = array
            args_broad_raveled += [array_broad.ravel()]
        result = func(*args_broad_raveled, **kwds)
        return result.reshape(shape_broad)
    return inner


class NoConvergence(Exception):
    pass


class MetaCop(ABCMeta):
    backend = "cython"

    def __new__(cls, name, bases, cls_dict):
        new_cls = super().__new__(cls, name, bases, cls_dict)
        new_cls.name = name.lower()
        # print("in MetaCop with " + name)
        if "cop_expr" in cls_dict:
            new_cls.dens_func = MetaCop.density_from_cop(new_cls)
            new_cls.cdf_given_u = MetaCop.cdf_given_u(new_cls)
            new_cls.cdf_given_v = MetaCop.cdf_given_v(new_cls)
            # new_cls.inv_cdf_given_u = MetaCop.inv_cdf_given_u(new_cls)
            # new_cls.inv_cdf_given_v = MetaCop.inv_cdf_given_v(new_cls)
            new_cls.copula_func = MetaCop.copula_func(new_cls)
        elif "dens_expr" in cls_dict and "dens_func" not in cls_dict:
            new_cls.dens_func = MetaCop.density_func(new_cls)
        if "inv_cdf_given_uu_expr" in cls_dict:
            new_cls.inv_cdf_given_u = MetaCop.inv_cdf_given_u(new_cls)
        if "inv_cdf_given_vv_expr" in cls_dict:
            new_cls.inv_cdf_given_v = MetaCop.inv_cdf_given_v(new_cls)
        return new_cls

    def copula_func(cls):
        uu, vv, *theta = sympy.symbols(cls.par_names)
        ufunc = ufuncify(cls, "copula",
                         [uu, vv] + theta, cls.cop_expr,
                         backend=MetaCop.backend,
                         tempdir=conf.ufunc_tmp_dir)
        if MetaCop.backend in ("f2py", "cython"):
            ufunc = broadcast_2d(ufunc)
        return positive(ufunc)

    def density_func(cls):
        uu, vv, *theta = sympy.symbols(cls.par_names)
        dens_expr = cls.dens_expr
        ufunc = ufuncify(cls, "density",
                         [uu, vv] + theta, dens_expr,
                         backend=MetaCop.backend,
                         tempdir=conf.ufunc_tmp_dir)
        if MetaCop.backend in ("f2py", "cython"):
            ufunc = broadcast_2d(ufunc)
        return positive(ufunc)

    def conditional_cdf(cls, conditioning):
        uu, vv, *theta = sympy.symbols(cls.par_names)
        expr_attr = "cdf_given_%s_expr" % conditioning
        try:
            conditional_cdf = getattr(cls, expr_attr)
        except AttributeError:
            with tools.shelve_open(conf.sympy_cache) as sh:
                key = ("%s_cdf_given_%s__%s" %
                       (cls.name, conditioning, tools.hash_cop(cls)))
                if key not in sh:
                    warnings.warn("Generating conditional %s" % cls.name)
                    # a good cop always stays positive!
                    good_cop = sympy.Piecewise((cls.cop_expr,
                                                cls.cop_expr > 0),
                                               (0, True))
                    conditional_cdf = sympy.diff(good_cop, conditioning)
                    conditional_cdf = sympy.simplify(conditional_cdf)
                    # conditional_cdf = \
                    #     sympy.Piecewise((0, uu < 1e-12),
                    #                     (0, vv < 1e-12),
                    #                     (1, uu > (1 - 1e-12)),
                    #                     (1, vv > (1 - 1e-12)),
                    #                     (conditional_cdf, True))
                    sh[key] = conditional_cdf
                conditional_cdf = sh[key]
            setattr(cls, expr_attr, conditional_cdf)
        ufunc = ufuncify(cls, "conditional_cdf",
                         [uu, vv] + theta, conditional_cdf,
                         backend=MetaCop.backend,
                         tempdir=conf.ufunc_tmp_dir)
        if MetaCop.backend in ("f2py", "cython"):
            ufunc = broadcast_2d(ufunc)
        return positive(ufunc)

    def cdf_given_u(cls):
        return cls.conditional_cdf(sympy.symbols("uu"))

    def cdf_given_v(cls):
        return cls.conditional_cdf(sympy.symbols("vv"))

    def inverse_conditional_cdf(cls, conditioning):
        uu, vv, qq, theta = sympy.symbols(("uu", "vv", "qq", "theta"))
        # conditioned = list(set((uu, vv)) - set([conditioning]))[0]
        # with tools.shelve_open(conf.sympy_cache) as sh:
        #     key = ("%s_inv_cdf_given_%s_%s" %
        #            (cls.name, conditioning, tools.hash_cop(cls)))
        #     if key not in sh:
        #         warnings.warn("Generating inverse conditional %s" %
        #                       cls.name)
        #         cdf_given_expr = getattr(cls,
        #                                  "cdf_given_%s_expr" % conditioning)
        #         try:
        #             inv_cdf = sympy.solve(cdf_given_expr - qq, conditioned)
        #         except NotImplementedError:
        #             warnings.warn("Derivation of inv.-conditional failed for" +
        #                           " %s" % cls.name)
        #             return
        #         sh[key] = inv_cdf
        #     inv_cdf = sh[key]
        # setattr(cls, "inv_cdf_given_%s" % conditioning, inv_cdf)
        inv_cdf = getattr(cls, "inv_cdf_given_%s_expr" % conditioning)
        ufunc = ufuncify(cls, "inv_cdf_given_%s" % conditioning,
                         [qq, conditioning, theta], inv_cdf,
                         backend=MetaCop.backend,
                         tempdir=conf.ufunc_tmp_dir)
        if MetaCop.backend in ("f2py", "cython"):
            ufunc = broadcast_2d(ufunc)
        return positive(ufunc)

    def inv_cdf_given_u(cls):
        return cls.inverse_conditional_cdf(sympy.symbols("uu"))

    def inv_cdf_given_v(cls):
        return cls.inverse_conditional_cdf(sympy.symbols("vv"))

    def density_from_cop(cls):
        """Copula density obtained by sympy differentiation compiled with
        cython.
        """
        uu, vv, *theta = sympy.symbols(cls.par_names)
        with tools.shelve_open(conf.sympy_cache) as sh:
            key = "%s_density_%s" % (cls.name, tools.hash_cop(cls))
            if key not in sh:
                warnings.warn("Generating density for %s" % cls.name)
                dens_expr = sympy.diff(cls.cop_expr, uu, vv)
                dens_expr = sympy.Piecewise((dens_expr, cls.cop_expr > 0),
                                            (0, True))
                sh[key] = sympy.simplify(dens_expr)
            dens_expr = sh[key]
        # for outer pleasure
        cls.dens_expr = dens_expr
        ufunc = ufuncify(cls, "density",
                         [uu, vv] + theta, dens_expr,
                         backend=MetaCop.backend,
                         tempdir=conf.ufunc_tmp_dir)
        if MetaCop.backend in ("f2py", "cython"):
            ufunc = broadcast_2d(ufunc)
        return positive(ufunc)


class MetaArch(MetaCop):

    def __new__(cls, name, bases, cls_dict):
        # print("in MetaArch with " + name)
        if ("gen_expr" in cls_dict) and ("cop_expr" not in cls_dict):
            gen = cls_dict["gen_expr"]
            uu, vv, x, t = sympy.symbols(("uu", "vv", "x", "t"))
            with tools.shelve_open(conf.sympy_cache) as sh:
                key = "%s_cop_%s" % (name, tools.hash_cop(gen))
                if key not in sh:
                    warnings.warn("Generating inv. gen for %s" % name)
                    if "gen_inv" not in cls_dict:
                        gen_inv = sympy.solve(gen - x, t)[0]
                    cop = gen_inv.subs(x, gen.subs(t, uu) + gen.subs(t, vv))
                    sh[key] = sympy.simplify(cop)
                cop = sh[key]
            cls_dict["cop_expr"] = cop
        new_cls = super().__new__(cls, name, bases, cls_dict)
        return new_cls


class Copulae(metaclass=MetaCop):

    """Base of all copula implementations, defining what a copula
    must implement to be a copula."""

    theta_bounds = None

    @abstractproperty
    def theta_start(self):
        """Starting solution for parameter estimation."""
        pass

    @abstractproperty
    def par_names(self):
        pass

    def __call__(self, *theta):
        return Frozen(self, *theta)

    def density(self, uu, vv, *theta):
        if len(theta) > 1:
            theta = [np.full_like(uu, the) for the in theta]
        else:
            theta = np.full_like(uu, theta),
        return self.dens_func(uu, vv, *theta)

    def inv_cdf_given_u(self, ranks_u, quantiles, theta=None):
        """Numeric inversion of the cdf_given_u, to be used as a last resort.
        """
        theta = self.theta if theta is None else theta
        ranks_u, quantiles = map(np.atleast_1d, (ranks_u, quantiles))
        return np.array([
            sp.optimize.brentq(lambda y: self.cdf_given_u(u, y, theta) - q,
                               # 1e-12, 1 - 1e-12,
                               0, 1)
            for u, q in zip(ranks_u, quantiles)])

    def inv_cdf_given_v(self, quantiles, ranks_v, theta=None):
        """Numeric inversion of the cdf_given_v, to be used as a last resort.
        """
        theta = self.theta if theta is None else theta
        ranks_v, quantiles = map(np.atleast_1d, (ranks_v, quantiles))
        return np.array([
            sp.optimize.brentq(lambda y: self.cdf_given_v(y, v, theta) - q,
                               # 1e-12, 1 - 1e-12,
                               0, 1)
            for v, q in zip(ranks_v, quantiles)])

    def sample(self, size, theta=None):
        uu = random_sample(size)
        xx = random_sample(size)
        vv = self.inv_cdf_given_u(uu, xx, *theta)
        return uu, vv

    def generate_fitted(self, ranks_u, ranks_v, *args, **kwds):
        """Returns a Fitted instance that contains ranks_u, ranks_v and the
        fitted theta.
        """
        theta = self.fit(ranks_u, ranks_v, *args, **kwds)
        return Fitted(self, ranks_u, ranks_v, theta)

    def fit(self, *args, **kwds):
        # overwrite this function in child implementations, if a
        # better method than maximum likelihood is available as
        # fitting procedure.
        return self.fit_ml(*args, **kwds)

    def fit_ml(self, ranks_u, ranks_v, method=None, verbose=False):
        """Maximum likelihood estimate."""

        def neg_log_likelihood(theta):
            # for (lower, upper), par in zip(self.theta_bounds, theta):
            #     if lower <= par <= upper:
            #         return -np.inf
            dens = self.density(ranks_u, ranks_v, theta)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # mask = (dens > 0) & np.isfinite(dens)
                # dens_masked = dens[mask]
                # if len(dens_masked) == 0:
                #     return -np.inf
                # loglike = -np.sum(np.log(dens_masked))
                mask = (dens <= 0) | ~np.isfinite(dens)
                if np.any(mask):
                    return -np.inf
                # dens[mask] = 1e-9
                loglike = -np.sum(np.log(dens))
            return loglike

        result = minimize(neg_log_likelihood, self.theta_start,
                          bounds=self.theta_bounds,
                          method=method,
                          options=(dict(disp=True) if verbose else None))
        self.theta = result.x
        self.likelihood = -result.fun if result.success else -np.inf
        if not result.success:
            raise NoConvergence
        return self.theta

    def plot_cop_dens(self, theta=None, scatter=True, kind="contourf",
                      opacity=.1):
        fig, axs = plt.subplots(ncols=2, subplot_kw=dict(aspect="equal"))
        self.plot_copula(theta=theta, ax=axs[0])
        self.plot_density(theta=theta, scatter=scatter, kind=kind,
                          opacity=opacity, ax=axs[1])
        return axs

    def plot_density(self, theta=None, scatter=True, ax=None,
                     kind="contourf", opacity=.1, sample_size=1000,
                     skwds=None):
        if theta is None:
            try:
                theta = self.theta
            except AttributeError:
                theta = self.theta_start
        if skwds is None:
            skwds = dict()
        uu = vv = stats.rel_ranks(np.arange(1000))
        density = self.density(uu[None, :], vv[:, None], *theta)
        if not isinstance(self, Independence):
            # get rid of large values for visualizations sake
            density[density >= np.sort(density.ravel())[-10]] = np.nan
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        if kind == "contourf":
            ax.contourf(uu, vv, density, 40)
        elif kind == "contour":
            ax.contour(uu, vv, density, 40)
        if scatter:
            try:
                u_sample, v_sample = self.sample(size=1000, theta=theta)
                ax.scatter(u_sample, v_sample,
                           marker="o",
                           facecolor=(0, 0, 0, 0),
                           edgecolor=(0, 0, 0, opacity),
                           **skwds)
            except ValueError:
                warnings.warn("Sampling %s does not work" % self.name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

    def plot_copula(self, theta=None, ax=None, kind="contourf"):
        if theta is None:
            try:
                theta = self.theta
            except AttributeError:
                theta = self.theta_start
        uu = vv = stats.rel_ranks(np.arange(1000))
        cc = self.copula_func(uu[None, :], vv[:, None], *theta)
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        if kind == "contourf":
            cb = ax.contourf(uu, vv, cc, 40)
        elif kind == "contour":
            cb = ax.contour(uu, vv, cc, 40)
        # plt.colorbar(cb)
        ax.set_title(sympy.printing.latex(self.cop_expr, mode="inline"))


class Frozen(object):

    def __init__(self, copula, *theta):
        """Copula with frozen parameters."""
        self.copula = copula
        self.theta = theta
        self.name = "frozen %s" % copula.name

    def __getattr__(self, name):
        return getattr(self.copula, name)

    def density(self, ranks_u, ranks_v):
        return self.copula.density(ranks_u, ranks_v, *self.theta)

    def sample(self, size):
        return self.copula.sample(size, *self.theta)

    def copula_func(self, uu, vv):
        return self.copula.copula_func(uu, vv, *self.theta)

    def cdf_given_u(self, uu, vv):
        return self.copula.cdf_given_u(uu, vv, *self.theta)

    def cdf_given_v(self, uu, vv):
        return self.copula.cdf_given_v(uu, vv, *self.theta)


@functools.total_ordering
class Fitted(object):

    def __init__(self, copula, ranks_u, ranks_v, *theta):
        self.copula = copula
        self.ranks_u = ranks_u
        self.ranks_v = ranks_v
        self.theta = theta
        self.name = "fitted %s" % copula.name
        density = copula.density(ranks_u, ranks_v, *theta)
        mask = (density > 0) & np.isfinite(density)
        dens_masked = density[mask]
        if len(dens_masked) == 0:
            self.likelihood = -np.inf
        else:
            self.likelihood = np.sum(np.log(dens_masked))

    def __getattr__(self, name):
        return getattr(self.copula, name)

    def __lt__(self, other):
        return self.likelihood < other

    def __repr__(self):
        return ("Fitted(%r, %r, %r, %r)" %
                (self.copula, self.ranks_u, self.ranks_v, self.theta))

    def cdf_given_u(self, uu, vv):
        return self.copula.cdf_given_u(uu, vv, *self.theta)

    def cdf_given_v(self, uu, vv):
        return self.copula.cdf_given_v(uu, vv, *self.theta)


class Archimedian(Copulae, metaclass=MetaArch):
    par_names = "uu", "vv", "theta"

    # def sample(self, size, theta):
    #     if hasattr(self, "sample_u"):
    #         print("sampling %s with KC" % self.name)
    #         ss, tt = np.random.rand(2 * size).reshape(2, -1)
    #         return self.sample_u(ss, tt), self.sample_v(ss, tt)
    #     else:
    #         return Copulae.sample(self, size, theta)


class Clayton(Archimedian):
    theta_start = 2,
    theta_bounds = [(1e-9, theta_large)]

    uu, vv, t, theta = sympy.symbols(("uu", "vv", "t", "theta"))
    # cop_expr = (uu ** (-theta) + vv ** (-theta) - 1) ** (-1 / theta)
    gen_expr = 1 / theta * (t ** (-theta) - 1)
    # gen_inv_expr = (1 + theta * t) ** (-1 / theta)
    kc_inv_expr = (theta + 1) ** (1 / theta)

    # def fit(self, uu, vv, *args, **kwds):
    #     tau = stats.kendalltau(uu, vv).correlation
    #     self.theta = 2 * tau / (1 - tau),
    #     return self.theta
clayton = Clayton()


class Frank(Archimedian):
    theta_start = 2.,
    theta_bounds = [(-theta_large, theta_large)]
    xx, uu, vv, t, theta = sympy.symbols(("xx", "uu", "vv", "t", "theta"))
    gen_expr = -ln((exp(-theta * t) - 1) /
                   (exp(-theta) - 1))
    gen_inv_expr = -1 / theta * ln(1 + exp(-xx) * (exp(-theta) - 1))
frank = Frank()


class GumbelBarnett(Archimedian):
    theta_start = .5,
    theta_bounds = [(0., 1.)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = ln(1 - theta * ln(t))
gumbelbarnett = GumbelBarnett()


class Nelsen02(Archimedian):
    theta_start = 1.5,
    theta_bounds = [(1 + 1e-9, theta_large)]
    uu, vv, t, theta = sympy.symbols(("uu", "vv", "t", "theta"))
    gen_expr = (1 - t) ** theta
    cop_expr = (1 - ((1 - uu) ** theta +
                     (1 - vv) ** theta) ** (1 / theta))
nelsen02 = Nelsen02()


class Nelsen06(Archimedian):
    theta_start = 1.5,
    theta_bounds = [(1., theta_large)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = -ln(1 - (1 - t) ** theta)
nelsen06 = Nelsen06()


# class Nelsen07(Archimedian):
#     theta_start = .5,
#     theta_bounds = [(0, 1)]
#     t, theta = sympy.symbols(("t", "theta"))
#     # this seems wrong! Why the parentheses around "1 - theta"?
#     # Should there be an exponent on that?
#     gen_expr = -ln(theta * t + (1 - theta))
# nelsen07 = Nelsen07()


class Nelsen08(Archimedian):
    theta_start = 1.5,
    theta_bounds = [(1., theta_large)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = (1 - t) / (1 + (theta - 1) * t)
nelsen08 = Nelsen08()


class Nelsen10(Archimedian):
    theta_start = .5,
    theta_bounds = [(0., 1.)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = ln(2 * t ** (-theta) - 1)
nelsen10 = Nelsen10()


class Nelsen11(Archimedian):
    theta_start = .4,
    theta_bounds = [(0, .5)]
    uu, vv, t, theta = sympy.symbols(("uu", "vv", "t", "theta"))
    gen_expr = ln(2 - t ** theta)
    cop_expr = (uu ** theta * vv ** theta -
                2 * (1 - uu ** theta) * (1 - vv ** theta)) ** (1 / theta)
nelsen11 = Nelsen11()


class Nelsen12(Archimedian):
    theta_start = 1.5,
    theta_bounds = [(1., theta_large)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = (1 / t - 1) ** theta
nelsen12 = Nelsen12()


class Nelsen13(Archimedian):
    theta_start = .5,
    theta_bounds = [(1e-12, theta_large)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = (1 - ln(t)) ** theta - 1
nelsen13 = Nelsen13()


class Nelsen14(Archimedian):
    theta_start = 3.,
    theta_bounds = [(1., theta_large)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = (t ** (-1 / theta) - 1) ** theta
nelsen14 = Nelsen14()


class Nelsen15(Archimedian):
    # TODO: this probably has a real name, look it up!
    theta_start = 3.,
    theta_bounds = [(1., theta_large)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = (1 - t ** (1 / theta)) ** theta
nelsen15 = Nelsen15()


class Nelsen16(Archimedian):
    theta_start = .5,
    theta_bounds = [(0, theta_large)]
    uu, vv, S, t, theta = sympy.symbols(("uu", "vv", "S", "t", "theta"))
    gen_expr = (theta / t + 1) * (1 - t)
    cop_expr = .5 * (S * sympy.sqrt(S ** 2 + 4 * theta))
    cop_expr = cop_expr.subs(S, uu + vv - 1 - theta * (1 / uu + 1 / vv - 1))
nelsen16 = Nelsen16()


# very slow class construction
class Nelsen17(Archimedian):
    theta_start = 1,
    theta_bounds = [(-theta_large, theta_large)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = -ln(((1 + t) ** (-theta) - 1) / (2 ** (-theta) - 1))
nelsen17 = Nelsen17()


class Nelsen18(Archimedian):
    theta_start = 100,
    theta_bounds = [(2., theta_large)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = exp(theta / (t - 1))
nelsen18 = Nelsen18()


class Nelsen19(Archimedian):
    theta_start = 10,
    theta_bounds = [(0., theta_large)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = exp(theta / t) - exp(theta)
nelsen19 = Nelsen19()


class Nelsen20(Archimedian):
    theta_start = .1,
    theta_bounds = [(1e-12, theta_large)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = exp(t ** (-theta)) - mp.e
nelsen20 = Nelsen20()


class Nelsen21(Archimedian):
    theta_start = 1.5,
    theta_bounds = [(1., theta_large)]
    uu, vv, t, theta = sympy.symbols(("uu", "vv", "t", "theta"))
    gen_expr = 1 - (1 - (1 - t) ** theta) ** (1 / theta)
    # cop_expr = (1 - (1 - ((1 - (1 - uu) ** theta) ** (1 / theta) +
    #                       (1 - (1 - vv) ** theta) ** (1 / theta) -
    #                       1) ** theta)) ** (1 / theta)
    # cop_expr = (1 - (1 - sympy.Max((1 - (1 - uu) ** theta) ** (1 / theta) +
    #                                (1 - (1 - vv) ** theta) ** (1 / theta) -
    #                                1, 0) ** theta)) ** (1 / theta)
nelsen21 = Nelsen21()


# class Nelsen22(Archimedian):
#     theta_start = .5,
#     theta_bounds = [(0, 1)]
#     t, theta = sympy.symbols(("t", "theta"))
#     gen_expr = asin(1 - t ** theta)
# nelsen22 = Nelsen22()


class Joe(Archimedian):
    theta_start = 10,
    theta_bounds = [(1 + 1e-9, theta_large)]
    xx, uu, vv, t, theta = sympy.symbols(("xx", "uu", "vv", "t", "theta"))
    gen_expr = -sympy.ln(1 - (1 - t) ** theta)
    # gen_inv_expr = 1 - (1 - sympy.exp(-xx)) ** (1 / theta)
joe = Joe()


# class Joe180(Archimedian):
#     theta_start = 2,
#     theta_bounds = [(1 + 1e-9, theta_large)]
#     uu, vv, theta = sympy.symbols(("uu", "vv", "theta"))
#     cop_expr = (1 - (uu ** theta + vv ** theta -
#                      uu ** theta * vv ** theta) ** (1 / theta))
# joe180 = Joe180()


class Gumbel(Archimedian):
    theta_start = 5.,
    # not sure about that
    # theta_bounds = [(1e-9, 1. - 1e-9)]
    theta_bounds = [(1. + 1e-9, theta_large)]
    xx, uu, vv, t, theta = sympy.symbols(("xx", "uu", "vv", "t", "theta"))
    gen_expr = (-ln(t)) ** theta
    gen_inv_expr = exp(-xx ** (1 / theta))

    # def fit(self, uu, vv, *args, **kwds):
    #     tau = stats.kendalltau(uu, vv).correlation
    #     self.theta = 1. / (1 - tau),
    #     return self.theta
gumbel = Gumbel()


class AliMikailHaqPos(Archimedian):
    theta_start = .1,
    theta_bounds = [(1e-9, 1.)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = ln((1 - theta * (1 - t)) / t)
alimikailhaqpos = AliMikailHaqPos()


class AliMikailHaqNeg(AliMikailHaqPos):
    # splitting AliMikailHaq into its positive and negative dependence
    # domain hopefully helps the optimizer
    theta_start = -.1
    theta_bounds = [(-1., -1e-9)]
alimikailhaqneg = AliMikailHaqNeg()


class Independence(Copulae):
    par_names = "uu", "vv"
    theta_start = None,
    uu, vv = sympy.symbols(par_names)
    cop_expr = uu * vv

    def fit(self, uu, vv, *args, **kwds):
        return None

    def sample(self, size, *args, **kwds):
        return random_sample(size), random_sample(size)
independence = Independence()


all_cops = OrderedDict((name, obj) for name, obj
                       in sorted(dict(locals()).items())
                       if isinstance(obj, Copulae))

if __name__ == '__main__':
    import doctest
    doctest.testmod()

    from weathercop import cop_conf
    from weathercop import plotting as cplt

    data_filepath = os.path.join(cop_conf.weathercop_dir, "code",
                                 "vg_data.npz")
    with np.load(data_filepath) as saved:
        data_summer = saved["summer"]
    ranks_u_tm1 = stats.rel_ranks(data_summer[5, :-1])
    ranks_rh = stats.rel_ranks(data_summer[4, 1:])

    for copula in all_cops.values():
        # copula.plot_cop_dens()
        # copula.plot_copula()
        # copula.plot_density()
        # plt.title(copula.name)
        try:
            fitted_cop = copula.generate_fitted(ranks_u_tm1, ranks_rh,
                                                # method="TNC",
                                                verbose=False)
        except NoConvergence:
            continue
        fig, axs = plt.subplots(ncols=2, subplot_kw=dict(aspect="equal"))
        fig.suptitle(copula.name + " " +
                     repr(fitted_cop.theta) +
                     "\n likelihood: %.2f" % fitted_cop.likelihood)
        opacity = .1
        fitted_cop.plot_density(ax=axs[0], opacity=opacity,
                                scatter=True, sample_size=10000,
                                kind="contour")
        cplt.hist2d(ranks_u_tm1, ranks_rh, ax=axs[1],
                    # kind="contourf",
                    scatter=False)
        axs[1].scatter(ranks_u_tm1, ranks_rh,
                       marker="o", facecolors=(0, 0, 0, 0),
                       edgecolors=(0, 0, 0, opacity))
    plt.show()
