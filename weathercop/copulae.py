"""Bivariate Copulas intended for Vine Copulas."""
import functools
import importlib
import warnings
import os
from abc import ABCMeta, abstractproperty
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import sympy
from mpmath import mp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.special import erf, erfinv
from scipy.stats import mvn
from scipy import stats as spstats
from sympy import exp, ln
from sympy.utilities import autowrap

from weathercop import cop_conf as conf, stats, tools


# used wherever theta can be inf in principle
theta_large = 1e3

# here expressions are kept that sympy has problems with
faillog_file = conf.ufunc_tmp_dir / "known_fail"
conf.ufunc_tmp_dir.mkdir(parents=True, exist_ok=True)

# allow printing of more complex equations
# from matplotlib import rc
# rc("text", usetex=True)


def ufuncify(cls, name, uargs, expr, *args, verbose=False, **kwds):
    expr_hash = tools.hash_cop(expr)
    module_name = "%s_%s_%s" % (cls.name, name, expr_hash)
    try:
        with tools.chdir(conf.ufunc_tmp_dir):
            ufunc = importlib.import_module("%s_0" % module_name).autofunc_c
    except ImportError:
        if verbose:
            print("Compiling %s" % repr(expr))
        _filename_orig = autowrap.CodeWrapper._filename
        _module_basename_orig = autowrap.CodeWrapper._module_basename
        _module_counter_orig = autowrap.CodeWrapper._module_counter
        # these attributes determine the file name of the generated code
        autowrap.CodeWrapper._filename = "%s_code" % module_name
        autowrap.CodeWrapper._module_basename = module_name
        autowrap.CodeWrapper._module_counter = 0
        ufunc = autowrap.ufuncify(uargs, expr,
                                  tempdir=conf.ufunc_tmp_dir,
                                  verbose=verbose,
                                  *args, **kwds)
        autowrap.CodeWrapper._module_basename = _module_basename_orig
        autowrap.CodeWrapper._module_counter = _module_counter_orig
        autowrap.CodeWrapper._filename = _filename_orig
    return ufunc


def mark_failed(key):
    mode = "r+" if faillog_file.exists() else "w+"
    with faillog_file.open(mode) as faillog:
        keys = faillog.readlines()
        if (key + "\n") not in keys:
            faillog.write(key + os.linesep)


def swap_symbols(expr, symbol1, symbol2):
    """Substitute symbol1 and symbol2 in the given sympy expression.

    >>> import sympy
    >>> x, y = sympy.symbols("x y")
    >>> swap_symbols(x - y, x, y)
    -x + y
    """
    try:
        # in case symbols are strings
        symbol1, symbol2 = sympy.symbols((symbol1, symbol2))
    except TypeError:
        # assume symbols are already sympy symbols
        pass
    xxx = sympy.symbols("xxx")
    expr = expr.subs({symbol1: xxx})
    expr = expr.subs({symbol2: symbol1})
    expr = expr.subs({xxx: symbol2})
    return expr


def random_sample(size, bound=1e-12):
    """Sample in the closed interval (0 + bound, 1 - bound)."""
    return (1 - 2 * bound) * np.random.random_sample(size) + bound


def positive(func):
    @functools.wraps(func)
    def inner(*args, **kwds):
        if isinstance(args[0], Copulae):
            args = args[1:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(*args, **kwds)
            result = np.squeeze(result)
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


def bipit(ranks_u, ranks_v):
    """Bivariabe probability integral transform."""
    ranks_u, ranks_v = map(np.squeeze, (ranks_u, ranks_v))
    n = len(ranks_u)
    ranks = np.empty(n)
    for i, (rank_u, rank_v) in enumerate(zip(ranks_u, ranks_v)):
        ranks[i] = np.sum((ranks_u < rank_u) & (ranks_v < rank_v))
    return (ranks + .5) / n


class NoConvergence(Exception):
    pass


class MetaCop(ABCMeta):
    backend = "cython"

    def __new__(cls, name, bases, cls_dict):
        new_cls = super().__new__(cls, name, bases, cls_dict)
        new_cls.name = name.lower()
        if "cop_expr" in cls_dict:
            # auto-rotate the copula expression
            if name.endswith(("90", "180", "270")):
                new_cls = MetaCop.rotate_expr(new_cls)

            if "known_fail" in cls_dict:
                MetaCop.mark_failed(new_cls)
            new_cls.dens_func = MetaCop.density_from_cop(new_cls)
            new_cls.cdf_given_u = MetaCop.cdf_given_u(new_cls)
            new_cls.cdf_given_v = MetaCop.cdf_given_v(new_cls)
            new_cls.copula_func = MetaCop.copula_func(new_cls)
            ufunc = MetaCop.inv_cdf_given_u(new_cls)
            if ufunc is not None:
                new_cls.inv_cdf_given_u = ufunc
            ufunc = MetaCop.inv_cdf_given_v(new_cls)
            if ufunc is not None:
                new_cls.inv_cdf_given_v = ufunc
        elif "dens_expr" in cls_dict and "dens_func" not in cls_dict:
            new_cls.dens_func = MetaCop.density_func(new_cls)
        return new_cls

    def mark_failed(new_cls):
        for method_name in new_cls.known_fail:
            key = "_".join((new_cls.name,
                            method_name,
                            tools.hash_cop(new_cls)))
            mark_failed(key)

    def rotate_expr(cls):
        """Rotate copula expression.

        Notes
        -----
        see p. 271
        """
        uu, vv = sympy.symbols(("uu vv"))
        if cls.name.endswith("90"):
            cls.cop_expr = uu - cls.cop_expr.subs({vv: 1 - vv})
        elif cls.name.endswith("180"):
            cls.cop_expr = uu + vv - 1 + cls.cop_expr.subs({uu: 1 - uu,
                                                            vv: 1 - vv})
        elif cls.name.endswith("270"):
            cls.cop_expr = vv - cls.cop_expr.subs({uu: 1 - uu})
        return cls

    def copula_func(cls):
        uu, vv, *theta = sympy.symbols(cls.par_names)
        ufunc = ufuncify(cls, "copula",
                         [uu, vv] + theta, cls.cop_expr,
                         backend=MetaCop.backend,
                         verbose=False)
        if MetaCop.backend in ("f2py", "cython"):
            ufunc = broadcast_2d(ufunc)
        return positive(ufunc)

    def density_func(cls):
        uu, vv, *theta = sympy.symbols(cls.par_names)
        dens_expr = cls.dens_expr
        ufunc = ufuncify(cls, "density",
                         [uu, vv] + theta, dens_expr,
                         backend=MetaCop.backend,
                         verbose=False)
        if MetaCop.backend in ("f2py", "cython"):
            ufunc = broadcast_2d(ufunc)
        return positive(ufunc)

    def conditional_cdf(cls, conditioning):
        uu, vv, *theta = sympy.symbols(cls.par_names)
        expr_attr = "cdf_given_%s_expr" % conditioning
        with tools.shelve_open(conf.sympy_cache) as sh:
            key = ("%s_cdf_given_%s_%s" %
                   (cls.name, conditioning, tools.hash_cop(cls)))
            if key not in sh:
                print("Generating %s-conditional %s" %
                      (conditioning, cls.name))
                # a good cop always stays positive!
                good_cop = sympy.Piecewise((cls.cop_expr,
                                            cls.cop_expr > 0),
                                           (0, True))
                # good_cop = cls.cop_expr
                conditional_cdf = sympy.diff(good_cop, conditioning)
                conditional_cdf = sympy.simplify(conditional_cdf)
                sh[key] = conditional_cdf
            conditional_cdf = sh[key]
            setattr(cls, expr_attr, conditional_cdf)
        ufunc = ufuncify(cls, "conditional_cdf",
                         [uu, vv] + theta, conditional_cdf,
                         backend=MetaCop.backend,
                         verbose=False)
        if MetaCop.backend in ("f2py", "cython"):
            ufunc = broadcast_2d(ufunc)
        return positive(ufunc)

    def cdf_given_u(cls):
        return cls.conditional_cdf(sympy.symbols("uu"))

    def cdf_given_v(cls):
        # if we are given an expression for the u-conditional, we can
        # substitute v for u to get the corresponding v-conditional
        try:
            cdf = getattr(cls, "cdf_given_uu_expr")
            cdf = swap_symbols(cdf, "uu", "vv")
            setattr(cls, "cdf_given_vv_expr", cdf)
        except AttributeError:
            pass
        return cls.conditional_cdf(sympy.symbols("vv"))

    def inverse_conditional_cdf(cls, conditioning):
        uu, vv, qq, theta = sympy.symbols(("uu", "vv", "qq", "theta"))
        conditioned = list(set((uu, vv)) - set([conditioning]))[0]
        key = ("%s_inv_cdf_given_%s_%s" %
               (cls.name, conditioning, tools.hash_cop(cls)))
        # keep a log of what does not work in order to not repeat ad
        # nauseum
        try:
            if (key + os.linesep) in open(faillog_file):
                return
        except FileNotFoundError:
            open(faillog_file, "a").close()
        attr_name = "inv_cdf_given_%s_expr" % conditioning
        # cached sympy derivation
        if not hasattr(cls, attr_name):
            with tools.shelve_open(conf.sympy_cache) as sh:
                if key not in sh:
                    print("Generating inverse %s-conditional %s" %
                          (conditioning, cls.name))
                    cdf_given_expr = getattr(cls,
                                             "cdf_given_%s_expr" %
                                             conditioning)
                    try:
                        inv_cdf = sympy.solve(cdf_given_expr - qq,
                                              conditioned)[0]
                    except NotImplementedError:
                        warnings.warn("Derivation of inv.-conditional " +
                                      "failed for" +
                                      " %s" % cls.name)
                        with open(faillog_file, "r+") as faillog:
                            keys = faillog.readlines()
                            if key not in keys:
                                faillog.write(key + os.linesep)
                        return
                    sh[key] = inv_cdf
                inv_cdf = sh[key]
                setattr(cls, attr_name, inv_cdf)
        inv_cdf = getattr(cls, attr_name)
        # compile sympy expression
        try:
            ufunc = ufuncify(cls, "inv_cdf_given_%s" % conditioning,
                             [qq, conditioning, theta], inv_cdf,
                             backend=MetaCop.backend,
                             verbose=False)
        except autowrap.CodeWrapError:
            warnings.warn("Could not compile inv.-conditional for %s" %
                          cls.name)
            mark_failed(key)
            return
        if MetaCop.backend in ("f2py", "cython"):
            ufunc = broadcast_2d(ufunc)
        return positive(ufunc)

    def inv_cdf_given_u(cls):
        return cls.inverse_conditional_cdf(sympy.symbols("uu"))

    def inv_cdf_given_v(cls):
        if not hasattr(cls, "inv_cdf_given_vv_expr"):
            # if we are given an expression for the u-conditional, we can
            # substitute v for u to get the corresponding v-conditional
            try:
                inv_cdf = getattr(cls, "inv_cdf_given_uu_expr")
                uu, vv = sympy.symbols("uu vv")
                inv_cdf = inv_cdf.subs({uu: vv})
                setattr(cls, "inv_cdf_given_vv_expr", inv_cdf)
            except AttributeError:
                pass
        return cls.inverse_conditional_cdf(sympy.symbols("vv"))

    def density_from_cop(cls):
        """Copula density obtained by sympy differentiation compiled with
        cython.
        """
        uu, vv, *theta = sympy.symbols(cls.par_names)
        with tools.shelve_open(conf.sympy_cache) as sh:
            key = "%s_density_%s" % (cls.name, tools.hash_cop(cls))
            if key not in sh:
                print("Generating density for %s" % cls.name)
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
                         verbose=False)
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
                    print("Generating inv. gen for %s" % name)
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

    theta_bounds = [(-np.inf, np.inf)]
    # zero, one = 1e-6, 1 - 1e-6
    zero, one = 1e-15, 1 - 1e-15

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

    def _inverse_conditional(self, conditional_func, ranks, quantiles, theta):
        """Numeric inversion of conditional_func (inv_cdf_given_u or
        inv_cdf_given_v), to be used as a last resort.

        """
        theta = np.array(self.theta if theta is None else theta)
        ranks1 = np.atleast_1d(ranks)
        ranks1, quantiles, thetas = map(np.atleast_1d, (ranks,
                                                        quantiles,
                                                        theta))
        quantiles, thetas = map(np.squeeze, (quantiles, thetas))
        if thetas.size == 1:
            thetas = np.full_like(ranks1, theta)
        if quantiles.size == 1:
            quantiles = np.full_like(ranks1, quantiles)

        ranks2 = np.empty_like(ranks1)
        ranks2_calc = np.linspace(self.zero, self.one, 500)
        ranks2_calc = np.concatenate(([0], ranks2_calc, [1]))
        for i, rank1 in enumerate(ranks1):
            quantile, theta = quantiles[i], thetas[i]
            quantiles_calc = conditional_func(rank1,
                                              ranks2_calc[1:-1],
                                              theta)
            quantiles_calc = np.squeeze(quantiles_calc)
            quantiles_calc = np.concatenate(([0], quantiles_calc, [1]))
            # if there is more than one zero involved, interpolating
            # between them gives unwanted results.
            # valid_mask = ((quantiles_calc != 0) & (quantiles_calc != 1))
            valid_mask = np.ones_like(quantiles_calc, dtype=bool)
            f_int = interp1d(quantiles_calc[valid_mask],
                             ranks2_calc[valid_mask],
                             # bounds_error=False,
                             # fill_value=(0, 1)
                             )
            ranks2[i] = f_int(quantile)
        return ranks2

    def inv_cdf_given_u(self, ranks_u, quantiles, theta=None):
        """Numeric inversion of cdf_given_u, to be used as a last resort.
        """
        return self._inverse_conditional(self.cdf_given_u, ranks_u,
                                         quantiles, theta=theta)

    def inv_cdf_given_v(self, ranks_v, quantiles, theta=None):
        """Numeric inversion of cdf_given_v, to be used as a last resort.
        """
        return self._inverse_conditional(self.cdf_given_v, ranks_v,
                                         quantiles, theta=theta)

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

    def fit_ml(self, ranks_u, ranks_v, method=None, x0=None, verbose=False):
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
                # if np.any(mask):
                #     return -1e12
                dens[mask] = 1e-9
                loglike = -np.sum(np.log(dens))
            return loglike

        if x0 is None:
            x0 = self.theta_start
        result = minimize(neg_log_likelihood, x0,
                          bounds=self.theta_bounds,
                          method=method,
                          # options=(dict(disp=True) if verbose else None)
                          )
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
                     s_kwds=None, c_kwds=None):
        if theta is None:
            try:
                theta = self.theta
            except AttributeError:
                theta = self.theta_start
        if s_kwds is None:
            s_kwds = dict()
        if c_kwds is None:
            c_kwds = dict(alpha=.5,
                          linewidth=.25)
        uu = vv = stats.rel_ranks(np.arange(100))
        density = self.density(uu[None, :], vv[:, None], *theta)
        if not isinstance(self, Independence):
            # get rid of large values for visualizations sake
            density[density >= np.sort(density.ravel())[-10]] = np.nan
            if ax is None:
                fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
            if kind == "contourf":
                ax.contourf(uu, vv, density, 40, **c_kwds)
            elif kind == "contour":
                ax.contour(uu, vv, density, 40, **c_kwds)
        if scatter:
            try:
                u_sample, v_sample = self.sample(size=1000, theta=theta)
                ax.scatter(u_sample, v_sample,
                           marker="o",
                           facecolor=(0, 0, 0, 0),
                           edgecolor=(0, 0, 0, opacity),
                           **s_kwds)
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
        uu = vv = stats.rel_ranks(np.arange(100))
        cc = self.copula_func(uu[None, :], vv[:, None], *theta)
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        if kind == "contourf":
            ax.contourf(uu, vv, cc, 40)
        elif kind == "contour":
            ax.contour(uu, vv, cc, 40)
        ax.set_title(sympy.printing.latex(self.cop_expr, mode="inline"))
        # ax.set_title(r"%s" % sympy.printing.latex(self.cop_expr))


class Frozen:

    def __init__(self, copula, *theta):
        """Copula with frozen parameters."""
        self.copula = copula
        self.theta = theta
        self.name = "frozen %s" % copula.name

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            attr = getattr(self.copula, name)
            if callable(attr):
                return lambda *args: attr(*(args + self.theta))
            return attr


@functools.total_ordering
class Fitted:

    def __init__(self, copula, ranks_u, ranks_v, *theta,
                 verbose=False):
        self.copula = copula
        self.ranks_u = ranks_u
        self.ranks_v = ranks_v
        self.theta = theta
        self.name = "fitted %s" % copula.name
        self.verbose = self.copula.verbose = verbose
        if isinstance(copula, Independence):
            self.likelihood = 0
        else:
            density = copula.density(ranks_u, ranks_v, *theta)
            # if np.any(~np.isfinite(density)):
            #     self.likelihood = -np.inf
            # else:
            #     self.likelihood = np.sum(np.log(density))
            mask = (density > 0) & np.isfinite(density)
            density[~mask] = 1e-60
            dens_masked = density[mask]
            if len(dens_masked) == 0:
                self.likelihood = -np.inf
            else:
                self.likelihood = np.sum(np.log(dens_masked))

            if verbose:
                print(4 * " ",
                      self.name[len("fitted "):],
                      self.likelihood, end="")
                # n_nonfin = np.sum(~mask)
                # if n_nonfin:
                #     print(" # nonfinite ",
                #           np.sum(~mask),
                #           )
                # else:
                #     print()

    def plot_qq(self, ax=None, opacity=.25, s_kwds=None, title=None,
                *args, **kwds):
        if s_kwds is None:
            s_kwds = dict(marker="o", s=1,
                          facecolors=(0, 0, 0, 0),
                          edgecolors=(0, 0, 0, opacity))

        empirical = bipit(self.ranks_u, self.ranks_v)
        theoretical = self.copula.copula_func(self.ranks_u,
                                              self.ranks_v,
                                              *self.theta)
        if ax is None:
            fig, ax = plt.subplots(
                # subplot_kw=dict(aspect="equal")
            )
        if fig is None:
            fig = plt.gcf()
        ax.scatter(empirical, theoretical - empirical, **s_kwds)
        if title is None:
            title = "qq " + self.name[len("fitted "):]
        ax.set_title(title)
        fig.tight_layout()
        return fig, ax

    def __getstate__(self):
        # this could be a rotated copula, for which the class is
        # generated dynamically. even dill has problems with pickling
        # these crazy instances!
        dict_ = dict(self.__dict__)
        dict_["copula"] = self.copula.name
        return dict_

    def __setstate__(self, dict_):
        dict_["copula"] = globals()[dict_["copula"]]
        self.__dict__ = dict_

    def __getattr__(self, name):
        return getattr(self.copula, name)

    def __lt__(self, other):
        return self.likelihood < other

    def __repr__(self):
        return ("Fitted(%r, theta=%r)" % (self.copula, self.theta))

    def cdf_given_u(self, *, conditioned, condition):
        return self.copula.cdf_given_u(condition, conditioned, *self.theta)

    def cdf_given_v(self, *, conditioned, condition):
        return self.copula.cdf_given_v(conditioned, condition, *self.theta)

    def inv_cdf_given_u(self, *, conditioned, condition):
        return self.copula.inv_cdf_given_u(condition, conditioned, *self.theta)

    def inv_cdf_given_v(self, *, conditioned, condition):
        return self.copula.inv_cdf_given_v(conditioned, condition, *self.theta)

    def plot_density(self, *args, **kwds):
        return self.copula.plot_density(theta=self.theta, *args, **kwds)


class NoRotations:
    """Use this as a base if Copula should not be rotated."""
    pass


class No90:
    """Use this as a base if Copula should not be rotated by 90
    degrees."""
    pass


class No180:
    """Use this as a base if Copula should not be rotated by 180
    degrees."""
    pass


class No270:
    """Use this as a base if Copula should not be rotated by 270
    degrees."""
    pass


class Archimedian(Copulae, metaclass=MetaArch):
    """Archimedian Copulas.

    Notes
    -----
    Nelsen-style archimedian copulas defined by a generator
    function. Not a LT as in Joe.
    """
    par_names = "uu", "vv", "theta"

    # def sample(self, size, theta):
    #     if hasattr(self, "sample_u"):
    #         print("sampling %s with KC" % self.name)
    #         ss, tt = np.random.rand(2 * size).reshape(2, -1)
    #         return self.sample_u(ss, tt), self.sample_v(ss, tt)
    #     else:
    #         return Copulae.sample(self, size, theta)


class Clayton(Copulae, No90, No270):
    par_names = "uu", "vv", "delta"
    theta_start = 2,
    theta_bounds = [(1e-8, 10.)]
    uu, vv, t, delta, qq = sympy.symbols("uu vv t delta qq")
    cop_expr = (uu ** -delta + vv ** -delta - 1) ** (-1 / delta)
    dens_expr = ((1 + delta) * (uu * vv) ** (-delta - 1) *
                 (uu ** -delta + vv ** -delta - 1) ** (-2 - 1 / delta))
    # gen_expr = 1 / theta * (t ** (-theta) - 1)
    # gen_inv_expr = (1 + theta * t) ** (-1 / theta)
    # cdf_given_uu_expr = ((1 + uu ** delta *
    #                       (vv ** -delta - 1)) ** (-1 - 1 / delta))
    # inv_cdf_given_uu_expr = ((qq ** (-delta / (1 + delta)) - 1) *
    #                          uu ** -delta + 1) ** (-1 / delta)
clayton = Clayton()


class FrankPos(Archimedian):
    theta_start = 2.,
    theta_bounds = [(1e-5, 35)]
    xx, uu, vv, t, theta = sympy.symbols("xx uu vv t theta")
    # gen_expr = -ln((exp(-theta * t) - 1) /
    #                (exp(-theta) - 1))
    # gen_inv_expr = -1 / theta * ln(1 + exp(-xx) * (exp(-theta) - 1))
    cop_expr = (-theta ** -1 * ln((1 - exp(-theta) -
                                   (1 - exp(-theta * uu)) *
                                   (1 - exp(-theta * vv))) /
                                  (1 - exp(-theta))))
    # cdf_given_uu_expr = (exp(-theta * uu) * ((1 - exp(-theta)) *
    #                                          (1 - exp(-theta * vv)) ** -1 -
    #                                          (1 - exp(-theta * uu))) ** -1)
    # qq = sympy.symbols("qq")
    # inv_cdf_given_uu_expr = (-theta ** -1 * ln(1 - (1 - exp(-theta)) /
    #                                            ((qq ** -1 - 1) *
    #                                             exp(-theta * uu) + 1)))
# frankpos = FrankPos()


class FrankNeg(FrankPos):
    theta_start = -2.,
    theta_bounds = [(-35, -1e-5)]
# frankneg = FrankNeg()


class GumbelBarnett(Archimedian):
    """Also Nelsen09"""
    theta_start = .5,
    theta_bounds = [(1e-9, 1. - 1e-9)]
    t, theta = sympy.symbols(("t theta"))
    gen_expr = ln(1 - theta * ln(t))
gumbelbarnett = GumbelBarnett()


# inv_cdf_given_u fails test
# class Nelsen02(Archimedian):
#     theta_start = 1.1,
#     theta_bounds = [(1 + 1e-9, 80)]
#     uu, vv, t, theta = sympy.symbols(("uu", "vv", "t", "theta"))
#     gen_expr = (1 - t) ** theta
#     cop_expr = (1 - ((1 - uu) ** theta +
#                      (1 - vv) ** theta) ** (1 / theta))
#     cop_expr = sympy.Piecewise((cop_expr, cop_expr > 0),
#                                (0, True))
# nelsen02 = Nelsen02()


# this is the joe copula (see below)
# class Nelsen06(Archimedian):
#     theta_start = 1.5,
#     theta_bounds = [(1., theta_large)]
#     t, theta = sympy.symbols(("t", "theta"))
#     gen_expr = -ln(1 - (1 - t) ** theta)
# nelsen06 = Nelsen06()


# cdf_given_u fails test
# class Nelsen07(Archimedian, No180):
#     theta_start = .5,
#     theta_bounds = [(0, 1)]
#     t, theta = sympy.symbols(("t", "theta"))
#     # this seems wrong! Why the parentheses around "1 - theta"?
#     # Should there be an exponent on that?
#     gen_expr = -ln(theta * t + (1 - theta))
# nelsen07 = Nelsen07()


# # cdf_given_u fails test
# class Nelsen08(Archimedian, NoRotations):
#     theta_start = 1.5,
#     theta_bounds = [(1. + 1e-3, theta_large)]
#     t, theta = sympy.symbols(("t", "theta"))
#     gen_expr = (1 - t) / (1 + (theta - 1) * t)
#     uu, vv = sympy.symbols("uu vv")
#     # cop_expr = ((theta ** 2 * uu * vv - (1 - uu) * (1 - vv)) /
#     #             (theta ** 2 - (theta - 1) ** 2 * (1 - uu) * (1 - vv)))
#     # cop_expr = sympy.Piecewise((cop_expr, cop_expr > 0),
#     #                            (0, True))
# nelsen08 = Nelsen08()


class Nelsen10(Archimedian):
    theta_start = .5,
    theta_bounds = [(1e-3, 1.)]
    t, theta = sympy.symbols(("t theta"))
    gen_expr = ln(2 * t ** (-theta) - 1)
nelsen10 = Nelsen10()


# cdf_given_u fails test
# class Nelsen11(Archimedian):
#     theta_start = .4,
#     theta_bounds = [(1e-6, .5)]
#     uu, vv, t, theta = sympy.symbols(("uu", "vv", "t", "theta"))
#     gen_expr = ln(2 - t ** theta)
#     cop_expr = (uu ** theta * vv ** theta -
#                 2 * (1 - uu ** theta) * (1 - vv ** theta)) ** (1 / theta)
#     cop_expr = sympy.Piecewise((cop_expr, cop_expr > 0),
#                                (0, True))
# nelsen11 = Nelsen11()


class Nelsen12(Archimedian, NoRotations):
    theta_start = 1.5,
    theta_bounds = [(1., 70)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = (1 / t - 1) ** theta
nelsen12 = Nelsen12()


class Nelsen13(Archimedian, NoRotations):
    theta_start = .5,
    theta_bounds = [(1e-12, 300)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = (1 - ln(t)) ** theta - 1
nelsen13 = Nelsen13()


class Nelsen14(Archimedian, No90, No270):
    theta_start = 3.,
    theta_bounds = [(1., 60.)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = (t ** (-1 / theta) - 1) ** theta
nelsen14 = Nelsen14()


# inv_cdf_given_u fails test
# class Nelsen15(Archimedian, NoRotations):
#     # TODO: this probably has a real name, look it up!
#     theta_start = 3.,
#     theta_bounds = [(1., 45)]
#     t, theta = sympy.symbols(("t", "theta"))
#     gen_expr = (1 - t ** (1 / theta)) ** theta
# nelsen15 = Nelsen15()


# cdf_given_u fails test
# class Nelsen16(Archimedian):
#     theta_start = .5,
#     theta_bounds = [(1e-3, theta_large)]
#     uu, vv, S, t, theta = sympy.symbols(("uu vv S t theta"))
#     gen_expr = (theta / t + 1) * (1 - t)
#     cop_expr = .5 * (S + sympy.sqrt(S ** 2 + 4 * theta))
#     cop_expr = cop_expr.subs(S, uu + vv - 1 - theta * (1 / uu + 1 / vv - 1))
#     known_fail = "inv_cdf_given_uu", "inv_cdf_given_vv"
# nelsen16 = Nelsen16()


# # very slow class construction
# class Nelsen17(Archimedian):
#     theta_start = 1,
#     theta_bounds = [(-theta_large, theta_large)]
#     t, theta = sympy.symbols(("t", "theta"))
#     gen_expr = -ln(((1 + t) ** (-theta) - 1) / (2 ** (-theta) - 1))
# nelsen17 = Nelsen17()


# inv_cdf_given_u fails test
# class Nelsen18(Archimedian, No270):
#     theta_start = 2.5,
#     theta_bounds = [(2., 10)]
#     uu, vv, t, theta = sympy.symbols(("uu", "vv", "t", "theta"))
#     gen_expr = exp(theta / (t - 1))
#     cop_expr = 1 + theta / ln(exp(theta / (uu - 1)) +
#                               exp(theta / (vv - 1)))
#     cop_expr = sympy.Piecewise((cop_expr, cop_expr > 0),
#                                (0, True))
# nelsen18 = Nelsen18()


# cdf_given_u fails test
# class Nelsen19(Archimedian):
#     theta_start = .1,
#     theta_bounds = [(0., .375)]
#     t, theta = sympy.symbols(("t", "theta"))
#     gen_expr = exp(theta / t) - exp(theta)
# nelsen19 = Nelsen19()


class Nelsen20(Archimedian, No90, No270):
    theta_start = .05,
    theta_bounds = [(1e-9, .9)]
    uu, vv, t, theta = sympy.symbols(("uu", "vv", "t", "theta"))
    gen_expr = exp(t ** (-theta)) - mp.e
    cop_expr = (ln(exp(uu ** -theta) +
                   exp(vv ** -theta) - mp.e)) ** (-1 / theta)
nelsen20 = Nelsen20()


# inv_cdf_given_u fails test
# class Nelsen21(Archimedian):
#     theta_start = 1.1,
#     theta_bounds = [(1., 5)]
#     uu, vv, t, theta = sympy.symbols(("uu", "vv", "t", "theta"))
#     gen_expr = 1 - (1 - (1 - t) ** theta) ** (1 / theta)
#     # cop_expr = (1 - (1 - ((1 - (1 - uu) ** theta) ** (1 / theta) +
#     #                       (1 - (1 - vv) ** theta) ** (1 / theta) -
#     #                       1) ** theta)) ** (1 / theta)
#     # piece = ((1 - (1 - uu) ** theta) ** (1 / theta) +
#     #          (1 - (1 - vv) ** theta) ** (1 / theta) - 1)
#     # piece = sympy.Piecewise((piece, piece > 0),
#     #                         (0, True))
#     # cop_expr = (1 - (1 - piece ** theta)) ** (1 / theta)
# nelsen21 = Nelsen21()


# class Nelsen22(Archimedian):
#     theta_start = .5,
#     theta_bounds = [(0, 1)]
#     t, theta = sympy.symbols(("t", "theta"))
#     gen_expr = asin(1 - t ** theta)
# nelsen22 = Nelsen22()


class Joe(Copulae, No90, No270):
    par_names = "uu vv theta".split()
    theta_start = 5.,
    theta_bounds = [(1 + 1e-9, 30)]
    xx, uu, vv, t, theta = sympy.symbols(("xx", "uu", "vv", "t", "theta"))
    # gen_expr = -ln(1 - (1 - t) ** theta)
    # gen_inv_expr = 1 - (1 - sympy.exp(-xx)) ** (1 / theta)
    cop_expr = (1 - ((1 - uu) ** theta +
                     (1 - vv) ** theta -
                     (1 - uu) ** theta *
                     (1 - vv) ** theta) ** (1 / theta))
    cdf_given_uu_expr = ((1 +
                          (1 - vv) ** theta *
                          (1 - uu) ** -theta -
                          (1 - vv) ** theta) ** (-1 + 1 / theta) *
                         (1 - (1 - vv) ** theta))
    # cdf_given_vv_expr = cdf_given_uu_expr.subs({uu: vv, vv: uu})
    cdf_given_vv_expr = ((1 +
                          (1 - uu) ** theta *
                          (1 - vv) ** -theta -
                          (1 - uu) ** theta) ** (-1 + 1 / theta) *
                         (1 - (1 - uu) ** theta))
joe = Joe()


class Gumbel(Copulae, NoRotations):
    par_names = "uu", "vv", "theta"
    theta_start = 5.,
    theta_bounds = [(1. + 1e-9, 10.)]
    xx, yy, uu, vv, t, theta = sympy.symbols(("xx yy uu vv t theta"))
    # gen_expr = (-ln(t)) ** theta
    # gen_inv_expr = exp(-xx ** (1 / theta))
    cop_expr = exp(-((-ln(uu)) ** theta +
                     (-ln(vv)) ** theta) ** (1 / theta))
    cdf_given_uu_expr = (uu ** -1 * exp(-(xx ** theta +
                                          yy ** theta) ** (1 / theta)) *
                         (1 + (yy / xx) ** theta) ** (1 / theta - 1))
    cdf_given_uu_expr = cdf_given_uu_expr.subs({xx: -ln(uu),
                                                yy: -ln(vv)})
gumbel = Gumbel()


class AliMikailHaq(Archimedian, No90, No270):
    theta_start = .9,
    # theta_bounds = [(1e-9, 1.)]
    theta_bounds = [(-1 + 1e-6, 1. - 1e-6)]
    t, theta = sympy.symbols(("t", "theta"))
    gen_expr = ln((1 - theta * (1 - t)) / t)
    known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
alimikailhaq = AliMikailHaq()


class Gaussian(Copulae, NoRotations):
    par_names = "uu", "vv", "rho"
    theta_start = .0,
    theta_bounds = [(-1. + 1e-12, 1. - 1e-12)]
    # uu, vv, rho = sympy.symbols(par_names)
    # xx, yy = sympy.symbols(("xx", "yy"))
    # dens_expr = ((1 - rho ** 2) ** (-.5) *
    #              exp(-.5 * (xx ** 2 + yy ** 2 - 2 * rho * xx * yy) *
    #                  (1 - rho ** 2)) *
    #              exp(0.5 * (xx ** 2 + yy ** 2)))
    # uni_inv = sympy.sqrt(2) * error_functions.erfinv(2 * xx - 1)
    # dens_expr = dens_expr.subs(xx, uni_inv.subs(xx, uu))
    # dens_expr = dens_expr.subs(yy, uni_inv.subs(xx, vv))
    # dens_func = np.vectorize(sympy.lambdify((uu, vv, rho), dens_expr,))

    # def dens_func(self, uu, vv, rho):
    #     rho = np.squeeze(rho)
    #     erfu = erfinv(2 * uu - 1)
    #     erfv = erfinv(2 * vv - 1)
    #     return ((-rho ** 2 + 1) ** (-0.5) *
    #             np.exp((-rho ** 2 + 1) *
    #                    (2.0 * rho * erfu * erfv -
    #                     erfu ** 2 - erfv ** 2)) *
    #             np.exp(erfu ** 2 + erfv ** 2))

    def dens_func(self, uu, vv, rho):
        uu, vv = np.atleast_1d(uu, vv)
        xx = spstats.norm.ppf(uu).reshape(uu.shape)
        yy = spstats.norm.ppf(vv).reshape(vv.shape)
        return ((1 - rho ** 2) ** (-.5) *
                np.exp(-.5 * (xx ** 2 + yy ** 2 - 2 * rho * xx * yy) *
                       (1 - rho ** 2)) *
                np.exp(0.5 * (xx ** 2 + yy ** 2)))

    def copula_func(self, uu, vv, rho):
        rho = np.squeeze(rho)
        uu, vv = np.atleast_1d(uu, vv)
        uu_normal = spstats.norm.ppf(uu)
        vv_normal = spstats.norm.ppf(vv)
        broadcast = np.ndim(uu) == 2
        quantiles = np.empty_like(((uu + vv).ravel()))
        if broadcast:
            uu_normal = uu_normal.repeat(len(vv_normal))
            vv_normal = (vv_normal
                         .repeat(len(uu_normal))
                         .reshape((len(uu_normal),
                                   len(vv_normal)))
                         .T
                         .ravel())
        lower = np.full(2, -20.)
        infin = np.full(2, 2, dtype=int)
        for i, (u_normal, v_normal) in enumerate(zip(uu_normal, vv_normal)):
            quantiles[i] = mvn.mvndst(lower,
                                      np.array([u_normal, v_normal]),
                                      infin,
                                      rho
                                      )[1]
        if broadcast:
            quantiles = quantiles.reshape(uu.size, vv.size)
        return quantiles

    def cdf_given_u(self, uu, vv, rho):
        rho = np.squeeze(rho)
        return (-0.5 * erf((rho *
                            erfinv(2 * uu - 1) -
                            erfinv(2 * vv - 1)) /
                           np.sqrt(-rho ** 2 + 1)) + 0.5)

    def cdf_given_v(self, uu, vv, rho):
        return self.cdf_given_u(vv, uu, rho)
        # rho = np.squeeze(rho)
        # return (-0.5 * erf((rho *
        #                     erfinv(2 * uu - 1) -
        #                     erfinv(2 * vv - 1)) /
        #                    np.sqrt(-rho ** 2 + 1)) + 0.5)

    def fit(self, ranks_u, ranks_v, *args, **kwds):
        return np.corrcoef(ranks_u, ranks_v)[0, 1],
# gaussian = Gaussian()


class Plackett(Copulae):
    par_names = "uu", "vv", "delta"
    theta_start = .1,
    theta_bounds = [(1e-4, 20)]
    uu, vv, delta, eta = sympy.symbols(par_names + ("eta",))
    cop_expr = (1 / (2 * eta) * (1 + eta * (uu + vv) -
                                 ((1 + eta * (uu + vv)) ** 2 -
                                  4 * delta * eta * uu * vv) ** .5))
    cop_expr = cop_expr.subs(eta, delta - 1)
    cdf_given_uu_expr = 0.5 - 0.5 * ((eta * uu + 1 - (eta + 2) * vv) /
                                     ((1 + eta * (uu + vv)) ** 2 -
                                      4 * delta * eta * uu * vv) ** .5)
    cdf_given_uu_expr = cdf_given_uu_expr.subs(eta, delta - 1)
    known_fail = "inv_cdf_given_uu", "inv_cdf_given_vv"
plackett = Plackett()


class Galambos(Copulae, No90, No270):
    par_names = "uu", "vv", "delta"
    theta_start = 2.5,
    theta_bounds = [(.011, 50)]
    uu, vv, delta = sympy.symbols(par_names)
    cop_expr = uu * vv * exp(((-ln(uu)) ** -delta +
                              (-ln(vv)) ** -delta) ** (-1 / delta))
    xx, yy = sympy.symbols(("xx", "yy"))
    cdf_given_uu_expr = (vv * exp((xx ** -delta +
                                   yy ** -delta) ** (-1 / delta)) *
                         (1 - (1 + (xx / yy) ** delta) ** (-1 - 1 / delta)))
    cdf_given_uu_expr = cdf_given_uu_expr.subs({xx: -ln(uu), yy: -ln(vv)})
    # cdf_given_vv_expr = (uu * exp((xx ** -delta +
    #                                yy ** -delta) ** (-1 / delta)) *
    #                      (1 - (1 + (xx / yy) ** delta) ** (-1 - 1 / delta)))
    # cdf_given_vv_expr = cdf_given_vv_expr.subs({xx: -ln(vv), yy: -ln(uu)})
galambos = Galambos()


class Independence(Copulae, NoRotations):
    # having the theta here prevents trouble down the road...
    par_names = "uu", "vv", "theta"
    theta_start = None,
    uu, vv, _ = sympy.symbols(par_names)
    cop_expr = uu * vv

    def fit(self, uu, vv, *args, **kwds):
        return

    def sample(self, size, *args, **kwds):
        return random_sample(size), random_sample(size)

    def inv_cdf_given_u(self, uu, qq, *args):
        return qq

    def inv_cdf_given_v(self, vv, qq, *args):
        return qq
independence = Independence()


all_cops = OrderedDict((name, obj) for name, obj
                       in sorted(dict(locals()).items())
                       if isinstance(obj, Copulae))
# rotate all the cops!!
turned_cops = OrderedDict()
for cop_name, obj in all_cops.items():
    if isinstance(obj, NoRotations):
        continue
    for norot_cls in (No90, No180, No270):
        if isinstance(obj, norot_cls):
            continue
        rot_str = norot_cls.__name__[len("No"):]
        new_name = "%s_%s" % (cop_name, rot_str)
        old_type = type(obj)
        # # look for the same object in the call stack (really hacky!)
        # found = False
        # for depth in range(1, 1000):
        #     try:
        #         frame = sys._getframe(depth)
        #     except ValueError:
        #         break
        #     try:
        #         TurnedCop = frame.f_globals[new_name.capitalize()]
        #         turned_cops[new_name] = frame.f_globals[new_name]
        #         found = True
        #         break
        #     except KeyError:
        #         continue
        # if not found:
        #     TurnedCop = type(new_name,
        #                      (old_type,),
        #                      dict(old_type.__dict__))
        #     turned_cops[new_name] = TurnedCop()
        # make the rotated copulas importable
        TurnedCop = type(new_name,
                         (old_type,),
                         dict(old_type.__dict__))
        turned_cops[new_name] = TurnedCop()
        globals()[new_name.capitalize()] = TurnedCop
        globals()[new_name] = turned_cops[new_name]
all_cops.update(turned_cops)
all_cops = OrderedDict((name, obj) for name, obj
                       in sorted(all_cops.items()))


frozen_cops = tuple((name, copulas(copulas.theta_start))
                    for name, copulas in sorted(all_cops.items()))


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
            print("No convergence for %s" % copula.name)
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
