"""Bivariate Copulas intended for Vine Copulas."""
import functools
import importlib
import os
import warnings
from abc import ABCMeta, abstractproperty
from collections import OrderedDict
from contextlib import suppress

import numpy as np
import matplotlib.pyplot as plt
import sympy
from mpmath import mp
from scipy import stats as spstats
from scipy.interpolate import interp1d
from scipy.optimize import brentq, minimize, newton
from scipy.special import erf, erfinv
from scipy.stats import mvn
from sympy import exp, ln
from sympy.utilities import autowrap
import dill

from weathercop import cop_conf as conf, stats, tools

from weathercop.normal_conditional import norm_inv_cdf_given_u, norm_cdf_given_u

# used wherever theta can be inf in principle
theta_large = 1e3
zero, one = 1e-19, 1 - 1e-19

# here expressions are kept that sympy has problems with
faillog_file = conf.ufunc_tmp_dir / "known_fail"
conf.ufunc_tmp_dir.mkdir(parents=True, exist_ok=True)

# allow printing of more complex equations
# from matplotlib import rc
# rc("text", usetex=True)

rederive = None,
# rederive = "bb1"
# rederive = "bb2"
# rederive = "plackett"
# rederive = "nelsen16",
# rederive = "clayton",
# rederive = "gumbel"
# rederive = "galambos",
# rederive = "frank",
# rederive = "alimikailhaq",


def ufuncify_cython(cls, name, uargs, expr, *args, verbose=False, **kwds):
    expr_hash = tools.hash_cop(expr)
    module_name = "%s_%s_%s" % (cls.name, name, expr_hash)
    try:
        with tools.chdir(conf.ufunc_tmp_dir):
            ufunc = importlib.import_module("%s_0" % module_name).autofunc_c
    except (ImportError, AttributeError):
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


# from sympy.printing.theanocode import theano_function
# def ufuncify_theano(cls, name, uargs, expr, *args, verbose=False, **kwds):
#     with tools.shelve_open(conf.theano_cache) as sh:
#         expr_hash = tools.hash_cop(expr)
#         key = "%s_%s_%s" % (cls.name, name, expr_hash)
#         try:
#             return sh[key]
#         except KeyError:
#             if verbose:
#                 print("Building theano function for %s" % repr(expr))
#             dims = {key: 1 for key in uargs}
#             dtypes = {key: "float64" for key in uargs}
#             sh[key] = theano_function(uargs, [expr], dims=dims,
#                                       dtypes=dtypes,
#                                       on_unused_input="ignore")
#             return sh[key]
# def ufuncify(*args, **kwds):
#     try:
#         return ufuncify_theano(*args, **kwds)
#     except KeyError:
#         return ufuncify_cython(*args, **kwds)


def ufuncify_numpy(cls, name, uargs, expr, *args, verbose=False, **kwds):
    return autowrap.ufuncify(uargs, expr,
                             tempdir=conf.ufunc_tmp_dir,
                             verbose=verbose,
                             backend="numpy",
                             *args, **kwds)


def ufuncify(*args, backend="cython", **kwds):
    if backend in ("cython", "f2py"):
        return ufuncify_cython(*args, backend=backend, **kwds)
    elif backend == "numpy":
        return ufuncify_numpy(*args, **kwds)


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


def random_sample(size, bound=1e-9):
    """Sample in the closed interval (bound, 1 - bound)."""
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
            result[result < zero] = zero
            result[result > one] = one
            # result[(result < 0) | (~np.isfinite(result))] = 1e-15
        return result
    return inner


def broadcast_2d(func):
    @functools.wraps(func)
    def inner(*args, **kwds):
        if isinstance(args[0], Copulae):
            args = args[1:]
        try:
            args = np.nditer(args, flags=['multi_index',
                                          'zerosize_ok',
                                          'refs_ok'],
                             order='C').itviews
            shape = args[0].shape
            result = func(*[arg.ravel() for arg in args], **kwds)
            return result.reshape(shape)
        except ValueError:
            args = np.atleast_2d(*args)
            shape_broad = np.array([arg.shape for arg in args]).max(axis=0)
            args_broad_raveled = []
            for array in args:
                array_broad = np.empty(shape_broad)
                array_broad[:] = array
                args_broad_raveled += [array_broad.ravel()]
            result = func(*args_broad_raveled, **kwds)
            return result.reshape(shape_broad)
    return inner


def gen_arr_fun(fun):
    return lambda x, *args: fun(np.atleast_1d(x), *args)


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
        with suppress(TypeError):
            new_cls.n_par = len(new_cls.par_names) - 2
        if "backend" not in cls_dict:
            new_cls.backend = MetaCop.backend
        if "cop_expr" in cls_dict:
            new_cls = MetaCop.symmetries(new_cls)
            # auto-rotate the copula expression
            if name.endswith(("90", "180", "270")):
                new_cls.cop_expr = MetaCop.rotate_expr(new_cls)
                new_cls.rotated = True
            else:
                new_cls.rotated = False

            if "known_fail" in cls_dict:
                MetaCop.mark_failed(new_cls)
                known_fail = cls_dict["known_fail"]
            else:
                known_fail = None,
            new_cls.dens_func = MetaCop.density_from_cop(new_cls)
            new_cls.cdf_given_u = MetaCop.cdf_given_u(new_cls)
            new_cls.cdf_given_v = MetaCop.cdf_given_v(new_cls)
            new_cls.copula_func = MetaCop.copula_func(new_cls)
            if "inv_cdf_given_u" not in known_fail:
                ufunc = MetaCop.inv_cdf_given_u(new_cls)
                if ufunc is not None:
                    new_cls.inv_cdf_given_u = ufunc
            if "inv_cdf_given_v" not in known_fail:
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

    def rotate_expr(expr, degrees=None):
        """Rotate copula expression.

        Notes
        -----
        see p. 271
        """
        if degrees is None:
            degrees = int(expr.name.rsplit("_")[1])
            expr = expr.cop_expr
        uu, vv = sympy.symbols("uu vv")
        if degrees == 90:
            expr = uu - expr.subs({vv: 1 - vv})
        elif degrees == 180:
            expr = uu + vv - 1 + expr.subs({uu: 1 - uu,
                                            vv: 1 - vv})
        elif degrees == 270:
            expr = vv - expr.subs({uu: 1 - uu})
        else:
            raise ValueError("degrees must be one of (None, 90, 180, 270)")
        return expr

    def symmetries(cls):
        """Determine if copula is symmetric and set boolean attributes.
        """
        # if copula and its 180-degree rotated version are the same
        # there is symmetry 1
        copula = cls.cop_expr
        copula180 = MetaCop.rotate_expr(copula, 180)
        cls.symmetry1 = sympy.simplify(copula - copula180) == 0
        # uu, vv = sympy.symbols("uu vv")
        # copula_swapped = swap_symbols(copula, uu, vv)
        # cls.symmetry2 = sympy.simplify(copula - copula_swapped) == 0
        cls.symmetry2 = False
        return cls

    def copula_func(cls):
        uu, vv, *theta = sympy.symbols(cls.par_names)
        ufunc = ufuncify(cls, "copula",
                         [uu, vv] + theta, cls.cop_expr,
                         backend=cls.backend,
                         verbose=False)
        return ufunc

    def density_func(cls):
        uu, vv, *theta = sympy.symbols(cls.par_names)
        dens_expr = cls.dens_expr
        ufunc = ufuncify(cls, "density",
                         [uu, vv] + theta, dens_expr,
                         backend=cls.backend,
                         verbose=False)
        return ufunc

    def conditional_cdf(cls, conditioning):
        uu, vv, *theta = sympy.symbols(cls.par_names)
        expr_attr = "cdf_given_%s_expr" % conditioning
        with tools.shelve_open(conf.sympy_cache) as sh:
            key = ("%s_cdf_given_%s_%s" %
                   (cls.name, conditioning, tools.hash_cop(cls)))
            if key not in sh or cls.name in rederive:
                print("Generating %s-conditional %s" %
                      (conditioning, cls.name))
                # a good cop always stays positive!
                # good_cop = sympy.Piecewise((cls.cop_expr,
                #                             cls.cop_expr > 0),
                #                            (0, True))
                good_cop = cls.cop_expr
                # good_cop = sympy.Min(cls.cop_expr, one)
                # good_cop = sympy.Piecewise((cls.cop_expr,
                #                             cls.cop_expr <= one),
                #                            (one, True))
                conditional_cdf = sympy.diff(good_cop, conditioning)
                conditional_cdf = sympy.simplify(conditional_cdf)
                sh[key] = conditional_cdf
            conditional_cdf = sh[key]
            setattr(cls, expr_attr, conditional_cdf)
        ufunc = ufuncify(cls, "conditional_cdf",
                         [uu, vv] + theta, conditional_cdf,
                         backend=cls.backend,
                         verbose=False)
        return ufunc

    def cdf_given_u(cls):
        return cls.conditional_cdf(sympy.symbols("uu"))

    def cdf_given_v(cls):
        return cls.conditional_cdf(sympy.symbols("vv"))

    def inverse_conditional_cdf(cls, conditioning):
        uu, vv, qq, theta = sympy.symbols("uu vv qq theta")
        conditioned = list(set((uu, vv)) - set([conditioning]))[0]
        key = ("%s_inv_cdf_given_%s_%s" %
               (cls.name, conditioning, tools.hash_cop(cls)))
        # keep a log of what does not work in order to not repeat ad
        # nauseum
        try:
            if (key + os.linesep) in faillog_file.open():
                return
        except FileNotFoundError:
            open(faillog_file, "a").close()
        attr_name = "inv_cdf_given_%s_expr" % conditioning
        if not hasattr(cls, attr_name) or cls.rotated:
            # cached sympy derivation
            with tools.shelve_open(conf.sympy_cache) as sh:
                if key not in sh or cls.name in rederive:
                    print("Generating inverse %s-conditional %s" %
                          (conditioning, cls.name))
                    cdf_given_expr = getattr(cls,
                                             "cdf_given_%s_expr" %
                                             conditioning)
                    try:
                        inv_cdf = sympy.solve(cdf_given_expr - qq,
                                              conditioned)[0]
                    except (NotImplementedError, ValueError, TypeError):
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
                             [conditioning, qq, theta], inv_cdf,
                             backend=cls.backend,
                             verbose=False)
        except (autowrap.CodeWrapError, TypeError):
            warnings.warn("Could not compile inv.-conditional for %s" %
                          cls.name)
            mark_failed(key)
            return
        return ufunc

    def inv_cdf_given_u(cls):
        return cls.inverse_conditional_cdf(sympy.symbols("uu"))

    def inv_cdf_given_v(cls):
        # if not hasattr(cls, "inv_cdf_given_vv_expr"):
        #     # if we are given an expression for the u-conditional, we can
        #     # substitute v for u to get the corresponding v-conditional
        #     try:
        #         inv_cdf = getattr(cls, "inv_cdf_given_uu_expr")
        #         uu, vv = sympy.symbols("uu vv")
        #         inv_cdf = inv_cdf.subs({uu: vv})
        #         setattr(cls, "inv_cdf_given_vv_expr", inv_cdf)
        #     except AttributeError:
        #         pass
        return cls.inverse_conditional_cdf(sympy.symbols("vv"))

    def density_from_cop(cls):
        """Copula density obtained by sympy differentiation compiled with
        cython.
        """
        uu, vv, *theta = sympy.symbols(cls.par_names)
        with tools.shelve_open(conf.sympy_cache) as sh:
            key = "%s_density_%s" % (cls.name, tools.hash_cop(cls))
            if key not in sh or cls.name in rederive:
                print("Generating density for %s" % cls.name)
                dens_expr = sympy.diff(cls.cop_expr, uu, vv)
                # dens_expr = sympy.Piecewise((dens_expr, cls.cop_expr > 0),
                #                             (0, True))
                sh[key] = sympy.simplify(dens_expr)
            dens_expr = sh[key]
        # for outer pleasure
        cls.dens_expr = dens_expr
        ufunc = ufuncify(cls, "density",
                         [uu, vv] + theta, dens_expr,
                         backend=cls.backend,
                         verbose=False)
        return ufunc


class MetaArch(MetaCop):

    def __new__(cls, name, bases, cls_dict):
        if ("gen_expr" in cls_dict) and ("cop_expr" not in cls_dict):
            gen = cls_dict["gen_expr"]
            uu, vv, x, t = sympy.symbols("uu vv x t")
            with tools.shelve_open(conf.sympy_cache) as sh:
                key = "%s_cop_%s" % (name, tools.hash_cop(gen))
                if key not in sh or name in rederive:
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
    zero, one = 1e-19, 1 - 1e-19
    # needed in inverse conditionals
    _ranks2_calc = np.linspace(zero, one, 5000)

    @abstractproperty
    def theta_start(self):
        """Starting solution for parameter estimation."""
        pass

    @abstractproperty
    def par_names(self):
        pass

    def __call__(self, *theta):
        return Frozen(self, *theta)

    def __getstate__(self):
        dict_ = dict(self.__dict__)
        if hasattr(self, "bad_attrs"):
            for attr in self.bad_attrs:
                del dict_[attr]
        return dict_

    def __setstate__(self, dict_):
        self.__dict__ = dict_
        self.__init__()

    def density(self, uu, vv, *theta):
        theta = tuple(np.full_like(uu, np.atleast_1d(the))
                      for the in theta)
        return self.dens_func(uu, vv, *theta)

    def _inverse_conditional(self, conditional_func, ranks, quantiles,
                             *theta, given_v=False):
        """Numeric inversion of conditional_func (inv_cdf_given_u or
        inv_cdf_given_v), to be used as a last resort.

        """
        theta = np.array(self.theta if theta is None else theta)
        ranks1, quantiles = np.atleast_1d(ranks, quantiles)
        quantiles = np.squeeze(quantiles)
        thetas = np.atleast_2d(theta)
        if thetas.size == 1:
            thetas = np.full(2 * ranks1.shape, theta)
        if quantiles.size == 1:
            quantiles = np.full_like(ranks1, quantiles)

        ranks2 = np.copy(ranks1)
        tol = zero
        ltol, utol = tol, 1 - tol
        rank_u_ar, rank_v_ar = np.empty((2, 1))
        theta_ar = np.empty((self.n_par, 1))
        for i, rank1 in enumerate(ranks1):
            quantile, theta = quantiles[i], thetas[:, i]
            theta_ar[:, 0] = theta

            def f(rank2):
                if given_v:
                    rank_u_ar[0], rank_v_ar[0] = rank2, rank1
                else:
                    rank_u_ar[0], rank_v_ar[0] = rank1, rank2
                quantile_calc = conditional_func(rank_u_ar,
                                                 rank_v_ar,
                                                 *theta_ar)
                # if np.isnan(quantile_calc):
                if not (ltol < rank2 < utol):
                    if rank2 < ltol:
                        quantile_calc = np.array((ltol,))
                    elif rank2 > utol:
                        quantile_calc = np.array((utol,))
                return quantile_calc[0] - quantile

            debug = False
            # cls = Clayton
            cls = AliMikailHaq
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                skip_else = False
                try:
                    x0 = self.x0(rank1, quantile, theta)
                    rank2 = newton(f, x0=x0)
                    if not (0 < rank2 < 1):
                        raise ValueError
                except RuntimeWarning:
                    # the elevated warning is not considered for the
                    # else block. bug in the warnings module?
                    skip_else = True
                except RuntimeError:
                    if tol < rank1 < (1 - tol):
                        ranks2[i] = rank1
                        if debug and isinstance(self, cls):
                            warnings.warn("Used rank2 = rank1 for %f in %s" %
                                          (rank1, self.name))
                        continue
                except ValueError:
                    pass
                else:
                    if skip_else:
                        continue
                    # trust newton only when it doesn't delivers extreme values
                    if tol < rank2 < (1 - tol):
                        ranks2[i] = rank2
                        if debug and isinstance(self, cls):
                            warnings.warn("Used newton for rank1 %.6f in %s"
                                          % (rank1, self.name))
                        continue

            try:
                rank2 = brentq(f, 1e-19, 1)
            except ValueError:
                pass
            else:
                if tol < rank2 < (1 - tol):
                    ranks2[i] = rank2
                    if debug and isinstance(self, cls):
                        warnings.warn("Used brentq for rank1 %.6f in %s" %
                                      (rank1, self.name))
                    continue

            # fall back to linear interpolation
            ranks2_calc = self._ranks2_calc
            ranks1_full = np.full_like(ranks2_calc, rank1)
            theta_full = tuple(np.full_like(ranks2_calc, the)
                               for the in theta)
            if given_v:
                ranks_u, ranks_v = ranks2_calc, ranks1_full
            else:
                ranks_u, ranks_v = ranks1_full, ranks2_calc

            quantiles_calc = conditional_func(ranks_u, ranks_v, *theta_full)
            quantiles_calc = np.squeeze(quantiles_calc)
            quantiles_calc[0] = self.zero
            quantiles_calc[-1] = self.one
            # if there is more than one zero involved, interpolating
            # between them gives unwanted results.
            f_int = interp1d(quantiles_calc,
                             ranks2_calc,
                             # bounds_error=True,
                             bounds_error=False,
                             fill_value=(zero, one)
                             )
            if debug and isinstance(self, cls):
                warnings.warn("Used interpolation for rank1 %.6f in %s" %
                              (rank1, self.name))
            ranks2[i] = f_int(quantile)

        return ranks2

    def inv_cdf_given_u(self, ranks_u, quantiles, *theta):
        """Numeric inversion of cdf_given_u, to be used as a last resort.
        """
        return self._inverse_conditional(self.cdf_given_u, ranks_u,
                                         quantiles, *theta)

    def inv_cdf_given_v(self, ranks_v, quantiles, *theta):
        """Numeric inversion of cdf_given_v, to be used as a last resort.
        """
        return self._inverse_conditional(self.cdf_given_v, ranks_v,
                                         quantiles, *theta,
                                         given_v=True)

    def sample(self, size, *theta):
        uu = random_sample(size)
        xx = random_sample(size)
        theta = tuple(np.full_like(uu, the)
                      for the in theta)
        vv = self.inv_cdf_given_u(uu, xx, *theta)
        return uu, vv

    def generate_fitted(self, ranks_u, ranks_v, *args, **kwds):
        """Returns a Fitted instance that contains ranks_u, ranks_v and the
        fitted theta.
        """
        theta = self.fit(ranks_u, ranks_v, *args, **kwds)
        return Fitted(self, ranks_u, ranks_v, *theta)

    def x0(self, rank, quantile, theta):
        """Starting-value for newton root finding in the inverse conditionals.

        Replace it with something more meaningfull in childs.
        """
        return rank

    def fit(self, *args, **kwds):
        # overwrite this function in child implementations, if a
        # better method than maximum likelihood is available as
        # fitting procedure.
        return self.fit_ml(*args, **kwds)

    def fit_ml(self, ranks_u, ranks_v, method=None, x0=None, verbose=False):
        """Maximum likelihood estimate."""

        def neg_log_likelihood(theta):
            dens = self.density(ranks_u, ranks_v, *theta)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mask = (dens <= 0) | ~np.isfinite(dens)
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
        self.plot_copula(theta=theta, fig=fig, ax=axs[0])
        self.plot_density(theta=theta, scatter=scatter, kind=kind,
                          opacity=opacity, fig=fig, ax=axs[1])
        return fig, axs

    def plot_density(self, *, theta=None, scatter=True, fig=None,
                     ax=None, kind="contourf", opacity=.1,
                     sample_size=1000, s_kwds=None, c_kwds=None):
        if theta is None:
            try:
                theta = self.theta,
                if isinstance(theta, sympy.core.symbol.Symbol):
                    raise AttributeError
            except AttributeError:
                theta = self.theta_start,
        if s_kwds is None:
            s_kwds = dict()
        if c_kwds is None:
            c_kwds = dict(alpha=.5,
                          # linewidth=.25
                          )
        n_per_dim = 100
        uu = vv = stats.rel_ranks(np.arange(n_per_dim))
        theta_dens = tuple(np.repeat(the, n_per_dim ** 2)
                           for the in theta[0])
        density = (self
                   .density(uu.repeat(n_per_dim),
                            np.tile(vv, n_per_dim),
                            *theta_dens)
                   .reshape(n_per_dim, n_per_dim)
                   )
        if fig is None or ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        if not isinstance(self, Independence):
            # get rid of large values for visualizations sake
            density[density >= np.sort(density.ravel())[-10]] = np.nan
            if kind == "contourf":
                ax.contourf(uu, vv, density, 40, **c_kwds)
            elif kind == "contour":
                ax.contour(uu, vv, density, 40, **c_kwds)

        if scatter:
            sample_size = 1000
            theta_sample = tuple(np.repeat(the, sample_size)
                                 for the in theta[0])
            u_sample, v_sample = self.sample(sample_size,
                                             *theta_sample)
            ax.scatter(u_sample, v_sample,
                       marker="o",
                       facecolor=(0, 0, 0, 0),
                       edgecolor=(0, 0, 0, opacity),
                       **s_kwds)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.suptitle(fr"{self.name} $\theta$ = {theta[0][0]:.3f}")
        return fig, ax

    def plot_copula(self, theta=None, fig=None, ax=None,
                    n_per_dim=100, kind="contourf"):
        if theta is None:
            try:
                theta = self.theta
                if isinstance(theta, sympy.core.symbol.Symbol):
                    raise AttributeError
            except AttributeError:
                # theta = tuple(self.theta_start)
                theta = tuple(np.repeat(the, n_per_dim ** 2)
                              for the in self.theta_start)
        uu = vv = stats.rel_ranks(np.arange(n_per_dim))
        cc = self.copula_func(uu.repeat(n_per_dim),
                              np.tile(vv, n_per_dim),
                              *theta)
        cc = cc.reshape(n_per_dim, n_per_dim)
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        if kind == "contourf":
            cax = ax.contourf(uu, vv, cc, 40, vmin=0, vmax=1)
        elif kind == "contour":
            cax = ax.contour(uu, vv, cc, 40, vmin=0, vmax=1)
        if hasattr(self, "cop_expr"):
            title = (sympy.printing
                     .latex(self.cop_expr, mode="inline")
                     .replace("uu", "u")
                     .replace("vv", "v"))
            if "cases" not in title:
                ax.set_title(title)
        else:
            ax.set_title(self.name)
        fig.colorbar(cax)
        return fig, ax


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
                def proxied(*args):
                    if not args:
                        return attr()
                    theta = tuple(np.full_like(args[0], the)
                                  for the in self.theta[0])
                    return attr(*(args + theta))
                return proxied
            return attr


@functools.total_ordering
class Fitted:

    def __init__(self, copula, ranks_u, ranks_v, *theta,
                 verbose=False):
        self.copula = copula
        self.ranks_u = ranks_u
        self.ranks_v = ranks_v
        self.theta = tuple(np.full_like(ranks_u, the)
                           for the in theta)
        self.name = "fitted %s" % copula.name
        self.verbose = self.copula.verbose = verbose
        if isinstance(copula, Independence):
            self.likelihood = 0
        else:
            density = copula.density(ranks_u, ranks_v, *theta)
            mask = np.isfinite(density)
            density[~mask] = 1e-12
            density[density <= 0] = 1e-12
            dens_masked = density[mask]
            if len(dens_masked) == 0:
                self.likelihood = -np.inf
            else:
                self.likelihood = np.sum(np.log(dens_masked))

            if verbose:
                print(4 * " ",
                      self.name[len("fitted "):],
                      self.likelihood, end="")

    def plot_qq(self, ax=None, fig=None, opacity=.25, s_kwds=None,
                title=None, *args, **kwds):
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

    def plot_density(self, *, scatter=True, ax=None, kind="contourf",
                     opacity=.1, n_per_dim=100, s_kwds=None,
                     c_kwds=None):
        theta = self.theta
        if s_kwds is None:
            s_kwds = dict()
        if c_kwds is None:
            c_kwds = dict(alpha=.5,
                          linewidth=.25)
        uu = vv = stats.rel_ranks(np.arange(n_per_dim))
        density = (self
                   .density(uu.repeat(n_per_dim),
                            np.tile(vv, n_per_dim),
                            np.repeat(theta[0][0],
                                      n_per_dim * n_per_dim))
                   .reshape(n_per_dim, n_per_dim)
                   )
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
        density_valid = not np.all(np.isnan(density))
        if density_valid and not isinstance(self.copula, Independence):
            # get rid of large values for visualizations sake
            density[density >= np.sort(density.ravel())[-10]] = np.nan
            if kind == "contourf":
                ax.contourf(uu, vv, density, 40, **c_kwds)
            elif kind == "contour":
                ax.contour(uu, vv, density, 40, **c_kwds)

        if scatter:
            ax.scatter(self.ranks_u, self.ranks_v,
                       marker="o",
                       facecolor=(0, 0, 0, 0),
                       edgecolor=(0, 0, 0, opacity),
                       **s_kwds)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.gcf().suptitle(fr"{self.name} $\theta$ = {theta[0].mean():.3f}")
        return ax

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
        return f"Fitted({self.copula.name})"

    def _call_cdf(self, name, t=None, *, conditioned, condition):
        if t is not None:
            theta = np.array([self.theta[0][0]])
        else:
            size = len(condition)
            theta = self.theta[0]
            if size == 1:
                theta = theta[:1]
        method = getattr(self.copula, name)
        return method(conditioned, condition, theta)

    def cdf_given_u(self, t=None, *, conditioned, condition):
        return self._call_cdf("cdf_given_u", t,
                              conditioned=condition,
                              condition=conditioned)

    def cdf_given_v(self, t=None, *, conditioned, condition):
        return self._call_cdf("cdf_given_v", t,
                              conditioned=conditioned,
                              condition=condition)

    def inv_cdf_given_u(self, t=None, *, conditioned, condition):
        return self._call_cdf("inv_cdf_given_u", t,
                              conditioned=condition,
                              condition=conditioned)

    def inv_cdf_given_v(self, t=None, *, conditioned, condition):
        return self._call_cdf("inv_cdf_given_v", t,
                              conditioned=conditioned,
                              condition=condition)


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




# # conditionals are returning values > 1 for theta=3
# class Clayton(Copulae, NoRotations):
#     """p. 168"""
#     # backend = "numpy"
#     par_names = "uu", "vv", "theta"
#     theta_start = 2.5,
#     # theta_start = np.array([2.5])
#     theta_bounds = [(-1 + 1e-8, 10.)]
#     # uu, vv, t, theta = sympy.symbols("uu vv t theta")
#     uu, vv, t, theta, qq = sympy.symbols("uu vv t theta qq")
#     cop_expr = (uu ** -theta + vv ** -theta - 1) ** (-1 / theta)
#     # dens_expr = ((1 + theta) * (uu * vv) ** (-theta - 1) *
#     #              (uu ** -theta + vv ** -theta - 1) ** (-2 - 1 / theta))
#     # gen_expr = (1 / theta) * (t ** (-theta) - 1)
#     # gen_inv_expr = (1 + theta * t) ** (-1 / theta)
#     cdf_given_uu_expr = (uu ** (1 - theta) *
#                          (1 + uu ** theta *
#                           (vv ** -theta)) ** ((theta + 1) / theta))
#     inv_cdf_given_uu_expr = ((qq ** (-theta / (1 + theta)) - 1) *
#                              uu ** -theta + 1) ** (-1 / theta)
#     # cdf_given_vv_expr = cdf_given_uu_expr.subs(dict(uu=vv))
#     # inv_cdf_given_vv_expr = inv_cdf_given_uu_expr.subs(dict(uu=vv))
#     # known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
# clayton = Clayton()


# inverted conditionals fail, fit also
class Frank(Archimedian, No180):
    """Also Nelsen05"""
    theta_start = 5.,
    theta_bounds = [(-theta_large, theta_large)]
    xx, uu, vv, t, theta = sympy.symbols("xx uu vv t theta")
    # gen_expr = -ln((exp(-theta * t) - 1) /
    #                (exp(-theta) - 1))
    # gen_inv_expr = -1 / theta * ln(1 + exp(-xx) * (exp(-theta) - 1))
    # cop_expr = (-theta ** -1 * ln(1 + ((exp(-theta * uu) - 1) *
    #                                    (exp(-theta * vv) - 1)) /
    #                               (exp(-theta) - 1)))
    cop_expr = (-theta ** -1 * ln((1 - exp(-theta) -
                                   (1 - exp(-theta * uu)) *
                                   (1 - exp(-theta * vv))) /
                                  (1 - exp(-theta))))
    dens_expr = ((theta * (1 - exp(-theta)) * (exp(-theta * (uu + vv)))) /
                 (1 - exp(-theta) -
                  (1 - exp(-theta * uu)) *
                  (1 - exp(-theta * vv))))
    # cdf_given_uu_expr = (exp(-theta * uu) * ((1 - exp(-theta)) *
    #                                          (1 - exp(-theta * vv)) ** -1 -
    #                                          (1 - exp(-theta * uu))) ** -1)
    # qq = sympy.symbols("qq")
    # inv_cdf_given_uu_expr = (-theta ** -1 * ln(1 -
    #                                            (1 - exp(-theta)) /
    #                                            ((qq ** -1 - 1) *
    #                                             exp(-theta * uu) + 1)))
    # known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
frank = Frank()


# inverse conditionals causing pain
class GumbelBarnett(Archimedian, No180, No270):
    """Also Nelsen09"""
    theta_start = .9,
    theta_bounds = [(1e-9, 1. - 1e-9)]
    # t, theta = sympy.symbols("t theta")
    # gen_expr = ln(1 - theta * ln(t))
    uu, vv, theta = sympy.symbols("uu vv theta")
    # cop_expr = (uu + vv - 1 +
    #             (1 - uu) * (1 - vv) *
    #             exp(-theta * ln(1 - uu) * ln(1 - vv)))
    cop_expr = uu * vv * exp(-theta * ln(uu) * ln(vv))
    # dens_expr = ((-theta +
    #               (1 - theta * ln(1 - uu)) *
    #               (1 - theta * ln(1 - vv))) *
    #              exp(-theta * ln(1 - uu) * ln(1 - vv)))
gumbelbarnett = GumbelBarnett()


# # conditionals are not in [0, 1]
# class Nelsen02(Archimedian, NoRotations):
#     theta_start = 2.5,
#     theta_bounds = [(1 + 1e-9, 80)]
#     uu, vv, t, theta = sympy.symbols("uu vv t theta")
#     gen_expr = (1 - t) ** theta
#     # cop_expr = (1 - ((1 - uu) ** theta +
#     #                  (1 - vv) ** theta) ** (1 / theta))
#     # cop_expr = sympy.Piecewise((cop_expr, cop_expr > 0),
#     #                            (0, True))
#     known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
# nelsen02 = Nelsen02()


# this is the joe copula (see below)
# class Nelsen06(Archimedian):
#     theta_start = 1.5,
#     theta_bounds = [(1., theta_large)]
#     t, theta = sympy.symbols("t theta")
#     gen_expr = -ln(1 - (1 - t) ** theta)
# nelsen06 = Nelsen06()


# # does not integrate to one
# class Nelsen07(Archimedian):
#     theta_start = .5,
#     theta_bounds = [(0, 1)]
#     t, theta = sympy.symbols("t theta")
#     # this seems wrong! Why the parentheses around "1 - theta"?
#     # Should there be an exponent on that?
#     gen_expr = -ln(theta * t + (1 - theta))
#     # known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
# nelsen07 = Nelsen07()


# # cdf_given_u fails test
# class Nelsen08(Archimedian):
#     theta_start = 1.5,
#     theta_bounds = [(1. + 1e-3, theta_large)]
#     t, theta = sympy.symbols("t theta")
#     gen_expr = (1 - t) / (1 + (theta - 1) * t)
#     # uu, vv = sympy.symbols("uu vv")
#     # cop_expr = ((theta ** 2 * uu * vv - (1 - uu) * (1 - vv)) /
#     #             (theta ** 2 - (theta - 1) ** 2 * (1 - uu) * (1 - vv)))
#     # cop_expr = sympy.Piecewise((cop_expr, cop_expr > 0),
#     #                            (0, True))
#     known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
# nelsen08 = Nelsen08()


# # i don't trust it!
# class Nelsen10(Archimedian, No180):
#     theta_start = .5,
#     theta_bounds = [(1e-3, 1.)]
#     t, theta = sympy.symbols("t theta")
#     gen_expr = ln(2 * t ** (-theta) - 1)
# nelsen10 = Nelsen10()


# # density does not integrate to 1
# class Nelsen11(Archimedian, No90, No270):
#     theta_start = .25,
#     theta_bounds = [(1e-6, .5)]
#     uu, vv, t, theta = sympy.symbols("uu vv t theta")
#     gen_expr = ln(2 - t ** theta)
#     cop_expr = (uu ** theta * vv ** theta -
#                 2 * (1 - uu ** theta) * (1 - vv ** theta)) ** (1 / theta)
#     # cop_expr = sympy.Piecewise((cop_expr, cop_expr > 0),
#     #                            (0, True))
# nelsen11 = Nelsen11()


class Nelsen12(Archimedian, NoRotations):
    theta_start = 1.5,
    theta_bounds = [(1., 70)]
    t, theta = sympy.symbols("t theta")
    gen_expr = (1 / t - 1) ** theta
nelsen12 = Nelsen12()


class Nelsen13(Archimedian, NoRotations):
    theta_start = .5,
    theta_bounds = [(1e-12, 300)]
    t, theta = sympy.symbols("t theta")
    gen_expr = (1 - ln(t)) ** theta - 1
nelsen13 = Nelsen13()


class Nelsen14(Archimedian, NoRotations):
    theta_start = 3.,
    theta_bounds = [(1., 60.)]
    t, theta = sympy.symbols("t theta")
    gen_expr = (t ** (-1 / theta) - 1) ** theta
nelsen14 = Nelsen14()


# # funky conditional
# class Nelsen15(Archimedian, NoRotations):
#     # TODO: this probably has a real name, look it up!
#     theta_start = 3.,
#     theta_bounds = [(1., 45)]
#     t, theta = sympy.symbols("t theta")
#     gen_expr = (1 - t ** (1 / theta)) ** theta
# nelsen15 = Nelsen15()


# # either the copula function or the conditionals are not in [0, 1]
# class Nelsen16(Archimedian, NoRotations):
#     theta_start = .25,
#     theta_bounds = [(1e-3, theta_large)]
#     uu, vv, S, t, theta = sympy.symbols("uu vv S t theta")
#     # gen_expr = (theta / t + 1) * (1 - t)
#     cop_expr = (S + sympy.sqrt(S ** 2 + 4 * theta)) / 2
#     # cop_expr = S + sympy.sqrt(S ** 2 + 4 * theta)
#     cop_expr = cop_expr.subs(S,
#                              uu + vv - 1 - theta * (1 / uu + 1 / vv - 1))
#     cdf_given_uu_expr = (theta / uu ** 2 + 1 -
#                          (2 * theta / uu ** 2 + 2) / 2 *
#                          (theta * (-1 + 1 / vv + 1 / uu) - uu - vv + 1) /
#                          sympy.sqrt(4 * theta +
#                                     (theta * (-1 + 1 / vv + 1 / uu) -
#                                      uu - vv + 1) ** 2))
#     cdf_given_vv_expr = cdf_given_uu_expr.subs(uu, vv)
#     # cop_expr = sympy.Piecewise((cop_expr, cop_expr > 0),
#     #                            (0, True))
#     known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
# nelsen16 = Nelsen16()


# # very slow class construction
# class Nelsen17(Archimedian):
#     theta_start = 1,
#     theta_bounds = [(-theta_large, theta_large)]
#     t, theta = sympy.symbols("t theta")
#     gen_expr = -ln(((1 + t) ** (-theta) - 1) / (2 ** (-theta) - 1))
# nelsen17 = Nelsen17()


# # funky conditional
# class Nelsen18(Archimedian, NoRotations):
#     theta_start = 2.5,
#     theta_bounds = [(2., 10)]
#     uu, vv, t, theta = sympy.symbols("uu vv t theta")
#     gen_expr = exp(theta / (t - 1))
#     cop_expr = 1 + theta / ln(exp(theta / (uu - 1)) +
#                               exp(theta / (vv - 1)))
#     # cop_expr = sympy.Piecewise((cop_expr, cop_expr > 0),
#     #                            (0, True))
#     known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
# nelsen18 = Nelsen18()


# # conditionals return nan for extremes
# class Nelsen19(Archimedian, No90):
#     theta_start = 1.,
#     theta_bounds = [(1e-9, theta_large)]
#     t, theta = sympy.symbols("t theta")
#     gen_expr = exp(theta / t) - exp(theta)
#     # uu, vv = sympy.symbols("uu vv")
#     # cop_expr = theta / (ln(exp(theta / uu) +
#     #                        exp(theta / vv) -
#     #                        exp(theta)))
#     # known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
# nelsen19 = Nelsen19()


# # TypeError: argument is not an mpz
# class Nelsen20(Archimedian, No90, No270):
#     theta_start = .05,
#     theta_bounds = [(1e-9, .9)]
#     uu, vv, t, theta = sympy.symbols("uu vv t theta")
#     gen_expr = exp(t ** (-theta)) - mp.e
#     cop_expr = (ln(exp(uu ** -theta) +
#                    exp(vv ** -theta) - mp.e)) ** (-1 / theta)
# nelsen20 = Nelsen20()


# # funky conditionals
# class Nelsen21(Archimedian):
#     theta_start = 3.,
#     theta_bounds = [(1., 5)]
#     uu, vv, t, theta = sympy.symbols("uu vv t theta")
#     gen_expr = 1 - (1 - (1 - t) ** theta) ** (1 / theta)
#     cop_expr = (1 - (1 - ((1 - (1 - uu) ** theta) ** (1 / theta) +
#                           (1 - (1 - vv) ** theta) ** (1 / theta) -
#                           1) ** theta)) ** (1 / theta)
#     piece = ((1 - (1 - uu) ** theta) ** (1 / theta) +
#              (1 - (1 - vv) ** theta) ** (1 / theta) - 1)
#     piece = sympy.Piecewise((piece, piece > 0),
#                             (0, True))
#     # cop_expr = (1 - (1 - piece ** theta)) ** (1 / theta)
#     # known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
# nelsen21 = Nelsen21()


# # copula function does not start at 0
# class Nelsen22(Archimedian, NoRotations):
#     from sympy import asin
#     theta_start = .5,
#     theta_bounds = [(0, 1)]
#     t, theta = sympy.symbols("t theta")
#     gen_expr = asin(1 - t ** theta)
#     known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
# nelsen22 = Nelsen22()


# conditional produces values > 1
class Joe(Copulae, NoRotations):
    par_names = "uu vv theta".split()
    theta_start = 5.,
    theta_bounds = [(1 + 1e-9, 30)]
    xx, uu, vv, t, theta = sympy.symbols("xx uu vv t theta")
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
    cdf_given_vv_expr = cdf_given_uu_expr.subs({uu: vv, vv: uu})
    # cdf_given_vv_expr = ((1 +
    #                       (1 - uu) ** theta *
    #                       (1 - vv) ** -theta -
    #                       (1 - uu) ** theta) ** (-1 + 1 / theta) *
    #                      (1 - (1 - uu) ** theta))
    # known_fail = "inv_cdf_given_u", "inv_cdf_given_v"

    def __init__(self):
        uu, p, delta = sympy.symbols("uu p delta")
        x0_expr = (1 - (((1 - p) ** (-delta / (1 + delta)) - 1) *
                        (1 - uu) ** -delta + 1) ** (-1 / delta))
        x0_expr = x0_expr.subs(delta, delta - 1)
        x0_func = ufuncify(self.__class__, "x0", [uu, p, delta],
                           x0_expr, backend=MetaArch.backend)
        self.x0_func = broadcast_2d(x0_func)

    def x0(self, rank, quantile, theta):
        return self.x0_func(rank, quantile, theta)
joe = Joe()


class Gumbel(Archimedian, NoRotations):
    theta_start = 2.5,
    theta_bounds = [(1 + 1e-9, theta_large)]
    t, theta = sympy.symbols("t theta")
    gen_expr = (-ln(t)) ** theta
    xx, yy, uu, vv, t, theta = sympy.symbols("xx yy uu vv t theta")
    cdf_given_uu_expr = (uu ** -1 * exp(-(xx ** theta +
                                          yy ** theta) ** (1 / theta)) *
                         (1 + (yy / xx) ** theta) ** (1 / theta - 1))
    cdf_given_uu_expr = cdf_given_uu_expr.subs({xx: -ln(uu),
                                                yy: -ln(vv)})
    known_fail = "inv_cdf_given_u", "inv_cdf_given_v"

    def _inverse_conditional(self, ranks, quantiles, theta,
                             given_v=False):
        theta = np.array(self.theta if theta is None else theta)
        ranks1, quantiles, thetas = np.atleast_1d(ranks, quantiles,
                                                  theta)
        quantiles, thetas = map(np.squeeze, (quantiles, thetas))
        if thetas.size == 1:
            thetas = np.full_like(ranks1, theta)
        if quantiles.size == 1:
            quantiles = np.full_like(ranks1, quantiles)
        ranks2 = np.empty_like(ranks1)

        x, y, z, p, delta = sympy.symbols("x y z p delta")
        h_expr = (z + (delta - 1) * ln(z) -
                  (x + (delta - 1) * ln(x) - ln(p)))
        h_prime_expr = 1 + (delta - 1) * z ** -1
        h_args = [z, x, p, delta]

        if given_v:
            h_expr = swap_symbols(h_expr, x, y)
            h_prime_expr = swap_symbols(h_prime_expr, x, y)
            h_args = [z, y, p, delta]

        h = ufuncify(self.__class__, "h", h_args, h_expr,
                     backend=self.backend)
        h_prime = ufuncify(self.__class__, "h_prime", h_args, h_prime_expr,
                           backend=self.backend)
        h, h_prime = map(broadcast_1d, (h, h_prime))

        for i, rank1 in enumerate(ranks1):
            x = -np.log(rank1)
            z_root = newton(h, x0=x,
                            fprime=h_prime,
                            args=(x, quantiles[i], theta))
            y_root = (z_root ** theta - x ** theta) ** (1 / theta)
            ranks2[i] = np.exp(-float(y_root))
        return ranks2

    def inv_cdf_given_u(self, ranks_u, quantiles, theta):
        return self._inverse_conditional(ranks_u, quantiles, theta)

    def inv_cdf_given_v(self, ranks_v, quantiles, theta):
        return self._inverse_conditional(ranks_v, quantiles, theta,
                                         given_v=True)
gumbel = Gumbel()


class AliMikailHaq(Archimedian, No90, No270):
    theta_start = .9,
    # theta_bounds = [(1e-9, 1.)]
    theta_bounds = [(-1 + 1e-6, 1. - 1e-6)]
    t, theta = sympy.symbols("t theta")
    # gen_expr = ln((1 - theta * (1 - t))) / t
    uu, vv = sympy.symbols("uu vv")
    cop_expr = uu * vv / (1 - theta * (1 - uu) * (1 - vv))
    cdf_given_uu_expr = ((vv * (1 - theta * (1 - vv))) *
                         (1 - theta * (1 - uu) * (1 - vv)) ** -2)
    # known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
alimikailhaq = AliMikailHaq()


class Gaussian(Copulae, NoRotations):
    par_names = "uu", "vv", "theta"
    theta_start = .75,
    theta_bounds = [(-1. + 1e-12, 1. - 1e-12)]
    symmetry1 = symmetry2 = True

    def dens_func(self, uu, vv, theta):
        uu, vv = np.atleast_1d(uu, vv)
        xx = spstats.norm.ppf(uu).reshape(uu.shape)
        yy = spstats.norm.ppf(vv).reshape(vv.shape)
        return ((1 - theta ** 2) ** (-.5) *
                np.exp(-.5 * (xx ** 2 + yy ** 2 - 2 * theta * xx * yy) *
                       (1 - theta ** 2)) *
                np.exp(0.5 * (xx ** 2 + yy ** 2)))

    def copula_func(self, uu, vv, theta):
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
                                      theta[i]
                                      )[1]
        if broadcast:
            quantiles = quantiles.reshape(uu.size, vv.size)
        return quantiles

    def cdf_given_u_(self, uu, vv, theta):
        theta = np.squeeze(theta)
        return (-0.5 * erf((theta *
                       erfinv(2 * uu - 1) -
                       erfinv(2 * vv - 1)) /
                      np.sqrt(-theta ** 2 + 1))
           + 0.5)

    def cdf_given_u(self, uu, vv, theta):
        qq = np.empty_like(uu)
        norm_cdf_given_u(uu, vv, theta, qq)
        return qq

    def inv_cdf_given_u_(self, uu, qq, theta):
        # theta = np.squeeze(theta)
        return (.5 * (1 +
                 erf(theta *
                     erfinv(2 * uu - 1) -
                     erfinv(-2 * (qq - .5)) *
                     np.sqrt(-theta ** 2 + 1))))

    def inv_cdf_given_u(self, uu, qq, theta):
        vv = np.empty_like(uu)
        norm_inv_cdf_given_u(uu, qq, theta, vv)
        return vv
    
    def cdf_given_v(self, uu, vv, theta):
        return self.cdf_given_u(vv, uu, theta)

    def fit(self, ranks_u, ranks_v, *args, **kwds):
        return np.corrcoef(ranks_u, ranks_v)[0, 1],
gaussian = Gaussian()


class Plackett(Copulae, NoRotations):
# class Plackett(Copulae):    
    par_names = "uu", "vv", "theta"
    theta_start = 2.,
    theta_bounds = [(1e-6, 20)]
    uu, vv, theta, eta = sympy.symbols(par_names + ("eta",))
    cop_expr = (1 / (2 * eta) * (1 + eta * (uu + vv) -
                                 sympy.sqrt((1 + eta * (uu + vv)) ** 2 -
                                            4 * theta * eta * uu * vv)))
    cop_expr = cop_expr.subs(eta, theta - 1)
    cdf_given_uu_expr = 0.5 - 0.5 * ((eta * uu + 1 - (eta + 2) * vv) /
                                     sympy.sqrt((1 + eta * (uu + vv)) ** 2 -
                                                4 * theta * eta * uu * vv))
    cdf_given_uu_expr = cdf_given_uu_expr.subs(eta, theta - 1)
    known_fail = "inv_cdf_given_u", "inv_cdf_given_v"
plackett = Plackett()


class Galambos(Copulae, NoRotations):
    """p. 174"""
    par_names = "uu", "vv", "delta"
    theta_start = 1.8,
    theta_bounds = [(.011, 50)]
    uu, vv, delta = sympy.symbols(par_names)
    cop_expr = uu * vv * exp(((-ln(uu)) ** -delta +
                              (-ln(vv)) ** -delta) ** (-1 / delta))
    x, y = sympy.symbols("x y")
    dens_expr = ((cop_expr / (uu * vv)) *
                 (1 -
                  (x ** -delta + y ** -delta) ** (-1 - 1 / delta) *
                  (x ** (-delta - 1) + y ** (-delta - 1)) +
                  (x ** -delta + y ** -delta) ** (-2 - 1 / delta) *
                  (x * y) ** (-delta - 1) *
                 (1 + delta + (x ** -delta + y ** -delta) ** (-1 / delta))))
    dens_expr = dens_expr.subs(dict(x=-ln(uu), y=-ln(vv)))
    # cdf_given_uu_expr = (vv *
    #                      exp((x ** -delta + y ** -delta) ** (-1 / delta)) *
    #                      (1 - (1 + (x / y) ** delta) ** (-1 - 1 / delta)))
    # cdf_given_vv_expr = cdf_given_uu_expr.subs(dict(x=-ln(vv), y=-ln(uu)))
    # cdf_given_uu_expr = cdf_given_uu_expr.subs(dict(x=-ln(uu), y=-ln(vv)))
    # cdf_given_vv_expr = swap_symbols(cdf_given_vv_expr, uu, vv)

    def __init__(self):
        x, y, p, delta = sympy.symbols("x y p delta")
        h_expr = (ln(p) + y -
                  (x ** -delta + y ** -delta) ** (-1 / delta) -
                  ln(1 - x ** (-delta - 1) *
                     (x ** -delta + y ** -delta) ** (-1 / delta - 1)))
        h_prime_expr = (1 - y ** (-delta - 1) *
                        (x ** -delta + y ** -delta) ** (-1 / delta - 1) +
                        (((1 + delta) *
                          x ** (-delta - 1) *
                          y ** (-delta - 1) *
                          (x ** -delta + y ** -delta) ** (-1 / delta - 2)) /
                         (1 - x ** (-delta - 1) *
                          (x ** -delta + y ** -delta) ** (-1 / delta - 1))))
        r = sympy.symbols("r")
        h1_expr = (ln(p) +
                   x * r ** (1 / delta) -
                   x * (1 + r ** -1) ** (-1 / delta) -
                   ln(1 - (1 + r ** -1) ** (-1 / delta - 1)))
        h1_prime_expr = (delta ** -1 * x * r ** (1 / delta - 1) -
                         delta ** -1 * x *
                         (1 + r ** -1) ** (-1 / delta - 1) * r ** -2 +
                         ((1 + delta ** -1) *
                          (1 + r ** -1) ** (-1 / delta - 2) * r ** -2) /
                         (1 - (1 + r ** -1) ** (-1 / delta - 1)))
        h_args = [y, x, p, delta]
        h1_args = [r, x, p, delta]
        h = ufuncify(self.__class__, "h", h_args, h_expr,
                     backend=self.backend)
        h_prime = ufuncify(self.__class__, "h_prime", h_args,
                           h_prime_expr, backend=self.backend)
        h1 = ufuncify(self.__class__, "h1", h1_args, h1_expr,
                      backend=self.backend)
        h1_prime = ufuncify(self.__class__, "h1_prime", h1_args,
                            h1_prime_expr, backend=self.backend)
        (self.h_u,
         self.h_prime_u,
         self.h1_u,
         self.h1_prime_u) = map(broadcast_1d, (h, h_prime, h1,
                                               h1_prime))
        h_expr = swap_symbols(h_expr, x, y)
        h_prime_expr = swap_symbols(h_prime_expr, x, y)
        h1_expr = swap_symbols(h1_expr, x, y)
        h1_prime_expr = swap_symbols(h1_prime_expr, x, y)
        h_args = [x, y, p, delta]
        h1_args = [r, y, p, delta]
        h = ufuncify(self.__class__, "h", h_args, h_expr,
                     backend=self.backend)
        h_prime = ufuncify(self.__class__, "h_prime", h_args,
                           h_prime_expr, backend=self.backend)
        h1 = ufuncify(self.__class__, "h1", h1_args, h1_expr,
                      backend=self.backend)
        h1_prime = ufuncify(self.__class__, "h1_prime", h1_args,
                            h1_prime_expr, backend=self.backend)
        (self.h_v,
         self.h_prime_v,
         self.h1_v,
         self.h1_prime_v) = map(broadcast_1d, (h, h_prime, h1,
                                               h1_prime))

    def _inverse_conditional(self, ranks, quantiles, theta,
                             given_v=False):
        theta = np.array(self.theta if theta is None else theta)
        ranks1, quantiles, thetas = np.atleast_1d(ranks, quantiles,
                                                  theta)
        quantiles, thetas = map(np.squeeze, (quantiles, thetas))
        if thetas.size == 1:
            thetas = np.full_like(ranks1, theta)
        if quantiles.size == 1:
            quantiles = np.full_like(ranks1, quantiles)
        ranks2 = np.empty_like(ranks1)

        if given_v:
            h, h_prime, h1, h1_prime = (self.h_v, self.h_prime_v,
                                        self.h1_v, self.h1_prime_v)
        else:
            h, h_prime, h1, h1_prime = (self.h_u, self.h_prime_u,
                                        self.h1_u, self.h1_prime_u)

        tol = zero

        def f(rank2, quantile):
            if given_v:
                rank_u, rank_v = rank2, rank1
                conditional_func = getattr(self, "cdf_given_v")
            else:
                rank_u, rank_v = rank1, rank2
                conditional_func = getattr(self, "cdf_given_u")
            quantile_calc = conditional_func(rank_u, rank_v, theta)
            if np.isnan(quantile_calc):
                if rank2 < tol:
                    quantile_calc = 0
                elif rank2 > (1 - tol):
                    quantile_calc = 1
                # if np.isclose(rank2, 0):
                #     quantile_calc = 0
                # elif np.isclose(rank2, 1):
                #     quantile_calc = 1
            return quantile_calc - quantile

        for i, rank1 in enumerate(ranks1):
            x = -np.log(rank1)
            try:
                y_root = newton(h, x0=x,
                                fprime=h_prime,
                                args=(x, quantiles[i], theta))
                ranks2[i] = np.exp(-float(y_root))
            except RuntimeError as exc:
                warnings.warn("Newton did not converge.")
                try:
                    r_root = newton(h1,
                                    x0=.5,
                                    fprime=h1_prime,
                                    args=(x, quantiles[i], theta))
                except RuntimeError:
                    ranks2[i] = brentq(f, 0, 1, args=(quantiles[i],))
                else:
                    ranks2[i] = x * float(r_root) ** (1 / theta)
        return ranks2

    def inv_cdf_given_u(self, ranks_u, quantiles, theta):
        return self._inverse_conditional(ranks_u, quantiles, theta)

    def inv_cdf_given_v(self, ranks_v, quantiles, theta):
        return self._inverse_conditional(ranks_v, quantiles, theta,
                                         given_v=True)
galambos = Galambos()


class Independence(Copulae, NoRotations):
    # having the theta here prevents trouble down the road...
    par_names = "uu", "vv", "theta"
    theta_start = None,
    uu, vv, _ = sympy.symbols(par_names)
    cop_expr = uu * vv
    known_fail = "inv_cdf_given_u", "inv_cdf_given_v"

    def fit(self, uu, vv, *args, **kwds):
        return 0

    def sample(self, size, *args, **kwds):
        return random_sample(size), random_sample(size)

    def cdf_given_u(self, uu, vv, *args):
        return uu

    def cdf_given_v(self, uu, vv, *args):
        return vv

    def inv_cdf_given_u(self, uu, qq, *args):
        return qq

    def inv_cdf_given_v(self, vv, qq, *args):
        return qq

    def generate_fitted(self, ranks_u, ranks_v, *args, **kwds):
        """Returns a Fitted instance that contains ranks_u, ranks_v and the
        fitted theta.
        """
        return Fitted(self, ranks_u, ranks_v, None)
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
        old_type = type(obj)
        new_name = "%s_%s" % (old_type.__name__, rot_str)
        # make the rotated copulas importable
        TurnedCop = type(new_name,
                         (old_type,) + old_type.__bases__,
                         dict(old_type.__dict__))
        turned_cops[new_name.lower()] = TurnedCop()
        globals()[new_name] = TurnedCop
        globals()[new_name.lower()] = turned_cops[new_name.lower()]
all_cops.update(turned_cops)
all_cops = OrderedDict((name, obj) for name, obj
                       in sorted(all_cops.items()))

frozen_cops = OrderedDict((name, copulas(copulas.theta_start))
                          for name, copulas in sorted(all_cops.items()))


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    # from weathercop import cop_conf
    # from weathercop import plotting as cplt

    frozen_cops["galambos"].sample(1000)

    # for frozen_cop in frozen_cops.values():
    #     frozen_cop.plot_cop_dens()
    # plt.show()

    # data_filepath = os.path.join(cop_conf.weathercop_dir, "code",
    #                              "vg_data.npz")
    # with np.load(data_filepath) as saved:
    #     data_summer = saved["summer"]
    # ranks_u_tm1 = stats.rel_ranks(data_summer[5, :-1])
    # ranks_rh = stats.rel_ranks(data_summer[4, 1:])

    # for copula in all_cops.values():
    #     # copula.plot_cop_dens()
    #     # copula.plot_copula()
    #     # copula.plot_density()
    #     # plt.title(copula.name)
    #     try:
    #         fitted_cop = copula.generate_fitted(ranks_u_tm1, ranks_rh,
    #                                             # method="TNC",
    #                                             verbose=False)
    #     except NoConvergence:
    #         print("No convergence for %s" % copula.name)
    #         continue
    #     fig, axs = plt.subplots(ncols=2, subplot_kw=dict(aspect="equal"))
    #     fig.suptitle(copula.name + " " +
    #                  repr(fitted_cop.theta) +
    #                  "\n likelihood: %.2f" % fitted_cop.likelihood)
    #     opacity = .1
    #     fitted_cop.plot_density(ax=axs[0], opacity=opacity,
    #                             scatter=True, sample_size=10000,
    #                             kind="contour")
    #     cplt.hist2d(ranks_u_tm1, ranks_rh, ax=axs[1],
    #                 # kind="contourf",
    #                 scatter=False)
    #     axs[1].scatter(ranks_u_tm1, ranks_rh,
    #                    marker="o", facecolors=(0, 0, 0, 0),
    #                    edgecolors=(0, 0, 0, opacity))
    # plt.show()
