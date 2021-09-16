# -*- coding: utf-8 -*-
import functools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as spstats
from weathercop import stats


def rel_ranks(data, method="average"):
    return (spstats.rankdata(data, method) - 0.5) / len(data)


def cache(*names, **name_values):
    """Use as a decorator, to supply *names attributes that can be used as
    a cache. The attributes are set to None during compile time. The
    wrapped function also has a 'clear_cache'-method to delete those
    variables.

    Parameter
    ---------
    *names : str
    """

    def wrapper(function):
        @functools.wraps(function)
        def cache_holder(*args, **kwds):
            return function(*args, **kwds)

        cache_holder._cache_names = names
        cache_holder._cache_name_values = name_values
        cache_holder.clear_cache = lambda: clear_def_cache(cache_holder)
        cache_holder.clear_cache()
        return cache_holder

    return wrapper


def clear_def_cache(function, cache_names=None, cache_name_values=None):
    """I often use a simplified function cache in the form of
    'function.attribute = value'.  This function helps cleaning it up,
    i.e. setting them to None.

    Parameter
    ---------
    function : object with settable attributes
    cache_names : sequence of str or None, optional
        if None, function should have an attribute called _cache_names with
        names of attributes that are cached.
    """
    if cache_names is None:
        cache_names = function._cache_names
    if cache_name_values is None:
        cache_name_values = function._cache_name_values
    for name in cache_names:
        setattr(function, name, None)
    for name, value in cache_name_values.items():
        setattr(function, name, value)


def ccplom(
    data,
    k=0,
    kind="img",
    transform=False,
    varnames=None,
    h_kwds=None,
    s_kwds=None,
    title=None,
    opacity=0.1,
    cmap=None,
    x_bins=15,
    y_bins=15,
    display_rho=True,
    display_asy=True,
    vmax_fct=1.0,
    fontsize=None,
    fontcolor="yellow",
    scatter=True,
    axs=None,
    fig=None,
    **fig_kwds
):
    """Cross-Copula-plot matrix. Values that appear on the x-axes are shifted
    back k timesteps. Data is assumed to be a 2 dim arrays with
    observations in rows."""
    if transform:
        ranks = np.array([stats.rel_ranks(values) for values in data])
    else:
        ranks = np.asarray(data)
    K, T = data.shape
    h_kwds = {} if h_kwds is None else h_kwds
    s_kwds = {} if s_kwds is None else s_kwds
    if fontsize is None:
        fontsize = mpl.rcParams["xtick.labelsize"]
    if varnames is None:
        n_variables = data.shape[0]
        varnames = [str(i) for i in range(n_variables)]
    else:
        n_variables = len(varnames)

    if n_variables == 2:
        # two variables don't need a plot matrix
        n_variables = 1

    if fig is None:
        fig, axs = plt.subplots(
            n_variables,
            n_variables,
            subplot_kw=dict(aspect="equal"),
            **fig_kwds
        )
    if n_variables == 1:
        axs = ((axs,),)

    x_slice = slice(None, None if k == 0 else -k)
    y_slice = slice(k, None)
    for ii in range(n_variables):
        for jj in range(n_variables):
            ax = axs[ii][jj]
            if n_variables == 1:
                jj = 1
            if ii == jj and n_variables > 1:
                ax.set_axis_off()
                continue
            ranks_x = ranks[jj, x_slice]
            ranks_y = ranks[ii, y_slice]
            hist2d(
                ranks_x,
                ranks_y,
                x_bins,
                y_bins,
                ax=ax,
                cmap=cmap,
                scatter=False,
                kind=kind,
            )
            if scatter:
                ax.scatter(
                    ranks_x,
                    ranks_y,
                    marker="o",
                    facecolors=(0, 0, 0, 0),
                    edgecolors=(0, 0, 0, opacity),
                    **s_kwds
                )
            if display_rho:
                rho = stats.spearmans_rank(ranks_x, ranks_y)
                ax.text(
                    0.5,
                    0.5,
                    r"$\rho = %.3f$" % rho,
                    fontsize=fontsize,
                    color=fontcolor,
                    horizontalalignment="center",
                )
            if display_asy:
                asy1 = stats.asymmetry1(ranks_x, ranks_y)
                asy2 = stats.asymmetry2(ranks_x, ranks_y)
                ax.text(
                    0.5,
                    0.75,
                    r"$a_1 = %.3f$" % asy1,
                    fontsize=fontsize,
                    color=fontcolor,
                    horizontalalignment="center",
                )
                ax.text(
                    0.5,
                    0.25,
                    r"$a_2 = %.3f$" % asy2,
                    fontsize=fontsize,
                    color=fontcolor,
                    horizontalalignment="center",
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(False)
            ax.set_yticklabels("")
            ax.set_xticklabels("")
            if (jj == 0) or (jj == 1 and ii == 0):
                ax.set_ylabel(varnames[ii] + ("(t)" if k else ""))
            K = n_variables
            if (ii == K - 1) or (ii == K - 2 and jj == K - 1):
                ax.set_xlabel(varnames[jj] + (("(t-%d)" % k) if k else ""))
    # reset the vlims, so that we have the same color scale in all plots
    for ax in np.ravel(axs):
        for im in ax.get_images():
            im.set_clim(vmax=vmax_fct * hist2d.h_max)
    if title:
        plt.suptitle(title)
    else:
        plt.suptitle("k = %d" % k)
    hist2d.clear_cache()
    return fig, axs


@cache(h_max=-np.inf)
def hist2d(
    x,
    y,
    n_xbins=15,
    n_ybins=15,
    kind="img",
    ax=None,
    cmap=None,
    scatter=True,
    opacity=0.6,
    vmax=None,
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    if cmap is None:
        # cmap = plt.get_cmap("coolwarm")
        cmap = plt.get_cmap("terrain")
    H, xedges, yedges = np.histogram2d(x, y, (n_xbins, n_ybins), density=True)
    # if this histogram is part of a plot-matrix, the plot-matrix
    # might want to set vmax to a common value.  expose h_max to the
    # outside here as a function attribute for that reason.
    h_max = np.max(H)
    if h_max > hist2d.h_max:
        hist2d.h_max = h_max
    if kind.startswith("contour"):
        x_bins = 0.5 * (xedges[1:] + xedges[:-1])
        y_bins = 0.5 * (yedges[1:] + yedges[:-1])
        if kind == "contourf":
            ax.contourf(x_bins, y_bins, H.T, cmap=cmap, vmin=0, vmax=vmax)
        elif kind == "contour":
            ax.contour(x_bins, y_bins, H.T, cmap=cmap, vmin=0, vmax=vmax)
    elif kind == "img":
        ax.imshow(
            H.T,
            extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
            origin="left",
            aspect="equal",
            interpolation="none",
            cmap=cmap,
            vmin=0,
            vmax=vmax,
        )
        x = (x - x.min() - 0.5 / n_xbins) * n_xbins
        y = (y - y.min() - 0.5 / n_ybins) * n_ybins
    if scatter:
        ax.scatter(
            x,
            y,
            marker="o",
            facecolors=(0, 0, 0, 0),
            edgecolors=(0, 0, 0, opacity),
        )
    return fig, ax


def plot_cross_corr(
    data,
    varnames=None,
    max_lags=10,
    figsize=None,
    fig=None,
    axs=None,
    *args,
    **kwds
):
    K = data.shape[0]
    if varnames is None:
        varnames = np.arange(K).astype(str)
    lags = np.arange(max_lags)
    # shape: (max_lags, K, K)
    cross_corrs = np.array([cross_corr(data, k) for k in lags])
    if fig is None and axs is None:
        size = {}
        if figsize:
            size["figsize"] = figsize
        fig, axs = plt.subplots(K, squeeze=True, **size)
    for var_i in range(K):
        lines = []
        for var_j in range(K):
            # want to set the same colors as before when called with a given
            # fig and axs
            colors = plt.rcParams["axes.prop_cycle"]
            color = colors[var_j % len(colors)]
            lines += axs[var_i].plot(
                lags, cross_corrs[:, var_i, var_j], color=color, *args, **kwds
            )
        axs[var_i].set_title(varnames[var_i])
        axs[var_i].grid(True)
    plt.subplots_adjust(right=0.75, hspace=0.25)
    fig.legend(lines, varnames, loc="center right")
    return fig, axs


def cross_corr(data, k):
    """Return the cross-correlation-coefficient matrix for lag k. Variables are
    assumed to be stored in rows, with time extending across the columns."""
    finite_ii = np.isfinite(data)
    stds = [np.std(row[row_ii]) for row, row_ii in zip(data, finite_ii)]
    stds = np.array(stds)[:, np.newaxis]
    stds_dot = stds * stds.T  # dyadic product of row-vector stds
    return cross_cov(data, k) / stds_dot


def nanavg(x, axis=None, ddof=0):
    ii = np.isfinite(x)
    return np.nansum(x, axis=axis) / (np.sum(ii, axis=axis) - ddof)


def cross_cov(data, k, means=None):
    """Return the cross-covariance matrix for lag k. Variables are assumed to
    be stored in rows, with time extending across the columns."""
    n_vars = data.shape[0]
    k_right = -abs(k) if k else None
    if means is None:
        means = nanavg(data, axis=1).reshape((n_vars, 1))
        ddof = 1
    else:
        ddof = 0
    cross = np.empty((n_vars, n_vars))
    for ii in range(n_vars):
        for jj in range(n_vars):
            cross[ii, jj] = nanavg(
                (data[ii, :k_right] - means[ii]) * (data[jj, k:] - means[jj]),
                ddof=ddof,
            )
    return cross
