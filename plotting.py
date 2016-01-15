# -*- coding: utf-8 -*-
import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def rel_ranks(data, method="average"):
    return (stats.rankdata(data, method) - .5) / len(data)


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


def ccplom(data, k=1, variable_names=None, h_kwds=None, s_kwds=None,
           title=None, opacity=.1, cmap=None, x_bins=20, y_bins=20,
           display_rho=True, display_asy=True, vmax_fct=1.,
           **fig_kwds):
    """Cross-Copula-plot matrix. Values that appear on the x-axes are shifted
    back k timesteps. Data is assumed to be a 2 dim arrays with
    observations in rows."""
    data = np.asarray(data)
    K, T = data.shape
    h_kwds = {} if h_kwds is None else h_kwds
    s_kwds = {} if s_kwds is None else s_kwds
    n_variables = data.shape[0]
    fig, axes = plt.subplots(n_variables, n_variables,
                             subplot_kw=dict(aspect="equal"),
                             **fig_kwds)
    ranks = np.array([rel_ranks(var) for var in data])
    x_slice = slice(None, None if k == 0 else -k)
    y_slice = slice(k, None)
    for ii in range(n_variables):
        for jj in range(n_variables):
            ax = axes[ii, jj]
            ranks_x = ranks[jj, x_slice]
            ranks_y = ranks[ii, y_slice]
            hist2d(ranks_x, ranks_y, x_bins, y_bins,
                   ax=ax, cmap=cmap, scatter=False)
            ax.scatter(ranks_x, ranks_y,
                       marker="o", facecolors=(0, 0, 0, 0),
                       edgecolors=(0, 0, 0, opacity), **s_kwds)
            if display_rho:
                rho = spearmans_rank(ranks_x, ranks_y)
                ax.text(.5, .5, r"$\rho = %.3f$" % rho,
                        horizontalalignment="center")
            if display_asy:
                asy1 = asymmetry1(ranks_x, ranks_y)
                asy2 = asymmetry2(ranks_x, ranks_y)
                ax.text(.5, .75, r"$a_1 = %.3f$" % asy1,
                        horizontalalignment="center")
                ax.text(.5, .25, r"$a_2 = %.3f$" % asy2,
                        horizontalalignment="center")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(False)
            # show ticklabels only on the margins
            if jj != 0:
                ax.set_yticklabels("")
            if ii != n_variables - 1:
                ax.set_xticklabels("")
            if jj == 0:
                ax.set_ylabel(variable_names[ii] + "(t)")
            if ii == n_variables - 1:
                ax.set_xlabel(variable_names[jj] + "(t-%d)" % k)
    # reset the vlims, so that we have the same color scale in all plots
    for ax in np.ravel(axes):
        for im in ax.get_images():
            im.set_clim(vmax=vmax_fct * hist2d.h_max)
    if title:
        plt.suptitle(title)
    else:
        plt.suptitle("k = %d" % k)
    hist2d.clear_cache()
    return fig, axes


@cache(h_max=-np.inf)
def hist2d(x, y, n_xbins=15, n_ybins=15, kind="img", ax=None, cmap=None,
           scatter=True, opacity=.6, vmax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    if cmap is None:
        cmap = plt.get_cmap("coolwarm")
    H, xedges, yedges = np.histogram2d(x, y, (n_xbins, n_ybins), normed=True)
    # if this histogram is part of a plot-matrix, the plot-matrix
    # might want to set vmax to a common value.  expose h_max to the
    # outside here as a function attribute for that reason.
    h_max = np.max(H)
    if h_max > hist2d.h_max:
        hist2d.h_max = h_max
    if kind == "contourf":
        x_bins = .5 * (xedges[1:] + xedges[:-1])
        y_bins = .5 * (yedges[1:] + yedges[:-1])
        ax.contourf(x_bins, y_bins, H.T, cmap=cmap, vmin=0, vmax=vmax)
    elif kind == "img":
        ax.imshow(H.T,
                  extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]),
                  origin="left", aspect="equal", interpolation="none",
                  cmap=cmap, vmin=0, vmax=vmax)
        x = (x - x.min() - .5 / n_xbins) * n_xbins
        y = (y - y.min() - .5 / n_ybins) * n_ybins
    if scatter:
        ax.scatter(x, y, marker="o", facecolors=(0, 0, 0, 0),
                   edgecolors=(0, 0, 0, opacity))
    return fig, ax
