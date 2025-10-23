# -*- coding: utf-8 -*-
from collections import namedtuple
import functools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as spstats
import xarray as xr
from weathercop import stats
from varwg import helpers as my


def suptitle_prepend(fig, name):
    if isinstance(fig, mpl.figure.Figure):
        try:
            suptitle = fig._suptitle.get_text()
        except AttributeError:
            suptitle = ""
    fig.suptitle(f"{name} {suptitle}")


def rel_ranks(data, method="average"):
    if isinstance(data, int):
        N = data
        return (np.arange(N) + 0.5) / N
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
    alpha=0.1,
    cmap=None,
    x_bins=15,
    y_bins=15,
    display_rho=True,
    display_asy=True,
    vmax_fct=1.0,
    fontsize=None,
    fontcolor="black",
    scatter=True,
    axs=None,
    fig=None,
    **fig_kwds,
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
            **fig_kwds,
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
                    edgecolors=(0, 0, 0, alpha),
                    **s_kwds,
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
    alpha=0.6,
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
            origin="lower",
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
            edgecolors=(0, 0, 0, alpha),
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
    **kwds,
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


BoundKuglitsch = namedtuple("BoundKuglitsch", ("day", "night"))
HeatWavesResult = namedtuple(
    "HeatWavesResult",
    ("mask", "begin_end", "bound"),
)


class HeatWaves:
    def __init__(self, data, reference=None, month_start=None, month_end=None):
        self.month_start, self.month_end = month_start, month_end
        self.data = data
        self.reference = data if reference is None else reference
        self._data_day = self._data_night = None
        self._reference_day = self._reference_night = None

    @property
    def data_day(self):
        if self._data_day is None:
            self._data_day = self.data.resample(time="D").max()
        return self._data_day

    @property
    def data_night(self):
        if self._data_night is None:
            self._data_night = self.data.resample(time="D").min()
        return self._data_night

    @property
    def reference_day(self):
        if self._reference_day is None:
            self._reference_day = self.reference.resample(time="D").max()
        return self._reference_day

    @property
    def reference_night(self):
        if self._reference_night is None:
            self._reference_night = self.reference.resample(time="D").min()
        return self._reference_night

    def _filter_min_length(self, mask, min_length):
        begin_end = my.gaps(mask.values)
        begin_end[:, 1] += 1
        heat_wave_lenghts = np.diff(begin_end, axis=1)
        begin_end = begin_end[(heat_wave_lenghts >= min_length)[:, 0]]
        hot_mask = xr.full_like(mask, False)
        for begin, end in begin_end:
            hot_mask.values[begin:end] = True
        return hot_mask, begin_end


class HeatWavesN(HeatWaves):
    def __init__(
        self,
        data,
        quantile=0.9,
        min_length=3,
        month_start=5,
        month_end=9,
        n_per_summer=3,
        bound=None,
        **kwds,
    ):
        super().__init__(
            data, reference=None, month_start=month_start, month_end=month_end
        )
        mask = xr.full_like(data.time, False, dtype=bool)
        for year, group in self.data_day.groupby("time.year"):
            months = group.time.dt.month
            month_mask = (months >= month_start) & (months <= month_end)
            if not month_mask.any():
                continue
            tri = (
                group.sel(time=month_mask)
                .rolling(time=min_length, center=True)
                .sum("time")
                .dropna("time")
            )
            sort_ii = list(np.argsort(tri.values))
            waves_ii = [sort_ii.pop()]
            while len(waves_ii) < n_per_summer:
                while (
                    np.min(
                        np.abs(
                            (candidate := sort_ii.pop()) - np.array(waves_ii)
                        )
                    )
                    <= min_length
                ):
                    pass
                waves_ii += [candidate]
            # fig, ax = plt.subplots(nrows=1, ncols=1)
            # ax.plot(group.time, group.values)
            # ax.plot(tri.time, tri.values)
            # for i in waves_ii:
            #     ax.axvline(tri.time[i].values, color="k", linestyle="--")
            # ax.grid()
            # plt.show()
            # expand to get the whole heat waves into the mask
            waves_ii = np.array(waves_ii)
            waves_ii = (
                waves_ii[:, None] + np.arange(min_length) - min_length // 2
            ).ravel()
            waves_ii = waves_ii[(waves_ii >= 0) & (waves_ii < len(group))]
            mask.loc[dict(time=group.time.isel(time=waves_ii))] = True
        self.mask = mask


class HeatWavesZitis(HeatWaves):
    def __init__(
        self,
        data,
        reference=None,
        quantile=0.9,
        window_len=5,
        min_length=3,
        month_start=5,
        month_end=9,
        bound=None,
    ):
        super().__init__(data, reference, month_start, month_end)
        if bound is None:
            if "realization" in self.reference.dims:
                bound_per_doy = (
                    self.reference_day.chunk(dict(time=-1, realization=-1))
                    .groupby("time.dayofyear")
                    .quantile(quantile, dim=("time", "realization"))
                    .load()
                )
            else:
                bound_per_doy = (
                    self.reference_day.rolling(time=window_len, center=True)
                    .construct("rolled")
                    .groupby("time.dayofyear")
                    .quantile(quantile, dim=("time", "rolled"))
                )
            bound = (
                xr.full_like(self.data_day, 0).groupby("time.dayofyear")
                + bound_per_doy
            ).load()
        self.bound = bound
        hot_days = self.data_day > bound
        months = self.data_day.time.dt.month
        mask = hot_days & (months >= month_start) & (months <= month_end)
        self.mask, self.begin_end = self._filter_min_length(mask, min_length)

    def plot(self, fig=None, axs=None, individual_wave_filepath=None):
        K = len(self.data.coords["variable"])
        if fig is None or axs is None:
            fig, axs = plt.subplots(
                nrows=K,
                ncols=1,
                sharex=True,
                constrained_layout=True,
                figsize=(10, 2 * K),
            )
        p_kwds = dict(linestyle="--", alpha=0.5, color="k")
        varnames = list(self.data.coords["variable"].values)
        # theta_data = self.data.sel(variable="theta").load()
        # theta_data_day = theta_data.resample(time="D").max()
        # theta_data_night = theta_data.resample(time="D").min()
        theta_data_day = self.data_day.sel(variable="theta")
        for varname, ax in zip(varnames, axs):
            values = self.data.sel(variable=varname)
            if varname == "theta":
                ax.plot(theta_data_day.time, theta_data_day, **p_kwds)
            else:
                ax.plot(values.time, values, **p_kwds)
            ax.set_title(varname)
            ax.grid(True)
        axs[varnames.index("theta")].plot(
            self.bound_day.time, self.bound_day.values, color="k"
        )
        for ax_i, ax in enumerate(axs):
            for left, right in self.begin_end:
                if ax_i == varnames.index("theta"):
                    ax.fill_between(
                        self.mask.time[left:right],
                        self.bound_day[left:right].values,
                        theta_data_day[left:right].values,
                        color="r",
                        alpha=0.5,
                    )
                else:
                    ax.axvspan(
                        self.mask.time[left].values,
                        self.mask.time[right - 1].values,
                        color="r",
                        alpha=0.5,
                    )
        if individual_wave_filepath is not None:
            individual_wave_filepath.parent.mkdir(exist_ok=True, parents=True)
            for year, yearly_mask in self.mask.groupby("time.year"):
                if not np.any(yearly_mask.values):
                    print(f"no heatwaves in {year}")
                    continue
                axs[0].set_xlim(
                    yearly_mask.time.sel(
                        time=f"{year}-{self.month_start:02d}-01",
                        # time=f"{year}-01-01",
                        method="nearest",
                    ).values,
                    yearly_mask.time.sel(
                        time=f"{year}-{self.month_end:02d}-30",
                        # time=f"{year}-12-31",
                        method="nearest",
                    ).values,
                )
                fig.savefig(
                    individual_wave_filepath.parent
                    / (individual_wave_filepath.stem + f"_{year}.png")
                )
        return fig, axs


class HeatwavesKuglitsch(HeatWaves):
    def __init__(
        self,
        data,
        reference=None,
        quantile=0.95,
        window_len=15,
        min_length=3,
        month_start=6,
        month_end=9,
        bound=None,
    ):
        super().__init__(data, reference, month_start, month_end)
        if bound is None:
            if "realization" in self.reference.dims:
                bound_per_doy_day = (
                    self.reference_day.chunk(dict(time=-1, realization=-1))
                    .groupby("time.dayofyear")
                    .quantile(quantile, dim=("time", "realization"))
                    .load()
                )
                bound_per_doy_night = (
                    self.reference_night
                    # .persist()
                    .chunk(dict(time=-1, realization=-1))
                    .groupby("time.dayofyear")
                    .quantile(quantile, dim=("time", "realization"))
                    .load()
                )
            else:
                bound_per_doy_day = (
                    self.reference_day.rolling(time=window_len, center=True)
                    .construct("rolled")
                    .groupby("time.dayofyear")
                    .quantile(quantile, dim=("time", "rolled"))
                )
                bound_per_doy_night = (
                    self.reference_night.rolling(time=window_len, center=True)
                    .construct("rolled")
                    .groupby("time.dayofyear")
                    .quantile(quantile, dim=("time", "rolled"))
                )
            bound_day = (
                xr.full_like(self.data_day, 0).groupby("time.dayofyear")
                + bound_per_doy_day
            ).load()
            bound_night = (
                xr.full_like(self.data_night, 0).groupby("time.dayofyear")
                + bound_per_doy_night
            ).load()
        self.bound = BoundKuglitsch(day=bound_day, night=bound_night)
        hot_days = self.data_day > self.bound.day
        hot_nights = self.data_night > self.bound.night
        months = self.data_day.time.dt.month
        mask = (
            hot_days
            & hot_nights
            & (months >= month_start)
            & (months <= month_end)
        )
        # one normal day/night in-between a hot day/night does not
        # interrupt the heat-wave
        mask_tri = (
            mask.rolling(time=3, center=True).construct("tri")
            == [True, False, True]
        ).all("tri")
        mask |= mask_tri
        self.mask, self.begin_end = self._filter_min_length(mask, min_length)
        # self.result = HeatWavesResult(
        #     mask=hot_mask,
        #     begin_end=begin_end,
        #     bound=bound
        #     # bound_day=bound_day,
        #     # bound_night=bound_night,
        # )

    def plot(self, fig=None, axs=None, individual_wave_filepath=None):
        K = len(self.data.coords["variable"])
        if fig is None or axs is None:
            fig, axs = plt.subplots(
                nrows=K,
                ncols=1,
                sharex=True,
                constrained_layout=True,
                figsize=(10, 2 * K),
            )
        p_kwds = dict(linestyle="--", alpha=0.5, color="k")
        varnames = list(self.data.coords["variable"].values)
        # theta_data = self.data.sel(variable="theta").load()
        # theta_data_day = theta_data.resample(time="D").max()
        # theta_data_night = theta_data.resample(time="D").min()
        theta_data_day = self.data_day.sel(variable="theta")
        theta_data_night = self.data_night.sel(variable="theta")
        for varname, ax in zip(varnames, axs):
            values = self.data.sel(variable=varname)
            if varname == "theta":
                ax.plot(theta_data_day.time, theta_data_day, **p_kwds)
                ax.plot(theta_data_night.time, theta_data_night, **p_kwds)
            else:
                ax.plot(values.time, values, **p_kwds)
            ax.set_title(varname)
            ax.grid(True)
        axs[varnames.index("theta")].plot(
            self.bound_day.time, self.bound_day.values, color="k"
        )
        axs[varnames.index("theta")].plot(
            self.bound_night.time,
            self.bound_night.values,
            color="k",
        )
        for ax_i, ax in enumerate(axs):
            for left, right in self.begin_end:
                if ax_i == varnames.index("theta"):
                    ax.fill_between(
                        self.mask.time[left:right],
                        self.bound_day[left:right].values,
                        theta_data_day[left:right].values,
                        color="r",
                        alpha=0.5,
                    )
                    ax.fill_between(
                        self.mask.time[left:right],
                        self.bound_night[left:right].values,
                        theta_data_night[left:right].values,
                        color="r",
                        alpha=0.5,
                    )
                else:
                    ax.axvspan(
                        self.mask.time[left].values,
                        self.mask.time[right - 1].values,
                        color="r",
                        alpha=0.5,
                    )
        if individual_wave_filepath is not None:
            individual_wave_filepath.parent.mkdir(exist_ok=True, parents=True)
            for year, yearly_mask in self.mask.groupby("time.year"):
                if not np.any(yearly_mask.values):
                    print(f"no heatwaves in {year}")
                    continue
                axs[0].set_xlim(
                    yearly_mask.time.sel(
                        time=f"{year}-{self.month_start:02d}-01",
                        # time=f"{year}-01-01",
                        method="nearest",
                    ).values,
                    yearly_mask.time.sel(
                        time=f"{year}-{self.month_end:02d}-30",
                        # time=f"{year}-12-31",
                        method="nearest",
                    ).values,
                )
                fig.savefig(
                    individual_wave_filepath.parent
                    / (individual_wave_filepath.stem + f"_{year}.png")
                )
        return fig, axs


def heatwaves_kuglitsch(
    data,
    reference=None,
    quantile=0.95,
    window_len=15,
    min_length=3,
    bound_day=None,
    bound_night=None,
):
    """Detect eastern mediterranean heat-waves according to Kuglitsch et a. 2010.

    Parameter:
    ----------
    data : xarray DataArray
        Hourly air temperature data.
    reference : xarray DataArray or None, optional
        Hourly air temperature data used to determine quantiles for
        hot days/nights.
        If None, data (first paramter) will be used.
    quantile : float, optional
        Quantile for determining hot days/nights bounds.
    window_len : int, optional
        Window lengths around each doy for determining hot days/nights bounds.
        If reference has a 'realization' dimension, window_len is
        ignored and bounds are calculated using all realizations for
        each doy instead.
    min_length : int, optional
        Minimum lengths of a heat waves [days].

    Returns:
    --------
    heat_waves_result : HeatWavesResult namedtuple
        Attributes:
            - mask : xarray DataArray
              boolean array masking timesteps that are part of a heat wave
            - begin_end : 2d numpy ndarray of ints
              Indices of begin and end of heat waves. Row: [begin,
              end]. 'end' index is inclusive.
            - bound_day :
    """
    if reference is None:
        reference = data
    data_day = data.resample(time="D").max()
    data_night = data.resample(time="D").min()
    if bound_day is None or bound_night is None:
        reference_day = reference.resample(time="D").max()
        reference_night = reference.resample(time="D").min()
        if "realization" in reference.dims:
            bound_per_doy_day = (
                reference_day
                # .persist()
                .chunk(dict(time=-1, realization=-1))
                .groupby("time.dayofyear")
                .quantile(quantile, dim=("time", "realization"))
                .load()
            )
            bound_per_doy_night = (
                reference_night
                # .persist()
                .chunk(dict(time=-1, realization=-1))
                .groupby("time.dayofyear")
                .quantile(quantile, dim=("time", "realization"))
                .load()
            )
        else:
            bound_per_doy_day = (
                reference_day.rolling(time=window_len, center=True)
                .construct("rolled")
                .groupby("time.dayofyear")
                .quantile(quantile, dim=("time", "rolled"))
            )
            bound_per_doy_night = (
                reference_night.rolling(time=window_len, center=True)
                .construct("rolled")
                .groupby("time.dayofyear")
                .quantile(quantile, dim=("time", "rolled"))
            )
        bound_day = (
            xr.full_like(data_day, 0).groupby("time.dayofyear")
            + bound_per_doy_day
        ).load()
        bound_night = (
            xr.full_like(data_night, 0).groupby("time.dayofyear")
            + bound_per_doy_night
        ).load()
    hot_days = data_day > bound_day
    hot_nights = data_night > bound_night
    months = data_day.time.dt.month
    hot_mask = hot_days & hot_nights & (months >= 6) & (months <= 9)
    # one normal day/night in-between a hot day/night does not
    # interrupt the heat-wave
    hot_mask_tri = (
        hot_mask.rolling(time=3, center=True).construct("tri")
        == [True, False, True]
    ).all("tri")
    hot_mask |= hot_mask_tri
    begin_end = my.gaps(hot_mask.values)
    begin_end[:, 1] += 1
    heat_wave_lenghts = np.diff(begin_end, axis=1)
    begin_end = begin_end[(heat_wave_lenghts >= min_length)[:, 0]]
    hot_mask = xr.full_like(hot_mask, False)
    for begin, end in begin_end:
        hot_mask.values[begin:end] = True
    return HeatWavesResult(
        mask=hot_mask,
        begin_end=begin_end,
        bound=BoundKuglitsch(day=bound_day, night=bound_night),
        # bound_day=bound_day,
        # bound_night=bound_night,
    )


def plot_heatwaves(
    data,
    reference_data=None,
    quantile=0.95,
    window_len=15,
    min_length=3,
    fig=None,
    axs=None,
    heat_waves=None,
    # bound_day=None,
    # bound_night=None,
    bound=None,
    individual_wave_filepath=None,
    definition="kuglitsch",
    *args,
    **kwds,
):
    if reference_data is None:
        reference_data = data
    match definition:
        case "kuglitsch":
            heatwaves_detection = heatwaves_kuglitsch
        case "zitis":
            heatwaves_detection = heatwaves_zitis
        case _:
            raise RuntimeError(
                f"{definition=} not understood. Must be 'kuglitsch' or 'zitis'"
            )
    K = len(data.coords["variable"])
    if fig is None or axs is None:
        fig, axs = plt.subplots(
            nrows=K,
            ncols=1,
            sharex=True,
            constrained_layout=True,
            figsize=(10, 2 * K),
        )
    p_kwds = dict(linestyle="--", alpha=0.5, color="k")
    varnames = list(data.coords["variable"].values)
    theta_data = data.sel(variable="theta").load()
    theta_data_day = theta_data.resample(time="D").max()
    if definition == "kuglitsch":
        theta_data_night = theta_data.resample(time="D").min()
    for varname, ax in zip(varnames, axs):
        values = data.sel(variable=varname)
        if varname == "theta":
            ax.plot(theta_data_day.time, theta_data_day, **p_kwds)
            if definition == "kuglitsch":
                ax.plot(theta_data_night.time, theta_data_night, **p_kwds)
        else:
            ax.plot(values.time, values, **p_kwds)
        ax.set_title(varname)
        ax.grid(True)
    if heat_waves is None:
        # heat_waves = heatwaves_kuglitsch(
        #     theta_data,
        #     reference_data.sel(variable="theta"),
        #     bound=bound,
        #     # bound_day=bound_day,
        #     # bound_night=bound_night,
        # )
        heat_waves = HeatwavesKuglitsch(
            theta_data, reference_data.sel(variable="theta"), bound=bound
        )
    axs[varnames.index("theta")].plot(
        heat_waves.bound.day.time, heat_waves.bound.day.values, color="k"
    )
    axs[varnames.index("theta")].plot(
        heat_waves.bound.night.time, heat_waves.bound.night.values, color="k"
    )
    for ax_i, ax in enumerate(axs):
        for left, right in heat_waves.begin_end:
            if ax_i == varnames.index("theta"):
                ax.fill_between(
                    heat_waves.mask.time[left:right],
                    heat_waves.bound.day[left:right].values,
                    theta_data_day[left:right].values,
                    color="r",
                    alpha=0.5,
                )
                ax.fill_between(
                    heat_waves.mask.time[left:right],
                    heat_waves.bound.night[left:right].values,
                    theta_data_night[left:right].values,
                    color="r",
                    alpha=0.5,
                )
            else:
                ax.axvspan(
                    heat_waves.mask.time[left].values,
                    heat_waves.mask.time[right - 1].values,
                    color="r",
                    alpha=0.5,
                )
    if individual_wave_filepath is not None:
        individual_wave_filepath.parent.mkdir(exist_ok=True, parents=True)
        for year, yearly_mask in heat_waves.mask.groupby("time.year"):
            if not np.any(yearly_mask.values):
                print(f"no heatwaves in {year}")
                continue
            axs[0].set_xlim(
                yearly_mask.time.sel(
                    time=f"{year}-06-01",
                    # time=f"{year}-01-01",
                    method="nearest",
                ).values,
                yearly_mask.time.sel(
                    time=f"{year}-09-30",
                    # time=f"{year}-12-31",
                    method="nearest",
                ).values,
            )
            fig.savefig(
                individual_wave_filepath.parent
                / (individual_wave_filepath.stem + f"_{year}.png")
            )
    return fig, axs
