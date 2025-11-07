import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np

from weathercop import (
    cop_conf,
    copulae,
    stats,
)
from weathercop import plotting as wplt
from weathercop import seasonal_cop as scop


def generate_fitted(args, verbose=False, **kwds):
    name, cop, ranks_u, ranks_v, fit_mask = args
    try:
        fitted_cop = cop.generate_fitted(
            ranks_u, ranks_v, fit_mask=fit_mask, **kwds
        )
        return fitted_cop
    except copulae.NoConvergence:
        if verbose > 1:
            print("No fit for %s" % cop)
        return -np.inf


def mml(
    ranks_u,
    ranks_v,
    cop_candidates=copulae.all_cops,
    dtimes=None,
    verbose=False,
    plot=False,
    asymmetry=False,
    fit_mask=None,
    scop_kwds=None,
    **kwds,
):
    """Find fitted copula with maximum likelihood.

    Parameter
    ---------
    ranks_u : (T,) float array
    ranks_v : (T,) float array
    cop_candidates : mapping of str to copulae.Copulae classes, optional
    dtimes : None or 1d array of datetime objects, optional
        If not None, SeasonalCop instances are also candidates.
    verbose : boolean, optional
    plot : boolean, optional
    asymmetry : boolean, optional
        Avoid symmetric copulae for asymmetric data
    fit_mask : boolean (T,) ndarray or None, optional
        Use only the corresponding time steps for fitting.
    scop_kwds : dict or None
        Additional keyword-args passed to SeasonalCop
    """
    if scop_kwds is None:
        scop_kwds = {}
    if asymmetry:
        for asy_type in (1, 2):
            asy_func = getattr(stats, f"asymmetry{asy_type}")
            asy_data = asy_func(ranks_u, ranks_v)
            if abs(asy_data) > 0.002:
                if verbose:
                    print(f"Asymmetry type {asy_type}: " f"{asy_data:.4f}")
                attr_name = f"symmetry{asy_type}"
                cop_candidates = {
                    cop_name: cop
                    for cop_name, cop in cop_candidates.items()
                    if not getattr(cop, attr_name)
                }
        # if none of the copulas offer the asymmetries, fall back to
        # the full set of copula families
        if not cop_candidates:
            if verbose:
                print("None of the copulas offer the asymmetries.")
            cop_candidates = copulae.all_cops
        with suppress(KeyError):
            del cop_candidates["independence"]
    if cop_conf.PROFILE:
        fitted_cops = map(
            generate_fitted,
            zip(
                list(cop_candidates.keys()),
                list(cop_candidates.values()),
                repeat(ranks_u),
                repeat(ranks_v),
                repeat(fit_mask),
            ),
        )
    else:
        with multiprocessing.Pool(cop_conf.n_nodes) as pool:
            fitted_cops = pool.map(
                generate_fitted,
                zip(
                    list(cop_candidates.keys()),
                    list(cop_candidates.values()),
                    repeat(ranks_u),
                    repeat(ranks_v),
                    repeat(fit_mask),
                ),
            )
    best_cop = max(fitted_cops)
    if dtimes is not None:
        cops = cop_candidates.values()
        cop_seasonal = scop.SeasonalCop(
            None,
            dtimes,
            ranks_u,
            ranks_v,
            asymmetry=asymmetry,
            cop_candidates=cops,
            fit_mask=fit_mask,
            **scop_kwds,
        )
        best_cop = max(best_cop, cop_seasonal)
    if verbose:
        print(f"{best_cop.name} (L={best_cop.likelihood:.2f})")
    if plot:
        n_cops = 9
        fig, axs = plt.subplots(
            nrows=int(np.sqrt(n_cops + 1)),
            ncols=int(np.sqrt(n_cops + 1)),
            subplot_kw=dict(aspect="equal"),
            constrained_layout=True,
        )
        axs = np.ravel(axs)
        wplt.hist2d(
            ranks_u,
            ranks_v,
            cmap=plt.get_cmap("coolwarm"),
            scatter=False,
            ax=axs[0],
        )
        for cop, ax in zip(sorted(fitted_cops)[::-1], axs[1:]):
            cop.plot_density(fig=fig, ax=ax, scatter=False, kind="img")
            name = cop.name_camel
            ax.set_title(
                rf"{name} $\theta$ = {cop.theta[0][0]:.3f}"
                f"\n{cop.likelihood:.2f}",
                fontsize=12,
            )
        plt.show()
    # prefer the independence copula explicitly if it is on par with
    # the best
    if best_cop.likelihood == 0:
        if not isinstance(best_cop, copulae.Independence):
            best_cop = copulae.independence.generate_fitted(ranks_u, ranks_v)
    return best_cop


def mml_futures(
    ranks_u, ranks_v, cops=copulae.all_cops, dtimes=None, verbose=False, **kwds
):
    with ProcessPoolExecutor(cop_conf.n_nodes) as executor:
        futures = {
            executor.submit(generate_fitted, cop, ranks_u, ranks_v): cop_name
            for cop_name, cop in cops.items()
        }
    fitted_cops = []
    for future in as_completed(futures):
        try:
            fitted_cop = future.result()
        except copulae.NoConvergence:
            if verbose:
                print("No fit for %s" % futures[future])
            continue
        fitted_cop.likelihood
        fitted_cops.append(fitted_cop)
    if dtimes is not None:
        fitted_cops += [scop.SeasonalCop(None, dtimes, ranks_u, ranks_v)]
    best_cop = max(fitted_cops)
    if verbose:
        print("%s (L=%.2f)" % (best_cop.name, best_cop.likelihood))
    # prefer the independence copula explicitly if it is on par with
    # the best
    if best_cop.likelihood == 0:
        if not isinstance(best_cop, copulae.Independence):
            best_cop = copulae.independence.generate_fitted(ranks_u, ranks_v)
    return best_cop


def mml_serial(
    ranks_u,
    ranks_v,
    cops=copulae.all_cops,
    dtimes=None,
    verbose=False,
    plot=False,
    **kwds,
):
    fitted_cops = []
    for cop_name, cop in cops.items():
        try:
            fitted_cop = cop.generate_fitted(ranks_u, ranks_v, **kwds)
        except copulae.NoConvergence:
            if verbose > 1:
                print("No fit for %s" % cop_name)
            continue
        else:
            fitted_cops.append(fitted_cop)
            if verbose > 1:
                print(fitted_cop.name, fitted_cop.likelihood)
    if dtimes is not None:
        fitted_cops += [scop.SeasonalCop(None, dtimes, ranks_u, ranks_v)]
    best_cop = max(fitted_cops)
    if verbose:
        print("%s (L=%.2f)" % (best_cop.name, best_cop.likelihood))
    if plot:
        n_cops = 9
        fig, axs = plt.subplots(
            nrows=int(np.sqrt(n_cops)),
            ncols=int(np.sqrt(n_cops)),
            subplot_kw=dict(aspect="equal"),
        )
        axs = np.ravel(axs)
        for cop, ax in zip(sorted(fitted_cops)[::-1], axs):
            cop.plot_density(ax=ax, scatter=False)
            ax.scatter(
                ranks_u,
                ranks_v,
                marker="x",
                s=1,
                facecolor=(0, 0, 0, 0),
                edgecolor=(0, 0, 0, 0.25),
            )
            ax.set_title(
                "%s\n%.2f" % (cop.name[len("fitted ") :], cop.likelihood),
                fontsize=8,
            )
        fig.tight_layout()
    # prefer the independence copula explicitly if it is on par with
    # the best
    if best_cop.likelihood == 0:
        if not isinstance(best_cop, copulae.Independence):
            best_cop = copulae.independence.generate_fitted(ranks_u, ranks_v)
    return best_cop


def mml_kdim(data, cops=copulae.all_cops, k=1):
    K = len(data)
    fitted_cops = {}
    for i in range(K):
        if k > 0:
            ranks_u = data[i, :-k]
        else:
            ranks_u = data[i]
        for j in range(K):
            if i == j:
                continue
            if k > 0:
                ranks_v = data[j, k:]
            else:
                ranks_v = data[j]
            fitted_cops[i, j] = mml_serial(ranks_u, ranks_v, cops)
            print()
    return fitted_cops


def plot_matrix(data, kind="contourf"):
    data_ranks = np.array([stats.rel_ranks(row) for row in data])
    fitted_cops = mml_kdim(data_ranks, copulae.all_cops, k=1)

    K = len(data_ranks)
    fig, axs = plt.subplots(
        K, K, subplot_kw=dict(aspect="equal"), figsize=(15, 15)
    )
    for i in range(K):
        for j in range(K):
            ax = axs[i, j]
            if i == j:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    cop_conf.var_names[i],
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            else:
                cop = fitted_cops[(i, j)]
                cop.plot_density(ax=ax, kind=kind, scatter=False)
                ranks_x, ranks_y = data_ranks[i], data_ranks[j]
                ax.scatter(
                    ranks_x,
                    ranks_y,
                    marker="o",
                    facecolor=(0, 0, 0, 0),
                    edgecolor=(0, 0, 0, 0.1),
                )
                ax.set_title(
                    "%s (%.2f)" % (cop.name[len("fitted ") :], cop.likelihood)
                )
                rho = stats.spearmans_rank(ranks_x, ranks_y)
                asy1 = stats.asymmetry1(ranks_x, ranks_y)
                asy2 = stats.asymmetry2(ranks_x, ranks_y)
                t_kwds = dict(
                    fontsize=25,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="red",
                )
                ax.text(0.5, 0.6, r"$\rho = %.2f$" % rho, **t_kwds)
                ax.text(0.5, 0.5, r"$a_1 = %.2f$" % (100 * asy1), **t_kwds)
                ax.text(0.5, 0.4, r"$a_2 = %.2f$" % (100 * asy2), **t_kwds)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.text(.5, .5, "like: %.2f" % cop.likelihood,
                #                horizontalalignment="center",
                #                verticalalignment="center",
                #                color="red")
    return fig, axs


if __name__ == "__main__":
    import os

    data_filepath = os.path.join(
        cop_conf.weathercop_dir, "code", "vg_data.npz"
    )
    with np.load(data_filepath) as saved:
        data_summer = saved["summer"]
        data_winter = saved["winter"]
    data = np.hstack((data_summer, data_winter))
    # data = data_summer
    # ranks_u_tm1 = copulae.rel_ranks(data_summer[5, :-1])
    # ranks_rh = copulae.rel_ranks(data_summer[4, 1:])
    # # fitted_cops = mml(ranks_u_tm1, ranks_rh, copulae.all_cops)
    # best_cop = mml_serial(ranks_u_tm1, ranks_rh, copulae.all_cops)
    # print(best_cop)

    fig, axs = plot_matrix(data_summer)
    fig.suptitle("summer")
    fig, axs = plot_matrix(data_winter)
    fig.suptitle("winter")
    plt.show()
