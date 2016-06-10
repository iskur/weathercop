from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
from weathercop import copulae, cop_conf, stats


pool = ProcessPoolExecutor()


def mml(ranks_u, ranks_v, cops=copulae.all_cops, verbose=False):
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(cop.generate_fitted, ranks_u, ranks_v):
                   cop_name
                   for cop_name, cop in cops.items()}
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
    best_cop = max(fitted_cops)
    print(best_cop.name)
    # prefer the independence copula explicitly if it is on par with
    # the best
    if best_cop.likelihood == 0:
        if not isinstance(best_cop, copulae.Independence):
            best_cop = copulae.independence.generate_fitted(None, None)
    return best_cop


def mml_serial(ranks_u, ranks_v, cops=copulae.all_cops, verbose=False):
    fitted_cops = []
    for cop_name, cop in cops.items():
        try:
            fitted_cop = cop.generate_fitted(ranks_u, ranks_v)
        except copulae.NoConvergence:
            if verbose:
                print("No fit for %s" % cop_name)
            continue
        else:
            fitted_cops.append(fitted_cop)
            if verbose:
                print(fitted_cop.name, fitted_cop.likelihood)
    best_cop = max(fitted_cops)
    print(best_cop.name)
    # prefer the independence copula explicitly if it is on par with
    # the best
    if best_cop.likelihood == 0:
        if not isinstance(best_cop, copulae.Independence):
            best_cop = copulae.independence.generate_fitted(None, None)
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
    fig, axs = plt.subplots(K, K, subplot_kw=dict(aspect="equal"),
                            figsize=(15, 15))
    for i in range(K):
        for j in range(K):
            ax = axs[i, j]
            if i == j:
                ax.set_axis_off()
                ax.text(.5, .5, cop_conf.var_names[i],
                        horizontalalignment="center",
                        verticalalignment="center")
            else:
                cop = fitted_cops[(i, j)]
                cop.plot_density(ax=ax, kind=kind, scatter=False)
                ranks_x, ranks_y = data_ranks[i], data_ranks[j]
                ax.scatter(ranks_x, ranks_y,
                           marker="o",
                           facecolor=(0, 0, 0, 0),
                           edgecolor=(0, 0, 0, .1))
                ax.set_title("%s (%.2f)" % (cop.name[len("fitted "):],
                                            cop.likelihood))
                rho = stats.spearmans_rank(ranks_x, ranks_y)
                asy1 = stats.asymmetry1(ranks_x, ranks_y)
                asy2 = stats.asymmetry2(ranks_x, ranks_y)
                t_kwds = dict(fontsize=25,
                              horizontalalignment="center",
                              verticalalignment="center",
                              color="red")
                ax.text(.5, .6, r"$\rho = %.2f$" % rho, **t_kwds)
                ax.text(.5, .5, r"$a_1 = %.2f$" % (100 * asy1),  **t_kwds)
                ax.text(.5, .4, r"$a_2 = %.2f$" % (100 * asy2), **t_kwds)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.text(.5, .5, "like: %.2f" % cop.likelihood,
                #                horizontalalignment="center",
                #                verticalalignment="center",
                #                color="red")
    return fig, axs


if __name__ == '__main__':
    import os
    data_filepath = os.path.join(cop_conf.weathercop_dir, "code",
                                 "vg_data.npz")
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
