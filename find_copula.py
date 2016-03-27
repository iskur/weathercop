from concurrent.futures import ProcessPoolExecutor, as_completed
from weathercop import copulae, cop_conf, stats


pool = ProcessPoolExecutor()


def mml(ranks_u, ranks_v, cops):
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(cop.generate_fitted, ranks_u, ranks_v):
                   cop_name
                   for cop_name, cop in cops.items()}
    fitted_cops = []
    for future in as_completed(futures):
        try:
            fitted_cop = future.result()
        except copulae.NoConvergence:
            print("No fit for %s" % futures[future])
            continue
        fitted_cop.likelihood
        fitted_cops.append(fitted_cop)
    return fitted_cops


def mml_serial(ranks_u, ranks_v, cops):
    fitted_cops = []
    for cop_name, cop in cops.items():
        try:
            fitted_cop = cop.generate_fitted(ranks_u, ranks_v)
        except copulae.NoConvergence:
            print("No fit for %s" % cop_name)
            continue
        else:
            fitted_cops.append(fitted_cop)
            print(fitted_cop.name, fitted_cop.likelihood)
    return max(fitted_cops)


def mml_kdim(data, cops):
    K = len(data)
    fitted_cops = {}
    for i in range(K):
        ranks_u = data[i, :-1]
        for j in range(K):
            if i == j:
                continue
            ranks_v = data[j, 1:]
            fitted_cops[i, j] = mml_serial(ranks_u, ranks_v, cops)
    return fitted_cops


if __name__ == '__main__':
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    data_filepath = os.path.join(cop_conf.weathercop_dir, "code",
                                 "vg_data.npz")
    with np.load(data_filepath) as saved:
        data_summer = saved["summer"]
    # ranks_u_tm1 = copulae.rel_ranks(data_summer[5, :-1])
    # ranks_rh = copulae.rel_ranks(data_summer[4, 1:])
    # # fitted_cops = mml(ranks_u_tm1, ranks_rh, copulae.all_cops)
    # best_cop = mml_serial(ranks_u_tm1, ranks_rh, copulae.all_cops)
    # print(best_cop)
    data_ranks = np.array([copulae.rel_ranks(row) for row in data_summer])
    fitted_cops = mml_kdim(data_ranks, copulae.all_cops)

    K = len(data_ranks)
    fig, axs = plt.subplots(K, K, subplot_kw=dict(aspect="equal"))
    for i in range(K):
        for j in range(K):
            if i == j:
                axs[i, j].set_axis_off()
            else:
                cop = fitted_cops[(i, j)]
                cop.plot_density(ax=axs[i, j])
                axs[i, j].set_title(cop.name)
    plt.show()
