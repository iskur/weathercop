import re
from warnings import warn

from tqdm import tqdm
import xarray as xr
import numpy as np
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt

from weathercop import (
    multisite as ms,
    multisite_conditional as msc,
    plotting as wpl,
    cop_conf,
)


def batch(
    ensemble_name, obs_xds, funcs=None, n_realizations=None, verbose=False
):
    if verbose:
        print(f"Plotting {funcs} for {ensemble_name}")
        progress = tqdm
    else:

        def progress(iterable, *args, **kwds):
            return iterable

    if isinstance(funcs, str):
        funcs = (funcs,)
    ms_filepath = ms.pickle_filepath(obs_xds)
    wc = msc.MultisiteConditional.load(ms_filepath)
    ensemble_dir = cop_conf.ensemble_root / ensemble_name
    if not ensemble_dir.exists() and not ensemble_dir.name.endswith("_disag"):
        ensemble_dir = ensemble_dir.parent / (ensemble_dir.name + "_disag")
    if not ensemble_dir.exists():
        RuntimeError(f"ensemble_dir does not exist\n{ensemble_dir}")
    pattern = re.compile(".*real_[0-9]*.nc")
    filepaths = [
        path for path in ensemble_dir.glob("*.nc") if pattern.match(path.name)
    ]
    if not len(filepaths):
        RuntimeError(f"No realizations found in\n{ensemble_dir}")
    plot_dir = ensemble_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    ensemble = (
        xr.open_mfdataset(
            filepaths, concat_dim="realization", combine="nested"
        )
        .assign_coords(realization=range(len(filepaths)))
        .to_array("dummy")
        .squeeze("dummy", drop=True)
    )
    if n_realizations is None:
        realizations = ensemble.coords["realization"].values
    else:
        realizations = range(n_realizations)
    for real_i in progress(realizations, total=n_realizations):
        sim_sea_dis = ensemble.sel(realization=real_i).load()
        sim_sea = sim_sea_dis.resample(time="D").mean()
        wc.sim_sea = sim_sea
        wc.sim_sea_dis = sim_sea_dis
        for funcname in funcs:
            figs_axss = getattr(wc, f"plot_{funcname}")(figsize=(20, 10))
            for station_name, (figs, axs) in figs_axss.items():
                if len(figs) > 1:
                    for fig_i, fig in enumerate(figs):
                        fig.savefig(
                            plot_dir
                            / (
                                f"{funcname}_{real_i:0{cop_conf.n_digits}}"
                                + f"_{station_name}_{fig_i}.png"
                            )
                        )
                figs[0].savefig(
                    plot_dir / f"{funcname}_{real_i:0{cop_conf.n_digits}}"
                    f"_{station_name}.png"
                )
                for fig in figs:
                    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path

    home = Path.home()
    root = home / "Projects/PostDoc/Research_Contracts/code"
    import os

    os.chdir(root)
    import postdoc_conf
    import kll_vg_conf as vg_conf

    ms.set_conf(vg_conf)
    # data_root = home / "data/opendata_dwd"
    data_root = home / "data"

    ms_conf = postdoc_conf.MS
    obs_xds = xr.open_dataset(postdoc_conf.MS.nc_clean_filepath)

    batch(
        # "kll-stale-20230110_disag",
        "kll-stale-20220926_disag",
        obs_xds,
        "meteogram_daily",
        # "hyd_year_sums",
        n_realizations=10,
        verbose=True,
    )

    # batch(
    #     "kll-stale-20220812_disag",
    #     obs_xds,
    #     "meteogram_daily",
    #     n_realizations=10,
    # )
    # batch(
    #     "kll-gradual-20220812_disag",
    #     obs_xds,
    #     "meteogram_daily",
    #     n_realizations=10,
    # )

    # for theta_incr in [1, 2, 3, 4]:
    #     batch(
    #         f"kll-+{theta_incr}_theta-20220811",
    #         obs_xds,
    #         "meteogram_daily",
    #         verbose=True,
    #     )
