import matplotlib.pyplot as plt
import xarray as xr
from weathercop import multisite
import opendata_vg_conf as vg_conf
multisite.set_conf(vg_conf)

xar = xr.open_dataarray("/home/dirk/data/opendata_dwd/"
                        "multisite_testdata.nc")
wc = multisite.Multisite(xar, verbose=False,
                         refit="R")
wc.simulate()
obs = wc.ranks.unstack("rank")
decorr = wc.cop_quantiles

print("corr coefs observed")
print(list(wc.varnames))
# print(np.corrcoef(obs.sel(station="Konstanz")))
print(np.corrcoef(obs.stack(stacked=("station", "time"))).round(3))
print("corr coefs decorr")
print(np.corrcoef(decorr.stack(stacked=("station", "time"))).round(3))

for station_name in wc.station_names:
    fig, axs = plt.subplots(nrows=wc.K, ncols=2)
    for var_i, varname in enumerate(wc.varnames):
        ax = axs[var_i]
        obs_data = obs.sel(variable=varname, station=station_name)
        decorr_data = decorr.sel(variable=varname, station=station_name)
        ax[0].plot(obs_data.time, obs_data.data, label="obs")
        ax[0].plot(decorr_data.time, decorr_data.data,
                           label="decorr")
        ax[1].set(aspect="equal")
        ax[1].scatter(obs_data, decorr_data, marker="o",
                              facecolor=(0, 0, 0, 0),
                              edgecolor=(0, 0, 0, 1))
        ax[0].set_title(varname)
        if var_i > 0:
            ax[0].get_shared_x_axes().join(ax[0],
                                           axs[var_i - 1, 0])
    axs[0, 0].legend(loc="best")
    fig.suptitle(station_name)
    plt.show()
