from functools import partial
from heapq import merge
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from weathercop import multisite as ms, tools


def harmonic(amp, period, shift, tt):
    return amp * np.cos(2 * np.pi * (tt - shift) / period)


def harmonic_anti_dev(amp, period, shift, tt):
    return (
        amp
        * (period / (2 * np.pi))
        * np.sin(2 * np.pi * (tt - shift) / period)
    )


def get_roots_ii(shift, period, T):
    if shift > 0:
        shift = shift % period
    if shift < period / 4:
        root_first = period / 4 + shift
    elif period / 4 <= shift < 3 / 4 * period:
        root_first = shift - period / 4
    elif 3 / 4 * period <= shift < period:
        root_first = shift - 3 / 4 * period
    return np.arange(int(round(root_first)), T, int(round(period / 2)))


def get_yearly_integs(amp, period, shift, hyd_years_ii, tt):
    roots_ii = get_roots_ii(shift, period, len(tt))
    integ_steps = np.array(list(merge(roots_ii, hyd_years_ii)))
    integs_all = np.diff(harmonic_anti_dev(amp, period, shift, integ_steps))
    # yearly_integs = np.zeros(len(hyd_years_ii) + 1)
    yearly_integs = np.zeros(len(hyd_years_ii))
    hyd_years_ii = np.concatenate((hyd_years_ii, [tt[-1]]))
    year_i = 0
    for step, integ in zip(integ_steps, integs_all):
        if step > hyd_years_ii[year_i]:
            year_i += 1
        yearly_integs[year_i] += integ
    # return yearly_integs
    return yearly_integs[:-1]


class MultisiteAnnual(ms.Multisite):
    def __init__(
        self,
        *args,
        varnames_annual="R",
        lower_period=None,
        upper_period=None,
        **kwds,
    ):
        super().__init__(*args, **kwds)
        if isinstance(varnames_annual, str):
            varnames_annual = list(varnames_annual)
        self.varnames_annual = varnames_annual
        self.lower_period = lower_period
        self.upper_period = upper_period
        self.annual_sums = self._get_annual_sums(self.data_trans)
        self.annual_sums_sorted = {
            station_name: np.sort(
                self.annual_sums.sel(station=station_name).data.squeeze()
            )
            for station_name in self.station_names
        }
        self.annual_mins = self.annual_sums.min(dim="hydyear")
        self.annual_maxs = self.annual_sums.max(dim="hydyear")
        self.annual_stds = self.annual_sums.std(dim="hydyear")
        self._hyd_years_ii = None

    def hyd_years_ii(self):
        if self._hyd_years_ii is None:
            # hyd_years_ii = np.where(np.diff(self.times.dt.year))[0] + 1
            self._hyd_years_ii = np.where(
                tools.get_hyd_years_mask(self.times_daily)
            )[0]
            # self._hyd_years_ii = np.concatenate(([0], _hyd_years_ii))
        return self._hyd_years_ii

    def _get_annual_sums(self, data, full_years=False):
        return tools.hyd_year_sums(data.sel(variable=self.varnames_annual))

    def variable_phases_ii(self, T):
        freqs = np.fft.rfftfreq(T)
        periods = freqs[1:] ** -1
        if self.lower_period is None:
            lower_period = periods[-1] + 1
        else:
            lower_period = self.lower_period
        if self.upper_period is None:
            upper_period = periods[0]
        else:
            upper_period = self.upper_period
        mask = np.full_like(periods, False, dtype=bool)
        mask[(periods >= lower_period) & (periods < upper_period)] = True
        return np.where(mask)[0]

    def _error_sum(self, annual_sums_sim):
        obs_sorted = self.annual_sums_sorted
        error = 0
        for station_name in self.station_names:
            sim_sorted = np.sort(
                annual_sums_sim.sel(station=station_name).data.squeeze()
            )
            error += sum((obs_sorted[station_name] - sim_sorted) ** 2)
        return error

    def simulate(self, *args, **kwds):
        if "phase_randomize_vary_mean" not in kwds:
            kwds["phase_randomize_vary_mean"] = False
        simulate_unconditional = partial(super().simulate, *args, **kwds)
        verbose_before = self.verbose
        self.verbose = False
        if "phase_randomize_vary_mean" in kwds:
            del kwds["phase_randomize_vary_mean"]
        sim_sea, rphases = simulate_unconditional(return_rphases=True)

        annual_sums_sim = self._get_annual_sums(self.sim)
        error = self._error_sum(annual_sums_sim)
        freqs = np.fft.fftfreq(self.T_sim)
        tt = np.arange(self.T_sim)
        periods = freqs[1:] ** -1
        for phase_i in self.variable_phases_ii(self.T_sim):
            shift = rphases[0][0, phase_i]
            for station_name in self.station_names:
                for varname in self.varnames_annual:
                    # subtract harmonic with current phase
                    amp = (
                        np.abs(
                            self.As.sel(
                                station=station_name,
                                variable=self.varnames_annual,
                            ).data[:, phase_i]
                        )
                        ** 2
                    )
                    period = periods[phase_i + 1]
                    shift = rphases[0][0, phase_i]
                    yearly_integs = get_yearly_integs(
                        amp, period, shift, self.hyd_years_ii(), tt
                    )
                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    ax.plot(yearly_integs, label="integs")
                    harmonic_xr = xr.DataArray(
                        [harmonic(amp, period, shift, tt)],
                        dims=["variable", "time"],
                        coords=[["R"], sim_sea.time],
                    )
                    annual_sums = self._get_annual_sums(
                        harmonic_xr, full_years=True
                    )
                    ax.plot(
                        annual_sums.data.squeeze(),
                        label="sums",
                    )
                    ax.legend(loc="best")
                    plt.show()
                    __import__("pdb").set_trace()

        annual_mins_sim = annual_sums_sim.min(dim="hydyear")
        annual_maxs_sim = annual_sums_sim.max(dim="hydyear")
        annual_stds_sim = annual_sums_sim.std(dim="hydyear")
        fig, axs = plt.subplots(
            nrows=self.n_stations, ncols=1, constrained_layout=True
        )
        prop_cycle = mpl.rcParams["axes.prop_cycle"]
        for ax, station_name, prop in zip(axs, self.station_names, prop_cycle):
            color = prop["color"]
            obs = self.annual_sums.sel(station=station_name).squeeze()
            sim = annual_sums_sim.sel(station=station_name).squeeze()
            ax.plot(obs.hydyear, obs.data, label=station_name, color=color)
            ax.plot(sim.hydyear, sim.data, "--", color=color)
            ax.set_title(station_name)
            ax.grid(True)
        plt.suptitle("Annual sums")
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(6, 3))
        for station_name in self.station_names:
            axs[0].plot(
                [1, 2],
                [
                    self.annual_mins.sel(station=station_name).data,
                    annual_mins_sim.sel(station=station_name).data,
                ],
                "-x",
                label=station_name,
            )
            axs[1].plot(
                [1, 2],
                [
                    self.annual_stds.sel(station=station_name).data,
                    annual_stds_sim.sel(station=station_name).data,
                ],
                "-x",
                label=station_name,
            )
            axs[2].plot(
                [1, 2],
                [
                    self.annual_maxs.sel(station=station_name).data,
                    annual_maxs_sim.sel(station=station_name).data,
                ],
                "-x",
                label=station_name,
            )
        axs[0].set_title("mins")
        axs[1].set_title("stds")
        axs[2].set_title("maxs")
        # for ax in axs:
        #     ax.legend(loc="best")
        plt.show()
        __import__("pdb").set_trace()
        self.verbose = verbose_before


if __name__ == "__main__":
    from pathlib import Path
    import xarray as xr
    import varwg
    from weathercop import copulae

    home = Path.home()
    root = home / "Projects/PostDoc/Research_Contracts/code"
    import os

    os.chdir(root)
    import postdoc_conf
    import kll_vg_conf as vg_conf

    ms.set_conf(vg_conf)
    data_root = home / "data/opendata_dwd"
    ms_conf = postdoc_conf.MS

    # seed = 19 * 1000
    seed = 0
    # np.random.seed(seed)
    varwg.reseed(seed)
    xds = xr.open_dataset(postdoc_conf.MS.nc_clean_filepath)
    xds = xds.interpolate_na("time")
    xar = xds.drop_vars(("latitude", "longitude")).to_array("station")
    xar_daily = xar.resample(time="D").mean("time")
    varnames = xar.coords["variable"].values
    # clayton is selected in theta-R where it is not good!
    # cop_candidates = {name: cop for name, cop in copulae.all_cops.items()
    #                   if not name.startswith("clayton")}
    cop_candidates = copulae.all_cops
    refit = False
    # refit = "U"
    # refit = "abs_hum"
    # refit = "theta"
    # refit = "R"
    refit_vine = False
    recalc = False
    dis_kwds = dict(var_names_dis=[name for name in varnames if name != "R"])
    # dis_kwds = None
    n_realizations = 500

    wc_dist = MultisiteAnnual(
        xds,
        lower_period=50,
        upper_period=3 * 365.25,
        refit_vine=refit_vine,
        primary_var="R",
        refit=refit,
        rain_method="distance",
        cop_candidates=cop_candidates,
        # verbose=True,
        verbose=2,
        **ms_conf.init,
    )
    wc_dist.simulate()
