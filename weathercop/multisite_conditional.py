from functools import reduce
import numpy as np
from scipy.optimize import minimize_scalar
from weathercop import multisite as ms
from weathercop import tools, cop_conf


def fft2rfft(params):
    params_rfft = []
    even = []
    for block in params:
        T = block.shape[1]
        if T % 2 == 0:
            params_rfft += [block[:, 1 : T // 2 + 1]]
            even += [True]
        else:
            params_rfft += [block[:, 1 : (T - 1) // 2 + 1]]
            even += [False]
    return params_rfft, even


def rfft2fft(params, even, fill=False):
    K = params[0].shape[0]
    fill = np.full((K, 1), fill)
    params_fft = []
    for block, iseven in zip(params, even):
        if iseven:
            params_fft += [np.hstack((fill, block, fill, block))]
        else:
            params_fft += [np.hstack((fill, block, block))]
    return params_fft


class MultisiteConditional(ms.Multisite):
    def simulate_conditional(self, conditions, *args, **kwds):
        if isinstance(conditions, Conditions):
            self.conditions = conditions
        else:
            self.conditions = Conditions(conditions)
        conditions = self.conditions
        verbose_before = self.verbose
        self.verbose = False
        del kwds["phase_randomize_vary_mean"]
        self.simulate(phase_randomize_vary_mean=False, *args, **kwds)
        # update conditions with information about T!
        self.conditions.update(self._rphases, self.vine)
        phases_rfft, even = fft2rfft(self._rphases)
        # shift the longest configured harmonic first
        variable_phases_ii = np.where(self.conditions.variable_phases)[1]
        for phase_i in variable_phases_ii[::-1]:

            def error_1d(phase):
                phases_rfft[0][:, phase_i] = phase
                rphases = rfft2fft(phases_rfft, even)
                sim_sea = self.simulate(
                    write_to_disk=False,
                    stop_at=conditions.vine_var_i,
                    rphases=rphases,
                    phase_randomize_vary_mean=False,
                )
                return self.conditions.error_sum(sim_sea)

            result = minimize_scalar(error_1d, bounds=[0, 2 * np.pi])
            phases_rfft[0][:, phase_i] = result.x
            rphases = rfft2fft(phases_rfft, even)
            self.rphases = self._rphases = rphases
            if verbose_before > 1:
                print(f"error={result.fun:.1f}")
            if np.isclose(result.fun, 0):
                break
        self.verbose = verbose_before
        # final pass for writing out data
        return self.simulate(
            write_to_disk=True,
            rphases=rphases,
            phase_randomize_vary_mean=False,
        )


class Conditions:
    def __init__(self, conditions):
        if isinstance(conditions, Condition):
            conditions = (conditions,)
        self._conditions = conditions
        self.vine = None

    def update(self, phases, vine):
        for condition in self:
            condition.update(phases)
        self.vine = vine
        # how deep do we have to go into the vine?
        self.vine_var_i = max(
            [self.vine.varnames.index(varname) for varname in self.varnames]
        )

    def __iter__(self):
        return iter(self._conditions)

    def __getitem__(self, key):
        return self._conditions[key]

    @property
    def varnames(self):
        return [condition.varname for condition in self]

    def error_sum(self, data):
        return sum(condition(data) for condition in self)

    @property
    def variable_phases(self):
        variable_phases = [condition.variable_phases for condition in self]
        return reduce(np.logical_or, variable_phases)


class Condition:
    variable_phases = None
    varname = None

    def update(self, phases):
        pass

    def lower_upper_mask(self, T, lower_period=None, upper_period=None):
        freqs = np.fft.rfftfreq(T)
        periods = freqs[1:] ** -1
        mask = np.full_like(periods, False, dtype=bool)
        mask[(periods >= lower_period) & (periods < upper_period)] = True
        return mask


class MinimumYearlyRain(Condition):
    varname = "R"

    def __init__(
        self, min_sum, station_name, lower_period=None, upper_period=None
    ):
        """For ensuring minimum yearly precipitation sum."""
        self.min_sum = min_sum
        self.station_name = station_name
        self.lower_period = lower_period
        self.upper_period = upper_period

    def update(self, phases):
        self.variable_phases = [
            self.lower_upper_mask(
                phases_.shape[1], self.lower_period, self.upper_period
            )
            for phases_ in phases
        ]

    def __call__(self, data_xar):
        rain = data_xar.sel(variable="R", station=self.station_name).load()
        rain_sums = tools.hyd_year_sums(rain).data
        rain_sums = rain_sums[rain_sums < self.min_sum]
        return np.sum((rain_sums - self.min_sum) ** 2)

    def plot(self, data_xar):
        rain = data_xar.sel(variable="R", station=self.station_name).load()
        rain_sums = tools.hyd_year_sums(rain).data
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.hist(rain_sums, 30)
        ax.axvline(self.min_sum, linestyle="--", color="k")
        return fig, ax


if __name__ == "__main__":
    from pathlib import Path
    import xarray as xr
    import matplotlib.pyplot as plt
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

    np.random.seed(1)
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
    recalc = True
    # dis_kwds = dict(var_names_dis=[name for name in varnames if name != "R"])
    dis_kwds = None

    wc_dist = MultisiteConditional(
        xds,
        # refit_vine=True,
        primary_var="R",
        refit=refit,
        rain_method="distance",
        cop_candidates=cop_candidates,
        verbose=2,
        **ms_conf.init,
    )

    minrain = MinimumYearlyRain(
        300,
        "MeromGolan",
        lower_period=int(0.5 * 365),
        upper_period=int(2 * 365),
    )
    sim_sea = wc_dist.simulate_conditional(
        minrain, phase_randomize_vary_mean=False
    )

    # minrain.plot(sim_sea)
    # wc_dist.plot_meteogram_daily_stat()
    # wc_dist.plot_meteogram_daily_decorr()
    # plt.show()
