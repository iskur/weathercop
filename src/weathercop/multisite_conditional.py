from functools import reduce, partial
import collections
from multiprocessing import current_process
from pathlib import Path
import random
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from weathercop import multisite as ms
from weathercop import tools, cop_conf
import varwg
from varwg.time_series_analysis import rain_stats

DEBUG = cop_conf.DEBUG


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


def fft2rfft_1d(params):
    params_rfft = []
    even = []
    for block in params:
        T = len(block)
        if T % 2 == 0:
            params_rfft += [block[1 : T // 2 + 1]]
            even += [True]
        else:
            params_rfft += [block[1 : (T - 1) // 2 + 1]]
            even += [False]
    return params_rfft, even


def rfft2fft(params, even, fill=False):
    K = params[0].shape[0]
    fill = np.full((K, 1), fill)
    params_fft = []
    for block, iseven in zip(params, even):
        if iseven:
            params_fft += [np.hstack((fill, block, fill, -block[:, ::-1]))]
        else:
            params_fft += [np.hstack((fill, block, -block[:, ::-1]))]
    return params_fft


def error_1d(
    phase,
    phase_i,
    phases_rfft,
    even,
    simulate_unconditional,
    conditions,
    real_i=None,
    debug=False,
):
    # phases_rfft = [np.copy(phases_rfft[0])]
    # phases_rfft[0][:, phase_i] = phase
    phases_rfft = np.array(phases_rfft)
    phases_rfft[:, :, phase_i] = phase
    rphases = rfft2fft(phases_rfft, even)
    sim_result = simulate_unconditional(
        rphases=rphases,
        # write_to_disk=False,
        stop_at=conditions.vine_var_i,
        phase_randomize_vary_mean=False,
    )
    # fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True)
    # for ax, station_name in zip(axs, sim_sea.station):
    #     ax.plot(
    #         sim_sea.time.data,
    #         sim_sea.sel(station=station_name, variable="R").data,
    #     )
    #     ax.set_title(str(station_name.data))
    # plt.show()
    # __import__("pdb").set_trace()
    return conditions(sim_result.sim_sea, debug=debug, real_i=real_i)


def error_nd(
    phases, phases_ii, phases_rfft, even, simulate_unconditional, conditions
):
    # phases_rfft = [np.copy(phases_rfft[0])]
    # phases_rfft[0][:, phases_ii] = phases
    phases_rfft = np.array(phases_rfft)
    phases_rfft[:, :, phases_ii] = phases
    rphases = rfft2fft(phases_rfft, even)
    sim_result = simulate_unconditional(
        rphases=rphases,
        # write_to_disk=False,
        stop_at=conditions.vine_var_i,
        phase_randomize_vary_mean=False,
    )
    return conditions(sim_result.sim_sea)


def _phase_1d_opt(
    rphases,
    variable_phases_ii,
    simulate_unconditional,
    conditions,
    real_i,
    verbose=False,
    # verbose=2,
):
    phases_rfft, even = fft2rfft(rphases)
    # variable_phases_ii = random.sample(
    #     tuple(variable_phases_ii), k=len(variable_phases_ii)
    # )
    error = error_old = np.inf
    max_runs = 10
    run_i = 0
    while error > 0:
        run_i += 1
        for phase_i in variable_phases_ii[::-1]:
            result = minimize_scalar(
                error_1d,
                # bounds=[-np.pi, np.pi],
                bounds=[0, 2 * np.pi],
                method="Bounded",
                args=(
                    phase_i,
                    phases_rfft,
                    even,
                    simulate_unconditional,
                    conditions,
                    real_i,
                ),
                # options=dict(disp=True),
            )
            if verbose > 1:
                phase_old = phases_rfft[0][0, phase_i]
                print(
                    f"{run_i=} error={result.fun:.3f} {phase_i=} "
                    f"old={phase_old:.3f} new={result.x:.3f}"
                )
                # for printing per-condition error
                error_1d(
                    result.x,
                    phase_i,
                    phases_rfft,
                    even,
                    simulate_unconditional,
                    conditions,
                    real_i,
                    debug=True,
                )
            phase_new = result.x
            if (error := result.fun) < error_old:
                # phases_rfft[0][:, phase_i] = phase_new
                phases_rfft = np.array(phases_rfft)
                phases_rfft[:, :, phase_i] = phase_new
                rphases = rfft2fft(phases_rfft, even)
                error_old = error
            if np.isclose(error, 0):
                break
        if run_i == max_runs:  # or np.isclose(error, error_old):
            if error > 0:
                rphases = None
            break
        error_old = error
        variable_phases_ii = np.concatenate(
            (
                variable_phases_ii,
                random.sample(
                    tuple(
                        set(range(1, phases_rfft[0].shape[1]))
                        - set(variable_phases_ii)
                    ),
                    k=5,
                ),
            )
        )
        variable_phases_ii = random.sample(
            tuple(variable_phases_ii), k=len(variable_phases_ii)
        )
    if result.fun > 0:
        if verbose:
            print(f"Redrawing random phases (error={result.fun:.1f})")
        return None
    return rphases


def _phase_nd_opt(
    rphases,
    variable_phases_ii,
    simulate_unconditional,
    conditions,
    verbose=False,
):
    phases_rfft, even = fft2rfft(rphases)
    x0 = phases_rfft[0][0, variable_phases_ii]
    result = minimize(
        error_nd,
        x0,
        # method="Nelder-Mead",
        method="TNC",
        bounds=len(variable_phases_ii) * [(0, 2 * np.pi)],
        args=(
            variable_phases_ii,
            phases_rfft,
            even,
            simulate_unconditional,
            conditions,
        ),
        options=dict(disp=True),
    )
    if not result.success:
        __import__("pdb").set_trace()
    phases_rfft[0][:, variable_phases_ii] = result.x
    phases = rfft2fft(phases_rfft, even)
    return phases


def sim_one(args):
    (
        real_i,
        total,
        wcop,
        filepath,
        filepath_daily,
        filepath_trans,
        filepath_rphases,
        filepath_rphases_src,
        sim_args,
        sim_kwds,
        dis_kwds,
        sim_times,
        csv,
        ensemble_dir,
        n_digits,
        write_to_disk,
        verbose,
    ) = args
    varwg.seed(1000 * real_i)
    if "phase_randomize_vary_mean" not in sim_kwds:
        sim_kwds["phase_randomize_vary_mean"] = False
    if DEBUG:
        print(
            current_process().name
            + " in conditional.sim_one. before first unconditional sim"
        )
    simulate_unconditional = partial(
        ms.simulate,
        wcop,
        sim_times,
        *sim_args,
        **sim_kwds,
    )
    if DEBUG:
        print(
            current_process().name
            + " in conditional.sim_one. after first unconditional sim"
        )
    if filepath_rphases_src:
        rphases = np.load(filepath_rphases_src.with_suffix(".npy"))
    else:
        with ms.lock:
            conditions = wcop.conditions
            variable_phases_ii = conditions.variable_phases_ii
        sim_result = simulate_unconditional(return_rphases=True)
        rphases = _phase_1d_opt(
            sim_result.rphases,
            variable_phases_ii,
            simulate_unconditional,
            conditions,
            real_i,
            verbose,
        )
    while rphases is None:
        print(f"reshuffling {real_i=}")
        conditions.reset_reals_cache(real_i)
        with ms.lock:
            phases = wcop.phases
            T_sim = len(sim_times)
            T_summed = wcop[0].T_summed
        rphases = ms._random_phases(phases, T_sim, T_summed)
        rphases = _phase_1d_opt(
            rphases,
            variable_phases_ii,
            simulate_unconditional,
            conditions,
            real_i,
            verbose,
            # verbose=2,
        )
    # sim_sea, sim_trans = simulate_unconditional(rphases=rphases)
    if DEBUG:
        print(
            current_process().name
            + " in conditional.sim_one. before second unconditional sim"
        )
    sim_result = simulate_unconditional(rphases=rphases)
    if DEBUG:
        print(
            current_process().name
            + " in conditional.sim_one. after second unconditional sim"
        )
    if (filepath_rphases_src is None) and (
        error := conditions(sim_result.sim_sea, debug=True)
    ):
        print(f"Condition violated for {real_i=} with {error:.3f=}")
    if dis_kwds is not None:
        if write_to_disk:
            if DEBUG:
                print(
                    current_process().name
                    + " in conditional.sim_one. before daily to_netcdf"
                )
                print(filepath_daily)
            sim_result.sim_sea.to_netcdf(filepath_daily)
            # sim_sea.to_dataset("variable").to_zarr(
            #     filepath_daily.with_suffix(".zarr")
            # )
            if DEBUG:
                print(
                    current_process().name
                    + " in conditional.sim_one. after daily to_netcdf"
                )
        varwg.seed(1000 * real_i)
        if DEBUG:
            print(
                current_process().name
                + " in conditional.sim_one. before disaggregate"
            )
        sim_sea_dis = wcop.disaggregate(**dis_kwds)
        if DEBUG:
            print(
                current_process().name
                + " in conditional.sim_one. after disaggregate"
            )
        sim_sea, sim_sea_dis = xr.align(
            sim_result.sim_sea, sim_sea_dis, join="outer"
        )
        sim_sea.loc[dict(variable=wcop.varnames)] = sim_sea_dis.sel(
            variable=wcop.varnames
        )
    if write_to_disk:
        if DEBUG:
            print(
                current_process().name
                + " in conditional.sim_one. before hourly to_netcdf"
            )
            print(filepath_daily)
        sim_sea.to_netcdf(filepath)
        if DEBUG:
            print(
                current_process().name
                + " in conditional.sim_one. after hourly to_netcdf"
            )
        if csv:
            real_str = f"real_{real_i:0{n_digits}}"
            csv_path = ensemble_dir / "csv" / real_str
            wcop.to_csv(csv_path, sim_sea, filename_prefix=f"{real_str}_")
        # sim_trans.to_netcdf(filepath_trans)
        np.save(filepath_rphases, rphases)
    return real_i


ms.sim_one = sim_one


class MultisiteConditional(ms.Multisite):
    def __init__(self, conditions, *args, **kwds):
        if isinstance(conditions, Conditions):
            self.conditions = conditions
        elif isinstance(conditions, Condition):
            self.conditions = Conditions(conditions)
        else:
            self.conditions = Conditions(*conditions)
        super().__init__(*args, **kwds)

    def simulate(self, *args, rphases=None, return_rphases=False, **kwds):
        if "phase_randomize_vary_mean" not in kwds:
            kwds["phase_randomize_vary_mean"] = False
        kwds = {key: val for key, val in kwds.items()}
        kwds["return_trans"] = False
        simulate_unconditional = partial(super().simulate, *args, **kwds)
        verbose_before = self.verbose_old or False
        self.verbose = False
        if "phase_randomize_vary_mean" in kwds:
            del kwds["phase_randomize_vary_mean"]
        if rphases is None:
            sim_result = simulate_unconditional(return_rphases=True)
            # update conditions with information about T and the power spectrum!
            self.conditions.update(sim_result.rphases, self.As, self.vine)
            rphases = _phase_1d_opt(
                sim_result.rphases,
                self.conditions.variable_phases_ii,
                simulate_unconditional,
                self.conditions,
                None,
                verbose_before,
            )
            while rphases is None:
                print("reshuffling (single_thread)")
                rphases = ms._random_phases(
                    self.phases, sim_result.sim_sea.time.size, self[0].T_summed
                )
                rphases = _phase_1d_opt(
                    rphases,
                    self.conditions.variable_phases_ii,
                    simulate_unconditional,
                    self.conditions,
                    None,
                    verbose_before,
                )
        self.verbose = verbose_before
        # final pass for writing out data
        sim_result = simulate_unconditional(
            # write_to_disk=True,
            rphases=rphases,
            phase_randomize_vary_mean=False,
        )
        if error := self.conditions(sim_result.sim_sea, debug=verbose_before):
            warnings.warn(f"Condition violated with {error=:.3f}")
            # raise RuntimeError(f"Condition violated with {error=:.3f}")
        return sim_result
        # if return_rphases:
        #     return sim_sea, rphases
        # return sim_sea


class Conditions:
    hyd_kwds = dict(full_years=True)

    def __init__(self, *conditions):
        if isinstance(conditions, Condition):
            conditions = (conditions,)
        self._conditions = conditions
        for condition in conditions:
            condition.hyd_kwds = self.hyd_kwds
        self.vine = None
        self._variable_phases_ii = None
        self._calculate_yearly_rain_sums = False
        self.As = self.vine = self.phases = None
        for condition in self:
            # if type(condition).__name__.startswith("YearlyRain"):
            if isinstance(condition, YearlyRain):
                self._calculate_yearly_rain_sums = True
                break

    def update(self, phases, As, vine):
        self._variable_phases_ii = None
        for condition in self:
            condition.update(phases)
        self.As = As
        self.vine = vine
        # how deep do we have to go into the vine?
        self.vine_var_i = max(
            [self.vine.varnames.index(varname) for varname in self.varnames]
        )

    def reset_reals_cache(self, real_i):
        for condition in self:
            condition.reset_reals_cache(real_i)

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     del state["phases"], state["As"], state["vine"]
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     if None not in (self.phases, self.As, self.vine):
    #         self.update(self.phases, self.As, self.vine)

    def __iter__(self):
        return iter(self._conditions)

    def __getitem__(self, key):
        return self._conditions[key]

    def __len__(self):
        return len(self._conditions)

    @property
    def varnames(self):
        return [condition.varname for condition in self]

    @property
    def station_names(self):
        return [condition.station_name for condition in self]

    def __call__(self, data, real_i=None, debug=False):
        rain_sums_dict = dict().fromkeys(self.station_names)
        if "hyd_years_mask" not in self.hyd_kwds:
            self.hyd_kwds["hyd_years_mask"] = rain_stats.get_hyd_years_mask(
                data.time.dt
            )
        if self._calculate_yearly_rain_sums:
            rain = data.sel(variable="R").load()
            for station_name, condition in zip(self.station_names, self):
                # if type(condition).__name__.startswith("YearlyRain"):
                if isinstance(condition, YearlyRain):
                    rain_sums_dict[station_name] = dict(
                        rain_sums=rain_stats.hyd_year_sums(
                            rain.sel(station=station_name), **self.hyd_kwds
                        ).data
                    )
        if debug:
            error_sum = 0
            error_dict = {}
            for condition in self:
                error = condition(
                    data,
                    real_i=real_i,
                    **rain_sums_dict[condition.station_name],
                )
                if error:
                    error_dict[str(condition)] = error
                    error_sum += error
            if error_sum:
                print(f"{error_sum=:.3f}:")
                for name, error in sorted(
                    error_dict.items(), key=lambda item: item[1], reverse=True
                ):
                    error_perc = error / error_sum * 100
                    print(f"\t{name}: {error:.3f} ({error_perc:.1f}%)")
        return sum(
            condition(
                data, real_i=real_i, **rain_sums_dict[condition.station_name]
            )
            for condition in self
        )

    def __add__(self, other):
        if not isinstance(other, Condition):
            raise RuntimeError(
                "Can only add Condition to Conditions not {type(other)}"
            )
        return Conditions(*(list(self) + [other]))

    @property
    def variable_phases_ii(self):
        """Phase indices sorted by decreasing power."""
        if self._variable_phases_ii is None:
            variable_phases = [condition.variable_phases for condition in self]
            combined = reduce(np.logical_or, variable_phases)[0]
            phases_ii = np.where(combined)[0]
            # As = self.As.sel(
            #     station=list(set(self.station_names)),
            #     variable=list(set(self.varnames)),
            # ).sum(dim=("station", "variable"))
            # As_rfft = fft2rfft_1d([As.data])[0][0]
            # power = np.abs(As_rfft[combined]) ** 2
            # self._variable_phases_ii = phases_ii[np.argsort(power)[::-1]]
            self._variable_phases_ii = phases_ii
        return self._variable_phases_ii

    def plot(self, data_xar):
        n_variables = len(set(self.varnames))
        fig, axs = plt.subplots(
            nrows=len(self),
            ncols=n_variables,
            sharex=True,
            constrained_layout=True,
            figsize=(6, 2 * len(self)),
        )
        for condition, ax in zip(self, axs):
            condition.plot(data_xar, fig=fig, ax=ax)
        return fig, axs


class Condition:
    variable_phases_ii = None
    varname = None
    reals_cache = collections.defaultdict(lambda: dict())

    def __str__(self):
        return f"{type(self).__name__} {self._str}"

    def lower_upper_mask(self, T, lower_period=None, upper_period=None):
        freqs = np.fft.rfftfreq(T)
        periods = freqs[1:] ** -1
        mask = np.full_like(periods, False, dtype=bool)
        mask[(periods >= lower_period) & (periods < upper_period)] = True
        return mask

    def update(self, phases):
        self.phases = phases
        self.variable_phases = [
            self.lower_upper_mask(
                phases_.shape[1], self.lower_period, self.upper_period
            )
            for phases_ in phases
        ]

    def reset_reals_cache(self, real_i):
        with ms.lock:
            self.reals_cache[real_i] = dict()

    def plot(self, data_xar, fig=None, ax=None):
        ax.text(
            0.5,
            0.5,
            "Not implemented",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )


class YearlyRain:
    pass


class YearlyRainBounds(Condition, YearlyRain):
    varname = "R"

    def __init__(
        self,
        *,
        lower=None,
        upper=None,
        station_name=None,
        lower_period=None,
        upper_period=None,
    ):
        """For keeping the precipitation sum (hydrolgical year) in bounds."""
        if lower is None and upper is None:
            raise RuntimeError("Must specify at least one of lower or upper")
        self.lower = lower
        self.upper = upper
        self.station_name = station_name
        self.lower_period = lower_period
        self.upper_period = upper_period
        self._str = f"{station_name=} varname={self.varname}"
        if lower is not None:
            self._str = f"{self._str} {lower=:.1f}"
        if upper is not None:
            self._str = f"{self._str} {upper=:.1f}"

    def __call__(self, data_xar, rain_sums=None, real_i=None):
        lower, upper = self.lower, self.upper
        if rain_sums is None:
            rain = data_xar.sel(
                variable=self.varname, station=self.station_name
            ).load()
            rain_sums = rain_stats.hyd_year_sums(rain, **self.hyd_kwds).data
        # if real_i is not None:
        #     with ms.lock:
        #         real_cache = self.reals_cache[real_i]
        #         lower_nudge = real_cache.get("lower_nudge", 0)
        #         upper_nudge = real_cache.get("upper_nudge", 0)
        # else:
        #     lower_nudge = 0
        #     upper_nudge = 0
        error = 0
        if lower is not None:
            error += np.sum((rain_sums[rain_sums < lower] - lower) ** 2)
        if upper is not None:
            error += np.sum((rain_sums[rain_sums > upper] - upper) ** 2)
        # if lower is not None:
        #     error += np.sum(
        #         (rain_sums[rain_sums < lower + lower_nudge] - lower) ** 2
        #     )
        # if upper is not None:
        #     error += np.sum(
        #         (rain_sums[rain_sums > upper - upper_nudge] - upper) ** 2
        #     )
        # if (
        #     error
        #     and real_i is not None
        #     and upper is not None
        #     and lower is not None
        # ):
        #     if lower_nudge == 0:
        #         # be stricter to achieve more natural distribution
        #         with ms.lock:
        #             real_cache["lower_nudge"] = (
        #                 random.random() * (upper - lower) / 10
        #             )
        #             self.reals_cache[real_i] = real_cache
        #     if upper_nudge == 0:
        #         with ms.lock:
        #             real_cache["upper_nudge"] = (
        #                 random.random() * (upper - lower) / 10
        #             )
        #             self.reals_cache[real_i] = real_cache
        return error

    def plot(self, data_xar, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        rain = data_xar.sel(
            variable=self.varname, station=self.station_name
        ).load()
        rain_sums = np.ravel(
            rain_stats.hyd_year_sums(rain, **self.hyd_kwds).data
        )
        ax.hist(rain_sums, 30, density=True)
        if self.lower is not None:
            ax.axvline(self.lower, linestyle="--", color="k")
        if self.upper is not None:
            ax.axvline(self.upper, linestyle="--", color="k")
        ax.set_title(self.station_name)
        ax.set_yticklabels([])
        # ax.set_xlabel("Yearly Precipitation sum [mm]")
        ax.grid(True)
        return fig, ax


class YearlyRainStd(Condition, YearlyRain):
    varname = "R"

    def __init__(
        self,
        *,
        std_max=None,
        station_name=None,
        lower_period=None,
        upper_period=None,
    ):
        if std_max is None:
            raise RuntimeError("Must specify std_max")
        if std_max <= 0:
            raise RuntimeError("std_max must be >0")
        self.std_max = std_max
        self.station_name = station_name
        self.lower_period = lower_period
        self.upper_period = upper_period
        self._str = f"{station_name=} varname={self.varname} {std_max=:.1f}"

    def __call__(self, data_xar, rain_sums=None, **kwds):
        if rain_sums is None:
            rain = data_xar.sel(
                variable=self.varname, station=self.station_name
            ).load()
            rain_sums = rain_stats.hyd_year_sums(rain, **self.hyd_kwds).data
        if (std_sim := rain_sums.std()) > self.std_max:
            return (std_sim - self.std_max) ** 2
        else:
            return 0


class YearlyRainMean(Condition, YearlyRain):
    varname = "R"

    def __init__(
        self,
        *,
        mean_lower=None,
        mean_upper=None,
        station_name=None,
        lower_period=None,
        upper_period=None,
    ):
        if mean_lower is None and mean_upper is None:
            raise RuntimeError(
                "Must specify at least one of mean_lower or mean_upper"
            )
        if mean_lower > mean_upper:
            raise RuntimeError("mean_lower must be lower than mean_upper")
        self.mean_lower = mean_lower
        self.mean_upper = mean_upper
        self.station_name = station_name
        self.lower_period = lower_period
        self.upper_period = upper_period
        self._str = (
            f"{station_name=} varname={self.varname} "
            f"{mean_lower=:.1f} {mean_upper=:.1f}"
        )

    def __call__(self, data_xar, rain_sums=None, **kwds):
        if rain_sums is None:
            rain = data_xar.sel(
                variable=self.varname, station=self.station_name
            ).load()
            rain_sums = rain_stats.hyd_year_sums(rain, **self.hyd_kwds)
        rain_mean = rain_sums.mean()
        mean_lower, mean_upper = (
            self.mean_lower,
            self.mean_upper,
        )
        if mean_lower is not None and rain_mean < mean_lower:
            return (mean_lower - rain_mean) ** 2
        if mean_upper is not None and rain_mean > mean_upper:
            return (mean_upper - rain_mean) ** 2
        return 0

    def plot(self, data_xar, fig=None, ax=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1)
        rain = data_xar.sel(
            variable=self.varname, station=self.station_name
        ).load()
        rain_sums = rain_stats.hyd_year_sums(rain, **self.hyd_kwds)
        rain_mean = rain_sums.values.mean()
        ax.scatter([rain_mean], [0.5])
        if self.mean_lower is not None:
            ax.axvline(self.mean_lower, linestyle="--", color="k")
        if self.mean_upper is not None:
            ax.axvline(self.mean_upper, linestyle="--", color="k")
        ax.set_title(self.station_name)
        ax.grid(True)
        return fig, ax


if __name__ == "__main__":
    from pathlib import Path
    from weathercop.configs import get_dwd_vg_config

    vg_conf = get_dwd_vg_config()

    # np.random.seed(1)
    # ms.set_conf(vg_conf)
    # data_root = Path().home() / "data/opendata_dwd"
    # verbose = True
    # refit = False
    # refit_vine = False
    # xds = xr.open_dataset(data_root / "multisite_testdata.nc")
    # weinbiet_bounds = YearlyRainBounds(
    #     lower=500 / 24,
    #     upper=790 / 24,
    #     station_name="Weinbiet",
    #     lower_period=int(0.5 * 365),
    #     upper_period=int(2 * 365),
    # )
    # wc = MultisiteConditional(
    #     weinbiet_bounds,
    #     xds,
    #     # refit_vine=True,
    #     primary_var="R",
    #     refit=refit,
    #     rain_method="distance",
    #     # cop_candidates=cop_candidates,
    #     verbose=2,
    # )
    # sim_sea = wc.simulate(phase_randomize_vary_mean=False)

    # wc_uc = ms.Multisite(
    #     xds, primary_var="R", refit=refit, rain_method="distance", verbose=True
    # )
    # sim_sea_uc = wc_uc.simulate(phase_randomize_vary_mean=False)
    # fig, ax = wc_uc.plot_cross_corr_var(transformed=True)
    # fig.suptitle("unconditional sim")

    # fig, ax = wc.plot_cross_corr_var(transformed=True)
    # fig.suptitle("after conditional")
    # weinbiet_bounds.plot(sim_sea)

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

    # seed = 19 * 1000
    seed = 0
    varwg.seed(seed)
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
    # n_realizations = 1200
    n_realizations = 32

    b_kwds = dict(
        lower_period=int(0.5 * 365),
        upper_period=int(2 * 365),
        # lower_period=int(0.75 * 365),
        # upper_period=int(1.2 * 365),
    )
    merom_bounds = YearlyRainBounds(
        # lower=360,
        # upper=1390,
        lower=312,
        upper=1441,
        station_name="MeromGolan",
        **b_kwds,
    )
    knaan_bounds = YearlyRainBounds(
        # lower=390,
        # upper=1070,
        lower=346,
        # upper=1111,
        station_name="Knaan",
        **b_kwds,
    )
    kfar_bounds = YearlyRainBounds(
        # lower=225,
        # upper=775,
        lower=192,
        upper=808,
        station_name="KfarBlum",
        **b_kwds,
    )
    kinneret_bounds = YearlyRainBounds(
        # lower=200,
        # upper=820,
        lower=169,
        upper=851,
        station_name="Kinneret_station_A",
        **b_kwds,
    )
    rain_sums = rain_stats.hyd_year_sums(
        xar_daily.sel(variable="R"), full_years=True
    )
    std_max_conditions = [
        YearlyRainStd(
            std_max=rain_sums.sel(station=station_name).values.std() * 1.1,
            station_name=station_name,
            **b_kwds,
        )
        for station_name in xar.station.values
    ]

    mean_conditions = [
        YearlyRainMean(
            mean_lower=rain_sums.sel(station=station_name)
            .quantile(0.35)
            .values,
            mean_upper=rain_sums.sel(station=station_name)
            .quantile(0.65)
            .values,
            station_name=station_name,
            **b_kwds,
        )
        for station_name in xar.station.values
    ]
    for condition in mean_conditions:
        print(condition)

    conditions = Conditions(
        merom_bounds,
        knaan_bounds,
        kfar_bounds,
        kinneret_bounds,
        *(std_max_conditions + mean_conditions),
    )

    wc_dist = MultisiteConditional(
        conditions,
        xds,
        refit_vine=refit_vine,
        primary_var="R",
        refit=refit,
        # rain_method="distance",
        rain_method="simulation",
        cop_candidates=cop_candidates,
        # verbose=True,
        # verbose=2,
        **ms_conf.init,
    )

    # with Path("/tmp/ms.pkl").open("rb") as fi:
    #     # ms.dill.dump(wc_dist, fi)
    #     wc_dist = ms.dill.load(fi)

    # warnings.filterwarnings("error")
    # sim_sea = wc_dist.simulate(
    #     phase_randomize_vary_mean=False,
    #     start_str="2006-10-01T00:00:00",
    #     stop_str="2015-09-30T00:00:00",
    # )
    # wc_dist.disaggregate(**dis_kwds)
    # wc_dist.conditions.plot(sim_sea)

    sim_dis = wc_dist.simulate_ensemble(
        n_realizations,
        name="kll-stale-distance-std_mean_bounds",
        clear_cache=recalc,
        dis_kwds=dis_kwds,
        phase_randomize_vary_mean=False,
        start_str="2006-10-01T00:00:00",
        stop_str="2019-09-30T00:00:00",
    )

    # with Path("./ms.pkl").open("wb") as fi:
    #     ms.dill.dump(wc_dist, fi)

    wc_dist.conditions.plot(sim_dis if dis_kwds is None else sim_dis / 24)

    # drought_bound = 700
    # knaan_drought_bound = YearlyRainBounds(
    #     upper=drought_bound, station_name="Knaan", **b_kwds
    # )
    # R_diff = (
    #     drought_bound
    #     - float(rain_sums.sel(station="Knaan").quantile(0.9).values)
    # ) / 365
    # print(f"{R_diff=}")

    # wc_dist = MultisiteConditional(
    #     conditions + knaan_drought_bound,
    #     xds,
    #     refit_vine=refit_vine,
    #     primary_var="R",
    #     refit=refit,
    #     rain_method="distance",
    #     cop_candidates=cop_candidates,
    #     # verbose=True,
    #     verbose=2,
    #     **ms_conf.init,
    # )

    # warnings.filterwarnings("error")
    # sim_sea = wc_dist.simulate(
    #     theta_incr=R_diff,
    #     phase_randomize_vary_mean=False,
    #     start_str="2006-10-01T00:00:00",
    #     stop_str="2015-09-30T00:00:00",
    # )
    # # wc_dist.conditions.plot(sim_sea)

    wc_dist.plot_meteogram_daily_stat()
    wc_dist.plot_meteogram_daily_decorr()

    # sim_dis = wc_dist.simulate_ensemble(
    #     n_realizations,
    #     name="kll-stale-distance-more-drought700",
    #     clear_cache=recalc,
    #     # csv=True,
    #     # phase_randomize_vary_mean=0.25,
    #     dis_kwds=dis_kwds,
    #     theta_incr=R_diff,
    #     start_str="2006-10-01T00:00:00",
    #     stop_str="2015-09-30T00:00:00",
    # )
    # # for realization in range(n_realizations):
    # #     fig, ax = minrain.plot((sim_dis.sel(realization=realization).load()))
    # #     fig.suptitle(f"{realization=}")
    # conditions.plot(sim_dis)

    plt.show()
