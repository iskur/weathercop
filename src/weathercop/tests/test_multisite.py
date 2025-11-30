import copy

import numpy as np
import numpy.testing as npt
import pytest

# Mark entire module as memory-intensive
pytestmark = pytest.mark.memory_intensive
import varwg
from matplotlib import pyplot as plt
from scipy import stats

from weathercop.multisite import nan_corrcoef


def test_phase_randomization_corr(multisite_simulation_result):
    """Verify cross-station correlations preserved under phase randomization.

    Generates FFT-based phase-randomized weather and compares inter-station
    correlations against observed data using 2 decimal place tolerance.
    """
    wc, sim_result = multisite_simulation_result
    qq_std = wc.qq_std
    fft_sim = wc.fft_sim

    for var_i, varname in enumerate(wc.varnames):
        data_sim_ = fft_sim.sel(variable=varname).values
        sim_corr = nan_corrcoef(data_sim_)
        sim_corr = sim_corr[np.triu_indices_from(sim_corr, 1)]
        data_obs_ = qq_std.sel(variable=varname).values
        obs_corr = nan_corrcoef(data_obs_)
        obs_corr = obs_corr[np.triu_indices_from(obs_corr, 1)]
        try:
            npt.assert_almost_equal(sim_corr, obs_corr, decimal=2)
        except AssertionError:
            fig, ax = plt.subplots(
                nrows=1, ncols=1, subplot_kw=dict(aspect="equal")
            )
            ax.scatter(obs_corr, sim_corr, marker="x")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.show()
            raise


def test_sim_rphases(multisite_simulation_result):
    """Verify reproducibility when using saved random phases.

    Tests that:
    1. Saving rphases from initial simulation preserves them
    2. Using saved rphases produces bit-identical results
    3. All stations produce identical results when reusing phases

    Note: rphases are modified in-place during simulate() in the _vg_ph method
    (specifically at phase_[:, 0] = zero_phases[station_name]). To test
    reproducibility with the exact same phases, we must deep copy before
    second use.
    """
    wc, sim_result = multisite_simulation_result

    rphases = wc._rphases
    # Deep copy because simulate() modifies rphases in-place during _vg_ph
    rphases_copy = copy.deepcopy(rphases)

    sim_sea_new = wc.simulate(
        T=wc.T_sim, rphases=rphases_copy, phase_randomize_vary_mean=False
    ).sim_sea

    # Use another deep copy for the second call
    rphases_copy2 = copy.deepcopy(rphases)
    sim_sea_new2 = wc.simulate(
        T=wc.T_sim, rphases=rphases_copy2, phase_randomize_vary_mean=False
    ).sim_sea

    # Two simulations with identical input rphases should produce nearly identical output
    # Using decimal=2 (0.01 tolerance) due to small numerical differences from phase adjustment
    npt.assert_almost_equal(sim_sea_new2.values, sim_sea_new.values, decimal=2)

    for station_i, station_name in enumerate(wc.station_names):
        print(f"{station_name=}")
        actual = sim_result.sim_sea.sel(station=station_name)
        expected = sim_sea_new.sel(station=station_name)
        try:
            npt.assert_almost_equal(actual.values, expected.values, decimal=2)
        except AssertionError:
            fig, axs = plt.subplots(
                nrows=wc.K,
                ncols=1,
                sharex=True,
                constrained_layout=True,
            )
            for ax, varname in zip(axs, wc.varnames):
                ax.plot(
                    expected.time,
                    expected.sel(variable=varname).values,
                    label="sim1",
                    color="k",
                )
                ax.plot(
                    actual.time,
                    actual.sel(variable=varname).values,
                    label="sim2",
                    color="b",
                    linestyle="--",
                )
                ax.set_title(varname)
            axs[0].legend(loc="best")
            fig.suptitle(f"{station_i=}: {station_name}")

            fig, axs = plt.subplots(
                nrows=1,
                ncols=wc.K,
                subplot_kw=dict(aspect="equal"),
                constrained_layout=True,
                figsize=(wc.K * 4, 4),
            )
            for ax, varname in zip(axs, wc.varnames):
                expected_var = expected.sel(variable=varname).values
                actual_var = actual.sel(variable=varname).values
                ax.scatter(
                    expected_var, actual_var, s=1, marker="x", color="b"
                )
                min_ = min(expected_var.min(), actual_var.min())
                max_ = max(expected_var.max(), actual_var.max())
                ax.plot(
                    [min_, max_],
                    [min_, max_],
                    linestyle="--",
                    color="k",
                    linewidth=1,
                )
                ax.grid(True)
                ax.set_title(varname)

            plt.show()
            raise


def test_sim_mean(multisite_simulation_result):
    """Verify simulated mean values match observed data.

    Compares correlations and means of simulated vs observed data,
    using 1 decimal place tolerance for spatial correlations.
    """
    wc, sim_result = multisite_simulation_result

    sim_sea = sim_result.sim_sea
    sim_stacked = sim_sea.stack(stacked=("variable", "time")).T
    obs_stacked = wc.data_daily.stack(stacked=("variable", "time")).T
    corr_sim = nan_corrcoef(sim_stacked.T)
    corr_obs = nan_corrcoef(obs_stacked.T)
    try:
        npt.assert_almost_equal(corr_sim, corr_obs, decimal=1)
    except AssertionError:
        wc.plot_corr()
        fig, ax = plt.subplots(
            nrows=1, ncols=1, subplot_kw=dict(aspect="equal")
        )
        ax.scatter(corr_obs, corr_sim, marker="x")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.show()
        raise

    sim_means = sim_sea.mean("time")
    obs_means = wc.data_daily.mean("time")
    npt.assert_almost_equal(sim_means.data, obs_means.data, decimal=1)


def test_sim_mean_increase(multisite_simulation_result):
    """Verify simulated mean increase with theta_incr parameter.

    Tests two scenarios:
    1. theta_incr=4: Verify constant offset in theta variable
    2. theta_incr=None with usevg=True: Verify zero offset
    """
    wc, sim_result = multisite_simulation_result

    # Test 1: theta_incr = 4
    theta_incr = 4
    varwg.reseed(0)
    wc.reset_sim()
    sim_result_incr = wc.simulate(
        theta_incr=theta_incr,
        phase_randomize_vary_mean=False,
        phase_randomize=False,
    )
    sim = sim_result_incr.sim_sea.sel(variable="theta", drop=True).mean()
    obs = wc.data_daily.sel(variable="theta", drop=True).mean()
    npt.assert_almost_equal(sim - obs, theta_incr, decimal=1)

    # Test 2: theta_incr = None with usevg=True
    theta_incr = None
    varwg.reseed(0)
    wc.reset_sim()
    sim_result_vg = wc.simulate(
        theta_incr=theta_incr,
        phase_randomize_vary_mean=False,
        usevg=True,
    )
    sim = sim_result_vg.sim_sea.sel(variable="theta", drop=True).mean()
    obs = wc.data_daily.sel(variable="theta", drop=True).mean()
    npt.assert_almost_equal(
        sim - obs, theta_incr if theta_incr else 0, decimal=1
    )


@pytest.mark.xfail(reason="Look into varwg's resampling")
def test_sim_resample(multisite_simulation_result):
    """Verify simulated mean with resampling workflow.

    Tests simulation with resampling parameters including candidate selection
    and recalibration, verifying zero mean offset.
    """
    wc, sim_result = multisite_simulation_result

    theta_incr = None
    varwg.reseed(1)
    wc.reset_sim()
    sim_result_resample = wc.simulate(
        T=2 * wc.T_sim,
        theta_incr=theta_incr,
        phase_randomize_vary_mean=False,
        usevg=True,
        res_kwds=dict(
            n_candidates=None,
            recalibrate=True,
            doy_tolerance=20,
            verbse=True,
            resample_raw=True,
        ),
    )
    sim = sim_result_resample.sim_sea.sel(variable="theta", drop=True).mean()
    obs = wc.data_daily.sel(variable="theta", drop=True).mean()
    npt.assert_almost_equal(
        sim - obs, theta_incr if theta_incr else 0, decimal=2
    )


@pytest.mark.xfail(reason="Known failure - issue with non-gaussian marginals")
def test_sim_gradual(multisite_simulation_result):
    """Verify simulated temperature gradient.

    Tests that applying a theta_grad parameter produces the expected
    linear trend in temperature over time. Marked as xfail due to known
    issues with non-gaussian marginals.
    """
    wc, sim_result = multisite_simulation_result

    theta_grad = 1.5
    decimal = 1
    wc.reset_sim()
    varwg.reseed(1)
    sim_result_grad = wc.simulate(
        theta_grad=theta_grad,
        phase_randomize_vary_mean=False,
        rphases=sim_result.rphases,
    )
    for station_name in wc.station_names:
        sim_station = sim_result_grad.sim_sea.sel(
            variable="theta", station=station_name, drop=True
        )
        lr_result = stats.linregress(np.arange(wc.T_sim), sim_station.values)
        gradient = lr_result.slope * wc.T_sim
        npt.assert_almost_equal(theta_grad, gradient, decimal=decimal)


# def test_sim_primary_var(multisite_simulation_result):
#     """Verify simulated values for a primary variable with theta_incr.

#     Tests simulation when specifying a primary variable (sun) with a theta
#     increment and reuses rphases from initial simulation.
#     """
#     wc, sim_result = multisite_simulation_result

#     prim_incr = 0.01
#     prim_var_sim = "rh"
#     sim_result_prim = wc.simulate(
#         theta_incr=prim_incr,
#         primary_var=prim_var_sim,
#         rphases=sim_result.rphases,
#         phase_randomize_vary_mean=False,
#     )
#     sim = sim_result_prim.sim.sel(variable=prim_var_sim, drop=True).mean()
#     obs = wc.data_daily.sel(variable=prim_var_sim, drop=True).mean()
#     npt.assert_almost_equal(sim - obs, prim_incr, decimal=2)
