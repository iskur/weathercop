"""Common assertion utilities for weathercop tests."""

import numpy as np
import xarray as xr


def assert_valid_ensemble_structure(ensemble):
    """Verify ensemble dataset has required xarray structure and dimensions.

    Parameters
    ----------
    ensemble : xr.Dataset
        The ensemble dataset to validate.

    Raises
    ------
    AssertionError
        If dataset is not an xarray Dataset or missing required dimensions.
    """
    assert isinstance(
        ensemble, (xr.core.dataset.Dataset, xr.core.dataarray.DataArray)
    ), f"Expected xr.Dataset or xr.DataArray, got {type(ensemble)}"

    required_dims = {"station", "variable", "time"}
    assert required_dims.issubset(
        ensemble.dims
    ), f"Missing dimensions. Has: {set(ensemble.dims)}, need: {required_dims}"


def assert_mean_preservation(
    simulated_data, observed_data, variable_name, tolerance=0.15
):
    """Verify simulated variable means are within tolerance of observed.

    Parameters
    ----------
    simulated_data : xr.DataArray
        Simulated variable values.
    observed_data : xr.DataArray
        Observed variable values.
    variable_name : str
        Name of variable being tested (for error messages).
    tolerance : float, default=0.15
        Acceptable relative difference in means (0.15 = 15%).

    Raises
    ------
    AssertionError
        If simulated mean differs from observed by more than tolerance.
    """
    sim_mean = float(simulated_data.mean())
    obs_mean = float(observed_data.mean())

    if obs_mean == 0:
        assert (
            sim_mean == obs_mean
        ), f"{variable_name}: simulated mean {sim_mean} != observed {obs_mean} (obs is zero)"
    else:
        relative_diff = abs(sim_mean - obs_mean) / abs(obs_mean)
        assert relative_diff <= tolerance, (
            f"{variable_name}: simulated mean {sim_mean:.3f} differs from "
            f"observed {obs_mean:.3f} by {relative_diff:.1%} (tolerance: {tolerance:.1%})"
        )


def assert_correlation_preservation(
    simulated_data, observed_data, variable_names, tolerance=0.15
):
    """Verify simulated correlations are within tolerance of observed.

    Parameters
    ----------
    simulated_data : xr.Dataset
        Simulated dataset with multiple variables.
    observed_data : xr.Dataset
        Observed dataset with multiple variables.
    variable_names : list of str
        Names of variables to correlate.
    tolerance : float, default=0.15
        Acceptable relative difference in correlations (0.15 = 15%).

    Raises
    ------
    AssertionError
        If any simulated correlation differs from observed by more than tolerance.
    """
    # Extract variables that exist in both datasets
    available_vars = [
        var
        for var in variable_names
        if var in simulated_data and var in observed_data
    ]

    # Need at least 2 variables to compute correlations
    if len(available_vars) < 2:
        return

    sim_values = np.column_stack(
        [simulated_data[var].values.ravel() for var in available_vars]
    )
    obs_values = np.column_stack(
        [observed_data[var].values.ravel() for var in available_vars]
    )

    # Compute correlations
    sim_corr = np.corrcoef(sim_values.T)
    obs_corr = np.corrcoef(obs_values.T)

    # Handle case where there's only one variable (corrcoef returns scalar)
    if len(available_vars) == 1:
        return

    # Compare off-diagonal elements
    n = len(available_vars)
    for i in range(n):
        for j in range(i + 1, n):
            sim_r = sim_corr[i, j]
            obs_r = obs_corr[i, j]
            relative_diff = abs(sim_r - obs_r)
            assert relative_diff <= tolerance, (
                f"Correlation {available_vars[i]}-{available_vars[j]}: "
                f"simulated {sim_r:.3f} differs from observed {obs_r:.3f} "
                f"by {relative_diff:.3f} (tolerance: {tolerance:.3f})"
            )
