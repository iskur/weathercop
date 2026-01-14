import warnings
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
from scipy import integrate

import matplotlib.pyplot as plt
import varwg
from weathercop import copulae as cop


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def eps():
    """Numerical epsilon for copula boundary tests."""
    return np.array([1e-5])


@pytest.fixture(scope="module")
def img_dir(tmp_path_factory):
    """Temporary directory for test diagnostic images (module-scoped).

    Uses pytest's tmp_path_factory to create an isolated temporary directory
    that's automatically cleaned up after the test session.
    """
    img_path = tmp_path_factory.mktemp("copula_test_images")
    return img_path


@pytest.fixture(scope="module")
def all_copulas():
    """All unfrozen copula classes for boundary and fitting tests."""
    # Optional: filter to specific copulas for debugging
    test_cop = "all"  # Change to copula name (e.g., "nelsen15") for debugging

    if test_cop != "all":
        from collections import OrderedDict
        filtered = OrderedDict(
            (name, obj) for name, obj in cop.all_cops.items()
            if test_cop in name
        )
        print(f"Testing only {list(filtered.keys())}")
        return filtered
    return cop.all_cops


@pytest.fixture(scope="module")
def frozen_copulas():
    """All frozen copula instances for evaluation tests."""
    test_cop = "all"  # Change to copula name for debugging

    if test_cop != "all":
        from collections import OrderedDict
        filtered = OrderedDict(
            (name, obj) for name, obj in cop.frozen_cops.items()
            if test_cop in name
        )
        return filtered
    return cop.frozen_cops


# ============================================================================
# Parametrized Test Functions
# ============================================================================


@pytest.mark.parametrize("copula_name,copula",
                         list(cop.all_cops.items()),
                         ids=list(cop.all_cops.keys()))
def test_bounds(copula_name, copula, eps):
    """Can we evaluate densities on theta bounds?"""
    uu = (
        np.linspace(eps, 1 - eps, 100)
        .repeat(100)
        .reshape(100, 100)
    )
    vv = np.tile(np.linspace(eps, 1 - eps, 100), 100)

    for theta in np.array(copula.theta_bounds).T:
        frozen_cop = copula(theta)
        dens = frozen_cop.density(uu, vv)
        assert np.all(np.isfinite(dens)), \
            f"{copula_name}: Non-finite density at theta={theta}"


@pytest.mark.parametrize("copula_name,frozen_cop",
                         list(cop.frozen_cops.items()),
                         ids=list(cop.frozen_cops.keys()))
def test_cop(copula_name, frozen_cop, eps, img_dir):
    """A few very basic copula requirements."""
    zero = frozen_cop.copula_func(eps, eps)
    one = frozen_cop.copula_func(1 - eps, 1 - eps)

    try:
        npt.assert_almost_equal(zero, eps, decimal=4)
        npt.assert_almost_equal(one, 1 - eps, decimal=4)
    except AssertionError:
        print(f"{copula_name}: zero={zero}, expected={eps}")
        print(f"{copula_name}: one={one}, expected={1 - eps}")
        fig, ax = frozen_cop.plot_copula()
        fig.savefig(img_dir / f"copula_{copula_name}.png")
        plt.close(fig)
        raise


@pytest.mark.parametrize("copula_name,copula",
                         list(cop.frozen_cops.items()),
                         ids=list(cop.frozen_cops.keys()))
def test_cdf_given_u(copula_name, copula, eps, img_dir):
    """Testing u-conditional cdfs by roundtrip."""
    kind_str = (
        "symbolic"
        if hasattr(copula, "inv_cdf_given_uu_expr")
        else "numeric"
    )

    # Test boundary conditions with adaptive epsilon
    test_eps = np.array([1e-9])
    zero = copula.cdf_given_u(1 - eps, test_eps)
    while zero > eps and test_eps > 1e-17:
        test_eps *= 0.1
        zero = copula.cdf_given_u(1 - eps, test_eps)

    test_eps = np.array([1e-9])
    one = copula.cdf_given_u(eps, 1 - test_eps)
    while one < 1 - eps and test_eps > 1e-17:
        test_eps *= 0.1
        one = copula.cdf_given_u(eps, 1 - test_eps)

    try:
        npt.assert_almost_equal(zero, eps, decimal=4)
        npt.assert_almost_equal(one, 1 - eps, decimal=4)
    except AssertionError:
        print(f"{copula_name}: zero={zero}, expected={eps}")
        print(f"{copula_name}: one={one}, expected={1 - eps}")
        warnings.warn(copula_name.upper())
        fig, axs = copula.plot_cop_dens()
        fig.suptitle(copula_name)
        fig.savefig(img_dir / f"cop_dens_{copula_name}")
        plt.close(fig)

    # Test monotonicity of conditional CDF
    eps_val = eps[0]
    qq = np.linspace(eps_val, 1 - eps_val, 200)
    ranks_u_low = np.full_like(qq, eps_val)
    ranks_u_high = np.full_like(qq, 1 - eps_val)
    ranks_v_low = copula.inv_cdf_given_u(ranks_u_low, qq)
    ranks_v_high = copula.inv_cdf_given_u(ranks_u_high, qq)

    low_cdf_func = lambda v: copula.cdf_given_u(
        np.full_like(v, 0.01), v
    )
    high_cdf_func = lambda v: copula.cdf_given_u(
        np.full_like(v, 0.99), v
    )

    try:
        npt.assert_array_less(1e-19, np.diff(low_cdf_func(qq)))
    except AssertionError:
        fig, ax = plt.subplots(
            nrows=1, ncols=1, subplot_kw=dict(aspect="equal")
        )
        low_vv = low_cdf_func(qq)
        ax.plot(qq, low_vv)
        ax.scatter(ranks_v_low, np.zeros_like(qq), marker="x")
        for rank_v, q in zip(ranks_v_low, qq):
            ax.plot([0, rank_v], 2 * [q], color=(0, 0, 0, 0.1))
            ax.plot([rank_v, rank_v], [0, q], color=(0, 0, 0, 0.1))
        ax.scatter(np.zeros_like(qq), qq, marker="x")
        ax.set_title(f"{copula.name} ({kind_str})")
        fig.savefig(img_dir / f"cdf_given_u_low_{copula_name}.png")
        plt.close(fig)
        raise

    try:
        npt.assert_array_less(1e-19, np.diff(high_cdf_func(qq)))
    except AssertionError:
        fig, ax = plt.subplots()
        ax.plot(qq, high_cdf_func(qq))
        ax.scatter(ranks_v_high, np.zeros_like(qq), marker="x")
        ax.scatter(np.zeros_like(qq), qq, marker="x")
        ax.set_title(copula.name)
        fig.savefig(img_dir / f"cdf_given_u_high_{copula_name}.png")
        plt.close(fig)
        raise


@pytest.mark.parametrize("copula_name,copula",
                         list(cop.frozen_cops.items()),
                         ids=list(cop.frozen_cops.keys()))
def test_cdf_given_v(copula_name, copula, eps, img_dir):
    """Testing v-conditional cdfs by roundtrip."""
    kind_str = (
        "symbolic"
        if hasattr(copula, "inv_cdf_given_vv_expr")
        else "numeric"
    )

    # Test boundary conditions with adaptive epsilon
    test_eps = np.array([1e-9])
    zero = copula.cdf_given_v(test_eps, 1 - eps)
    while zero > eps and test_eps > 1e-17:
        test_eps *= 0.1
        zero = copula.cdf_given_v(test_eps, 1 - eps)

    test_eps = np.array([1e-9])
    one = copula.cdf_given_v(1 - test_eps, eps)
    while one < 1 - eps and test_eps > 1e-17:
        test_eps *= 0.1
        one = copula.cdf_given_v(1 - test_eps, eps)

    try:
        npt.assert_array_less(zero, eps)
        npt.assert_array_less(1 - eps, one)
    except AssertionError:
        print(f"{copula_name}: zero={zero}, expected<{eps}")
        print(f"{copula_name}: one={one}, expected>{1 - eps}")
        warnings.warn(copula_name.upper())
        fig, ax = copula.plot_cop_dens()
        fig.suptitle(copula_name)
        fig.savefig(img_dir / f"cop_dens_{copula_name}")
        plt.close(fig)
        raise

    # Test monotonicity of conditional CDF
    eps_val = eps[0]
    qq = np.linspace(eps_val, 1 - eps_val, 200)
    ranks_v_low = np.full_like(qq, eps_val)
    ranks_v_high = np.full_like(qq, 1 - eps_val)
    ranks_u_low = copula.inv_cdf_given_v(ranks_v_low, qq)
    ranks_u_high = copula.inv_cdf_given_v(ranks_v_high, qq)

    low_cdf_func = lambda u: copula.cdf_given_v(
        u, np.full_like(u, 0.01)
    )
    high_cdf_func = lambda u: copula.cdf_given_v(
        u, np.full_like(u, 0.99)
    )

    try:
        npt.assert_array_less(1e-19, np.diff(low_cdf_func(qq)))
    except AssertionError:
        fig, ax = plt.subplots(
            nrows=1, ncols=1, subplot_kw=dict(aspect="equal")
        )
        low_uu = low_cdf_func(qq)
        ax.plot(qq, low_uu)
        ax.scatter(ranks_u_low, np.zeros_like(qq), marker="x")
        for rank_u, q in zip(ranks_u_low, qq):
            ax.plot([0, rank_u], 2 * [q], color=(0, 0, 0, 0.1))
            ax.plot([rank_u, rank_u], [0, q], color=(0, 0, 0, 0.1))
        ax.scatter(np.zeros_like(qq), qq, marker="x")
        ax.set_title(f"{copula.name} ({kind_str})")
        fig.savefig(img_dir / f"cdf_given_v_low_{copula_name}.png")
        plt.close(fig)
        raise

    try:
        npt.assert_array_less(1e-19, np.diff(high_cdf_func(qq)))
    except AssertionError:
        fig, ax = plt.subplots()
        ax.plot(qq, high_cdf_func(qq))
        ax.scatter(ranks_u_high, np.zeros_like(qq), marker="x")
        ax.scatter(np.zeros_like(qq), qq, marker="x")
        ax.set_title(copula.name)
        fig.savefig(img_dir / f"cdf_given_v_high_{copula_name}.png")
        plt.close(fig)
        raise


@pytest.mark.parametrize("copula_name,copula",
                         list(cop.frozen_cops.items()),
                         ids=list(cop.frozen_cops.keys()))
def test_inv_cdf_given_u(copula_name, copula, eps, img_dir):
    """Testing inverse u-conditional cdfs by roundtrip."""
    kind_str = (
        "symbolic"
        if hasattr(copula, "inv_cdf_given_uu_expr")
        else "numeric"
    )

    uu = np.squeeze(np.linspace(eps, 1 - eps, 100))
    vv_exp = np.copy(uu)

    for i, u in enumerate(uu):
        u = np.full_like(vv_exp, u)
        qq = copula.cdf_given_u(u, vv_exp)
        vv_actual = copula.inv_cdf_given_u(u, qq)

        try:
            npt.assert_almost_equal(vv_actual, vv_exp, decimal=2)
        except AssertionError:
            fig, ax = plt.subplots()
            ax.plot(vv_exp, vv_actual, "-x")
            ax.plot([0, 1], [0, 1], color="k")
            ax.set_xlabel("vv_exp")
            ax.set_ylabel("vv_actual")
            ax.set_title(
                f"{copula_name} ({kind_str}) inv_cdf_given_u "
                f"u={np.squeeze(u)[0]:.6f}"
            )
            fig.savefig(
                img_dir / f"inv_cdf_given_u_{copula_name}_{i:03d}.png"
            )
            plt.close(fig)
            print(f"test failed for {copula_name} at i={i}")
            raise


@pytest.mark.parametrize("copula_name,copula",
                         list(cop.frozen_cops.items()),
                         ids=list(cop.frozen_cops.keys()))
def test_inv_cdf_given_v(copula_name, copula, eps, img_dir):
    """Testing inverse v-conditional cdfs by roundtrip."""
    kind_str = (
        "symbolic"
        if hasattr(copula, "inv_cdf_given_vv_expr")
        else "numeric"
    )

    vv = np.squeeze(np.linspace(eps, 1 - eps, 100))
    uu_exp = np.copy(vv)

    for i, v in enumerate(vv):
        v = np.full_like(uu_exp, v)
        qq = copula.cdf_given_v(uu_exp, v)

        try:
            npt.assert_array_less(0, qq + 1e-12)
            npt.assert_array_less(qq, 1 + 1e-12)
            uu_actual = copula.inv_cdf_given_v(v, qq)
            npt.assert_almost_equal(uu_actual, uu_exp, decimal=2)
        except AssertionError:
            fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
            ax.plot([0, 1], [0, 1], color="k")
            ax.plot(uu_exp, uu_actual, "-x")
            ax.set_xlabel("uu_exp")
            ax.set_ylabel("uu_actual")
            ax.set_title(
                f"{copula_name} ({kind_str}) inv_cdf_given_v "
                f"v={np.squeeze(v)[0]:.6f}"
            )
            fig.savefig(
                img_dir / f"inv_cdf_given_v_{copula_name}_{i:03d}.png"
            )
            plt.close(fig)
            # Note: Original had this as non-fatal, keeping that behavior
            # raise


@pytest.mark.parametrize("copula_name,frozen_cop",
                         list(cop.frozen_cops.items()),
                         ids=list(cop.frozen_cops.keys()))
def test_density(copula_name, frozen_cop, eps, img_dir):
    """Does the density integrate to 1?"""
    try:
        # Extract scalar from numpy array fixture
        eps_val = float(np.asarray(eps).flat[0])

        def density(x, y):
            # ufuncs return arrays; extract scalar for scipy.integrate
            result = frozen_cop.density(np.array([x]), np.array([y]))
            return float(np.asarray(result).flat[0])

        with warnings.catch_warnings():
            # Suppress scipy's internal deprecation warning about array-to-scalar
            # conversion from _quadpack. This is scipy's implementation detail.
            warnings.filterwarnings('ignore', category=DeprecationWarning,
                                    message='.*Conversion of an array.*')
            one = integrate.nquad(
                density,
                ([eps_val, 1 - eps_val], [eps_val, 1 - eps_val]),
                opts={'epsabs': 1e-4, 'epsrel': 1e-4},
            )[0]
    except integrate.IntegrationWarning:
        pytest.skip(f"Numerical integration of {copula_name} is problematic")
    else:
        try:
            npt.assert_almost_equal(one, 1.0, decimal=4)
        except AssertionError:
            fig, ax = frozen_cop.plot_cop_dens()
            fig.savefig(img_dir / f"density_{copula_name}.png")
            fig.suptitle(f"{copula_name} integral={one}")
            plt.close(fig)
            raise


@pytest.mark.parametrize("copula_name,copula",
                         list(cop.all_cops.items()),
                         ids=list(cop.all_cops.keys()))
def test_fit(copula_name, copula, img_dir):
    """Is fit able to reproduce parameters of a self-generated sample?"""
    varwg.reseed(1)

    sample_x, sample_y = copula.sample(10000, *copula.theta_start)

    try:
        fitted_theta = copula.fit(sample_x, sample_y)
    except cop.NoConvergence:
        pytest.skip(f"{copula_name}: fitting did not converge")
    else:
        if isinstance(copula, cop.Independence):
            pytest.skip("Independence copula has no parameters to fit")

        try:
            npt.assert_almost_equal(
                fitted_theta, copula.theta_start, decimal=1
            )
        except AssertionError:
            fig, ax = copula.plot_density()
            ax.scatter(
                sample_x,
                sample_y,
                marker="x",
                facecolor=(0, 0, 1, 0.1),
            )
            fig2, ax2 = copula.plot_cop_dens(scatter=True)
            fig2.savefig(img_dir / f"cop_dens_{copula_name}.png")
            plt.close(fig)
            plt.close(fig2)
            raise


@pytest.mark.parametrize(
    "copula_name,copula",
    [
        ("gaussian", cop.gaussian),
        ("gumbel", cop.gumbel),
        ("clayton", cop.clayton),
        ("independence", cop.independence),
    ],
)
def test_fitted_pickle(copula_name, copula):
    """Test that Fitted instances pickle/unpickle correctly.

    Regression test for issue where Fitted.__setstate__ used globals()
    which could resolve to wrong objects when unpickling in different
    contexts (e.g., installed package vs source).
    """
    import pickle

    varwg.reseed(42)

    # Generate sample data and create a Fitted instance
    ranks_u = np.random.random(100)
    ranks_v = np.random.random(100)
    fitted = copula.generate_fitted(ranks_u, ranks_v)

    # Store original copula reference
    original_copula = fitted.copula

    # Verify copula is an instance, not a class
    assert not isinstance(original_copula, type), \
        f"fitted.copula should be an instance, got {type(original_copula)}"

    # Pickle and unpickle
    pickled_data = pickle.dumps(fitted)
    unpickled = pickle.loads(pickled_data)

    # Verify copula was restored correctly
    assert unpickled.copula is original_copula, \
        f"Unpickled copula should be the same singleton instance"

    # Verify copula is still an instance, not corrupted
    assert not isinstance(unpickled.copula, type), \
        f"unpickled.copula should be an instance, got {type(unpickled.copula)}"

    # Verify copula has expected attributes
    assert hasattr(unpickled.copula, "inv_cdf_given_u"), \
        f"unpickled.copula missing inv_cdf_given_u method"
    assert hasattr(unpickled.copula, "_inverse_conditional"), \
        f"unpickled.copula missing _inverse_conditional method"

    # Verify methods work correctly after unpickling
    test_u = np.array([0.5])
    test_q = np.array([0.5])
    if not isinstance(copula, cop.Independence):
        test_theta = fitted.theta[0][0]
    else:
        test_theta = 0.0

    try:
        result = unpickled.copula.inv_cdf_given_u(test_u, test_q, test_theta)
        assert np.isfinite(result).all(), \
            f"inv_cdf_given_u returned non-finite values after unpickling"
    except AttributeError as e:
        pytest.fail(
            f"Method call failed after unpickling {copula_name}: {e}"
        )


# ============================================================================
# Module execution (for legacy compatibility)
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
