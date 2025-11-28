"""Tests for configuration flags."""
import pytest
from weathercop import cop_conf


def test_memory_flags_exist():
    """Verify memory optimization flags exist in configuration."""
    assert hasattr(cop_conf, 'SKIP_INTERMEDIATE_RESULTS_TESTING')
    assert hasattr(cop_conf, 'AGGRESSIVE_CLEANUP')


def test_memory_flags_are_boolean():
    """Verify memory optimization flags are boolean types."""
    assert isinstance(cop_conf.SKIP_INTERMEDIATE_RESULTS_TESTING, bool)
    assert isinstance(cop_conf.AGGRESSIVE_CLEANUP, bool)


def test_memory_flags_set_in_testing(configure_for_testing):
    """Verify flags are enabled during testing."""
    assert cop_conf.SKIP_INTERMEDIATE_RESULTS_TESTING is True
    assert cop_conf.AGGRESSIVE_CLEANUP is True


def test_dwd_config_applied():
    """Verify that DWD config is applied to VarWG before transformations.

    This test catches regressions where tox or other import contexts
    cause VarWG to use default config instead of DWD config.
    """
    import varwg
    from weathercop.configs import get_dwd_vg_config

    dwd_conf = get_dwd_vg_config()

    # Check that relative humidity uses empirical distribution (KDE)
    # This is the key difference between DWD and template config
    assert varwg.conf.dists['rh'] == 'empirical', (
        f"VarWG is using wrong config. "
        f"Expected 'rh': 'empirical' (DWD), "
        f"got 'rh': {varwg.conf.dists['rh']} (likely template config)"
    )

    # Verify the config came from DWD
    assert varwg.conf is dwd_conf
