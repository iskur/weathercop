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
