"""Debug tests to trace config setup issues in tox."""

import varwg
from weathercop.configs import get_dwd_vg_config


def test_debug_01_config_exists():
    """Check if varwg.conf exists."""
    assert hasattr(varwg, "conf"), "varwg.conf does not exist!"
    print(f"\n✓ varwg.conf exists: {varwg.conf}")


def test_debug_02_config_has_dists():
    """Check if config has dists."""
    assert hasattr(varwg.conf, "dists"), "varwg.conf.dists does not exist!"
    print(f"\n✓ varwg.conf.dists exists: {varwg.conf.dists}")


def test_debug_03_rh_dist():
    """Check what distribution is used for rh."""
    rh_dist = varwg.conf.dists.get("rh", "NOT FOUND")
    print(f"\n✓ varwg.conf.dists['rh'] = {rh_dist}")
    assert rh_dist == "empirical", (
        f"Expected 'empirical', got {rh_dist}. " f"This is the problem!"
    )


def test_debug_04_dwd_conf_object():
    """Check what dwd_conf looks like."""
    dwd_conf = get_dwd_vg_config()
    print(f"\n✓ dwd_conf object: {dwd_conf}")
    print(f"  - dwd_conf.dists['rh'] = {dwd_conf.dists['rh']}")
    assert dwd_conf.dists["rh"] == "empirical"


def test_debug_05_config_identity():
    """Check if varwg.conf is the same object as dwd_conf."""
    dwd_conf = get_dwd_vg_config()
    is_same = varwg.conf is dwd_conf
    print(f"\n✓ varwg.conf is dwd_conf: {is_same}")
    if not is_same:
        print(f"  - varwg.conf: {varwg.conf}")
        print(f"  - dwd_conf: {dwd_conf}")
        print(f"  - id(varwg.conf): {id(varwg.conf)}")
        print(f"  - id(dwd_conf): {id(dwd_conf)}")


def test_debug_06_module_paths():
    """Check where varwg modules are loaded from."""
    from varwg.core import core as vg_core
    import varwg.core.base as vg_base

    print(f"\n✓ varwg module path: {varwg.__file__}")
    print(f"✓ vg_core module path: {vg_core.__file__}")
    print(f"✓ vg_base module path: {vg_base.__file__}")

    # Check which config each module has
    print(f"✓ varwg.conf: {getattr(varwg, 'conf', 'NOT SET')}")
    print(f"✓ vg_core.conf: {getattr(vg_core, 'conf', 'NOT SET')}")
    print(f"✓ vg_base.conf: {getattr(vg_base, 'conf', 'NOT SET')}")
