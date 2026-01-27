"""Integration test verifying Quick Start example code works end-to-end."""

import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from weathercop.example_data import get_example_dataset_path, get_dwd_config
from weathercop.multisite import Multisite, set_conf


def test_quick_start_imports_work():
    """Verify all imports from Quick Start example are available."""
    # These should not raise ImportError
    from weathercop.example_data import (
        get_example_dataset_path,
        get_dwd_config,
    )
    from weathercop.multisite import Multisite, set_conf

    # Verify functions are callable
    assert callable(get_example_dataset_path)
    assert callable(get_dwd_config)
    assert callable(set_conf)


def test_tangled_quick_start_example_executes():
    """Execute the tangled quick_start.py example from README.org.

    This test ensures that the code documented in README.org actually works
    end-to-end. By executing the tangled file directly, we verify:
    - The docs contain working, tested code
    - Any changes to the example are automatically tested
    - The test always matches the documentation exactly
    """
    # Locate the tangled quick start script
    quick_start_file = (
        Path(__file__).parent.parent / "examples" / "quick_start.py"
    )

    if not quick_start_file.exists():
        raise FileNotFoundError(
            f"Tangled quick start script not found at {quick_start_file}. "
            "Run: pandoc README.org --lua-filter=scripts/tangle.lua"
        )

    # Execute the tangled script in a controlled namespace
    namespace = {
        "xr": xr,
        "get_example_dataset_path": get_example_dataset_path,
        "get_dwd_config": get_dwd_config,
        "Multisite": Multisite,
        "set_conf": set_conf,
        "plt": plt,
    }

    with open(quick_start_file) as f:
        code = f.read()

    try:
        exec(code, namespace)
    except Exception as e:
        raise AssertionError(
            f"Quick Start example failed to execute: {e}"
        ) from e

    # Save generated plots to documentation
    plots_dir = quick_start_file.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for plot_name in ("meteogram", "qq"):
        fig_dict = namespace[f"fig_{plot_name}"]
        for station_i, (fig, axs) in enumerate(fig_dict.values()):
            fig.savefig(
                plots_dir / f"ensemble_{plot_name}_{station_i}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
