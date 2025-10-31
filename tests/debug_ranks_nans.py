"""Debug script to identify NaN sources in rank calculation."""
import sys
import numpy as np
import xarray as xr
from weathercop.multisite import Multisite
from weathercop.configs import get_dwd_vg_config
from weathercop.cop_conf import set_conf
from pathlib import Path

def debug_ranks_initialization():
    """Instrument Multisite init to see where NaNs come from."""
    # Setup
    vg_conf = get_dwd_vg_config()
    set_conf(vg_conf)
    data_root = Path("src/weathercop/tests/data")
    xds = xr.open_dataset(data_root / "multisite_testdata.nc")

    # Create instance but catch before the assertion fails
    try:
        wc = Multisite(
            xds,
            verbose=False,
            refit=True,
            refit_vine=False,
            reinitialize_vgs=True,
            fit_kwds=dict(seasonal=True),
        )
    except AssertionError as e:
        print(f"AssertionError caught: {e}")
        # The error happens in _init_vgs at line 1136
        # We need to inspect the internal state before that point
        # This will require modifying multisite.py temporarily

    xds.close()

if __name__ == "__main__":
    debug_ranks_initialization()
