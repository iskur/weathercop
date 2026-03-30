"""Pre-generate VARWG cache for memory-intensive tests.

Without a pre-built cache, each test worker fits VG models from scratch with a
non-deterministic random state, making statistical tests like
test_sim_mean_increase flaky (the model parameters affect the back-transformed
mean, so ``sim - obs != theta_incr`` by a small but consistent margin).

Running this script once before the test suite ensures all workers load the same
cached VG models (identified by fixtures/vg_cache_ready).
"""

from pathlib import Path

import varwg
import xarray as xr

from weathercop.configs import get_dwd_vg_config
from weathercop.multisite import Multisite, set_conf

set_conf(get_dwd_vg_config())
varwg.reseed(0)

cache_dir = Path(__file__).parent.parent / "src" / "weathercop" / "tests" / "fixtures"
xds = xr.open_dataset(cache_dir / "multisite_testdata.nc")

print("Fitting VARWG instances and writing cache to:", cache_dir)
wc = Multisite(
    xds,
    verbose=True,
    refit=True,
    reinitialize_vgs=True,
    infilling="vg",
    fit_kwds=dict(seasonal=True),
    vgs_cache_dir=cache_dir,
)
wc.close()

(cache_dir / "vg_cache_ready").touch()
print("Done. VG cache ready.")
