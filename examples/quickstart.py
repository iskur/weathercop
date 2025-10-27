"""Quick start example: generate a small weather ensemble."""

import xarray as xr
from pathlib import Path
from weathercop.multisite import Multisite, set_conf
from weathercop.configs import get_dwd_vg_config

vg_conf = get_dwd_vg_config()

# Load test data
set_conf(vg_conf)
xds = xr.open_dataset(Path.home() / "data/opendata_dwd/multisite_testdata.nc")

# Initialize and generate ensemble
multisite = Multisite(xds, verbose=True)
# Keep ensemble in memory (avoids dask dependency for mfdataset loading)
ensemble = multisite.simulate_ensemble(n_realizations=5, name="example", write_to_disk=False)

# Visualize
fig_axs = multisite.plot_ensemble_meteogram_daily()
for station, (fig, axs) in fig_axs.items():
    fig.savefig(f"ensemble_{station}.png", dpi=100, bbox_inches='tight')

multisite.close()
print("Done! Check ensemble_*.png files")
