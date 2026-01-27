import xarray as xr
import varwg as vg
from weathercop.multisite import Multisite, set_conf

# Configure VarWG (e.g., with your config module)
import your_vg_conf
set_conf(your_vg_conf)

# Load multisite weather data as xarray Dataset
# Expected dimensions: (time, station, variable)
xds = xr.open_dataset("path/to/multisite_data.nc")

# Initialize the multisite weather generator
wc = Multisite(
    xds,
    verbose=True,
)

# Generate a single realization
sim_result = wc.simulate()

# Generate an ensemble of 20 realizations
wc.simulate_ensemble(20)

# Visualize results
wc.plot_ensemble_stats()
wc.plot_ensemble_meteogram_daily()
wc.plot_ensemble_qq()

