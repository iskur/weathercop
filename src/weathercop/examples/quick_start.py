import xarray as xr
from weathercop.example_data import get_example_dataset_path, get_dwd_config
from weathercop.multisite import Multisite, set_conf

# Configure VarWG with DWD settings
set_conf(get_dwd_config())

# Load example multisite weather data as xarray Dataset
# This dataset contains 3 stations with temperature, precipitation, and radiation
xds = xr.open_dataset(get_example_dataset_path())

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
fig_meteogram = wc.plot_ensemble_meteogram_daily()
fig_qq = wc.plot_ensemble_qq()

