import xarray as xr

xds = (
    xr.open_dataset(
        "/home/dirk/data/opendata_dwd/"
        "multisite_data_clean.nc"
        # "/home/dirk/data/opendata_dwd/"
        # "multisite_st_simulation.nc"
    )
    # .sel(station=["Hohenpeißenberg",
    #               "Regensburg",
    #               "Kempten",
    #               "Stötten"])
    .isel(
        station=slice(-5, None),
        # variable=slice(3),
    ).sel(
        time=slice("2010", "2018"),
        # variable=("R", "theta", "sun")
    )
)
xds = xds.sel(station=xds.station[::-1])

for key in xds:
    if key in ("latitude", "longitude"):
        continue
    if key not in xds.coords["station"]:
        xds = xds.drop(key)
xds = xds.interpolate_na("time")
# xds = xds.rename_dims(met_variable="variable").rename_vars(
#     met_variable="variable"
# )
print(xds)
xds.to_netcdf("/home/dirk/data/opendata_dwd/multisite_testdata.nc")
