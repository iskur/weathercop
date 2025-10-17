import numpy as np
import xarray as xr

station_names = [0, 1]
times = np.arange(5)
varnames = ["a", "b"]

# data_dict = {station_name: np.full((len(varnames), len(times)),
#                                    station_name)
#              for station_name in station_names}
data = np.empty((len(station_names), len(varnames), len(times)))
data[0] = 0
data[1] = 1

xar = xr.DataArray(data,
                   coords=(station_names, varnames, times),
                   dims=("station", "variable", "time")
                   )

print(xar)
print(xar.stack(stacked=("time", "station")))
print(xar.stack(stacked=("station", "time")))

