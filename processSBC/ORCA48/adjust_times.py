import xarray as xr
import numpy as np

ds = xr.open_dataset('DFS5.2_03_y2013_merged.nc', decode_cf=False)
ds['time'] = (ds.time * 3600).astype('float64')
ds.time.attrs['units'] = 'seconds since 2013-01-01 03:00:00'
ds.to_netcdf('DFS5.2_03_y2013_merged_formatted_time.nc')
