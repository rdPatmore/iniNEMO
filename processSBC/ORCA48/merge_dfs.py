import xarray as xr

a = xr.open_mfdataset('DFS5.2_03_y2013*', chunks={'time':1})
a.to_netcdf('DFS5.2_03_y2013_merged.nc', unlimited_dims='time')
