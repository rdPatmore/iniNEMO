import xarray as xr

a = xr.open_mfdataset('DFS5.2_03_y2013_*', chunks={'time':1})
#b = xr.open_dataset('DFS5.2_03_y2013merged.nc', chunks={'time':1})
#a['time'] = (b.time * 3600).astype('float64')
#a.time.attrs['units'] = 'seconds since 2013-01-01 03:00:00'
a.to_netcdf('DFS5.2_03_y2013_merged.nc', unlimited_dims='time')
