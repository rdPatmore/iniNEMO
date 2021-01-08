import xarray as xr

ds = xr.open_dataset('mesh_zgr.nc', chunks={'z':10})

ds.attrs['DOMAIN_number_total'] = 1

ds.to_netcdf('mesh_zgr_updated.nc')
