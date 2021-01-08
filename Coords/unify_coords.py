import xarray as xr

temp = xr.load_dataset('data_1m_potential_temperature_nomask.nc')
salt = xr.load_dataset('data_1m_salinity_nomask.nc')

time_source = xr.load_dataset('sst_data.nc')
coords_source = xr.load_dataset('coordinates.nc',decode_times=False)

#salt = salt.assign_coords(time_counter=time_source.time_counter)
salt['nav_lon'] = coords_source.nav_lon
salt['nav_lat'] = coords_source.nav_lat
#salt.time_counter.attrs.update({'units': 'seconds since 1999-01-01'})
salt = salt.assign_coords(time_counter=temp.time_counter)
temp['nav_lon'] = coords_source.nav_lon
temp['nav_lat'] = coords_source.nav_lat
temp = temp.drop_vars(['lat','lon'])


temp0=temp.isel(time_counter=0).squeeze()
salt0=salt.isel(time_counter=0).squeeze()
print (salt0)

temp.to_netcdf('temperature_1_year.nc')
salt.to_netcdf('salinity_1_year.nc')
temp0.to_netcdf('temperature_ini.nc')
salt0.to_netcdf('salinity_ini.nc')
