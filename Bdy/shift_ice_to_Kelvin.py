import xarray as xr

def convert_to_Kelvin(ds, temperature='votemper'):

    ds[temperature] = ds[temperature] + 273.15
    ds[temperature].attrs['units'] = 'Kelvin'
    
    return ds

def kelvin_ice_bdy(year=''):
    ''' conform bdy to TEOS10 '''

    # load
    ds = xr.open_dataset('../DataOut/ORCA12/bdy_I_ring' + year + '.nc',
            chunks={'time_counter':1})#, 'xbt':10})

    # convert
    convert_to_Kelvin(ds, temperature='sitemp')

    # fill nan with frozen temp
    ds['sitemp'] = ds.sitemp.fillna(270)

    # save
    path = '../DataOut/ORCA12/bdy_I_ring' + year + '_kelvin.nc'
    ds.to_netcdf(path, unlimited_dims='time_counter')

kelvin_ice_bdy(year='_y2014')
