import xarray as xr

def de_nan_and_name():
    ds = xr.open_dataset('DataIn/restart.nc')
    ds = ds.fillna(0.0)
    ds = ds.squeeze('T')
    ds = ds.rename({'sossheig':'sshn',
                    'votemper':'tn',
                    'vosaline':'sn',
                    'vozocrtx':'un',
                    'vomecrty':'vn'})

    ds.to_netcdf('DataOut/restart_conform.nc')
    
de_nan_and_name()
