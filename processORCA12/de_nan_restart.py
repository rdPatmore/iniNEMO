import xarray as xr

def de_nan_and_name():
    zps = 1
    ds = xr.open_dataset('DataIn/restart.nc')
    cfg = xr.open_dataset('../SourceData/domain_cfg.nc')
    # shift up ssh
    ds['sossheig'] = ds.sossheig - ds.sossheig.mean(skipna=True)
    ds = ds.fillna(0.0)
    #ds = ds.squeeze('T')
    ds = ds.drop('deptht')
    if zps:
        ds['nav_lev'] = cfg.nav_lev.rename({'z':'Z'})
        ds = ds.set_coords('nav_lev')
        ds = ds.swap_dims({'Z':'nav_lev'}).drop('Z')
    print (ds)
    ds = ds.rename({'sossheig':'sshn',
                    'votemper':'tn',
                    'vosaline':'sn',
                    'vozocrtx':'un',
                    'vomecrty':'vn'})
    

    ds.to_netcdf('DataOut/restart_conform.nc')
    
de_nan_and_name()
