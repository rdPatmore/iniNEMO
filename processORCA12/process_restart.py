import xarray as xr
import gsw

def convert_to_TEOS10(ds, temperature='votemper', salinity='vosaline',
                      ssh='sossheig'):
    '''
    converts T and S to conform with TESO10
     -> potential temperature to conservative temperature
     -> practical salinity to absolute salinity
    '''

    p = gsw.p_from_z(ds[ssh], ds.nav_lat)
    ds[salinity] = gsw.conversions.SA_from_SP(ds[salinity], p,
                                              ds.nav_lon, ds.nav_lat)
    ds[salinity].attrs['long_name'] = 'Absolute Salinity'
    ds[temperature] = gsw.conversions.CT_from_pt(ds[salinity], ds[temperature])
    ds[temperature].attrs['long_name'] = 'Conservative Temperature'
    
    return ds

def de_nan_and_name(TEOS10=False):
    zps = 1
    ds = xr.open_dataset('DataIn/restart.nc')
    cfg = xr.open_dataset('../SourceData/domain_cfg.nc')
    # shift up ssh
    ds['sossheig'] = ds.sossheig - ds.sossheig.mean(skipna=True)
    ds = ds.fillna(0.0)
    #ds = ds.squeeze('T')
    ds = ds.drop('deptht')

    if TEOS10:
        convert_to_TEOS10(ds)

    if zps:
        ds['nav_lev'] = cfg.nav_lev.rename({'z':'Z'})
        ds = ds.set_coords('nav_lev')
        ds = ds.swap_dims({'Z':'nav_lev'}).drop('Z')

    ds = ds.rename({'sossheig':'sshn',
                    'votemper':'tn',
                    'vosaline':'sn',
                    'vozocrtx':'un',
                    'vomecrty':'vn'})
    

    ds.to_netcdf('DataOut/restart_conform.nc')
    
de_nan_and_name(TEOS10=True)
