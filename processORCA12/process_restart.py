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
    ds = xr.open_dataset('DataIn/restart12.nc')

    if TEOS10:
        convert_to_TEOS10(ds)

    ds = ds.rename({'sossheig':'sshn',
                    'votemper':'tn',
                    'vosaline':'sn',
                    'vozocrtx':'un',
                    'vomecrty':'vn'})
    var_list = ['sshn', 'tn', 'un', 'vn']
    for var in var_list:
        ds[var] = ds[var].fillna(0.0)
    
    ds['sn'] = ds['sn'].fillna(34.0)
    ds.to_netcdf('DataOut/restart_conform.nc', unlimited_dims='T')

def de_nan_and_name_ice():

    ds = xr.open_dataset('DataIn/restart12_ice.nc')

    for var in ['siconc', 'sithic', 'snthic']:
        print ('var', var)
        ds[var] = ds[var].fillna(0.0)
    
    ds.to_netcdf('DataOut/restart_ice_conform.nc', unlimited_dims='T')
    
de_nan_and_name()
    
