import xarray as xr
import gsw

def convert_to_TEOS10(ds, temperature='votemper', salinity='vosaline',
                                  ssh='sossheig'):
    '''
    converts T and S to conform with TESO10
     -> potential temperature to conservative temperature
     -> practical salinity to absolute salinity
    '''

    print (ds)
    p = gsw.p_from_z(ds[ssh], ds.nav_lat)
    ds[salinity] = gsw.conversions.SA_from_SP(ds[salinity], p,
                                              ds.nav_lon, ds.nav_lat)
    ds[salinity].attrs['long_name'] = 'Absolute Salinity'
    ds[temperature] = gsw.conversions.CT_from_pt(ds[salinity], ds[temperature])
    ds[temperature].attrs['long_name'] = 'Conservative Temperature'
    
    return ds

def TEOS10_bdy():
    ''' conform bdy to TEOS10 '''

    # load
    ds = xr.open_dataset('BdyOut/bdy_T_ring.nc')

    # convert
    convert_to_TEOS10(ds)

    # save
    path = 'BdyOut/bdy_T_ring_TEOS10.nc'
    ds.to_netcdf(path, unlimited_dims='time_counter')

TEOS10_bdy()
