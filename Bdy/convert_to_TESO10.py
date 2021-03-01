import xarray as xr
import gsw

def convert_to_TEOS10(ds, temperature='votemper', salinity='vosaline'):
    '''
    converts T and S to conform with TESO10
     -> potential temperature to conservative temperature
     -> practical salinity to absolute salinity
    '''

    sbc = xr.open_dataset('../processSBC/DFS3.2

    ds[salinity] = gsw.conversions.SA_from_SP(ds[salinity], sbc.slp,
                                              ds.nav_lon, ds.nav_lat)
    ds[salinity].attrs['long_name'] = 'Absolute Salinity'
    ds[temperature] = gsw.conversions.CT_from_pt(da[salinity], ds[temperature])
    ds[temperature].attrs['long_name'] = 'Conservative Temperature'
    
    return ds

def TEOS10_bdy():
    ''' conform bdy to TEOS10 '''

    for pos in {'T','U','V'}:
        # load
        ds = xr.open_dataset('BdyOut/bdy_' + pos + '_ring.nc')

        # convert
        convert_to_TEOS10(ds)

        # save
        path = 'BdyOut/bdy_' + pos + '_ring_TEOS10.nc'
        ds.to_netcdf(path, unlimited_dims='time_counter')

TEOS10_bdy()
