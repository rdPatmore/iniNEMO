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
    indir = '/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/SOCHIC_48/'
    ds = xr.open_dataset(indir + 'restart_y2012m01.nc')

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
    ds.to_netcdf('../DataOut/ORCA48/restart_conform.nc', unlimited_dims='T')

def de_nan_and_name_ice():

    indir = '/work/n02/n02/ryapat30/nemo/nemo/tools/SIREN/SOCHIC_12/'
    ds = xr.open_dataset(indir + 'restart_ice_y2012m01.nc')

    for var in ['siconc', 'sithic', 'snthic', 'sitemp', 'u_ice', 'v_ice']:
        print ('var', var)
        ds[var] = ds[var].fillna(0.0)
    ds['sisalt'] = ds['sisalt'].fillna(6.3) # set in g/m^3?
    ds['sitemp'] = ds['sitemp'] + 273.15 # convert to kelvin
    #ds['nn_fsbc'] = 1
    #s['kt_ice'] = 1

    #ds = ds.rename({'siconc': 'a_i',
    #                'sithic': 'v_i',
    #                'snthic': 'v_s',
    #                'sitemp': 't_su',
    #                'sisalt': 'sv_i'})
    
    ds.to_netcdf('../DataOut/restart_ice_conform.nc', unlimited_dims='T')
    
de_nan_and_name(TEOS10=False)
