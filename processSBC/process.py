import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd


coord = xr.open_dataset('../SourceData/coordinates.nc', decode_times=False)
ECMWF = xr.open_dataset('../SourceData/ECMWF.nc')#.isel(time=slice(None,3))

def regrid(ds, coord, var):
    print ('variable: ', var)
    print ('time length', ds[var].time.values.shape)
    ds_new_grid = []
    for i, ds_iter in enumerate(ds[var]):
        print ('time', i)
        xi, xj = np.meshgrid(ds_iter.longitude.values,
                             ds_iter.latitude.values)
        points = np.squeeze(np.dstack((xi.ravel(), xj.ravel())))
        values = ds_iter.values.ravel()
        grid_x = coord.nav_lon.values
        grid_y = coord.nav_lat.values
        
        grid = griddata(points, values, (grid_x, grid_y))
        ds_new_grid.append(grid[None,:,:])
    
    ds_regridded_var = np.concatenate(ds_new_grid, axis=0)
    ds = ds.assign({var: (['time','Y', 'X'], ds_regridded_var, 
                                ds[var].attrs)})
    return ds

def clean_coords(ds):
    ds = ds.drop_dims(['latitude','longitude'])
    print (ds)
    ds = ds.assign({'nav_lon': coord.nav_lon, 'nav_lat': coord.nav_lat})
    return ds

#def calc_specific_humidity(ds):
#    T0 = 273.18 # Kelvin freezing point
#    Rd = 287 # gas constant for dry air
#    Rv = 461 # gas constant for water vapour
#
#    # Teten's constants
#    c0 = 6.1078
#    c1_neg = 21.875
#    c2_neg = 265.5 
#    c1_pos = 17.27
#    c2_pos = 237.3
#
#    def humid_formula(c0, c1, c2, T, T0):
#        return c0 * np.exp( c1 * (T - T0) / (T - T0 + c2))
# 
#    es = xr.where(ds.d2m >= 0, humid_formula(c0, c1_pos, c2_pos, ds['d2m'], T0),
#                               humid_formula(c0, c1_neg, c2_neg, ds['d2m'], T0))
#    R_ratio = Rd/Rv
#    humid = R_ratio * es / (ds.sp - (1-R_ratio) * es)
#    #humid = humid * 100 # convert to percentage
#    ds = ds.assign({'humid':humid})
#    return ds

def calc_specific_humidity(ds):
    ''' calculation of sepcific humitity following meom-group DFTOOlS'''

    # vapor pressure at saturation (hPa)
    psat = ( 10**(10.79574*(1 - 273.16/ds.d2m) - 5.028*np.log10(ds.d2m/273.16) 
                 + 1.50475*10**(-4)*(1 - 10**(-8.2969*(ds.d2m/273.16 - 1))) 
                 + 0.42873*10**(-3)*(10**(4.76955*(1 - 273.16/ds.d2m)) - 1)
                 + 0.78614) )
    # convert to Pa
    psat = psat * 100

    # calculate humidity    
    R_ratio = 0.62197
    humid = R_ratio * psat / (ds.sp - (1-R_ratio) * psat)
    humid.attrs = {'units':'kg/kg', 'long name': 'specific humidity'}
    ds = ds.assign({'humid':humid})
    return ds
    

def correct_units(ds):
    time_period = 10800 # seconds
    density = 1000 # kg * m^-3

    snow   = ds.sf * density# / time_period
    precip = ds.tp * density# / time_period
    precip.attrs['units'] = 'kg m**-2 s**-1'
    snow.attrs['units']   = 'kg m**-2 s**-1'

    short_wave = ds.ssrd# / time_period
    long_wave  = ds.strd# / time_period
    #short_wave = ds.ssrd
    #long_wave  = ds.strd
    short_wave.attrs['units']   = 'W m**-2'
    long_wave.attrs['units']   = 'W m**-2'

    #p = ds.sp / 100
    mslp = ds.msl / 100
    #p.attrs['units'] = 'hPa'
    mslp.attrs['units'] = 'hPa'

    ds = ds.assign({'tp': precip, 'sf': snow,
                    'ssrd': short_wave, 'strd': long_wave,
                    'mslp': mslp})
    #                'sp': p, 'mslp': mslp})
    #ds = ds.assign({'sp': p, 'mslp': mslp})
    return ds

def daily_average(ds, var_keys):
    ''' 
    return two datasets of differeing time scales
    one on original time frame
    one of daily averages
    '''

    # get list of averaged quanitities
    arr_list = []
    ds = xr.decode_cf(ds)
    for var in day_keys:
        midday = ds[var].sel(time=datetime.time(12))
        midnight = ds[var].sel(time=datetime.time(0))
        midnight['time'] = midnight.time - 1
        cumlat_times = xr.merge([midday, midnight])
        day_mean = cumlat_times.groupby('time.dayofyear').sum(
                                   'time', keep_attrs=True) / 86400
        arr_list.append(day_mean)

    # merge data arrays
    ds_24 = xr.merge(arr_list)
    ds_24 = ds_24.rename({'dayofyear':'time'})
    ds_24['time'] = (ds_24.time * 24. - 12.)
    ds_24.time.attrs = {'long_name': 'time', 'units': 'hours since 2015-01-01',
                        'calendar': 'gregorian'}
    #ds_24['nav_lon'] = ds.nav_lon
    #ds_24['nav_lat'] = ds.nav_lat
    ds_24 = xr.decode_cf(ds_24)

    # drop averaged keys from original dataset
    ds_short = ds.drop(var_keys)
        
    return ds_short, ds_24

def five_day_average(ds, var_keys):
    ''' 
    return two datasets of differeing time scales
    one on original time frame
    one of five day averages
    '''
    ds_short, ds_24 = daily_average(ds, var_keys)

    time = pd.date_range("2015-01-03 12:00:00", freq="5D", periods=6)
    ds_5d = ds_24.interp(time=time)
    ds_5d.time.encoding['dtype'] = np.float64
    return ds_short, ds_5d

def five_day_average_all(ds, var_keys):
    ''' 
    five day averages of all quantities
    '''

    ds_short, ds_24 = daily_average(ds, var_keys)

    def interpolate(ds):
        time = pd.date_range("2015-01-03 12:00:00", freq="5D", periods=6)
        ds = ds.interp(time=time)
        ds.time.encoding['dtype'] = np.float64
        return ds

    ds_short = interpolate(ds_short)
    ds_24 = interpolate(ds_24)
    ds = xr.merge([ds_short, ds_24])

    return ds

def daily_average_all(ds, var_keys):
    ''' 
    daily averages of all quantities
    '''

    ds_short, ds_24 = daily_average(ds, var_keys)

    ds_short = ds_short.interp(time=ds_24.time)
    ds = xr.merge([ds_short, ds_24])
    return ds

for var in ECMWF.keys():
    ECMWF = regrid(ECMWF, coord, var)
ECMWF = clean_coords(ECMWF)
ECMWF = correct_units(ECMWF)
ECMWF = calc_specific_humidity(ECMWF)

day_keys = ['sf', 'tp', 'ssrd', 'strd']
#ECMWF_3, ECMWF_24 = daily_average(ECMWF, day_keys)
#ECMWF_3, ECMWF_5d = five_day_average(ECMWF, day_keys)
#ECMWF_5d_all = five_day_average_all(ECMWF, day_keys)
ECMWF_24_all = daily_average_all(ECMWF, day_keys)

#ECMWF_3.to_netcdf('ECMWF_3_conform.nc', unlimited_dims='time')
#ECMWF_5d.to_netcdf('ECMWF_5d_conform.nc', unlimited_dims='time')
#ECMWF_5d_all.to_netcdf('ECMWF_5d_all_conform.nc', unlimited_dims='time')
ECMWF_24_all.to_netcdf('ECMWF_24_all_conform.nc', unlimited_dims='time')
