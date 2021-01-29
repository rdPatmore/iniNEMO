import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np


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

def calc_specific_humidity(ds):
    T0 = 273.14 # Kelvin freezing point
    Rd = 287 # gas constant for dry air
    Rv = 461 # gas constant for water vapour

    # Teten's constants
    c0 = 6.1078
    c1_neg = 21.875
    c2_neg = 265.5 
    c1_pos = 17.27
    c2_pos = 237.3

    def humid_formula(c0, c1, c2, T, T0):
        return c0 * np.exp( c1 * (T - T0) / (T - T0 + c2))
 
    es = xr.where(ds.d2m >= 0, humid_formula(c0, c1_pos, c2_pos, ds['d2m'], T0),
                               humid_formula(c0, c1_neg, c2_neg, ds['d2m'], T0))
    R_ratio = Rd/Rv
    humid = R_ratio * es / (ds.sp - (1-R_ratio) * es)
    #humid = humid * 100 # convert to percentage
    ds = ds.assign({'humid':humid})
    return ds

def correct_units(ds):
    time_period = 10800 # seconds
    density = 1000 # kg * m^-3

    snow   = ds.sf * density / time_period
    precip = ds.tp * density / time_period
    precip.attrs['units'] = 'kg m**-2 s**-1'
    snow.attrs['units']   = 'kg m**-2 s**-1'

    short_wave = ds.ssrd / time_period
    long_wave  = ds.strd / time_period
    short_wave.attrs['units']   = 'W m**-2'
    long_wave.attrs['units']   = 'W m**-2'

    p = ds.sp / 100
    mslp = ds.msl / 100
    p.attrs['units'] = 'hPa'
    mslp.attrs['units'] = 'hPa'

    ds = ds.assign({'tp': precip, 'sf': snow,
                    'ssrd': short_wave, 'strd': long_wave,
                    'sp': p, 'mslp': mslp})
    return ds

for var in ECMWF.keys():
    ECMWF = regrid(ECMWF, coord, var)
ECMWF = clean_coords(ECMWF)
ECMWF = correct_units(ECMWF)
ECMWF = calc_specific_humidity(ECMWF)

ECMWF.to_netcdf('ECMWF_conform.nc')
