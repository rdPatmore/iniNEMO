import xarray as xr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd



def regrid(ds, coord):
    print ('time length', ds.time_counter.values.shape)
    ds_new_grid = []
    for i, ds_iter in enumerate(ds):
        print ('time', i)
        #xi, xj = np.meshgrid(ds_iter.x.values,
        #                     ds_iter.y.values)
        print (ds_iter)
        points = np.squeeze(np.dstack((ds.nav_lon.values.ravel(), ds.nav_lon.values.ravel())))
        values = ds_iter.values.ravel()
        print (points.shape)
        print (values.shape)
        grid_x = coord.nav_lon.values
        grid_y = coord.nav_lat.values
        
        grid = griddata(points, values, (grid_x, grid_y))
        ds_new_grid.append(grid[None,:,:])
    
    ds_regridded_var = np.concatenate(ds_new_grid, axis=0)
    ds = xr.dataArray({var: (['time','Y', 'X'], ds_regridded_var)})
    return ds

def clean_coords(ds):
    ds = ds.drop_dims(['latitude','longitude'])
    print (ds)
    ds = ds.assign({'nav_lon': coord.nav_lon, 'nav_lat': coord.nav_lat})
    return ds

def regrid_dfs(coord, ORCA12):
    flux=ORCA12.tohfls
    print (flux)
    def cut_ds(ds):
        ds = ds.where(ds.nav_lat < -57, drop=True)
        ds = ds.where(ds.nav_lat > -63, drop=True)
        ds = ds.where(ds.nav_lon < 3, drop=True)
        ds = ds.where(ds.nav_lon > -3, drop=True)
        return ds
    #flux = cut_ds(flux)
    #print (flux.nav_lon.values)
    #print (flux.nav_lat.values)

    flux = flux.isel(x=slice(3430,3480), y=slice(550,650)).fillna(0.0)
    #flux = regrid(flux, coord)
#

    #ds = xr.merge(ds)
    #ds = clean_coords(ds)

    flux.to_netcdf('orca12_patch_surface_flux.nc',
                 unlimited_dims='time')

coord = xr.open_dataset('../SourceData/coordinates.nc', decode_times=False)
ORCA12 = xr.open_mfdataset('DataIn/ORCA0083-N06_2015*d05T.nc')
regrid_dfs(coord, ORCA12)
