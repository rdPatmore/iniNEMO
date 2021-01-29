import xarray as xr
import numpy as np
from scipy.interpolate import griddata

def regrid(ds, coord, var):
    print ('variable: ', var)
    time_labels  = set(['time', 'time_counter'])
    depth_labels = set(['deptht', 'depthu', 'depthv'])
    dims = set(ds[var].dims)
    print (dims)
    if list(dims & depth_labels):
        depth_name = list(dims & depth_labels)[0]
    time_name = list(dims & time_labels)[0]
    ds_new_grid = []
    nav_lon = ds.nav_lon.isel({time_name:0})
    nav_lat = ds.nav_lat.isel({time_name:0})
    for i, ds_iter in enumerate(ds[var]):
        ds_new_lev = []
        if list(dims & depth_labels):
            for j, dep in enumerate(ds_iter):
                dep = dep.load()
                print ('time: ', i, ', lev: ', j)
                xi = nav_lon.values
                xj = nav_lat.values
                points = np.squeeze(np.dstack((xi.ravel(), xj.ravel())))
                values = dep.values.ravel()
                grid_x = coord.nav_lon.values
                grid_y = coord.nav_lat.values
                
                grid_lev = griddata(points, values, (grid_x, grid_y))
                ds_new_lev.append(grid_lev[None,:,:])
            ds_regridded_lev = np.concatenate(ds_new_lev, axis=0)
            ds_new_grid.append(ds_regridded_lev[None,:,:,:])
        else:
            print ('time: ', i)
            xi = nav_lon.values
            xj = nav_lat.values
            points = np.squeeze(np.dstack((xi.ravel(), xj.ravel())))
            values = ds_iter.values.ravel()
            grid_x = coord.nav_lon.values
            grid_y = coord.nav_lat.values
            
            grid_lev = griddata(points, values, (grid_x, grid_y))
            ds_new_grid.append(grid_lev[None,:,:])
    ds_regridded_var = np.concatenate(ds_new_grid, axis=0)
    
    if list(dims & depth_labels):
        dims = [time_name, depth_name, 'Y', 'X']
    else:
        dims = [time_name, 'Y', 'X']
    ds = ds.assign({var: (dims, ds_regridded_var, ds[var].attrs)})
    return ds
