import xarray as xr
import config
from dask.distributed import Client, LocalCluster
import dask
import numpy as np

def calc_mean_and_std(case):
    print ('begin')

    # load data
    path = config.data_path() + case + '/'
    Ro = xr.open_dataarray(path + 'rossby_number.nc',
                           chunks=dict(time_counter=1))
    mld = xr.open_dataset(path + 'SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc',
                          chunks=dict(time_counter=1)).mldr10_3
    mld = mld.isel(x=slice(1,-2), y=slice(1,-2))
    Ro = Ro.rename({'depth':'deptht'})

    # hack for time missalignment
    Ro = Ro.assign_coords({'time_counter':mld.time_counter}) 

    def wherey(x):
        print(x.time_counter.values)
        return x.where(x.deptht < mld.sel(time_counter=x.time_counter), drop=True)
        
    #for label, group in Ro.groupby('time_counter'):
    Ro = np.abs(Ro.groupby('time_counter').map(wherey))

    Ro_std = Ro.std(['x', 'y', 'deptht']).load()
    Ro_std.name = 'std'
    Ro_mean = Ro.mean(['x', 'y', 'deptht']).load()
    Ro_mean.name = 'mean'
    Ro_stats = xr.merge([Ro_mean, Ro_std])
    Ro_stats.to_netcdf(path + 'Stats/Ro_stats.nc')
   
if __name__ == '__main__':

    #dask.config.set({'temporary_directory': 'Scratch'})
    dask.config.set(scheduler='single-threaded')
    cluster = LocalCluster(n_workers=1)
    client = Client(cluster)

    calc_mean_and_std('EXP10')
