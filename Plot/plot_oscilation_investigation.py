import matplotlib.pyplot as plt
import xarray as xr
import config
import numpy as np

path = config.data_path()

theta12 = xr.open_dataset(path + 'EXP13' +
                    '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').votemper
theta24 = xr.open_dataset(path + 'EXP08' +
                    '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').votemper
theta48 = xr.open_dataset(path + 'EXP10' +
                    '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').votemper

fig, axs = plt.subplots(3, figsize=(3.0,3))

def render(ax, data, deps=[10],x=50,y=50,time1=100):
    d0 = data.sel(deptht=deps, method='nearest')
    d0 = d0.isel(x=x,y=y,time_counter=slice(0,time1)).squeeze()
    print (d0)
    #detrend
    #n = len(d0)
    #t = np.arange(n)
    #p = np.polyfit(t, d0, 3)
    #d0 = d0 - np.polyval(p, t)
    #grouped= d0.groupby('time_counter.date')
    #d0 = d0 - d0.mean()
    #d0 = d0.assign_coords({'time':d0.time_counter.dt.time})
    #d0 = d0.assign_coords({'dayofyear':d0.time_counter.dt.dayofyear})
    
    d0 = d0.groupby('time_counter.time') - d0.groupby('time_counter.time').first()#.groupby('time_counter.dayofyear')
    d0 = d0.assign_coords({'time':d0.time_counter.dt.strftime('%H:%M:%S')})
    print (d0)

    #ax.plot(d0.time_counter.dt.strftime('%H:%M'), d0, alpha=0.2, c='b')
    ax.plot(d0.time, d0, alpha=0.2, c='b')

deps = [100]
render(axs[0], theta12, deps=deps)
render(axs[1], theta24, deps=deps)
render(axs[2], theta48, deps=deps)
plt.show()
   
