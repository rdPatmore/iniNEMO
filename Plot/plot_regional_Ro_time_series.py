import config
import xarray as xr
import matplotlib.pyplot as plt

def get_ro_time_series(path, x, y):   #, lat, lon, depth=10)

    # load and set coords
    so = xr.open_dataarray(path, chunks={'time_counter':1})
    so = so.assign_coords({'lat': so.nav_lat.isel(x=0),
                           'lon': so.nav_lon.isel(y=0)})
    so = so.swap_dims({'x':'lon', 'y':'lat'})

    # restrict depth and lat-lon
    so = so.sel(depth=10, method='nearest')
    so_patch = abs(so.sel(lon=slice(x[0],x[1]), lat=slice(y[0],y[1])))

    # means
    so_mean = so_patch.mean(['lat','lon'])
    so_mean.name = 'Ro_mean'
    
    # std
    so_std = so_patch.std(['lat','lon'])
    so_std.name = 'Ro_std'

    # merge
    so = xr.merge([so_mean, so_std]).load()

    return so

def plt_time_series(so, ls='-', c='red'):

    plt.plot(so.time_counter, so.Ro_mean, c=c, ls=ls)
    plt.fill_between(so.time_counter, so.Ro_mean-so.Ro_std,
                                       so.Ro_mean+so.Ro_std,
                                       facecolor=c, alpha=0.2)


def plot_north_south_patch():
    x = [-2,2]
    y0 = [-57.5,-56.5]
    y1 = [-57.5,-56.5]
    path = config.data_path_old() + 'EXP10/rossby_number.nc'
    #so = get_ro_time_series(path, x=[-2,2], y=)
    #    son_patch = abs(so.sel(lon=slice(-2,2),
    #                 lat=slice(-57.5,-56.5)))
    #    sos_patch = abs(so.sel(lon=slice(-2,2),
    #                 lat=slice(-63.0,-62.0)))
    plt_time_series(son, sos)
    
    path = config.data_path() + 'EXP08/rossby_number.nc'
    so = get_ro_time_series(path)
    plt_time_series(son, sos, ls='--')
    
    path = config.data_path() + 'EXP13/rossby_number.nc'
    so = get_ro_time_series(path)
    plt_time_series(so, sos, ls='--')
    plt.show()

def plot_patch_size():
    x0 = [-2,2]
    y0 = [-57.5,-56.5]

    path = config.data_path_old() + 'EXP10/rossby_number.nc'

    # small
    so = get_ro_time_series(path, x=[-1,1], y=[-57.25,-56.75])
    plt_time_series(so, c='red')
    
    # medium
    so = get_ro_time_series(path, x=[-2,2], y=[-57.5,-56.5])
    plt_time_series(so, c='blue')
    
    # large
    so = get_ro_time_series(path, x=[-4,4], y=[-58.5,-55.5])
    plt_time_series(so, c='green')
    plt.show()

plot_patch_size()
