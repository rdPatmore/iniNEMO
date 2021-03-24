import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import datetime as datetime
from common import time_mean_to_orca

def add_slice(ax, ds, time, depth):

    ds = ds.isel(time_counter=time, deptht=depth)
    
    lev = np.linspace(-2,2,11)
    p = ax.pcolor(ds.nav_lon, ds.nav_lat, ds.vozocrtx, vmin=-0.4, vmax=0.4,
                  cmap=plt.cm.inferno)

    return p

def plot(orca_path, sochic_path, model, depth0, depth1):
    '''
    main plotting routine adding six time slices over two depths
    '''

    # intialise plots
    fig, axs = plt.subplots(2,6, figsize=(6.5, 2), dpi=300)
    plt.subplots_adjust(bottom=0.15, top=0.98, right=0.90, left=0.10,
                        wspace=0.1, hspace=0.1)

    # load data
    orca   = xr.open_dataset(orca_path, decode_cf=False)
    sochic = xr.open_dataset(sochic_path, decode_cf=True)

    # translate orca dates to cf
    orca.time_counter.attrs['units'] = 'seconds since 1900-01-01'
    orca.time_centered.attrs['units'] = 'seconds since 1900-01-01'
    orca = xr.decode_cf(orca)

    #sochic['time_counter'] = sochic.indexes['time_counter'].to_datetimeindex()

    #if model == 'orca': 
    m = orca
    #if model == 'sochic':
    #    # align time steps
    #m = time_mean_to_orca(orca, sochic)
  
    # plot six time steps at depth0
    for i, ax in enumerate(axs[0]):
        print ('a', i)
        add_slice(ax, m, i, depth0)

    # plot six time steps at depth1
    for i, ax in enumerate(axs[1]):
        print ('b', i)
        p = add_slice(ax, m, i, depth1)

    pos = axs[0,1].get_position()
    cbar_ax = fig.add_axes([0.91, pos.y0, 0.02, pos.y1 - pos.y0])
    cbar = fig.colorbar(p, cax=cbar_ax)
    #cbar.locator = ticker.MaxNLocator(nbins=3)
    #cbar.update_ticks()
    #cbar.ax.text(4.5, 0.5, labels[i], fontsize=8, rotation=90,
    cbar.ax.text(4.5, 0.5, 'temperature', fontsize=8, rotation=90,
                 transform=cbar.ax.transAxes, va='center', ha='right')
    for ax in axs[0,:]:
        ax.set_xticks([])
    for ax in axs[0,1:]:
        ax.set_yticks([])
    for ax in axs[1,1:]:
        ax.set_yticks([])
    for ax in axs.flatten():
        ax.set_aspect('equal')

    plt.savefig(model + 'orca_u_orca_mean.png')

if __name__ == '__main__':
    model = 'EXP106'
    orca_path = '../processORCA12/DataOut/ORCA_PATCH_U.nc'
    #orca_path = '../processORCA12/DataOut/ORCA0083-N06_T_conform.nc'
    sochic_path = '../Output/EXP106/SOCHIC_PATCH_1h_20150106_20150118_grid_U.nc'
   #sochic_path = '../Output/EXP74/SOCHIC_PATCH_1h_20150106_20150130_grid_T.nc'
    #sochic_path = '../Output/EXP60/SOCHIC_PATCH_1h_20150101_20150121_grid_T.nc'

    #SOCHIC_PATCH_1h_20150101_20150130_grid_T.nc'
    #plot(orca_path, sochic_path, 'orca', depth0=0, depth1=20)
    plot(orca_path, sochic_path, model, depth0=0, depth1=20)
