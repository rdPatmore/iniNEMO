import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from common import time_mean_to_orca
import cmocean

def plot(orca_path, sochic_path, var, vmin, vmax):
    '''
    main plotting routine adding six time slices over two depths
    '''

    # intialise plots
    fig, axs = plt.subplots(3,6, figsize=(6.5, 3.5))
    plt.subplots_adjust(bottom=0.1, top=0.98, right=0.86, left=0.1,
                        wspace=0.05, hspace=0.05)

    # load data
    orca   = xr.open_dataset(orca_path, decode_cf=False)
    sochic = xr.open_dataset(sochic_path, decode_cf=True)

    # translate orca dates to cf
    orca.time_counter.attrs['units'] = 'seconds since 1900-01-01'
    orca.time_centered.attrs['units'] = 'seconds since 1900-01-01'
    orca = xr.decode_cf(orca)

    # align time steps
    sochic['time_counter'] = sochic.indexes['time_counter'].to_datetimeindex()
    sochic = time_mean_to_orca(orca, sochic)
  
    # align dim labels
    orca = orca.rename_dims({'X':'x', 'Y':'y'})
  
    # get model differences
    diff = sochic[var] - orca[var]

    # plot six time steps of orca
    for i, ax in enumerate(axs[0]):
        ds = orca.isel(time_counter=i)
        p = ax.pcolor(ds.nav_lon, ds.nav_lat, ds[var], vmin=vmin, vmax=vmax,
                      cmap=cmocean.cm.balance)

    # plot six time steps of sochic
    for i, ax in enumerate(axs[1]):
        ds = sochic.isel(time_counter=i)
        p = ax.pcolor(ds.nav_lon, ds.nav_lat, ds[var], vmin=vmin, vmax=vmax,
                      cmap=cmocean.cm.balance)

    # plot six time steps of sochic
    for i, ax in enumerate(axs[2]):
        arr = diff.isel(time_counter=i)
        print (arr)
        vmax = max(np.abs(arr.max()), np.abs(arr.min()))
        vmin = - vmax
        p = ax.pcolor(sochic.nav_lon, sochic.nav_lat, arr, vmin=vmin, vmax=vmax,
                      cmap=cmocean.cm.balance)

    pos = axs[0,1].get_position()
    cbar_ax = fig.add_axes([0.88, pos.y0, 0.02, pos.y1 - pos.y0])
    cbar = fig.colorbar(p, cax=cbar_ax)
    #cbar.locator = ticker.MaxNLocator(nbins=3)
    #cbar.update_ticks()
    #cbar.ax.text(4.5, 0.5, labels[i], fontsize=8, rotation=90,
    cbar.ax.text(4.5, 0.5, var, fontsize=8, rotation=90,
                 transform=cbar.ax.transAxes, va='center', ha='right')
    for ax in axs[1,:].flatten():
        ax.set_xticks([])
    for ax in axs[:,1:].flatten():
        ax.set_yticks([])

    plt.savefig('flux_diff_' + var + '.png')

if __name__ == '__main__':
    orca_path = '../processORCA12/DataOut/ORCA_PATCH_T.nc'
    sochic_path = '../Output/EXP50/SOCHIC_PATCH_1h_20150101_20150128_grid_T.nc'
    #sochic_path = '../Output/EXP60/SOCHIC_PATCH_1h_20150101_20150121_grid_T.nc'

    plot(orca_path, sochic_path, 'sowaflup', -1e-4, 1e-4)
    plot(orca_path, sochic_path, 'soshfldo', -300, 300)
    plot(orca_path, sochic_path, 'sohefldo', -210, 210)
