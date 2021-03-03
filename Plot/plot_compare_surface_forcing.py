import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from common import time_mean_to_orca
import cmocean

plt.style.use('paper')

def plot(model, orca_path, sochic_path, var, vmin, vmax):
    '''
    main plotting routine adding six time slices over two depths
    '''

    # intialise plots
    fig, axs = plt.subplots(3,6, figsize=(6.5, 3.0))
    plt.subplots_adjust(bottom=0.15, top=0.97, right=0.86, left=0.15,
                        wspace=0.1, hspace=0.1)

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
        p0 = ax.pcolor(ds.nav_lon, ds.nav_lat, ds[var], vmin=vmin, vmax=vmax,
                      cmap=cmocean.cm.balance)

    # plot six time steps of sochic - orca
    for i, ax in enumerate(axs[2]):
        arr = diff.isel(time_counter=i)
        #maximum = max(np.abs(arr.max()), np.abs(arr.min()))
        #minimum = - maximum
        minimum=vmin/10
        maximum=vmax/10
        p1 = ax.pcolor(sochic.nav_lon, sochic.nav_lat, arr,
                       vmin=minimum, vmax=maximum, cmap=cmocean.cm.balance)

    pos0 = axs[0,-1].get_position()
    pos1 = axs[1,-1].get_position()
    cbar_ax = fig.add_axes([0.88, pos1.y0, 0.02, pos0.y1 - pos1.y0])
    cbar = fig.colorbar(p0, cax=cbar_ax)

    pos2 = axs[2,-1].get_position()
    cbar_ax = fig.add_axes([0.88, pos2.y0, 0.02, pos2.y1 - pos2.y0])
    cbar = fig.colorbar(p1, cax=cbar_ax)

    #cbar.locator = ticker.MaxNLocator(nbins=3)
    #cbar.update_ticks()
    #cbar.ax.text(4.5, 0.5, labels[i], fontsize=8, rotation=90,

    cbar.ax.text(4.5, 0.5, var, fontsize=8, rotation=90,
                 transform=cbar.ax.transAxes, va='center', ha='right')

    for ax in axs[:2,:].flatten():
        ax.set_xticks([])
    for ax in axs[:,1:].flatten():
        ax.set_yticks([])
    for ax in axs.flatten():
        ax.set_aspect('equal')
    for ax in axs[:,0]:
        ax.set_ylabel('latitude')
    for ax in axs[2,:]:
        ax.set_xlabel('longitude')

    plt.savefig(model + '_flux_diff_' + var + '.png')

if __name__ == '__main__':
    orca_path = '../processORCA12/DataOut/ORCA_PATCH_T.nc'
    sochic_path = '../Output/EXP52/SOCHIC_PATCH_1h_20150101_20150128_grid_T.nc'
    #sochic_path = '../Output/EXP60/SOCHIC_PATCH_1h_20150101_20150121_grid_T.nc'

    model = 'EXP52'
    plot(model, orca_path, sochic_path, 'sowaflup', -1e-4, 1e-4)
    plot(model, orca_path, sochic_path, 'soshfldo', -300, 300)
    plot(model, orca_path, sochic_path, 'sohefldo', -210, 210)
