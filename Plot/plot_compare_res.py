import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from common import time_mean_to_orca
import cmocean
from processORCA12.gridding import regrid

plt.style.use('paper')

def compare2(path1, path2, time=3, depth_dict=None):
    '''
    compare 2 models over a given 5 day averaged period
     - {T,S,U,V}
    '''

    # intialise plots
    fig, axs = plt.subplots(2,4, figsize=(6.5, 4.5), dpi=600)
    plt.subplots_adjust(bottom=0.25, top=0.98, right=0.98, left=0.1,
                        wspace=0.05, hspace=0.15)

    # load data
    orca_path = '../processORCA12/DataOut/ORCA_PATCH_T.nc'
    orca   = xr.open_dataset(orca_path, decode_cf=False)
    sochic1 = {'U': xr.open_dataset(path1 + 'U.nc', decode_cf=True),
               'V': xr.open_dataset(path1 + 'V.nc', decode_cf=True),
               'T': xr.open_dataset(path1 + 'T.nc', decode_cf=True)}
    sochic2 = {'U': xr.open_dataset(path2 + 'U.nc', decode_cf=True),
               'V': xr.open_dataset(path2 + 'V.nc', decode_cf=True),
               'T': xr.open_dataset(path2 + 'T.nc', decode_cf=True)}

    # translate orca dates to cf
    orca.time_counter.attrs['units'] = 'seconds since 1900-01-01'
    orca.time_centered.attrs['units'] = 'seconds since 1900-01-01'
    orca = xr.decode_cf(orca)

    for pos in ['U', 'V', 'T']:
        tag = 'depth' + pos.lower()
        sochic1[pos] = sochic1[pos].isel({tag:0})
        sochic2[pos] = sochic2[pos].isel({tag:0})

        ## align time steps
        #try:
        #    sochic1[pos]['time_counter'] = sochic1[pos].indexes['time_counter'].to_datetimeindex()
        #    sochic2[pos]['time_counter'] = sochic2[pos].indexes['time_counter'].to_datetimeindex()
        #except:
        #    print ('leap skipping to_datetimeindex')

        sochic1[pos] = time_mean_to_orca(orca, sochic1[pos])
        sochic2[pos] = time_mean_to_orca(orca, sochic2[pos])
  
        sochic1[pos] = sochic1[pos].isel(time_counter=3)
        sochic2[pos] = sochic2[pos].isel(time_counter=3)



    def get_lims(var, pos, sym_bounds=False):
        vmin = min(sochic1[pos][var].min(), sochic2[pos][var].min())
        vmax = max(sochic1[pos][var].max(), sochic2[pos][var].max())
        if sym_bounds:
            vmax = max(np.abs(vmin), np.abs(vmax))
            vmin = -vmax
        return vmin, vmax

    # plot temperature
    vmin, vmax = get_lims('votemper', 'T', sym_bounds=False)
    cmap = cmocean.cm.thermal
    axs[0,0].pcolor(sochic1['T'].nav_lon, sochic1['T'].nav_lat,
                        sochic1['T'].votemper,
                        vmin=vmin, vmax=vmax, cmap=cmap)
    p0 = axs[1,0].pcolor(sochic2['T'].nav_lon, sochic2['T'].nav_lat,
                        sochic2['T'].votemper,
                        vmin=vmin, vmax=vmax, cmap=cmap)

    # plot salinity
    vmin, vmax = get_lims('vosaline', 'T', sym_bounds=False)
    vmin = 33.5
    vmax = 34.0
    cmap = cmocean.cm.haline
    axs[0,1].pcolor(sochic1['T'].nav_lon, sochic1['T'].nav_lat,
                        sochic1['T'].vosaline,
                        vmin=vmin, vmax=vmax, cmap=cmap)
    p1 = axs[1,1].pcolor(sochic2['T'].nav_lon, sochic2['T'].nav_lat,
                        sochic2['T'].vosaline,
                        vmin=vmin, vmax=vmax, cmap=cmap)

    # plot u
    vmin, vmax = get_lims('vozocrtx', 'U', sym_bounds=True)
    cmap = cmocean.cm.balance
    axs[0,2].pcolor(sochic1['U'].nav_lon, sochic1['U'].nav_lat,
                        sochic1['U'].vozocrtx,
                        vmin=vmin, vmax=vmax, cmap=cmap)
    p2 = axs[1,2].pcolor(sochic2['U'].nav_lon, sochic2['U'].nav_lat,
                        sochic2['U'].vozocrtx,
                        vmin=vmin, vmax=vmax, cmap=cmap)

    # plot v
    vmin, vmax = get_lims('vomecrty', 'V', sym_bounds=True)
    cmap = cmocean.cm.balance
    axs[0,3].pcolor(sochic1['V'].nav_lon, sochic1['V'].nav_lat,
                        sochic1['V'].vomecrty,
                        vmin=vmin, vmax=vmax, cmap=cmap)
    p3 = axs[1,3].pcolor(sochic2['V'].nav_lon, sochic2['V'].nav_lat,
                        sochic2['V'].vomecrty,
                        vmin=vmin, vmax=vmax, cmap=cmap)

    ## colourbars
    pos = axs[0,0].get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p0, cax=cbar_ax, orientation='horizontal')
    cbar.ax.text(0.5, -6.0, 'potential temperature\n' + r'($^{\circ}$C)',
                 fontsize=8, rotation=0,
                 transform=cbar.ax.transAxes, va='bottom', ha='center')

    pos = axs[0,1].get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
    cbar.ax.text(0.5, -6.0, 'practical salinity\n(psu)',
                 fontsize=8, rotation=0,
                 transform=cbar.ax.transAxes, va='bottom', ha='center')

    pos = axs[0,2].get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p2, cax=cbar_ax, orientation='horizontal')
    cbar.ax.text(0.5, -5.0, r'$u$ (m s$^{-1}$)',
                 fontsize=8, rotation=0,
                 transform=cbar.ax.transAxes, va='bottom', ha='center')

    pos = axs[0,3].get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p3, cax=cbar_ax, orientation='horizontal')
    cbar.ax.text(0.5, -5.0, r'$v$ (m s$^{-1}$)',
                 fontsize=8, rotation=0,
                 transform=cbar.ax.transAxes, va='bottom', ha='center')

    #cbar.locator = ticker.MaxNLocator(nbins=3)
    #cbar.update_ticks()
    #cbar.ax.text(4.5, 0.5, labels[i], fontsize=8, rotation=90,

    for ax in axs[:1,:].flatten():
        ax.set_xticklabels([])
    for ax in axs[:,1:].flatten():
        ax.set_yticks([])
    for ax in axs.flatten():
        ax.set_aspect('equal')
    for ax in axs[:,0]:
        ax.set_ylabel('latitude')
    for ax in axs[1,:]:
        ax.set_xlabel('longitude')

    plt.savefig('compare_orca12_orca24_state.png', transparent=True)

if __name__ == '__main__':
    model1 = 'EXP74'
    model2 = 'EXP106'

    orca_path = '../processORCA12/DataOut/ORCA_PATCH_T.nc'
    path1 = '../Output/' + model1 + '/SOCHIC_PATCH_1h_20150106_20150130_grid_'
    path2 = '../Output/' + model2 + '/SOCHIC_PATCH_1h_20150106_20150118_grid_'
    compare2(path1, path2, time=0)
