import xarray as xr
import config
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 8})

def theta_Ro():

    # load theta 
    t12_snap = xr.open_dataset(config.data_path() + 
                       'EXP13/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').tos
    t24_snap = xr.open_dataset(config.data_path() + 
                       'EXP08/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').tos
    t48_snap = xr.open_dataset(config.data_path() + 
                       'EXP10/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').tos
    # load Ro 
    t12_Ro = xr.open_dataarray(config.data_path() + 'EXP13/rossby_number.nc')
    t24_Ro = xr.open_dataarray(config.data_path() + 'EXP08/rossby_number.nc')
    t48_Ro = xr.open_dataarray(config.data_path() + 'EXP10/rossby_number.nc')
    
    tlist = [t12_snap, t24_snap, t48_snap, t12_Ro, t24_Ro, t48_Ro]
    
    fig, axs = plt.subplots(2,3, figsize=(6.5,4.0))
    plt.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.1, wspace=0.05,
                        hspace=0.05)

    flat_axs = axs.ravel()

    # plot temperature    
    for i, t in enumerate(tlist[:3]):
        halo=(1%3) + 1
        t = t.sel(time_counter='2013-02-20 00:00:00', method='nearest')
        t = t.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        
        p0 = flat_axs[i].pcolor(t.nav_lon, t.nav_lat, t, shading='nearest',
                      cmap=plt.cm.inferno, vmin=-0.5, vmax=1.2)
        flat_axs[i].set_aspect('equal')
        flat_axs[i].set_xticklabels([])
        flat_axs[i].set_xlim([-3.8,3.8])
        flat_axs[i].set_ylim([-63.8,-56.2])

    # plot Ro
    for i, t in enumerate(tlist[3:]):
        i = i + 3
        t = t.sel(time_counter='2013-02-20 00:00:00', method='nearest')
        t = t.isel(depth=10)
        
        p1 = flat_axs[i].pcolor(t.nav_lon, t.nav_lat, t, shading='nearest',
                      cmap=plt.cm.RdBu, vmin=-0.5, vmax=0.5)
        flat_axs[i].set_aspect('equal')
        flat_axs[i].set_xlabel('Longitude')
        flat_axs[i].set_xlim([-3.8,3.8])
        flat_axs[i].set_ylim([-63.8,-56.2])

    for i in [1,2]:
        axs[0,i].yaxis.set_ticklabels([])
        axs[1,i].yaxis.set_ticklabels([])
    
    axs[0,0].set_ylabel('Latitude')
    axs[1,0].set_ylabel('Latitude')
    
    axs[0,0].set_title(r'1/12$^{\circ}$', fontsize=8)
    axs[0,1].set_title(r'1/24$^{\circ}$', fontsize=8)
    axs[0,2].set_title(r'1/48$^{\circ}$', fontsize=8)
    
    pos = axs[0,2].get_position()
    cbar_ax = fig.add_axes([0.89, pos.y0, 0.02, pos.y1 - pos.y0])
    cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
    cbar.ax.text(4.3, 0.5, r'Temperature ($^{\circ}$C)', fontsize=8,
               rotation=90, transform=cbar.ax.transAxes, va='center', ha='left')

    pos = axs[1,2].get_position()
    cbar_ax = fig.add_axes([0.89, pos.y0, 0.02, pos.y1 - pos.y0])
    cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
    cbar.ax.text(4.3, 0.5, r'$\zeta / f$', fontsize=8,
               rotation=90, transform=cbar.ax.transAxes, va='center', ha='left')
    
    plt.savefig('T_Ro_12_24_48_saturated', dpi=600)

theta_Ro()
