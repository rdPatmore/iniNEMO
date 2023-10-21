import xarray as xr
import config
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cmocean

matplotlib.rcParams.update({'font.size': 8})

def Ro_sea_ice():

    fig, axs = plt.subplots(1,2, figsize=(5.0,3))
    plt.subplots_adjust(bottom=0.3, top=0.99, right=0.99, left=0.1,
                        wspace=0.05)

    # load sea ice and Ro
    si = xr.open_dataset(config.data_path() + 
                    'EXP10/SOCHIC_PATCH_3h_20121209_20130331_icemod.nc').siconc
    Ro = xr.open_dataset(config.data_path_old() + 
                    'EXP10/rossby_number.nc').Ro
    # plot sea ice
    halo=(1%3) + 1
    si = si.sel(time_counter='2012-12-30 00:00:00', method='nearest')
    si = si.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
    
    p0 = axs[0].pcolor(si.nav_lon, si.nav_lat, si, shading='nearest',
                       cmap=cmocean.cm.ice, vmin=0, vmax=1)
    axs[0].set_aspect('equal')
    axs[0].set_xlabel('Longitude')
    axs[0].set_xlim([-3.7,3.7])
    axs[0].set_ylim([-63.8,-56.2])

    # plot Ro
    Ro = Ro.sel(time_counter='2012-12-30 00:00:00', method='nearest')
    Ro = Ro.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
    Ro = Ro.isel(depth=10)
    
    p1 = axs[1].pcolor(Ro.nav_lon, Ro.nav_lat, Ro, shading='nearest',
                  cmap=plt.cm.RdBu, vmin=-0.45, vmax=0.45)
    axs[1].set_aspect('equal')
    axs[1].set_xlabel('Longitude')
    axs[1].set_xlim([-3.7,3.7])
    axs[1].set_ylim([-63.8,-56.2])

    axs[1].yaxis.set_ticklabels([])
    
    axs[0].set_ylabel('Latitude')
    
    pos = axs[0].get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p0, cax=cbar_ax, orientation='horizontal')
    cbar.ax.text(0.5, -4.3, r'Sea Ice Concentration', fontsize=8,
               rotation=0, transform=cbar.ax.transAxes, va='top', ha='center')

    pos = axs[1].get_position()
    cbar_ax = fig.add_axes([pos.x0, 0.13, pos.x1 - pos.x0, 0.02])
    cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
    cbar.ax.text(0.5, -4.3, r'$\zeta / f$', fontsize=8,
               rotation=0, transform=cbar.ax.transAxes, va='top', ha='center')

    axs[0].text(0.1, 0.9, '2012-12-30', transform=axs[0].transAxes, c='w')
    
    plt.savefig('SI_Ro.png', dpi=600)

Ro_sea_ice()
