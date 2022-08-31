import xarray as xr
import config
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cmocean

matplotlib.rcParams.update({'font.size': 8})

def Ro_theta_sea_ice():

    fig, axs = plt.subplots(1,3, figsize=(6.5,3.0))
    plt.subplots_adjust(bottom=0.28, top=0.99, right=0.98, left=0.1,
                        wspace=0.1)

    # load sea ice and Ro
    si = xr.open_dataset(config.data_path() + 
                   'EXP10/SOCHIC_PATCH_3h_20121209_20130331_icemod.nc').siconc
    Ro = xr.open_dataset(config.data_path_old() + 
                   'EXP10/rossby_number.nc').Ro
    T  = xr.open_dataset(config.data_path() + 
                   'EXP10/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').votemper

    # ~~~~~ plot Theta ~~~~~ #

    # prep data
    T = T.sel(time_counter='2012-12-30 00:00:00', method='nearest')
    T = T.isel(deptht=10)

    # plot
    p0 = axs[0].pcolor(T.nav_lon, T.nav_lat, T, shading='nearest',
                  cmap=cmocean.cm.thermal, vmin=-1.5, vmax=0.5)

    # ~~~~~ plot sea ice ~~~~~ #

    # prep data
    si = si.sel(time_counter='2012-12-30 00:00:00', method='nearest')
    
    # plot
    p1 = axs[1].pcolor(si.nav_lon, si.nav_lat, si, shading='nearest',
                       cmap=cmocean.cm.ice, vmin=0, vmax=1)

    # ~~~~~ plot Ro ~~~~~ #

    Ro = Ro.sel(time_counter='2012-12-30 00:00:00', method='nearest')
    Ro = Ro.isel(depth=10)
    
    p2 = axs[2].pcolor(Ro.nav_lon, Ro.nav_lat, Ro, shading='nearest',
                  cmap=cmocean.cm.balance, vmin=-0.45, vmax=0.45)

    # format axes
    axs[0].set_ylabel('Latitude')
    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlabel('Longitude')
        ax.set_xlim([-3.7,3.7])
        ax.set_ylim([-63.8,-56.2])

    # drop y labels
    for ax in axs[1:]:
        ax.yaxis.set_ticklabels([])
    
    
    def set_cbar(ax, p, txt):
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0+0.02, 0.16, pos.x1 - pos.x0-0.04, 0.02])
        cbar = fig.colorbar(p, cax=cbar_ax, orientation='horizontal')
        cbar.ax.text(0.5, -4.3, txt, fontsize=8,
                     rotation=0, transform=cbar.ax.transAxes,
                     va='top', ha='center')

    # colour bars
    set_cbar(axs[0], p0, r'$\Theta$ [$^\circ$C]')
    set_cbar(axs[1], p1, 'Sea Ice Area Fraction [-]')
    set_cbar(axs[2], p2, r'$\zeta / f$ [-]')

    axs[1].text(0.1, 0.9, '2012-12-30', transform=axs[1].transAxes, c='w')
    
    plt.savefig('Theta_SI_Ro.png', dpi=600)

Ro_theta_sea_ice()
