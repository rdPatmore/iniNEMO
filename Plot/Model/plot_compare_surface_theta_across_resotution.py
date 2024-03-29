import xarray as xr
import config
import matplotlib.pyplot as plt

def mean_slices():
    t12 = xr.open_dataset(config.data_path() + 
                       'EXP04/SOCHIC_PATCH_24h_20120101_20121231_grid_T.nc').tos
    t24 = xr.open_dataset(config.data_path() + 
                        'EXP08/SOCHIC_PATCH_6h_20120101_20120701_grid_T.nc').tos
    t48 = xr.open_dataset(config.data_path() + 
                       'EXP10/SOCHIC_PATCH_24h_20120101_20121231_grid_T.nc').tos
    
    # get daily mean for t24
    t24 = t24.groupby('time_counter.dayofyear').mean('time_counter')
    time_counter = xr.DataArray(
                     t48.time_counter.isel(time_counter=slice(None,183)).values,
                         dims=('dayofyear'))
    t24 = t24.assign_coords({'time_counter':time_counter}) 
    t24 = t24.swap_dims({'dayofyear':'time_counter'})
    
    tlist = [t12, t24, t48]
    
    fig, axs = plt.subplots(1,3, figsize=(8.5,3.5))
    plt.subplots_adjust(left=0.1, right=0.88, top=0.9, bottom=0.1, wspace=0.05)
    
    for i, t in enumerate(tlist):
        t = t.sel(time_counter='2012-05-01 00:00:00', method='nearest')
        t = t.isel(x=slice(1*(i+1), -1*(i+1)), y=slice(1*(i+1), -1*(i+1)))
        
        p = axs[i].pcolor(t.nav_lon, t.nav_lat, t, shading='nearest',
                      cmap=plt.cm.inferno, vmin=-1, vmax=1)
        axs[i].set_aspect('equal')
        axs[i].set_xlabel('Longitude')
        axs[i].set_xlim([-3.8,3.8])
        axs[i].set_ylim([-63.8,-56.2])
    
    for i in [1,2]:
        axs[i].yaxis.set_ticklabels([])
    
    axs[0].set_ylabel('Latitude')
    
    axs[0].set_title(r'1/12$^{\circ}$')
    axs[1].set_title(r'1/24$^{\circ}$')
    axs[2].set_title(r'1/48$^{\circ}$')
    
    pos = axs[2].get_position()
    cbar_ax = fig.add_axes([0.89, pos.y0, 0.02, pos.y1 - pos.y0])
    cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
    cbar.ax.text(4.3, 0.5, r'Temperature ($^{\circ}$C)', fontsize=10,
               rotation=90, transform=cbar.ax.transAxes, va='center', ha='left')
    
    plt.savefig('T_12_24_48.png', dpi=600)

def mean_versus_snapshots():
    # mean data
    t12_mean = xr.open_dataset(config.data_path() + 
                       'EXP13/SOCHIC_PATCH_24h_20120101_20121231_grid_T.nc').tos
    t24_mean = xr.open_dataset(config.data_path() + 
                       'EXP08/SOCHIC_PATCH_24h_20120101_20121231_grid_T.nc').tos
    t48_mean = xr.open_dataset(config.data_path() + 
                       'EXP10/SOCHIC_PATCH_24h_20120101_20121231_grid_T.nc').tos

    # sanpshot data
    t12_snap = xr.open_dataset(config.data_path() + 
                       'EXP13/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').tos
    t24_snap = xr.open_dataset(config.data_path() + 
                       'EXP08/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc').tos
    t48_snap = xr.open_dataset(config.data_path() + 
                       'EXP10/SOCHIC_PATCH_24h_20120101_20121231_grid_T.nc').tos
    
    tlist = [t12_mean, t24_mean, t48_mean, t12_snap, t24_snap, t48_snap]
    
    fig, axs = plt.subplots(2,3, figsize=(6.5,5.5))
    plt.subplots_adjust(left=0.1, right=0.88, top=0.9, bottom=0.1, wspace=0.05)
    
    for i, t in enumerate(tlist):
        flat_axs = axs.ravel()
        halo=(1%3) + 1
        t = t.sel(time_counter='2012-12-20 00:00:00', method='nearest')
        t = t.isel(x=slice(1*halo, -1*halo), y=slice(1*halo, -1*halo))
        
        p = flat_axs[i].pcolor(t.nav_lon, t.nav_lat, t, shading='nearest',
                      cmap=plt.cm.inferno, vmin=-1, vmax=1)
        flat_axs[i].set_aspect('equal')
        flat_axs[i].set_xlabel('Longitude')
        flat_axs[i].set_xlim([-3.8,3.8])
        flat_axs[i].set_ylim([-63.8,-56.2])
    
    for i in [1,2]:
        axs[0,i].yaxis.set_ticklabels([])
        axs[1,i].yaxis.set_ticklabels([])
    
    axs[0,0].set_ylabel('Latitude')
    axs[1,0].set_ylabel('Latitude')
    
    axs[0,0].set_title(r'1/12$^{\circ}$')
    axs[0,1].set_title(r'1/24$^{\circ}$')
    axs[0,2].set_title(r'1/48$^{\circ}$')
    
    pos0 = axs[0,2].get_position()
    pos1 = axs[1,2].get_position()
    cbar_ax = fig.add_axes([0.89, pos0.y0, 0.02, pos0.y1 - pos1.y0])
    cbar = fig.colorbar(p, cax=cbar_ax, orientation='vertical')
    cbar.ax.text(4.3, 0.5, r'Temperature ($^{\circ}$C)', fontsize=10,
               rotation=90, transform=cbar.ax.transAxes, va='center', ha='left')
    
    plt.savefig('T_12_24_48_snap_mean.png', dpi=600)

mean_versus_snapshots()
