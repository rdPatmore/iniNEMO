import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cmocean
import config

def make_salt_slice_movie_2param(model, outfreq='3h'):
    ''' make movie of salinity with and without sea ice mask '''
     

    # prep movie maker
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1e6)

    # ini plot
    fig, axs = plt.subplots(1,2,figsize=(5.5,4.0))

    # get data
    if outfreq == '3h':
        fn = 'SOCHIC_PATCH_3h_20121209_20130331_'
        outdir = config.data_path()
    if outfreq == '24h':
        fn = 'SOCHIC_PATCH_24h_20120101_20121231_'
        outdir = config.data_path_old()
    salt = xr.open_dataset(outdir + model + '/' + fn + 'grid_T.nc').sos
    salt = salt.isel(x=slice(45,-45),y=slice(45,-45)).T
    p = []
    for ax in axs:
        #p.append(ax.imshow(salt.isel(time_counter=0),
        #          vmin=33.5, vmax=34.35, origin='lower',
        #          aspect='equal', animated=True, cmap=cmocean.cm.haline))
        p.append(ax.pcolor(salt.nav_lon, salt.nav_lat,
                  salt.isel(time_counter=0),
                  vmin=33.5, vmax=34.35,
                  cmap=cmocean.cm.haline, shading='nearest'))


    ice = xr.open_dataset(outdir + model + '/' + fn + 'icemod.nc').siconc
    ice = ice.where(ice > 0)
    ice = ice.isel(x=slice(45,-45),y=slice(45,-45)).T
    #pice = axs[0].imshow(ice.isel(time_counter=0),
    #              vmin=0, vmax=1, origin='lower',
    #              aspect='equal', animated=True, cmap=cmocean.cm.ice)
    pice = axs[0].pcolor(ice.nav_lon, ice.nav_lat,
                 ice.isel(time_counter=300),
                  vmin=0, vmax=1,
                  cmap=cmocean.cm.ice, shading='nearest', zorder=10)

#    def animate(i):
#        print (i)
#        for j in range(2):
#            p[j].set_array(salt.isel(time_counter=i))
#        pice.set_array(ice.isel(time_counter=i))
#        plt.suptitle(salt.time_counter[i].dt.strftime(
#                      '%y-%m-%d %H:%M').values)

    def animate(i):
        print (i)
        for j in range(2):
            p[j].set_array(salt.isel(time_counter=i).stack(z=('x','y')))
        ice_i = ice.isel(time_counter=i).stack(z=('x','y'))
        pice.set_array(ice_i)
        #pice = axs[0].pcolor(ice.nav_lon, ice.nav_lat,
        #         ice.isel(time_counter=i),
        #          vmin=0, vmax=1,
        #          cmap=cmocean.cm.ice, shading='nearest', zorder=10)
        plt.suptitle(salt.time_counter[i].dt.strftime(
                      '%y-%m-%d %H:%M').values)
        return pice
        #return p[0],p[1],pice
    
 
    axs[1].set_yticklabels([])

    axs[0].set_xlabel('longitude')
    axs[1].set_xlabel('longitude')
    axs[0].set_ylabel('latitude')

    plt.subplots_adjust(bottom=0.12, left=0.11, right=0.69,top=0.9, hspace=0.05,
                        wspace=0.05)
    
    pos = axs[1].get_position()
    cbar_ax = fig.add_axes([0.70, pos.y0, 0.02, pos.y1 - pos.y0])
    cbar = fig.colorbar(p[0], cax=cbar_ax, orientation='vertical')
    cbar.ax.text(4.9, 0.5, r'salinity', fontsize=8,
               rotation=90, transform=cbar.ax.transAxes, va='center', ha='left')

    cbar_ax = fig.add_axes([0.85, pos.y0, 0.02, pos.y1 - pos.y0])
    cbar = fig.colorbar(pice, cax=cbar_ax, orientation='vertical')
    cbar.ax.text(4.9, 0.5, r'sea ice concentration', fontsize=8,
               rotation=90, transform=cbar.ax.transAxes, va='center', ha='left')

    frames = int((len(salt.coords['time_counter']) - 1) )
    print (frames)
    #frames = 100
    line_ani = animation.FuncAnimation(fig, animate,frames=frames, blit=False)
    fname = 'surface_salt_and_ice_' + outfreq + '.mp4'
    line_ani.save(fname, writer=writer, dpi=300)
 
#models = ['EXP37','EXP39','EXP40','EXP41']
model = 'EXP10'
#make_salt_slice_movie_2param(model, outfreq='3h')
make_salt_slice_movie_2param(model, outfreq='24h')
