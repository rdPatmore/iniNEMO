import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cmocean
import config

def make_w_slice_movie_4model(models, depth):
    ''' make movie of vertical velocities '''
     

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1e6)

    fig, axs = plt.subplots(2,2,figsize=(6.5,5.5))

    #p0 = axs[0,0].imshow(u.isel(time_counter=0), extent=[0,300,0,300],
    #                  vmin=-1e-1, vmax=1e-1,
    #                  aspect='auto', animated=True, cmap=cmocean.cm.balance)
    #p1= axs[1,0].imshow(v.isel(time_counter=0), extent=[0,300,0,300],
    #                  vmin=-1e-1, vmax=1e-1,
    #                  aspect='auto', animated=True, cmap=cmocean.cm.balance)

    fn = 'SOCHIC_PATCH_1h_20120101_20120408_grid_'
    p = []
    w_list = []
    for i, model in enumerate(models):
        w = xr.open_dataset(config.data_path() + model + '/' + fn + 'W.nc').wo
        w = w.isel(x=slice(2,-2),y=slice(2,-2))
        w = w.sel(depthw=depth, method='nearest')
        w_list.append(w)
        p.append(axs.flatten()[i].imshow(w.isel(time_counter=0),
                      vmin=-3e-4, vmax=3e-4,
                      aspect='equal', animated=True, cmap=cmocean.cm.balance))
                      #extent=[0,300,0,300]s
    def animate(i):
        print (i)
        for j in range(3):
            p[j].set_array(w_list[j].isel(time_counter=i*2))
        plt.suptitle(w_list[0].time_counter[i*2].dt.strftime(
                      '%y-%m-%d %H:%M').values)
 
    #axs[1,0].set_xlabel('x [m]')
    #axs[1,1].set_xlabel('x [m]')
    #axs[0,0].set_ylabel('y [m]')
    #axs[1,0].set_ylabel('y [m]')
    axs[0,0].set_xticklabels([])
    axs[0,1].set_xticklabels([])
    axs[0,1].set_yticklabels([])
    axs[1,1].set_yticklabels([])
    axs[0,0].text(0.1,0.1,'std', bbox=dict(fc='white', alpha=0.6), 
                  ha='left',va='top', size=6, transform=axs[0,0].transAxes)
    axs[0,1].text(0.1,0.1,'orlanski 8 rim', bbox=dict(fc='white', alpha=0.6),
                  ha='left',va='top', size=6, transform=axs[0,1].transAxes)
    axs[1,0].text(0.1,0.1,'frs 8 rim', bbox=dict(fc='white', alpha=0.6),
                  ha='left',va='top', size=6, transform=axs[1,0].transAxes)
    #axs[1,1].text(0.1,0.1,'wind smooth 10 day', bbox=dict(fc='white', alpha=0.6),
    #              ha='left',va='top', size=6, transform=axs[1,1].transAxes)
 
    #for ax in axs.flatten():
    #    x_label_list = ['', 'B2', 'C2', 'D2']
    #    ax.set_xticks([-0.75,-0.25,0.25,0.75])

    plt.subplots_adjust(bottom=0.1,left=0.05, right=0.79,top=0.9, hspace=0.05,
                        wspace=0.05)
    #cbar = plt.colorbar(p[0])
    
    pos0 = axs[0,1].get_position()
    pos1 = axs[1,1].get_position()
    cbar_ax = fig.add_axes([0.80, pos1.y0, 0.02, pos0.y1 - pos1.y0])
    cbar = fig.colorbar(p[0], cax=cbar_ax, orientation='vertical')
    cbar.ax.text(8.3, 0.5, r'w [m s$^{-1}$]', fontsize=8,
               rotation=90, transform=cbar.ax.transAxes, va='center', ha='left')

    #cbar.set_label(units)
    #frames = len(w.coords['time_counter']) - 1
    frames = int((len(w.coords['time_counter']) - 1) / 2)
    print (frames)
    #frames = 10
    line_ani = animation.FuncAnimation(fig, animate,frames=frames, blit=False)
    fname = 'edge_smoothing_z500.mp4'
    line_ani.save(fname, writer=writer, dpi=300)
 
#models = ['EXP37','EXP39','EXP40','EXP41']
models = ['EXP37','EXP42','EXP43']
make_w_slice_movie_4model(models, 500)
