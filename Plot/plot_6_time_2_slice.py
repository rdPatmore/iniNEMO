import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


def interp_to_orca(orca, sochic):
    return sochic.interp(time_counter=orca.time_counter)

def add_slice(ax, ds, time, depth, x, y):
    ds = ds.isel(time_counter=time, deptht=depth)#.fillna(0.0)
    
    lev = np.linspace(-2,2,11)
    p = ax.pcolor(ds[x], ds[y], ds.votemper, vmin=-1.2, vmax=1.2, cmap=plt.cm.inferno)
    #p = ax.imshow(ds.x, ds.y, ds.votemper, cmap=plt.cm.inferno, levels=lev)

    return p

def plot(orca_path, sochic_path, model, depth0, depth1):
    '''
    main plotting routine adding six time slices over two depths
    '''

    # intialise plots
    fig, axs = plt.subplots(2,6, figsize=(6.5, 3.5))
    plt.subplots_adjust(bottom=0.1, top=0.98, right=0.86, left=0.1,
                        wspace=0.05, hspace=0.05)

    # load data
    orca   = xr.open_dataset(orca_path, decode_cf=False)
    sochic = xr.open_dataset(sochic_path, decode_cf=False)#.drop_attrs()#.isel(
    #sochic.time_counter.attrs['calendar'] = 'gregorian'
    sochic = xr.decode_cf(sochic)
    orca = xr.decode_cf(orca)
    sochic['time_counter'] = sochic.indexes['time_counter'].to_datetimeindex()
    print (orca.time_counter)
    print (sochic.time_counter)
                                                #  time_counter=slice(None, -2))

    # align time steps
    sochic = interp_to_orca(orca, sochic)
    #print (sochic.votemper)
  
    if model == 'orca':
        m = orca
        x = 'X'
        y = 'Y'
    if model == 'sochic':
        m = sochic
        x = 'x'
        y = 'y'
  
    # plot six time steps at depth0
    for i, ax in enumerate(axs[0]):
        add_slice(ax, m, i, depth0, x, y)

    # plot six time steps at depth1
    for i, ax in enumerate(axs[1]):
        p = add_slice(ax, m, i, depth1, x, y)

    pos = axs[0,1].get_position()
    cbar_ax = fig.add_axes([0.88, pos.y0, 0.02, pos.y1 - pos.y0])
    cbar = fig.colorbar(p, cax=cbar_ax)
    #cbar.locator = ticker.MaxNLocator(nbins=3)
    #cbar.update_ticks()
    #cbar.ax.text(4.5, 0.5, labels[i], fontsize=8, rotation=90,
    cbar.ax.text(4.5, 0.5, 'tempurature', fontsize=8, rotation=90,
                 transform=cbar.ax.transAxes, va='center', ha='right')
    for ax in axs[0,:]:
        ax.set_xticks([])
    for ax in axs[0,1:]:
        ax.set_yticks([])
    for ax in axs[1,1:]:
        ax.set_yticks([])

    plt.savefig('temp_' + model + '_compare.png')

if __name__ == '__main__':
    orca_path = '../processORCA12/DataOut/ORCA_PATCH_T.nc'
    sochic_path = '../Output/EXP01/SOCHIC_PATCH_20ts_20150101_20150131_grid_T.nc'
    plot(orca_path, sochic_path, 'orca', depth0=0, depth1=20)
    plot(orca_path, sochic_path, 'sochic', depth0=0, depth1=20)
