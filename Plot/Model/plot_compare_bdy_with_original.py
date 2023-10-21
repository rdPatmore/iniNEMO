import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import matplotlib.tri as tri
from scipy.interpolate import griddata
import pandas as pd
import seaborn as sns


def interp_to_orca(orca, sochic):
    return sochic.interp(time_counter=orca.time_counter)

#def add_slice(ax, ds, time, depth, x, y, coord):
#    print (ds)
#    ds = ds.isel(time_counter=time, deptht=depth).fillna(0.0)
#    print (ds)
#    
#    #print (ds.nav_lon)
#    lev = np.linspace(-2,2,11)
#    #p = ax.pcolor(ds[x], ds[y], ds.votemper, vmin=-1.2, vmax=1.2,
#    #              cmap=plt.cm.inferno)
#    data = ds.votemper[0].values
#
#    # use your x,y and z arrays here
#    x = ds[x][0]
#    y = ds[y][0]
#    z = ds.votemper[0]
#    new_x = coord.nav_lon.values
#    new_y = coord.nav_lat.values
#
#    #yy, xx = np.meshgrid(y,x)
#    #grid_lev = griddata(points, values, (grid_x, grid_y))
#    #zz = griddata((x,y),z,(new_x,new_y), method='nearest')
#    #ax.scatter(x,y,z)
#    lon_vals, lon_idx = np.unique(x, return_inverse=True)
#    lat_vals, lat_idx = np.unique(y, return_inverse=True)
#    new_x, new_y = np.meshgrid(lon_vals, lat_vals)
#    vals_array = np.empty(lon_vals.shape + lat_vals.shape)
#    vals_array.fill(np.nan) # or whatever yor desired missing data flag is
#    vals_array[lon_idx, lat_idx] = z
#    p = ax.pcolor(new_x.T, new_y.T, vals_array, vmin=-1.2, vmax=1.2,
#                  cmap=plt.cm.inferno)
#    #ax.plot(x,y, 'k.', ms=1)
#    #p = ax.pcolor(ds[x][0], ds[y][0].T, data,
#    #              cmap=plt.cm.inferno)
#    #p = ax.imshow(ds[x], ds[y], ds.votemper, cmap=plt.cm.inferno, levels=lev)
#    print ('YES    YES ')
#
#    return p

def add_slice(ax, ds, time, depth, x, y, coord):
    ds = ds.isel(time_counter=time, deptht=depth)#.fillna(0.0)
    
    p = ax.pcolor(ds[x], ds[y], ds.votemper, vmin=-1.2, vmax=1.2, cmap=plt.cm.inferno)

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
    #print (orca.time_counter)
    #print (sochic.time_counter)
    #time_counter=slice(None, -2))

    # align time steps
    #print (orca)
    #sochic = interp_to_orca(orca, sochic)
    #orca = orca.where(orca.nav_lon == sochic.nav_lon.values)#, drop=True)
    #orca = orca.where(orca.nav_lat==sochic.nav_lat.values)#, drop=True)
    #sochic = sochic.isel(time_counter=slice(7,19,2))
  
    if model == 'orca':
        m = orca
    #    x = 'X'
    #    y = 'Y'
        x = 'nav_lon'
        y = 'nav_lat'
    if model == 'sochic':
        m = sochic
        x = 'nav_lon'
        y = 'nav_lat'
    #    x = 'x'
    #    y = 'y'
  
    # plot six time steps at depth0
    for i, ax in enumerate(axs[0]):
        add_slice(ax, m, i, depth0, x, y, coord=orca)
    #p = add_slice(axs[0,0], m, 0, depth0, x, y, coord=orca)

    # plot six time steps at depth1
    for i, ax in enumerate(axs[1]):
        p = add_slice(ax, m, i, depth1, x, y, coord=orca)

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

    plt.savefig('temp_' + model + '_BDY_short_compare.png')

if __name__ == '__main__':
    orca_path = '../processORCA12/DataOut/ORCA_PATCH_T.nc'
    sochic_path = '../Bdy/BdyOut/bdy_T_ring.nc'
    
    plot(orca_path, sochic_path, 'orca', depth0=0, depth1=20)
    #plot(orca_path, sochic_path, 'sochic', depth0=0, depth1=20)
