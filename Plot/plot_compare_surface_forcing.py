import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from common import time_mean_to_orca
import cmocean
from processORCA12.gridding import regrid

plt.style.use('paper')

def plot(model, orca_path, sochic_path, var, vmin=-1, vmax=1, depth_dict=None,
         cmap=cmocean.cm.balance, sym_bounds=False):
    '''
    main plotting routine adding six time slices over two depths
    '''

    # intialise plots
    fig, axs = plt.subplots(3,3, figsize=(6.5, 4.0))
    plt.subplots_adjust(bottom=0.12, top=0.95, right=0.86, left=0.1,
                        wspace=0.15, hspace=0.12)

    # load data
    orca   = xr.open_dataset(orca_path, decode_cf=False)
    sochic = xr.open_dataset(sochic_path, decode_cf=True)

    if depth_dict:
        print ('oui')
        #orca = orca.isel(deptht=list(depth_dict.values())[0])
        orca = orca.isel(depth_dict)
        sochic = sochic.isel(depth_dict)
    

    # translate orca dates to cf
    print ('space')
    print ('space')
    print ('space')
    print ('space')
    print ('space')
    print ('space')
    print (orca.time_counter.attrs)
    print ('space')
    print ('space')
    print ('space')
    print ('space')
    print ('space')
    print ('space')
    print ('space')
    #orca.time_counter.attrs['units'] = 'seconds since 1900-01-01'
    #orca.time_centered.attrs['units'] = 'seconds since 1900-01-01'
    #try:
    #except:
    #    print ('no time centered')
    orca = xr.decode_cf(orca)

    # align time steps
    try:
        sochic['time_counter'] = sochic.indexes['time_counter'].to_datetimeindex()
    except:
        print ('leap skipping to_datetimeindex')
    sochic = time_mean_to_orca(orca, sochic)
  
    # align dim labels
    try:
        orca = orca.rename_dims({'X':'x', 'Y':'y'})
    except:
        print ('x y already are dims')
    #try:
    #    sochic = sochic.rename({'longitude':'nav_lon', 'latitude':'nav_lat'})
    #except:
    #    print ('x y already are dims')
    #try:
    #    orca = orca.rename({'longitude':'nav_lon', 'latitude':'nav_lat'})
    #except:
    #    print ('x y already are dims')

    # cut to start and end
    #orca = orca.isel(time_counter=slice(1,-1))
    #sochic = sochic.isel(time_counter=slice(1,-1))
  

    vmin = orca[var].min()
    vmax = orca[var].max()
    if sym_bounds:
        vmax = max(np.abs(vmin), np.abs(vmax))
        vmin = -vmax
    print ('')
    print ('')
    print ('')
    print ('')
    print ('')
    print ('')
    #print (sochic)
    print ()
    print ('')
    print ('')
    print ('')
    print ('')
    print ('')

    # plot six time steps of orca
    for i, ax in enumerate(axs[0]):
        ds = orca.isel(time_counter=i)
        p = ax.pcolor(ds.nav_lon, ds.nav_lat, ds[var], vmin=vmin, vmax=vmax,
                      cmap=cmap)

    # plot six time steps of sochic
    for i, ax in enumerate(axs[1]):
        ds = sochic.isel(time_counter=i)
        p0 = ax.pcolor(ds.nav_lon, ds.nav_lat, ds[var], vmin=vmin, vmax=vmax,
                      cmap=cmap)

    # get model differences
    orca_new = regrid(orca, sochic, var, x='x', y='y')
    diff = sochic[var] - orca_new[var]

    # plot six time steps of sochic - orca
    for i, ax in enumerate(axs[2]):
        arr = diff.isel(time_counter=i)
        maximum = np.abs(max(vmin, vmax) / 10)
        minimum = - maximum
        p1 = ax.pcolor(sochic.nav_lon, sochic.nav_lat, arr,
                       vmin=minimum, vmax=maximum, cmap=cmocean.cm.balance)

    pos0 = axs[0,-1].get_position()
    pos1 = axs[1,-1].get_position()
    cbar_ax = fig.add_axes([0.88, pos1.y0, 0.02, pos0.y1 - pos1.y0])
    cbar0 = fig.colorbar(p0, cax=cbar_ax)

    pos2 = axs[2,-1].get_position()
    cbar_ax = fig.add_axes([0.88, pos2.y0, 0.02, pos2.y1 - pos2.y0])
    cbar1 = fig.colorbar(p1, cax=cbar_ax)

    #cbar.locator = ticker.MaxNLocator(nbins=3)
    #cbar.update_ticks()
    #cbar.ax.text(4.5, 0.5, labels[i], fontsize=8, rotation=90,

    # translate names
    var_lookup = {'votemper': 'temperature',
                  'vosaline': 'salinity',
                  'vozocrtx': 'u',
                  'vomecrty': 'v',
                  'sowaflup': 'fresh water flux',
                  'soshfldo': 'shortwave radiation',
                  'sohefldo': 'heat flux',
                  'siconc': 'sea ice concentration',
                  'sithic': 'sea ice thickness',
                  'snthic': 'snow thickness'}

    # colourbars
    var_human = var_lookup[var] 
    cbar0.ax.text(5.5, 0.5, var_human, fontsize=8, rotation=90,
                 transform=cbar0.ax.transAxes, va='center', ha='right')

    cbar1.ax.text(5.5, 0.5, 'difference', fontsize=8, rotation=90,
                 transform=cbar1.ax.transAxes, va='center', ha='right')

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
    for i, ax in enumerate(axs[0]):
        dt = orca.time_counter[i].dt
        #title = str(dt.dayofyear.values) + ' ' + str(dt.hour.values)
        title = str(dt.dayofyear.values) + ' ' +  dt.strftime('%b').values
        print ('title', title)
        ax.set_title(title)

    plt.savefig(model + '_flux_diff_' + var + '.png')

if __name__ == '__main__':
    model = 'EXP005'
    outdir = '../Output/'
    outpath = outdir + model 
    dates = '20151104_20151114'
    orca_path = '../processORCA12/DataOut/ORCA0083-N06_y2015m11_T_conform.nc'
    #sochic_path = '../Output/EXP54/SOCHIC_PATCH_1h_20150101_20150128_grid_T.nc'
    sochic_path = outpath + '/SOCHIC_PATCH_1h_' + dates + '_grid_T.nc'
    #sochic_path = '../Output/EXP60/SOCHIC_PATCH_1h_20150101_20150121_grid_T.nc'

    plot(model, orca_path, sochic_path, 'sowaflup', -1e-4, 1e-4, sym_bounds=True)
    plot(model, orca_path, sochic_path, 'soshfldo', 300,
         cmap=plt.cm.inferno)
    plot(model, orca_path, sochic_path, 'sohefldo', -210, 210, sym_bounds=True)
#    plot(model, orca_path, sochic_path, 'votemper', -1.2, 1.2, 
###         depth_dict={'deptht':0}, cmap=plt.cm.inferno)
###    plot(model, orca_path, sochic_path, 'vosaline', 33, 34.5, 
###         depth_dict={'deptht':0}, cmap=plt.cm.inferno)

    #orca_path = '../processORCA12/DataOut/ORCA_PATCH_y2015m11_U.nc'
    orca_path = '../processORCA12/DataOut/ORCA0083-N06_y2015m11_U_conform.nc'
    sochic_path = outpath + '/SOCHIC_PATCH_1h_' + dates + '_grid_U.nc'
    plot(model, orca_path, sochic_path, 'vozocrtx', -1.2, 1.2, sym_bounds=True,
         depth_dict={'depthu':0})

    #orca_path = '../processORCA12/DataOut/ORCA_PATCH_y2015m11_V.nc'
    orca_path = '../processORCA12/DataOut/ORCA0083-N06_y2015m11_V_conform.nc'
    sochic_path = outpath + '/SOCHIC_PATCH_1h_' + dates + '_grid_V.nc'
    plot(model, orca_path, sochic_path, 'vomecrty', -1.2, 1.2, sym_bounds=True, 
         depth_dict={'depthv':0})

    orca_path = '../processORCA12/DataOut/ORCA_PATCH_y2015m11_I.nc'
    #orca_path = '../processORCA12/DataOut/ORCA0083-N06_y2015m11_I_conform.nc'
    sochic_path = outpath + '/SOCHIC_PATCH_1h_' + dates + '_icemod.nc'
    plot(model, orca_path, sochic_path, 'siconc', 0, 1, sym_bounds=False,
         cmap=cmocean.cm.ice)
    plot(model, orca_path, sochic_path, 'sithic', 0, 0.5, sym_bounds=False,
         cmap=cmocean.cm.ice)
     # no sea ice thickness in data... ## Find its name... 
#    plot(model, orca_path, sochic_path, 'snthic', 0, 0.5, sym_bounds=False,
#         cmap=cmocean.cm.ice)
