import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from common import time_mean_to_orca, monthly_mean
import cmocean
from processORCA12.gridding import regrid
import calendar
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 0.2 #set the value globally
#plt.style.use('paper')

def plot(model, orca_path, sochic_path, var, vmin=-1, vmax=1, depth_dict=None,
         cmap=cmocean.cm.balance, sym_bounds=False, orca_time=True):
    '''
    main plotting routine adding six time slices over two depths
    '''

    # intialise plots
    fig, axs = plt.subplots(3,12, figsize=(10.5, 4.0))
    plt.subplots_adjust(bottom=0.12, top=0.92, right=0.90, left=0.08,
                        wspace=0.15, hspace=0.12)

    # load data
    orca   = xr.open_dataset(orca_path, decode_cf=False)
    sochic = xr.open_dataset(sochic_path, decode_cf=True)

    # translate var names
    try:
        sochic = sochic.rename({'wfo'    : 'sowaflup',
                                'qsr_oce': 'soshfldo',
                                'qt_oce' : 'sohefldo'})
    except:
        print ('not new T')

    try:
        sochic = sochic.rename({ 'uo'     : 'vozocrtx'})
    except:
        print ('not new U')

    try:
        sochic = sochic.rename({'vo'     : 'vomecrty'})
    except:
        print ('not new V')

    if var == 'siconc':
        var = 'icepres'
        orca = orca.rename({'siconc'     : 'icepres'})


    if depth_dict:
        print ('oui')
        #orca = orca.isel(deptht=list(depth_dict.values())[0])
        orca = orca.isel(depth_dict)
        sochic = sochic.isel(depth_dict)

    # translate orca dates to cf
    #orca.time_counter.attrs['units'] = 'seconds since 1900-01-01'
    #orca.time_centered.attrs['units'] = 'seconds since 1900-01-01'
    #    print ('no time centered')
    orca = xr.decode_cf(orca)

    # align time steps
    try:
        sochic['time_counter'] = sochic.indexes['time_counter'].to_datetimeindex()
    except:
        print ('leap skipping to_datetimeindex')
    if orca_time:
        time_var = 'time_counter'
        sochic = time_mean_to_orca(orca, sochic)
    else:
        time_var = 'month'
  
    # align dim labels
    try:
        orca = orca.rename_dims({'X':'x', 'Y':'y'})
    except:
        print ('x y already are dims')

    # remove x and y coordinate (it messes with plotting)
    orca = orca.reset_coords(['X','Y'], drop=True)
    #print (sochic[var][0])
    #print (sochic[var][1])
    #print (sochic[var][2])
    #print (sochic[var][:,0])
    #print (sochic[var][:,1])
    #print (sochic[var][:,2])
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
    sochic = sochic.reindex(month=orca.month)

    vmin = orca[var].min()
    vmax = orca[var].max()
    if sym_bounds:
        vmax = max(np.abs(vmin), np.abs(vmax))
        vmin = -vmax
    if var == 'siconc':
        orca['siconc'] = orca.siconc.fillna(0.0)
    if var == 'siconc':
        orca['icepres'] = orca.icepres.fillna(0.0)
    if var == 'sithic':
        orca['sithic'] = orca.sithic.fillna(0.0)

    # plot six time steps of orca
    for i, ax in enumerate(axs[0]):
        ds = orca.isel({time_var:i})
        p = ax.pcolor(ds.nav_lon, ds.nav_lat, ds[var], vmin=vmin, vmax=vmax,
                      cmap=cmap, shading='nearest')

    # plot six time steps of sochic
    for i, ax in enumerate(axs[1]):
        ds = sochic.isel({time_var:i})
        #p0 = ax.imshow(ds[var].values, vmin=vmin, vmax=vmax,
        #                   cmap=cmap, interpolation='none')
        p0 = ax.pcolormesh(ds.nav_lon, ds.nav_lat,ds[var].values,
                           vmin=vmin, vmax=vmax,
                           cmap=cmap, shading='nearest')

    # get model differences
    orca_new = regrid(orca, sochic, var, x='x', y='y')
    diff = sochic[var] - orca_new[var]

    # plot six time steps of sochic - orca
    for i, ax in enumerate(axs[2]):
        arr = diff.isel({time_var:i})
        maximum = (orca[var].max() - orca[var].min()) / 10
        minimum = - maximum
        #maximum = np.abs(max(vmin, vmax) / 10)
        p1 = ax.pcolor(sochic.nav_lon, sochic.nav_lat, arr,
                       vmin=minimum, vmax=maximum, cmap=cmocean.cm.balance, 
                       shading='nearest')

    pos0 = axs[0,-1].get_position()
    pos1 = axs[1,-1].get_position()
    cbar_ax = fig.add_axes([0.92, pos1.y0, 0.02, pos0.y1 - pos1.y0])
    cbar0 = fig.colorbar(p0, cax=cbar_ax)

    pos2 = axs[2,-1].get_position()
    cbar_ax = fig.add_axes([0.92, pos2.y0, 0.02, pos2.y1 - pos2.y0])
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
                  'icepres': 'sea ice presence',
                  'siconc': 'sea ice concentration',
                  'sithic': 'sea ice thickness',
                  'snthic': 'snow thickness'}

    # colourbars
    var_human = var_lookup[var] 
    cbar0.ax.text(4.0, 0.5, var_human, fontsize=8, rotation=90,
                 transform=cbar0.ax.transAxes, va='center', ha='right')

    cbar1.ax.text(4.0, 0.5, 'difference', fontsize=8, rotation=90,
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
        dt = orca[time_var][i]
    #    dt = orca[time_var][i].dt
    #    #title = str(dt.dayofyear.values) + ' ' + str(dt.hour.values)
    #    title = str(dt.dayofyear.values) + ' ' +  dt.strftime('%b').values
    #    print ('title', title)
        title = calendar.month_abbr[int(dt)]
        ax.set_title(title)
    #for ax in axs.flatten():
         #ax.axis('off')
         #ax.spines['top'].set_visible(False)
         #ax.spines['right'].set_visible(False)
         #ax.spines['bottom'].set_visible(False)
         #ax.spines['left'].set_visible(False)

    #    ax.set_xlim([sochic.nav_lon.min()-0.4, sochic.nav_lon.max()+0.2])
    #    ax.set_ylim([sochic.nav_lat.min()-0.4, sochic.nav_lat.max()+0.2])

    plt.savefig(model + '_flux_diff_' + var + '.png', dpi=1200)

if __name__ == '__main__':
    model = 'EXP08'
    #outdir = '../Output/'
    #outdir = '/work/n02/n02/ryapat30/nemo/nemo/cfgs/SOCHIC_PATCH_ICE/'
    #outpath = outdir + model 
    #dates = '20120101_20121231'
    #orca_path = '../OrcaCutData/ORCA0083-N06_y2015m11_T_conform.nc'
    #orca_path = '../OrcaCutData/ORCA_PATCH_2012_T.nc'
    #sochic_path = '../Output/EXP54/SOCHIC_PATCH_1h_20150101_20150128_grid_T.nc'
    #sochic_path = outpath + '/SOCHIC_PATCH_3h_' + dates + '_grid_T.nc'
    #sochic_path = '../Output/EXP60/SOCHIC_PATCH_1h_20150101_20150121_grid_T.nc'

#    orca_path   = 'tmp/orca_T.nc'
#    sochic_path = 'tmp/sochic_' + model + '_T.nc'
#    plot(model, orca_path, sochic_path, 'sowaflup', sym_bounds=True,
#         orca_time=False)
#    plot(model, orca_path, sochic_path, 'soshfldo', cmap=plt.cm.inferno,
#         orca_time=False)
#    plot(model, orca_path, sochic_path, 'sohefldo', sym_bounds=True,
#         orca_time=False)
#    plot(model, orca_path, sochic_path, 'votemper', 
#         depth_dict={'deptht':0}, cmap=plt.cm.inferno, orca_time=False)
#    plot(model, orca_path, sochic_path, 'vosaline',
#         depth_dict={'deptht':0}, cmap=plt.cm.inferno, orca_time=False)

    #orca_path = '../processORCA12/DataOut/ORCA_PATCH_y2015m11_U.nc'
    #orca_path = '../processORCA12/DataOut/ORCA0083-N06_y2015m11_U_conform.nc'
    #sochic_path = outpath + '/SOCHIC_PATCH_1h_' + dates + '_grid_U.nc'
#    orca_path = 'tmp/orca_U.nc'
#    sochic_path = 'tmp/sochic_' + model + '_U.nc'
#    plot(model, orca_path, sochic_path, 'vozocrtx', sym_bounds=True,
#         depth_dict={'depthu':0}, orca_time=False)
#
#    #orca_path = '../processORCA12/DataOut/ORCA_PATCH_y2015m11_V.nc'
#    #orca_path = '../processORCA12/DataOut/ORCA0083-N06_y2015m11_V_conform.nc'
#    #sochic_path = outpath + '/SOCHIC_PATCH_1h_' + dates + '_grid_V.nc'
#    orca_path = 'tmp/orca_V.nc'
#    sochic_path = 'tmp/sochic_' + model + '_V.nc'
#    plot(model, orca_path, sochic_path, 'vomecrty', sym_bounds=True, 
#         depth_dict={'depthv':0}, orca_time=False)

   # orca_path = '../processORCA12/DataOut/ORCA_PATCH_y2015m11_I.nc'
   # sochic_path = outpath + '/SOCHIC_PATCH_1h_' + dates + '_icemod.nc'
    orca_path = 'tmp/orca_I.nc'
    sochic_path = 'tmp/sochic_' + model + '_I.nc'
    plot(model, orca_path, sochic_path, 'siconc', sym_bounds=False,
         cmap=cmocean.cm.ice_r, orca_time=False)
    plot(model, orca_path, sochic_path, 'sithic', sym_bounds=False,
         cmap=cmocean.cm.ice_r, orca_time=False)
#     # no sea ice thickness in data... ## Find its name... 
##    plot(model, orca_path, sochic_path, 'snthic', 0, 0.5, sym_bounds=False,
##         cmap=cmocean.cm.ice)
