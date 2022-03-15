import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import config
import itertools
import matplotlib
matplotlib.rcParams.update({'font.size': 8})


def get_transects(data, concat_dim='distance', method='cycle', shrink=None):
    if method == '2nd grad':
        a = np.abs(np.diff(data.lat, 
        append=data.lon.max(), prepend=data.lon.min(), n=2))# < 0.001))[0]
        idx = np.where(a>0.006)[0]
    crit = [0,1,2,3]
    if method == 'cycle':
        idx=[]
        crit_iter = itertools.cycle(crit)
        start = True
        a = next(crit_iter)
        for i in range(data[concat_dim].size)[::shrink]:
            da = data.isel({concat_dim:i})
            if (a == 0) and (start == True):
                test = ((da.lat < -60.04) and (da.lon > 0.176))
            elif a == 0:
                test = (da.lon > 0.176)
            elif a == 1:
                test = (da.lat > -59.93)
            elif a == 2:
                test = (da.lon < -0.173)
            elif a == 3:
                test = (da.lat > -59.93)
            if test: 
                start = False
                idx.append(i)
                a = next(crit_iter)
    da = np.split(data, idx)
    transect = np.arange(len(da))
    pop_list=[]
    for i, arr in enumerate(da):
        if len(da[i]) < 1:
            pop_list.append(i) 
        else:
            da[i] = da[i].assign_coords({'transect':i})
    for i in pop_list:
        da.pop(i)
    da = xr.concat(da, dim=concat_dim)

    # remove initial and mid path excursions
    da = da.where(da.transect>1, drop=True)
    da = da.where(da.transect != da.lat.idxmin().transect, drop=True)
    # restrict to 1st 19 transects
    da = da.where(da.transect < 10, drop=True)
    return da

def get_sampled_path(model, append, post_transect=True):
    path = config.data_path() + model + '/'
    file_path = path + 'GliderRandomSampling/glider_uniform_' + \
                append +  '_00.nc'
    glider = xr.open_dataset(file_path).sel(ctd_depth=10, method='nearest')
    glider['lon'] = glider.lon - glider.attrs['lon_offset']
    glider['lat'] = glider.lat - glider.attrs['lat_offset']
    if post_transect:
        glider = get_transects(glider.votemper)
    return glider

def get_raw_path():
    glider_raw = xr.open_dataset(config.root() + 'Giddy_2020/merged_raw.nc')
    glider_raw = glider_raw.rename({'longitude': 'lon', 'latitude': 'lat'})
    index = np.arange(glider_raw.ctd_data_point.size)
    glider_raw = glider_raw.assign_coords(ctd_data_point=index)
    glider_raw = get_transects(glider_raw.dives, concat_dim='ctd_data_point',
                               shrink=100)
    return glider_raw

def plot_path():
    #p500  = get_interp_path('EXP10', '500') 
    #p1000 = get_interp_path('EXP10', '1000') 
    #p2000 = get_interp_path('EXP10', '2000') 
    #raw = get_raw_path()

    fig, axs = plt.subplots(1,1, figsize=(10,10))

     
    full_path = get_sampled_path('EXP10', 'interp_1000_trans') 
    cmap = plt.cm.inferno(np.linspace(0,1,full_path.transect.max().values+1))
    for (l,trans) in full_path.groupby('transect'):
        print (l)
        axs.plot(trans.lon, trans.lat)
        #axs.plot(trans.lon, trans.lat, c=cmap[l])
    #for (l,trans) in p1000.groupby('transect'):
    #    print (l)
    #    axs[1].plot(trans.lon, trans.lat)
    #for (l,trans) in p2000.groupby('transect'):
    #    print (l)
    #    axs[2].plot(trans.lon, trans.lat)
    plt.show()

    
def plot_paths():
    fig, axs = plt.subplots(2,4, figsize=(7.5,2.5))
    plt.subplots_adjust(hspace=0.05,wspace=0.05, bottom=0.15,right=0.98,left=0.1)

    raw = get_raw_path()
    cmap = plt.cm.inferno(np.linspace(0,1,raw.transect.max().values+1))
    for (l,trans) in raw.groupby('transect'):
        axs[0,3].plot(trans.lon, trans.lat)

    # plot post transect
    full_path = get_sampled_path('EXP10', 'interp_1000') 
    cmap = plt.cm.inferno(np.linspace(0,1,full_path.transect.max().values+1))
    for (l,trans) in full_path.groupby('transect'):
        axs[0,0].plot(trans.lon, trans.lat)

    every_2 = get_sampled_path('EXP10', 'every_2') 
    cmap = plt.cm.inferno(np.linspace(0,1,every_2.transect.max().values+1))
    for (l,trans) in every_2.groupby('transect'):
        axs[0,1].plot(trans.lon, trans.lat)

    every_8 = get_sampled_path('EXP10', 'every_8') 
    cmap = plt.cm.inferno(np.linspace(0,1,every_8.transect.max().values+1))
    for (l,trans) in every_8.groupby('transect'):
        axs[0,2].plot(trans.lon, trans.lat)

    # plot pre transect
    full_path = get_sampled_path('EXP10', 'interp_1000_transects',
                                 post_transect=False) 
    cmap = plt.cm.inferno(np.linspace(0,1,
                                        int(full_path.transect.max().values)+1))
    for (l,trans) in full_path.groupby('transect'):
        axs[1,0].plot(trans.lon, trans.lat)

    every_2 = get_sampled_path('EXP10', 'every_2_transects',
                                 post_transect=False) 
    cmap = plt.cm.inferno(np.linspace(0,1,int(every_2.transect.max().values)+1))
    for (l,trans) in every_2.groupby('transect'):
        axs[1,1].plot(trans.lon, trans.lat)

    every_8 = get_sampled_path('EXP10', 'every_8_transects',
                                 post_transect=False) 
    cmap = plt.cm.inferno(np.linspace(0,1,int(every_8.transect.max().values)+1))
    for (l,trans) in every_8.groupby('transect'):
        axs[1,2].plot(trans.lon, trans.lat)

    for ax in axs.ravel():
        ax.set_aspect('equal')
        ax.set_xlim([-0.235,0.235])
        ax.set_ylim([-60.12,-59.88])
    axs[0,0].set_ylabel('latitude')
    axs[1,0].set_ylabel('latitude')
    for ax in axs[1]:
        ax.set_xlabel('longitude')
    for ax in axs[:,1:].ravel():
        ax.set_yticks([])
    for ax in axs[0]:
        ax.set_xticks([])
    axs[0,0].set_title('full path (interp)')
    axs[0,1].set_title('every 2 (interp)')
    axs[0,2].set_title('every 8 (interp)')
    axs[0,3].set_title('raw path')
    axs[1,3].axis('off')
    plt.savefig('transect_method.png', dpi=300)

plot_paths()
