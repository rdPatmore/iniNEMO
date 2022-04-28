import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import config
import itertools
import matplotlib
from get_transects import get_transects
matplotlib.rcParams.update({'font.size': 8})

def rotate(data, theta):
    ''' this is copied from model_data.py, for ds only
        get transects has a da version'''

    # translation lengths
    xt = data.lon.median()
    yt = data.lat.median()

    # translate to origin
    lon_orig = data.lon - xt
    lat_orig = data.lat - yt

    # rotate
    lon_rotated =  lon_orig * np.cos(theta) - lat_orig * np.sin(theta)
    lat_rotated =  lon_orig * np.sin(theta) + lat_orig * np.cos(theta)

    # translate to original position
    data['lon'] = lon_rotated + xt
    data['lat'] = lat_rotated + yt

    return data

def get_sampled_path(model, append, post_transect=True, rotation=None):
    path = config.data_path() + model + '/'
    file_path = path + 'GliderRandomSampling/glider_uniform_' + \
                append +  '_00.nc'
    glider = xr.open_dataset(file_path).sel(ctd_depth=10, method='nearest')
    glider['lon_offset'] = glider.attrs['lon_offset']
    glider['lat_offset'] = glider.attrs['lat_offset']
    glider = glider.set_coords(['lon_offset','lat_offset','time_counter'])
    #glider = rotate(glider, np.radians(-90))

    #glider['lon'] = glider.lon - glider.attrs['lon_offset']
    #glider['lat'] = glider.lat - glider.attrs['lat_offset']

    if post_transect:
        print ('rotation', rotation)
        glider = get_transects(glider.votemper, offset=True, rotation=rotation)
    return glider

def get_raw_path():
    glider_raw = xr.open_dataset(config.root() + 'Giddy_2020/merged_raw.nc')
    glider_raw = glider_raw.rename({'longitude': 'lon', 'latitude': 'lat'})
    index = np.arange(glider_raw.ctd_data_point.size)
    glider_raw = glider_raw.assign_coords(ctd_data_point=index)
    glider_raw = get_transects(glider_raw.dives, concat_dim='ctd_data_point',
                               shrink=100, offset=False)
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

def test_get_vertex():
    fig, ax = plt.subplots(1,1, figsize=(7.5,2.5))
    #plt.subplots_adjust(hspace=0.05,wspace=0.05, bottom=0.15,right=0.98,left=0.1)
    # plot post transect
    full_path = get_sampled_path('EXP10', 'interp_1000_rotate_90',
                post_transect=True, rotation=np.radians(90))
    #full_path = full_path.where(da.vertex==2., drop=True)
    #v0 = v0.swap_dims({'ctd_data_point':'transects'})
    for (l, v) in full_path.groupby('vertex'):
        plt.plot(v.lon, v.lat, label=l)
    plt.legend()
    plt.show()
    cmap = plt.cm.inferno(np.linspace(0,1,full_path.transect.max().values+1))
    for (l,trans) in full_path.groupby('transect'):
        ax.plot(trans.lon, trans.lat)
    
test_get_vertex()
