import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import config
import itertools
import matplotlib
from iniNEMO.Process.Glider.get_transects import get_transects
import matplotlib.colors as mcolors
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

def get_sampled_path_set(model, rotation=None):
    ''' load all 100 glider paths '''

    n=100
    def expand_sample_dim(ds):
        ds['lon_offset'] = ds.attrs['lon_offset']
        ds['lat_offset'] = ds.attrs['lat_offset']
        ds = ds.set_coords(['lon_offset','lat_offset','time_counter'])
        da = ds['b_x_ml']
        return da

    # load samples
    prep = '/GliderRandomSampling/glider_uniform_interp_1000_'
    if rotation:
        rotation_label = 'rotate_' + str(rotation) + '_' 
        rotation_rad = np.radians(rotation)
    else:
        rotation_label = ''
        rotation_rad = rotation # None type 

    sample_list = [config.data_path() + model + prep + rotation_label +
                   str(i).zfill(2) + '.nc' for i in range(n)]
    samples = xr.open_mfdataset(sample_list, 
                                 combine='nested', concat_dim='sample',
                                 preprocess=expand_sample_dim).load()

    # select depth
    samples = samples.sel(ctd_depth=10, method='nearest')

    # get transects
    sample_list = []
    for i in range(samples.sample.size):
        print ('sample: ', i)
        var10 = samples.isel(sample=i)
        sample_transect = get_transects(var10, offset=True,
                          rotation=rotation_rad)
        sample_list.append(sample_transect)
    samples=xr.concat(sample_list, dim='sample')
 
    return samples

def get_sampled_path(model, append, post_transect=True, rotation=None,
                     drop_meso=False):
    ''' load a single glider path '''
    path = config.data_path() + model + '/'
    file_path = path + 'GliderRandomSampling/glider_uniform_' + \
                append +  '_00.nc'
    glider = xr.open_dataset(file_path).sel(ctd_depth=10, method='nearest')
    glider['lon_offset'] = glider.attrs['lon_offset']
    glider['lat_offset'] = glider.attrs['lat_offset']
    glider = glider.set_coords(['lon_offset','lat_offset','time_counter'])

    if post_transect:
        print ('rotation', rotation)
        glider = get_transects(glider.votemper, offset=True, rotation=rotation,
                               method='from interp_1000')

    if drop_meso:
        glider = glider.where(glider.meso_transect==1, drop=True)
    return glider

def get_raw_path(drop_meso=False):
    glider_raw = xr.open_dataset(config.root() + 'Giddy_2020/merged_raw.nc')
    glider_raw = glider_raw.rename({'longitude': 'lon', 'latitude': 'lat'})
    index = np.arange(glider_raw.ctd_data_point.size)
    glider_raw = glider_raw.assign_coords(ctd_data_point=index)
    glider_raw = get_transects(glider_raw.dives, concat_dim='ctd_data_point',
                               shrink=100, offset=False)
    if drop_meso:
        glider_raw = glider_raw.where(glider_raw.meso_transect==1, drop=True)
    return glider_raw

def plot_path():
    #p500  = get_interp_path('EXP10', '500') 
    #p1000 = get_interp_path('EXP10', '1000') 
    #p2000 = get_interp_path('EXP10', '2000') 
    #raw = get_raw_path()


     
    #every_8 = get_sampled_path('EXP10', 'interp_1000') 
    #print (every_8)
    #print (lkjsdf)
    every_8 = get_sampled_path('EXP10', 'every_8') 
    cmap = plt.cm.inferno(np.linspace(0,1,every_8.transect.max().values+1))
    fig, axs = plt.subplots(5,5, figsize=(10,10))
    plt.subplots_adjust(hspace=0.02, wspace=0.02)
    axs = axs.flatten()
    for i, (l,trans) in enumerate(every_8.groupby('transect')):
        print (l)
        axs[i%25].plot(trans.lon, trans.lat)
        axs[i%25].set_xlim(every_8.lon.min(), every_8.lon.max())
        axs[i%25].set_ylim(every_8.lat.min(), every_8.lat.max())
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()
        #plt.clear()
    #raw = get_raw_path()
    #axs.set_xlim(raw.lon.min(), raw.lon.max())
    #axs.set_ylim(raw.lat.min(), raw.lat.max())
        #axs.plot(trans.lon, trans.lat, c=cmap[l])
    #for (l,trans) in p1000.groupby('transect'):
    #    print (l)
    #    axs[1].plot(trans.lon, trans.lat)
    #for (l,trans) in p2000.groupby('transect'):
    #    print (l)
    #    axs[2].plot(trans.lon, trans.lat)

    
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
    

def plot_model_buoyancy_gradients_patch(ax, da, sample):
    ''' restrict the model time to glider time and sample areas '''

    # add lat-lon to dimensions
    da = da.assign_coords({'lon':(['x'], da.nav_lon.isel(y=0)),
                           'lat':(['y'], da.nav_lat.isel(x=0))})
    da = da.swap_dims({'x':'lon','y':'lat'})

    # get limts of sample
    x0 = float(sample.lon.min())
    x1 = float(sample.lon.max())
    y0 = float(sample.lat.min())
    y1 = float(sample.lat.max())

    patch = da.sel(lon=slice(x0,x1),
                   lat=slice(y0,y1))
    ax.pcolor(patch.nav_lon, patch.nav_lat, patch, cmap=plt.cm.binary,
              shading='nearest')


def plot_patch_sampling():
    ''' plot domain surface temperature with glider paths overlian '''

    model='EXP10'

    fig, ax = plt.subplots(1,1, figsize=(5.5,5.5))
    plt.subplots_adjust(right=0.78)

    path = config.data_path() + model + '/'
    file_path = path + '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc'
    m = xr.open_dataset(file_path, chunks='auto').sel(
                                       time_counter='2013-01-15 00:00:00',
                                       deptht=0,
                                       method='nearest').votemper
    m = m.isel(x=slice(10,-10),y=slice(10,-10))

    p = ax.pcolor(m.nav_lon, m.nav_lat, m, cmap=plt.cm.inferno,
                  shading='nearest')
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    samples = get_sampled_path_set('EXP10')
    for (_, sample) in samples.groupby('sample'):
        #plot_model_buoyancy_gradients_patch(ax, m, sample)
        for (l, v) in sample.groupby('vertex'):
            print (l)
            c = list(mcolors.TABLEAU_COLORS)[int(l)]
            plt.plot(v.lon, v.lat, c=c)

    ax.set_aspect('equal')
    ax.set_xlim(-3.8,3.5)
    ax.set_ylim(-63,-58)
    pos = ax.get_position()
    cbar_ax = fig.add_axes([0.82, pos.y0, 0.02, pos.y1 - pos.y0])
    cbar = fig.colorbar(p, cax=cbar_ax)

    cbar.ax.text(7.0, 0.5, r'$\theta (^{\circ} C)$', fontsize=8, rotation=90,
                 transform=cbar.ax.transAxes, va='center', ha='right')

    plt.savefig('temp_with_glider_paths.png', dpi=600)
#plot_patch_sampling()
