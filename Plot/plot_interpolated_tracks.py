import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import config
import itertools

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
                test = ((da.lat < -60.10) and (da.lon > 0.176))
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
                print (test)
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
    return da

def get_interp_path(model, interp):
    path = config.data_path() + model + '/'
    file_path = path + 'GliderRandomSampling/glider_uniform_interp_' + \
                interp +  '_00.nc'
    glider = xr.open_dataset(file_path).sel(ctd_depth=10, method='nearest')
    glider['lon'] = glider.lon - glider.attrs['lon_offset']
    glider['lat'] = glider.lat - glider.attrs['lat_offset']
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

def plot_paths():
    #p500  = get_interp_path('EXP10', '500') 
    #p1000 = get_interp_path('EXP10', '1000') 
    #p2000 = get_interp_path('EXP10', '2000') 
    raw = get_raw_path()

    fig, axs = plt.subplots(1,1, figsize=(10,10))

     
    cmap = plt.cm.inferno(np.linspace(0,1,raw.transect.max().values+1))
    for (l,trans) in raw.groupby('transect'):
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
    
plot_paths()
