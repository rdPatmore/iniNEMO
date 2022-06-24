import xarray as xr
import numpy  as np
import itertools

def rotate_path(data, theta):
    ''' this is copied from model_data.py'''

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
    # using drop to stop old coord being replaced
    data['lon'] = (lon_rotated + xt).drop(['lon','lat']) 
    data['lat'] = (lat_rotated + yt).drop(['lon','lat']) 
    print (data)

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

def get_transects(da, concat_dim='distance', method='cycle',
                  shrink=None, drop_trans=[False,False,False,False],
                  offset=False, rotation=None):
    '''
    da is xr DataArray only

    drop_tran: drop sides
               SW -> NE
               E
               SE -> NW
               W
    '''

    if method == '2nd grad':
        a = np.abs(np.diff(da.lat, 
        append=da.lon.max(), prepend=da.lon.min(), n=2))# < 0.001))[0]
        idx = np.where(a>0.006)[0]
    crit = [0,1,2,3]
    if method == 'cycle':
        #da = da.isel(distance=slice(0,400))

        # some paths are saved rotated
        if rotation: # shift back to unrotated
            da = rotate_path(da, -rotation)


        # shift back to origin
        if offset:
            da['orig_lon'] = da.lon - da.lon_offset
            da['orig_lat'] = da.lat - da.lat_offset
        else:
            da['orig_lon'] = da.lon
            da['orig_lat'] = da.lat
 

        idx=[]
        crit_iter = itertools.cycle(crit)
        start = True
        a = next(crit_iter)
        for i in range(da[concat_dim].size)[::shrink]:
            dp = da.isel({concat_dim:i})
            if (a == 0) and (start == True):
                test = ((dp.orig_lat < -60.04) & (dp.orig_lon > 0.176))
            elif a == 0:
                test = (dp.orig_lon > 0.176)
            elif a == 1:
                test = (dp.orig_lat > -59.93)
            elif a == 2:
                test = (dp.orig_lon < -0.173)
            elif a == 3:
                test = (dp.orig_lat > -59.93)
            if test: 
                start = False
                idx.append(i)
                a = next(crit_iter)


    da = np.split(da, idx)
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
    print (da)
    da = da.where(da.transect>1, drop=True)
    # this should work but there is a bug in xarray perhaps
    # idxmin drops all coordinates...
    #da = da.where(da.transect != da.lat.idxmin(skipna=True).transect,drop=True)
    transect_south = da.isel(distance=da.lat.argmin(skipna=True).values)
    da = da.where(da.transect != transect_south, drop=True)

    # re-rotate
    if rotation:
        da = rotate_path(da, rotation)

    # catagorise
    category = da.transect%4
    da = da.assign_coords({'vertex': da.transect%4})

    #category = (np.tile([0,1,2,3], 1 + (da.size/4)))[:ds.size]
    #print (np.unique(category))
    #print(da)
    #import matplotlib.pyplot as plt
    #da['vertex'] = da.vertex.where(da.vertex==2.)
    #da = da.where(da.vertex==2., drop=True)
    #print (np.unique(da.vertex))
    #v0 = v0.swap_dims({'ctd_data_point':'transects'})
    #for (_, v) in da.groupby('vertex'):
    #    plt.plot(v.lon, v.lat)
    #plt.show()
    return da

