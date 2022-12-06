import xarray as xr
import numpy  as np
import itertools
import matplotlib.pyplot as plt
import config

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

def get_sampled_path(model, append, post_transect=True, rotation=None,
                     cut_meso=True):
    ''' load single gilder path sampled from model '''
    path = config.data_path() + model + '/'
    file_path = path + 'GliderRandomSampling/glider_uniform_' + \
                append +  '_00.nc'
    glider = xr.open_dataset(file_path).sel(ctd_depth=10, method='nearest')
    glider['lon_offset'] = glider.attrs['lon_offset']
    glider['lat_offset'] = glider.attrs['lat_offset']
    glider = glider.set_coords(['lon_offset','lat_offset','time_counter'])
    if post_transect:
        glider = get_transects(glider.votemper, offset=True,
                               rotation=rotation, cut_meso=cut_meso)
    #glider = rotate(glider, np.radians(-90))
    return glider

def remove_meso(da, rotation=None):
        ''' removes the mesoscale north-south transects '''

        # some paths are saved rotated
        if rotation: # shift back to unrotated
            da = rotate_path(da, -rotation)

        da = da.where(da.transect>1, drop=True)
        #da = da.reset_coords('transect')
        # this should work but there is a bug in xarray perhaps
        # idxmin drops all coordinates...
        # this works when transect is a coordinate rather than a variable
        # it also cannot deal with chunks... 
        if 'ctd_depth' in da.dims:
            idxmin = da.lat.idxmin(skipna=True, dim='distance').transect
        else:
            idxmin = da.lat.idxmin(skipna=True).transect
        da = da.where(da.transect != idxmin, drop=True)
        # transect_south = da.isel(distance=da.lat.argmin(skipna=True).values)
        # da = da.where(da.transect != transect_south, drop=True)

        # re-rotate
        if rotation:
            da = rotate_path(da, rotation)

        return da

def get_transects(da, concat_dim='distance', method='cycle',
                  shrink=None, drop_trans=[False,False,False,False],
                  offset=False, rotation=None, cut_meso=False):
    '''
    da is xr DataArray only

    drop_tran: drop sides
               SW -> NE
               E
               SE -> NW
               W
    cut meso: remove long north-south transects
    '''
    print ('i')

    skip=False # skip transect finding due to copy from existing array
    # some paths are saved rotated
    if rotation: # shift back to unrotated
        da = rotate_path(da, -rotation)

    if method == '2nd grad':
        # finds the maxima in the second order gradients 

        a = np.abs(np.diff(da.lat, 
        append=da.lon.max(), prepend=da.lon.min(), n=2))# < 0.001))[0]
        idx = np.where(a>0.006)[0]

    if method == 'cycle':
        # loops through the data and interates transect when lat-lon
        # thresholds are met
 
        crit = [0,1,2,3]
        #da = da.isel(distance=slice(0,400))

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
            print (test)
            if test.any(): 
                start = False
                idx.append(i)
                a = next(crit_iter)

    if method == 'find e-w':
        # find local maximums in east west travel
        # a = np.abs(np.diff(da.lon.isel(distance=slice(None,None,10)), n=2))
        # works ok but not uniform accross the data reductions (every 2 etc.)

        def differ(arr):
            print (arr.lon)
            a = np.abs(arr.lon.isel(max_diff=-1) - arr.lon.isel(max_diff=0))
            a.name = 'max_dist'
            print (a)
            a['lon'] = arr.isel(max_diff=2).lon.values
            a['lat'] = arr.isel(max_diff=2).lat.values
            return a
            
        #rolled.name = 'votemper'
        arr = da.reset_coords('lon')
        rolled = arr.rolling(distance=5, center=True).construct('max_diff')
        max_dist = rolled.groupby('distance').map(differ)
        max_dist = np.abs(np.diff(max_dist,
                    append=max_dist.lon.max(), n=1))
        #a = np.abs(np.diff(da.lon,
        #                   append=da.lon.max(), prepend=da.lon.min(), n=2))
        #da['max_dist'] = xr.DataArray(max_dist, dims=('distance'))
        #max_dist = np.where(max_dist>0.01, max_dist, np.nan)[0]
        #da = da.where(da.max_dist>0.01, drop=True)
        #print (da)
        #p = plt.scatter(da.lon, da.lat, c=da.max_dist, alpha=0.2)
        #plt.colorbar(p)
        #plt.show()
        #idx = da.max_dist.values
        idx = np.where(max_dist>0.009)[0]

    if method == 'from interp_1000':
        # get transects from interp_1000_00
        # this data has the mesoscale transects included

        path = config.data_path() + 'EXP10/'
        file_path = path + 'GliderRandomSampling/' + \
                    'glider_uniform_interp_1000_00.nc'
        glider = xr.open_dataset(file_path).sel(ctd_depth=10, method='nearest')
        
        glider['lon_offset'] = glider.attrs['lon_offset']
        glider['lat_offset'] = glider.attrs['lat_offset']
        glider = glider.set_coords(['lon_offset','lat_offset','time_counter'])
        glider = get_transects(glider.votemper, offset=True, method='cycle',
                               cut_meso=False)
        
        # assign high res transects to low res path
        trans = glider.swap_dims({'distance':'ctd_data_point'}
                                    ).dropna('ctd_data_point')
        trans = trans.sel(ctd_data_point=da.ctd_data_point.values,
                         method='nearest').transect.astype('int').values

        dim = list(da.dims)[0] # target dim
        da = da.assign_coords({'transect': xr.DataArray(trans, dims=(dim))})

        skip=True

    if not skip:
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

    # remove initial and mid-path mesoscale excursions
    # defunct due to next naming lines
    if cut_meso:
        da = remove_meso(da)

    else:
        # name mesoscale transects
        # 1 for bow tie, 0 for n-s transects
        meso = xr.where(da.transect>1, 1, 0) # get 1st transect
        lat_nan_t0 = da.lat.where(meso) # temp arr with 1st path removed
        # get second 
        if 'ctd_depth' in da.dims:
            idxmin = lat_nan_t0.idxmin(skipna=True, dim='distance').transect
        else:
            idxmin = lat_nan_t0.idxmin(skipna=True).transect

        meso = xr.where(da.transect != idxmin, meso, 0)
        da = da.assign_coords({'meso_transect': meso})

    # re-rotate
    if rotation:
        da = rotate_path(da, rotation)

    # catagorise
    category = da.transect % 4
    da = da.assign_coords({'vertex': da.transect % 4})

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

