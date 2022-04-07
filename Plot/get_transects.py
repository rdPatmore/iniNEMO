import xarray as xr
import numpy  as np
import itertools

def get_transects(data, concat_dim='distance', method='cycle',
                  shrink=None, drop_trans=[False,False,False,False],
                  offset=False):
    '''
    drop_tran: drop sides
               SW -> NE
               E
               SE -> NW
               W
    '''

    if method == '2nd grad':
        a = np.abs(np.diff(data.lat, 
        append=data.lon.max(), prepend=data.lon.min(), n=2))# < 0.001))[0]
        idx = np.where(a>0.006)[0]
    crit = [0,1,2,3]
    if method == 'cycle':
        #data = data.isel(distance=slice(0,400))

        # shift back to origin
        if offset:
            data['orig_lon'] = data.lon - data.lon_offset
            data['orig_lat'] = data.lat - data.lat_offset
        else:
            data['orig_lon'] = data.lon
            data['orig_lat'] = data.lat

        idx=[]
        crit_iter = itertools.cycle(crit)
        start = True
        a = next(crit_iter)
        for i in range(data[concat_dim].size)[::shrink]:
            da = data.isel({concat_dim:i})
            if (a == 0) and (start == True):
                test = ((da.orig_lat < -60.04) and (da.orig_lon > 0.176))
            elif a == 0:
                test = (da.orig_lon > 0.176)
            elif a == 1:
                test = (da.orig_lat > -59.93)
            elif a == 2:
                test = (da.orig_lon < -0.173)
            elif a == 3:
                test = (da.orig_lat > -59.93)
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
 
    # catagorise
    #category = (np.tile([0,1,2,3], 1 + (da.size/4)))[:ds.size]
    category = da.transect%4
    print (np.unique(category))
    da = da.assign_coords({'vertex': da.transect%4})
    print(da)
    import matplotlib.pyplot as plt
    v0 = da.where(da.vertex==0)
    print (v0)
    v0 = v0.swap_dims({'ctd_data_point':'transects'})
    for (_, v) in v0.groupby('transects'):
        plt.plot(v.lon, v.lat)
    plt.show()
    return da

