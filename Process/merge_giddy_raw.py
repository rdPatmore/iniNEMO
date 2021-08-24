import config
import xarray as xr
import glidertools as gt

def show_available_variables():

    path = config.root()
    filenames = path + 'Giddy_2020/Raw/sg643/p6430121.nc'#p643*.nc'
    a = gt.load.seaglider_show_variables(filenames)
    print (a.to_html())

def load_vars():
    path = config.root()
    filenames = path + 'Giddy_2020/Raw/sg643/p643*.nc'#p643*.nc'
    names = [
        'ctd_depth',
        'ctd_time',
        'ctd_pressure',
    ]
    ds_dict = gt.load.seaglider_basestation_netCDFs(
        filenames, names,
        return_merged=False,
        keep_global_attrs=False
    )
    ds_dict['ctd_data_point'].to_netcdf(path + 'Giddy_2020/merged_raw.nc')
    print (ds_dict)
    

load_vars()
def merge_giddy_raw():

    #def set_coords(ds):
    #    dims = ['aa4831_data_point',
    #            'auxCompass_data_point',
    #            'ctd_data_point',
    #            'gc_event',
    #            'gc_state',
    #            'gps_info',
    #            'qsp2150_data_point',
    #            'sbect_data_point',
    #            'scicon_wlbb2fl_wlbb2fl_data_point',
    #            'trajectory',
    #            'wlbb2fl_data_point']

    #    ds = ds.drop_dims(dims, errors='ignore')
    #    ds['index'] = ds.
    #    try:
    #        ds = ds.set_coords('time')
    #    except:
    #        print ('no')
    #        print ('no')
    #        print ('no')
    #        print ('no')
    #        print (ds)
    #    return ds
    dims = ['aa4831_data_point',
            'auxCompass_data_point',
            'sg_data_point',
            'depth_data_point',
            'gc_event',
            'gc_state',
            'gps_info',
            'qsp2150_data_point',
            'sbect_data_point',
            'scicon_wlbb2fl_wlbb2fl_data_point',
            'trajectory',
            'wlbb2fl_data_point']

    # load data
    path = config.root()
   
    ds = xr.open_dataset(path + 'Giddy_2020/Raw/sg643/p6430001.nc')
    ds = ds.drop_dims(dims, errors='ignore')
    ds['index'] = xr.DataArray(range(ds.sizes['ctd_data_point']),
                               dims='ctd_data_point')
    ds = ds.swap_dims({'ctd_data_point': 'index'})
    ds['dives'] = 1
    #print (ds)

    station_len = 503
    for i in range(2, station_len):
        print (i)
        try:
            data = xr.open_dataset(path + 'Giddy_2020/Raw/sg643/p6430' +
                                   str(i).zfill(3) + '.nc')
            data = ds
            data = data.drop_dims(dims, errors='ignore')
            inds = range(ds.index[-1].values + 1,
                         ds.index[-1].values +1 + data.sizes['ctd_data_point'])
            data['index'] = xr.DataArray(inds, dims='ctd_data_point')
            data['dives'] = i
            #print (data)
            #data = data.set_dims('index')
            #data = data.swap_dims({'ctd_data_point': 'index'})
            data = data.swap_dims({'ctd_data_point': 'index'})
            print (ds.index)
            print (data.index)
            ds = xr.concat([ds,data], dim='ctd_data_point')
        except:
            print ('missing file')
      

    # get time and depth
    ds.to_netcdf(path + 'Giddy_2020/merged_raw.nc')

#merge_giddy_raw()
