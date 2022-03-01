import xarray as xr
import config
import numpy as np
import gsw
import dask
import matplotlib.pyplot as plt
import glidertools as gt
from math import radians, cos, sin, asin, sqrt
from dask.distributed import Client, LocalCluster


class model(object):
    ''' get model object and process '''
 
    def __init__(self, case):
        self.case = case
        self.root = config.root()
        self.path = config.data_path()
        self.data_path = config.data_path() + self.case + '/'

        self.loaded_p = False

    def load_all(self):
        def drop_coords(ds):
            for var in ['e3t','e3u','e3v']:
                try:
                    ds = ds.drop(var)
                except:
                    print ('no win', var)
            return ds.reset_coords(drop=True)
        #with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        self.ds = {}
        self.grid_keys = ['icemod']
        self.grid_keys = ['grid_T', 'grid_U', 'grid_V', 'grid_W', 'icemod']
        self.file_names = ['/SOCHIC_PATCH_3h_20120101_20121231_',
                           '/SOCHIC_PATCH_3h_20130101_20140101_']
        #self.file_names = ['/SOCHIC_PATCH_24h_20120101_20121231_']
        #self.file_names = ['/SOCHIC_PATCH_6h_20120101_20120701_',
        #                  '/SOCHIC_PATCH_6h_20120702_20121231_']
        for pos in self.grid_keys:
        #    self.ds[pos] = xr.open_mfdataset(self.data_path +
        #                '/SOCHIC_PATCH_3h*_' + pos + '.nc',
        #              chunks={'time_counter':100}, decode_cf=False)
        #                #compat='different', coords='all',
            data_set = []
            for file_name in self.file_names:
                data = xr.open_dataset(self.data_path +
                            file_name + pos + '.nc',
                          chunks={'time_counter':1},
                          decode_cf=False)
                try:
                    data = data.drop(['sbt','mldr10_1','mldkz5'])
                except:
                    print ('no drop')
                try:
                    data = data.drop(['vocetr_eff'])
                except:
                    print ('no drop')
                try:
                    data = data.drop(['difvho','vovematr','av_wave',
                                      'bflx_iwm','pcmap_iwm','emix_iwm',
                                      'av_ratio'])
                except:
                    print ('no drop')
                data_set.append(data) 
            self.ds[pos] = xr.concat(data_set, dim='time_counter',
                                               data_vars='minimal',
                                               coords='minimal')
            self.ds[pos] = xr.decode_cf(self.ds[pos])
            self.ds[pos] = self.ds[pos].isel(x=slice(1,-1), y=slice(1,-1))
        #self.ds = self.ds.drop_vars('time_instant')
        #self.ds = xr.open_mfdataset(self.data_path + '/SOCHIC_201201_T.nc',
        #                            #compat='override',coords='minimal',
        #                            chunks={'time_counter':10})#, 'x':10,'y':10})
        #self.data = xr.open_dataset(self.data_path +
        #            self.file_names[0] + 'grid_T.nc',
        #          chunks={'time_counter':1},
        #          decode_cf=False)

        # load obs
        self.giddy     = xr.open_dataset(self.root + 
                         'Giddy_2020/sg643_grid_density_surfaces.nc')
        self.giddy_raw = xr.open_dataset(self.root + 
                         'Giddy_2020/merged_raw.nc')
        self.giddy_raw = self.giddy_raw.rename({'longitude': 'lon',
                                                'latitude': 'lat'})
        index = np.arange(self.giddy_raw.ctd_data_point.size)
        self.giddy_raw = self.giddy_raw.assign_coords(ctd_data_point=index)

    def load_gridT_and_giddy(self):
        ''' minimal loading for glider sampling of model '''

        # grid T
        self.ds = {}
        self.file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
        path = self.data_path + self.file_id + 'grid_T.nc'
        self.ds['grid_T'] = xr.open_dataset(path, chunks={'time_counter':10})
        self.ds['grid_T'] = self.ds['grid_T'].isel(x=slice(1,-1), y=slice(1,-1))

        # drop variables
        self.ds['grid_T'] = self.ds['grid_T'].drop(['tos', 'sos', 'zos',
                            'wfo', 'qsr_oce', 'qns_oce',
                            'qt_oce', 'sfx', 'taum', 'windsp',
                            'precip', 'snowpre', 'bounds_nav_lon',
                            'bounds_nav_lat', 'deptht_bounds',
                            'area', 'e3t','time_centered_bounds',
                            'time_counter_bounds', 'time_centered',
                            'mldr10_3', 'time_instant',
                            'time_instant_bounds'])#.isel(x=slice(0,50),
                                                   #      y=slice(0,50)).load()
        print (self.ds['grid_T'])

        # glider
        self.giddy_raw = xr.open_dataset(self.root + 
                         'Giddy_2020/merged_raw.nc')
        self.giddy_raw = self.giddy_raw.rename({'longitude': 'lon',
                                                'latitude': 'lat'})
        index = np.arange(self.giddy_raw.ctd_data_point.size)
        self.giddy_raw = self.giddy_raw.assign_coords(ctd_data_point=index)

    def merge_state(self):
        ''' merge all state variables into one file '''

        grid_T = xr.open_dataset(self.data_path + 
                                 'SOCHIC_PATCH_3h_20120101_20121231_grid_T.nc',
                                    chunks={'time_counter':1})
        alpha = xr.open_dataarray(self.data_path + 'alpha.nc',
                                    chunks={'time_counter':1})
        beta = xr.open_dataarray(self.data_path + 'beta.nc',
                                    chunks={'time_counter':1})
        absolute_salinity = xr.open_dataarray(self.data_path + 
                                    'absolute_salinity.nc',
                                    chunks={'time_counter':1})
        conservative_temperature = xr.open_dataarray(self.data_path + 
                                        'conservative_temperature.nc',
                                        chunks={'time_counter':1})
        self.ds = xr.merge([alpha, beta, absolute_salinity,
                            conservative_temperature,
                            grid_T.mldr10_3])

        # make grid regular
        self.x_y_to_lat_lon()
 
        self.ds.to_netcdf('state.nc')


    def x_y_to_lat_lon(self, grid='grid_T'):
        ''' change x y dimentions to lat lon '''

        self.ds[grid] = self.ds[grid].assign_coords(
                                       {'lon': self.ds[grid].nav_lon.isel(y=0),
                                        'lat': self.ds[grid].nav_lat.isel(x=0)})

        self.ds[grid] = self.ds[grid].swap_dims({'x':'lon', 'y':'lat'})

    def get_pressure(self, save=False):
        ''' calculate pressure from depth '''
        if self.loaded_p:
            print ('p already loaded') 
        else:
            self.loaded_p = True
            data = self.ds['grid_T']
            self.p = gsw.p_from_z(-data.deptht, data.nav_lat)
            self.p.name = 'p'
            if save:
                self.p.to_netcdf(self.data_path + self.file_id + 'p.nc')

    def get_conservative_temperature(self, save=False):
        ''' calulate conservative temperature '''
        data = self.ds['grid_T'].chunk({'time_counter':1})
        #self.cons_temp = gsw.conversions.CT_from_pt(data.vosaline,
        #                                            data.votemper)
        self.cons_temp = xr.apply_ufunc(gsw.conversions.CT_from_pt,
                                        data.vosaline, data.votemper,
                                        dask='parallelized',
                                        output_dtypes=[data.vosaline.dtype])
        #self.cons_temp.compute()
        self.cons_temp.name = 'cons_temp'
        if save:
            self.cons_temp.to_netcdf(self.data_path + self.file_id
                                   + 'conservative_temperature.nc',)

    def get_absolute_salinity(self, save=False):
        ''' calulate absolute_salinity '''
        self.get_pressure()
        data = self.ds['grid_T'].chunk({'time_counter':1})
        #self.abs_sal = gsw.conversions.SA_from_SP(data.vosaline, 
        #                                          self.p,
        #                                          data.nav_lon,
        #                                          data.nav_lat)
        self.abs_sal = xr.apply_ufunc(gsw.conversions.SA_from_SP,data.vosaline, 
                                      self.p, data.nav_lon, data.nav_lat,
                                      dask='parallelized', 
                                      output_dtypes=[data.vosaline.dtype])
        #self.abs_sal.compute()
        self.abs_sal.name = 'abs_sal'
        if save:
            self.abs_sal.to_netcdf(self.data_path + self.file_id 
                                + 'absolute_salinity.nc')


    def get_alpha_and_beta(self, save=False):
        ''' calculate the themo-haline contaction coefficients '''
        #self.open_ct_as_p()
        self.ds['alpha'] = gsw.density.alpha(self.ds.abs_sal, self.ds.cons_temp,
                                             self.ds.p)
        self.ds['beta'] = gsw.density.beta(self.ds.abs_sal, self.ds.cons_temp,
                                           self.ds.p)

        if save:
            self.ds.alpha.to_netcdf(config.data_path() + 'alpha.nc')
            self.ds.beta.to_netcdf(config.data_path() + 'beta.nc')

    def get_rho(self):
        '''
        calculate buoyancy from conservative temperature and
        absolute salinity    
        '''
        
        # load temp, sal, alpha, beta
        gsw_file = xr.open_dataset(self.data_path + self.file_id +  'gsw.nc',
                              chunks={'time_counter':1})
        ct = gsw_file.cons_temp
        a_sal = gsw_file.abs_sal

        rho = xr.apply_ufunc(gsw.density.sigma0, a_sal, ct,
                             dask='parallelized', output_dtypes=[a_sal.dtype]
                             ) + 1000

        # save
        rho.name = 'rho'
        rho.to_netcdf(self.data_path + self.file_id + 'rho.nc')

    def save_all_gsw(self):
        ''' save p, conservative temperature and absolute salinity to netcdf '''

        self.get_pressure()
        self.get_conservative_temperature()
        self.get_absolute_salinity()
        gsw = xr.merge([self.p, self.cons_temp, self.abs_sal])
        gsw.to_netcdf(self.data_path + self.file_id + 'gsw.nc')
        
    def get_nemo_glider_time(self, start_month='01'):
        ''' take a time sample based on time difference in glider sample '''
        time = self.giddy.time.isel(density=50)
        time_diff = time.diff('distance').pad(distance=(0,1)).fillna(0).cumsum()
        start_date = np.datetime64('2012-' + start_month + '-01 00:00:00')
        time_span = start_date + time_diff
        return time_span
    #self.ds = self.ds.interp(time_counter=time_span.values, method='nearest')
        
    def random_glider_lat_lon_shift(self, grid='grid_T', load=True):

        if load:
            # load shifted data
            data = xr.open_dataset(self.data_path + 
                             'GliderRandomSampling/glider_uniform_interp_1000_'
                                 + str(self.ind).zfill(2) + '.nc')
            # get random shifts within nemo bounds
            self.lon_shift = data.attrs['lon_offset']
            self.lat_shift = data.attrs['lat_offset']

        else:
            # nemo limits
            nlon0 = self.ds[grid].lon.min()
            nlon1 = self.ds[grid].lon.max()
            nlat0 = self.ds[grid].lat.min()
            nlat1 = self.ds[grid].lat.max()

            # glider limits
            glon0 = self.giddy_raw.lon.min()
            glon1 = self.giddy_raw.lon.max()
            glat0 = self.giddy_raw.lat.min()
            glat1 = self.giddy_raw.lat.max()

            # max shift distances
            lon_dist = glon0 - nlon0 + nlon1 - glon1
            lat_dist = glat1 - nlat1 + nlat0 - glat0
  
            # lat lon space to left and bottom
            left_space = glon0 - nlon0
            bottom_space = glat1 - nlat1


            # get random shifts within nemo bounds
            self.lon_shift = (- left_space   + 
                             (lon_dist * np.random.random())).values
            self.lat_shift = (- bottom_space +
                             (lat_dist * np.random.random())).values
    
    def resample_original_raw_glider_path(self, sample_dist):
        self.giddy_raw['distance'] = xr.DataArray( 
                 gt.utils.distance(self.giddy_raw.lon,
                                   self.giddy_raw.lat).cumsum(),
                                   dims='ctd_data_point')
        self.giddy_raw = self.giddy_raw.set_coords('distance')
        self.giddy_raw = self.giddy_raw.swap_dims(
                                                 {'ctd_data_point': 'distance'})

        # remove duplicate index values
        _, index = np.unique(self.giddy_raw['distance'], return_index=True)
        self.giddy_raw = self.giddy_raw.isel(distance=index)

        distance_interp = np.arange(0,self.giddy_raw.distance.max(),
                                    sample_dist)
        # change time units for interpolation 
        timedelta = self.giddy_raw.ctd_time-np.datetime64('1970-01-01 00:00:00')
        self.giddy_raw['ctd_time'] = timedelta.astype(np.int64)

        self.giddy_raw = self.giddy_raw.interp(distance=distance_interp)

        # convert time units back to datetime64
        self.giddy_raw['ctd_time'] = self.giddy_raw.ctd_time / 1e9 
        unit = "seconds since 1970-01-01 00:00:00"
        self.giddy_raw.ctd_time.attrs['units'] = unit
        self.giddy_raw = xr.decode_cf(self.giddy_raw)
        self.giddy_raw = self.giddy_raw.swap_dims({'distance':'ctd_data_point'})
        self.giddy_raw = self.giddy_raw.drop('distance')

    def mould_glider_path_to_shape(self, sample_dist):
        '''
        take distance in glider path reshape the path along distance
        preserving dives and depth
        '''

        # first shape is a square
        length_dist = 4000 # meters
         
        self.giddy_raw['distance'] = xr.DataArray( 
                 gt.utils.distance(self.giddy_raw.lon,
                                   self.giddy_raw.lat).cumsum(),
                                   dims='ctd_data_point')
        self.giddy_raw = self.giddy_raw.set_coords('distance')
        self.giddy_raw = self.giddy_raw.swap_dims(
                                                 {'ctd_data_point': 'distance'})

        # remove duplicate index values
        _, index = np.unique(self.giddy_raw['distance'], return_index=True)
        self.giddy_raw = self.giddy_raw.isel(distance=index)
  
        # iterate over sides
        for i in int(range(giddy_raw.distance.max()/lenth_dist)):
            side = self.giddy_raw.sel(distance=slice(
                         i * length_dist, (i + 1) * length_dist))
        #    if ns:
        #         ds = 
             

    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return c * r
        

        distance_interp = np.arange(0,self.giddy_raw.distance.max(),
                                    sample_dist)
        # change time units for interpolation 
        timedelta = self.giddy_raw.ctd_time-np.datetime64('1970-01-01 00:00:00')
        self.giddy_raw['ctd_time'] = timedelta.astype(np.int64)

        self.giddy_raw = self.giddy_raw.interp(distance=distance_interp)

        # convert time units back to datetime64
        self.giddy_raw['ctd_time'] = self.giddy_raw.ctd_time / 1e9 
        unit = "seconds since 1970-01-01 00:00:00"
        self.giddy_raw.ctd_time.attrs['units'] = unit
        self.giddy_raw = xr.decode_cf(self.giddy_raw)
        self.giddy_raw = self.giddy_raw.swap_dims({'distance':'ctd_data_point'})
        self.giddy_raw = self.giddy_raw.drop('distance')


    def rotate_original_raw_glider_path(self, theta):

        # translation lengths
        xt = self.giddy_raw.lon.median()
        yt = self.giddy_raw.lat.median()
 
        # translate to origin
        lon_orig = self.giddy_raw.lon - xt
        lat_orig = self.giddy_raw.lat - yt

        # rotate
        lon_rotated =  lon_orig * np.cos(theta) - lat_orig * np.sin(theta)
        lat_rotated =  lon_orig * np.sin(theta) + lat_orig * np.cos(theta)

        # translate to original position
        self.giddy_raw['lon'] = lon_rotated + xt
        self.giddy_raw['lat'] = lat_rotated + yt

    def prep_interp_to_raw_obs(self, resample_path=False, sample_dist=10,
                                     rotate=False, rotation=np.pi/2):
        '''
        preliminary processing for sampling model like a glider
        '''
        
        rho = xr.open_dataarray(self.data_path + self.file_id + 'rho.nc')
        self.ds['grid_T'] = xr.merge([self.ds['grid_T'], rho])

        # shift glider time to nemo time
        time_delta = (np.datetime64('2018') 
                    - np.datetime64('2012')).astype('timedelta64[ns]')
        self.giddy_raw['ctd_time'] = self.giddy_raw['ctd_time'] - time_delta

        # get glider data limits
        time0 = self.giddy_raw.ctd_time.min(skipna=True) - np.timedelta64(1,'D')
        time1 = self.giddy_raw.ctd_time.max(skipna=True) + np.timedelta64(1,'D')

        # reduce to glider limits
        self.ds['grid_T'] = self.ds['grid_T'].sel(
                                                deptht=slice(None,1100),
                                                time_counter=slice(time0,time1))


        # alter path to test sampling methods
        if resample_path:
            self.resample_original_raw_glider_path(sample_dist)

        if rotate:
            self.rotate_original_raw_glider_path(rotation)

        
        ## get glider lat-lons
        #self.glider_lon = xr.DataArray(self.giddy_raw.lon.values,
        #                      dims='ctd_data_point')
        #self.glider_lat = xr.DataArray(self.giddy_raw.lat.values,
        #                      dims='ctd_data_point')

        # add lat-lat lon to grid_T dimentions
        self.x_y_to_lat_lon('grid_T')

    def prep_remove_dives(self, remove='climb'):
        ''' remove dives and/or climbs before sampling and processing '''
        if remove == 'climb':
            token = 0.0
            remove_index = self.giddy_raw.dives % 1
        if remove == 'dive':
            token = 0.5
            remove_index = self.giddy_raw.dives % 1
        if remove == 'every_2':
            # remove every other dive pair
            token = 1.0 
            remove_index = np.floor(self.giddy_raw.dives) % 2
        if remove == 'every_8':
            token = 1.0 
            remove_index = np.floor(self.giddy_raw.dives) % 8
        if remove == 'every_8_and_climb':
            # remove all dives and sample every 8
            token = 0.0 
            remove_index = self.giddy_raw.dives % 8
        if remove == 'every_8_and_dive':
            # remove all climbs and sample every 8
            token = 0.5 
            remove_index = self.giddy_raw.dives % 8
        if remove == 'burst_3_20':
            # sample 3 on 20 off
            token = 0.0 
            remove_index = self.giddy_raw.dives % 24
            remove_index = xr.where(remove_index < 3, 0.0, remove_index)

        self.giddy_raw = self.giddy_raw.assign_coords(
                         {'remove_index': remove_index})
        #self.giddy_raw = self.giddy_raw.set_index(
        #         ctd_data_point=['ctd_data_point','dive_direction','ctd_time'])
        self.giddy_raw = self.giddy_raw.swap_dims(
                                            {'ctd_data_point':'remove_index'})
        self.giddy_raw = self.giddy_raw.sel(remove_index=token)
        self.giddy_raw = self.giddy_raw.swap_dims(
                                            {'remove_index':'ctd_data_point'})
        self.giddy_raw = self.giddy_raw.drop('remove_index')

    def interp_to_raw_obs_path(self, random_offset=False, save=False, ind='',
                               append='', load_offset=False):
        '''
        sample model along glider's raw path
        using giddy (2020)
        '''
 
        # get glider lat-lons
        self.glider_lon = xr.DataArray(self.giddy_raw.lon.values,
                              dims='ctd_data_point')
        self.glider_lat = xr.DataArray(self.giddy_raw.lat.values,
                              dims='ctd_data_point')

        # alias lat-lon for interpolation
        self.x = self.glider_lon
        self.y = self.glider_lat

        if random_offset:
            # this shift is centered on the model and may shift
            # glider out of bounds
            self.random_glider_lat_lon_shift(load=load_offset)
            self.x = self.x + self.lon_shift
            self.y = self.y + self.lat_shift


        # lon lat gets overriden if these remain
        self.giddy_raw_no_ll = self.giddy_raw.drop(['lon','lat'])

        # reduce computational load - local reduction of time and space
        xmin, xmax = self.x.min(), self.x.max()
        ymin, ymax = self.y.min(), self.y.max()
        tmin = self.giddy_raw.ctd_time.min()
        tmax = self.giddy_raw.ctd_time.max()
        dmax = self.giddy_raw.ctd_depth.max()
        grid_T_redu = self.ds['grid_T'].sel(lon=slice(xmin,xmax),
                                            lat=slice(ymin,xmax),
                                            time_counter=slice(tmin,tmax),
                                            deptht=slice(None,dmax))
        # interpolate
        grid_T_redu = grid_T_redu.chunk({'time_counter':-1})
        self.glider_nemo = grid_T_redu.interp(lon=self.x, lat=self.y,
                                     deptht=self.giddy_raw_no_ll.ctd_depth,
                                     time_counter=self.giddy_raw_no_ll.ctd_time)

        self.glider_nemo['dives'] = self.giddy_raw_no_ll.dives
        if random_offset:
            self.glider_nemo.attrs['lon_offset'] = self.lon_shift
            self.glider_nemo.attrs['lat_offset'] = self.lat_shift

        # drop obsolete coords
        self.glider_nemo = self.glider_nemo.drop_vars(['nav_lon',
                                                       'nav_lat'])

        if save:
            self.glider_nemo.to_netcdf(self.data_path +
                                     'GliderRandomSampling/glider_raw_nemo_' + 
                                      append + '_' + ind + '.nc')

    def interp_raw_obs_path_to_uniform_grid(self, ind='', append=''):
        '''
           interpolate glider path sampled model data to 
           1 m vertical and 1 km horizontal grids
           following giddy (2020)
        '''

        #glider_raw = xr.open_dataset(self.data_path +
        #                             'GliderRandomSampling/glider_raw_nemo_' + 
        #                             ind + '.nc')
                                    #chunks={'ctd_data_point': 1000})
        glider_raw = self.glider_nemo.load()
        glider_raw['distance'] = xr.DataArray( 
                 gt.utils.distance(glider_raw.lon,
                                   glider_raw.lat).cumsum(),
                                   dims='ctd_data_point')
        glider_raw = glider_raw.set_coords('distance')

        # make time a variable so it doesn't dissapear on interp
        glider_raw = glider_raw.reset_coords('time_counter')
        #glider_raw = glider_raw.isel(ctd_data_point=slice(0,100))

        # change time units for interpolation 
        timedelta = glider_raw.time_counter-np.datetime64('1970-01-01 00:00:00')
        glider_raw['time_counter'] = timedelta.astype(np.int64)

        uniform_distance = np.arange(0, glider_raw.distance.max(),
                                     self.interp_dist)

        glider_uniform_i = []
        # interpolate to 1 m vertical grid
        for (label, group) in glider_raw.groupby('dives'):
            if group.sizes['ctd_data_point'] < 2:
                continue
            group = group.swap_dims({'ctd_data_point': 'ctd_depth'})
 
            group = group.drop('dives')

            # remove duplicate index values
            _, index = np.unique(group['ctd_depth'], return_index=True)
            group = group.isel(ctd_depth=index)
        
            # interpolate
            depth_uniform = group.interp(ctd_depth=np.arange(0.0,999.0,1))

            uniform = depth_uniform.expand_dims(dives=[label])
            glider_uniform_i.append(uniform)

        glider_uniform = xr.concat(glider_uniform_i, dim='dives')


        # interpolate to 1 km horzontal grid
        glider_uniform_i = []
        for (label, group) in glider_uniform.groupby('ctd_depth'):
            group = group.swap_dims({'dives': 'distance'})
                
            group = group.sortby('distance')
            group = group.dropna('distance', how='all')

            # remove duplicate index values
            _, index = np.unique(group['distance'], return_index=True)
            group = group.isel(distance=index)

            if group.sizes['distance'] < 2:
                continue

            group = group.interpolate_na('distance')
           
            uniform = group.interp(distance=uniform_distance)
            glider_uniform_i.append(uniform)

        glider_uniform = xr.concat(glider_uniform_i, dim='ctd_depth')

        # convert time units back to datetime64
        glider_uniform['time_counter'] = glider_uniform.time_counter / 1e9 
        unit = "seconds since 1970-01-01 00:00:00"
        glider_uniform.time_counter.attrs['units'] = unit
        glider_uniform = xr.decode_cf(glider_uniform)

        # add mixed layer depth
        glider_uniform = self.get_mld_from_interpolated_glider(glider_uniform)

        # add buoyancy gradient
        glider_uniform = self.buoyancy_gradients_in_mld_from_interp_data(
                              glider_uniform)


        glider_uniform.to_netcdf(self.data_path + 
                                 'GliderRandomSampling/glider_uniform_'
                                  + append + str(ind).zfill(2) + '.nc')

    def get_mld_from_interpolated_glider(self, glider_sample,
                                         ref_depth=10, threshold=0.03):
        '''
        Use giddy method for finding mixed layer depth
        Use on uniformly interpolated glider sampled
        add quanitites to random samples
        '''
         
        mld_set = []
        for i in range(len(glider_sample.rho[1,:])):
            density_i = glider_sample.rho.isel(distance=i)
            density_ref = density_i.sel(ctd_depth=ref_depth, method='nearest')
            dens_diff = np.abs(density_i-density_ref)
            mld_i = glider_sample.deptht.where(dens_diff >= threshold).min()
            mld_set.append(mld_i)
        mld = xr.concat(mld_set, dim='distance')

        glider_sample['mld'] = mld
        return glider_sample

    def buoyancy_gradients_in_mld_from_interp_data(self, glider_sample):
        '''
        add buoyancy gradients to randomly sampled and
        uniformly interpolated model data
        '''
       
        # constants
        g = 9.81
        rho_0 = 1027 
        dx = 1000

        # buoyancy gradient
        b = g * (1 - glider_sample.rho / rho_0)
        b_x = b.diff('distance') / dx

        # buoyancy within mixed layer
        glider_sample['b_x_ml'] = b_x.where( 
                            glider_sample.deptht < glider_sample.mld, drop=True)
        return glider_sample


    def interp_to_obs_path(self, random_offset=False):
        '''
          interpolate grid to glider path 
          initially applied to giddy 2020
        '''
         
        # shift glider time to nemo time
        time = self.giddy.time.isel(density=50)
        time_delta = (np.datetime64('2018') 
                    - np.datetime64('2012')).astype('timedelta64[ns]')
        glider_time = time - time_delta

        # get x - y grid
        self.x = xr.DataArray(self.giddy.lon.isel(density=50).values,
                              dims='distance')
        self.y = xr.DataArray(self.giddy.lat.isel(density=50).values,
                              dims='distance')
     
        if random_offset:
            self.random_glider_lat_lon_shift()
            self.x = self.x + self.lon_shift
            self.y = self.y + self.lat_shift

        # get time bounds
        time_delta = (np.datetime64('2018') 
                    - np.datetime64('2012')).astype('timedelta64[ns]')
        time0 = self.giddy.time.min(skipna=True) - time_delta
        time1 = self.giddy.time.max(skipna=True) - time_delta

        # get space bounds
        lon0 = self.giddy.lon.max()
        lon1 = self.giddy.lon.min()
        lat0 = self.giddy.lat.max()
        lat1 = self.giddy.lat.min()

        # reduce to glider limits
        self.ds['grid_T'] = self.ds['grid_T'].sel(
                                                deptht=slice(0,300),
                                                time_counter=slice(time0,time1))
        self.x_y_to_lat_lon('grid_T')

        # interpolate
        self.glider_nemo = self.ds['grid_T'].interp(lon=self.x, lat=self.y,
                                                    time_counter=glider_time)
        self.glider_nemo = self.glider_nemo.swap_dims(
                                                   {'distance':'time_counter'})

        # drop obsolete coords
        self.glider_nemo = self.glider_nemo.drop_vars(['nav_lon',
                                                       'nav_lat'])

    def save_glider_nemo_state(self):
        ''' save nemo data on glider path '''
        
        #state = xr.merge([self.glider_nemo.votemper,
        #                  self.glider_nemo.vosaline,
        #                  self.glider_nemo.mldr10_3])
        self.glider_nemo.to_netcdf(self.data_path + 'glider_nemo.nc')

    def save_area_mean_all(self):
        ''' save lateral mean of all data '''

        for grid in self.grid_keys:
            print ('mean :', grid)
            ds = self.ds[grid].mean(['x','y'])#.load()
            for key in ds.keys():
                ds = ds.rename({key: key + '_mean'})
            ds = ds.drop('area_mean')
            ds.to_netcdf(self.data_path +
                         'Stats/SOCHIC_PATCH_mean_' + grid + '.nc')

    #def save_area_mean_all_test(self):
    #    ''' save lateral mean of all data '''
#
#        ds = self.data.mean(['x','y']).load()
#        for key in ds.keys():
#            ds = ds.rename({key: key + '_mean'})
#        ds.to_netcdf(self.data_path +
#                     'Stats/SOCHIC_PATCH_mean_grid_T.nc')

    def save_area_std_all(self):
        ''' save lateral standard deviation of all data '''

        for grid in self.grid_keys:
            print ('std :', grid)
            ds = self.ds[grid].std(['x','y']).load()
            for key in ds.keys():
                ds = ds.rename({key: key + '_std'})
            ds = ds.drop('area_std')
            ds.to_netcdf(self.data_path +
                         'Stats/SOCHIC_PATCH_std_' + grid + '.nc')

    def save_month(self):
        jan = self.ds.sel(time_counter='2012-01')
        jan.time_counter.encoding['dtype'] = np.float64
        comp = dict(zlib=False, complevel=6)
        encoding = {var: comp for var in jan.data_vars}
        jan.to_netcdf(self.data_path + 'SOCHIC_201201_T.nc', encoding=encoding)

if __name__ == '__main__':
  
    dask.config.set(scheduler='single-threaded')
    #dask.config.set({'temporary_directory': 'Scratch'})
    #cluster = LocalCluster(n_workers=1)
    #client = Client(cluster)
    #from dask.distributed import Client, progress
    #client = Client(threads_per_worker=1, n_workers=10)

    def get_rho():
        m = model('EXP10')
        m.load_gridT_and_giddy()
        #m.save_all_gsw()
        m.get_rho()

    def glider_sampling(remove=False, append='', interp_dist=1000):
        m = model('EXP10')
        m.interp_dist=interp_dist
        m.load_gridT_and_giddy()
        #m.save_area_mean_all()
        #m.save_area_std_all()
        #m.save_month()
        #m.get_conservative_temperature(save=True)
        #sample_dist=5000
        #m.prep_interp_to_raw_obs(resample_path=True, sample_dist=sample_dist)
        m.prep_interp_to_raw_obs()
        if remove:
            m.prep_remove_dives(remove=append)
        for ind in range(100):
            m.ind = ind
            print ('ind: ', ind)
            m.interp_to_raw_obs_path(random_offset=True, load_offset=True)
            print ('done part 1')
            m.interp_raw_obs_path_to_uniform_grid(ind=ind, append=append + '_')
            print ('done part 2')
        #inds = np.arange(100)
        #m.ds['grid_T'] = m.ds['grid_T'].expand_dims(ind=inds)
        #m.interp_to_raw_obs_path(random_offset=True, load_offset=True)
        #m.interp_raw_obs_path_to_uniform_grid(append=append)
        #print (m.ds['grid_T'])
        ##print (inds)
        ##print (djkfs)
        #futures = client.map(process_all, inds, **dict(m=m))
        #client.gather(futures)
        #xr.apply_ufunc(process_all, inds, dask="parallelized")
        print (' ')
        print ('successfully ended')
        print (' ')
    glider_sampling(remove=True, append='every_8_and_climb', interp_dist=1000)
    glider_sampling(remove=True, append='every_8_and_dive', interp_dist=1000)

    def interp_obs_to_model():
        m.prep_interp_to_raw_obs()
        m.interp_to_raw_obs_path()
        m.interp_raw_obs_path_to_uniform_grid(ind='', append='climbs')
    
    
    #print ('start')
    #m.get_conservative_temperature(save=True)
    #print ('part1/3')
    #m.get_absolute_salinity(save=True)
    #print ('part2/3')
    #m.get_alpha_and_beta(save=True)
    #print ('part3/3... end')
