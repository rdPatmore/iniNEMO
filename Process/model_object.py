import xarray as xr
import config
import numpy as np
import gsw
import dask
import matplotlib.pyplot as plt
import glidertools as gt

#dask.config.set(scheduler='single-threaded')

class model(object):
    ''' get model object and process '''
 
    def __init__(self, case):
        self.case = case
        self.root = config.root()
        self.path = config.data_path()
        self.data_path = config.data_path() + self.case + '/'
        def drop_coords(ds):
            for var in ['e3t','e3u','e3v']:
                try:
                    ds = ds.drop(var)
                except:
                    print ('no win', var)
            return ds.reset_coords(drop=True)
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            self.ds = {}
            #self.grid_keys = ['icemod']
            self.grid_keys = ['grid_T', 'grid_U', 'grid_V', 'grid_W', 'icemod']
            file_names = ['/SOCHIC_PATCH_3h_20120101_20121231_',
                          '/SOCHIC_PATCH_3h_20130101_20140101_']
            for pos in self.grid_keys:
            #    self.ds[pos] = xr.open_mfdataset(self.data_path +
            #                '/SOCHIC_PATCH_3h*_' + pos + '.nc',
            #              chunks={'time_counter':100}, decode_cf=False)
            #                #compat='different', coords='all',
                data_set = []
                for file_name in file_names:
                    data = xr.open_dataset(self.data_path +
                                file_name + pos + '.nc',
                              chunks={'time_counter':10}, decode_cf=False)
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

        # load obs
        self.giddy     = xr.open_dataset(self.root + 
                         'Giddy_2020/sg643_grid_density_surfaces.nc')
        self.giddy_raw = xr.open_dataset(self.root + 
                         'Giddy_2020/merged_raw.nc')
        self.giddy_raw = self.giddy_raw.rename({'longitude': 'lon',
                                                'latitude': 'lat'})


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
        ''' calculate pressure from sea surface height '''
        self.ds['p'] = gsw.p_from_z(self.ds.zos, self.ds.nav_lat)
 
        if save:
            self.ds.p.to_netcdf(config.data_path() + self.case + '/p.nc')

    def get_conservative_temperature(self, save=False):
        ''' calulate conservative temperature '''
        self.ds['cons_temp'] = gsw.conversions.CT_from_pt(self.ds.vosaline,
                                                          self.ds.votemper)
        if save:
            self.ds.cons_temp.to_netcdf(config.data_path() + self.case + 
                                    '/conservative_temperature.nc')

    def get_absolute_salinity(self, save=False):
        ''' calulate absolute_salinity '''
        self.get_pressure()
        self.ds['abs_sal'] = gsw.conversions.SA_from_SP(self.ds.vosaline, 
                                                        self.ds.p,
                                                        self.ds.nav_lon,
                                                        self.ds.nav_lat)
        if save:
            self.ds.abs_sal.to_netcdf(config.data_path() + self.case + 
                                  '/absolute_salinity.nc')

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
    



        
    def get_nemo_glider_time(self, start_month='01'):
        ''' take a time sample based on time difference in glider sample '''
        time = self.giddy.time.isel(density=50)
        time_diff = time.diff('distance').pad(distance=(0,1)).fillna(0).cumsum()
        start_date = np.datetime64('2012-' + start_month + '-01 00:00:00')
        time_span = start_date + time_diff
        return time_span
        #self.ds = self.ds.interp(time_counter=time_span.values, method='nearest')
        
    def random_glider_lat_lon_shift(self, grid='grid_T'):

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
  
        # fractional distance to bound
        lon_frac = (glon0 - nlon0) / lon_dist
        lat_frac = (glat0 - nlat0) / lat_dist

        # lat lon space to left and bottom
        left_space = glon0 - nlon0
        bottom_space = glat1 - nlat1

        print (' ')
        print (' ')
        print (' ')
        print (' ')
        print (' ')
        print (lon_dist)
        print (lat_dist)
        print (left_space)
        print (bottom_space)

        # get random shifts within nemo bounds
        self.lon_shift = (- left_space + (lon_dist * np.random.random())) * 0.8
        self.lat_shift = (- bottom_space + (lat_dist * np.random.random())) *0.8
    
        print (self.lon_shift)
        print (self.lat_shift)
        print (' ')
        print (' ')
        print (' ')
        print (' ')
        print (' ')

    def prep_interp_to_raw_obs(self):
        '''
        preliminary processing for sampling model like a glider
        '''
        
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

        # drop surface variables
        self.ds['grid_T'] = self.ds['grid_T'].drop(['tos', 'sos', 'zos',
                            'mldr10_3', 'wfo', 'qsr_oce', 'qns_oce',
                            'qt_oce', 'sfx', 'taum', 'windsp',
                            'precip', 'snowpre', 'bounds_nav_lon',
                            'bounds_nav_lat', 'deptht_bounds',
                             'area', 'e3t'])

        # get glider lat-lons
        self.x = xr.DataArray(self.giddy_raw.lon.values,
                              dims='ctd_data_point')
        self.y = xr.DataArray(self.giddy_raw.lat.values,
                              dims='ctd_data_point')

        # add lat-lat lon to grid_T dimentions
        self.x_y_to_lat_lon('grid_T')
        print (self.ds['grid_T'])
        


    def interp_to_raw_obs_path(self, random_offset=False, save=False, ind=''):
        '''
        sample model along glider's raw path
        using giddy (2020)
        '''


        if random_offset:
            # this shift is centered on the model and may shift
            # glider out of bounds
            self.random_glider_lat_lon_shift()
            self.x = self.x + self.lon_shift
            self.y = self.y + self.lat_shift

      
        print (' ')
        print (self.x.values)
        print (self.y.values)
        print (' ')

        # interpolate
        self.glider_nemo = self.ds['grid_T'].interp(lon=self.x,
                                                    lat=self.y,
                                        deptht=self.giddy_raw.ctd_depth,
                                        time_counter=self.giddy_raw.ctd_time)
        #self.glider_nemo = self.glider_nemo.swap_dims(
        #                                        {'distance':'time_counter'})
        #glider_raw=self.glider_nemo#.isel(ctd_data_point=slice(None,250000))
        print ('')
        print ('')
        print (self.glider_nemo.votemper.max().values)
        print (self.glider_nemo.votemper.min().values)
        print ('')
        print ('')

        self.glider_nemo['dives'] = self.giddy_raw.dives
        if random_offset:
            self.glider_nemo.attrs['lon_offset'] = self.lon_shift.values
            self.glider_nemo.attrs['lat_offset'] = self.lat_shift.values

        # drop obsolete coords
        self.glider_nemo = self.glider_nemo.drop_vars(['nav_lon',
                                                       'nav_lat'])

        if save:
            self.glider_nemo.to_netcdf(self.data_path +
                                     'GliderRandomSampling/glider_raw_nemo_' + 
                                       ind + '.nc')

    def interp_raw_obs_path_to_uniform_grid(self, ind=''):
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
        print (glider_raw)
        #glider_raw=glider_raw.isel(ctd_data_point=slice(None,250000))
        #plt.figure()
        #plt.plot(glider_raw.ctd_data_point, glider_raw.votemper)
        #plt.show()

        # distance between samples
        #glider_raw['distance'] = xr.DataArray(np.r_[0, 
        #             gt.utils.distance(glider_raw.longitude,
        #                               glider_raw.latitude).cumsum()],
        #                               dims='ctd_data_point')
        
        # set distance as dimension
        # make depth a dimension

        #glider_uniform_i = glider_raw.interp(deptht=np.arange(0,999,1))
        #print (glider_uniform_i)
        #print (sdfhjklas)
 
        #reshaped = []
        #for (label, group) in glider_raw.groupby('deptht'):
        #    print (group)
        #    reshaped.append(group.expand_dims('depth'))
        #glider_raw = xr.concat(reshaped, dim='depth')

        #def expand_depth_dim(ds):
        #    return ds.expand_dims('depht')
   
        #glider_raw.groupby('deptht').map(expand_depth_dim)
#
#        print (glider_raw)  
#        print (dskjlh)
        uniform_distance = np.arange(0, glider_raw.distance.max(),1000)
        #print (uniform_distance)

        glider_uniform_i = []
        #glider_raw = glider_raw.isel(ctd_data_point=slice(None,10000))
        #print (glider_raw)
        #glider_raw = glider_raw.dropna('ctd_data_point', how='all')
        # interpolate to 1 m vertical grid
        for (label, group) in glider_raw.groupby('dives'):
            #print (label)
            #try:
            if group.sizes['ctd_data_point'] < 2:
                continue
            group = group.swap_dims({'ctd_data_point': 'ctd_depth'})

            # remove duplicate index values
            _, index = np.unique(group['ctd_depth'], return_index=True)
            group = group.isel(ctd_depth=index)

            depth_uniform = group.interp(ctd_depth=np.arange(0.0,999.0,1))
            #depth_uniform = depth_uniform.reset_coords(
            #['lat','lon','latitude','longitude','ctd_depth'])
            uniform = depth_uniform.expand_dims(dive=[label])
            glider_uniform_i.append(uniform)
            #except:
            #    print ('fail 1')
        glider_uniform = xr.concat(glider_uniform_i, dim='dive')
        print (glider_uniform)

        # interpolate to 1 km horzontal grid
        glider_uniform_i = []
        for (label, group) in glider_uniform.groupby('ctd_depth'):
            #try:
            group = group.swap_dims({'dive': 'distance'})
                
            #group = group.dropna('distance')
            group = group.sortby('distance')
            group = group.dropna('distance', how='all')

            # remove duplicate index values
            _, index = np.unique(group['distance'], return_index=True)
            group = group.isel(distance=index)

            group = group.interpolate_na('distance')
           
            uniform = group.interp(distance=uniform_distance)
            glider_uniform_i.append(uniform)
            #except ValueError:
            #    print ('fail 2')

        glider_uniform = xr.concat(glider_uniform_i, dim='ctd_depth')
        #glider_uniform_i = []
        #for (label, group) in glider_raw.groupby('dives'):
        #    try:
        #        #depth_uniform = depth_uniform.sortby('distance')
        #        #plt.figure()
        #        #plt.plot(depth_uniform.distance)
        #        #plt.show()
        #        depth_uniform = depth_uniform.dropna('distance')
        #        print (depth_uniform.distance.values)
        #        depth_uniform = depth_uniform.interpolate_na('distance')
        #        uniform = uniform.expand_dims(dive=[label])
        #        glider_uniform_i.append(uniform)
        #    except:
        #        print ('fail')
        #print (glider_uniform)
        glider_uniform.to_netcdf(self.data_path + 
                                 'GliderRandomSampling/glider_uniform_'
                                  + str(ind) + '.nc')

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
            ds = self.ds[grid].mean(['x','y']).load()
            for key in ds.keys():
                ds = ds.rename({key: key + '_mean'})
            ds.to_netcdf(self.data_path +
                        'Stats/SOCHIC_PATCH_mean_' + grid + '.nc')

    def save_area_std_all(self):
        ''' save lateral standard deviation of all data '''

        for grid in self.grid_keys:
            print ('std :', grid)
            ds = self.ds[grid].std(['x','y']).load()
            for key in ds.keys():
                ds = ds.rename({key: key + '_std'})
            ds.to_netcdf(self.data_path +
                         'Stats/SOCHIC_PATCH_std_' + grid + '.nc')

    def save_month(self):
        jan = self.ds.sel(time_counter='2012-01')
        jan.time_counter.encoding['dtype'] = np.float64
        comp = dict(zlib=False, complevel=6)
        encoding = {var: comp for var in jan.data_vars}
        jan.to_netcdf(self.data_path + 'SOCHIC_201201_T.nc', encoding=encoding)


m = model('EXP02')
#m.save_area_mean_all()
#m.save_area_std_all()
#m.save_month()
m.prep_interp_to_raw_obs()
for ind in range(10):
    print ('ind: ', ind)
    m.interp_to_raw_obs_path(save=False, random_offset=True)
    print ('done part 1')
    m.interp_raw_obs_path_to_uniform_grid(ind=ind)
    print ('done part 2')

#print ('start')
#m.get_conservative_temperature(save=True)
#print ('part1/3')
#m.get_absolute_salinity(save=True)
#print ('part2/3')
#m.get_alpha_and_beta(save=True)
#print ('part3/3... end')
