from dask.distributed import Client, LocalCluster
import xarray as xr
import numpy  as np
import config

class profile_stats(object):

    def __init__(self, case):
        self.root = config.root_old()
        self.case = case
        self.data_path = config.data_path_old() + self.case + '/'

        self.loaded_gp = False 

    def restrict_model_time_and_space(self):
        ''' restrict data to glider time and space '''

        clean_float_time = self.samples.time_counter
        start = clean_float_time.min().astype('datetime64[ns]')
        end   = clean_float_time.max().astype('datetime64[ns]')
        
        east = self.sample.lon.max()
        west = self.sample.lon.min()
        north = self.sample.lat.max()
        south = self.sample.lat.min()

        dmax=self.sample.ctd_depth.max()

        restricted = self.model.sel(time_counter=slice(start,end),
                         lon=slice(west,east), lat=slice(south,north),
                         deptht=slice(None,dmax))
        restricted = restricted.swap_dims({'lon':'x', 'lat':'y'})

        return restricted

    def get_glider_patches(self):
        ''' load set of processed glider emulators '''

        self.loaded_gp = True

        def expand_sample_dim(ds):
            ds = ds.expand_dims('sample')
            return ds
        file_paths = [self.data_path + 'GliderRandomSampling/glider_uniform_' +
                      str(i).zfill(2) + '.nc' for i in range(100)]
        self.samples = xr.open_mfdataset(file_paths,
                                         combine='nested', concat_dim='sample',
                                         preprocess=expand_sample_dim)

        # set time to float for averaging
        float_time = self.samples.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        self.samples['time_counter'] = clean_float_time

        self.samples = self.samples.dropna('distance', how='all')
        self.samples = self.samples.dropna('ctd_depth', how='all')

    def get_model_patches(self):
        
        if not self.loaded_gp:
            self.get_glider_patches()

        print ('a')
        # get_model_data
        t2012 = xr.open_dataset(self.data_path + 
                                'SOCHIC_PATCH_3h_20120101_20121231_grid_T.nc',
                                chunks={'deptht':1})
        t2012 = t2012.drop(['mldkz5','mldr10_1','sbt','tos','sos','zos','mldr10_3',
                                      'wfo','qsr_oce','qns_oce','qt_oce',
                                      'sfx','taum','windsp','precip',
                                      'snowpre','e3t','bounds_nav_lon',
                                      'bounds_nav_lat','area',
                                      'deptht_bounds','time_centered_bounds',
                                      'time_counter_bounds'])
        t2013 = xr.open_dataset(self.data_path + 
                                'SOCHIC_PATCH_3h_20130101_20140101_grid_T.nc',
                                chunks={'deptht':1})
        t2013 = t2013.drop(['tos','sos','zos','mldr10_3',
                                      'wfo','qsr_oce','qns_oce','qt_oce',
                                      'sfx','taum','windsp','precip',
                                      'snowpre','e3t','bounds_nav_lon',
                                      'bounds_nav_lat','area',
                                      'deptht_bounds','time_centered_bounds',
                                      'time_counter_bounds'])
        self.model = xr.concat([t2012,t2013], 'time_counter')

        self.model = self.model.assign_coords(
                               {'lon': self.model.nav_lon.isel(y=0),
                                'lat': self.model.nav_lat.isel(x=0)})
        self.model = self.model.swap_dims({'x':'lon', 'y':'lat'})

        print ('b')
        # size of sample set 
        set_size = self.samples.sizes['sample']
        print ('c')

        mean_model_patches = []
        decile_model_patches = []
        std_model_patches = []
        for i, sample in enumerate(range(set_size)):
            print (i, ' / ', set_size)
            self.sample = self.samples.isel(sample=sample)
            model_patch = self.restrict_model_time_and_space().load()#.chunk(
                                                            #{'time_counter':-1})

            # calculate stats over horizontal plane
            mean = model_patch.mean(['x','y','time_counter'])
            quant = model_patch.quantile([0.1,0.5,0.9],['x','y','time_counter'])
            std = model_patch.std(['x','y','time_counter'])
           
            mean_model_patches.append(mean)
            decile_model_patches.append(quant)
            std_model_patches.append(std)

        self.mean_model_patches = xr.concat(mean_model_patches, 'sets')
        self.decile_model_patches = xr.concat(decile_model_patches, 'sets')
        self.std_model_patches = xr.concat(std_model_patches, 'sets')

    def calc_glider_set(self):

        if not self.loaded_gp:
            self.get_glider_patches()

        # calculate stats over horizontal plane
        mean_mean = self.samples.mean(['distance','sample']) # and samples
        quant = self.samples.quantile([0.1,0.5,0.9],'distance')
        std = self.samples.std('distance')

        # average over sets
        quant_mean = quant.mean('sample') 
        std_mean = std.mean('sample') 

        # save
        mean_mean.to_netcdf(self.data_path + 'Stats/glider_profiles_mean.nc')
        quant_mean.to_netcdf(self.data_path + 'Stats/glider_profiles_decile.nc')
        std_mean.to_netcdf(self.data_path + 'Stats/glider_profiles_std.nc')


    def calc_model_set(self):
        ''' calculate depth profile stats over glider patches '''

        # get patches that align with glider samples
        self.get_model_patches()

        print ('e')
        # average over sets
        mean_mean = self.mean_model_patches.mean('sets')
        print ('f')
        decile_mean = self.decile_model_patches.mean('sets')
        print ('g')
        std_mean = self.std_model_patches.mean('sets')
        print ('h')

        # save
        mean_mean.to_netcdf(self.data_path + 'Stats/model_profiles_mean.nc')
        decile_mean.to_netcdf(self.data_path + 'Stats/model_profiles_decile.nc')
        std_mean.to_netcdf(self.data_path + 'Stats/model_profiles_std.nc')

if __name__ == '__main__':
    # set dask cluster env
    cluster = LocalCluster(n_workers=1)
    client = Client(cluster)

    ps = profile_stats('EXP02')
    #ps.calc_glider_set()
    ps.calc_model_set()
