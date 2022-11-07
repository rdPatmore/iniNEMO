import xarray as xr
import numpy as np
import config

class patch_set(object):
    '''
    calculate over glider patches

    NOTE: some of these calcualtion remain in plot_compare_glider_path

    '''

    def __init__(self, case):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'

        self.hist_range = (0,2e-8)
        #self.sample = xr.open_dataset(self.data_path + 
        #       'GliderRandomSampling/glider_uniform_rotated_path.nc')

    def get_glider_samples(self, rotation=None):
        ''' load processed glider samples '''

        # files definitions
        prep = 'GliderRandomSampling/glider_uniform_interp_1000'
        rotation_label = ''
        if rotation:
            rotation_label = '_rotate_' + str(rotation) 

        # get samples
        self.samples = xr.open_dataarray(self.data_path + prep +
             rotation_label + '_b_x_abs_10m_post_transects.nc', 
             decode_times=False)

    def get_model_buoyancy_gradients_patch_set(self, stats=None, rolling=False):
        ''' 
        restrict the model time to glider time and sample areas
        Note: repition of function in plot_compare_gider_path
        '''

        rolling_str, stats_str = '', ''
        # model
        bg = xr.open_dataset(config.data_path() + self.case +
                             '/SOCHIC_PATCH_3h_20121209_20130331_bg.nc')
                             #chunks={'time_counter':113})
                             #chunks='auto')
                             #chunks={'time_counter':1})
        bg = np.abs(bg.sel(deptht=10, method='nearest')).load()

        # get norm
        bg['bg_norm'] = (bg.bx ** 2 + bg.by ** 2) ** 0.5

        # add lat-lon to dimensions
        bg = bg.assign_coords({'lon':(['x'], bg.nav_lon.isel(y=0)),
                               'lat':(['y'], bg.nav_lat.isel(x=0))})
        bg = bg.swap_dims({'x':'lon','y':'lat'})


        clean_float_time = self.samples.time_counter
        start = clean_float_time.min().astype('datetime64[s]')
        end   = clean_float_time.max().astype('datetime64[s]')

        bg = bg.sel(time_counter=slice(start,end))
        
        patch_set = []
        for (l, sample) in self.samples.groupby('sample'):
            # get limts of sample
            x0 = float(sample.lon.min())
            x1 = float(sample.lon.max())
            y0 = float(sample.lat.min())
            y1 = float(sample.lat.max())
 
            patch = bg.sel(lon=slice(x0,x1),
                           lat=slice(y0,y1)).expand_dims(sample=[l])

            dims = ['lon','lat','time_counter']
            if rolling:
                rolling_str = '_rolling'
                # contstruct allows for mean/std over multiple dims
                #patch = patch.sortby('time_counter')
                patch = patch.resample(time_counter='1H').median()
                patch = patch.rolling(time_counter=168, center=True).construct(
                                                               'weekly_rolling')
                dims = ['lat','lon','weekly_rolling']

            if stats == 'mean':
                patch = patch.mean(dims).load()
                stats_str = '_mean'
            if stats == 'std':
                patch = patch.std(dims).load()
                stats_str = '_std'
            if stats == 'median':
                stats_str = '_median'
                patch = patch.median(dims).load()
                patch = patch.std(dims).load()

            if stats == 'time_mean_space_quantile':
                patch_mean = patch.rolling(time_counter=168).mean()
                patch_std = patch.rolling(time_counter=168).std()
                for k in list(patch_mean.keys()):
                    patch_mean = patch_mean.rename({k:k + '_rolling_mean'})
                    patch_std = patch_std.rename({k:k + '_rolling_std'})
                print ('a', l)

                patch = xr.merge([patch_mean, patch_std]).load()

            print ('before before')
            patch_set.append(patch)

        print ('before')
        self.model_patches = xr.concat(patch_set, dim='sample')
        print ('efta')

        # space median
        if stats == 'time_mean_space_quantile':
            print (jkhasl)
            stats_str = '_time_mean_space_quantile'
            space_dims=['sample','lat','lon']
            qs = [0.1,0.5,0.9]
            self.model_patches = self.model_patches.quantile(qs, space_dims)

        # save
        self.model_patches.to_netcdf(config.data_path() + self.case +
                '/PatchSets/SOCHIC_PATCH_3h_20121209_20130331_bg_patch_set' +
                rolling_str + stats_str + '.nc')

exp10 = patch_set('EXP10')
exp10.get_glider_samples()
exp10.get_model_buoyancy_gradients_patch_set(stats='time_mean_space_quantile')
