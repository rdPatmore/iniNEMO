import xarray as xr
import numpy as np
import config
import dask

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
        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'

    def get_10m_glider_samples(self, rotation=None):
        """ load processed glider samples restricted to 10 m depth """

        # files definitions
        prep = "GliderRandomSampling/glider_uniform_interp_1000"

        # get samples
        self.samples = xr.open_dataarray(self.data_path + prep +
             rotation_label + '_b_x_abs_10m_post_transects.nc', 
             decode_times=False, chunks=-1)

    def get_glider_samples(self, rotation=None):
        """ load processed glider samples """

        # files definitions
        prep = 'GliderRandomSampling/glider_uniform_interp_1000'
        rotation_label = ""
        if rotation:
            rotation_label = "_rotate_" + str(rotation) 

        # get samples
        self.samples = xr.open_dataset(self.data_path + prep +
             rotation_label + ".nc", decode_times=False, chunks=-1)

    def process_bg(self):
        """
        Get buoyancy gradients at 10m depth for full domain and calculate
        euclidian norm.
        """

        # model
        bg = np.abs(xr.open_dataset(config.data_path() + self.case +
                             self.file_id + 'bg_z10m.nc'))

        # get norm
        bg['bg_norm'] = (bg.bx ** 2 + bg.by ** 2) ** 0.5

        # assign alias
        self.ds = bg


    def get_N2(self):
        """
        Save N2 for full domain averaged over the mixed layer 
        """

        # get data
        W_path = self.data_path + "RawOutput/" + self.file_id + "grid_W.nc"
        T_path = self.data_path + "RawOutput/" + self.file_id + "grid_T.nc"
        kwargs = {'chunks':{'time_counter':100} ,'decode_cf':False} 
        N2 = xr.open_dataset(W_path, **kwargs).bn2
        mld = xr.open_dataset(T_path, **kwargs).mldr10_3
      
        # dubious merge of offset time_counter
        mld["time_counter"] = N2.time_counter

        # find mean N over mixed layer depth
        ds = N2.where(N2.depthw < mld).mean("depthw")
     
        # save
        ds.to_netcdf(self.data_path + "ProcessedVars/" + self.file_id +
                     "N2_mld.mc")
        

    def get_model_patch_set(self, stats=None, rolling=False, var="bg"):
        ''' 
        restrict the model time to glider time and sample areas
        Note: repition of function in plot_compare_gider_path
        '''

        rolling_str, stats_str = '', ''

        # get time glider bounds
        clean_float_time = self.samples.time_counter
        start = clean_float_time.min().astype('datetime64[s]')
        end   = clean_float_time.max().astype('datetime64[s]')

        # restrict model to glider time
        ds = self.ds.sel(time_counter=slice(start,end))
        
        # add lat-lon to dimensions
        ds = ds.assign_coords({'lon':(['x'], ds.nav_lon.isel(y=0)),
                               'lat':(['y'], ds.nav_lat.isel(x=0))})
        ds = ds.swap_dims({'x':'lon','y':'lat'})

        patch_set = []
        for (l, sample) in self.samples.groupby('sample'):
            # get limts of sample
            x0 = float(sample.lon.min())
            x1 = float(sample.lon.max())
            y0 = float(sample.lat.min())
            y1 = float(sample.lat.max())
 
            patch = ds.sel(lon=slice(x0,x1),
                           lat=slice(y0,y1))

            xi = np.arange(len(patch.lon))
            yi = np.arange(len(patch.lat))
            patch = patch.assign_coords({'xi':(['lon'], xi),
                                         'yi':(['lat'], yi)})
            patch = patch.swap_dims({'lon':'xi','lat':'yi'})
            patch = patch.reset_coords(['lon','lat'])
            patch = patch.expand_dims(sample=[l])

            dims = ['lon','lat','time_counter']
            if rolling:
                rolling_str = '_rolling'
                # contstruct allows for mean/std over multiple dims
                patch = patch.resample(time_counter='1H').median()
                patch = patch.rolling(time_counter=168, center=True).construct(
                                                               'weekly_rolling')
                dims = ['lat','lon','weekly_rolling']

            if stats == 'mean':
                patch = patch.mean(dims)
                stats_str = '_mean'
            if stats == 'std':
                patch = patch.std(dims)
                stats_str = '_std'
            if stats == 'median':
                stats_str = '_median'
                patch = patch.median(dims)
                patch = patch.std(dims)
            
            if stats == 'time_mean_space_quantile':
                kwargs = dict(time_counter=168, center=True, min_periods=1)
                patch_mean = patch.rolling(**kwargs).mean()
                patch_std = patch.rolling(**kwargs).std()
                for k in list(patch_mean.keys()):
                    if k in ['lat','lon']: continue
                    patch_mean = patch_mean.rename({k:k + '_rolling_mean'})
                    patch_std = patch_std.rename({k:k + '_rolling_std'})
                print ('a', l)

                patch = xr.merge([patch_mean, patch_std])
            patch_set.append(patch)
        self.model_patches = xr.concat(patch_set, dim='sample')

        # remove sample chunks to avoid clash with .quantile
        self.model_patches = self.model_patches.chunk({"sample":-1})

        #    patch.to_netcdf('Scratch/patch_' + str(l) + '.nc')
        # space median
        if stats == 'time_mean_space_quantile':
            stats_str = '_time_mean_space_quantile'
            space_dims=['sample','xi','yi']
            qs = [0.1,0.5,0.9]
            self.model_patchs = self.model_patches.quantile(qs, space_dims)

        # save
        self.model_patches.to_netcdf(config.data_path() + self.case +
                '/PatchSets/SOCHIC_PATCH_3h_20121209_20130331_' + var +
                '_patch_set' + rolling_str + stats_str + '.nc')

    def group_patch_files(self, stats=None):
        patch_set = xr.open_mfdataset('Scratch/patch*.nc')
                             
        # space median
        if stats == 'time_mean_space_quantile':
            stats_str = '_time_mean_space_quantile'
            space_dims=['sample','lat','lon']
            qs = [0.1,0.5,0.9]
            patch_set = patch_set.quantile(qs, space_dims)

        # save
        patch_set.to_netcdf(config.data_path() + self.case +
                '/PatchSets/SOCHIC_PATCH_3h_20121209_20130331_bg_patch_set' +
                rolling_str + stats_str + '.nc')


if __name__ == '__main__':
     dask.config.set(scheduler='single-threaded')
     exp10 = patch_set('EXP10')
     #exp10.get_glider_samples()
     exp10.get_N2()
     #exp10.get_model_patch_set(stats="time_mean_space_quantile", var="N2")
