import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import numpy as np

class bootstrap_glider_samples(object):
    '''
    for ploting bootstrap samples of buoyancy gradients
    '''

    def __init__(self, case, offset=False):
        self.root = config.root_old()
        self.case = case
        self.data_path = config.data_path_old() + self.case + '/'

    def get_glider_uniform(self):
        def expand_sample_dim(ds):
            ds = ds.expand_dims('sample')
            return ds
        file_paths = [self.data_path + 'GliderRandomSampling/glider_uniform_' +
                      self.append + str(i).zfill(2) + '.nc' for i in range(100)]
        self.samples = xr.open_mfdataset(file_paths,
                                         combine='nested', concat_dim='sample',
                                         preprocess=expand_sample_dim)

        # set time to float for averaging
        float_time = self.samples.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        self.samples['time_counter'] = clean_float_time

        # depth average
        #self.samples = self.samples.reset_coords(['lon','lat']) # to preserve coords
        #self.samples = self.samples.mean('ctd_depth', skipna=True)
        #self.samples = self.samples.set_coords(['lon','lat']) # to preserve coords
 

        # absolute value of buoyancy gradients
        self.samples['b_x_ml'] = np.abs(self.samples.b_x_ml)


    def get_global_buoyancy_gradients(self):
        ''' load and processs global buoyancy gradient '''
    
        bg_global = xr.open_dataset(config.data_path_old() + self.case +
                                   '/buoyancy_gradients.nc', 
                                 chunks={'time_counter':10})
        
        bg_global = bg_global.assign_coords(
                               {'lon': bg_global.nav_lon.isel(y=0),
                                'lat': bg_global.nav_lat.isel(x=0)})
        self.bg_global = bg_global.swap_dims({'x':'lon', 'y':'lat'})

        # mean over mixed layer
        self.bg_global = self.bg_global.mean('deptht')

        self.bg_global = np.abs(self.bg_global)
         

    def restrict_time_and_space(self, variable):
        ''' restrict data to glider time and space '''

        clean_float_time = self.samples.time_counter
        start = clean_float_time.min().astype('datetime64[ns]')
        end   = clean_float_time.max().astype('datetime64[ns]')
        
        east = self.sample.lon.max()
        west = self.sample.lon.min()
        north = self.sample.lat.max()
        south = self.sample.lat.min()

        variable = variable.sel(time_counter=slice(start,end),
                         lon=slice(west,east), lat=slice(south,north))
        variable = variable.swap_dims({'lon':'x', 'lat':'y'})

        return variable

    def calc_mean_model_bg_stats(self, direction='bgx'):
        ''' calculate hist of buoyancy gradient stats over glider patches '''

        self.get_global_buoyancy_gradients()
        set_size = self.samples.sizes['sample']

        set_of_hists = []
        for i, sample in enumerate(range(set_size)):
            print (i, ' / ', set_size)
            self.sample = self.samples.isel(sample=sample)
            model_patch = self.restrict_time_and_space(
                                                      self.bg_global[direction])
            stacked = model_patch.stack(z=('time_counter','x','y'))
            hist, bins = np.histogram(stacked,
                                      range=(1e-9,5e-8), density=True, bins=100)
            bin_centers = bins[:-1] + bins[1:] / 2
            hist = xr.DataArray(hist, dims=('bin_centers'), 
                                      coords={'bin_centers':bin_centers})
            set_of_hists.append(hist)

        set_of_hists = xr.concat(set_of_hists, 'sets')

        mean = set_of_hists.mean(['sets'])
        quant = set_of_hists.quantile([0.1,0.9],['sets'])
        std = set_of_hists.std(['sets'])

        mean.to_netcdf( direction + '_bootstraped_over_patches_mean.nc')
        quant.to_netcdf(direction + '_bootstraped_over_patches_decile.nc')
        std.to_netcdf(  direction + '_bootstraped_over_patches_std.nc')

    def calc_mean_glider_stats(self, append=''):
        '''
        calculate averages of the statistics from all 100 glider samples
            - mean of means
            - mean of deciles
            - mean of standard deviations
        '''
        self.append=append
        self.get_glider_uniform()

        set_size = self.samples.sizes['sample']

        set_of_hists = []
        for sample in range(set_size):
            sample_set = self.samples.isel(sample=sample).b_x_ml
            #sample_set = sample_set.chunk(chunks={'sample':-1})
            stacked = sample_set.stack(z=('distance','ctd_depth'))
            hist, bins = np.histogram(sample_set,
                                      range=(1e-9,5e-8), density=True, bins=100)
            bin_centers = bins[:-1] + bins[1:] / 2
            hist = xr.DataArray(hist, dims=('bin_centers'), 
                                      coords={'bin_centers':bin_centers})
            set_of_hists.append(hist)

        set_of_hists = xr.concat(set_of_hists, 'sets')

        mean = set_of_hists.mean(['sets'])
        quant = set_of_hists.quantile([0.1,0.9],['sets'])
        std = set_of_hists.std(['sets'])

        mean.to_netcdf('glider_' + self.append +
                       'bootstraped_over_patches_mean.nc')
        quant.to_netcdf('glider_' + self.append + 
                        'bootstraped_over_patches_decile.nc')
        std.to_netcdf('glider_' + self.append + 
                      'bootstraped_over_patches_std.nc')

    def plot_model_stats(self, direction='bgx', c='black'):

        mean   = xr.load_dataarray(direction +
                                   '_bootstraped_over_patches_mean.nc')
        decile = xr.load_dataarray(direction +
                                   '_bootstraped_over_patches_decile.nc')
        hist_l = decile.sel(quantile=0.1)
        hist_u = decile.sel(quantile=0.9)
        self.ax.plot(mean.bin_centers, mean, c=c, lw=1, zorder=1,
                     label='Model')
        self.ax.fill_between(hist_l.bin_centers, hist_l, hist_u,
                     color=c, edgecolor=None, alpha=0.2)

    def plot_glider_stats(self, c='red'):

        mean   = xr.load_dataarray('glider_' + self.append +
                                   'bootstraped_over_patches_mean.nc')
        decile = xr.load_dataarray('glider_' + self.append + 
                                   'bootstraped_over_patches_decile.nc')

        hist_l = decile.sel(quantile=0.1)
        hist_u = decile.sel(quantile=0.9)
        self.ax.plot(mean.bin_centers, mean, c=c, lw=1, zorder=1,
                     label='glider')
        self.ax.fill_between(hist_l.bin_centers, hist_l, hist_u,
                     color=c, edgecolor=None, alpha=0.2)
        

    def histogram_buoyancy_gradients_and_samples(self, append=''):
        ''' 
        plot histogram of buoyancy gradients 
        ''' 

        self.append = append
        self.figure, self.ax = plt.subplots(figsize=(5.5,4.5))

        sample_sizes = [10, 100, 1000]
        colours = ['g', 'b', 'r', 'y', 'c']
        self.plot_model_stats(direction='bgx')
        self.plot_model_stats(direction='bgy', c='grey')
        self.plot_glider_stats()

        self.ax.set_xlabel('Buoyancy Gradient')
        self.ax.set_ylabel('PDF')

        self.ax.set_ylim(0,3e8)
        self.ax.set_xlim(1.5e-9,7e-8)

        plt.legend()
        plt.savefig('EXP02_bg_' + append + 'sampling_comparison.png', dpi=600)

m = bootstrap_glider_samples('EXP02')
#m.calc_mean_glider_stats(append='dive_')
m.histogram_buoyancy_gradients_and_samples(append='dive_')
m.histogram_buoyancy_gradients_and_samples(append='climb_')
m.histogram_buoyancy_gradients_and_samples(append='')
