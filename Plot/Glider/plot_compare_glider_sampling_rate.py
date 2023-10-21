import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import dask

class bootstrap_glider_samples(object):
    '''
    for ploting bootstrap samples of buoyancy gradients
    '''

    def __init__(self, case, offset=False):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'
        def expand_sample_dim(ds):
            ds = ds.expand_dims('sample')
            return ds
        self.samples = xr.open_mfdataset(self.data_path + 
               'GliderRandomSampling/glider_uniform_pre_resampled_path_*_00.nc',
                                         combine='nested', concat_dim='sample',
                                         preprocess=expand_sample_dim)

        # set time to float for averaging
        float_time = self.samples.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        self.samples['time_counter'] = clean_float_time

        # depth average
        self.samples = self.samples.mean('ctd_depth', skipna=True)
 
        # absolute value of buoyancy gradients
        self.samples['b_x_ml'] = np.abs(self.samples.b_x_ml)

        #self.model_rho = xr.open_dataset(self.data_path + 'rho.nc')
        #self.model_mld = xr.open_dataset(self.data_path +
        #                   'SOCHIC_PATCH_3h_20120101_20121231_grid_T.nc').mld
        

    def get_model_buoyancy_gradients(self):
        ''' restrict the model time to glider time '''
    
        bg = xr.open_dataset(config.data_path() + self.case +
                                   '/buoyancy_gradients.nc')
        
        bg = bg.assign_coords({'lon': bg.nav_lon.isel(y=0),
                                   'lat': bg.nav_lat.isel(x=0)})
        bg = bg.swap_dims({'x':'lon', 'y':'lat'})
         
        clean_float_time = self.samples.time_counter
        start = clean_float_time.min().astype('datetime64[ns]')
        end   = clean_float_time.max().astype('datetime64[ns]')
        
        print (self.samples)
        east = self.sample.lon.max()
        west = self.sample.lon.min()
        north = self.sample.lat.max()
        south = self.sample.lat.min()

        print (' ')
        print (' ')
        print ('start', start.values)
        print ('end', end.values)
        print (' ')
        print (' ')
        self.bg = bg.sel(time_counter=slice(start,end),
                         lon=slice(west,east), lat=slice(south,north))
        self.bg = self.bg.swap_dims({'lon':'x', 'lat':'y'})

    def add_sample(self, n=1, c='green'):
        '''
        add sample set of means and std to histogram
        n = sample size
        '''
 
        self.sample = xr.open_dataset(self.data_path + 
               'GliderRandomSampling/glider_uniform_pre_resampled_path_'+
                str(n) + '_00.nc')
        bx = np.abs(self.sample.b_x_ml)
        bx = bx.where(bx < 5e-8, drop=True)

        bx_stacked = bx.stack(z=('distance','ctd_depth'))

        plt.hist(bx_stacked, bins=100, density=True, alpha=1.0,
                 label='freq (m): ' + str(n), fill=False, edgecolor=c,
                 histtype='step')

        #hist, bins = np.histogram(bx_stacked.dropna('z', how='all'),
        #                        range=(1e-9,5e-8), density=True, bins=100)


        #bin_centers = bins[:-1] + bins[1:] / 2

        #hist_l_quant, hist_u_quant = bx_stacked.quantile([0.1,0.9],'z')
        #print (hist_l_quant)

        #self.ax.plot(bin_centers, hist, c=c, label='freq (m): ' + str(n),
        #             lw=1)
        #self.ax.fill_between(bin_centers, hist_l_quant,
        #                                  hist_u_quant,
        #                     color=c, edgecolor=None, alpha=0.2)

    def glider_sample_bootstrap_stats(self, n):
        set_size = self.samples.sizes['sample']
        random = np.random.randint(set_size, size=(set_size,n))

        set_of_means = []
        set_of_quants = []
        set_of_stds = []
        for sample in random:
            sample_set = self.samples.isel(sample=sample).b_x_ml
            print (sample_set)
            sample_set = sample_set.chunk(chunks={'sample':-1})
            set_mean = sample_set.mean(['sample','distance'])
            set_quant = sample_set.quantile([0.1,0.9],['sample','distance'])
            set_std = sample_set.std(['sample','distance'])
            set_of_means.append(set_mean)
            set_of_quants.append(set_quant)
            set_of_stds.append(set_std)

        chunks={'sets':-1}
        set_of_means = xr.concat(set_of_means, 'sets').chunk(chunks)
        set_of_quants = xr.concat(set_of_quants, 'sets').chunk(chunks)
        set_of_stds = xr.concat(set_of_stds, 'sets').chunk(chunks)

        mean = set_of_means.mean()
        quant = set_of_quants.mean()
        std = set_of_stds.mean()
        #quant = set_of_means.quantile([0.1,0.9])
        
        
        return mean, quant, std

    def add_model_lines(self):

        print ('a')
        # load buoyancy gradients       
        self.get_model_buoyancy_gradients()
        print (self.bg)

        print ('b')
        print (self.bg)
        self.bg = self.bg.mean('deptht')
        self.bg = np.abs(self.bg)

        print ('c')
        stacked_bgx = self.bg.bgx.stack(z=('time_counter','x','y'))
        stacked_bgy = self.bg.bgy.stack(z=('time_counter','x','y'))

        hist_x, bins = np.histogram(stacked_bgx.dropna('z', how='all'),
                                    range=(1e-9,5e-8), density=True, bins=100)
        hist_y, bins = np.histogram(stacked_bgy.dropna('z', how='all'),
                                    range=(1e-9,5e-8), density=True, bins=100)

        bin_centers = bins[:-1] + bins[1:] / 2
        self.ax.plot(bin_centers, hist_x, c='black', lw=1, zorder=1,
                     label='Model')

    def add_model_hist(self):
        '''
        add model buoyancy gradient of means and std to histogram
        '''

        #abs_bg = np.abs(bg)
        #abs_bg.where(abs_bgy<2e-8, drop=True)

        # load buoyancy gradients       
        self.get_model_buoyancy_gradients()

        self.bg = self.bg.mean('deptht')
        self.bg = np.abs(self.bg)
        self.bg = self.bg.where(self.bg < 5e-8, drop=True)
        stacked_bgx = self.bg.bgx.stack(z=('time_counter','x','y'))
        stacked_bgy = self.bg.bgy.stack(z=('time_counter','x','y'))
        
        print (stacked_bgx)
        plt.hist(stacked_bgx, bins=100, density=True, alpha=1.0,
                 label='model bgx', fill=False, edgecolor='gray',
                 histtype='step', zorder=11)
        plt.hist(stacked_bgy, bins=100, density=True, alpha=1.0,
                 label='model bgy', fill=False, edgecolor='black',
                 histtype='step', zorder=11)
        

    def histogram_buoyancy_gradients_and_samples(self, n):
        ''' 
        plot histogram of buoyancy gradients 
        n = sample_size
        '''

        self.figure, self.ax = plt.subplots(figsize=(5.5,4.5))

        sample_sizes = [10, 100, 1000]
        colours = ['g', 'b', 'r', 'y', 'c']

        for i in range(len(sample_sizes)):
            print ('sample', i)
            self.add_sample(n=sample_sizes[i], c=colours[i])
        print ('model')
        self.add_model_hist()

        self.ax.set_xlabel('Buoyancy Gradient')
        self.ax.set_ylabel('PDF')

        plt.legend()
        plt.savefig('EXP02_bg_glider_sample_freq.png', dpi=300)

    def plot_error_bars(self):
        
        means = []
        quants= []
        stds= []
        sample_sizes = [1, 2, 4, 10, 20]
        for n in sample_sizes:
            print ('n     :', n)
            mean , quant, std = self.glider_sample_bootstrap_stats(n)
            means.append(mean.values)
            #quants.append(quant.values)
            stds.append([mean.values - std.values, mean.values + std.values])
        #quants = np.transpose(np.array(quants))
        stds = np.transpose(np.array(stds))

        #print (quants.shape)
        plt.figure()
        plt.errorbar(sample_sizes, means, stds)
        plt.show()



m = bootstrap_glider_samples('EXP02')
m.histogram_buoyancy_gradients_and_samples(2)
#m.plot_error_bars()

#def plot_histogram():
#    m = glider_nemo('EXP03')
#    m.load_nemo_bg()
#    #m.load_glider_nemo()
#    #m.sub_sample_nemo()
#    m.histogram_buoyancy_gradient()
#
#plot_histogram()
