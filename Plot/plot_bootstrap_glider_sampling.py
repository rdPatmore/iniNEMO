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
                                    'GliderRandomSampling/glider_uniform_*.nc',
                                         combine='nested', concat_dim='sample',
                                         preprocess=expand_sample_dim)
        self.samples = self.samples.mean('ctd_depth')
        self.samples['b_x_ml'] = np.abs(self.samples.b_x_ml)
        #time_unit = 'seconds since 1971-01-01 00:00:00'
        #self.samples.time_counter.attrs['units'] = time_unit
        #self.samples = xr.decode_cf(self.samples)
        print (self.samples)
        #self.model_rho = xr.open_dataset(self.data_path + 'rho.nc')
        #self.model_mld = xr.open_dataset(self.data_path +
        #                   'SOCHIC_PATCH_3h_20120101_20121231_grid_T.nc').mld
        
        # time is missing from data need to solve
        #self.cut_model_time()

    def get_model_buoyancy_gradients(self):
        ''' restrict the model time to glider time '''
    
        bg = xr.open_dataset(config.data_path() + self.case +
                                   '/buoyancy_gradients.nc')
        
         
        float_time = self.samples.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        start = clean_float_time.min().astype('datetime64[ns]')
        end   = clean_float_time.max().astype('datetime64[ns]')
        print (' ')
        print (' ')
        print ('start', start.values)
        print ('end', end.values)
        print (' ')
        print (' ')
        self.bg = bg.sel(time_counter=slice(start,end))

    def add_sample(self, n=1, colour='green'):
        '''
        add sample set of means and std to histogram
        n = sample size
        '''
 
        set_size = self.samples.sizes['sample']

        print (' ')
        print ('            set_size', set_size)
        print (' ')
        print ('setsize/n', set_size/n)
        print ('int setsize/n', int(set_size/n))
        # get random group
        random = np.random.randint(set_size, size=(set_size,n))
        for sample in random:
            sample_set = self.samples.isel(sample=sample).b_x_ml
            set_stacked = sample_set.stack(z=('distance','sample'))
            print (set_stacked)
        print (random)
        print (random.shape)
        print (jsdfhkl)
        






        ## get sampling indexes
        #random = (np.random.random(n) * set_size).astype('int')
        #print (random.shape)
        #random = (np.random.random(n) * set_size)

        # select from sample set using random as index
        self.samples = self.samples.assign_coords(sets=('rand_sample', random))

        # mean over sample sets
        set_means = self.samples.groupby('sets').mean()
    
        # plot histogram 
        plt.hist(set_means, bins=50, density=True, alpha=0.3,
                 label='sample bx', fill=False, edgecolor=colour,
                 histtype='step')

    def glider_sample_bootstrap_stats(self, n):
        set_size = self.samples.sizes['sample']
        random = np.random.randint(set_size, size=(set_size,n))

        set_of_means = []
        set_of_quants = []
        for sample in random:
            sample_set = self.samples.isel(sample=sample).b_x_ml
            print (sample_set)
            sample_set = sample_set.chunk(chunks={'sample':-1})
            set_mean = sample_set.mean(['sample','distance'])
            set_quant = sample_set.quantile([0.1,0.9],['sample','distance'])
            set_of_means.append(set_mean)
            set_of_quants.append(set_quant)

        set_of_means = xr.concat(set_of_means, 'sets')
        set_of_means = set_of_means.chunk(chunks={'sets':-1})

        set_of_quants = xr.concat(set_of_quants, 'sets')
        set_of_quants = set_of_quants.chunk(chunks={'sets':-1})

        mean = set_of_means.mean()
        quant = set_of_quants.mean()
        #quant = set_of_means.quantile([0.1,0.9])
        
        
        return mean, quant

    def add_model(self):
        '''
        add model buoyancy gradient of means and std to histogram
        '''

        #abs_bg = np.abs(bg)
        #abs_bg.where(abs_bgy<2e-8, drop=True)

        # load buoyancy gradients       
        self.get_model_buoyancy_gradients()

        self.bg = self.bg.mean('deptht')
        self.bg = np.abs(self.bg)
        self.bg = self.bg.where(self.bg < 1e-7, drop=True)
        stacked_bgx = self.bg.bgx.stack(z=('time_counter','x','y'))
        stacked_bgy = self.bg.bgy.stack(z=('time_counter','x','y'))
        
        print (stacked_bgx)
        plt.hist(stacked_bgx, bins=50, density=True, alpha=0.3,
                 label='model bgx', fill=False, edgecolor='red',
                 histtype='step')
        plt.hist(stacked_bgy, bins=50, density=True, alpha=0.3,
                 label='model bgy', fill=False, edgecolor='blue',
                 histtype='step')
        

    def histogram_buoyancy_gradients_and_samples(self, n):
        ''' 
        plot histogram of buoyancy gradients 
        n = sample_size
        '''

        self.figure = plt.figure()

        #self.add_model()
        self.glider_sample_bootstrap_stats(n)
        print (dfjk)
        self.add_sample(n=n)

        plt.legend()
        plt.show()

    def plot_error_bars(self):
        
        means = []
        quants= []
        sample_sizes = [1, 2, 4, 10, 20]
        for n in sample_sizes:
            print ('n     :', n)
            mean , quant = self.glider_sample_bootstrap_stats(n)
            means.append(mean.values)
            quants.append(quant.values)
        quants = np.transpose(np.array(quants))

        print (quants.shape)
        plt.figure()
        plt.errorbar(sample_sizes, means, quants)
        plt.show()



m = bootstrap_glider_samples('EXP02')
#m.histogram_buoyancy_gradients_and_samples(2)
m.plot_error_bars()

#def plot_histogram():
#    m = glider_nemo('EXP03')
#    m.load_nemo_bg()
#    #m.load_glider_nemo()
#    #m.sub_sample_nemo()
#    m.histogram_buoyancy_gradient()
#
#plot_histogram()
