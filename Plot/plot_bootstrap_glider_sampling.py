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
                                    'GliderRandomSampling/glider_uniform_6*.nc',
                                         combine='nested', concat_dim='sample',
                                         preprocess=expand_sample_dim)
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
        start = self.samples.time_counter.min()
        end   = self.samples.time_counter.max()
        self.bg = bg.sel(time_counter=slice(start,end))

    def add_sample(self, n=1):
        '''
        add sample set of means and std to histogram
        n = sample size
        '''
 
        # get sampling indexes
        random = int(np.random.random(set_size) * set_size)

        # select from sample set using random as index
        self.samples = self.samples.assign_coords(sets=('sample', random))

        # mean over sample sets
        set_means = self.samples.groupby('sets').mean()
    
        # plot histogram 
        self.plt.hist(set_means, bins=100, density=True, alpha=0.3,
                 label='sample bx', fill=False, edgecolor='l',
                 histtype='step')
        

    def add_model(self):
        '''
        add model buoyancy gradient of means and std to histogram
        '''

        #abs_bg = np.abs(bg)
        #abs_bg.where(abs_bgy<2e-8, drop=True)

        # load buoyancy gradients       
        self.get_model_buoyancy_gradients()
        
        self.plt.hist(self.bg.bgx_mean, bins=100, density=True, alpha=0.3,
                 label='model bgx', fill=False, edgecolor='red',
                 histtype='step')
        self.plt.hist(self.bg.bgy, bins=100, density=True, alpha=0.3,
                 label='model bgy', fill=False, edgecolor='l',
                 histtype='step')
        

    def histogram_buoyancy_gradients_and_samples(self, n):
        ''' 
        plot histogram of buoyancy gradients 
        n = sample_size
        '''

        self.plt = plt.figure()

        self.add_model()
        #self.add_sample(n=n)

        plt.legend()
        plt.show()


m = bootstrap_glider_samples('EXP02')
m.histogram_buoyancy_gradients_and_samples(1)

#def plot_histogram():
#    m = glider_nemo('EXP03')
#    m.load_nemo_bg()
#    #m.load_glider_nemo()
#    #m.sub_sample_nemo()
#    m.histogram_buoyancy_gradient()
#
#plot_histogram()
