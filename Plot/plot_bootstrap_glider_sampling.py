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
        self.model_rho = xr.open_dataset(self.data_path + 'rho.nc')
        
        self.cut_model_time()


    def cut_model_time(self):
        ''' restrict the model time to glider time '''
    
        start = self.samples.time_counter.min()
        end   = self.samples.time_counter.max()
        self.model_rho = self.model_rho.sel(time_counter=slice(start,end))

    def add_sample(n=1):
        '''
        add sample set of means and std to histogram
        n = sample size
        '''
 
        # get sampling indexes
        random = int(np.random.random(set_size) * set_size)

        # select from sample set using random as index
         
    
        # reshape accoring to n 
   
        

        

    def sample_times(self, data, start_month='01'):
        ''' take a time sample based on time differnce in glider sample '''
        self.load_giddy()
        time = self.giddy.time.isel(density=50)
        #time_diff = time.diff('distance').pad(distance=(0,1)).fillna(0).cumsum()
        time_diff = time.diff('distance').fillna(0).cumsum()
        start_date = np.datetime64('2012-' + start_month + '-01 00:00:00')
        time_span = start_date + time_diff
        data = data.interp(time_counter=time_span.values, method='nearest')
        return data

    def histogram_buoyancy_gradient(self):
        ''' 
        plot histogram of buoyancy gradients 
        '''

        abs_bg = np.abs(bg)
        abs_bg.where(abs_bgy<2e-8, drop=True)

        plt.hist(abs_bg, bins=100, density=True, alpha=0.3,
                 label='glider bg', fill=False, edgecolor='red',
                 histtype='step')
        plt.legend()
        plt.show()


m = bootstrap_glider_samples('EXP02')

#def plot_histogram():
#    m = glider_nemo('EXP03')
#    m.load_nemo_bg()
#    #m.load_glider_nemo()
#    #m.sub_sample_nemo()
#    m.histogram_buoyancy_gradient()
#
#plot_histogram()
