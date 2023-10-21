import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.dates as mdates
import numpy as np
import dask
import matplotlib
import datetime
import matplotlib.gridspec as gridspec
import scipy.stats as stats
from get_transects import get_transects

matplotlib.rcParams.update({'font.size': 8})

class bg_diff(object):
    '''
    for ploting bootstrap samples of buoyancy gradients
    '''

    def __init__(self, case, offset=False, 
                 subset='', transect=False, interp='1000'):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'

        # subset domain
        self.subset=subset
        self.append=''
        patch=''
        if self.subset=='north':
            self.append='_n'
            patch = '_north_patch'
        if self.subset=='south':
            self.append='_s'
            patch = '_south_patch'
        
        self.append = self.append + '_interp_' + interp

        # load samples
        prep = 'GliderRandomSampling/glider_uniform_interp_'+ interp + \
               patch + '.nc'
        self.samples = xr.open_dataset(self.data_path + prep)

        # drop unused variables
        self.samples = self.samples.drop(['rho','bg_norm',
                                          'vosaline','votemper'])

        # depth slice
        self.samples = self.samples.sel(ctd_depth=10, method='nearest')
        #self.samples = self.samples.isel(sample=slice(None,3)).load()


        # unify times
        self.samples['time_counter'] = self.samples.time_counter.isel(sample=0)
    
        # get transects and remove 2 n-s excursions
        # this cannot currently deal with depth-distance data (1-d only)
        bg_norm = self.samples.bg_norm_ml.groupby('sample').map(
                                      get_transects, offset=True, cut_meso=True)
        b_x = self.samples.b_x_ml.groupby('sample').map(
                                      get_transects, offset=True, cut_meso=True)

        self.samples = xr.merge([bg_norm,b_x])

        # set time to float for averaging
        float_time = self.samples.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        self.samples['time_counter'] = clean_float_time
 
        # absolute value of buoyancy gradients
        self.samples = np.abs(self.samples)

    def get_stats(self, samp, random, n):
        #find stats for each bootstrap sample
        stats_bs = []
        for i, sample in enumerate(random):
            sample_set = samp.isel(sample=sample)#.b_x_ml
            set_stacked = sample_set.stack(z=('distance','sample'))
           
            # get stats
            std = set_stacked.std('z')
            std.name = 'bs_std'

            mean = set_stacked.mean('z')
            mean.name = 'bs_mean'

            quants = [0.02,0.05,0.1,0.2,0.5,0.8,0.9,0.95,0.98]
            quantiles = set_stacked.quantile(quants, 'z')
            quantiles.name = 'bs_quantiles'

            # form dataset
            stats_bs.append(xr.merge([std,mean,quantiles]))
        
        stats_set = xr.concat(stats_bs, dim='bootstrap_sample').mean(
                    'bootstrap_sample')
        stats_set = stats_set.assign_coords({'sample_size':n})

        return stats_set

    def calc_diff_in_bg_terms(self, N=30):
        ''' 
        calc the difference between bg_norm_ml and b_x_ml at each point
        bootstrapped according to glider number

        N = total number of gliders tested
        '''

        # get fractional difference at each point: (pred-truth)/truth
        bg_diff_frac = (self.samples.b_x_ml - self.samples.bg_norm_ml) / \
                        self.samples.bg_norm_ml 

        # convert to percent
        bg_diff = np.abs(bg_diff_frac) * 100
        # ~~~ bootstrap ~~~ #

        set_size = self.samples.sizes['sample']
        n_set = []
        for n in range(1,N+1):
            # get random group, shape (bootstrap iters, number of gliders)
            random = np.random.randint(set_size, size=(set_size,n))

            # bootstrap stats for n gliders
            n_set.append(self.get_stats(bg_diff, random, n))

        n_set = xr.concat(n_set, dim='sample_size')

        n_set.to_netcdf(self.data_path + 
                       '/BgGliderSamples/diff_bg_norm_and_b_x.nc')
        
    def plot_diff_in_bg_terms(self):
        ''' 
        plot the difference between bg_norm_ml and b_x_ml at each point
        bootstrapped according to glider number
        '''

m = bg_diff('EXP10', subset='', transect=True)
m.calc_diff_in_bg_terms()
