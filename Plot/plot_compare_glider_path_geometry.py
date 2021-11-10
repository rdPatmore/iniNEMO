import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import dask
import matplotlib

matplotlib.rcParams.update({'font.size': 8})

class glider_path_geometry(object):
    '''
    for ploting results of adjusting the geometry of the glider path
    '''

    def __init__(self, case, offset=False):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'
        #self.sample = xr.open_dataset(self.data_path + 
        #       'GliderRandomSampling/glider_uniform_rotated_path.nc')

        ## set time to float for averaging
        #float_time = self.sample.time_counter.astype('float64')
        #clean_float_time = float_time.where(float_time > 0, np.nan)
        #self.sample['time_counter'] = clean_float_time

        ## depth average
        #self.sample = self.sample.mean('ctd_depth', skipna=True)
 
        ## absolute value of buoyancy gradients
        #self.sample['b_x_ml'] = np.abs(self.sample.b_x_ml)


    def get_model_buoyancy_gradients(self):
        ''' restrict the model time to glider time '''
    
        bg = xr.open_dataset(config.data_path() + self.case +
                                   '/buoyancy_gradients.nc')
        
        bg = bg.assign_coords({'lon': bg.nav_lon.isel(y=0),
                                   'lat': bg.nav_lat.isel(x=0)})
        bg = bg.swap_dims({'x':'lon', 'y':'lat'})
         
        float_time = self.sample.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        start = clean_float_time.min().astype('datetime64[ns]')
        end   = clean_float_time.max().astype('datetime64[ns]')
        
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

    def add_sample(self, c='green', label='', path_append=''):
        '''
        add sample set of means and std to histogram
        n = sample size
        '''
 
        self.sample = xr.open_dataset(self.data_path + 
               'GliderRandomSampling/glider_uniform_' + path_append + '_00.nc')
        bx = np.abs(self.sample.b_x_ml)
        bx = bx.where(bx < 2e-8, drop=True)

        bx_stacked = bx.stack(z=('distance','ctd_depth'))

        plt.hist(bx_stacked, bins=100, density=True, alpha=0.3,
                 label=label, fill=False, edgecolor=c,
                 histtype='step')

    def add_model_hist(self):
        '''
        add model buoyancy gradient of means and std to histogram
        '''

        # load buoyancy gradients       
        self.get_model_buoyancy_gradients()

        self.bg = self.bg.mean('deptht')
        self.bg = np.abs(self.bg)
        self.bg = self.bg.where(self.bg < 2e-8, drop=True)
        stacked_bgx = self.bg.bgx.stack(z=('time_counter','x','y'))
        stacked_bgy = self.bg.bgy.stack(z=('time_counter','x','y'))
        
        plt.hist(stacked_bgx, bins=100, density=True, alpha=0.3,
                 label='model bgx', fill=False, edgecolor='gray',
                 histtype='step', zorder=11)
        plt.hist(stacked_bgy, bins=100, density=True, alpha=0.3,
                 label='model bgy', fill=False, edgecolor='black',
                 histtype='step', zorder=11)
        

    def histogram_buoyancy_gradients_and_sample(self):
        ''' 
        plot histogram of buoyancy gradients 
        n = sample_size
        '''

        self.figure, self.ax = plt.subplots(figsize=(4.5,4.0))


        self.add_sample(c='b', label='rotated', path_append='rotated_path')
        self.add_sample(c='r', label='non-rotated', path_append='non_rotated_path')
        self.add_model_hist()

        self.ax.set_xlabel('Buoyancy Gradient')
        self.ax.set_ylabel('PDF')

        plt.legend()
        plt.savefig('EXP02_bg_glider_rotation.png', dpi=300)

m = glider_path_geometry('EXP02')
m.histogram_buoyancy_gradients_and_sample()
