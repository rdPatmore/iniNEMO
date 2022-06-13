import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import dask
import matplotlib
from get_transects import get_transects

dask.config.set({"array.slicing.split_large_chunks": True})

matplotlib.rcParams.update({'font.size': 8})

class plot_buoyancy_gradients(object):
    '''
    for ploting results of adjusting the geometry of the glider path
    '''

    def __init__(self, case, offset=False):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'
        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'

    def save_bg_10(self):
        ''' save the 10th z-level of the buoyancy gradients '''

        bg = xr.open_dataset(config.data_path() + self.case +
                             self.file_id + 'bg.nc', chunks='auto')
        bg = bg.sel(deptht=10, method='nearest')
        bg.to_netcdf(config.data_path() + self.case +
                     self.file_id + 'bg_z10m.nc')

    def model_buoyancy_gradients_at_depth(self, depth=10):
    
        # model
        bg = xr.open_dataset(config.data_path() + self.case +
                             '/SOCHIC_PATCH_3h_20121209_20130331_bg.nc',
                             chunks='auto')
        bg = np.abs(bg.sel(deptht=depth, method='nearest'))

        bg = bg.sel(time_counter='2013-01-01', method='nearest')

        fig, ax= plt.subplots(1, figsize=(4.5,4.5))
        ax.pcolor(bg, cmap=plt.cm.RdBu_r, vmin=-1e-8, vmax=1e-8) 
        plt.show()

    def render_time_mean_bg(self, ax, bg, bg_type='norm'):
        '''
        adds time-mean horizontal slice of buoyancy gradients to a subplot
        '''

        bg = self.bg.mean('time_counter').load()
        if bg_type == 'norm':
            bg = (bg.bx ** 2 + bg.by ** 2) ** 0.5
        if bg_type == 'bx':
            bg = bg.bx
        if bg_type == 'by':
            bg = bg.by

        vmax = bg.max()
        ax.pcolor(bg.nav_lon, bg.nav_lat, bg, vmin=0, vmax=4.0e-8)
        ax.set_aspect('equal')

    def render_snapshot_bg(self, ax, bg):
        '''
        adds time-mean horizontal slice of buoyancy gradients to a subplot
        '''

        bg = bg.sel(time_counter='2013-01-01 00:00:00', method='nearest')
        bg = (bg.bx ** 2 + bg.by ** 2) ** 0.5

        vmax = bg.max()
        ax.pcolor(bg.nav_lon, bg.nav_lat, bg, vmin=0, vmax=4.0e-8)
        ax.set_aspect('equal')

    def render_cumulative_mean(self, ax, bg, bg_type='bx'):
        ''' 
        render cumulative mean
        '''

        t_index = xr.DataArray(np.arange(bg.sizes['time_counter']) + 1,
                               dims=['time_counter'])
        bg_cummean = bg.cumsum('time_counter') / t_index
        ax.plot(bg.time_counter, bg_cummean[bg_type].stack(z=['x','y']),
                alpha=0.005, color='navy')
        
    def plot_time_mean_bg(self, depth=10):
        '''
        plot time mean of buoyancy gradients at defined depth
        depth: assigns the depth level slice
        '''


        # load bg
        self.bg = xr.open_dataset(config.data_path() + self.case +
                                  self.file_id + 'bg_z10m.nc', chunks='auto')
       
        self.bg = self.bg.isel(x=slice(20,-20), y=slice(20,-20))
        self.bg = np.abs(self.bg)

        # initialise plots
        fig, axs = plt.subplots(3,2, figsize=(5.5,4.0))
        plt.subplots_adjust()
        
        self.render_cumulative_mean(axs[2,0], self.bg, bg_type='bx')
        self.render_cumulative_mean(axs[2,1], self.bg, bg_type='by')
        self.render_time_mean_bg(axs[0,0], self.bg, bg_type='bx')
        self.render_time_mean_bg(axs[0,1], self.bg, bg_type='by')
        self.render_time_mean_bg(axs[1,0], self.bg, bg_type='norm')
        self.render_snapshot_bg(axs[1,1], self.bg)

        #plt.show()
        plt.savefig('bg_means.png')
        

        

p = plot_buoyancy_gradients('EXP10')
p.plot_time_mean_bg()
