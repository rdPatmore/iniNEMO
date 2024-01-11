import xarray as xr
import config
import iniNEMO.Process.Common.model_object as mo
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import dask
import matplotlib
from iniNEMO.Process.Glider.get_transects import get_transects
import matplotlib.dates as mdates
import iniNEMO.Plot.Utils.utils as utils

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
        p = ax.pcolor(bg.nav_lon, bg.nav_lat, bg, vmin=0, vmax=5.0e-8)
        ax.set_aspect('equal')
 
        return p 

    def render_snapshot_bg(self, ax, bg, bg_type='norm'):
        '''
        adds time-mean horizontal slice of buoyancy gradients to a subplot
        '''

        bg = bg.sel(time_counter='2013-01-01 00:00:00', method='nearest')
        if bg_type == 'norm':
            bg = (bg.bx ** 2 + bg.by ** 2) ** 0.5
        if bg_type == 'bx':
            bg = bg.bx
        if bg_type == 'by':
            bg = bg.by

        vmax = bg.max()
        p = ax.pcolor(bg.nav_lon, bg.nav_lat, bg, vmin=0, vmax=2.0e-7)
        ax.set_aspect('equal')
        return p

    def render_cumulative_mean(self, ax, bg_type='bx', dcdt=False):
        ''' 
        render cumulative mean in buoyancy gradients
        
        bg_type: direction of buoyancy gradient
        dcdt : rate of change over time
        '''

        t_index = xr.DataArray(np.arange(self.bg.sizes['time_counter']) + 1,
                               dims=['time_counter'])
        bg_cummean = bg.cumsum('time_counter') / t_index

        if dcdt:
            # difference in cummean over time
            dc = bg_cummean.diff('time_counter')
            # time in seconds
            dt = (bg_cummean.astype('int')/1e9).diff('time_counter')
            # gradient
            bg_cummean = dc/dt

        ax.plot(bg.time_counter, bg_cummean[bg_type].stack(z=['x','y']),
                alpha=0.005, color='navy')

    def render_buoyancy_gradient_mean_and_std(self, ax, bg):
        '''
        render a time-series of buoyancy gradients over time
        '''

        # stats
        bg_mean = bg.mean(['x','y'])
        chunk=dict(x=-1,y=-1)
        bg_quant = bg.chunk(chunks=chunk).quantile([0.05,0.95],['x','y'])
        bg_l, bg_u = bg_quant.sel(quantile=0.05), bg_quant.sel(quantile=0.95)

        # colours
        c0 = 'k'
        c1 = '#f18b00'

        # render x
        ax.plot(bg_mean.bx.time_counter, bg_mean.bx, c=c0,
                label=r'$db/dx$')
        ax.fill_between(bg_l.bx.time_counter, bg_l.bx, bg_u.bx,
                        color=c0, alpha=0.5)

        # render y
        ax.plot(bg_mean.bx.time_counter, bg_mean.bx, c=c1,
                label=r'$db/dy$')
        ax.fill_between(bg_l.by.time_counter, bg_l.by, bg_u.by,
                        color=c1, alpha=0.5)

    def plot_bg_timeseries_with_north_south(self, depth=10):
        '''
        plot timeseries of buoyancy gradients at defined depth
        with split of north and south
        depth: assigns the depth level slice
        '''

        # load bg
        self.bg = xr.open_dataset(config.data_path() + self.case +
                                  self.file_id + 'bg_z10m.nc', chunks='auto')
        self.bg = self.bg.isel(x=slice(20,-20), y=slice(20,-20))
        self.bg = np.abs(self.bg)
 
        # initialise plots
        fig, axs = plt.subplots(3,1, figsize=(5.5,4.0))
        plt.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.15,
                            hspace=0.13)

        print (self.bg.y.max().values)
        bg_s = self.bg.sel(y=slice(int(self.bg.y.max()/2),-1))
        bg_n = self.bg.sel(y=slice(0,int(self.bg.y.max()/2)))
        
        self.render_buoyancy_gradient_mean_and_std(axs[0], self.bg)
        self.render_buoyancy_gradient_mean_and_std(axs[1], bg_s)
        self.render_buoyancy_gradient_mean_and_std(axs[2], bg_n)

        # axs labels
        for ax in axs[:2]:
            ax.set_xticklabels([])
        for ax in axs:
            ax.set_ylabel('buoyancy\ngradient')
            ax.set_ylim(0, 1.7e-7)
            ax.set_xlim(self.bg.time_counter.min(), self.bg.time_counter.max())

        # region labels
        axs[0].text(0.98, 0.88, 'full domain', transform=axs[0].transAxes,
                    ha='right')
        axs[1].text(0.98, 0.88, 'north', transform=axs[1].transAxes,
                    ha='right')
        axs[2].text(0.98, 0.88, 'south', transform=axs[2].transAxes,
                    ha='right')

        # date labels
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        # Rotates and right-aligns 
        for label in axs[2].get_xticklabels(which='major'):
            label.set(rotation=35, horizontalalignment='right')
        axs[2].set_xlabel('date')

        plt.savefig('bg_time_series.png')

    def plot_bg_timeseries_full_model(self, depth=10):
        '''
        plot timeseries of buoyancy gradients at defined depth
        with split of north and south
        depth: assigns the depth level slice
        '''

        # load bg
        self.bg = xr.open_dataset(config.data_path() + self.case +
                                  self.file_id + 'bg_z10m.nc', chunks='auto')
        self.bg = self.bg.isel(x=slice(20,-20), y=slice(20,-20))
        self.bg = np.abs(self.bg)

        # initialise plots
        fig, ax = plt.subplots(1,1, figsize=(4.0,2.0))
        plt.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.3,
                            hspace=0.13)
        
        self.render_buoyancy_gradient_mean_and_std(ax, self.bg)

        # legend
        plt.legend()

        # axis params
        ax.set_ylabel('buoyancy gradient')
        ax.set_ylim(0, 1.7e-7)
        ax.set_xlim(self.bg.time_counter.min(), self.bg.time_counter.max())

        # date labels
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        # Rotates and right-aligns 
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=35, horizontalalignment='right')
        ax.set_xlabel('date')

        plt.savefig('bg_time_series_full_model.png', dpi=600)
        
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
        fig, axs = plt.subplots(2,3, figsize=(5.5,3.8))
        plt.subplots_adjust(left=0.1, right=0.88, top=0.97, bottom=0.10,
                            hspace=0.05, wspace=0.06)
        
        #self.render_cumulative_mean(axs[2,0], self.bg, bg_type='bx')
        #self.render_cumulative_mean(axs[2,1], self.bg, bg_type='by')
        self.render_time_mean_bg(axs[0,0], self.bg, bg_type='bx')
        self.render_time_mean_bg(axs[0,1], self.bg, bg_type='by')
        p0 = self.render_time_mean_bg(axs[0,2], self.bg, bg_type='norm')
        self.render_snapshot_bg(axs[1,0], self.bg, bg_type='bx')
        self.render_snapshot_bg(axs[1,1], self.bg, bg_type='by')
        p1 = self.render_snapshot_bg(axs[1,2], self.bg, bg_type='norm')
        #self.render_buoyancy_gradient_mean_and_std(axs[2,0])

        # colour bar
        pos = axs[0,2].get_position()
        cbar_ax = fig.add_axes([0.89, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.3, 0.5, 'buoyancy gradient', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left')

        pos = axs[1,2].get_position()
        cbar_ax = fig.add_axes([0.89, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.3, 0.5, 'buoyacy gradient', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left')
        for ax in axs[:,1:].flatten():
            ax.set_yticklabels([])
        for ax in axs[0,:]:
            ax.set_xticklabels([])
        for ax in axs[1,:]:
            ax.set_xlabel('longitude')
        for ax in axs[:,0]:
            ax.set_ylabel('latitude')

        # titles
        axs[0,0].set_title(r'$b_x$')
        axs[0,1].set_title(r'$b_y$')
        axs[0,2].set_title(r'$|b_{(x,y)}}|$')

        plt.savefig('bg_means.png', dpi=600)

    def plot_buoyancy_gradient_directional_bias(self):
        '''
        plot histogram of the meridional and zonal components of the 
        buoyancy gradiets
        '''

        # load
        bg_hist = xr.open_dataset(config.data_path() + self.case +
                                  self.file_id + 'bg_model_hist.nc').load()

        # get step boundaries
        stair_edges = np.unique(np.concatenate((bg_hist.bin_left.values, \
                                               bg_hist.bin_right.values)))

        # plot
        c1 = '#f18b00'
        colours=[c1, 'purple', 'green']# 'navy','turquoise']
        fig, ax = plt.subplots(1, figsize=(3.2,4))
        ax.stairs(bg_hist.hist_norm, stair_edges, orientation='horizontal',
                  label=r'$|\nabla b|$', color='grey', lw=2)
        ax.stairs(bg_hist.hist_x, stair_edges, orientation='horizontal',
                  label=r'$db/dx$', color=colours[1], lw=2)
        ax.stairs(bg_hist.hist_y, stair_edges, orientation='horizontal',
                  label=r'$db/dy$', color=colours[0], lw=2)

        # axis params
        ax.xaxis.get_offset_text().set_visible(False)
        ax.set_ylabel(r'buoyancy gradient ($\times 10^{-8}$ s$^{-2}$)')
        ax.yaxis.get_offset_text().set_visible(False)
        ax.set_xlabel(r'PDF ($\times 10 ^{-8}$)')
        ax.set_ylim(stair_edges[0],stair_edges[-1])

        ax.legend()


        plt.show()

p = plot_buoyancy_gradients('EXP10')
#p.plot_bg_timeseries_with_north_south()
p.plot_bg_timeseries_full_model()
#p.plot_time_mean_bg()
