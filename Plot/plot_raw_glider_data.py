import xarray as xr
import config
import matplotlib.pyplot as plt
import numpy as np
import dask
import glidertools as gt
import matplotlib

matplotlib.rcParams.update({'font.size': 8})

class raw_glider_plotting(object):
    '''
    plotting of raw glider data
    '''

    def __init__(self):
        self.root = config.root()
        self.giddy = xr.open_dataset(self.root + 'Giddy_2020/merged_raw.nc')

        # reduce to single time coordinate
        self.giddy = self.giddy.drop('ctd_time_dt64')

    def interpolate_to_unifrom_depths(self):
        ''' interpolate raw glider data to uniform 1 m grid in vertical '''

        # change time units for interpolation 
        timedelta = self.giddy.ctd_time-np.datetime64('1970-01-01 00:00:00')
        self.giddy['ctd_time'] = timedelta.astype(np.int64)/ 1e9

        #uniform_distance = np.arange(0, self.giddy.distance.max(),1000)

        glider_uniform_i = []
        # interpolate to 1 m vertical grid
        for (label, group) in self.giddy.groupby('dives'):
            if group.sizes['ctd_data_point'] < 2:
                continue
            group = group.swap_dims({'ctd_data_point': 'ctd_depth'})

            # remove duplicate index values
            _, index = np.unique(group['ctd_depth'], return_index=True)
            group = group.isel(ctd_depth=index)

            # interpolate - 1 m 
            depth_uniform = group.interp(ctd_depth=np.arange(0.0,999.0,1))

            uniform = depth_uniform.expand_dims(dive=[label])
            glider_uniform_i.append(uniform)

        self.glider_uniform = xr.concat(glider_uniform_i, dim='dive')

    def get_sampling_distance(self):
        ''' caluclates distance between samples '''

        distance_k = []
        for (label, group) in self.glider_uniform.groupby('ctd_depth'):
            distance = xr.DataArray(
                                gt.utils.distance(group.longitude,
                                                  group.latitude),
                                                  dims='dive')/1000 # km
            distance_k.append(distance)
        self.glider_uniform['distance'] = xr.concat(distance_k, dim='ctd_depth')

    def get_sampling_rates(self):
        ''' calculates time between samples '''
        
        time_i = []
        for (label, group) in self.glider_uniform.groupby('ctd_depth'):
            dt = group.ctd_time.diff('dive')/3600 # hours
            dt = dt.where(dt > 0, drop=True)
            #dt = dt.pad(ctd_data_point=(0,1))
            #group = group.swap_dims({'distance': 'dive'})

            time_i.append(dt)
        self.glider_uniform['dt'] = xr.concat(time_i, dim='ctd_depth')

    def plot_sampling_rates(self):
        ''' Plot raw glider sampling rates. Spatial and temporal '''

        # plot prep
        fig, axs = plt.subplots(1, 2, figsize=(4.5,4))
        plt.subplots_adjust(right=0.83, top=0.98, bottom=0.15)

        # calcs
        self.interpolate_to_unifrom_depths()
        self.get_sampling_distance()
        self.get_sampling_rates()
 
        # load for faster plotting
        self.glider_uniform = self.glider_uniform.load()

        glider_stacked = self.glider_uniform.stack(z=('dive','ctd_depth'))
        glider_stacked = glider_stacked.fillna(0.0)
        #glider_stacked = glider_stacked.where(glider_stacked.distance>0.0,
        #                                      drop=True)
        #glider_stacked = glider_stacked.dropna('z')

        y_range=[0,1000]

        x_range=[0,5]
        # plot sampling distance bar chart
        print (glider_stacked.distance.values)
        axs[0].hist2d(glider_stacked.distance, glider_stacked.ctd_depth,
                      bins=[100,1000], range=[x_range,y_range],
                      norm=matplotlib.colors.LogNorm(),
                      vmin=1, vmax=500)
        axs[0].set_ylabel('Depth (m)')
        axs[0].set_xlabel('Distance Between Samples\n(km)')

        x_range=[0,5]
        # plot sampling rate bar chart
        p = axs[1].hist2d(glider_stacked.dt, glider_stacked.ctd_depth,
                          bins=[100,1000], range=[x_range,y_range],
                          norm=matplotlib.colors.LogNorm(),
                          vmin=1, vmax=500)
        axs[1].set_xlabel('Time Between Samples\n(Hours)')
        axs[1].yaxis.set_ticklabels([])

        axs[0].invert_yaxis()
        axs[1].invert_yaxis()
      
        # colour bar
        print (p[-1])
        pos = axs[1].get_position()
        cbar_ax = fig.add_axes([0.85, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p[-1], cax=cbar_ax, orientation='vertical')
        cbar.ax.text(5.8, 0.5, 'Number of Samples', fontsize=8, rotation=90,
                     transform=cbar.ax.transAxes, va='center', ha='right')
        plt.savefig('glider_sampling_frequencies.png', dpi=600)

        
glider = raw_glider_plotting()
glider.plot_sampling_rates()
