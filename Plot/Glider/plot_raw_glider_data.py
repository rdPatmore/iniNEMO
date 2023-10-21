import xarray as xr
import config
import matplotlib.pyplot as plt
import numpy as np
import glidertools as gt
import matplotlib

matplotlib.rcParams.update({'font.size': 8})

class raw_glider_sampling(object):
    '''
    process glider data to find time and depth sampling rates
    '''

    def __init__(self, fn='merged_raw.nc'):
        self.data_path = config.root() + 'Giddy_2020/'
        self.glider = xr.open_dataset(self.data_path + fn)

        # reduce to single time coordinate
        #self.glider = self.glider.drop('ctd_time')
        if list(self.glider.dims)[0] == 'distance':
            self.glider = self.glider.swap_dims({'distance':'ctd_data_point'})
        
        if 'lon' in list(self.glider.coords): 
            self.glider = self.glider.rename({'lon': 'longitude',
                                              'lat': 'latitude'})

    def interpolate_to_uniform_depths(self):
        ''' interpolate raw glider data to uniform 1 m grid in vertical '''

        # change time units for interpolation 
        timedelta = self.glider.ctd_time_dt64 \
                  - np.datetime64('1970-01-01 00:00:00')
        self.glider['ctd_time'] = timedelta.astype(np.int64)/ 1e9

        #uniform_distance = np.arange(0, self.glider.distance.max(),1000)

        glider_uniform_i = []
        # interpolate to 1 m vertical grid
        for (label, group) in self.glider.groupby('dives'):
            if group.sizes['ctd_data_point'] < 2:
                continue
            group = group.swap_dims({'ctd_data_point': 'ctd_depth'})

            # remove duplicate index values
            _, index = np.unique(group['ctd_depth'], return_index=True)
            group = group.isel(ctd_depth=index)

            # interpolate - 1 m 
            depth_uniform = group.interp(ctd_depth=np.arange(0.0,999.0,1))

            # concatenations is along dimension not coord
            #uniform = depth_uniform.swap_dims({'ctd_depth':'dives'})
            depth_uniform = depth_uniform.drop('dives')
            uniform = depth_uniform.expand_dims(dives=[label])
            glider_uniform_i.append(uniform)

        self.glider_uniform = xr.concat(glider_uniform_i, dim='dives')

    def get_sampling_distance(self):
        ''' caluclates distance between samples '''

        distance_k = []
        for (label, group) in self.glider_uniform.groupby('ctd_depth'):
            distance = xr.DataArray(
                                gt.utils.distance(group.longitude,
                                                  group.latitude),
                                                  dims='dives')/1000 # km
            distance_k.append(distance)
        self.glider_uniform['distance'] = xr.concat(distance_k, dim='ctd_depth')

    def get_sampling_rates(self):
        ''' calculates time between samples '''
        
        time_i = []
        for (label, group) in self.glider_uniform.groupby('ctd_depth'):
            dt = group.ctd_time.diff('dives')/3600 # hours
            dt = dt.where(dt > 0, drop=True)
            print (dt.dives)
            #dt = dt.pad(ctd_data_point=(0,1))
            #dt = dt.swap_dims({'dives':'ctd_time'})
            time_i.append(dt)
        self.glider_uniform['dt'] = xr.concat(time_i, dim='ctd_depth')

    def get_sampling_freqencies(self):
        '''
        Return sampling distance and times
    
        Interpolates paths to uniform depths and finds distance and times
        between vertical profiles
        '''

        # calcs
        self.interpolate_to_uniform_depths()
        self.get_sampling_distance()
        self.get_sampling_rates()
 
        # load for faster plotting
        self.glider_uniform = self.glider_uniform.load()

class plot_sampling_frequencies(object):

    def sub_set(self, ds, token):
        '''
        Subset by dive number
     
        arguments
        ---------
        token: flag for removing dive or climb - 0.0 for climb, 0.5 for dive
        '''

        print (ds.dt.values)
        remove_index = ds.dives % 1
        dsn = ds.assign_coords({'remove_index': remove_index})
        dsn = dsn.swap_dims({'dives':'remove_index'})
        #ds = ds.where(remove_index!=token, drop=True)
        dsn = dsn.sel(remove_index=token)
        dsn = dsn.swap_dims({'remove_index':'dives'})
        dsn = dsn.drop('remove_index')

        return dsn

    def plot_sampling_frequencies_at_10m(self, paths):
        '''
        plot histogram of time and depth between samples.

        arguments
        ---------
        paths: list of paths to be plotted
        '''

        def render_path(path, c):

            def plot_hist(ds, row, c, x_r, y_r):
                bins = 100

                # plot sampling distance bar chart
                axs[row,0].hist(ds.distance, bins=bins, range=x_r,
                                color=c, alpha=0.5)

                # plot sampling rate bar chart
                print (ds.dt.values)
                p = axs[row,1].hist(ds.dt, bins=bins, range=x_r,
                                    color=c, alpha=0.5)
                return p

            # get sampling rates
            g = raw_glider_sampling(path)
            g.get_sampling_freqencies()

            # subset to 10 m depth
            g.glider_uniform = g.glider_uniform.sel(ctd_depth=10, 
                                                            method='nearest')
            # remove climbs
            glider_dive = self.sub_set(g.glider_uniform, 0.0)
            # remove dives
            glider_climb = self.sub_set(g.glider_uniform, 0.5)

            glider_dive = glider_dive.fillna(-1.0)
            glider_climb = glider_climb.fillna(-1.0)

            y_range=[0,1000]
            x_range=[0,6]
            pd = plot_hist(glider_dive, 0, c, x_range, y_range)
            pc = plot_hist(glider_climb, 1, c, x_range, y_range)

           # para = raw_glider_sampling('artificial_straight_line_transects.nc')

        # plot prep
        fig, axs = plt.subplots(2, 2, figsize=(4.5,4))
        plt.subplots_adjust(right=0.83, top=0.98, bottom=0.15, wspace=0.05,
                            hspace=0.06)

        colours = ['k', 'r']
        for i, path in enumerate(paths):
            render_path(path, colours[i])

        axs[0,0].set_ylabel('Count')
        axs[1,0].set_ylabel('Count')
        axs[1,0].set_xlabel('Distance Between Samples\n(km)')
        axs[1,1].set_xlabel('Time Between Samples\n(Hours)')
        axs[0,1].yaxis.set_ticklabels([])
        axs[1,1].yaxis.set_ticklabels([])
        axs[0,0].xaxis.set_ticklabels([])
        axs[0,1].xaxis.set_ticklabels([])

        for ax in axs[0]:
            ax.text(0.95, 0.95, 'climb -> dive', transform=ax.transAxes,
                    bbox=dict(facecolor='lightgrey', alpha=1.0,
                              edgecolor='None'), ha='right', va='top',
                    fontsize=6)
        for ax in axs[1]:
            ax.text(0.95, 0.05, 'dive -> climb', transform=ax.transAxes,
                    bbox=dict(facecolor='lightgrey', alpha=1.0,
                              edgecolor='None'), ha='right', va='bottom',
                    fontsize=6)
        plt.show()

    def plot_sampling_rates(self):
        ''' Plot raw glider sampling rates. Spatial and temporal ''' 

        # plot prep
        fig, axs = plt.subplots(2, 2, figsize=(4.5,4))
        plt.subplots_adjust(right=0.83, top=0.98, bottom=0.15, wspace=0.05,
                            hspace=0.06)

        # get glider sampling rates
        g = raw_glider_sampling()
        g.get_sampling_freqencies()

        glider_dive = self.sub_set(g.glider_uniform, 0.0)  # remove climbs
        glider_climb = self.sub_set(g.glider_uniform, 0.5) # remove dives

        glider_stacked_d = glider_dive.stack(z=('dive','ctd_depth'))
        glider_stacked_d = glider_stacked_d.fillna(0.0)

        glider_stacked_c = glider_climb.stack(z=('dive','ctd_depth'))
        glider_stacked_c = glider_stacked_c.fillna(0.0)
        #glider_stacked = glider_stacked.where(glider_stacked.distance>0.0,
        #                                      drop=True)
        #glider_stacked = glider_stacked.dropna('z')

        y_range=[0,1000]
        x_range=[0,5]

        def plot_hist(ds, row, cmap):
            # plot sampling distance bar chart
            axs[row,0].hist2d(ds.distance, ds.ctd_depth,
                          bins=[100,1000], range=[x_range,y_range],
                          norm=matplotlib.colors.LogNorm(), cmap=cmap,
                          vmin=1, vmax=500)

            # plot sampling rate bar chart
            p = axs[row,1].hist2d(ds.dt, ds.ctd_depth,
                              bins=[100,1000], range=[x_range,y_range],
                              norm=matplotlib.colors.LogNorm(), cmap=cmap,
                              vmin=1, vmax=500)
            return p

        pd = plot_hist(glider_stacked_d, 0, plt.cm.inferno)
        pc = plot_hist(glider_stacked_c, 1, plt.cm.inferno)

        axs[0,0].set_ylabel('Depth [m]')
        axs[1,0].set_ylabel('Depth [m]')
        axs[1,0].set_xlabel('Distance Between Samples\n[km]')
        axs[1,1].set_xlabel('Time Between Samples\n[Hours]')
        axs[0,1].yaxis.set_ticklabels([])
        axs[1,1].yaxis.set_ticklabels([])
        axs[0,0].xaxis.set_ticklabels([])
        axs[0,1].xaxis.set_ticklabels([])

        for ax in axs.ravel():
            ax.invert_yaxis()
      
        # colour bar
        pos0 = axs[0,1].get_position()
        pos1 = axs[1,1].get_position()
        cbar_ax = fig.add_axes([0.85, pos1.y0, 0.02, pos0.y1 - pos1.y0])
        cbar = fig.colorbar(pc[-1], cax=cbar_ax, orientation='vertical')
        cbar.ax.text(5.8, 0.5, 'Number of Samples', fontsize=8, rotation=90,
                     transform=cbar.ax.transAxes, va='center', ha='right')

        for ax in axs[0]:
            ax.text(0.95, 0.95, 'climb -> dive', transform=ax.transAxes,
                    bbox=dict(facecolor='lightgrey', alpha=1.0,
                              edgecolor='None'), ha='right', va='top',
                    fontsize=6)
        for ax in axs[1]:
            ax.text(0.95, 0.05, 'dive -> climb', transform=ax.transAxes,
                    bbox=dict(facecolor='lightgrey', alpha=1.0,
                              edgecolor='None'), ha='right', va='bottom',
                    fontsize=6)

        plt.savefig('glider_sampling_frequencies.png', dpi=1200)

        
paths = ['merged_raw.nc', 'artificial_straight_line_transects.nc']
#paths = ['artificial_straight_line_transects.nc']
p = plot_sampling_frequencies()
p.plot_sampling_frequencies_at_10m(paths)
