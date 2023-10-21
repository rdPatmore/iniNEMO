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

    arguments
    ---------
    single_depth: optional argument to slice depth level
    '''

    def __init__(self, fn='merged_raw.nc', single_depth=None):
        self.data_path = config.root() + 'Giddy_2020/'
        self.glider = xr.open_dataset(self.data_path + fn)

        # reduce to single time coordinate
        #self.glider = self.glider.drop('ctd_time')
        if list(self.glider.dims)[0] == 'distance':
            self.glider = self.glider.swap_dims({'distance':'ctd_data_point'})
        
        if 'lon' in list(self.glider.coords): 
            self.glider = self.glider.rename({'lon': 'longitude',
                                              'lat': 'latitude'})
        # assign depth slice choice
        self.single_depth = single_depth

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

            # select interpolation choice
            if self.single_depth:
                depths = [self.single_depth]
            else:
                depths = np.arange(1.0,999.0,1)

            # interpolate 
            depth_uniform = group.interp(ctd_depth=depths)

            # concatenations is along dimension not coord
            #uniform = depth_uniform.swap_dims({'ctd_depth':'dives'})
            depth_uniform = depth_uniform.drop('dives')
            uniform = depth_uniform.expand_dims(dives=[label])
            glider_uniform_i.append(uniform)

        self.g_uniform = xr.concat(glider_uniform_i, dim='dives')

    def get_sampling_distance(self):
        ''' caluclates distance between samples '''

        def get_dx(ds):
            dx = xr.DataArray(gt.utils.distance(ds.longitude, ds.latitude),
                                                  dims='dives')/1000 # km
            return dx 
        
        self.g_uniform['dx'] = self.g_uniform.groupby('ctd_depth').map(get_dx)

    def get_sampling_rates(self):
        ''' calculates time between samples '''
        
        def get_dt(ds):
            dt = ds.ctd_time.diff('dives')/3600 # hours
            dt = dt.where(dt > 0, drop=True)

            return dt

        self.g_uniform['dt'] = self.g_uniform.groupby('ctd_depth').map(get_dt)

    def get_sampling_freqencies(self):
        '''
        Return sampling distance and times
    
        Interpolates paths to uniform depths and finds distance and times
        between vertical profiles

        '''

        # interplolate to uniform depth
        self.interpolate_to_uniform_depths()
        
        # load for faster plotting
        self.g_uniform = self.g_uniform.load()

        # calculate space and time between samples
        self.get_sampling_distance()
        self.get_sampling_rates()
 
        # remove ctd_depth (when using depth slice)
        self.g_uniform = self.g_uniform.squeeze()

        # remove distances < 1 km
        self.g_uniform = self.g_uniform.where(self.g_uniform.dx > 1)

class plot_sampling_frequencies(object):

    def subset(self, ds, token):
        '''
        Subset by dive number
     
        arguments
        ---------
        token: flag for removing dive or climb - 0.0 for climb, 0.5 for dive
        '''

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

            def plot_hist(ds, c, x_r):
                bins = 100

                # plot sampling distance bar chart
                axs[0].hist(ds.dx, bins=bins, range=x_r,
                                color=c, alpha=0.5)

                # plot sampling rate bar chart
                p = axs[1].hist(ds.dt, bins=bins, range=x_r,
                                    color=c, alpha=0.5)
                return p

            # get sampling rates
            g = raw_glider_sampling(path, single_depth=10)
            g.get_sampling_freqencies()

            # subset by climb/dive
            #glider_dive = self.subset(g.g_uniform, 0.0) # remove climbs
            #glider_climb = self.subset(g.g_uniform, 0.5) # remove dives

            x_range=[1,7]
            ph = plot_hist(g.g_uniform, c, x_range)
            return ph

        # plot prep
        fig, axs = plt.subplots(1, 2, figsize=(5.5,3))
        plt.subplots_adjust(left=0.1, right=0.98, top=0.90, bottom=0.15,
                            wspace=0.05)

        # render
        colours = ['k', '#f18b00']
        p = []
        for i, path in enumerate(paths):
            p.append(render_path(path, colours[i]))

        print (p)

        # set axes
        axs[0].set_ylabel('Count')
        axs[0].set_xlabel('Distance Between Samples (km)')
        axs[1].set_xlabel('Time Between Samples (Hours)')
        axs[1].yaxis.set_ticklabels([])

        fig.legend(['bow-tie', 'straight path'],
                    loc='lower center', bbox_to_anchor=(0.5, 0.91), 
                    ncol=len(paths), fontsize=8)

        # show
        plt.savefig('straight_line_bow_tie_sampling_rate_hist.png', dpi=1200)

    def plot_sampling_rates(self):
        ''' Plot raw glider sampling rates. Spatial and temporal ''' 

        # plot prep
        fig, axs = plt.subplots(2, 2, figsize=(4.5,4))
        plt.subplots_adjust(right=0.83, top=0.98, bottom=0.15, wspace=0.05,
                            hspace=0.06)

        # get glider sampling rates
        g = raw_glider_sampling()
        g.get_sampling_freqencies()

        glider_dive = self.sub_set(g.g_uniform, 0.0)  # remove climbs
        glider_climb = self.sub_set(g.g_uniform, 0.5) # remove dives

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
