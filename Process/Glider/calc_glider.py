import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
import dask

matplotlib.rcParams.update({'font.size': 8})

class glider_nemo(object):

    def __init__(self, case, offset=False):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'

    def load_glider_nemo(self):
        self.glider_nemo = xr.open_dataset(self.data_path + 'glider_nemo_bg.nc',
                                    chunks={'time_counter':1})

    def load_nemo_bg(self):
        ''' load *subset to giddy* extent buoyancy gradient of model '''
        self.nemo = xr.open_dataset(self.data_path + 'nemo_bg.nc',
                                    chunks={'time_counter':1})

    def load_giddy(self):
        self.giddy = xr.open_dataset(self.root + 
                     'Giddy_2020/sg643_grid_density_surfaces.nc',
                      chunks={'distance':1})

    def load_glider_uniform_bg(self):
        ''' load glider sampled model data on a unifrom grid'''
        self.glider_uniform = xr.open_dataset(self.data_path + 
                                   'glider_uniform_raw_path.nc',
                                   chunks={'ctd_depth':1})

    def sub_sample_nemo(self, glider_path, save=True):
        ''' restrict time and space of nemo to glider path '''

        bg = xr.open_dataset(self.data_path + 
                                  'buoyancy_gradients.nc',
                                  chunks={'time_counter':1,'deptht':1})

        # restrict nemo to glider time
        bg_cut_grid = self.subset_times_to_obs(bg)

        bounds = [glider_path.lon.min(), glider_path.lon.max(),
                  glider_path.lat.min(), glider_path.lat.max()]

        bg_cut_grid = self.cut_grid_to_obs_patch(bg_cut_grid, bounds)
        bg_cut_grid = bg_cut_grid.drop(['nav_lon', 'nav_lat'])

        self.nemo_bg_ml = bg_cut_grid.load()
  
        if save:
            self.nemo_bg_ml.to_netcdf(self.data_path + 'nemo_bg.nc',
                                      unlimited_dims='time_counter')
        
    def cut_grid_to_obs_patch(self, data, bounds):
        # xy -> latlon
        data = data.assign_coords({'lon': data.nav_lon.isel(y=0),
                                   'lat': data.nav_lat.isel(x=0)})
        data = data.swap_dims({'x':'lon', 'y':'lat'})

        data = data.sel(lon=slice(bounds[0],bounds[1]),
                        lat=slice(bounds[2],bounds[3]))
        return data

    def subset_times_to_obs(self, data, start_month='01'):
        ''' subset model according to glider time '''
        self.load_giddy()
        time = self.giddy.time.isel(density=50)
        time_delta = np.datetime64('2018-01-01') - np.datetime64('2012-01-01')
        time_span = time - time_delta
        data = data.sel(time_counter=slice(time_span[1],time_span[-1]))
        return data

    def sample_times_interp(self, data, start_month='01'):
        ''' sample time according to glider time '''
        self.load_giddy()
        time = self.giddy.time.isel(density=50)
        time_delta = np.datetime64('2018-01-01') - np.datetime64('2012-01-01')
        time_span = time - time_delta
        data = data.interp(time_counter=time_span.values, method='linear')
        data['time_counter'] = data.time_counter.astype('int64')/1e9
        data.time_counter.attrs['units'] = 'seconds since 1970-01-01'
        return data

    def get_randomly_sampled_buoyancy_gradient_along_glider_path(self,
                                                  save=False, samples=1):

        model = mo.model(self.case)
        random_samples = []
        for sample in range(samples):
            print (sample)
            model.interp_to_obs_path(random_offset=True)

            dx = 1000
            dCT_dx = model.glider_nemo.cons_temp.diff('distance') / dx
            dAS_dx = model.glider_nemo.abs_sal.diff('distance') / dx
            dCT_dx = dCT_dx.pad(glider_path=(0,1))
            dAS_dx = dAS_dx.pad(glider_path=(0,1))
            alpha = model.glider_nemo.alpha
            beta = model.glider_nemo.beta
            g=9.81
            bg = g * ( alpha * dCT_dx - beta * dAS_dx ) 

            # restrict to mixed layer depth
            bg = bg.where(bg.deptht < model.glider_nemo.mldr10_3, drop=True)

            bg.name = 'bg'
            random_samples.append(bg)
            print (bg)
        self.bg_rand = xr.concat(random_samples, dim='random_samples')

        if save:
            self.bg_rand.to_netcdf(self.data_path +
                                  'random_sampled_glider_bg.nc')

    def subset_nemo_to_randomised_nemo_paths(self, save=False, samples=1):

        glider_random_path_bg = xr.open_dataarray(self.data_path + 
                                     'random_sampled_glider_bg.nc')

        nemo_samples=[]
        for i in range(samples):
            print (i)
            self.sub_sample_nemo(glider_random_path_bg.isel(random_samples=i))
            print (self.nemo_bg_ml.sizes)
            #self.nemo_bg_ml = self.nemo_bg_ml.assign_coords(
            #                  {'x':range(self.nemo_bg_ml.sizes['lon']),
            #                   'y':range(self.nemo_bg_ml.sizes['lat'])})
            #self.nemo_bg_ml = self.nemo_bg_ml.swap_dims({'lon':'x','lat':'y'})
            self.nemo_bg_ml = self.nemo_bg_ml.reset_index(['lat','lon'])
            print (self.nemo_bg_ml)
            self.nemo_bg_ml = self.nemo_bg_ml.reset_coords(['lat_','lon_'])
            nemo_samples.append(self.nemo_bg_ml)
        nemo_rand = xr.concat(nemo_samples, coords='minimal',
                               dim='random_samples')
        if save:
            nemo_rand.to_netcdf(self.data_path +
                                      'nemo_subset_rand_set.nc')

    def combine_random_bg_nemo_and_glider(self):
        glider_rand = xr.open_dataarray(self.data_path + 
                                     'random_sampled_glider_bg.nc')
        nemo_rand = xr.open_dataarray(self.data_path + 
                                      'nemo_subset_rand_set.nc')
        random_sampled_nemo_and_glider = xr.merge([glider_rand, nemo_rand])
        random_sampled_nemo_and_glider.to_netcdf(self.data_path +
                                           'random_sampled_nemo_glider_pair.nc')
        
    def histogram_buoyancy_gradient_ensemble(self):
        ''' 
        plot histogram of buoyancy gradient comparision for different
        sampling methods
        using set of random glider samples
        '''
        plt.figure(figsize=(4.5,4.5))
        samples = 10

        #self.load_glider_uniform_bg()
        #glider_abs_bg = np.abs(self.glider_uniform.bg_x_ml)
        glider_bg_hist = glider_abs_bg.where(glider_abs_bg<2e-8, drop=True)


        for i in range(samples):
            glider_bg_hist_sample = glider_bg_hist.isel(random_samples=i)

            abs_bgx = np.abs(self.nemo_bg_ml.bgx)
            bgx_hist = abs_bgx.where(abs_bgx<2e-8, drop=True)
            abs_bgy = np.abs(self.nemo.bgy)
            bgy_hist = abs_bgy.where(abs_bgy<2e-8, drop=True)

            bgx_hist.plot.hist(bins=100, density=True, alpha=0.3,
                               label='nemo bgx',
                               fill=False, edgecolor='gray', histtype='step')
            bgy_hist.plot.hist(bins=100, density=True, alpha=0.3,
                               label='nemo bgy',
                               fill=False, edgecolor='black', histtype='step')

            glider_bg_hist_sample.plot.hist(bins=100, density=True, alpha=0.3,
                                 label='glider bg', fill=False, edgecolor='red',
                                 histtype='step')
        plt.legend()
        plt.xlabel
        #plt.hist(bgx_hist, 50)
        #plt.savefig('Plots/buoyany_gradient_histogram.png', dpi=300)
        plt.show()

    def histogram_buoyancy_gradient(self):
        ''' 
        plot histogram of buoyancy gradient comparision of glider sample
        against nemo gradients
        nemo gradients are subsampled to glider area and time 
        '''
        plt.figure(figsize=(4.5,4.0))

        self.load_glider_uniform_bg()
        glider_abs_bg = np.abs(self.glider_uniform.b_x_ml)
        glider_bg_hist = glider_abs_bg.where(glider_abs_bg<2e-8, drop=True)

        self.load_nemo_bg()
        abs_bgx = np.abs(self.nemo.bgx)
        bgx_hist = abs_bgx.where(abs_bgx<2e-8, drop=True)
        abs_bgy = np.abs(self.nemo.bgy)
        bgy_hist = abs_bgy.where(abs_bgy<2e-8, drop=True)

        abs_bgx.plot.hist(bins=100, density=True, alpha=0.3,
                           label='nemo bgx', range=(0,2e-8),
                           fill=False, edgecolor='gray', histtype='step')
        abs_bgy.plot.hist(bins=100, density=True, alpha=0.3,
                           label='nemo bgy', range=(0,2e-8),
                           fill=False, edgecolor='black', histtype='step')

        glider_abs_bg.plot.hist(bins=100, density=True, alpha=0.3,
                             label='glider bg', fill=False, edgecolor='red',
                             range=(0,2e-8), histtype='step')
        plt.legend()
        plt.xlabel('Buoyancy Gradient')
        plt.ylabel('PDF')
        plt.title('')
        plt.savefig('Plots/buoyany_gradient_histogram.png', dpi=600)

    def plot_glider_path(self, ax):
        self.load_giddy()

        lon = self.giddy.lon.isel(density=50)
        lat = self.giddy.lat.isel(density=50)
        dist = self.giddy.distance#.isel(density=50)

        colour = plt.cm.plasma
        plt.scatter(lon, lat, c=dist, cmap=colour, s=3)

    def plot_max_bg(self, ax):
        self.load_nemo_bg()
        self.nemo = self.sample_times(self.nemo)

        bgx_hist = self.nemo.bgx.where(self.nemo.bgx<2e-8, drop=True)
        bgy_hist = self.nemo.bgy.where(self.nemo.bgy<2e-8, drop=True)
        max_abs_bgx = np.abs(bgx_hist).max(['time_counter','deptht'])
        max_abs_bgy = np.abs(bgy_hist).max(['time_counter','deptht'])
        max_abs_bg = xr.where(max_abs_bgx<max_abs_bgy, max_abs_bgy, max_abs_bgx)
 
        p = ax.pcolor(self.nemo.lon, self.nemo.lat, max_abs_bg,
                      cmap=plt.cm.gray, shading='nearest')
        return p
        
    def plot_path_and_max_bg(self): 
        ''' 
        Plot the maximum buoyancy gradined for each lat-lon point
        then overlay glider path.
        '''
   
        fig, axs = plt.subplots(1)

        p = self.plot_max_bg(axs)
        self.plot_glider_path(axs)

        cbar = plt.colorbar(p)
        cbar.set_label('maximum buoyancy gradient')

        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.title('glider path')

        plt.savefig('Plots/giddy_path.png', dpi=300)

    def movie_bg_grad_and_glider_path(self):
        
        self.load_nemo_bg()
        self.nemo = self.sample_times(self.nemo)

        self.load_giddy()
        self.giddy = self.giddy.isel(density=50, distance=slice(1,None))
        self.giddy['time_counter'] = self.nemo.time_counter
        self.giddy = self.giddy.drop_dims('time_counter')
        self.giddy = self.giddy.swap_dims({'distance':'time_counter'})

        self.nemo = self.nemo.isel(time_counter=slice(None,None,20))
        self.giddy = self.giddy.isel(time_counter=slice(None,None,20))

        bgx = self.nemo.bgx.where(self.nemo.bgx<2e-8, drop=True).mean('deptht')
        bgy = self.nemo.bgy.where(self.nemo.bgy<2e-8, drop=True).mean('deptht')

        fig, axs = plt.subplots(2,1, figsize=(3.2,6))
        
        vmin = min(bgx.min(), bgy.min()).values
        vmax = max(bgx.max(), bgy.max()).values

        p0 = axs[0].pcolormesh(self.nemo.lon, self.nemo.lat,
                               bgx.isel(time_counter=0), cmap=plt.cm.gray,
                               vmin=vmin, vmax=vmax, shading='nearest')

        p1 = axs[1].pcolormesh(self.nemo.lon, self.nemo.lat,
                               bgy.isel(time_counter=0), cmap=plt.cm.gray,
                               vmin=vmin, vmax=vmax, shading='nearest')
        fig.colorbar(p0, ax=axs[0])
        fig.colorbar(p1, ax=axs[1])
 
        axs[0].set_aspect('equal')
        axs[1].set_aspect('equal')
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1e3)

        
        frames = len(self.nemo.time_counter) - 1
        
        def animate(i):
            print (i + 1, ' / ', frames)

            axs[0].pcolormesh(self.nemo.lon, self.nemo.lat,
                                   bgx.isel(time_counter=i), cmap=plt.cm.gray,
                               vmin=vmin, vmax=vmax, shading='nearest')
            axs[1].pcolormesh(self.nemo.lon, self.nemo.lat,
                                   bgy.isel(time_counter=i), cmap=plt.cm.gray,
                               vmin=vmin, vmax=vmax, shading='nearest')

            colour = plt.cm.plasma
            axs[0].scatter(self.giddy.lon[:i], self.giddy.lat[:i],
                           c=self.giddy.distance[:i], cmap=colour, s=3)
            axs[1].scatter(self.giddy.lon[:i], self.giddy.lat[:i],
                           c=self.giddy.distance[:i], cmap=colour, s=3)

            axs[0].set_aspect('equal')
            axs[1].set_aspect('equal')
        
        ani = animation.FuncAnimation(
              fig, animate, frames=frames, blit=False, save_count=50)
        
        # To save the animation, use e.g.
        #
        #
        # or
        #
        # writer = animation.FFMpegWriter(
        #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("movie.mp4", writer=writer, dpi=300)


m = glider_nemo('EXP02')
m.load_giddy()
print (m.giddy)
#m.histogram_buoyancy_gradient()


# this is for creating nemo_bg for bg plotting
# histogram buoyancy gradient!
#m.load_glider_uniform_bg()
#m.sub_sample_nemo(m.glider_uniform)


#m.get_randomly_sampled_buoyancy_gradient_along_glider_path(save=True,samples=10)
#m.subset_nemo_to_randomised_nemo_paths(save=True, samples=10)
#m.combine_random_bg_nemo_and_glider()
#m.histogram_buoyancy_gradient()
#m.movie_bg_grad_and_glider_path()
def plot_path():
    m = glider_nemo('EXP03')
    #m.sub_sample_nemo()
    m.plot_path_and_max_bg() 

def plot_histogram():
    m = glider_nemo('EXP02')
    #m.load_glider_nemo()
    #m.sub_sample_nemo()
    m.histogram_buoyancy_gradient()


    #m.get_randomly_sampled_buoyancy_gradient_along_glider_path()
