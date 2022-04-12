import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import dask
import matplotlib
from get_transects import get_transects

matplotlib.rcParams.update({'font.size': 8})

class glider_path_geometry(object):
    '''
    for ploting results of adjusting the geometry of the glider path
    '''

    def __init__(self, case, offset=False):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'

        self.hist_range = (0,2e-8)
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

    def get_glider_sample_lims(self):
        ''' find the x and y extend of each glider sample '''

        self.get_glider_samples()

        x0_set, x1_set, y0_set, y1_set = [], [], [], []
        for sample in random:
            # limits for each sample of each sample set
            sample_set = self.samples.isel(sample=sample)#.b_x_ml
            x0_set.append(
                 sample_set.lon.min(dim='distance').expand_dims('sample_set'))
            x1_set.append(
                 sample_set.lon.max(dim='distance').expand_dims('sample_set'))
            y0_set.append(
                 sample_set.lat.min(dim='distance').expand_dims('sample_set'))
            y1_set.append(
                 sample_set.lat.max(dim='distance').expand_dims('sample_set'))

        x0_set = xr.concat(x0_set, dim='sample_set')
        x1_set = xr.concat(x1_set, dim='sample_set')
        y0_set = xr.concat(y0_set, dim='sample_set')
        y1_set = xr.concat(y1_set, dim='sample_set')

        x0_set.name = 'x0'
        x1_set.name = 'x1'
        y0_set.name = 'y0'
        y1_set.name = 'y1'
 
        self.latlon_lims = xr.merge([x0_set, x1_set, y0_set, y1_set])

    def get_model_buoyancy_gradients_patch(self):
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


    def get_glider_samples(self):
        ''' get set of 100 glider samples '''
        def expand_sample_dim(ds):
            ds['lon_offset'] = ds.attrs['lon_offset']
            ds['lat_offset'] = ds.attrs['lat_offset']
            ds = ds.set_coords(['lon_offset','lat_offset','time_counter'])
            da = ds['b_x_ml']
            return da

        # load samples
        prep = 'GliderRandomSampling/glider_uniform_interp_1000_'
        sample_list = [self.data_path + prep + 
                       str(i).zfill(2) + '.nc' for i in range(100)]
        self.samples = xr.open_mfdataset(sample_list, 
                                     combine='nested', concat_dim='sample',
                                     preprocess=expand_sample_dim).load()

        # select depth
        self.samples = self.samples.sel(ctd_depth=10, method='nearest')

        # get transects
        sample_list = []
        for i in range(self.samples.sample.size):
            print ('sample: ', i)
            var10 = self.samples.isel(sample=i)
            sample_transect = get_transects(var10, offset=True)
            sample_list.append(sample_transect)
        self.samples=xr.concat(sample_list, dim='sample')

        # set time to float for averaging
        float_time = self.samples.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        self.samples['time_counter'] = clean_float_time
 
        # absolute value of buoyancy gradients
        self.samples = np.abs(self.samples)


    def get_hist_stats(self, hist_set, bins):    
        ''' get mean, lower and upper deciles of group of histograms '''
        bin_centers = (bins[:-1] + bins[1:]) / 2
        hist_array = xr.DataArray(hist_set, dims=('sets', 'bin_centers'), 
                          coords={'bin_centers': bin_centers,
                                  'bin_left':  (['bin_centers'], bins[:-1]),
                                  'bin_right': (['bin_centers'], bins[1:])})
        mean = hist_array.mean('sets')
        quant = hist_array.quantile([0.05,0.1,0.25,0.75,0.9,0.95],'sets')
        mean.name = 'hist_mean'
        quant.name = 'hist_quant'
        hist_stats = xr.merge([mean,quant])
        return hist_stats

    def calc_hist(self, da):
        ''' calculate histogram from dataset '''

        set_size = da.sizes['sample']
        # get random samples
        random = np.random.randint(set_size, size=(set_size))

        hists = []
        for i, sample in enumerate(random):
            sample = da.isel(sample=sample)
            #set_stacked = sample_set.stack(z=('distance','sample'))
            hist, bins = np.histogram(sample.dropna('distance', how='all'),
                                range=self.hist_range, density=True, bins=20)
            hists.append(hist)
        hist_set = self.get_hist_stats(hists, bins)
        return hist_set

    def get_vertex_sets_all(self):
        ''' 
        save a file with all combinations of vertex choices
        add time coordinate for sub-sampling full model data afterwards

        remove sides of glider path

        left n-s transect  [2]
        right n-s transect [0]
        ne -> sw transect  [1]
        nw -> se transect  [3]
        two sides          [0,2]
        cross              [1,3]

        nb: need to check this
        '''

        # get sample set
        self.get_glider_samples()

        # reduce vertices and get hist
        vertex_list = [[3],[1],[2],[0],[1,3],[0,2]]
        ver_label = ['left','right','diag_ur','diag_ul','parallel','cross']
        hist_list=[]
        for i,choice in enumerate(vertex_list):
            ver_sampled = self.samples.where(self.samples.vertex.isin(choice),
                                             drop=True)
            hist = self.calc_hist(ver_sampled).expand_dims(
                                                  vertex_choice=[ver_label[i]])
            hist_list.append(hist)
        hist_ds= xr.concat(hist_list, dim='vertex_choice')

        hist_ds.to_netcdf(self.data_path +  '/BgGliderSamples' + 
                  '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_vertex_hist.nc')

 ######  11/04/22 ####
 ###### comments need to be veted before removal ####
#    def add_sample(self, c='green', label='', path_append=''):
#        '''
#        add sample set of means and std to histogram
#        n = sample size
#        '''
# 
#        self.sample = xr.open_dataset(self.data_path + 
#               'GliderRandomSampling/glider_uniform_' + path_append + '_00.nc')
#        bx = np.abs(self.sample.b_x_ml)
#        bx = bx.where(bx < 2e-8, drop=True)
#
#        bx_stacked = bx.stack(z=('distance','ctd_depth'))
#
#        plt.hist(bx_stacked, bins=100, density=True, alpha=0.3,
#                 label=label, fill=False, edgecolor=c,
#                 histtype='step')

    #def add_model_hist(self):
    #    '''
    #    add model buoyancy gradient of means and std to histogram
    #    '''

    #    # load buoyancy gradients       
    #    self.get_model_buoyancy_gradients()

    #    self.bg = self.bg.mean('deptht')
    #    self.bg = np.abs(self.bg)
    #    self.bg = self.bg.where(self.bg < 2e-8, drop=True)
    #    stacked_bgx = self.bg.bgx.stack(z=('time_counter','x','y'))
    #    stacked_bgy = self.bg.bgy.stack(z=('time_counter','x','y'))
    #    
    #    plt.hist(stacked_bgx, bins=100, density=True, alpha=0.3,
    #             label='model bgx', fill=False, edgecolor='gray',
    #             histtype='step', zorder=11)
    #    plt.hist(stacked_bgy, bins=100, density=True, alpha=0.3,
    #             label='model bgy', fill=False, edgecolor='black',
    #             histtype='step', zorder=11)

    def render_glider_sample_set(self, da, c='green', style='plot'):
        print (da)
        if style=='bar':
            self.ax.bar(da.bin_left, 
                    da.hist_quant.sel(quantile=0.95) -
                    da.hist_quant.sel(quantile=0.05), 
                    width=da.bin_right - da.bin_left,
                    color=c,
                    alpha=0.6,
                    bottom=da.hist_quant.sel(quantile=0.05), 
                    align='edge',
                    label='vertex: ' + str(da.vertex_choice.values))
            self.ax.scatter(da.bin_centers, da.hist_mean, c=c, s=4, zorder=10)
        if style=='plot':
            self.ax.fill_between(da.bin_centers, 
                                 da.hist_quant.sel(quantile=0.05),
                                 da.hist_quant.sel(quantile=0.95),
                                 color=c, edgecolor=None, alpha=0.2)
            self.ax.plot(da.bin_centers, da.hist_mean, c=c, lw=0.8,
                         label='vertex: ' + str(da.vertex_choice.values))
        
    def add_model_means(self, style='plot'):
        ds = xr.open_dataset(self.data_path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist.nc')
        if style=='bar':
            self.ax.hlines(ds.hist_x, ds.bin_left, ds.bin_right,
                       transform=self.ax.transData,
                       colors='black', lw=0.8, label='model_bx')
            self.ax.hlines(ds.hist_y, ds.bin_left, ds.bin_right,
                       transform=self.ax.transData,
                       colors='orange', lw=0.8, label='model_by')
        if style=='plot':
            self.ax.plot(ds.bin_centers, ds.hist_x, c='black', lw=0.8,
                         label='model bx')
            self.ax.plot(ds.bin_centers, ds.hist_y, c='red', lw=0.8,
                         label='model by')

    def plot_histogram_buoyancy_gradients_and_samples(self):
        ''' 
        plot histogram of buoyancy gradients 
        n = sample_size
        '''

        self.figure, self.ax = plt.subplots(figsize=(4.5,4.0))

        vertex = ['left','right','diag_ur','diag_ul']
        colours = ['g', 'b', 'r', 'y']

        ds = xr.open_dataset(self.data_path + '/BgGliderSamples' + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_' +
                          'vertex_hist.nc')
        print (ds)
        for i, ver in enumerate(vertex):
            da = ds.sel(vertex_choice=ver)
            print ('sample', i)
            self.render_glider_sample_set(da, c=colours[i], style='plot')
        print ('model')
        self.add_model_means(style='bar')

        self.ax.set_xlabel('Buoyancy Gradient')
        self.ax.set_ylabel('PDF')

        plt.legend()
        self.ax.set_xlim(self.hist_range[0], self.hist_range[1])
        self.ax.set_ylim(0, 3e8)
        plt.savefig(self.case + '_bg_vertex_select.png', dpi=600)

 ######  11/04/22 ####
 ###### comments need to be veted before removal ####
#    def histogram_buoyancy_gradients_and_sample(self):
#        ''' 
#        plot histogram of buoyancy gradients 
#        n = sample_size
#        '''
#
#        self.figure, self.ax = plt.subplots(figsize=(4.5,4.0))
#
#
#        self.add_sample(c='b', label='rotated', path_append='rotated_path')
#        self.add_sample(c='r', label='non-rotated', path_append='non_rotated_path')
#        self.add_model_hist()
#
#        self.ax.set_xlabel('Buoyancy Gradient')
#        self.ax.set_ylabel('PDF')
#
#        plt.legend()
#        plt.savefig('EXP02_bg_glider_rotation.png', dpi=300)

m = glider_path_geometry('EXP10')
m.plot_histogram_buoyancy_gradients_and_samples()
#m.get_vertex_sets_all()
