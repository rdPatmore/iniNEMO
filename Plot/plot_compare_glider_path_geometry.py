import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import dask
import matplotlib
from get_transects import get_transects

dask.config.set({"array.slicing.split_large_chunks": True})

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

    def get_model_buoyancy_gradients_patch_set(self, ml=False, stats=None):
        ''' restrict the model time to glider time and sample areas '''
    
        # model
        bg = xr.open_dataset(config.data_path() + self.case +
                             '/SOCHIC_PATCH_3h_20121209_20130331_bg.nc',
                             chunks='auto')
                             #chunks={'time_counter':1})
        print (bg)
        bg = np.abs(bg.sel(deptht=10, method='nearest'))

        # add lat-lon to dimensions
        bg = bg.assign_coords({'lon':(['x'], bg.nav_lon.isel(y=0)),
                               'lat':(['y'], bg.nav_lat.isel(x=0))})
        bg = bg.swap_dims({'x':'lon','y':'lat'})


        clean_float_time = self.samples.time_counter
        start = clean_float_time.min().astype('datetime64[ns]')
        end   = clean_float_time.max().astype('datetime64[ns]')

        #print (' ')
        #print (' ')
        #print ('start', start.values)
        #print ('end', end.values)
        #print (' ')
        #print (' ')
        bg = bg.sel(time_counter=slice(start,end))
        
        patch_set = []
        for (l, sample) in self.samples.groupby('sample'):
            # get limts of sample
            x0 = float(sample.lon.min())
            x1 = float(sample.lon.max())
            y0 = float(sample.lat.min())
            y1 = float(sample.lat.max())
 
            patch = bg.sel(lon=slice(x0,x1),
                           lat=slice(y0,y1)).expand_dims(sample=[l])
            if stats == 'mean':
                patch = patch.mean(['lon','lat','time_counter']).load()
            if stats == 'std':
                patch = patch.std(['lon','lat','time_counter']).load()
            patch_set.append(patch)

        self.model_patches = xr.concat(patch_set, dim='sample')#.load()#.chunk('auto')
                                                      #{'lat':10,'lon':10})

        return self.model_patches
         
    def get_sample_and_glider_diff_rotation(self, percentage=False,
                                            samples=100):
        '''
        For each rotation choice,
        get difference between mean/std bg for each sample-patch combo
        '''

        def get_diff(n, sample_type=''):
            ''' get diff between model and glider'''

            # glider stats
            self.get_glider_samples(n=n, sample_type=sample_type)
            g_mean = self.samples.mean('distance')
            g_std = self.samples.std('distance')

            # model stats
            m_mean = self.get_model_buoyancy_gradients_patch_set(stats='mean')
            m_std  = self.get_model_buoyancy_gradients_patch_set(stats='std')
            #stat_dims = ['lat','lon','time_counter']
            #m_mean = self.model_patches.mean(stat_dims).load()
            #m_std  = self.model_patches.std(stat_dims).load()

            if percentage:
                denom_m_bx = np.abs(m_mean.bx) / 100.0
                denom_m_by = np.abs(m_mean.by) / 100.0
                denom_s_bx = np.abs(m_std.bx) / 100.0
                denom_s_by = np.abs(m_std.by) / 100.0
            else: 
                denom = 1.0

            diff_x_mean = (m_mean.bx - g_mean) / denom_m_bx
            diff_y_mean = (m_mean.by - g_mean) / denom_m_by
            diff_x_std = (m_std.bx - g_std) / denom_s_bx
            diff_y_std = (m_std.by - g_std) / denom_s_by

            if sample_type != '':
                sample_type = '_' + sample_type
            diff_x_mean.name = 'diff_bx_mean' + sample_type
            diff_y_mean.name = 'diff_by_mean' + sample_type
            diff_x_std.name = 'diff_bx_std' + sample_type
            diff_y_std.name = 'diff_by_std' + sample_type

            return [diff_x_mean, diff_y_mean, diff_x_std, diff_y_std]

        non_rot = get_diff(samples)
        rot     = get_diff(samples, 'rotate')

        diff_ds = xr.merge(non_rot + rot)

        if percentage:
            f_append = '_percent'
        else: 
            f_append = ''

        diff_ds.to_netcdf(self.data_path +  '/BgGliderSamples' + 
                  '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_rotate_diff' +
                  f_append + '_' + str(samples) + '_samples.nc')

    def get_sample_and_glider_diff_vertex_set(self, percentage=False,
                                              samples=100):
        '''
        For each vertex choice,
        get difference between mean/std bg for each sample-patch combo
        '''

        # glider samples
        self.get_glider_samples(n=samples)

        # model patches
        m_mean = self.get_model_buoyancy_gradients_patch_set(stats='mean')
        m_std = self.get_model_buoyancy_gradients_patch_set(stats='std')

        # glider stats
        vertex_list = [[3],[1],[2],[0],[1,3],[0,2],[0,1,2,3]]
        ver_label = ['left','right','diag_ur','diag_ul',
                     'parallel','cross','all']
        g_std_list=[]
        g_mean_list=[]
        for i,choice in enumerate(vertex_list):
            ver_sampled = self.samples.where(self.samples.vertex.isin(choice),
                                             drop=True)
            g_mean_list.append(ver_sampled.mean('distance'))
            g_std_list.append(ver_sampled.std('distance'))

        # join along new coord
        vertex_coord = xr.DataArray(ver_label, dims='vertex_choice',
                                               name='vertex_choice')
        g_mean = xr.concat(g_mean_list, dim=vertex_coord)
        g_std = xr.concat(g_std_list, dim=vertex_coord)

        if percentage:
            denom_m_bx = np.abs(m_mean.bx) / 100.0
            denom_m_by = np.abs(m_mean.by) / 100.0
            denom_s_bx = np.abs(m_std.bx) / 100.0
            denom_s_by = np.abs(m_std.by) / 100.0
            f_append = '_percent'
        else: 
            denom = 1.0
            f_append = ''

        diff_x_mean = (m_mean.bx - g_mean) / denom_m_bx
        diff_y_mean = (m_mean.by - g_mean) / denom_m_by
        diff_x_std = (m_std.bx - g_std) / denom_s_bx
        diff_y_std = (m_std.by - g_std) / denom_s_by

        diff_x_mean.name = 'diff_bx_mean'
        diff_y_mean.name = 'diff_by_mean'
        diff_x_std.name = 'diff_bx_std'
        diff_y_std.name = 'diff_by_std'

        diff_ds = xr.merge([diff_x_mean,diff_y_mean,
                            diff_x_std,diff_y_std])

        diff_ds.to_netcdf(self.data_path +  '/BgGliderSamples' + 
                  '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_vertex_diff' +
                  f_append + '_' + str(samples) + '_samples.nc')

    def get_glider_samples(self, n=100, sample_type=''):
        ''' get set of 100 glider samples '''
        def expand_sample_dim(ds):
            ds['lon_offset'] = ds.attrs['lon_offset']
            ds['lat_offset'] = ds.attrs['lat_offset']
            ds = ds.set_coords(['lon_offset','lat_offset','time_counter'])
            da = ds['b_x_ml']
            return da

        # load samples
        prep = 'GliderRandomSampling/glider_uniform_interp_1000_'
        print (self.data_path)
        print (prep)
        print (sample_type)
        print (n)
        if sample_type != '':
            sample_type = sample_type + '_' 
        print ([str(i).zfill(2) for i in range(n)])
        sample_list = [self.data_path + prep + sample_type +
                       str(i).zfill(2) + '.nc' for i in range(n)]
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
        #print (da)
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
        #print (ds)
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

class glider_path_geometry_plotting(object):
    def __init__(self, append=''):
        self.data_path = config.data_path()
        if append == '':
            self.append = ''
        else:
            self.append='_' + append

    def plot_model_and_glider_diff_bar(self, cases, samples):
        ''' 
        bar chart of difference between model patches and gliders
        mean and std across samples
        '''

        fig, axs = plt.subplots(2,1,figsize=(4.5,4.0))

        vertex = ['left','right','diag_ur','diag_ul']
        colours = ['g', 'b', 'r', 'y']


        x_pos = np.linspace(0.5,6.5,7)
        offset = [-0.3, 0, 0.3]
        for i, case in enumerate(cases):
            path = self.data_path + case
            prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
            ds = xr.open_dataset(path + prepend +  'glider_vertex_diff' +
                                 self.append + '_percent_' + str(samples) + 
                                 '_samples.nc')

            ds_mean = ds.mean('sample')
            ds_l_quant = ds.quantile(0.05, 'sample')
            ds_u_quant = ds.quantile(0.95, 'sample')

            axs[0].bar(x_pos + offset[i], 
                       ds_u_quant.diff_bx_mean - ds_l_quant.diff_bx_mean,
                       width=0.15,
                       alpha=0.2,
                       bottom=ds_l_quant.diff_bx_mean,
                       color=colours[i],
                       tick_label=ds.vertex_choice)
            axs[0].hlines(ds_mean.diff_bx_mean,
                          x_pos + offset[i]-0.075,
                          x_pos + offset[i]+0.075,
                          lw=2)

            axs[0].bar(x_pos + offset[i]+0.15, 
                       ds_u_quant.diff_by_mean - ds_l_quant.diff_by_mean,
                       width=0.15,
                       alpha=0.2,
                       bottom=ds_l_quant.diff_by_mean,
                       color=colours[i],
                       tick_label=ds.vertex_choice)
            axs[0].hlines(ds_mean.diff_by_mean,
                          x_pos + offset[i]+0.075,
                          x_pos + offset[i]+0.225,
                          lw=2)

            axs[1].bar(x_pos + offset[i], 
                       ds_u_quant.diff_bx_std - ds_l_quant.diff_bx_std,
                       width=0.15,
                       bottom=ds_l_quant.diff_bx_std,
                       color=colours[i],
                       alpha=0.2,
                       tick_label=ds.vertex_choice)
            axs[1].hlines(ds_mean.diff_bx_std,
                          x_pos + offset[i]-0.075,
                          x_pos + offset[i]+0.075,
                          lw=2)

            axs[1].bar(x_pos + offset[i]+0.15, 
                       ds_u_quant.diff_by_std - ds_l_quant.diff_by_std,
                       width=0.15,
                       alpha=0.2,
                       bottom=ds_l_quant.diff_by_std,
                       color=colours[i],
                       tick_label=ds.vertex_choice)
            axs[1].hlines(ds_mean.diff_by_std,
                          x_pos + offset[i]+0.075,
                          x_pos + offset[i]+0.225,
                          lw=2,
                          zorder=10)
        for ax in axs:
             ax.axhline(0, lw=0.8)
             ax.set_ylim(-100,100)

        
        #self.ax.set_xlabel('Buoyancy Gradient')
        #self.ax.set_ylabel('PDF')

        #plt.show()
        #plt.savefig('multi_model_vertex_skill.png', dpi=600)
        plt.savefig('multi_model_vertex_skill_' + str(samples) + 
                    '_samples.png', dpi=600)

    def plot_model_and_glider_diff_rotate_bar(self, case, samples):
        ''' 
        bar chart of difference between model patches and gliders
        mean and std across samples
        '''

        fig, axs = plt.subplots(2,1,figsize=(3.0,4.0))

        vertex = ['left','right','diag_ur','diag_ul']
        colours = ['g', 'b', 'r', 'y']


        x_pos = np.linspace(0.5,1.5,2)
        print (x_pos)
        offset = [-0.3, 0, 0.3]
        path = self.data_path + case
        prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
        ds = xr.open_dataset(path + prepend +  'glider_rotate_diff' +
                             self.append + '_percent_' + str(samples) + 
                             '_samples.nc')

        ds_mean = ds.mean('sample')
        ds_l_quant = ds.quantile(0.05, 'sample')
        ds_u_quant = ds.quantile(0.95, 'sample')

        def render(ax, mean, l_quant, u_quant, x_pos, stat, rota='', c='navy'):
            # non-rotated
            var = 'diff_bx_' + stat + rota

            ax.bar(x_pos, u_quant[var] - l_quant[var],
                   width=0.25, alpha=0.2, bottom=l_quant[var],
                   color=c)
            ax.hlines(mean[var], x_pos-0.125, x_pos+0.125, lw=2)

            var = 'diff_by_' + stat + rota
            ax.bar(x_pos+0.25, u_quant[var] - l_quant[var],
                   width=0.25, alpha=0.2, bottom=l_quant[var],
                   color=c)
            ax.hlines(ds_mean[var], x_pos+0.125,x_pos+0.375,lw=2)

        render(axs[0], ds_mean, ds_l_quant, ds_u_quant,
               x_pos=x_pos[0], stat='mean')
        render(axs[1], ds_mean, ds_l_quant, ds_u_quant,
               x_pos=x_pos[0], stat='std')
        render(axs[0], ds_mean, ds_l_quant, ds_u_quant,
               x_pos=x_pos[1], stat='mean', rota='_rotate', c='green')
        render(axs[1], ds_mean, ds_l_quant, ds_u_quant,
               x_pos=x_pos[1], stat='std',  rota='_rotate', c='green')


        for ax in axs:
             ax.axhline(0, lw=0.8)
             ax.set_ylim(-60,60)

        
        #self.ax.set_xlabel('Buoyancy Gradient')
        #self.ax.set_ylabel('PDF')

        plt.show()
        #plt.savefig(case + '_rotation_test_' + str(samples) + 
        #            '_samples.png', dpi=600)

    def plot_model_and_glider_diff_scatter(self,cases):

        fig, ax = plt.subplots(figsize=(4.5,4.0))

        vertex = ['left','right','diag_ur','diag_ul']
        colours = ['g', 'b', 'r', 'y']


        x_pos = np.linspace(0.5,6.5,7)
        offset = [-0.3, 0, 0.3]
        for i, case in enumerate(cases):
            path = self.data_path + case
            prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
            ds = xr.open_dataset(path + prepend +  'glider_vertex_diff' +
                                 self.append + '.nc')
            print (ds)

            ax.scatter(ds.lat_offset, 
                           ds.diff_bx_mean.sel(vertex_choice=6), 
                           color=colours[i], label='bx mean')
            ax.scatter(ds.lat_offset, 
                           ds.diff_by_mean.sel(vertex_choice=6), 
                           color=colours[i], label='by mean')
        plt.show()

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

#cases = ['EXP10']
#cases = ['EXP13','EXP08','EXP10']
#m.plot_model_and_glider_diff_bar(cases, samples=50)
#m.plot_model_and_glider_diff_bar(cases, samples=100)
#m.plot_model_and_glider_diff_bar(cases, samples=200)
m = glider_path_geometry_plotting()
m.plot_model_and_glider_diff_rotate_bar('EXP10', samples=100)


# save file of vertex percentage error
#cases = ['EXP10']
#cases = ['EXP13','EXP08','EXP10']
#for  case in cases:
#    m = glider_path_geometry(case)
#    m.get_sample_and_glider_diff_rotation(percentage=True, samples=50)
#    m.get_sample_and_glider_diff_rotation(percentage=True, samples=100)
#    m.get_sample_and_glider_diff_rotation(percentage=True, samples=200)


#m.get_model_buoyancy_gradients_patch_set(ml=True)
#m.plot_histogram_buoyancy_gradients_and_samples()
#m.get_vertex_sets_all()
