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

    def get_model_buoyancy_gradients_patch_set(self, ml=False, stats=None,
                                               rolling=False):
        ''' restrict the model time to glider time and sample areas '''
    
        # model
        bg = xr.open_dataset(config.data_path() + self.case +
                             '/SOCHIC_PATCH_3h_20121209_20130331_bg.nc',
                             chunks={'time_counter':113})
                             #chunks='auto')
                             #chunks={'time_counter':1})
        bg = np.abs(bg.sel(deptht=10, method='nearest'))

        # get norm
        bg['bg_norm'] = (bg.bx ** 2 + bg.by ** 2) ** 0.5

        # add lat-lon to dimensions
        bg = bg.assign_coords({'lon':(['x'], bg.nav_lon.isel(y=0)),
                               'lat':(['y'], bg.nav_lat.isel(x=0))})
        bg = bg.swap_dims({'x':'lon','y':'lat'})


        clean_float_time = self.samples.time_counter
        start = clean_float_time.min().astype('datetime64[ns]')
        end   = clean_float_time.max().astype('datetime64[ns]')

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

            dims = ['lon','lat','time_counter']
            if rolling:
                # contstruct allows for mean/std over multiple dims
                patch = patch.dropna('time_counter').sortby('time_counter')
                patch = patch.resample(time_counter='1H').interpolate()
                patch = patch.rolling(time_counter=168, center=True).construct(
                                                               'weekly_rolling')
                dims = ['lat','lon','weekly_rolling']


            if stats == 'mean':
                patch = patch.mean(dims).load()
            if stats == 'std':
                patch = patch.std(dims).load()

            patch_set.append(patch)

        self.model_patches = xr.concat(patch_set, dim='sample')

        return self.model_patches
         
    def get_sample_and_glider_diff_rotation(self, percentage=False,
                                            samples=100, rolling=False):
        '''
        For each rotation choice,
        get difference between mean/std bg for each sample-patch combo

        keep_time: retain the time dimention when making model calcs to
                   remove problem of seasonal change skewing the stats
        03/05 RDP don't think keep_time makes sense...
        '''

        def get_diff(n, rotation=None):
            ''' get diff between model and glider'''

            # glider stats
            self.get_glider_samples(n=n, rotation=rotation)
            if rolling:
                mean_time = self.samples.time_counter.mean('sample')
                mean_time = mean_time.astype('datetime64[ns]')
                samples = self.samples.assign_coords({'time_counter':mean_time})
                samples = samples.swap_dims({'distance':'time_counter'})
                samples = samples.sortby('time_counter')
                samples = samples.resample(time_counter='1H').interpolate()
                samples = samples.rolling(time_counter=168,
                                        center=True).construct('weekly_rolling')
                g_mean = samples.mean('weekly_rolling').load()
                g_std = samples.std('weekly_rolling').load()

            else:
                g_mean = self.samples.mean('distance').load()
                g_std = self.samples.std('distance').load()

            # model stats
            m_mean = self.get_model_buoyancy_gradients_patch_set(stats='mean',
                                                            rolling=rolling)
            m_std  = self.get_model_buoyancy_gradients_patch_set(stats='std',
                                                            rolling=rolling)
            #stat_dims = ['lat','lon','time_counter']
            #m_mean = self.model_patches.mean(stat_dims).load()
            #m_std  = self.model_patches.std(stat_dims).load()

            if percentage:
                denom_m_bx      = np.abs(m_mean.bx)      / 100.0
                denom_m_by      = np.abs(m_mean.by)      / 100.0
                denom_m_bg_norm = np.abs(m_mean.bg_norm) / 100.0
                denom_s_bx      = np.abs(m_std.bx)       / 100.0
                denom_s_by      = np.abs(m_std.by)       / 100.0
                denom_s_bg_norm = np.abs(m_std.bg_norm)  / 100.0
            else: 
                denom = 1.0

            diff_x_mean = (m_mean.bx - g_mean) / denom_m_bx
            diff_y_mean = (m_mean.by - g_mean) / denom_m_by
            diff_norm_mean = (m_mean.bg_norm - g_mean) / denom_m_bg_norm
            diff_x_std = (m_std.bx - g_std) / denom_s_bx
            diff_y_std = (m_std.by - g_std) / denom_s_by
            diff_norm_std = (m_std.bg_norm - g_std) / denom_s_bg_norm

            if rotation != '':
                label = '_rotate'
            diff_x_mean.name = 'diff_bx_mean' + label
            diff_y_mean.name = 'diff_by_mean' + label
            diff_norm_mean.name = 'diff_bg_norm_mean' + label
            diff_x_std.name = 'diff_bx_std' + label
            diff_y_std.name = 'diff_by_std' + label
            diff_norm_std.name = 'diff_bg_norm_std' + label

            diff = xr.merge([diff_x_mean, diff_y_mean, diff_x_std, diff_y_std,
                             diff_norm_mean, diff_norm_std])
            return diff

        non_rot = get_diff(samples)
        rot_90  = get_diff(samples, 90)
        rot_180 = get_diff(samples, 180)
        rot_270 = get_diff(samples, 270)
        # join along new coord
        rotation_coord = xr.DataArray([0,90,180,270], dims='rotation',
                                                      name='rotation')
        diff_ds = xr.concat([non_rot,rot_90,rot_180,rot_270],
                             dim=rotation_coord)


        if percentage:
            f_append = '_percent'
        else: 
            f_append = ''

        if rolling:
            f_append = f_append + '_rolling'

        diff_ds.to_netcdf(self.data_path +  '/BgGliderSamples' + 
                  '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_rotate_diff' +
                  f_append + '_' + str(samples) + '_samples.nc')

    def get_sample_and_glider_diff_vertex_set(self, percentage=False,
                                              samples=100, rolling=False):
        '''
        For each vertex choice,
        get difference between mean/std bg for each sample-patch combo
        '''

        # glider samples
        self.get_glider_samples(n=samples)

        # model patches
        m_mean = self.get_model_buoyancy_gradients_patch_set(stats='mean',
                                                            rolling=rolling)
        m_std  = self.get_model_buoyancy_gradients_patch_set(stats='std',
                                                            rolling=rolling)

        # glider stats
        vertex_list = [[3],[1],[2],[0],[1,3],[0,2],[0,1,2,3]]
        ver_label = ['left','right','diag_ur','diag_ul',
                     'parallel','cross','all']
        g_std_list=[]
        g_mean_list=[]
        for i,choice in enumerate(vertex_list):
            ver_sampled = self.samples.where(self.samples.vertex.isin(choice),
                                             drop=True)
            if rolling:
                # conform time_counter to 1d
                mean_time = ver_sampled.time_counter.mean('sample')
                mean_time = mean_time.astype('datetime64[ns]')
                g = ver_sampled.assign_coords({'time_counter':mean_time})

                # set time_counter as index
                g = g.swap_dims({'distance':'time_counter'})

                # get 1h uniform time (required for rolling)
                uniform_time = np.arange('2012-12-01','2013-04-01', 
                            dtype='datetime64[h]')[:len(ver_sampled.distance)]
                uniform_time_arr = xr.DataArray(uniform_time,
                                         dims='time_counter')

                # interpolate to uniform time
                g = g.interp(time_counter=uniform_time_arr)

                # calcualte rolling object
                g = g.rolling(time_counter=168,
                                        center=True).construct('weekly_rolling')

                # get weekly rolling mean and std
                g_mean = g.mean('weekly_rolling').load()
                g_std = g.std('weekly_rolling').load()

            else:
                g_mean = self.samples.mean('distance').load()
                g_std = self.samples.std('distance').load()

            g_mean_list.append(g_mean)
            g_std_list.append(g_std)

        # join along new coord
        vertex_coord = xr.DataArray(ver_label, dims='vertex_choice',
                                               name='vertex_choice')
        g_mean = xr.concat(g_mean_list, dim=vertex_coord)
        g_std = xr.concat(g_std_list, dim=vertex_coord)

        if percentage:
            denom_m_bx = np.abs(m_mean.bx) / 100.0
            denom_m_by = np.abs(m_mean.by) / 100.0
            denom_m_bg_norm = np.abs(m_mean.bg_norm) / 100.0
            denom_s_bx = np.abs(m_std.bx) / 100.0
            denom_s_by = np.abs(m_std.by) / 100.0
            denom_s_bg_norm = np.abs(m_std.bg_norm) / 100.0
            f_append = '_percent'
        else: 
            denom = 1.0
            f_append = ''
        if rolling:
            f_append = f_append + '_rolling'

        diff_x_mean = (m_mean.bx - g_mean) / denom_m_bx
        diff_y_mean = (m_mean.by - g_mean) / denom_m_by
        diff_norm_mean = (m_mean.bg_norm - g_mean) / denom_m_bg_norm
        diff_x_std = (m_std.bx - g_std) / denom_s_bx
        diff_y_std = (m_std.by - g_std) / denom_s_by
        diff_norm_std = (m_std.bg_norm - g_std) / denom_s_bg_norm

        diff_x_mean.name = 'diff_bx_mean'
        diff_y_mean.name = 'diff_by_mean'
        diff_norm_mean.name = 'diff_bg_norm_mean'
        diff_x_std.name = 'diff_bx_std'
        diff_y_std.name = 'diff_by_std'
        diff_norm_std.name = 'diff_bg_norm_std'

        diff_ds = xr.merge([diff_x_mean,diff_y_mean,
                            diff_x_std,diff_y_std,
                            diff_norm_mean,diff_norm_std])

        diff_ds.to_netcdf(self.data_path +  '/BgGliderSamples' + 
                  '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_vertex_diff' +
                  f_append + '_' + str(samples) + '_samples.nc')

    def get_glider_samples(self, n=100, block=False, rotation=None):
        ''' get set of 100 glider samples '''

        # files definitions
        prep = 'GliderRandomSampling/glider_uniform_interp_1000'
        if rotation:
            rotation_label = 'rotate_' + str(rotation) + '_' 
            rotation_rad = np.radians(rotation)
        else:
            rotation_label = ''
            rotation_rad = rotation # None type 

        if block:
            self.samples = xr.open_dataset(self.data_path + prep + 
                                           rotation_label + '.nc')

        else:
            def expand_sample_dim(ds):
                ds['lon_offset'] = ds.attrs['lon_offset']
                ds['lat_offset'] = ds.attrs['lat_offset']
                ds = ds.set_coords(['lon_offset','lat_offset','time_counter'])
                da = ds['b_x_ml']
                return da

            print ([str(i).zfill(2) for i in range(n)])
            sample_list = [self.data_path + prep + '_' + rotation_label +
                           str(i).zfill(2) + '.nc' for i in range(n)]
            self.samples = xr.open_mfdataset(sample_list, 
                                         combine='nested', concat_dim='sample',
                                         preprocess=expand_sample_dim).load()

            # select depth
            self.samples = self.samples.sel(ctd_depth=10, method='nearest')

            # get transects and cut meso
            sample_list = []
            for i in range(self.samples.sample.size):
                print ('sample: ', i)
                var10 = self.samples.isel(sample=i)
                sample_transect = get_transects(var10, offset=True,
                                  rotation=rotation_rad, cut_meso=True)
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

    def lon_lat_to_km(self, case, lon, lat):
        ''' convert lat-lon to euclidean grid using domain_cfg'''
 
        path = config.data_path() + case + '/'
        cfg = xr.open_dataset(path + 'domain_cfg.nc').isel(z=0).squeeze()
        cfg['dist_x'] = cfg.e1t.cumsum('x')
        cfg['dist_y'] = cfg.e2t.cumsum('y')
        cfg = cfg.assign_coords(lon_1d=cfg.nav_lon.isel(y=0),
                                lat_1d=cfg.nav_lat.isel(x=0))
        cfg = cfg.swap_dims({'x':'lon_1d','y':'lat_1d'})
        print (cfg)
        print (lon)
        print (lat)
        cfg_glider_grid = cfg.interp(lon_1d=lon, lat_1d=lat)#.stack(distance=('lon_1d','lat_1d'))
        plt.figure(1000)
        plt.plot(cfg_glider_grid.dist_x, cfg_glider_grid.dist_y)
        plt.show()
        print (cfg_glider_grid)
        print (cfg_glider_grid.dist_x.dropna('distance'))
        print (cfg_glider_grid.dist_y.dropna('distance'))
        print (fkljsd)

    def get_sampled_path(self, model, append, post_transect=True, 
                         rotation=None):
        ''' load single gilder path sampled from model '''

        path = config.data_path() + model + '/'
        file_path = path + 'GliderRandomSampling/glider_uniform_' + \
                    append +  '_00.nc'
        glider = xr.open_dataset(file_path).sel(ctd_depth=10, method='nearest')
        glider['lon_offset'] = glider.attrs['lon_offset']
        glider['lat_offset'] = glider.attrs['lat_offset']
        coords = ['lon_offset','lat_offset','time_counter']
        glider = glider.set_coords(coords)
        if post_transect:
            glider = get_transects(glider.votemper, offset=True,
                                   rotation=rotation)
        return glider

    def render(self, ax, ds, x_pos, stat, rotate=None,
               label_var='vertex_choice'):
        ''' render stats onto axes'''

        dims = ['sample','time_counter']
        mean = ds.mean(dims)
        std = ds.std(dims)
        l_quant = mean - std
        u_quant = mean + std
        #l_quant = ds.quantile(0.10, dims) 
        #u_quant = ds.quantile(0.90, dims)

        if rotate:
            var = 'diff_bg_norm_' + stat + '_rotate' 
        else:
            var = 'diff_bg_norm_' + stat
        ax.bar(x_pos, u_quant[var] - l_quant[var],
               width=0.25, alpha=0.2, bottom=l_quant[var],
               color='navy', tick_label=ds[label_var], align='edge')
        ax.hlines(mean[var], x_pos, x_pos+0.25, lw=2)

        if rotate:
            var = 'diff_by_' + stat + '_rotate'
        else:
            var = 'diff_by_' + stat
        ax.bar(x_pos+0.25, u_quant[var] - l_quant[var],
               width=0.25, alpha=0.2, bottom=l_quant[var],
               color='navy', tick_label=ds[label_var], align='edge')
        ax.hlines(mean[var], x_pos+0.25,x_pos+0.5,lw=2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    def plot_model_and_glider_diff_vertex_bar(self, cases, samples):
        ''' 
        bar chart of difference between model patches and gliders
        mean and std across samples

        NB: 31/08/22 introduced bg_norm to render()
        Path now included meso transect. Need to remove.
        '''

        import matplotlib.colors as mcolors
        # initialise figure
        fig = plt.figure(figsize=(4.0,4.0))

        # initialise gridspec
        gs0 = gridspec.GridSpec(ncols=1, nrows=2, right=0.97)#, figure=fig)
        gs1 = gridspec.GridSpec(ncols=7, nrows=1, right=0.97)#, figure=fig)
    
        gs0.update(top=0.98, bottom=0.25, left=0.18, hspace=0.1)
        gs1.update(top=0.15, bottom=0.02, left=0.18)

        axs0, axs1 = [], []
        for i in range(2):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(7):
            axs1.append(fig.add_subplot(gs1[i]))

        x_pos = np.linspace(0.5,6.5,7)
        offset = [-0.3, 0, 0.3]
        for i, case in enumerate(cases):
            path = self.data_path + case
            prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
            ds = xr.open_dataset(path + prepend +  'glider_vertex_diff' +
                               self.append + '_percent_rolling_' + str(samples) + 
                                 '_samples.nc')

            self.render(axs0[0], ds, x_pos=x_pos, stat='mean')
            self.render(axs0[1], ds, x_pos=x_pos, stat='std')
        for ax in axs0:
             ax.axhline(0, lw=0.8)
             ax.set_ylim(-105,105)

        axs0[0].set_xticks([])
        axs0[0].set_ylabel('% difference\n [mean]')
        axs0[1].set_ylabel('% difference\n [standard deviation]')
        axs0[1].set_xlabel('vertex')

        axs0[0].text(0.625, 100, r'$b_x$', transform=axs0[0].transData,
                     ha='center', va='center')
        axs0[0].text(0.875, 100, r'$b_y$', transform=axs0[0].transData,
                     ha='center', va='center')

        # add glider paths
        vertex_list     = [[3],[1],[2],[0],[1,3],[0,2],[0,1,2,3]]
        vertex_list_inv = [[0,1,2],[0,2,3],[0,1,3],[1,2,3],[0,2],[1,3]]
        path = self.get_sampled_path('EXP10','interp_1000',post_transect=True)

        # plot removed paths
        for i, choice in enumerate(vertex_list_inv):
            ver_sampled_inv = path.where(path.vertex.isin(choice), drop=True)
            for (l, v) in ver_sampled_inv.groupby('vertex'):
                axs1[i].plot(v.lon, v.lat, c='navy', alpha=0.2)

        # plot kept paths
        for i, choice in enumerate(vertex_list):
            ver_sampled = path.where(path.vertex.isin(choice), drop=True)
            for j, (l, v) in enumerate(ver_sampled.groupby('vertex')):
                c = list(mcolors.TABLEAU_COLORS)[vertex_list[i][j]]
                axs1[i].plot(v.lon, v.lat, c=c)

            axs1[i].axis('off')
            axs1[i].set_xlim(path.lon.min(),path.lon.max())
            axs1[i].set_ylim(path.lat.min(),path.lat.max())

        
        plt.savefig('multi_model_vertex_skill_' + str(samples) + 
                    '_samples_rolling.png', dpi=600)


    def plot_model_and_glider_diff_vertex_timeseries(self, case, samples):
        ''' 
        Time series of difference between model patches and gliders
        mean and std across samples.
        Same as bar chart but for weely chunks.
        '''

        import matplotlib.colors as mcolors

        # initialise figure
        fig = plt.figure(figsize=(6.5,6.5))

        # initialise gridspec
        gs0 = gridspec.GridSpec(ncols=1, nrows=4, right=0.97)#, figure=fig)
        gs1 = gridspec.GridSpec(ncols=1, nrows=4, right=0.97)#, figure=fig)
        gs2 = gridspec.GridSpec(ncols=4, nrows=1, right=0.97)#, figure=fig)
        gs3 = gridspec.GridSpec(ncols=3, nrows=1, right=0.97)#, figure=fig)
    
        gs0.update(top=0.99, bottom=0.15, left=0.15, right=0.55, hspace=0.05)
        gs1.update(top=0.99, bottom=0.15, left=0.58, right=0.98, hspace=0.05)
        gs2.update(top=0.06, bottom=0.00, left=0.15, right=0.55)
        gs3.update(top=0.06, bottom=0.00, left=0.58, right=0.98)

        axs0, axs1, axs2, axs3 = [], [], [], []
        for i in range(4):
            axs0.append(fig.add_subplot(gs0[i]))
            axs1.append(fig.add_subplot(gs1[i]))
        for i in range(4):
            axs2.append(fig.add_subplot(gs2[i]))
        for i in range(3):
            axs3.append(fig.add_subplot(gs3[i]))

        colours = ['navy', 'gold', 'teal']
    
        # get data
        path = self.data_path + case
        prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
        ds = xr.open_dataset(path + prepend +  'glider_vertex_diff' +
                             self.append + '_percent_rolling_' + str(samples) + 
                             '_samples.nc')

        def render(da, ax, l='', c=list(mcolors.TABLEAU_COLORS)):
            da = da.squeeze()

            # scale by number of samples at each time
            #sample_scale = xr.where(np.isnan(da), 0, 1).sum('sample')/100
            #da = da * sample_scale

            # remove times with few samples
            sample_scale = xr.where(np.isnan(da), 0, 1).sum('sample')
            da = da.where(sample_scale>90)
            mean = da.mean('sample')

            #l_quant, u_quant = da.quantile([0.05, 0.95], 'sample')

            # std err
            #sample_scale = np.sqrt(xr.where(np.isnan(da),0,1).sum('sample'))
            #std_err = da.std('sample') / sample_scale

            std = da.std('sample')
            l_quant, u_quant = mean - std, mean + std

            ax.plot(mean.time_counter, mean, c=c[i], label=l,lw=0.8)
            ax.fill_between(l_quant.time_counter, l_quant, u_quant,
                                 color=c[i], alpha=0.4, edgecolor=None)

        ds0 = ds.sel(vertex_choice=['left','right','diag_ur','diag_ul'])
        mc = list(mcolors.TABLEAU_COLORS)
        mcolors_reorder= [mc[3], mc[1], mc[2], mc[0]]
        for i, (l, da), in enumerate(ds0.groupby('vertex_choice')):
            render(da.diff_bx_mean, axs0[0], l=l, c=mcolors_reorder)
            render(da.diff_bx_std,  axs0[1], l=l, c=mcolors_reorder)
            render(da.diff_by_mean, axs0[2], l=l, c=mcolors_reorder)
            render(da.diff_by_std,  axs0[3], l=l, c=mcolors_reorder)

        ds1 = ds.sel(vertex_choice=['parallel','cross','all'])
        for i, (l, da), in enumerate(ds1.groupby('vertex_choice')):
            render(da.diff_bx_mean, axs1[0], l=l, c=colours)
            render(da.diff_bx_std,  axs1[1], l=l, c=colours)
            render(da.diff_by_mean, axs1[2], l=l, c=colours)
            render(da.diff_by_std,  axs1[3], l=l, c=colours)

        for ax in axs0 + axs1:
            ax.axhline(0, lw=0.5, c='black')
            ax.axhline(20, lw=0.5, ls='--', c='black')
            ax.axhline(-20, lw=0.5, ls='--', c='black')
            ax.set_ylim(-150,150)
            ax.set_xlim(ds.time_counter.min(), ds.time_counter.max())

        #axs0[0].set_xticks([])
        axs0[0].set_ylabel('% difference\n [mean]')
        axs0[1].set_ylabel('% difference\n [standard deviation]')
        axs0[2].set_ylabel('% difference\n [mean]')
        axs0[3].set_ylabel('% difference\n [standard deviation]')
        axs0[3].set_xlabel('date')
        axs1[3].set_xlabel('date')

        for i in [0,1,2]:
            axs0[i].set_xticks([])
            axs1[i].set_xticks([])
        for ax in axs1:
            ax.set_yticks([])
        for label in axs0[3].get_xticklabels():
            label.set_rotation(20)
            label.set_ha('right')
        for label in axs1[3].get_xticklabels():
            label.set_rotation(20)
            label.set_ha('right')

        # add glider paths
        vertex_list     = [[3],[1],[2],[0],[1,3],[0,2],[0,1,2,3]]
        vertex_list_inv = [[0,1,2],[0,2,3],[0,1,3],[1,2,3],[0,2],[1,3]]
        path = self.get_sampled_path('EXP10','interp_1000',post_transect=True)

        def render_paths(ax, vertex_list, vertex_list_inv, path,
                         repeat_c=None):
            # plot removed paths
            for i, choice in enumerate(vertex_list_inv):
                ver_sampled_inv = path.where(path.vertex.isin(choice),drop=True)
                for (l, v) in ver_sampled_inv.groupby('vertex'):
                    ax[i].plot(v.lon, v.lat, c='navy', alpha=0.2)

            # plot kept paths
            for i, choice in enumerate(vertex_list):
                ver_sampled = path.where(path.vertex.isin(choice), drop=True)
                for j, (l, v) in enumerate(ver_sampled.groupby('vertex')):
                    if repeat_c:
                        c = repeat_c[i]
                    else:
                        c = list(mcolors.TABLEAU_COLORS)[vertex_list[i][j]]
                    ax[i].plot(v.lon, v.lat, c=c)

                ax[i].axis('off')
                ax[i].set_xlim(path.lon.min(),path.lon.max())
                ax[i].set_ylim(path.lat.min(),path.lat.max())
                ax[i].set_aspect('equal')
        render_paths(axs2, vertex_list[:4], vertex_list_inv[:4], path)
        render_paths(axs3, vertex_list[4:], vertex_list_inv[4:], path,
                     repeat_c=colours)

        # save
        plt.savefig(case + '_diff_vertex_time_series_' + str(samples) + 
                    '_samples.png', dpi=600)

    def plot_model_and_glider_diff_rotate_timeseries(self, case, samples):
        ''' 
        Time series of difference between model patches and gliders
        mean and std across samples.
        Same as bar chart but for weely chunks.
        '''
        import matplotlib.colors as mcolors

        # initialise figure
        fig = plt.figure(figsize=(4.0,5.0))

        # initialise gridspec
        gs0 = gridspec.GridSpec(ncols=1, nrows=4, right=0.99)#, figure=fig)
        gs1 = gridspec.GridSpec(ncols=4, nrows=1, right=0.99)#, figure=fig)
    
        gs0.update(top=0.99, bottom=0.25, left=0.20, hspace=0.05)
        gs1.update(top=0.12, bottom=0.02, left=0.22)

        axs0, axs1 = [], []
        for i in range(4):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(4):
            axs1.append(fig.add_subplot(gs1[i]))
    
        # get data
        path = self.data_path + case
        prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
        ds = xr.open_dataset(path + prepend +  'glider_rotate_diff' +
                             self.append + '_percent_rolling_' + str(samples) + 
                             '_samples.nc')

        def render(da, ax, c):
            sample_scale = xr.where(np.isnan(da), 0, 1).sum('sample')
            da = da.where(sample_scale>90)

            mean = da.mean('sample')
            #l_quant, u_quant = da.quantile([0.25, 0.75], 'sample')
            #l_quant, u_quant = da.quantile([0.05, 0.95], 'sample')
            #std = da.std('sample')*2
            #l_quant, u_quant = mean - std, mean + std
            std = da.std('sample')
            l_quant, u_quant = mean - std, mean + std


            ax.plot(mean.time_counter, mean, c=c[i], lw=0.8)
            ax.fill_between(l_quant.time_counter, l_quant, u_quant,
                                 color=c[i], edgecolor=None, alpha=0.3)

        colours = ['navy', 'purple', 'green', 'gold']
        for i, (l, da), in enumerate(ds.groupby('rotation')):
            render(da.diff_bx_mean_rotate, axs0[0], c=colours)
            render(da.diff_bx_std_rotate,  axs0[1], c=colours)
            render(da.diff_by_mean_rotate, axs0[2], c=colours)
            render(da.diff_by_std_rotate,  axs0[3], c=colours)

        for ax in axs0:
            ax.axhline(0, lw=0.5, c='black')
            ax.axhline(20, lw=0.5, ls='--', c='black')
            ax.axhline(-20, lw=0.5, ls='--', c='black')
            ax.set_ylim(-150,150)
            ax.set_xlim(ds.time_counter.min(), ds.time_counter.max())

        #axs0[0].set_xticks([])
        axs0[0].set_ylabel('% difference\n [mean]')
        axs0[1].set_ylabel('% difference\n [standard deviation]')
        axs0[2].set_ylabel('% difference\n [mean]')
        axs0[3].set_ylabel('% difference\n [standard deviation]')
        axs0[3].set_xlabel('time')

        for i in [0,1,2]:
            axs0[i].set_xticks([])
        for label in axs0[3].get_xticklabels():
            label.set_rotation(20)
            label.set_ha('right')

        # add glider paths
        rotations = [0, 90, 180, 270]

        for i, ax in enumerate(axs1):
            if rotations[i] == 0:
                path = self.get_sampled_path('EXP10', 
                                    'interp_1000',
                                    post_transect=True)
            else:
                path = self.get_sampled_path('EXP10', 
                                    'interp_1000_rotate_' + str(rotations[i]),
                                    post_transect=True,
                                    rotation=np.radians(rotations[i]))
            for (l, v) in path.groupby('vertex'):
                ax.plot(v.lon, v.lat, label=l)
            circle = plt.Circle((0.5, 1.1), 0.1, color=colours[i],
                                transform=ax.transAxes, clip_on=False)
            ax.add_patch(circle)
            ax.axis('off')

        # save
        plt.savefig(case + '_diff_rotation_time_series_' + str(samples) + 
                    '_samples.png', dpi=600)

    def plot_model_and_glider_diff_rotate_bar(self, case, samples):
        ''' 
        bar chart of difference between model patches and gliders
        mean and std across samples
        '''

        # initialise figure
        fig = plt.figure(figsize=(3.0,4.0))

        # initialise gridspec
        gs0 = gridspec.GridSpec(ncols=1, nrows=2, right=0.97)#, figure=fig)
        gs1 = gridspec.GridSpec(ncols=4, nrows=1, right=0.97)#, figure=fig)
    
        gs0.update(top=0.98, bottom=0.25, left=0.24, hspace=0.1)
        gs1.update(top=0.15, bottom=0.02, left=0.24)

        axs0, axs1 = [], []
        for i in range(2):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(4):
            axs1.append(fig.add_subplot(gs1[i]))
    
        # get data
        path = self.data_path + case
        prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
        ds = xr.open_dataset(path + prepend +  'glider_rotate_diff' +
                             self.append + '_percent_rolling_' + str(samples) + 
                             '_samples.nc')

        x_pos = np.linspace(0.5,3.5,4)
        self.render(axs0[0], ds, x_pos=x_pos, stat='mean', rotate=True,
                    label_var='rotation')
        self.render(axs0[1], ds, x_pos=x_pos, stat='std', rotate=True,
                    label_var='rotation')


        for ax in axs0:
            ax.axhline(0, lw=0.8)
            ax.set_ylim(-85,85)

        axs0[0].set_xticks([])
        axs0[0].set_ylabel('% difference\n [mean]')
        axs0[1].set_ylabel('% difference\n [standard deviation]')
        axs0[1].set_xlabel('rotation [degrees]')

        axs0[0].text(0.625, 70, r'$b_x$', transform=axs0[0].transData,
                     ha='center', va='center')
        axs0[0].text(0.875, 70, r'$b_y$', transform=axs0[0].transData,
                     ha='center', va='center')

        # add glider paths
        rotations = [0, 90, 180, 270]

        for i, ax in enumerate(axs1):
            if rotations[i] == 0:
                path = self.get_sampled_path('EXP10', 
                                    'interp_1000',
                                    post_transect=True)
            else:
                path = self.get_sampled_path('EXP10', 
                                    'interp_1000_rotate_' + str(rotations[i]),
                                    post_transect=True,
                                    rotation=np.radians(rotations[i]))

            for (l, v) in path.groupby('vertex'):
                ax.plot(v.lon, v.lat, label=l)
            ax.axis('off')

        plt.savefig(case + '_rotation_' + str(samples) + 
                    '_samples_rolling.png', dpi=600)
        

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

cases = ['EXP10']
#cases = ['EXP13','EXP08','EXP10']
#m.plot_model_and_glider_diff_bar(cases, samples=50)
#m.plot_model_and_glider_diff_bar(cases, samples=200)
m = glider_path_geometry_plotting()
#m.plot_model_and_glider_diff_rotate_bar('EXP10', samples=100)
#m.plot_model_and_glider_diff_rotate_timeseries('EXP10', samples=100)
#m.plot_model_and_glider_diff_vertex_timeseries('EXP10', samples=100)
m.plot_model_and_glider_diff_vertex_bar(cases, samples=100)


# save file of vertex percentage error
#cases = ['EXP13','EXP08','EXP10']
#cases = ['EXP10']
#for  case in cases:
#    m = glider_path_geometry(case)
#    m.get_sample_and_glider_diff_vertex_set(percentage=True, samples=100,
#                                             rolling=True)
#    m.get_sample_and_glider_diff_rotation(percentage=True, samples=100,
#                                          rolling=True)
#    m.get_sample_and_glider_diff_rotation(percentage=True, samples=50)
#    m.get_sample_and_glider_diff_rotation(percentage=True, samples=200)


#m.get_model_buoyancy_gradients_patch_set(ml=True)
#m.plot_histogram_buoyancy_gradients_and_samples()
#m.get_vertex_sets_all()
