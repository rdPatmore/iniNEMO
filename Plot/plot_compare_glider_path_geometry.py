import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import dask
import matplotlib
from get_transects import get_transects, get_sampled_path
import cartopy.crs as ccrs

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

    def save_model_buoyancy_gradients_patch_set(self, stats=None,
                                                                 rolling=False):
        ''' restrict the model time to glider time and sample areas '''
    

        rolling_str, stats_str = '', ''
        # model
        bg = xr.open_dataset(config.data_path() + self.case +
                             '/SOCHIC_PATCH_3h_20121209_20130331_bg.nc')
                             #chunks={'time_counter':113})
                             #chunks='auto')
                             #chunks={'time_counter':1})
        bg = np.abs(bg.sel(deptht=10, method='nearest')).load()

        # get norm
        bg['bg_norm'] = (bg.bx ** 2 + bg.by ** 2) ** 0.5

        # add lat-lon to dimensions
        bg = bg.assign_coords({'lon':(['x'], bg.nav_lon.isel(y=0)),
                               'lat':(['y'], bg.nav_lat.isel(x=0))})
        bg = bg.swap_dims({'x':'lon','y':'lat'})


        clean_float_time = self.samples.time_counter
        start = clean_float_time.min().astype('datetime64[s]')
        end   = clean_float_time.max().astype('datetime64[s]')

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
                rolling_str = '_rolling'
                # contstruct allows for mean/std over multiple dims
                #patch = patch.sortby('time_counter')
                patch = patch.resample(time_counter='1H').median()
                patch = patch.rolling(time_counter=168, center=True).construct(
                                                               'weekly_rolling')
                dims = ['lat','lon','weekly_rolling']


            if stats == 'mean':
                patch = patch.mean(dims).load()
                stats_str = '_mean'
            if stats == 'std':
                patch = patch.std(dims).load()
                stats_str = '_std'
            if stats == 'median':
                stats_str = '_median'
                patch = patch.median(dims).load()

            patch_set.append(patch)

        self.model_patches = xr.concat(patch_set, dim='sample')

        # save
        self.model_patches.to_netcdf(config.data_path() + self.case +
                '/PatchSets/SOCHIC_PATCH_3h_20121209_20130331_bg_patch_set' +
                rolling_str + stats_str + '.nc')

    def get_model_buoyancy_gradients_patch_set(self, stats=None, rolling=False):
        ''' load patch set that corresponds to glider samples '''
  
        # prep file name
        rolling_str, stats_str = '', ''
        if stats:
            stats_str = '_' + stats
        if rolling:
            rolling_str = '_rolling'
        
        # load
        da = xr.open_dataset(config.data_path() + self.case +
                '/PatchSets/SOCHIC_PATCH_3h_20121209_20130331_bg_patch_set' +
                rolling_str + stats_str + '.nc')

        return da

         
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
            self.get_glider_samples(rotation=rotation)
            if rolling:
                # conform time_counter to 1d
                mean_time = self.samples.time_counter.mean('sample')
                mean_time = mean_time.astype('datetime64[s]')
                samples = self.samples.assign_coords({'time_counter':mean_time})
                samples = samples.swap_dims({'distance':'time_counter'})
                # get 1h uniform time (required for rolling)
                uniform_time = np.arange('2012-12-01','2013-04-01', 
                            dtype='datetime64[h]')
                uniform_time_arr = xr.DataArray(uniform_time,
                                         dims='time_counter')

                # interpolate to uniform time
                samples = samples.interp(time_counter=uniform_time_arr)

                # calcualte rolling object
                samples = samples.rolling(time_counter=168,
                                        center=True).construct('weekly_rolling')

                # set minimum amount of data-points in week
                samples = samples.where(samples.count('weekly_rolling') > 48)

                #samples = samples.sortby('time_counter')
                #samples = samples.resample(time_counter='1H').median()
                #samples = samples.rolling(time_counter=168,
                #                        center=True).construct('weekly_rolling')
                # get weekly rolling mean and std
                g_mean = samples.mean('weekly_rolling').load()
                g_std = samples.std('weekly_rolling').load()
                g_med = samples.median('weekly_rolling').load()

            else:
                g_mean = self.sample_set.mean('distance').load()
                g_std = self.sample_set.std('distance').load()
                g_med = self.sample_set.median('weekly_rolling').load()

            # model stats
            m_mean = self.get_model_buoyancy_gradients_patch_set(stats='mean',
                                                            rolling=rolling)
            m_std  = self.get_model_buoyancy_gradients_patch_set(stats='std',
                                                            rolling=rolling)
            m_med  = self.get_model_buoyancy_gradients_patch_set(stats='median',
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
                denom_med_bx      = np.abs(m_med.bx)       / 100.0
                denom_med_by      = np.abs(m_med.by)       / 100.0
                denom_med_bg_norm = np.abs(m_med.bg_norm)  / 100.0
            else: 
                denom = 1.0

            diff_x_mean = (m_mean.bx - g_mean) / denom_m_bx
            diff_y_mean = (m_mean.by - g_mean) / denom_m_by
            diff_norm_mean = (m_mean.bg_norm - g_mean) / denom_m_bg_norm
            diff_x_std = (m_std.bx - g_std) / denom_s_bx
            diff_y_std = (m_std.by - g_std) / denom_s_by
            diff_norm_std = (m_std.bg_norm - g_std) / denom_s_bg_norm
            diff_x_med = (m_med.bx - g_med) / denom_med_bx
            diff_y_med = (m_med.by - g_med) / denom_med_by
            diff_norm_med = (m_med.bg_norm - g_med) / denom_med_bg_norm

            if rotation != '':
                label = '_rotate'
            diff_x_mean.name = 'diff_bx_mean' + label
            diff_y_mean.name = 'diff_by_mean' + label
            diff_norm_mean.name = 'diff_bg_norm_mean' + label
            diff_x_std.name = 'diff_bx_std' + label
            diff_y_std.name = 'diff_by_std' + label
            diff_norm_std.name = 'diff_bg_norm_std' + label
            diff_x_med.name = 'diff_bx_med' + label
            diff_y_med.name = 'diff_by_med' + label
            diff_norm_med.name = 'diff_bg_norm_med' + label

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
        print (non_rot)
        print (rot_90)
        print (rot_180)
        print (rot_270)
        diff_ds = xr.concat([non_rot,rot_90,rot_180,rot_270],
                             dim=rotation_coord)


        if percentage:
            f_append = '_percent'
        else: 
            f_append = ''

        if rolling:
            f_append = f_append + '_rolling'
        print (self.data_path)
        print (f_append)
        print (str(samples))
        print (diff_ds)

        diff_ds.to_netcdf(self.data_path +  '/BgGliderSamples' + 
          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_rotate_diff_threshold' +
                  f_append + '_' + str(samples) + '_samples.nc')

    def get_sample_and_glider_diff_vertex_set(self, percentage=False,
                                              samples=100, rolling=False):
        '''
        For each vertex choice,
        get difference between mean/std bg for each sample-patch combo
        '''

        # glider samples
        self.get_glider_samples()

        # model patches
        m_mean = self.get_model_buoyancy_gradients_patch_set(stats='mean',
                                                            rolling=rolling)
        m_std  = self.get_model_buoyancy_gradients_patch_set(stats='std',
                                                            rolling=rolling)
        m_med  = self.get_model_buoyancy_gradients_patch_set(stats='median',
                                                            rolling=rolling)
        print ('done model patches')

        # glider stats
        vertex_list = [[3],[1],[2],[0],[1,3],[0,2],[0,1,2,3]]
        ver_label = ['left','right','diag_ur','diag_ul',
                     'parallel','cross','all']

        g_std_list, g_mean_list, g_med_list = [], [], []
        for i,choice in enumerate(vertex_list):
            print ('i vert: ', i)
            ver_sampled = self.samples.where(self.samples.vertex.isin(choice),
                                             drop=True)
            # conform time_counter to 1d
            mean_time = ver_sampled.time_counter.mean('sample')
            mean_time = mean_time.astype('datetime64[s]')
            g = ver_sampled.assign_coords({'time_counter':mean_time})

            # set time_counter as index
            g = g.swap_dims({'distance':'time_counter'})

            if rolling:
                # get 1h uniform time (required for rolling)
                uniform_time = np.arange('2012-12-01','2013-04-01', 
                            dtype='datetime64[h]')
                uniform_time_arr = xr.DataArray(uniform_time,
                                         dims='time_counter')

                # interpolate to uniform time
                g = g.interp(time_counter=uniform_time_arr)

# this method is consistent with model calcs (replace with uniform time method)
#                g = g.sortby('time_counter')
#                g = g.resample(time_counter='1H').median()


                #g = g.fillna(1000)
                # calcualte rolling object
                g = g.rolling(time_counter=168,
                                        center=True).construct('weekly_rolling')

                # set minimum amount of data-points in week
#                print (' ')
#                print (' ')
#                print (' ')
#                print (' ')
#                plt.figure(10)
#                plt.plot(g.count('weekly_rolling').isel(sample=0), label=ver_label[i])
#                plt.legend()
#                print (g.count('weekly_rolling').isel(sample=0))
#                print (' ')
#                print (' ')
#                print (' ')
#                print (' ')
#                print (' ')
                g = g.where(g.count('weekly_rolling') > 48)
                
                ## get weekly rolling mean and std
                g_mean = g.mean('weekly_rolling').load()
                g_std = g.std('weekly_rolling').load()
                g_med = g.median('weekly_rolling').load()

            else:
                g_mean = g.mean('time_counter').load()
                g_std = g.std('time_counter').load()
                g_med = g.median('time_counter').load()

            g_mean_list.append(g_mean)
            g_std_list.append(g_std)
            g_med_list.append(g_med)

#        plt.show()
        def get_frac(var_list, model_ds, stat_str):
            # join along new coord
            vertex_coord = xr.DataArray(ver_label, dims='vertex_choice',
                                                   name='vertex_choice')
            g = xr.concat(var_list, dim=vertex_coord)

            if percentage:
                denom_bx = np.abs(model_ds.bx) / 100.0
                denom_by = np.abs(model_ds.by) / 100.0
                denom_bg_norm = np.abs(model_ds.bg_norm) / 100.0
                f_append = '_percent'
            else: 
                denom = 1.0
                f_append = ''
            if rolling:
                f_append = f_append + '_rolling'

            diff_x = (model_ds.bx - g) / denom_bx
            diff_y = (model_ds.by - g) / denom_by
            diff_norm = (model_ds.bg_norm - g) / denom_bg_norm

            diff_x.name = 'diff_bx_' + stat_str
            diff_y.name = 'diff_by_' + stat_str
            diff_norm.name = 'diff_bg_norm_' + stat_str

#            print (g)
#            print (model_ds)
#            print (diff_norm)
#            fig, axs = plt.subplots(3)
#            axs[0].plot(g.sel(vertex_choice='parallel').isel(sample=0))
#            axs[1].plot(model_ds.bg_norm.isel(sample=1))
#            axs[2].plot(diff_norm.sel(vertex_choice='parallel').isel(sample=0))
#            plt.show()

            return [diff_x, diff_y, diff_norm], f_append

        means, f_append = get_frac(g_mean_list, m_mean, 'mean')
        print ('done means')
        stds,         _ = get_frac(g_std_list, m_std, 'std')
        print ('done stds')
        medians,      _ = get_frac(g_med_list, m_med, 'median')
        print ('done meds')

        diff_ds = xr.merge(means + stds + medians)

        diff_ds.to_netcdf(self.data_path +  '/BgGliderSamples' + 
         '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_vertex_diff_threshold' +
                  f_append + '_' + str(samples) + '_samples.nc')

    def save_glider_samples(self, block=False, rotation=None):
        ''' get set of 100 glider samples '''

        # number of glider samples
        n=100

        # files definitions
        prep = 'GliderRandomSampling/glider_uniform_interp_1000'
        if rotation:
            rotation_label = '_rotate_' + str(rotation) 
            rotation_rad = np.radians(rotation)
        else:
            rotation_label = ''
            rotation_rad = rotation # None type 

        if block:
            self.samples = xr.open_dataset(self.data_path + prep + 
                                           rotation_label + '.nc',
                                          decode_times=False,
                                         ).b_x_ml.load()

        else:
            def expand_sample_dim(ds):
                ds['lon_offset'] = ds.attrs['lon_offset']
                ds['lat_offset'] = ds.attrs['lat_offset']
                ds = ds.set_coords(['lon_offset','lat_offset','time_counter'])
                da = ds['b_x_ml']
                return da

            print ([str(i).zfill(2) for i in range(n)])
            sample_list = [self.data_path + prep + rotation_label + '_' +
                           str(i).zfill(2) + '.nc' for i in range(n)]
            self.samples = xr.open_mfdataset(sample_list, 
                                         combine='nested', concat_dim='sample',
                                         decode_times=False,
                                         preprocess=expand_sample_dim).load()

        # select depth
        self.samples = self.samples.sel(ctd_depth=10, method='nearest')

        print (self.samples)
        # get transects and cut meso
        def get_transects_all_samples(arr): 
            transected_samps = get_transects(arr, offset=True,
                                           rotation=rotation_rad, cut_meso=True)
            return transected_samps
        self.samples = self.samples.groupby('sample').map(
                                                     get_transects_all_samples)
        
        # save

        #    sample_list.append(sample_transect)

        #    def get_complex_arg(arr):
        #        return np.angle(arr)
        #    Tu_phi = np.abs(xr.apply_ufunc(get_complex_arg, Tu_comp,
        #                            dask='parallelized'))


        # get transects and cut meso
        #sample_list = []
        #for i in range(self.samples.sample.size):
        #    print ('sample: ', i)
        #    var10 = self.samples.isel(sample=i)
        #    sample_transect = get_transects(var10, offset=True,
        #                      rotation=rotation_rad, cut_meso=True)
        #    sample_list.append(sample_transect)
        #self.samples=xr.concat(sample_list, dim='sample')

        # set time to float for averaging
        # not required if decode times = false 
        #float_time = self.samples.time_counter.astype('float64')
        #clean_float_time = float_time.where(float_time > 0, np.nan)
        #self.samples['time_counter'] = clean_float_time

        # absolute value of buoyancy gradients
        self.samples = np.abs(self.samples)

        # save
        self.samples.to_netcdf(self.data_path + prep + rotation_label +
                               '_b_x_abs_10m_post_transects.nc')

    def get_glider_samples(self, rotation=None):
        ''' load processed glider samples '''

        # files definitions
        prep = 'GliderRandomSampling/glider_uniform_interp_1000'
        rotation_label = ''
        if rotation:
            rotation_label = '_rotate_' + str(rotation) 

        # get samples
        self.samples = xr.open_dataarray(self.data_path + prep +
             rotation_label + '_b_x_abs_10m_post_transects.nc', 
             decode_times=False)

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

#    def get_sampled_path(self, model, append, post_transect=True, 
#                         rotation=None):
#        ''' load single gilder path sampled from model '''
#
#        path = config.data_path() + model + '/'
#        file_path = path + 'GliderRandomSampling/glider_uniform_' + \
#                    append +  '_00.nc'
#        glider = xr.open_dataset(file_path).sel(ctd_depth=10, method='nearest')
#        glider['lon_offset'] = glider.attrs['lon_offset']
#        glider['lat_offset'] = glider.attrs['lat_offset']
#        coords = ['lon_offset','lat_offset','time_counter']
#        glider = glider.set_coords(coords)
#        if post_transect:
#            glider = get_transects(glider.votemper, offset=True,
#                                   rotation=rotation)
#        return glider

    def render_bx_by(self, ax, ds, x_pos, stat, rotate=None,
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
            var = 'diff_bx_' + stat + '_rotate' 
        else:
            var = 'diff_bx_' + stat
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

    def render_bg_norm(self, ax, ds, x_pos, stat, rotate=None,
                       label_var='vertex_choice'):
        ''' render norm stats onto axes '''

        # get stats
        dims = ['time_counter']
        mean = ds.mean(dims)
        std = ds.std(dims)
        l_quant = mean - std
        u_quant = mean + std

        # bootstrap
        mean = mean.mean('sample')
        std = std.mean('sample')
        l_quant = l_quant.mean('sample')
        u_quant = u_quant.mean('sample')

        # check for rotation
        if rotate:
            var = 'diff_bg_norm_' + stat + '_rotate' 
        else:
            var = 'diff_bg_norm_' + stat

        colours = ['#dad1d1', '#7e9aa5', '#55475a']
        # plot bars 
        print (x_pos)
        print (u_quant[var])
        ax.bar(x_pos, u_quant[var] - l_quant[var],
               width=0.5, alpha=1.0, bottom=l_quant[var],
               color=colours[0], tick_label=ds[label_var], align='center')

        # plot mean line 
        ax.hlines(mean[var], x_pos-0.25, x_pos+0.25, lw=2, colors=colours[2])

        # remove frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def plot_model_and_glider_diff_vertex_bar_norm(self, cases, samples):
        ''' 
        bar chart of difference between model patches and gliders
        mean and std across samples

        NB: 31/08/22 introduced bg_norm to render()
        Path now included meso transect. Need to remove.
        '''

        import matplotlib.colors as mcolors

        # ~~~~~ define figure layout ~~~~ #

        # initialise figure
        fig = plt.figure(figsize=(4.0,4.0))

        # initialise gridspec
        gs0 = gridspec.GridSpec(ncols=1, nrows=2, right=0.97)#, figure=fig)
        gs1 = gridspec.GridSpec(ncols=7, nrows=1, right=0.97)#, figure=fig)
    
        # set frame bounds
        gs0.update(top=0.98, bottom=0.25, left=0.18, hspace=0.1)
        gs1.update(top=0.15, bottom=0.02, left=0.18)

        # assign axes to lists
        axs0, axs1 = [], []
        for i in range(2):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(7):
            axs1.append(fig.add_subplot(gs1[i]))

        # ~~~~~ load and render data ~~~~~ #

        x_pos = np.linspace(0.5,6.5,7)
        offset = [-0.3, 0, 0.3]
        for i, case in enumerate(cases):
            path = self.data_path + case
            prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
            ds = xr.open_dataset(path + prepend +  'glider_vertex_diff' +
                             self.append + '_percent_rolling_' + str(samples) + 
                                 '_samples.nc')

            self.render_bg_norm(axs0[0], ds, x_pos=x_pos, stat='mean')
            self.render_bg_norm(axs0[1], ds, x_pos=x_pos, stat='std')


        # ~~~~ axes parameters ~~~~ #

        for ax in axs0:
             ax.set_ylim(0,80)

        axs0[0].set_xticks([])
        axs0[0].set_ylabel('% difference\n [mean]')
        axs0[1].set_ylabel('% difference\n [standard deviation]')
        axs0[1].set_xlabel('vertex')

        axs0[0].text(0.625, 100, r'$b_x$', transform=axs0[0].transData,
                     ha='center', va='center')
        axs0[0].text(0.875, 100, r'$b_y$', transform=axs0[0].transData,
                     ha='center', va='center')

        self.add_paths_vertex(axs1)

        # save
        plt.savefig('multi_model_vertex_skill_' + str(samples) + 
                    '_samples_rolling_norm_bootstrap.png', dpi=600)

    def plot_model_and_glider_diff_vertex_bar_vec(self, cases, samples):
        ''' 
        bar chart of difference between model patches and gliders
        mean and std across samples

        NB: 31/08/22 introduced bg_norm to render()
        Path now included meso transect. Need to remove.
        '''

        # ~~~~~ define figure layout ~~~~ #

        # initialise figure
        fig = plt.figure(figsize=(4.0,4.0))

        # initialise gridspec
        gs0 = gridspec.GridSpec(ncols=1, nrows=2, right=0.97)#, figure=fig)
        gs1 = gridspec.GridSpec(ncols=7, nrows=1, right=0.97)#, figure=fig)
    
        # set frame bounds
        gs0.update(top=0.98, bottom=0.25, left=0.18, hspace=0.1)
        gs1.update(top=0.15, bottom=0.02, left=0.18)

        # assign axes to lists
        axs0, axs1 = [], []
        for i in range(2):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(7):
            axs1.append(fig.add_subplot(gs1[i]))

        # ~~~~~ load and render data ~~~~~ #

        x_pos = np.linspace(0.5,6.5,7)
        offset = [-0.3, 0, 0.3]
        for i, case in enumerate(cases):
            path = self.data_path + case
            prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
            ds = xr.open_dataset(path + prepend +  'glider_vertex_diff' +
                             self.append + '_percent_rolling_' + str(samples) + 
                                 '_samples.nc')

            self.render_bx_by(axs0[0], ds, x_pos=x_pos, stat='mean')
            self.render_bx_by(axs0[1], ds, x_pos=x_pos, stat='std')

        # ~~~~ axes parameters ~~~~ #

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

        self.add_paths_vertex(axs1)

        plt.savefig('multi_model_vertex_skill_' + str(samples) + 
                    '_samples_rolling_vectors.png', dpi=600)

    def add_paths_vertex(self, axs):
        ''' add glider paths to vertex plot '''

        import matplotlib.colors as mcolors

        # add glider paths
        vertex_list     = [[3],[1],[2],[0],[1,3],[0,2],[0,1,2,3]]
        vertex_list_inv = [[0,1,2],[0,2,3],[0,1,3],[1,2,3],[0,2],[1,3]]
        path = get_sampled_path('EXP10','interp_1000', post_transect=True,
                                     cut_meso=True)

        # plot removed paths
        for i, choice in enumerate(vertex_list_inv):
            ver_sampled_inv = path.where(path.vertex.isin(choice), drop=True)
            for (l, v) in ver_sampled_inv.groupby('vertex'):
                axs[i].plot(v.lon, v.lat, c='navy', alpha=0.2)

        # plot kept paths
        for i, choice in enumerate(vertex_list):
            ver_sampled = path.where(path.vertex.isin(choice), drop=True)
            for j, (l, v) in enumerate(ver_sampled.groupby('vertex')):
                c = list(mcolors.TABLEAU_COLORS)[vertex_list[i][j]]
                c_choice = vertex_list[i][j]
                c1 = '#f18b00'
                path_cset=[c1,'navy','lightseagreen','purple'][c_choice]
                axs[i].plot(v.lon, v.lat, c=path_cset)#, lw=0.5)

            axs[i].axis('off')
            axs[i].set_xlim(path.lon.min(),path.lon.max())
            axs[i].set_ylim(path.lat.min(),path.lat.max())
        


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
        path = get_sampled_path('EXP10','interp_1000',post_transect=True)

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
                path = get_sampled_path('EXP10', 
                                    'interp_1000',
                                    post_transect=True)
            else:
                path = get_sampled_path('EXP10', 
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

    def plot_model_and_glider_diff_rotate_bar_norm(self, case, samples):
        '''
        under construction
        
        bar chart of difference between gliders and model patches
        as a percentage, testing sensitivity to rotated paths
        '''

        # initialise figure
        fig = plt.figure(figsize=(3.2,4.0))

        # initialise gridspec
        gs0 = gridspec.GridSpec(ncols=1, nrows=2, right=0.97)#, figure=fig)
        gs1 = gridspec.GridSpec(ncols=4, nrows=1, right=0.97)#, figure=fig)
    
        gs0.update(top=0.92, bottom=0.25, left=0.24, hspace=0.1)
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

        x_pos = np.linspace(1,4,4) # seems to have no effect
        self.render_bg_norm(axs0[0], ds, x_pos=x_pos, stat='mean', rotate=True,
                    label_var='rotation')
        self.render_bg_norm(axs0[1], ds, x_pos=x_pos, stat='std', rotate=True,
                    label_var='rotation')

        for ax in axs0:
            ax.set_ylim(-10,75)

        axs0[0].set_xticks([])
        axs0[0].set_ylabel('% difference\n in mean')
        axs0[1].set_ylabel('% difference\n in standard deviation')
        axs0[1].set_xlabel('path rotation [degrees]')

        axs0[0].text(0.5, 1.1, 
                   'difference in buoyancy gradients\nbetween model and glider',
                     transform=axs0[0].transAxes,
                     ha='center', va='center')

        # add glider paths
        rotations = [0, 90, 180, 270]

        for i, ax in enumerate(axs1):
            if rotations[i] == 0:
                path = get_sampled_path('EXP10', 
                                    'interp_1000',
                                    post_transect=True)
            else:
                path = get_sampled_path('EXP10', 
                                    'interp_1000_rotate_' + str(rotations[i]),
                                    post_transect=True,
                                    rotation=np.radians(rotations[i]))

            c1 = '#f18b00'
            path_cset=[c1,'navy','lightseagreen','purple']
            for i, (l,trans) in enumerate(path.groupby('transect')):
                ax.plot(trans.lon, trans.lat, c=path_cset[int(trans.vertex[0])])
            ax.axis('off')

        plt.savefig(case + '_rotation_' + str(samples) + 
                    '_samples_rolling_bg_norm.png', dpi=600)

    def plot_model_and_glider_diff_rotate_bar_vec(self, case, samples):
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
                path = get_sampled_path('EXP10', 
                                    'interp_1000',
                                    post_transect=True)
            else:
                path = get_sampled_path('EXP10', 
                                    'interp_1000_rotate_' + str(rotations[i]),
                                    post_transect=True,
                                    rotation=np.radians(rotations[i]))

            for (l, v) in path.groupby('vertex'):
                ax.plot(v.lon, v.lat, label=l)
            ax.axis('off')

        plt.savefig(case + '_rotation_' + str(samples) + 
                    '_samples_rolling_bg_vec.png', dpi=600)

    def plot_model_and_glider_diff_rotate_and_transect(self, case, samples):
        ''' 
        bar chart of difference between model patches and gliders
        mean and std across samples

        NB: 31/08/22 introduced bg_norm to render()
        Path now included meso transect. Need to remove.
        '''

        import matplotlib.colors as mcolors

        # ~~~~~ define figure layout ~~~~ #

        # initialise figure
        fig = plt.figure(figsize=(3.2,4.0))

        # initialise gridspec
        gs0 = gridspec.GridSpec(ncols=1, nrows=2, right=0.97)#, figure=fig)
        gs1 = gridspec.GridSpec(ncols=4, nrows=1, right=0.97)#, figure=fig)
    
        # set frame bounds
        gs0.update(top=0.92, bottom=0.25, left=0.18, right=0.95, hspace=0.1)
        gs1.update(top=0.15, bottom=0.02, left=0.18)

        # assign axes to lists
        axs0, axs1 = [], []
        for i in range(2):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(4):
            axs1.append(fig.add_subplot(gs1[i]))

        # ~~~~~ load and render data ~~~~~ #

        x_pos = np.linspace(0.5,4.5,3)
        offset = [-0.3, 0, 0.3]
        path = self.data_path + case
        prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'

        # transect selection
        ds_t = xr.open_dataset(path + prepend + 'glider_vertex_diff_threshold' +
                         self.append + '_percent_rolling_' + str(samples) + 
                                 '_samples.nc')

        ds = ds_t.sel(vertex_choice=['parallel','cross','all'])
        self.render_bg_norm(axs0[0], ds, x_pos=x_pos, stat='mean')
        self.render_bg_norm(axs0[1], ds, x_pos=x_pos, stat='std')

        # rotations
        x_pos = np.array([6.5])
        ds = xr.open_dataset(path + prepend +'glider_rotate_diff_threshold' +
                             self.append + '_percent_rolling_' + str(samples) + 
                             '_samples.nc').sel(rotation=90)

        #ds_r = ds_r.assign({'vertex_choice':'90'})
        #ds_r = ds_r.expand_dims('vertex_choice').set_coords('vertex_choice')
        #for key in ds_r:
        #    print (key)
        #    ds_r = ds_r.rename({key:key.rstrip('_rotate')})
        #ds = xr.merge([ds_r,ds_t], compat='minimal')
        print (ds)
        #for key,var in ds_t.groupby('vertex_choice'):
        #    print (var.dropna('time_counter'))
        #    plt.figure(10)
        #    plt.plot(var.isel(sample=1).diff_bg_norm_mean, label=key)
        #plt.legend()
        #plt.show()
        self.render_bg_norm(axs0[0], ds, x_pos=x_pos, stat='mean', rotate=True,
                            label_var='rotation')
        self.render_bg_norm(axs0[1], ds, x_pos=x_pos, stat='std', rotate=True,
                            label_var='rotation')

        # ~~~~ axes parameters ~~~~ #

        for ax in axs0:
             ax.set_ylim(-5,70)

        import matplotlib.colors as mcolors

        # add glider paths
        vertex_list     = [[1,3],[0,2],[0,1,2,3]]
        vertex_list_inv = [[0,2],[1,3]]
        path = get_sampled_path('EXP10','interp_1000', post_transect=True,
                                     cut_meso=True)

        colours = ['#dad1d1', '#7e9aa5', '#55475a']
        # plot removed paths
        for i, choice in enumerate(vertex_list_inv):
            ver_sampled_inv = path.where(path.vertex.isin(choice), drop=True)
            for (l, v) in ver_sampled_inv.groupby('vertex'):
                axs1[i].plot(v.lon, v.lat, c=colours[0], alpha=1.0)

        # plot kept paths
        for i, choice in enumerate(vertex_list):
            ver_sampled = path.where(path.vertex.isin(choice), drop=True)
            for j, (l, v) in enumerate(ver_sampled.groupby('vertex')):
                c = list(mcolors.TABLEAU_COLORS)[vertex_list[i][j]]
                c_choice = vertex_list[i][j]
                c1 = '#f18b00'
                path_cset=[c1,'navy','lightseagreen','purple'][c_choice]
                axs1[i].plot(v.lon, v.lat, c=path_cset)#, lw=0.5)

            axs1[i].axis('off')
            axs1[i].set_xlim(path.lon.min(),path.lon.max())
            axs1[i].set_ylim(path.lat.min(),path.lat.max())


        axs0[0].set_xticks([])
        axs0[1].set_xticks(np.linspace(0.5,6.5,4))
        axs0[1].set_xticklabels(['parallel', 'cross', 'standard',
                                 r'90$^{\circ}$ rotation'])
        axs0[0].set_ylabel('% difference in\n temporal mean')
        axs0[1].set_ylabel('% difference in\n temporal standard deviation')
        axs0[1].set_xlabel('path choice')

        axs0[0].text(0.5, 1.1, 
                   'difference in buoyancy gradients\nbetween model and glider',
                     transform=axs0[0].transAxes,
                     ha='center', va='center')

        ## add glider paths
        path = get_sampled_path('EXP10', 
                            'interp_1000_rotate_90',
                            post_transect=True,
                            rotation=np.radians(90))

        c1 = '#f18b00'
        path_cset=[c1,'navy','lightseagreen','purple']
        for i, (l,trans) in enumerate(path.groupby('transect')):
            axs1[3].plot(trans.lon,trans.lat,c=path_cset[int(trans.vertex[0])])
        axs1[-1].axis('off')

        # save
        plt.savefig('multi_model_rotate_and_transect_skill_' + str(samples) + 
                    '_samples_rolling_norm_bootstrap.png', dpi=600)
        

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

# ~~~~ plot vertex percentage error ~~~~ #
#m.plot_model_and_glider_diff_bar(cases, samples=50)
#m.plot_model_and_glider_diff_bar(cases, samples=200)
m = glider_path_geometry_plotting()
##m.plot_model_and_glider_diff_rotate_bar_norm('EXP10', samples=100)
m.plot_model_and_glider_diff_rotate_and_transect('EXP10', samples=100)
#m.plot_model_and_glider_diff_rotate_timeseries('EXP10', samples=100)
#m.plot_model_and_glider_diff_vertex_timeseries('EXP10', samples=100)
#m.plot_model_and_glider_diff_vertex_bar_norm(['EXP10'], samples=100)


# ~~~~ save file of vertex percentage error ~~~~ #

#cases = ['EXP10']
#for  case in cases:
#    m = glider_path_geometry(case)
#    # model patches
#    #m.save_glider_samples(rotation=180)
#    #m.save_glider_samples(rotation=270)
##    m.get_glider_samples()
#    #m_mean = m.save_model_buoyancy_gradients_patch_set(stats='mean',
#    #                                                    rolling=True)
#    #m_std  = m.save_model_buoyancy_gradients_patch_set(stats='std',
#    #                                                    rolling=True)
#    #m_med  = m.save_model_buoyancy_gradients_patch_set(stats='median',
#    #                                                    rolling=True)
#    m.get_sample_and_glider_diff_vertex_set(percentage=True, samples=100,
#                                            rolling=True)
#    m.get_sample_and_glider_diff_rotation(percentage=True, samples=100,
#                                            rolling=True)
##    m.get_sample_and_glider_diff_rotation(percentage=True, samples=50)
##    m.get_sample_and_glider_diff_rotation(percentage=True, samples=200)


#m.get_model_buoyancy_gradients_patch_set(ml=True)
#m.plot_histogram_buoyancy_gradients_and_samples()
#m.get_vertex_sets_all()
