import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.dates as mdates
import numpy as np
import dask
import matplotlib
import datetime
import matplotlib.gridspec as gridspec
import scipy.stats as stats
#import itertools
from get_transects import get_transects

#matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 8})
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

class bootstrap_glider_samples(object):
    '''
    for ploting bootstrap samples of buoyancy gradients
    '''

    def __init__(self, case, offset=False, var='b_x_ml', load_samples=True,
                 subset='', transect=False, interp='1000'):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'
        self.var = var

        self.hist_range = (0,2e-8)
        def expand_sample_dim(ds):
            ds['lon_offset'] = ds.attrs['lon_offset']
            ds['lat_offset'] = ds.attrs['lat_offset']
            ds = ds.set_coords(['lon_offset','lat_offset','time_counter'])
            da = ds[var]
       
            # these two lines fail
            #da = da.sel(ctd_depth=10, method='nearest')
            #da = get_transects(da)
            return da
        if load_samples:
            sample_size = 100
        else:
            sample_size = 1

        # subset domain
        self.subset=subset
        self.append=''
        patch=''
        if self.subset=='north':
            self.append='_n'
            patch = '_north_patch'
        if self.subset=='south':
            self.append='_s'
            patch = '_south_patch'
        
        self.append = self.append + '_interp_' + interp

        # load samples
        prep = 'GliderRandomSampling/glider_uniform_interp_'+ interp + \
               patch + '.nc'
        #sample_list = [self.data_path + prep + 
        #               str(i).zfill(2) + '.nc' for i in range(sample_size)]
        #self.samples = xr.open_mfdataset(sample_list, 
        #                             combine='nested', concat_dim='sample',
        #                             preprocess=expand_sample_dim).load()
        self.samples = xr.open_dataset(self.data_path + prep,
                                       chunks={'sample':1})[var]

        # depth average
        #self.samples = self.samples.mean('ctd_depth', skipna=True)
        self.samples = self.samples.sel(ctd_depth=10, method='nearest')
        #self.samples = self.samples.sel(ctd_depth=10, method='nearest')


        # unify times
        self.samples['time_counter'] = self.samples.time_counter.isel(sample=0)
    
        # get transects and remove 2 n-s excursions
        # this cannot currently deal with depth-distance data (1-d only)
        old=False
        print (self.samples)
        if transect:
            if old:
                sample_list = []
                for i in range(self.samples.sample.size):
                    print ('sample: ', i)
                    var10 = self.samples.isel(sample=i)
                    sample_transect = get_transects(var10, offset=True)
                    sample_list.append(sample_transect)
                self.samples=xr.concat(sample_list, dim='sample')
            else:
                self.samples = self.samples.groupby('sample').map(
                                                     get_transects, offset=True)

        # set time to float for averaging
        float_time = self.samples.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        self.samples['time_counter'] = clean_float_time

 
        # absolute value of buoyancy gradients
        self.samples = np.abs(self.samples)
        #self.samples['b_x_ml'] = np.abs(self.samples.b_x_ml)

        #for i in range(self.samples.sample.size):
        #    self.samples[i] = self.get_transects(self.samples.isel(sample=i))

        #self.model_rho = xr.open_dataset(self.data_path + 'rho.nc')
        #self.model_mld = xr.open_dataset(self.data_path +
        #                   'SOCHIC_PATCH_3h_20120101_20121231_grid_T.nc').mld



    def get_model_buoyancy_gradients(self, ml=False):
        ''' restrict the model time to glider time '''
    
        bg = xr.open_dataset(config.data_path() + self.case +
                             '/SOCHIC_PATCH_3h_20121209_20130331_bg.nc',
                             chunks='auto')
                             #chunks={'x':10,'y':10})
        mld = xr.open_dataset(config.data_path() + self.case +
                              '/SOCHIC_PATCH_3h_20121209_20130331_grid_T.nc',
                              chunks={'time_counter':10}).mldr10_3.load()
        mld = mld.isel(x=slice(1,-2),y=slice(1,-2))

        if ml:
            bg = bg.where(bg.deptht < mld)
        
        # for some reason this needs to be replicated from __init__ 
        float_time = self.samples.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        print (clean_float_time.min())
        start = clean_float_time.min().astype('datetime64[ns]')
        end   = clean_float_time.max().astype('datetime64[ns]')

        print (' ')
        print (' ')
        print ('start', start.values)
        print ('end', end.values)
        print (' ')
        print (' ')
        self.bg = bg.sel(time_counter=slice(start,end))

    def get_glider_timeseries(self, ensemble_range=range(1,2), save=False):
        ''' get upper and lower deciles and mean time series of
            of glider sample sets
            + weekly and daily std
        '''
        set_size = self.samples.sizes['sample']

        ensemble_set=[]
        for n in ensemble_range:
            # get random group
            random = np.random.randint(set_size, size=(set_size,n))

            d_mean_set = [] # set of time_series
            w_mean_set = [] # set of time_series
            d_std_set = [] # set of time_series
            w_std_set = [] # set of time_series
            mean_set = []
            for i, samples in enumerate(random):
                print (i)
                sample_set = self.samples.isel(sample=samples)

                # mean
                sample_set = sample_set.reset_coords('time_counter') # retain t
                #sample_mean = sample_set.mean(['sample','ctd_depth']) 
                sample_mean = sample_set.mean(['sample']) 
                sample_mean = sample_mean.set_coords('time_counter')
                mean_set.append(sample_mean)

                # standard deviations
                sample_set['time_counter'] = sample_set.time_counter.astype(
                                                               'datetime64[ns]')
                sample_set = sample_set.stack(
                                 {'ensemble':('distance','sample')})
                                #{'ensemble':('distance','sample','ctd_depth')})
                sample_set = sample_set.swap_dims({'ensemble':'time_counter'})
                sample_set = sample_set.dropna('time_counter').sortby(
                                                                 'time_counter')
                dims=['time_counter']
                d_mean = sample_set.resample(
                                   time_counter='1D',skipna=True).mean(dim=dims)
                w_mean = sample_set.resample(
                                   time_counter='1W',skipna=True).mean(dim=dims)
                d_std = sample_set.resample(
                                    time_counter='1D',skipna=True).std(dim=dims)
                w_std = sample_set.resample(
                                    time_counter='1W',skipna=True).std(dim=dims)
                d_mean_set.append(d_mean)
                w_mean_set.append(w_mean)
                d_std_set.append(d_std)
                w_std_set.append(w_std)

            d_mean = xr.concat(d_mean_set, dim='sets')
            w_mean = xr.concat(w_mean_set, dim='sets')
            d_std = xr.concat(d_std_set, dim='sets')
            w_std = xr.concat(w_std_set, dim='sets')
            mean_arr = xr.concat(mean_set, dim='sets')
                
            # rename time for compatability
            d_mean    = d_mean.rename({'time_counter':'day'})
            w_mean    = w_mean.rename({'time_counter':'day'})
            d_mean    = d_mean.rename({'b_x_ml':'b_x_ml_day_mean'})
            w_mean    = w_mean.rename({'b_x_ml':'b_x_ml_week_mean'})
            d_std    = d_std.rename({'time_counter':'day'})
            w_std    = w_std.rename({'time_counter':'day'})
            d_std    = d_std.rename({'b_x_ml':'b_x_ml_day_std'})
            w_std    = w_std.rename({'b_x_ml':'b_x_ml_week_std'})
            mean_arr = mean_arr.rename({'b_x_ml':'b_x_ml_mean'})

            ts_array = xr.merge([d_mean,w_mean,d_std,w_std,mean_arr])
            ts_array = ts_array.assign_coords({'time_counter_mean':
                                            ts_array.time_counter.mean('sets')})

            # stats
            set_mean = ts_array.mean('sets')
            set_quant = ts_array.quantile([0.05,0.1,0.25,0.75,0.9,0.95],'sets')
            set_mean = set_mean.rename({
                                'b_x_ml_day_mean':'b_x_ml_day_mean_set_mean',
                                'b_x_ml_week_mean':'b_x_ml_week_mean_set_mean',
                                'b_x_ml_day_std':'b_x_ml_day_std_set_mean',
                                'b_x_ml_week_std':'b_x_ml_week_std_set_mean',
                                'b_x_ml_mean':'b_x_ml_mean_set_mean'})
            set_quant = set_quant.rename({
                                'b_x_ml_day_mean':'b_x_ml_day_mean_set_quant',
                                'b_x_ml_week_mean':'b_x_ml_week_mean_set_quant',
                                'b_x_ml_day_std':'b_x_ml_day_std_set_quant',
                                'b_x_ml_week_std':'b_x_ml_week_std_set_quant',
                                'b_x_ml_mean':'b_x_ml_mean_set_quant'})

            # create ds
            ds = xr.merge([ts_array,set_mean,set_quant], compat='override')
            ensemble_set.append(ds)
       
        # group of stats for different ensemble sizes
        ds_ensembles = xr.concat(ensemble_set, dim='ensemble_size')

        if save:
            ds_ensembles.to_netcdf(self.data_path + '/BgGliderSamples' + 
                   '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_timeseries' +
                    self.append + '.nc')

    def get_full_model_day_week_std(self, save=False):
        ''' get std x,y,day/week of full model data '''

        self.get_model_buoyancy_gradients(ml=False)
        self.bg = self.bg.sel(deptht=10, method='nearest').load()
        self.bg = np.abs(self.bg)

        self.bg['bg_norm'] = (self.bg.bx ** 2 + self.bg.by ** 2) ** 0.5

        # subset model
        if self.subset=='north':
            self.bg = self.bg.where(self.bg.nav_lat>-59.9858036, drop=True)
        if self.subset=='south':
            self.bg = self.bg.where(self.bg.nav_lat<-59.9858036, drop=True)
 
        dims=['x','y','time_counter']
        daily_std = self.bg.resample(time_counter='1D').std(dim=dims)
        weekly_std = self.bg.resample(time_counter='1W').std(dim=dims)

        # rename time for compatability
        daily_std = daily_std.rename({'time_counter':'day'})
        weekly_std = weekly_std.rename({'time_counter':'day'})

        bg_d = daily_std.rename({'bx':'bx_ts_day_std', 'by':'by_ts_day_std',
                                 'bg_norm':'bg_norm_ts_day_std'})
        bg_w = weekly_std.rename({'bx':'bx_ts_week_std', 'by':'by_ts_week_std',
                                 'bg_norm':'bg_norm_ts_week_std'})
        bg_stats = xr.merge([bg_w,bg_d])
        if save:
            bg_stats.to_netcdf(self.data_path + '/BgGliderSamples' +
              '/SOCHIC_PATCH_3h_20121209_20130331_bg_day_week_std_timeseries' + 
                    self.append + '.nc')

    def get_full_model_day_week_sdt_and_mean_bg(self, save=False,
                                                space_quantile=False):
        ''' 
        get mean bg for each day/week of full model data 
        out put retains x and y coordinates
        '''

        self.get_model_buoyancy_gradients(ml=False)
        self.bg = self.bg.sel(deptht=10, method='nearest').load()
        self.bg = np.abs(self.bg)
        self.bg = np.abs(self.bg)

        self.bg['bg_norm'] = (self.bg.bx ** 2 + self.bg.by ** 2) ** 0.5

        # subset model
        if self.subset=='north':
            self.bg = self.bg.where(self.bg.nav_lat>-59.9858036, drop=True)
        if self.subset=='south':
            self.bg = self.bg.where(self.bg.nav_lat<-59.9858036, drop=True)
 
        # reduce in x,y by finding quantiles
        if space_quantile:
            self.bg = self.bg.quantile([0.1,0.2,0.5,0.8,0.9], ('x','y'))
            self.append = self.append + '_space_quantile'

        # stats dims
        dims=['time_counter']

        # time means
        daily_mean = self.bg.resample(time_counter='1D').mean(dim=dims)
        weekly_mean = self.bg.resample(time_counter='1W').mean(dim=dims)

        # time stds
        daily_std = self.bg.resample(time_counter='1D').std(dim=dims)
        weekly_std = self.bg.resample(time_counter='1W').std(dim=dims)

        # rename time for compatability
        daily_mean = daily_mean.rename({'time_counter':'day'})
        weekly_mean = weekly_mean.rename({'time_counter':'day'})
        daily_std = daily_std.rename({'time_counter':'day'})
        weekly_std = weekly_std.rename({'time_counter':'day'})

        bg_d_mean = daily_mean.rename({'bx':'bx_ts_day_mean',
                                  'by':'by_ts_day_mean',
                                  'bg_norm':'bg_norm_ts_day_mean'})

        bg_w_mean = weekly_mean.rename({'bx':'bx_ts_week_mean',
                                   'by':'by_ts_week_mean',
                                   'bg_norm':'bg_norm_ts_week_mean'})

        bg_d_std = daily_std.rename({'bx':'bx_ts_day_std',
                                  'by':'by_ts_day_std',
                                  'bg_norm':'bg_norm_ts_day_std'})

        bg_w_std = weekly_std.rename({'bx':'bx_ts_week_std',
                                   'by':'by_ts_week_std',
                                   'bg_norm':'bg_norm_ts_week_std'})

        bg_stats = xr.merge([bg_w_mean,bg_d_mean,bg_w_std,bg_d_std])
        if save:
            bg_stats.to_netcdf(self.data_path + '/BgGliderSamples' +
            '/SOCHIC_PATCH_3h_20121209_20130331_bg_day_week_std_mean_timeseries' 
                     + self.append + '.nc')

    def get_full_model_timeseries_stats(self, save=False):
        ''' 
           get model mean, std, and quantiles time_series
               - buoyancy
        '''

        self.get_model_buoyancy_gradients()
        self.bg = self.bg.sel(deptht=10, method='nearest')
        self.bg = np.abs(self.bg)

        self.bg['bg_norm'] = (self.bg.bx ** 2 + self.bg.by ** 2) ** 0.5

        # subset model
        if self.subset=='north':
            self.bg = self.bg.where(self.bg.nav_lat>-59.9858036, drop=True)
        if self.subset=='south':
            self.bg = self.bg.where(self.bg.nav_lat<-59.9858036, drop=True)

        bg_mean  = self.bg.mean(['x','y'])
        bg_std   = self.bg.std(['x','y'])
        self.bg  = self.bg.chunk(chunks={'x':-1,'y':-1})
        bg_quant = self.bg.quantile([0.1,0.2,0.5,0.8,0.9], ['x','y'])
        bg_mean  = bg_mean.rename({'bx':'bx_ts_mean', 'by':'by_ts_mean',
                                   'bg_norm':'bg_norm_ts_mean'})
        bg_std   = bg_std.rename({'bx':'bx_ts_std', 'by':'by_ts_std',
                                  'bg_norm':'bg_norm_ts_std'})
        bg_quant   = bg_quant.rename({'bx':'bx_ts_quant', 'by':'by_ts_quant',
                                  'bg_norm':'bg_norm_ts_quant'})
        bg_stats = xr.merge([bg_mean,bg_std,bg_quant])
        if save:
            bg_stats.to_netcdf(self.data_path + '/BgGliderSamples' +
                    '/SOCHIC_PATCH_3h_20121209_20130331_bg_stats_timeseries' + 
                    self.append + '.nc')

    def get_full_model_timeseries_norm_bg(self, save=False):
        ''' 
        get timeseries of norm bg at z10
        '''

        self.get_model_buoyancy_gradients()
        self.bg = self.bg.sel(deptht=10, method='nearest')
        self.bg = np.abs(self.bg)

        self.bg['bg_norm'] = (self.bg.bx ** 2 + self.bg.by ** 2) ** 0.5

        # subset model
        if self.subset=='north':
            self.bg = self.bg.where(self.bg.nav_lat>-59.9858036, drop=True)
        if self.subset=='south':
            self.bg = self.bg.where(self.bg.nav_lat<-59.9858036, drop=True)

        self.bg  = self.bg.chunk(chunks={'x':-1,'y':-1})
        if save:
            self.bg.to_netcdf(self.data_path +
                  '/SOCHIC_PATCH_3h_20121209_20130331_bg_norm_timeseries_z10' + 
                    self.append + '.nc')

    def get_hist_stats(self, hist_set, bins):    
        ''' get mean, lower and upper deciles of group of histograms '''
        bin_centers = (bins[:-1] + bins[1:]) / 2
        hist_array = xr.DataArray(hist_set, dims=('sets', 'bin_centers'), 
                                  coords={'bin_centers': bin_centers})
        hist_mean = hist_array.mean('sets')
        hist_l_quant, hist_u_quant = hist_array.quantile([0.1,0.9],'sets')
        return hist_mean, hist_l_quant, hist_u_quant

    def get_rmse_stats(self, hist_set, bins):    
        ''' get mean, lower and upper deciles of group of histograms '''
        bin_centers = (bins[:-1] + bins[1:]) / 2
        glider_hist = xr.DataArray(hist_set, dims=('sets', 'bin_centers'), 
                                  coords={'bin_centers': bin_centers})

        model_hist = self.get_full_model_hist(subset='')

        # rmse :: pred - truth / truth
        frac_diff = (glider_hist - model_hist.hist_norm) / model_hist.hist_norm
        rmsep_diff = np.abs(frac_diff) * 100
        
        # get stats
        hist_mean = rmsep_diff.mean('sets')
        hist_u_quant = rmsep_diff.quantile([0.1,0.9],'sets')
        hist_l_quant, hist_u_quant = rmsep_diff.quantile([0.1,0.9],'sets')

        return hist_mean, hist_l_quant, hist_u_quant


    def get_glider_sampled_hist(self, n=1, save=False, by_time=None):
        '''
        add sample set of means and std to histogram
        n      : sample size
        by_time: get stats over time - i.e. get weekly stats
                 - week is only option for now
        '''
 
        set_size = self.samples.sizes['sample']

        # get random group, shape (bootstrap iters, number of gliders)
        random = np.random.randint(set_size, size=(set_size,n))

        def get_stats_across_hists(samp):
            sample_size = len(samp.time_counter)
            # calculate set of histograms
            hists = []
            for i, sample in enumerate(random):
                sample_set = samp.isel(sample=sample)#.b_x_ml
                set_stacked = sample_set.stack(z=('time_counter','sample'))
                hist, bins = np.histogram(set_stacked.dropna('z', how='all'),
                                   range=self.hist_range, density=True, bins=20)
                #                    range=(1e-9,5e-8), density=True)
                hists.append(hist)
            
            # calculate rmse across histogram set
            rmse_mean, rmse_l_quant, rmse_u_quant = self.get_rmse_stats(
                                                                    hists, bins)

            # calculate spread across histogram set
            hist_mean, hist_l_quant, hist_u_quant = self.get_hist_stats(
                                                                    hists, bins)

            bin_centers = (bins[:-1] + bins[1:]) / 2
            hist_ds = xr.Dataset({'hist_mean':(['bin_centers'], hist_mean),
                                  'hist_l_dec':(['bin_centers'], hist_l_quant),
                                  'hist_u_dec':(['bin_centers'], hist_u_quant),
                                  'rmse_mean':(['bin_centers'], rmse_mean),
                                  'rmse_l_dec':(['bin_centers'], rmse_l_quant),
                                  'rmse_u_dec':(['bin_centers'], rmse_u_quant),
                                  'sample_size':(sample_size)},
                                      coords={
                                  'bin_centers': (['bin_centers'], bin_centers),
                                  'bin_left'   : (['bin_centers'], bins[:-1]),
                                  'bin_right'  : (['bin_centers'], bins[1:])})
            return hist_ds

        samples = self.samples
        samples['time_counter'] = samples.time_counter.astype(
                                   'datetime64[ns]')
        samples = samples.swap_dims({'distance':'time_counter'})
        samples = samples.dropna('time_counter')
        #plt.scatter(np.arange(len(samples.time_counter)),samples.time_counter)
        #plt.show()
        #samples = samples.resample(time_counter='1H').interpolate()

        def get_rolling_hists(ts):
            '''
            calculate distribution of buoyancy gradients over a 
            rolling mean
            in prep: functionality currently only works with regular time steps
            '''

            # get week centred dates
            week_dates = list(samples.time_counter.resample(
                              time_counter='1W').groups.keys())

            # create rolling object as dataset with extra rolled dimension
            rolled = samples.rolling(time_counter=ts, center=True).construct(
                         by_time).sel(time_counter=week_dates, method='nearest')

            # swap time labels
            rolled = rolled.rename({'time_counter':'time'})
            rolled = rolled.rename({by_time:'time_counter'})

            # caculate histograms
            hist_ds = rolled.groupby('time').map(get_stats_across_hists)

            # return labels
            hist_ds = hist_ds.rename({'time':'time_counter'})

            return hist_ds

        date_list = [np.datetime64('2012-12-10 00:00:00') +
                     np.timedelta64(i, 'W')
                     for i in range(16)]
        mid_date = [date_list[i] + (date_list[i+1] - date_list[i])/2
                   for i in range(15)]
        if by_time == 'weekly':
            # split into groups of weeks
            hist_ds = samples.resample(time_counter='1W', skipna=True).map(
                                                         get_stats_across_hists)
        elif by_time == '1W_rolling':
            # split into 1 week samples, sampled by week
            hist_ds = samples.groupby_bins('time_counter', date_list,
                                    labels=mid_date).map(get_stats_across_hists)
            hist_ds = hist_ds.rename({'time_counter_bins':'time_counter'})
        elif by_time == '2W_rolling':
            # split into 2 week samples, sampled by week
            mid_date=mid_date[1:]
            l_dl = date_list[::2] + np.timedelta64(84, 'h')
            l_label = mid_date[::2]
            hist_ds_l = samples.groupby_bins('time_counter', l_dl,
                         labels=l_label).map(get_stats_across_hists)
            u_dl = date_list[1:-1:2] + np.timedelta64(84, 'h')
            u_label = mid_date[1:-1:2]# + np.timedelta64(1, 'W')
            hist_ds_u = samples.groupby_bins('time_counter', u_dl,
                         labels=u_label).map(get_stats_across_hists)
            hist_ds = xr.merge([hist_ds_u, hist_ds_l])
            hist_ds = hist_ds.rename({'time_counter_bins':'time_counter'})
        elif by_time == '3W_rolling':
            # split into 3 week samples, sampled by week
            mid_date=mid_date[1:]
            l_dl = date_list[::3]
            l_label = mid_date[::3]
            hist_ds_l = samples.groupby_bins('time_counter', l_dl,
                         labels=l_label).map(get_stats_across_hists)
            m_dl = date_list[1:-1:3]
            m_label = mid_date[1:-1:3]
            hist_ds_m = samples.groupby_bins('time_counter', m_dl,
                         labels=m_label).map(get_stats_across_hists)
            u_dl = date_list[2:-1:3]
            u_label = mid_date[2:-1:3]
            hist_ds_u = samples.groupby_bins('time_counter', u_dl,
                         labels=u_label).map(get_stats_across_hists)
            hist_ds = xr.merge([hist_ds_u, hist_ds_m, hist_ds_l])
            hist_ds = hist_ds.rename({'time_counter_bins':'time_counter'})
        else:
            # entire timeseries
            hist_ds = get_stats_across_hists(samples)
            
        if save:
            hist_ds.to_netcdf(self.data_path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_' + 
                          self.var + '_glider_' +
                          str(n).zfill(2) + '_hist' + self.append + '.nc')
        return hist_ds

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

    def glider_sample_bootstrap_stats(self, n):
        set_size = self.samples.sizes['sample']
        random = np.random.randint(set_size, size=(set_size,n))

        set_of_means = []
        set_of_quants = []
        set_of_stds = []
        for sample in random:
            sample_set = self.samples.isel(sample=sample).b_x_ml
            print (sample_set)
            sample_set = sample_set.chunk(chunks={'sample':-1})
            set_mean = sample_set.mean(['sample','distance'])
            set_quant = sample_set.quantile([0.1,0.9],['sample','distance'])
            set_std = sample_set.std(['sample','distance'])
            set_of_means.append(set_mean)
            set_of_quants.append(set_quant)
            set_of_stds.append(set_std)

        chunks={'sets':-1}
        set_of_means = xr.concat(set_of_means, 'sets').chunk(chunks)
        set_of_quants = xr.concat(set_of_quants, 'sets').chunk(chunks)
        set_of_stds = xr.concat(set_of_stds, 'sets').chunk(chunks)

        mean = set_of_means.mean()
        quant = set_of_quants.mean()
        std = set_of_stds.mean()
        #quant = set_of_means.quantile([0.1,0.9])

        return mean, quant, std

    def get_bg_z_hist(self, bg, bins=20):
        ''' calculate histogram and assign to xarray dataset '''

        # stack dimensions
        stacked_bgx = bg.bx.stack(z=('time_counter','x','y'))
        stacked_bgy = bg.by.stack(z=('time_counter','x','y'))

        # bg norm - warning: not gridded appropriately on T-pts
        stacked_bg_norm = (stacked_bgx**2 + stacked_bgy**2)**0.5

        # histogram
        hist_x, bins = np.histogram(stacked_bgx.dropna('z', how='all'),
                            range=self.hist_range, density=True, bins=bins)
        hist_y, bins = np.histogram(stacked_bgy.dropna('z', how='all'),
                            range=self.hist_range, density=True, bins=bins)
        hist_norm, bins = np.histogram(
                            stacked_bg_norm.dropna('z', how='all'),
                            range=self.hist_range, density=True, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # assign to dataset
        hist_ds = xr.Dataset({'hist_x':(['bin_centers'], hist_x),
                              'hist_y':(['bin_centers'], hist_y),
                              'hist_norm':(['bin_centers'], hist_norm)},
                   coords={'bin_centers': (['bin_centers'], bin_centers),
                           'bin_left'   : (['bin_centers'], bins[:-1]),
                           'bin_right'  : (['bin_centers'], bins[1:])})
        return hist_ds

    def get_full_model_hist(self, save=False, subset='', by_time=None):
        '''
        make histgram of buoyancy gradients for full model domain
        by_time: splits caluculation over time
                 currently implemeted for weekly splitting only
        '''
        # load buoyancy gradients       
        self.get_model_buoyancy_gradients()

        # reduce data
        self.bg = np.abs(self.bg.sel(deptht=10, method='nearest'))
        #mean_bg = (self.bg.bx + self.bg.by) / 2
        #mean_bg = self.bg.by

        # subset model
        if self.subset=='north':
            self.bg = self.bg.where(self.bg.nav_lat>-59.9858036, drop=True)
        if self.subset=='south':
            self.bg = self.bg.where(self.bg.nav_lat<-59.9858036, drop=True)


        def get_rolling_hists(ts):
            '''
            calculate distribution of buoyancy gradients over a 
            rolling mean
            '''

            # get week centred dates
            week_dates = list(self.bg.time_counter.resample(
            time_counter='1W').groups.keys())

            # create rolling object as dataset with extra rolled dimension
            self.bg = self.bg.chunk(dict(time_counter=162,x=64,y=64))
            rolled = self.bg.rolling(time_counter=ts, center=True).construct(
                         by_time).sel(time_counter=week_dates, method='nearest')

            # swap time labels
            rolled = rolled.rename({'time_counter':'time'})
            rolled = rolled.rename_dims({by_time:'time_counter'})

            # caculate histograms
            hist_ds = rolled.groupby('time').map(self.get_bg_z_hist)

            # return labels
            hist_ds = hist_ds.rename({'time':'time_counter'})

            return hist_ds

        date_list = [np.datetime64('2012-12-10 00:00:00') +
                     np.timedelta64(i, 'W')
                     for i in range(16)]
        mid_date = [date_list[i] + (date_list[i+1] - date_list[i])/2
                   for i in range(15)]
        if by_time == 'weekly':
            # split into groups of weeks
            hist_ds = self.bg.resample(time_counter='1W', skipna=True).map(
                                                         self.get_bg_z_hist)
        elif by_time == '1W_rolling':
            # split into 1 week samples, sampled by week
            hist_ds = self.bg.groupby_bins('time_counter', date_list,
                                    labels=mid_date).map(self.get_bg_z_hist)
            hist_ds = hist_ds.rename({'time_counter_bins':'time_counter'})
        elif by_time == '2W_rolling':
            # split into 2 week samples, sampled by week
            mid_date=mid_date[1:]
            l_dl = date_list[::2] + np.timedelta64(84, 'h')
            l_label = mid_date[::2]
            hist_ds_l = self.bg.groupby_bins('time_counter', l_dl,
                         labels=l_label).map(self.get_bg_z_hist)
            u_dl = date_list[1:-1:2] + np.timedelta64(84, 'h')
            u_label = mid_date[1:-1:2]# + np.timedelta64(1, 'W')
            hist_ds_u = self.bg.groupby_bins('time_counter', u_dl,
                         labels=u_label).map(self.get_bg_z_hist)
            hist_ds = xr.merge([hist_ds_u, hist_ds_l])
            hist_ds = hist_ds.rename({'time_counter_bins':'time_counter'})
        elif by_time == '3W_rolling':
            # split into 3 week samples, sampled by week
            mid_date=mid_date[1:]
            l_dl = date_list[::3]
            l_label = mid_date[::3]
            hist_ds_l = self.bg.groupby_bins('time_counter', l_dl,
                         labels=l_label).map(self.get_bg_z_hist)
            m_dl = date_list[1:-1:3]
            m_label = mid_date[1:-1:3]
            hist_ds_m = self.bg.groupby_bins('time_counter', m_dl,
                         labels=m_label).map(self.get_bg_z_hist)
            u_dl = date_list[2:-1:3]
            u_label = mid_date[2:-1:3]
            hist_ds_u = self.bg.groupby_bins('time_counter', u_dl,
                         labels=u_label).map(self.get_bg_z_hist)
            hist_ds = xr.merge([hist_ds_u, hist_ds_m, hist_ds_l])
            hist_ds = hist_ds.rename({'time_counter_bins':'time_counter'})
        else:
            # entire timeseries
            hist_ds = self.get_bg_z_hist(self.bg)

        if save:
            hist_ds.to_netcdf(self.data_path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                          self.append + '.nc')
        return hist_ds

    def get_sampled_model_hist(self):
        ''' return mean and std of sampled model hists within a sample set '''


        ### this is for testing how good the glider is at sampling 
        ### the patch of ocean it is in rather than a given region
        ### no need to have n samples

        hists_x, hists_y = [], []
        print ('MODEL')
        for (label, sample_set) in self.latlon_lims.groupby('sample_set'):
            print ('sample set', label)
            stacked_bgx, stacked_bgy = [], []
            for (label, group) in sample_set.groupby('sample'):
                print ('sample', label)
                subset_bg = self.bg.where((self.bg.nav_lon > group.x0) &
                                          (self.bg.nav_lon < group.x1) &
                                          (self.bg.nav_lat > group.y0) &
                                          (self.bg.nav_lat < group.y1),
                                           drop=True)
                stacked_bgx.append(
                                subset_bg.bx.stack(z=('time_counter','x','y')))
                stacked_bgy.append(
                                subset_bg.by.stack(z=('time_counter','x','y')))
            stacked_bgx = xr.concat(stacked_bgx, dim='z')
            stacked_bgy = xr.concat(stacked_bgy, dim='z')

            hist_x, bins = np.histogram(stacked_bgx.dropna('z', how='all'),
                                 range=self.hist_range, density=True, bins=100)
            hist_y, bins = np.histogram(stacked_bgy.dropna('z', how='all'),
                                 range=self.hist_range, density=True, bins=100)
            hists_x.append(hist_x)
            hists_y.append(hist_y)
        x_mean, x_l_dec, x_u_dec = self.get_hist_stats(hists_x, bins)
        y_mean, y_l_dec, y_u_dec = self.get_hist_stats(hists_y, bins)
        return x_mean, x_l_dec, x_u_dec, y_mean, y_l_dec, y_u_dec

    def render_glider_sample_set(self, n=1, c='green', style='plot'):
        ds = xr.open_dataset(self.data_path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_' +
                           str(n).zfill(2) + '_hist' + self.append + '.nc')
        if style=='bar':
            self.ax.bar(ds.bin_left, 
                    ds.hist_u_dec - ds.hist_l_dec, 
                    width=ds.bin_right - ds.bin_left,
                    color=c,
                    alpha=0.2,
                    bottom=ds.hist_l_dec, 
                    align='edge',
                    label='gliders: ' + str(n))
            self.ax.scatter(ds.bin_centers, ds.hist_mean, c=c, s=4, zorder=10)
        if style=='plot':
            self.ax.fill_between(ds.bin_centers, ds.hist_l_dec,
                                                 ds.hist_u_dec,
                                 color=c, edgecolor=None, alpha=0.2)
            self.ax.plot(ds.bin_centers, ds.hist_mean, c=c, lw=0.8,
                         label='gliders: ' + str(n))

    def add_model_means(self, style='plot'):
        ds = xr.open_dataset(self.data_path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                        self.append + '.nc')
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

    def add_model_bootstrapped_samples(self):
        ''' add model buoyancy gradients and std as a bars '''

        # load buoyancy gradients       
        self.get_model_buoyancy_gradients()

        #self.bg = self.bg.mean('deptht')
        self.bg = self.bg.sel(deptht=10, method='nearest')
        self.bg = np.abs(self.bg)

        self.get_sampled_model_hist()

    def add_model_hist(self):
        '''
        add model buoyancy gradient of means and std to histogram
        '''

        #abs_bg = np.abs(bg)
        #abs_bg.where(abs_bgy<2e-8, drop=True)

        # load buoyancy gradients       
        self.get_model_buoyancy_gradients()

        #self.bg = self.bg.mean('deptht')
        # 10 m depth
        self.bg = self.bg.sel(deptht=10, method='nearest')

        self.bg = np.abs(self.bg)
        self.bg = self.bg.where(self.bg < 1e-7, drop=True)
        stacked_bgx = self.bg.bgx.stack(z=('time_counter','x','y'))
        stacked_bgy = self.bg.bgy.stack(z=('time_counter','x','y'))
        
        print (stacked_bgx)
        plt.hist(stacked_bgx, bins=20, density=True, alpha=0.3,
                 label='model bgx', fill=False, edgecolor='red',
                 histtype='step')
        plt.hist(stacked_bgy, bins=20, density=True, alpha=0.3,
                 label='model bgy', fill=False, edgecolor='blue',
                 histtype='step')
        

    def plot_histogram_buoyancy_gradients_and_samples(self):
        ''' 
        plot histogram of buoyancy gradients 
        n = sample_size
        '''

        self.figure, self.ax = plt.subplots(figsize=(4.5,4.0))

        sample_sizes = [1, 4, 20]
        colours = ['g', 'b', 'r', 'y', 'c']

        for i, n in enumerate(sample_sizes):
            print ('sample', i)
            self.render_glider_sample_set(n=n, c=colours[i], style='bar')
        print ('model')
        self.add_model_means(style='bar')

        self.ax.set_xlabel('Buoyancy Gradient')
        self.ax.set_ylabel('PDF')

        plt.legend()
        self.ax.set_xlim(self.hist_range[0], self.hist_range[1])
        self.ax.set_ylim(0, 3e8)
        plt.savefig(self.case + '_bg_sampling_skill' + self.append + '.png',
                    dpi=600)


    def plot_rmse_over_ensemble_sizes(self):
        ''' plot the root mean squared error of the 1 s.d. from the 
            **real** mean
        '''
        m = xr.open_dataset(self.data_path + 
                     '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                     self.append + '.nc')
        
        def pre_proc(ds):
            ds = ds.expand_dims('ensemble_size')
            return ds
         
        prep = '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_'
        ensemble_list = [self.data_path + prep + str(i).zfill(2) + '_hist' + 
                         self.append + '.nc'
                         for i in range(1,31)]
        ensembles = xr.open_mfdataset(ensemble_list, 
                                   combine='nested', concat_dim='ensemble_size',
                                     preprocess=pre_proc).load()
        ensembles = ensembles.assign_coords(ensemble_size=np.arange(1,31))

        m_bg_abs = (0.5*(m.hist_x**2 + m.hist_y**2))** 0.5

        # rmse
        def rmsep(pred, true):
            norm = (pred - true)/true 
            return np.sqrt(((norm)**2).mean(dim='bin_centers')) * 100

        rmse_l = rmsep(ensembles.hist_l_dec, m_bg_abs)
        rmse_u = rmsep(ensembles.hist_u_dec, m_bg_abs)
        rmse_mean = rmsep(ensembles.hist_mean, m_bg_abs)

        fig, ax = plt.subplots(1)
        #ax.plot(m_bg_abs, c='black')
        #ax.plot(m.hist_x, c='navy')
        #ax.plot(m.hist_y, c='navy')
        #ax.plot(ensembles.hist_l_dec.isel(ensemble_size=1), c='green')
        #ax.plot(ensembles.hist_u_dec.isel(ensemble_size=1), c='green')
        #ax.plot(ensembles.hist_l_dec.isel(ensemble_size=19), c='red')
        #ax.plot(ensembles.hist_u_dec.isel(ensemble_size=19), c='red')
        ax.plot(rmse_u.ensemble_size, rmse_u, c='navy', label='upper decile')
        ax.plot(rmse_l.ensemble_size, rmse_l, c='green', label='lower decile')
        #ax.plot(rmse_mean.ensemble_size, rmse_mean, c='black')

        plt.legend()

        ax.set_ylim(0,75)
        ax.set_xlabel('ensemble size')
        ax.set_ylabel('RMSE of buoyancy gradients (%)')

        plt.savefig(self.case + '_bg_RMSE' + self.append + '.png', dpi=600)
        
    def plot_error_bars(self):
        
        means = []
        quants= []
        stds= []
        sample_sizes = [1, 2, 4, 10, 20]
        for n in sample_sizes:
            print ('n     :', n)
            mean , quant, std = self.glider_sample_bootstrap_stats(n)
            means.append(mean.values)
            #quants.append(quant.values)
            stds.append([mean.values - std.values, mean.values + std.values])
        #quants = np.transpose(np.array(quants))
        stds = np.transpose(np.array(stds))

        #print (quants.shape)
        plt.figure()
        plt.errorbar(sample_sizes, means, stds)
        plt.show()

    def plot_timeseries(self):

        def pre_proc(ds):
            ds = ds.expand_dims('ensemble_size')
            return ds

        # get data
        prep = 'BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_'
        ensemble_list = [self.data_path + prep + str(i).zfill(2) +
                       '_timeseries' + self.append + '.nc' for i in range(1,31)]
        ensembles = xr.open_mfdataset(ensemble_list, 
                                   combine='nested', concat_dim='ensemble_size',
                                     preprocess=pre_proc).load()
        ensembles = ensembles.assign_coords(ensemble_size=np.arange(1,31))
        m = xr.open_dataset(self.data_path + 'BgGliderSamples' + 
                     '/SOCHIC_PATCH_3h_20121209_20130331_bg_stats_timeseries' + 
                          self.append + '.nc')
        
        # define fig
        self.figure, self.ax = plt.subplots(figsize=(6.5,4.0))

        # plot
        ensemble_list = [1,4,30]
        colours = ['green', 'red', 'navy', 'orange']
        for i, l in enumerate(ensemble_list):
            e = ensembles.sel(ensemble_size=l)
            e = e.reset_coords('time_counter').mean('sets')
            e['time_counter'] = e.time_counter / 1e9 
            unit = "seconds since 1970-01-01 00:00:00"
            e.time_counter.attrs['units'] = unit
            e = xr.decode_cf(e)
            e = e.where((e.time_counter>np.datetime64('2012-12-15')) &
                        (e.time_counter<np.datetime64('2013-03-15')))
            #e = e.sel(time_counter=slice('2012-12-15','2013-03-15'))
            print (e)
            self.ax.fill_between(e.time_counter, 
                                 e.bg_ts_dec.sel(quantile=0.1),
                                 e.bg_ts_dec.sel(quantile=0.9),
                                 color=colours[i], edgecolor=None, alpha=1.0,
                                 label=str(l))
        m_bg_abs = (0.5*(m.bx_ts_mean**2 + m.by_ts_mean**2))** 0.5
        m_bg_abs = m_bg_abs.sel(time_counter=
                                slice('2012-12-15','2013-03-15 00:00:00'))
        self.ax.plot(m_bg_abs.time_counter, m_bg_abs, c='cyan', lw=0.8,
                     label='model mean')

        # legend
        plt.legend(title='ensemble size', fontsize=6)
 
        # axis limits
        self.ax.set_xlim(e.time_counter.min(skipna=True),
                         e.time_counter.max(skipna=True))
        self.ax.set_ylim(-1e-8,1.3e-7)

        # add labels
        self.ax.set_xlabel('time')
        self.ax.set_ylabel('buoyancy gradients')

        # save
        plt.savefig(self.case + '_bg_ensemble_timeseries' + self.append +
                    '.png', dpi=600)

    def plot_quantify_delta_bg(self, t0='2013-01-01', t1='2013-03-15'):
        ''' plot glider and model changes in buoyancy gradients over time '''

        # time series data 
        def pre_proc(ds):
            ds = ds.expand_dims('ensemble_size')
            return ds

        # hack to remove interp from append (can this be removed from __init__?)
        self.append = self.append.split('_interp')[0]
        prep = 'BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_'
        ensemble_list = [self.data_path + prep + str(i).zfill(2) +
                       '_timeseries' + self.append
                        + '.nc' for i in range(1,31)]
        ensembles = xr.open_mfdataset(ensemble_list, 
                                   combine='nested', concat_dim='ensemble_size',
                                     preprocess=pre_proc).load()
        ensembles = ensembles.assign_coords(ensemble_size=np.arange(1,31))
        m_ts = xr.open_dataset(self.data_path + 'BgGliderSamples' + 
                     '/SOCHIC_PATCH_3h_20121209_20130331_bg_stats_timeseries' + 
                          self.append + '.nc')

        # initialise figure
        fig = plt.figure(figsize=(6.5, 4.5))
        gs0 = gridspec.GridSpec(ncols=1, nrows=1)
        gs1 = gridspec.GridSpec(ncols=2, nrows=1)
        gs0.update(top=0.99, bottom=0.50, left=0.10, right=0.80)
        gs1.update(top=0.38, bottom=0.09, left=0.10, right=0.80, wspace=0.1)

        axs0 = fig.add_subplot(gs0[0])
        axs1 = []
        for i in range(2):
            axs1.append(fig.add_subplot(gs1[i]))

        # plot interdecile range of glider samples
        ensemble_list = [1,4,30]
        #colours = ['lightgrey', 'gray', 'black', 'orange']
        #colours = ['lightsalmon', 'cadetblue', 'black', 'orange']
        fill = []
        colours = ['#dad1d1', '#7e9aa5', '#55475a']
        for i, l in enumerate(ensemble_list):
            e = ensembles.sel(ensemble_size=l)
            e = e.reset_coords('time_counter').mean('sets')
            e['time_counter'] = e.time_counter / 1e9 
            unit = "seconds since 1970-01-01 00:00:00"
            e.time_counter.attrs['units'] = unit
            e = xr.decode_cf(e)
            e = e.where((e.time_counter>np.datetime64('2012-12-15')) &
                        (e.time_counter<np.datetime64('2013-03-15')))
            #e = e.sel(time_counter=slice('2012-12-15','2013-03-15'))
            fill.append(axs0.fill_between(e.time_counter, 
                                  e.bg_ts_dec.sel(quantile=0.1),
                                  e.bg_ts_dec.sel(quantile=0.9),
                                  color=colours[i], edgecolor=None, alpha=1.0,
                                  label=str(l)))
        

        # reneder model time series
        c = '#fd0000'
        c = '#dd175f'
        c = 'turquoise'
        #c = 'lightseagreen'
        m_ts = m_ts.sel(time_counter=slice('2012-12-15','2013-03-15 00:00:00'))
        #axs0.plot(m_ts.time_counter, m_ts.bg_norm_ts_quant.sel(quantile=0.2),
        #              c=c, lw=0.8, ls='--', label='__nolegend__')
        #p = axs0.plot(m_ts.time_counter,
        #              m_ts.bg_norm_ts_quant.sel(quantile=0.5),
        #              c=c, lw=0.8, ls='-', label='full model')
        #axs0.plot(m_ts.time_counter, m_ts.bg_norm_ts_quant.sel(quantile=0.8),
        #              c=c, lw=0.8, ls='--', label='__nolegend__')

        end = m_ts.isel(time_counter=-1)

        # add model labels
        print (end)
        axs0.text(end.time_counter + np.timedelta64(12, 'h'),
                 end.bg_norm_ts_quant.sel(quantile=0.1),
                 'Lower Decile', ha='left',va='center', fontsize=6, c=c)
        axs0.text(end.time_counter + np.timedelta64(12, 'h'),
                  end.bg_norm_ts_quant.sel(quantile=0.5),
                 'Median', ha='left',va='center', fontsize=6, c=c)
        axs0.text(end.time_counter + np.timedelta64(12, 'h'),
                  end.bg_norm_ts_quant.sel(quantile=0.9),
                 'Upper Decile', ha='left',va='center', fontsize=6, c=c)


        # diff data
        g = xr.open_dataset(self.data_path + 'BgGliderSamples' + 
                    '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_timeseries' + 
                    self.append + '.nc').dropna(dim='day')
        m = xr.open_dataset(self.data_path + 'BgGliderSamples' + 
           '/SOCHIC_PATCH_3h_20121209_20130331_bg_day_week_std_mean_timeseries' 
              + self.append + '.nc').dropna(dim='day')
        
#        # change in bg - model differences over time
        m0 = m.sel(day=t0, method='nearest') # x,y
        m1 = m.sel(day=t1, method='nearest') # x,y
#        m_diff = m0 - m1
        qs = [0.1,0.2,0.5,0.8,0.9] # quantiles
#        deltaM_mean_stats = m_diff.bg_norm_ts_week_mean.quantile(qs,('x','y'))
#        deltaM_std_stats  = m_diff.bg_norm_ts_week_std.quantile(qs,('x','y'))

        # calculate spatial average then difference
        m0_stats = m0.quantile(qs,('x','y'))
        m1_stats = m1.quantile(qs,('x','y'))
        deltaM_stats = m0_stats - m1_stats
        deltaM_mean_stats = deltaM_stats.bg_norm_ts_week_mean
        deltaM_std_stats  = deltaM_stats.bg_norm_ts_week_std

#        ### --- add median delta(bg) to time series --- ###
#        # find t0 and t1 of median delta(bg) 
#        diff_mean = m_diff.bg_norm_ts_week_mean
#        diff_mean = diff_mean.assign_coords(
#                           {'x':np.arange(diff_mean.sizes['x']),
#                            'y':np.arange(diff_mean.sizes['y'])})
#
#        # select postion of median delta(bg) for week_mean data
#        ind0 = (np.abs(diff_mean - deltaM_mean_stats.sel(quantile=0.5))
#               ).argmin(dim='x')
#        diff_mean_1d = diff_mean.isel(x=ind0)
#        ind1 = (np.abs(diff_mean_1d - deltaM_mean_stats.sel(quantile=0.5))
#               ).argmin(dim='y')
#        dm = diff_mean_1d.isel(y=ind1) # median difference (x,y) in week-mean
#        m0_median = m0.isel(x=dm.x, y=dm.y).bg_norm_ts_week_mean
#        m1_median = m1.isel(x=dm.x, y=dm.y).bg_norm_ts_week_mean

        # alternative to above
        m0_median = m0_stats.sel(quantile=0.5).bg_norm_ts_week_mean
        m1_median = m1_stats.sel(quantile=0.5).bg_norm_ts_week_mean

        # select postion of median delta(bg) for full model time series
        m_ts = xr.open_dataset(self.data_path + 
                    'SOCHIC_PATCH_3h_20121209_20130331_bg_norm_timeseries_z10' +
                    self.append + '.nc').bg_norm
        m_ts = m_ts.sel(time_counter=slice('2012-12-15','2013-03-15 00:00:00'))

        m_ts_rolling_mean = m_ts.rolling(time_counter=56, min_periods=1,
                                         center=True).mean()
        qs = [0.1,0.2,0.5,0.8,0.9]
        m_ts = m_ts_rolling_mean.quantile(qs,('x','y'))

        # 
        ##m_ts_median = m_ts.isel(x=dm.x, y=dm.y)
        #m_ts_median = m_ts.isel(x=2, y=2)
        #p = axs0.plot(m_ts_median.time_counter,
        #              m_ts_median,
        #              c=c, lw=0.8, ls='-', label='full model')
        #m_ts = m_ts.quantile([0.1,0.2,0.5,0.8,0.9], ['x','y'])
        axs0.plot(m_ts.time_counter, m_ts.sel(quantile=0.1),
                      c=c, lw=0.8, ls='-', label='__nolegend__')
        p = axs0.plot(m_ts.time_counter,
                      m_ts.sel(quantile=0.5),
                      c=c, lw=0.8, ls='-', label='full model')
        axs0.plot(m_ts.time_counter, m_ts.sel(quantile=0.9),
                      c=c, lw=0.8, ls='-', label='__nolegend__')


        # plot t0 and t1 weeks
        c_mm = '#17dd95' # model mean colour
        c_mm = '#00ffa2' # model mean colour
        c_mm = 'orange'
        c_mm = 'lightseagreen'
        c_mm = '#f18b00'
        axs0.hlines(m0_median, m0.day.values - np.timedelta64(84, 'h'), 
                   m0.day.values + np.timedelta64(84, 'h'),
                   transform=axs0.transData, colors=c_mm, zorder=10)
        axs0.hlines(m1_median, m1.day.values - np.timedelta64(84, 'h'), 
                    m1.day.values + np.timedelta64(84, 'h'),
                    transform=axs0.transData, colors=c_mm, zorder=10)
        axs0.hlines(m0_median, m0.day.values, m1.day.values,
                    transform=axs0.transData, colors=c_mm, lw=0.5,
                    zorder=0)

        from matplotlib.patches import FancyArrowPatch

        x = m1.day
        y0 = m1_median
        y1 = m0_median
        myArrow = FancyArrowPatch(posA=(x, y0), posB=(x, y1),
                                  arrowstyle='<|-|>', color=c_mm,
                                  mutation_scale=5, shrinkA=0, shrinkB=0)
        axs0.add_artist(myArrow)

        # add bounding box for labels
        x0 = m0.day
        y0 = m0_median
        w = np.timedelta64(96, 'h')
        h = 1e-8
        rect = patches.Rectangle((x0, y0), w, h,
                                 linewidth=1, edgecolor=None,
                                 facecolor='w', alpha=1.0)
        # Add the patch to the Axes
        #axs0.add_patch(rect)

        # median delta(bg) labels
        #axs0.text(m0_median.day, m0_median + 1e-9,
        #         r'$\overline{\nabla \mathbf{b}|_{t_0}}$',
        #          ha='left',va='bottom', fontsize=6, c=c_mm)
        #axs0.text(m1_median.day, m1_median + 1e-9,
        #         r'$\overline{\nabla \mathbf{b}|_{t_1}}$',
        #         ha='left',va='bottom', fontsize=6, c=c_mm)
        #axs0.text(m1_median.day + np.timedelta64(7, 'D'),
        axs0.text(x - np.timedelta64(5,'h'),
                  m0_median,
        r'Modelled Change in Median $|\nabla b|$',
                 ha='right',va='bottom', fontsize=6, c=c_mm)


        # legend
        ensemble_text = ['1 Glider','4 Gliders','30 Gliders']
        legend1 =  axs0.legend(fill + p, ensemble_text + ['Model']) 
                       #loc='lower center', bbox_to_anchor=(0.5, 0.94), 
                       #ncol=4, fontsize=8)
 
        # axis limits
        axs0.set_xlim(e.time_counter.min(skipna=True).values,
                      e.time_counter.max(skipna=True).values)
        axs0.set_ylim(0,1.3e-7)

        # add labels
        axs0.set_ylabel(r'$|\nabla b|$' + '\n' + 
                        r' ($\times 10^{-7}$ s$^{-2}$)')
        axs0.yaxis.get_offset_text().set_visible(False)


        # change in bg - glider differences
        g0 = g.sel(day=t0, method='nearest')  
        g1 = g.sel(day=t1, method='nearest')
        g_diff = g0 - g1
        qs = [0.01,0.05,0.1,0.9,0.95,0.99] # quantiles
        deltaG_mean_stats = g_diff.b_x_ml_week_mean.quantile(qs,'sets')
        deltaG_std_stats  = g_diff.b_x_ml_week_std.quantile(qs,'sets')


        # plot
        def render(ax, g, m):
            cg_0='grey'
            cg_1='black'
            cg_2='lightgrey'
            p0 = ax.fill_between(g.ensemble_size + 1, g.sel(quantile=0.01),
                            g.sel(quantile=0.99), facecolor=cg_0,
                            edgecolor=None)
            p1 = ax.fill_between(g.ensemble_size + 1, g.sel(quantile=0.05),
                            g.sel(quantile=0.95), facecolor=cg_1,
                            edgecolor=None)
            p2 = ax.fill_between(g.ensemble_size + 1, g.sel(quantile=0.10),
                            g.sel(quantile=0.90), facecolor=cg_2,
                            edgecolor=None)
            p3 = ax.axhline(m.sel(quantile=0.1), c=c_mm, lw=1.0)
            ax.axhline(m.sel(quantile=0.5), c=c_mm, lw=1.0)
            ax.axhline(m.sel(quantile=0.9), c=c_mm, lw=1.0)
            ax.axhline(0, c='lightgrey', lw=0.5, ls='--')

            # add model labels
            ax.text(29.5, m.sel(quantile=0.1)-3e-9, 'Lower Decile',
                    ha='right',va='top', fontsize=6, c=c_mm)
            ax.text(1.5, m.sel(quantile=0.5)-3e-9, 'Median',
                    ha='left',va='top', fontsize=6, c=c_mm)
            ax.text(29.5, m.sel(quantile=0.9)-3e-9, 'Upper Decile',
                    ha='right',va='top', fontsize=6, c=c_mm)
            return [p0,p1,p2,p3]

        render(axs1[0], deltaG_mean_stats, deltaM_mean_stats)
        leg = render(axs1[1], deltaG_std_stats, deltaM_std_stats)

        # add legend
        axs1[1].legend(leg, ['Gliders 98% CI', 'Gliders 95% CI',
                             'Gliders 80% CI', 'Model'],
                   loc='upper left', bbox_to_anchor=(1.00,1.01), fontsize=8)


        # labels 
        axs1[0].set_ylabel(r'Detected Change in $|\nabla b|$' + '\n' +
                           r' ($\times 10^{-8}$ s$^{-2}$)')
        axs1[1].set_yticklabels([])

        # set ax1 titles
        axs1[0].text(0.5, 1.01, 'Temporal Mean',
                  transform=axs1[0].transAxes, ha='center', va='bottom')
        axs1[1].text(0.5, 1.01, 'Temporal Standard Deviation',
                  transform=axs1[1].transAxes, ha='center', va='bottom')

        axs1[0].set_ylim(-2e-8,9.0e-8)
        axs1[1].set_ylim(-2e-8,9.0e-8)
        for ax in axs1:
            ax.set_xlim(1,30)
            ax.set_xlabel('Number of Gliders')
            ax.yaxis.get_offset_text().set_visible(False)
            ax.set_xticks([1,5,10,15,20,25,30])

        # date labels
        axs0.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        axs0.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        axs0.set_xlabel('Date')

        # letters
        axs0.text(0.01, 0.98, '(a)',
                     transform=axs0.transAxes, ha='left', va='top')
        axs1[0].text(0.02, 1.01, '(b)',
                     transform=axs1[0].transAxes, ha='left', va='bottom')
        axs1[1].text(0.02, 1.01, '(c)',
                     transform=axs1[1].transAxes, ha='left', va='bottom')

        print ('')
        print ('')
        print (t0)
        print ('')
        print ('')
        plt.savefig(self.case + '_bg_change_err_estimate' + self.append +
                   '_'+ t0 + '_' + t1, dpi=600)

    def collect_hists(self, var):
        ''' group histogram files into one '''

        var_str = var + '_glider_'

        preamble = self.data_path + '/SOCHIC_PATCH_3h_20121209_20130331_'

        ds_all, ds_rolling = [], []
        for n in range(1,31):
            # entier glider time series
            g_all = xr.open_dataset(preamble + var_str + str(n).zfill(2) + 
                                   '_hist_interp_1000.nc')
            roll_coord = xr.DataArray(['full_time'], dims='rolling')
            quant_coord = xr.DataArray([n], dims='glider_quantity')
            g_all = g_all.assign_coords({'rolling':         roll_coord,
                                         'glider_quantity': quant_coord})
            ds_all.append(g_all)

            rolls = ['1W_rolling','2W_rolling','3W_rolling']
            ds = []
            for i, roll_freq in enumerate(rolls):
            # weekly glider time series
                try:
                    g = xr.open_dataset(preamble + var_str + str(n).zfill(2) +
                                    '_hist_interp_1000_' + roll_freq + '.nc')
                except:
                    g = xr.open_dataset(preamble + var_str + str(n).zfill(2) +
                                    '_hist_' + roll_freq + '.nc')
                roll_coord = xr.DataArray([roll_freq], dims='rolling')
                g = g.assign_coords({'rolling'        : roll_coord,
                                     'glider_quantity': quant_coord})
                ds.append(g)
                
            ds_rolling.append(xr.concat(ds, dim='rolling'))
        ds_rolling = xr.concat(ds_rolling, dim='glider_quantity')
        ds_all = xr.concat(ds_all, dim='glider_quantity')

        # save
        ds_rolling.to_netcdf(preamble + 'hist_interp_1000_rolling_' +
                             var + '.nc')
        ds_all.to_netcdf(preamble + 'hist_interp_1000_full_time_' +
                             var + '.nc')

class bootstrap_plotting(object):
    def __init__(self, append='', bg_method='norm', interp='1000'):
        self.data_path = config.data_path()
        if append == '':
            self.append = ''
        else:
            self.append='_' + append

        self.interp = '_interp_' + interp

        self.hist_range = (0,2e-8)
        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
        self.bg_method = bg_method

    def plot_variance(self, cases):

        fig, axs = plt.subplots(3,2, figsize=(6.5,5.5))

        def render(ax, g_ens, c='black', var='mean'):
            pre = 'b_x_ml_' + var
            if var == 'mean':
                x = 'time_counter_mean'
            else:
                x = 'day'
            ax.fill_between(g_ens[x],
                            g_ens[pre + '_set_quant'].sel(quantile=0.05),
                            g_ens[pre + '_set_quant'].sel(quantile=0.95),
                            color=c, alpha=0.2)
            ax.plot(g_ens[x], g_ens[pre + '_set_mean'], c=c)
            ax.set_xlim(g_ens[x].min(skipna=True),
                        g_ens.dropna('day')[x].max(skipna=True))

        for i, case in enumerate(cases): 

            # get data
            path = self.data_path + case
            prepend = '/BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_'
            g = xr.open_dataset(path + prepend +  'glider_timeseries' +
                                self.append + '.nc')
            m_std = xr.open_dataset(path + prepend + 'day_week_std_timeseries' +
                                 self.append + '.nc')
            m_mean = xr.open_dataset(path + prepend + 'stats_timeseries' + 
                                 self.append + '.nc')

            # convert to datetime
            g['time_counter_mean'] = g.time_counter_mean.astype(
                                                               'datetime64[ns]')
            #g = g.dropna('day')
            #m_std = m_std.dropna('day')
            #g['b_x_ml_week_std_set_quant'] = g.b_x_ml_week_std_set_quant.dropna('day')

            # model - sdt
            axs[i,1].plot(m_std.day, m_std.bx_ts_day_std, c='cyan')
            axs[i,1].plot(m_std.day, m_std.by_ts_day_std, c='cyan', ls=':')

            # model - mean
            axs[i,0].plot(m_mean.time_counter, m_mean.bx_ts_mean, c='cyan')
            axs[i,0].plot(m_mean.time_counter, m_mean.by_ts_mean, c='cyan',
                                                                    ls=':')
            
            # 1 glider
            g1 = g.isel(ensemble_size=0)
            render(axs[i,1], g1, c='green', var='day_std')
            render(axs[i,0], g1, c='green', var='mean')

            # 4 glider
            g1 = g.isel(ensemble_size=3)
            render(axs[i,1], g1, c='red', var='day_std')
            render(axs[i,0], g1, c='red', var='mean')

            # 30 glider
            g1 = g.isel(ensemble_size=29)
            render(axs[i,1], g1, c='navy', var='day_std')
            render(axs[i,0], g1, c='navy', var='mean')

        for ax in axs[:,0]:
            ax.set_ylim(0,2e-7)
        for ax in axs[:,1]:
            ax.set_ylim(0,1e-7)

        plt.show()

    def render_glider_sample_set_v(self, n=1, c='green', style='plot',
                                   by_time=None):
        # weekly glider time series
        ds = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_' +
                           str(n).zfill(2) + '_hist' + self.append 
                          + '_' + by_time + self.interp + '.nc')

        # entier glider time series
        ds_all = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_' +
                          str(n).zfill(2) + '_hist' + self.append + '.nc')

        # regular weeks
        date_list = np.array([(np.datetime64('2012-12-13')
                              + np.timedelta64(i, 'W')).astype('datetime64[D]')
                               for i in range(16)])
        if style=='bar':
            for (l, week) in ds.groupby('time_counter'):
                i = int(np.argwhere(date_list==l.astype('datetime64[D]')))
                self.axs.flatten()[i].barh(week.bin_left, 
                                 week.hist_u_dec - week.hist_l_dec, 
                                 height=week.bin_right - week.bin_left,
                                 color=c,
                                 alpha=1.0,
                                 left=week.hist_l_dec, 
                                 align='edge',
                                 label='gliders: ' + str(n))
                self.axs.flatten()[i].text(0.3, 0.9, 
                                  week.time_counter.dt.strftime('%m-%d').values,
                                  transform=self.axs.flatten()[i].transAxes,
                                  fontsize=6)
                self.axs.flatten()[i].text(0.3, 0.8, 
                                  str(week.sample_size.values),
                                  transform=self.axs.flatten()[i].transAxes,
                                  fontsize=6)

            self.axs[1,-1].barh(ds_all.bin_left, 
                             ds_all.hist_u_dec - ds_all.hist_l_dec, 
                             height=ds_all.bin_right - ds_all.bin_left,
                             color=c,
                             alpha=1.0,
                             left=ds_all.hist_l_dec, 
                             align='edge',
                             label='gliders: ' + str(n))
            self.axs[1,-1].text(0.3, 0.9, 'all',
                              transform=self.axs[1,-1].transAxes,
                              fontsize=6)
            #    self.ax.scatter(ds.bin_centers, ds.hist_mean, c=c, s=4, zorder=10)
            if style=='plot':
                self.ax.fill_between(ds.bin_centers, ds.hist_l_dec,
                                                     ds.hist_u_dec,
                                     color=c, edgecolor=None, alpha=0.2)
                self.ax.plot(ds.bin_centers, ds.hist_mean, c=c, lw=0.8,
                             label='gliders: ' + str(n))

    def add_model_means_v(self, style='plot', by_time=None):
        ''' add model means of the normed buoyancy gradients of the model '''
        ds = xr.open_dataset(self.path + 
                           '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                           self.append + '_' + by_time + '.nc')
        ds_all = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                        self.append + '.nc')
        # mean direction
        if self.bg_method == 'mean':
            ds['hist'] = (ds.hist_x + ds.hist_y) / 2
            ds_all['hist'] = (ds_all.hist_x + ds_all.hist_y) / 2

        # vector norm
        if self.bg_method == 'norm':
            ds['hist'] = ds.hist_norm
            ds_all['hist'] = ds_all.hist_norm

        date_list = np.array([(np.datetime64('2012-12-13')
                              + np.timedelta64(i, 'W')).astype('datetime64[D]')
                               for i in range(16)])
        if style=='bar':
            for (l, week) in ds.groupby('time_counter'):
                i = int(np.argwhere(date_list==l.astype('datetime64[D]')))
                self.axs.flatten()[i].vlines(week.hist,
                        week.bin_left, week.bin_right,
                       transform=self.axs.flatten()[i].transData,
                       colors='k', lw=0.8, label='model bgx')
            self.axs[1,-1].vlines(ds_all.hist,
                       ds_all.bin_left, ds_all.bin_right,
                       transform=self.axs[1,-1].transData,
                       colors='k', lw=0.8, label='model bgx')
        if style=='plot':
            self.ax.plot(ds.bin_centers, ds.hist, c='black', lw=0.8,
                         label='model bg')

    def add_model_means_averaged_over_weeks(self, axs, c='k'):
        ''' 
        add model means of normed buoyancy gradients of model averaged over
        time slots
        '''
        ds_all = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                        self.append + '.nc')
        # mean direction
        if self.bg_method == 'mean':
            ds_all['hist'] = (ds_all.hist_x + ds_all.hist_y) / 2

        # vector norm
        if self.bg_method == 'norm':
            ds_all['hist'] = ds_all.hist_norm

        rolls = ['1W_rolling','2W_rolling','3W_rolling']
        for i, roll_freq in enumerate(rolls):
            ds = xr.open_dataset(self.path + 
                           '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                           self.append + '_' + roll_freq + '.nc')
            ds = ds.mean('time_counter')

            # mean direction
            if self.bg_method == 'mean':
                ds['hist'] = (ds.hist_x + ds.hist_y) / 2
      
            # vector norm
            if self.bg_method == 'norm':
                ds['hist'] = ds.hist_norm

            axs.flatten()[i].vlines(ds.hist,
                   ds.bin_left, ds.bin_right,
                   transform=axs.flatten()[i].transData,
                   colors=c, lw=0.8, label='model')
        l = axs[-1].vlines(ds_all.hist,
                   ds_all.bin_left, ds_all.bin_right,
                   transform=axs[-1].transData,
                       colors=c, lw=0.8, label='model')
        return l

    def render_glider_sample_set_averaged_over_weeks(self, axs, sample_sizes,
                                                     c='green'):

        l = []
        for i, n in enumerate(sample_sizes):
            print ('sample', i)
            ds_rolling = self.ds_rolling.isel(glider_quantity=n)
            ds_rolling = ds_rolling.mean('time_counter')
            for j, (_, ds) in enumerate(ds_rolling.groupby('rolling')):
                # weekly glider time series
                axs.flatten()[j].barh(ds.bin_left, 
                                 ds.hist_u_dec - ds.hist_l_dec, 
                                 height=ds.bin_right - ds.bin_left,
                                 color=c[i],
                                 alpha=1.0,
                                 left=ds.hist_l_dec, 
                                 align='edge',
                                 label='gliders: ' + str(n))

            ds_all = self.ds_all.isel(glider_quantity=n)
            l.append(axs[-1].barh(ds_all.bin_left, 
                             ds_all.hist_u_dec - ds_all.hist_l_dec, 
                             height=ds_all.bin_right - ds_all.bin_left,
                             color=c[i],
                             alpha=1.0,
                             left=ds_all.hist_l_dec, 
                             align='edge',
                             label='gliders: ' + str(n)))
        return l

    def add_giddy(self, by_time=None):
        ''' add giddy buoyancy gradient distribution '''

        # get glider data
        root = config.root()
        giddy = xr.open_dataset(root + 'Giddy_2020/sg643_linterp.nc')
        giddy_10 = giddy.sel(depth=10, method='nearest')
        giddy_10 = giddy_10.set_coords('time')

        # calculate buoyancy gradients
        g = 9.81     # gravity 
        rho_0 = 1026 # reference density
        b = g*(1-giddy_10.dens/rho_0)
        dx = 1000
        dbdx = b.diff('distance') / dx

        def get_hist(bx):
            hist, bins = np.histogram(bx.dropna('time', how='all'),
                                  range=self.hist_range, density=True, bins=20)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # assign to dataset
            hist_ds = xr.Dataset({'hist':(['bin_centers'], hist)},
                       coords={'bin_centers': (['bin_centers'], bin_centers),
                               'bin_left'   : (['bin_centers'], bins[:-1]),
                               'bin_right'  : (['bin_centers'], bins[1:])})
            return hist_ds
       
        # time splitting - hists are loaded for other variables
        #                - maybe move this to calcs section and save as file...
        dbdx = dbdx.swap_dims({'distance':'time'})

        # base list of weeks for rolling
        date_list = [np.datetime64('2018-12-10 00:00:00') +
                     np.timedelta64(i, 'W')
                     for i in range(16)]
        # mid week dates
        mid_date = [date_list[i] + (date_list[i+1] - date_list[i])/2
                   for i in range(15)]

        if by_time == 'weekly':
            # split into groups of weeks
            hist_ds = dbdx.resample(time='1W', skipna=True).map(
                                                         get_hist)
        elif by_time == '1W_rolling':
            # split into 1 week samples, sampled by week
            hist_ds = dbdx.groupby_bins('time', date_list,
                                    labels=mid_date).map(get_hist)
            hist_ds = hist_ds.rename({'time_bins':'time'})
        elif by_time == '2W_rolling':
            # split into 2 week samples, sampled by week
            mid_date=mid_date[1:]
            l_dl = date_list[::2] + np.timedelta64(84, 'h')
            l_label = mid_date[::2]
            hist_ds_l = dbdx.groupby_bins('time', l_dl,
                         labels=l_label).map(get_hist)
            u_dl = date_list[1:-1:2] + np.timedelta64(84, 'h')
            u_label = mid_date[1:-1:2]# + np.timedelta64(1, 'W')
            hist_ds_u = dbdx.groupby_bins('time', u_dl,
                         labels=u_label).map(get_hist)
            hist_ds = xr.merge([hist_ds_u, hist_ds_l])
            hist_ds = hist_ds.rename({'time_bins':'time'})
        elif by_time == '3W_rolling':
            # split into 3 week samples, sampled by week
            mid_date=mid_date[1:]
            l_dl = date_list[::3]
            l_label = mid_date[::3]
            hist_ds_l = dbdx.groupby_bins('time', l_dl,
                         labels=l_label).map(get_hist)
            m_dl = date_list[1:-1:3]
            m_label = mid_date[1:-1:3]
            hist_ds_m = dbdx.groupby_bins('time', m_dl,
                         labels=m_label).map(get_hist)
            u_dl = date_list[2:-1:3]
            u_label = mid_date[2:-1:3]
            hist_ds_u = dbdx.groupby_bins('time', u_dl,
                         labels=u_label).map(get_hist)
            hist_ds = xr.merge([hist_ds_u, hist_ds_m, hist_ds_l])
            hist_ds = hist_ds.rename({'time_bins':'time'})

        # plot over rolling intervals
        for i, (_,t) in enumerate(hist_ds.groupby('time')):
            if i == 15: continue
            ax = self.axs.flatten()[i]
            ax.vlines(t.hist, t.bin_left, t.bin_right,
                      transform=ax.transData, colors='orange', lw=0.8,
                      label='Giddy et al. (2020)')

        # entire timeseries
        hist_ds = get_hist(dbdx)
        ax = self.axs[-1,-1]
        ax.vlines(hist_ds.hist, hist_ds.bin_left, hist_ds.bin_right,
                  transform=ax.transData, colors='orange', lw=0.8,
                  label='Giddy et al. (2020)')

    def plot_histogram_buoyancy_gradients_and_samples_over_time(self, case,
                                       by_time):
        ''' 
        plot histogram of buoyancy gradients in week portions
        '''

        self.path = self.data_path + case
        self.figure, self.axs = plt.subplots(2,8, figsize=(6.5,3.5))
        plt.subplots_adjust(wspace=0.3, bottom=0.15, left=0.08, right=0.98,
                            top=0.95)
        #self.add_giddy(self.axs[0,0])
        self.add_giddy(by_time=by_time)

        sample_sizes = [1, 4, 20]
        #colours = ['g', 'b', 'r', 'y', 'c']
        colours = ['#dad1d1', '#7e9aa5', '#55475a']

        for i, n in enumerate(sample_sizes):
            print ('sample', i)
            self.render_glider_sample_set_v(n=n, c=colours[i], style='bar',
                                            by_time=by_time)
        self.add_model_means_v(style='bar', by_time=by_time)
        self.add_giddy(by_time=by_time)

        for ax in self.axs[:,0]:
            ax.set_ylabel('Buoyancy Gradient')
        for ax in self.axs[1]:
            ax.set_xlabel('PDF', labelpad=12)

        #plt.legend()
        for ax in self.axs.flatten():
            ax.set_ylim(self.hist_range[0], self.hist_range[1])
            ax.set_xlim(0, 3e8)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
        for ax in self.axs[:,1:].flatten():
            ax.set_yticklabels([])
        for ax in self.axs[0]:
            ax.set_xticklabels([])
        
        if self.bg_method == 'norm':
            norm_str = '_pre_norm'
        else:
            norm_str = '_vector_mean'

        interp_str = self.interp

        plt.savefig(case + '_bg_sampling_skill' + self.append + '_'
                    + by_time + norm_str + interp_str + '.png', dpi=600)

    def plot_histogram_bg_pdf_averaged_weekly_samples(self, case, var='b_x_ml'):
        '''
        Plot buoyancy gradients over rolling frequency averaged over
        all rolling objects. Plots different ensemble sizes and different
        sampling lengths
        '''

        # data paths
        file_id = '/SOCHIC_PATCH_3h_20121209_20130331_' 
        self.path = self.data_path + case 
        self.preamble = self.path + file_id

        self.figure, self.axs = plt.subplots(1,4, figsize=(6.5,3.0))
        plt.subplots_adjust(wspace=0.3, bottom=0.14, left=0.1, right=0.91,
                            top=0.95)
        #self.add_giddy(self.axs[0,0])
        #self.add_giddy(by_time=by_time)

        sample_sizes = [1, 4, 20]
        #colours = ['g', 'b', 'r', 'y', 'c']
        colours = ['#dad1d1', '#7e9aa5', '#55475a']

        # entier glider time series
        # load glider data 
        self.ds_all = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_' + var + '.nc')
        self.ds_rolling = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_' + var + '.nc')

        # render
        self.render_glider_sample_set_averaged_over_weeks(self.axs,
                                                          sample_sizes, 
                                                          c=colours)
        self.add_model_means_averaged_over_weeks(self.axs, c='orange')
        #self.add_giddy(by_time=by_time)

        self.axs[0].set_ylabel(
                  r'$|\nabla b|$ [$\times 10^{-8}$ s$^{-1}$]')
        for ax in self.axs:
            ax.set_xlabel(r'PDF [$\times 10 ^{-8}$]', labelpad=12)

        # legend
        self.axs[-1].legend(loc='upper left', bbox_to_anchor=(0.80,1.0),
                            fontsize=6, borderaxespad=0)

        for ax in self.axs:
            ax.set_ylim(self.hist_range[0], self.hist_range[1])
            ax.set_xlim(0, 3e8)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            ax.xaxis.get_offset_text().set_visible(False)
        self.axs[0].yaxis.get_offset_text().set_visible(False)

        # align labels
        ypos = -0.1  # axes coords
        for ax in self.axs:
            ax.xaxis.set_label_coords(0.5, ypos)

        for ax in self.axs[1:]:
            ax.set_yticklabels([])

        # add time span info
        time_txt = ['1 week', '2 week', '3 week', '3.5 month']
        for i, txt in enumerate(time_txt):
            self.axs[i].text(0.3, 0.95, txt,
                          transform=self.axs[i].transAxes,
                          fontsize=6)

        if self.bg_method == 'norm':
            norm_str = '_pre_norm'
        else:
            norm_str = '_vector_mean'

        interp_str = self.interp

        plt.savefig(case + '_bg_sampling_skill_time_mean_' +
                    var + self.append + '_'
                    + norm_str + interp_str + '.png', dpi=600)

    def plot_histogram_bg_pdf_averaged_weekly_samples_multi_var(self, case):
        '''
        Plot buoyancy gradients over rolling frequency averaged over
        all rolling objects. Plots different ensemble sizes and different
        sampling lengths

        Use multiple variables
        '''

        # data paths
        file_id = '/SOCHIC_PATCH_3h_20121209_20130331_' 
        self.path = self.data_path + case 
        self.preamble = self.path + file_id

        self.figure, self.axs = plt.subplots(2,4, figsize=(5.5,4.5))
        plt.subplots_adjust(wspace=0.1, bottom=0.08, left=0.13, right=0.98,
                            top=0.94, hspace=0.05)
        #self.add_giddy(self.axs[0,0])
        #self.add_giddy(by_time=by_time)

        sample_sizes = [1, 4, 20]
        #colours = ['g', 'b', 'r', 'y', 'c']
        colours = ['#dad1d1', '#7e9aa5', '#55475a']

        def render(row, var):
            # load glider data 
            self.ds_all = xr.open_dataset(self.preamble + 'hist' 
                     + self.interp + self.append + '_full_time_' + var + '.nc')
            self.ds_rolling = xr.open_dataset(self.preamble + 'hist' 
                     + self.interp + self.append + '_rolling_' + var + '.nc')

            # render
            lines = self.render_glider_sample_set_averaged_over_weeks(row,
                                                              sample_sizes, 
                                                              c=colours)
            l = self.add_model_means_averaged_over_weeks(row, c='orange')

            return [l] + lines
            #self.add_giddy(by_time=by_time)

        # render variables
        p = render(self.axs[0], var='b_x_ml')
        render(self.axs[1], var='bg_norm_ml')

        l2 = r'$|\nabla b|$ ($\times 10^{-8}$ s$^{-2}$)'
        self.axs[0,0].set_ylabel('Along-Track Sampled\n' + l2)
        self.axs[1,0].set_ylabel('Across-Front Sampled\n' + l2)
        for ax in self.axs[1]:
            ax.set_xlabel(r'PDF ($\times 10 ^{-8}$)', labelpad=12)

        # legend
        #self.axs[0,-1].legend(loc='upper left', bbox_to_anchor=(1.02,1.0),
        #                    fontsize=6, borderaxespad=0)
        self.figure.legend(p, ['Model','1 Glider', '4 Gliders', '20 Gliders'],
                       loc='lower center', bbox_to_anchor=(0.555, 0.94), 
                       ncol=4, fontsize=8)

        for ax in self.axs.flatten():
            ax.set_ylim(self.hist_range[0], self.hist_range[1])
            ax.set_xlim(0, 2e8)
            #ax.spines['right'].set_visible(False)
            #ax.spines['top'].set_visible(False)
            #ax.spines['left'].set_visible(False)
            ax.xaxis.get_offset_text().set_visible(False)
        for row in self.axs:
            row[0].yaxis.get_offset_text().set_visible(False)
        for col in self.axs[0]:
            col.set_xticklabels([])

        # align labels
        ypos = -0.1  # axes coords
        for ax in self.axs[1]:
            ax.xaxis.set_label_coords(0.5, ypos)

        for ax in self.axs[:,1:].flatten():
            ax.set_yticklabels([])

        # add time span info
        time_txt = ['1-Week', '2-Week', '3-Week', '3.5-Month']
        letters = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
        for j, row in enumerate(self.axs):
            for i, txt in enumerate(time_txt):
                row[i].text(0.95, 0.98, 
                          letters[i + (4*j)] + '\n' + txt + '\nDeployment',
                          transform=row[i].transAxes,
                          fontsize=6, ha='right', va='top')
        #self.figure.text(0.5, 0.99, 'Along-Track Sampling', fontsize=8, 
        #                 ha='center', va='top', fontweight='bold')
        #self.figure.text(0.5, 0.50, 'Across-Front Sampling', fontsize=8, 
        #                 ha='center', fontweight='bold')

        if self.bg_method == 'norm':
            norm_str = '_pre_norm'
        else:
            norm_str = '_vector_mean'

        interp_str = self.interp

        plt.savefig(case + '_bg_sampling_skill_time_mean_multi_var_' +
                     self.append + norm_str + interp_str + '.png', dpi=600)

    def print_bg_rmse_averaged_weekly_samples_multi_var(self,case):
        ''' print rmse error for paper table '''

        # data paths
        file_id = '/SOCHIC_PATCH_3h_20121209_20130331_' 
        self.path = self.data_path + case 
        self.preamble = self.path + file_id

        # get data
        b_x_roll = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_b_x_ml.nc')
        bg_norm_roll = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_bg_norm_ml.nc')
        b_x_full = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_b_x_ml.nc')
        bg_norm_full = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_bg_norm_ml.nc')

        # select rmse
        b_x_roll     = b_x_roll.rmse_mean
        bg_norm_roll = bg_norm_roll.rmse_mean
        b_x_full     = b_x_full.rmse_mean
        bg_norm_full = bg_norm_full.rmse_mean

        # restict to 1-5 gliders
        b_x_roll     = b_x_roll.sel(glider_quantity=slice(1,5))
        bg_norm_roll = bg_norm_roll.sel(glider_quantity=slice(1,5))
        b_x_full     = b_x_full.sel(glider_quantity=slice(1,5))
        bg_norm_full = bg_norm_full.sel(glider_quantity=slice(1,5))

        # mean
        b_x_f_mean = b_x_full.mean('bin_centers')
        bg_norm_f_mean = bg_norm_full.mean('bin_centers')
        b_x_r_mean = b_x_roll.mean(['bin_centers','time_counter'])
        bg_norm_r_mean = bg_norm_roll.mean(['bin_centers','time_counter'])
        print (b_x_f_mean.round(0))
        print (bg_norm_f_mean.round(0))
        print (b_x_r_mean.round(0))
        print (bg_norm_r_mean.round(0))
    
    def plot_histogram_bg_rmse_averaged_weekly_samples_multi_var(self, case):
        '''
        Plot percentage error of bg histograms ~~ glider versus model ~~.
   
        3 x 2 plot
        - Top row is line plots of percentage error against bg grad
          for different glider numbers, with diff_bg_norm and diff_b_x
        - Bottom row is mean across bg of the above percentage error
        Across rows changes samplign length from 1 week up to the 
        full 3.5 months.
        '''

        # data paths
        file_id = '/SOCHIC_PATCH_3h_20121209_20130331_' 
        self.path = self.data_path + case 
        self.preamble = self.path + file_id

        ## initialise plot
        #fig, axs = plt.subplots(2,5, figsize=(6.5,4))
        #plt.subplots_adjust()

        # initialised figure
        fig = plt.figure(figsize=(5.5, 6.0))
        gs0 = gridspec.GridSpec(ncols=4, nrows=2)
        gs1 = gridspec.GridSpec(ncols=2, nrows=1)
        gs0.update(top=0.95, bottom=0.57, left=0.12, right=0.98, hspace=0.1,
                   wspace=0.1)
        gs1.update(top=0.39, bottom=0.08, left=0.12, right=0.98, wspace=0.1)

        axs0, axs1 = [], []
        for i in range(8):
            axs0.append(fig.add_subplot(gs0[i]))
        for i in range(2):
            axs1.append(fig.add_subplot(gs1[i]))

        # get data
        b_x_roll = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_b_x_ml.nc')
        bg_norm_roll = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_rolling_bg_norm_ml.nc')
        b_x_full = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_b_x_ml.nc')
        bg_norm_full = xr.open_dataset(self.preamble + 'hist' 
                 + self.interp + self.append + '_full_time_bg_norm_ml.nc')
        

        # plot rmse across bg
        nums = [1,4,20]
        colours = ['#dad1d1', '#7e9aa5', '#55475a']
        p = []
        for i, num in enumerate(nums):
            b_x = b_x_full.sel(glider_quantity=num)
            print ('')
            print ('')
            print ('')
            print ('')
            print (b_x.rmse_mean.max().values)
            l = axs0[3].bar(b_x.bin_left, 
                        b_x.rmse_mean, 
                        color=colours[i],
                        width=b_x.bin_right - b_x.bin_left,
                        align='edge')

            bg_norm = bg_norm_full.sel(glider_quantity=num)
            axs0[7].bar(bg_norm.bin_left, 
                        bg_norm.rmse_mean, 
                        color=colours[i],
                        width=bg_norm.bin_right - bg_norm.bin_left,
                        align='edge')
                        #label='gliders: ' + str(n))
            p.append(l)

            for j, roll in enumerate(['1W_rolling','2W_rolling','3W_rolling']):
                b_x = b_x_roll.sel(glider_quantity=num, rolling=roll)
                b_x = b_x.mean('time_counter')
                print (b_x.rmse_mean.max().values)
                print ('')
                axs0[j].bar(b_x.bin_left, 
                             b_x.rmse_mean, 
                             color=colours[i],
                             width=b_x.bin_right - b_x.bin_left,
                             align='edge')

                bg_norm = bg_norm_roll.sel(glider_quantity=num, rolling=roll)
                bg_norm = bg_norm.mean('time_counter')
                axs0[j+4].bar(bg_norm.bin_left, 
                             bg_norm.rmse_mean, 
                             color=colours[i],
                             width=bg_norm.bin_right - bg_norm.bin_left,
                             align='edge')

        print ('')
        print ('')
        print ('')
        print ('')
        fig.legend(p, ['1 Glider', '4 Gliders', '20 Gliders'],
                       loc='lower center', bbox_to_anchor=(0.555, 0.95), 
                       ncol=4, fontsize=8)

        # ~~~ plot rmse across glider number ~~~ #

        c1 = '#f18b00'
        colours=[c1, 'purple', 'green' 'navy']#,'turquoise']
        colours = ['#dad1d1', '#7e9aa5', '#55475a']
        c0 = '#7e9aa5'
        #c0 = '#dad1d1'
        c= ['lightgrey', 'grey', 'black', c1]
        #c = ['#dad1d1', '#7e9aa5', '#55475a', c1]
        #c=[c1, 'purple', 'green', 'navy']#,'turquoise']
        #cg_0='grey'
        #cg_1='black'
        #cg_2='lightgrey'

        b_x_f_mean = b_x_full.mean('bin_centers')
        bg_norm_f_mean = bg_norm_full.mean('bin_centers')
        b_x_r_mean = b_x_roll.mean(['bin_centers','time_counter'])
        bg_norm_r_mean = bg_norm_roll.mean(['bin_centers','time_counter'])

        # cut small bg magnitudes
        b_x_full_cut = b_x_full.isel(bin_centers=slice(3,None))
        b_x_roll_cut = b_x_roll.isel(bin_centers=slice(3,None))
        b_x_f_mean_cut = b_x_full_cut.mean('bin_centers')
        b_x_r_mean_cut = b_x_roll_cut.mean(['bin_centers','time_counter'])

        bg_norm_full_cut = bg_norm_full.isel(bin_centers=slice(3,None))
        bg_norm_roll_cut = bg_norm_roll.isel(bin_centers=slice(3,None))
        bg_norm_f_mean_cut = bg_norm_full_cut.mean('bin_centers')
        bg_norm_r_mean_cut=bg_norm_roll_cut.mean(['bin_centers','time_counter'])

        p0, = axs1[0].plot(b_x_f_mean.glider_quantity, b_x_f_mean.rmse_mean,
                     c=c[3], lw=1.5, zorder=10)
        p1, = axs1[0].plot(b_x_f_mean_cut.glider_quantity,
                     b_x_f_mean_cut.rmse_mean,
                     c=c[3], ls='--', lw=1.5)
        axs1[1].plot(bg_norm_f_mean.glider_quantity, bg_norm_f_mean.rmse_mean,
                     c=c[3], lw=1.5)

        p = []
        for i, roll in enumerate(['1W_rolling','2W_rolling','3W_rolling']):
            b_x_mean = b_x_r_mean.sel(rolling=roll)
            b_x_mean_cut = b_x_r_mean_cut.sel(rolling=roll)
            bg_norm_mean = bg_norm_r_mean.sel(rolling=roll)
            bg_norm_mean_cut = bg_norm_r_mean_cut.sel(rolling=roll)
            l, = axs1[0].plot(b_x_mean.glider_quantity, b_x_mean.rmse_mean, 
                         c=c[i], lw=1.5)
            axs1[1].plot(bg_norm_mean.glider_quantity, bg_norm_mean.rmse_mean,
                         c=c[i], lw=1.5)
            p.append(l)
            #ax.text(0.9,0.9,roll,transform=ax.transAxes)

        p = p + [p0]# + [p1]
        #labs = ['1-Week', '2-Week', '3-Week', '3.5-Month', '3.5-Month '+ thresh]
        labs = ['1-Week', '2-Week', '3-Week', '3.5-Month', '3.5-Month']
        fig.legend(p, labs, loc='lower center', title='Deployment',
                       title_fontsize=8,
                       bbox_to_anchor=(0.555, 0.39), ncol=4, fontsize=8)

        for ax in axs0:
            ax.set_ylim(0,160)
            ax.xaxis.get_offset_text().set_visible(False)
            ax.set_xlim(self.hist_range[0], self.hist_range[1])
        for ax in axs0[:4]:
            ax.set_xticklabels([])
        for ax in axs0[1:4] + axs0[5:] + [axs1[1]]:
            ax.set_yticklabels([])
        for ax in axs1:
            ax.set_ylim(0,120)

        for ax in axs1:
            ax.set_xlim(1,30)
            ax.set_xlabel('Number of Gliders')
            ax.set_xticks([1,5,10,15,20,25,30])

        l0 = r'$|\nabla b|$'
        l1 = r'($\times 10^{-8}$ s$^{-2}$)'
        axs0[4].set_xlabel(l0 + ' 1-Week\n' + l1)
        axs0[5].set_xlabel(l0 + ' 2-Week\n' + l1)
        axs0[6].set_xlabel(l0 + ' 3-Week\n' + l1)
        axs0[7].set_xlabel(l0 + ' 3.5-Month\n' + l1)

        l2 = ' RMSE (%)'
        axs0[0].set_ylabel('Along-Track\nSampled' + l2)
        axs0[4].set_ylabel('Across-Front\nSampled'+ l2)

        l2 = ' RMSE (%)'
        axs0[0].set_ylabel('Along-Track\nSampled' + l2)
        axs0[4].set_ylabel('Across-Front\nSampled'+ l2)
       
        axs1[0].set_ylabel('Mean RMSE (%)')

        axs1[0].text(0.98, 0.96, 'Along-Track', va='top', ha='right',
                    transform=axs1[0].transAxes)
        axs1[1].text(0.98, 0.96, 'Across-Front', va='top', ha='right',
                    transform=axs1[1].transAxes)

        thresh = r'$|\nabla b|> 4 \times 10^{-9}$ s$^{-2}$'
        axs1[0].text(29.5, 32, thresh, va='bottom', ha='right',
                    transform=axs1[0].transData, c=c[3], fontsize=6)
        # letters
        letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
        for i, ax in enumerate(axs0):
            ax.text(0.95, 0.95, letters[i], transform=ax.transAxes, va='top',
                    ha='right')

        letters = ['(i)', '(j)']
        for i, ax in enumerate(axs1):
            ax.text(0.03, 0.96, letters[i], transform=ax.transAxes, va='top',
                    ha='left')


        plt.savefig('paper_hist_rmse.png', dpi=1200)

    def get_ensembles(self, case, by_time):
        '''
        load weekly model hists and calculate mean buoyancy gradients in x and y
            - for rmse differences
        '''

        # load weekly model hists
        m = xr.open_dataset(self.data_path + case + 
                     '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                     self.append + '_' + by_time + '.nc')#.isel(
                    # time_counter=slice(None,-1))
        
        # pre merge function
        def pre_proc(ds):
            ds = ds.expand_dims('ensemble_size')
            return ds
         
        # load weekly glider hists
        prep = case + '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_'
        ensemble_list = [self.data_path + prep + str(i).zfill(2) + '_hist' + 
                         self.append + '_' + by_time + '.nc'
                         for i in range(1,31)]
        ensembles = xr.open_mfdataset(ensemble_list, 
                                   combine='nested', concat_dim='ensemble_size',
                                     preprocess=pre_proc).load()
        ensembles = ensembles.assign_coords(ensemble_size=np.arange(1,31))
        ensembles = ensembles.set_coords('sample_size')

        # mean of the histograms
        if self.bg_method == 'mean':
            ensembles['m_bg_abs'] = (m.hist_x + m.hist_y) / 2

        # norm of the vectors - calculated before hist
        elif self.bg_method == 'norm':
            ensembles['m_bg_abs'] = m.hist_norm

        return ensembles

    def rmsep(self, pred, truth):
        ''' calulate root mean squared percentage error '''

        norm = (pred - truth)/truth 

        return np.sqrt(((norm)**2).mean(dim='bin_centers')) * 100

    def get_spatial_mean_and_std_bg(self, case):
        '''
        calcualte spatial mean and standard deviation of buoyancy gradients
        at 10 m depth
        '''

        # load bg
        bg = xr.open_dataset(self.data_path + case +
                             self.file_id + 'bg_z10m.nc', chunks='auto')
        # cut edges of domain
        bg = bg.isel(x=slice(20,-20), y=slice(20,-20))
  
        # absolute value of vector mean
        if self.bg_method == 'mean':
            bg = np.abs((bg.bx + bg.by)/2).load()

        # bg normed
        elif self.bg_method == 'norm':
            bg = ((bg.bx**2 + bg.by**2) ** 0.5).load()

        else:
            print ('method not recognised')

        bg_mean = bg.mean(['x','y'])
        bg_std = bg.std(['x','y'])
 
        return bg_mean, bg_std

    def plot_rmse_over_ensemble_sizes_and_week(self, case, by_time):
        ''' plot the root mean squared error of the 1 s.d. (? not decile)
            from the **real** mean over week and ensemble size
            contourf
        '''

        ensembles = self.get_ensembles(case, by_time, method='norm')

        # calculate rmse
        rmse_l = self.rmsep(ensembles.hist_l_dec, ensembles.m_bg_abs)
        rmse_u = self.rmsep(ensembles.hist_u_dec, ensembles.m_bg_abs)
        rmse_mean = self.rmsep(ensembles.hist_mean, ensembles.m_bg_abs)

        # initialised figure
        fig = plt.figure(figsize=(6.5, 4), dpi=300)
        gs0 = gridspec.GridSpec(ncols=1, nrows=2)
        gs1 = gridspec.GridSpec(ncols=1, nrows=1)
        gs0.update(top=0.98, bottom=0.35, left=0.13, right=0.87, hspace=0.1)
        gs1.update(top=0.30, bottom=0.15, left=0.13, right=0.87)

        axs0 = []
        for i in range(2):
            axs0.append(fig.add_subplot(gs0[i]))
        axs1 = fig.add_subplot(gs1[0])

        # initialise plot
        #fig, axs = plt.subplots(2, figsize=(6.5,4))
        #plt.subplots_adjust(left=0.08, right=0.87, hspace=0.1, bottom=0.15,
        #                    top=0.98)

        # render
        cmap = plt.cm.inferno
        lev = np.linspace(0,300,11)
        p0 = axs0[0].contourf(rmse_u.time_counter, rmse_u.ensemble_size, rmse_u,
                             levels=lev, cmap=cmap)
        lev = np.linspace(0,100,11)
        p1 = axs0[1].contourf(rmse_l.time_counter, rmse_l.ensemble_size, rmse_l,
                             levels=lev, cmap=cmap)

        # colour bar upper
        pos = axs0[0].get_position()
        cbar_ax = fig.add_axes([0.88, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.1, 0.5, 'RMSE of\nbuoyancy gradients (%)', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left', multialignment='center')

        # colour bar lower
        pos = axs0[1].get_position()
        cbar_ax = fig.add_axes([0.88, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.1, 0.5, 'RMSE of\nbuoyancy gradients (%)', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left', multialignment='center')

        # text labels
        axs0[0].text(0.99, 0.98, 'upper decile', c='w', va='top', ha='right',
                    transform=axs0[0].transAxes)
        axs0[1].text(0.99, 0.98, 'lower decile', c='w', va='top', ha='right',
                    transform=axs0[1].transAxes)

        # axes labels
        for ax in axs0:
            ax.set_ylabel('ensemble size')
            ax.set_xticks(rmse_mean.time_counter)

        # set xlabels
        axs0[0].set_xticklabels([])
        axs0[1].set_xticklabels([])


        # add time series of sample size
        sample_size = ensembles.sample_size.isel(ensemble_size=0)
        axs1.plot(sample_size.time_counter, sample_size)

        # add time series of bg - standard deviation
        _, std = self.get_spatial_mean_and_std_bg(case)
        axs1_2 = axs1.twinx()
        axs1_2.plot(std.time_counter, std)

        # set xticks
        week_labels = sample_size.time_counter.dt.strftime('%m-%d').values
        axs1.set_xlim(sample_size.time_counter.min(),
                      sample_size.time_counter.max())
        axs1.set_ylim(0,500)
        axs1.set_xticklabels(week_labels)
        axs1.set_xticks(sample_size.time_counter)
        axs1.set_xlabel('date (MM-DD)')
        axs1.set_ylabel('sample\nsize')

        # rotate labels
        for label in axs1.get_xticklabels(which='major'):
            label.set(rotation=35, horizontalalignment='right')

        plt.savefig(case + '_bg_RMSE_' + by_time 
                    + self.append + '.png', dpi=600)

    def plot_rmse_over_ensemble_sizes_and_week_3_panel(self, case):
        ''' plot the root mean squared error of the 1 s.d. (? not decile)
            from the **real** mean over week and ensemble size
            contourf
        '''

        # initialised figure
        fig = plt.figure(figsize=(6.5, 4), dpi=300)
        gs0 = gridspec.GridSpec(ncols=3, nrows=2)
        gs1 = gridspec.GridSpec(ncols=3, nrows=1)
        gs0.update(top=0.95, bottom=0.35, left=0.08, right=0.86, hspace=0.1,
                   wspace=0.07)
        gs1.update(top=0.32, bottom=0.15, left=0.08, right=0.86,
                   wspace=0.07)

        axs0, axs1 = [], []
        for j in range(3):
            for i in range(2):
                axs0.append(fig.add_subplot(gs0[i,j]))
        for i in range(3):
            axs1.append(fig.add_subplot(gs1[i]))

        # render
        def calc_and_render(by_time, a0, a1):

            # load data
            ensembles = self.get_ensembles(case, by_time, method='norm')

            # calculate rmse
            rmse_l = self.rmsep(ensembles.hist_l_dec, ensembles.m_bg_abs)
            rmse_u = self.rmsep(ensembles.hist_u_dec, ensembles.m_bg_abs)
            rmse_mean = self.rmsep(ensembles.hist_mean, ensembles.m_bg_abs)

            cmap = plt.cm.inferno

            # upper decile
            lev = np.linspace(0,300,11)
            p0 = a0[0].contourf(rmse_u.time_counter, rmse_u.ensemble_size,
                                  rmse_u, levels=lev, cmap=cmap)

            # lower decile
            lev = np.linspace(0,100,11)
            p1 = a0[1].contourf(rmse_l.time_counter, rmse_l.ensemble_size,
                                    rmse_l, levels=lev, cmap=cmap)

            # add time series of sample size
            sample_size = ensembles.sample_size.isel(ensemble_size=0)
            p2, = a1.plot(sample_size.time_counter, sample_size, lw=0.8)
            a1.yaxis.label.set_color(p2.get_color())
            a1.tick_params(axis='y', colors=p2.get_color())

            return p0, p1, ensembles.time_counter

        # render each column
        calc_and_render('1W_rolling', axs0[:2], axs1[0])
        calc_and_render('2W_rolling', axs0[2:4], axs1[1])
        p0, p1, time_counter = calc_and_render('3W_rolling', axs0[4:], axs1[2])

        # add time series of bg - standard deviation
        _, std = self.get_spatial_mean_and_std_bg(case)
        twin_axes, p  = [], []
        for ax in axs1[:2]:
            a1_2 = ax.twinx()
            twin_axes.append(a1_2)
            p.append(a1_2.plot(std.time_counter, std, c='g', lw=0.8)[0])
            a1_2.set_yticklabels([])
        a1_2 = axs1[-1].twinx()
        twin_axes.append(a1_2)
        p.append(a1_2.plot(std.time_counter, std, c='g', lw=0.8)[0])
        a1_2.set_ylabel(r'$\sigma_{bg}$' + '\n' + r'$[\times 10^{-8}]$')
        a1_2.yaxis.get_offset_text().set_visible(False)
        for i, ax in enumerate(twin_axes):
            ax.yaxis.label.set_color(p[i].get_color())
            ax.tick_params(axis='y', colors=p[i].get_color())

        # top two rows
        for ax in axs0:
            ax.set_xticklabels([])
            ax.set_xticks(time_counter)
        for ax in axs0[:2]:
            ax.set_ylabel('ensemble size')
        for ax in axs0[2:]:
            ax.set_yticklabels([])

        # bottom row axes details
        for ax in axs1:
            week_labels = time_counter.dt.strftime('%m-%d').values
            week_labels[::2] = ''
            ax.set_xlim(time_counter.min(), time_counter.max())
            ax.set_ylim(0,500)
            ax.set_xticklabels(week_labels)
            ax.set_xticks(time_counter)
            ax.set_xlabel('date (MM-DD)')

            # rotate labels
            for label in ax.get_xticklabels(which='major'):
                label.set(rotation=35, horizontalalignment='right')

        axs1[0].set_ylabel('sample size')
        for ax in axs1[1:]:
            ax.set_yticklabels([])

        # colour bar upper
        pos = axs0[4].get_position()
        cbar_ax = fig.add_axes([0.87, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(3.7, 0.5, 'RMSE of\nbuoyancy gradients\n[%]', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left', multialignment='center')

        # colour bar lower
        pos = axs0[5].get_position()
        cbar_ax = fig.add_axes([0.87, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(3.7, 0.5, 'RMSE of\nbuoyancy gradients\n[%]', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left', multialignment='center')

        # text labels
        by_time_labels = ['1-Week Rolling', '2-Week Rolling', '3-Week Rolling']
        for i, ax in enumerate(axs0[::2]):
            ax.text(0.99, 0.98, 'upper decile', c='w', va='top', ha='right',
                    transform=ax.transAxes, fontsize=6)
            ax.text(0.5, 1.01, by_time_labels[i], va='bottom', ha='center',
                    transform=ax.transAxes)
        for ax in axs0[1::2]:
            ax.text(0.99, 0.98, 'lower decile', c='w', va='top', ha='right',
                    transform=ax.transAxes, fontsize=6)

        # align labels
        xpos = -0.21  # axes coords
        for ax in axs0[:2]:
            ax.yaxis.set_label_coords(xpos, 0.5)
        axs1[0].yaxis.set_label_coords(xpos, 0.5)

        plt.savefig(case + '_bg_RMSE_3_panel_' + self.append + 'pre_norm.png',
                    dpi=600)


    def plot_correlation_rmse(self, case):
        '''
        scatter plots of RMSE against
            - ensemble size
            - sample size
            - spatial? standard deviation in bg
        '''
 
        # initialise figure
        fig, axs = plt.subplots(3, 1, figsize=(3.2,3.5))
        plt.subplots_adjust(top=0.95, bottom=0.13, left=0.09, right=0.78,
                            hspace=0.15)

        # get standard deviation of bg for correlations
        _, std = self.get_spatial_mean_and_std_bg(case)

        def all_by_times(by_time, std, c='r', pos=1):

            ensembles = self.get_ensembles(case, by_time)
            std = std.interp(time_counter=ensembles.time_counter)

            # calculate rmse
            rmse_l = self.rmsep(ensembles.hist_l_dec, ensembles.m_bg_abs)
            rmse_u = self.rmsep(ensembles.hist_u_dec, ensembles.m_bg_abs)

            # drop 100 % errors
            rmse_l = rmse_l.where(rmse_l != 100)
            rmse_u = rmse_u.where(rmse_u != 100)
            rmse_l['bg_std'] = std.interp(time_counter=rmse_l.time_counter)
            rmse_u['bg_std'] = std.interp(time_counter=rmse_u.time_counter)

            rmse_l = rmse_l.stack(z=['ensemble_size','time_counter'])
            rmse_u = rmse_u.stack(z=['ensemble_size','time_counter'])
            l_ensemble_size_norm=rmse_l.ensemble_size/rmse_l.ensemble_size.max()
            u_ensemble_size_norm=rmse_u.ensemble_size/rmse_u.ensemble_size.max()
            #sample_size_max = 448 # max for all rolling lengths
            #l_sample_size_norm = rmse_l.sample_size/sample_size_max
            #u_sample_size_norm = rmse_u.sample_size/sample_size_max
            l_sample_size_norm = rmse_l.sample_size/rmse_l.sample_size.max()
            u_sample_size_norm = rmse_u.sample_size/rmse_u.sample_size.max()
            l_bg_std_norm = rmse_l.bg_std/rmse_l.bg_std.max()
            u_bg_std_norm = rmse_u.bg_std/rmse_u.bg_std.max()
            normalise = matplotlib.colors.Normalize(vmin=0, vmax=448)

            m='|'
            s=5
            a=0.4
            axs[0].scatter(rmse_u, pos * np.ones(len(rmse_u)) + 0.2, 
                           c=rmse_u.ensemble_size, s=s, alpha=a, marker=m)
            axs[1].scatter(rmse_u, pos * np.ones(len(rmse_u)) + 0.2, 
                           c=rmse_u.sample_size, s=s, alpha=a, marker=m,
                           norm=normalise)
            axs[2].scatter(rmse_u, pos * np.ones(len(rmse_u)) + 0.2, 
                           c=rmse_u.bg_std, s=s, alpha=a, marker=m)
            p0 = axs[0].scatter(rmse_l, pos * np.ones(len(rmse_l)) - 0.2, 
                           c=rmse_l.ensemble_size, s=s, alpha=a, marker=m)
            p1 = axs[1].scatter(rmse_l, pos * np.ones(len(rmse_l)) - 0.2, 
                           c=rmse_l.sample_size, s=s, alpha=a, marker=m,
                           norm=normalise)
            p2 = axs[2].scatter(rmse_l, pos * np.ones(len(rmse_l)) - 0.2, 
                           c=rmse_l.bg_std, s=s, alpha=a, marker=m)

            # colour bar
            pos0 = axs[0].get_position()
            cbar_ax = fig.add_axes([0.86, pos0.y0, 0.02, pos0.y1 - pos0.y0])
            cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
            cbar.ax.text(5.3, 0.5, 'ensemble size',
                         fontsize=6, rotation=90, transform=cbar.ax.transAxes,
                         va='center', ha='left', multialignment='center')
            cbar.ax.tick_params(labelsize=6)
            cbar.ax.yaxis.get_offset_text().set_visible(False)
            pos1 = axs[1].get_position()
            cbar_ax = fig.add_axes([0.86, pos1.y0, 0.02, pos1.y1 - pos1.y0])
            cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
            cbar.ax.text(5.3, 0.5, 'sample size',
                         fontsize=6, rotation=90, transform=cbar.ax.transAxes,
                         va='center', ha='left', multialignment='center')
            cbar.ax.tick_params(labelsize=6)
            cbar.ax.yaxis.get_offset_text().set_visible(False)
            pos2 = axs[2].get_position()
            cbar_ax = fig.add_axes([0.86, pos2.y0, 0.02, pos2.y1 - pos2.y0])
            cbar = fig.colorbar(p2, cax=cbar_ax, orientation='vertical')
            cbar.ax.text(5.3, 0.5, r'$\sigma_{bg} [\times 10^{-8}]$',
                         fontsize=6, rotation=90, transform=cbar.ax.transAxes,
                         va='center', ha='left', multialignment='center')
            cbar.ax.tick_params(labelsize=6)
            cbar.ax.yaxis.get_offset_text().set_visible(False)

            for ax in axs:
                ax.text(0, pos + 0.2, 'u', va='center', ha='center',
                        fontsize=6, transform=ax.transData)
                ax.text(0, pos - 0.2, 'l', va='center', ha='center',
                        fontsize=6, transform=ax.transData)
                ax.text(350, 4, r'$r$',
                        va='center', ha='center',
                        fontsize=6, transform=ax.transData)

            # pearsons rank correlation
            print (rmse_l)
            rmse_l = rmse_l.dropna('z')
            u_ens_pr = round(stats.pearsonr(rmse_u.ensemble_size, rmse_u)[0], 3)
            l_ens_pr = round(stats.pearsonr(rmse_l.ensemble_size, rmse_l)[0], 3)
            u_sam_pr = round(stats.pearsonr(rmse_u.sample_size, rmse_u)[0], 3)
            l_sam_pr = round(stats.pearsonr(rmse_l.sample_size, rmse_l)[0], 3)
            u_std_pr = round(stats.pearsonr(rmse_u.bg_std, rmse_u)[0], 3)
            l_std_pr = round(stats.pearsonr(rmse_l.bg_std, rmse_l)[0], 3)

            axs[0].text(350, pos + 0.2, str(u_ens_pr),
                    va='center', ha='center',
                    fontsize=6, transform=axs[0].transData)
            axs[0].text(350, pos - 0.2, str(l_ens_pr),
                    va='center', ha='center',
                    fontsize=6, transform=axs[0].transData)
            axs[1].text(350, pos + 0.2, str(u_sam_pr),
                    va='center', ha='center',
                    fontsize=6, transform=axs[1].transData)
            axs[1].text(350, pos - 0.2, str(l_sam_pr),
                     va='center', ha='center',
                    fontsize=6, transform=axs[1].transData)
            axs[2].text(350, pos + 0.2, str(u_std_pr),
                     va='center', ha='center',
                     fontsize=6, transform=axs[2].transData)
            axs[2].text(350, pos - 0.2, str(l_std_pr),
                    va='center', ha='center',
                    fontsize=6, transform=axs[2].transData)

        all_by_times('1W_rolling', std, c='r', pos=3)
        all_by_times('2W_rolling', std, c='g', pos=2)
        all_by_times('3W_rolling', std, c='b', pos=1)
 
        # axis details
        ax_labels = ['ensemble size', 'sample size', r'$\sigma_{bg}$']
        for i, ax in enumerate(axs):
            ax.set_xlim(0,335)
            ax.set_ylim(0,4)
            ax.set_yticks([1,2,3])
            ax.set_yticklabels(['3W', '2W', '1W'])
            ax.text(0.5, 1.01, ax_labels[i], va='bottom', ha='center',
                    fontsize=8, transform=ax.transAxes)

            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            #ax.tick_params(tick1On=False)
            #ax.tick_params(axis='y', which='both', length=0)
            ax.tick_params(left=False)

        axs[2].set_xlabel('RMSE of buoyancy gradients [%]')
        for ax in axs[:2]:
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])

        plt.savefig('EXP10_bg_rmse_corr_pre_norm.png', dpi=600)
  
def plot_correlations():
    boot = bootstrap_plotting(bg_method='norm')
    boot.plot_correlation_rmse('EXP10')

#plot_correlations()

def plot_hist(by_time=None):
    cases = ['EXP10', 'EXP08', 'EXP13']
    cases = ['EXP10']
    if by_time:
        boot = bootstrap_plotting(bg_method='norm', interp='1000')
        #boot.plot_histogram_buoyancy_gradients_and_samples_over_time(
        #                                                      'EXP10', by_time)
        #boot.plot_rmse_over_ensemble_sizes_and_week('EXP10', by_time)
        #boot.plot_rmse_over_ensemble_sizes_and_week_3_panel('EXP10')
        #boot.plot_histogram_bg_pdf_averaged_weekly_samples('EXP10',
        #                                                          var='b_x_ml')
    
    else:
        boot = bootstrap_plotting()
        #boot.plot_histogram_bg_pdf_averaged_weekly_samples_multi_var('EXP10')
        boot.print_bg_rmse_averaged_weekly_samples_multi_var('EXP10')
        #boot.plot_histogram_bg_rmse_averaged_weekly_samples_multi_var('EXP10')
            #m = bootstrap_glider_samples(case, var='b_x_ml', load_samples=False,
            #                             subset='')
            #m.plot_histogram_buoyancy_gradients_and_samples()
            #m.plot_rmse_over_ensemble_sizes()

def collect_hists():
    m = bootstrap_glider_samples('EXP10')
    m.collect_hists('b_x_ml')
    m.collect_hists('bg_norm_ml')

#collect_hists()
def prep_hist(by_time=None, interp='1000'):
    '''
    Create files for histogram plotting. Takes output from model_object.

    Select variable from model object files.
    Expectation: - b_x_ml (along-track gradients) 
                 - bg_norm (glider sample of model norm)
    '''

    cases = ['EXP10']
    #for var in ['bg_norm_ml','b_x_ml']:
    for var in ['b_x_ml']:
        for case in cases:
            m = bootstrap_glider_samples(case, var=var, load_samples=True,
                                       subset='', transect=False, interp=interp)
            if by_time:
                 m.append =  m.append + '_' + by_time
            #m.get_full_model_hist(save=True, by_time=by_time)
            #m.get_glider_sampled_hist(n=1, save=True, by_time=by_time)
            for n in range(30,31):
                print (n)
                m.get_glider_sampled_hist(n=n, save=True, by_time=by_time)

def prep_timeseries(subset='', interp='1000'):
    cases = ['EXP10']
    #cases = ['EXP10', 'EXP08', 'EXP13']
    for case in cases:
        m = bootstrap_glider_samples(case, var='b_x_ml', load_samples=True,
                                     subset=subset, interp=interp)
        m.get_full_model_timeseries(save=True)
        for n in range(1,31):
            print ('n :', n)
            m.get_glider_timeseries(n=n, save=True)

def plot_timeseries():
    cases = ['EXP10', 'EXP08', 'EXP13']
    cases = ['EXP10']
    for case in cases:
        print ('case: ', case)
        m = bootstrap_glider_samples(case, var='b_x_ml', load_samples=False,
                                    subset='south')
        m.plot_timeseries()

def plot_quantify_delta_bg(subset=''):
    ''' quantify the skill of estimated change in bg with time from glider '''
    cases = ['EXP10']
    for case in cases:
        print ('case: ', case)
        m = bootstrap_glider_samples(case, var='b_x_ml', load_samples=False,
                                     subset=subset)
        m.plot_quantify_delta_bg(t0 = '2013-01-01', t1 = '2013-03-01')
#plot_quantify_delta_bg(subset='south')
##plot_quantify_delta_bg(subset='north')

# -------- paper plot --------- #

#plot_quantify_delta_bg(subset='south')

# ----------------------------- #

#bootstrap_plotting().plot_variance(['EXP13','EXP08','EXP10'])

#for exp in ['EXP10','EXP13','EXP08']:
#for exp in ['EXP13','EXP08']:
#for exp in ['EXP10']:
#    m = bootstrap_glider_samples(exp, var='b_x_ml', load_samples=True,
#                                  subset='north')
#    m.get_glider_timeseries(ensemble_range=range(1,31), save=True)
#    m.get_full_model_day_week_std(save=True)
#    m = bootstrap_glider_samples(exp, var='b_x_ml', load_samples=True,
#                                  subset='south')
#    m.get_glider_timeseries(ensemble_range=range(1,31), save=True)
#    m.get_full_model_day_week_std(save=True)

#m = bootstrap_glider_samples('EXP10', load_samples=False, subset='south')
#m.get_full_model_day_week_sdt_and_mean_bg(save=True)
#m.get_full_model_timeseries(save=True)
#m.get_full_model_timeseries_norm_bg(save=True)

plot_hist()
#prep_hist(by_time='3W_rolling')
#prep_hist(by_time='1W_rolling', interp='1000')
#prep_hist(by_time='2W_rolling', interp='1000')
#prep_hist(by_time='3W_rolling', interp='1000')
#prep_hist(interp='1000')
#plot_hist()
#prep_timeseries()
#plot_timeseries()
#plot_quantify_delta_bg()
print ('done 1')
#m = bootstrap_glider_samples('EXP08')
#m.histogram_buoyancy_gradients_and_samples()
#print ('done 2')
m = bootstrap_glider_samples('EXP10')
#m.histogram_buoyancy_gradients_and_samples()
m.plot_error_bars()

#def plot_histogram():
#    m = glider_nemo('EXP03')
#    m.load_nemo_bg()
#    #m.load_glider_nemo()
#    #m.sub_sample_nemo()
#    m.histogram_buoyancy_gradient()
#
#plot_histogram()
