import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import dask
import matplotlib
import datetime
#import itertools
from get_transects import get_transects

matplotlib.rcParams.update({'font.size': 8})

class bootstrap_glider_samples(object):
    '''
    for ploting bootstrap samples of buoyancy gradients
    '''

    def __init__(self, case, offset=False, var='b_x_ml', load_samples=True,
                 subset='', transect=False):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'

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

        # load samples
        prep = 'GliderRandomSampling/glider_uniform_interp_1000' + patch + '.nc'
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
        
         
        clean_float_time = self.samples.time_counter
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
            + weekly and daily sdt
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

        bg_d = daily_std.rename({'bx':'bx_ts_day_std', 'by':'by_ts_day_std'})
        bg_w = weekly_std.rename({'bx':'bx_ts_week_std', 'by':'by_ts_week_std'})
        bg_stats = xr.merge([bg_w,bg_d])
        if save:
            bg_stats.to_netcdf(self.data_path + '/BgGliderSamples' +
              '/SOCHIC_PATCH_3h_20121209_20130331_bg_day_week_std_timeseries' + 
                    self.append + '.nc')


    def get_full_model_timeseries(self, save=False):
        ''' 
           get model mean and std time_series
               - buoyancy
        '''

        self.get_model_buoyancy_gradients()
        self.bg = self.bg.sel(deptht=10, method='nearest')
        self.bg = np.abs(self.bg)

        # subset model
        if self.subset=='north':
            self.bg = self.bg.where(self.bg.nav_lat>-59.9858036, drop=True)
        if self.subset=='south':
            self.bg = self.bg.where(self.bg.nav_lat<-59.9858036, drop=True)

        bg_mean  = self.bg.mean(['x','y'])
        bg_std   = self.bg.std(['x','y'])
        bg_mean  = bg_mean.rename({'bx':'bx_ts_mean', 'by':'by_ts_mean'})
        bg_std   = bg_std.rename({'bx':'bx_ts_std', 'by':'by_ts_std'})
        bg_stats = xr.merge([bg_mean,bg_std])
        if save:
            bg_stats.to_netcdf(self.data_path + '/BgGliderSamples' +
                    '/SOCHIC_PATCH_3h_20121209_20130331_bg_stats_timeseries' + 
                    self.append + '.nc')

    def get_hist_stats(self, hist_set, bins):    
        ''' get mean, lower and upper deciles of group of histograms '''
        bin_centers = (bins[:-1] + bins[1:]) / 2
        hist_array = xr.DataArray(hist_set, dims=('sets', 'bin_centers'), 
                                  coords={'bin_centers': bin_centers})
        hist_mean = hist_array.mean('sets')
        hist_l_quant, hist_u_quant = hist_array.quantile([0.1,0.9],'sets')
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

        hists = []


        def get_stats_across_hists(samp):
            sample_size = len(samp.time_counter)
            print (samp.time_counter.min().values)
            print (samp.time_counter.max().values)
            print (sample_size)
            # calculate set of histograms
            for i, sample in enumerate(random):
                sample_set = samp.isel(sample=sample)#.b_x_ml
                set_stacked = sample_set.stack(z=('time_counter','sample'))
                hist, bins = np.histogram(set_stacked.dropna('z', how='all'),
                                   range=self.hist_range, density=True, bins=20)
                #                    range=(1e-9,5e-8), density=True)
                hists.append(hist)
            
            # calculate spread across histogram set
            hist_mean, hist_l_quant, hist_u_quant = self.get_hist_stats(
                                                                    hists, bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            hist_ds = xr.Dataset({'hist_mean':(['bin_centers'], hist_mean),
                                  'hist_l_dec':(['bin_centers'], hist_l_quant),
                                  'hist_u_dec':(['bin_centers'], hist_u_quant),
                                  'sample_size':(sample_size)},
                                      coords={
                                  'bin_centers': (['bin_centers'], bin_centers),
                                  'bin_left'   : (['bin_centers'], bins[:-1]),
                                  'bin_right'  : (['bin_centers'], bins[1:])})
           #hist_ds = hist_ds.assign_coords({'week_start':samples.time_counter})
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
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_' +
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

        def get_hist(bg):
            ''' calculate histogram and assign to xarray dataset '''

            # stack dimensions
            stacked_bgx = bg.bx.stack(z=('time_counter','x','y'))
            stacked_bgy = bg.by.stack(z=('time_counter','x','y'))

            # histogram
            hist_x, bins = np.histogram(stacked_bgx.dropna('z', how='all'),
                                   range=self.hist_range, density=True, bins=20)
            hist_y, bins = np.histogram(stacked_bgy.dropna('z', how='all'),
                                   range=self.hist_range, density=True, bins=20)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            # assign to dataset
            hist_ds = xr.Dataset({'hist_x':(['bin_centers'], hist_x),
                                  'hist_y':(['bin_centers'], hist_y)},
                       coords={'bin_centers': (['bin_centers'], bin_centers),
                               'bin_left'   : (['bin_centers'], bins[:-1]),
                               'bin_right'  : (['bin_centers'], bins[1:])})
            return hist_ds

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
            hist_ds = rolled.groupby('time').map(get_hist)

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
                                                         get_hist)
        elif by_time == '1W_rolling':
            # split into 1 week samples, sampled by week
            hist_ds = self.bg.groupby_bins('time_counter', date_list,
                                    labels=mid_date).map(get_hist)
            hist_ds = hist_ds.rename({'time_counter_bins':'time_counter'})
        elif by_time == '2W_rolling':
            # split into 2 week samples, sampled by week
            mid_date=mid_date[1:]
            l_dl = date_list[::2] + np.timedelta64(84, 'h')
            l_label = mid_date[::2]
            hist_ds_l = self.bg.groupby_bins('time_counter', l_dl,
                         labels=l_label).map(get_hist)
            u_dl = date_list[1:-1:2] + np.timedelta64(84, 'h')
            u_label = mid_date[1:-1:2]# + np.timedelta64(1, 'W')
            hist_ds_u = self.bg.groupby_bins('time_counter', u_dl,
                         labels=u_label).map(get_hist)
            hist_ds = xr.merge([hist_ds_u, hist_ds_l])
            hist_ds = hist_ds.rename({'time_counter_bins':'time_counter'})
        elif by_time == '3W_rolling':
            # split into 3 week samples, sampled by week
            mid_date=mid_date[1:]
            l_dl = date_list[::3]
            l_label = mid_date[::3]
            hist_ds_l = self.bg.groupby_bins('time_counter', l_dl,
                         labels=l_label).map(get_hist)
            m_dl = date_list[1:-1:3]
            m_label = mid_date[1:-1:3]
            hist_ds_m = self.bg.groupby_bins('time_counter', m_dl,
                         labels=m_label).map(get_hist)
            u_dl = date_list[2:-1:3]
            u_label = mid_date[2:-1:3]
            hist_ds_u = self.bg.groupby_bins('time_counter', u_dl,
                         labels=u_label).map(get_hist)
            hist_ds = xr.merge([hist_ds_u, hist_ds_m, hist_ds_l])
            hist_ds = hist_ds.rename({'time_counter_bins':'time_counter'})
        else:
            # entire timeseries
            hist_ds = get_hist(self.bg)

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
        plt.legend(title='ensemble size')
 
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

        def pre_proc(ds):
            ds = ds.expand_dims('ensemble_size')
            return ds

        # get data
        #prep = 'BgGliderSamples/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_'
        #ensemble_list = [self.data_path + prep + str(i).zfill(2) +
        #                 '_timeseries' + self.append + '.nc' 
        #                 for i in range(1,31)]
        #ensembles = xr.open_mfdataset(ensemble_list, 
        #                           combine='nested', concat_dim='ensemble_size',
        #                           preprocess=pre_proc).load()
        #ensembles = ensembles.assign_coords(ensemble_size=np.arange(1,31))
        #ensembles['time_counter'] = ensembles.time_counter / 1e9 
        #unit = "seconds since 1970-01-01 00:00:00"
        #ensembles.time_counter.attrs['units'] = unit
        #ensembles = xr.decode_cf(ensembles)

        g = xr.open_dataset(self.data_path + 'BgGliderSamples' + 
                    '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_timeseries' + 
                    self.append + '.nc').dropna(dim='day')
        m = xr.open_dataset(self.data_path + 'BgGliderSamples' + 
                    '/SOCHIC_PATCH_3h_20121209_20130331_bg_stats_timeseries' + 
                    self.append + '.nc')

        # change in bg
        m_week = m.resample(time_counter='1W',skipna=True).mean()
        m0 = m_week.sel(time_counter=t0, method='nearest')
        m1 = m_week.sel(time_counter=t1, method='nearest')

        print (g)
        g0_mean = g.b_x_ml_week_mean.sel(day=t0, method='nearest')
        g1_mean = g.b_x_ml_week_mean.sel(day=t1, method='nearest')
        g0_std = g.b_x_ml_week_std.sel(day=t0, method='nearest')
        g1_std = g.b_x_ml_week_std.sel(day=t1, method='nearest')
       
        #m0 = m.sel(time_counter=slice('2013-01-01 00:00:00',
        #                         '2013-01-15 00:00:00')).mean('time_counter')
        #m1 = m.sel(time_counter=slice('2013-03-01 00:00:00',
        #                         '2013-03-15 00:00:00')).mean('time_counter')
        #g0 = ensembles.where(
        #         (ensembles.time_counter>np.datetime64('2013-01-01 00:00:00')) &
        #         (ensembles.time_counter<np.datetime64('2013-01-15 00:00:00'))
        #                     ).mean('distance')
        #g1 = ensembles.where(
        #         (ensembles.time_counter>np.datetime64('2013-03-01 00:00:00')) &
        #         (ensembles.time_counter<np.datetime64('2013-03-15 00:00:00'))
        #                     ).mean('distance')
        model_delta_x_mean = m0.bx_ts_mean - m1.bx_ts_mean
        model_delta_y_mean = m0.by_ts_mean - m1.by_ts_mean
        model_delta_x_std = m0.bx_ts_std - m1.bx_ts_std
        model_delta_y_std = m0.by_ts_std - m1.by_ts_std
        glider_delta_mean = (g0_mean - g1_mean).quantile([0.05,0.95],'sets')
        glider_delta_std  = (g0_std  - g1_std ).quantile([0.05,0.95],'sets')

        # define fig
        self.figure, self.axs = plt.subplots(2,1, figsize=(6.5,4.0))

        # plot
        def render(ax, m_x, m_y, g):
            ax.plot(g.ensemble_size, 
                    g.sel(quantile=0.05), c='black')
            ax.plot(g.ensemble_size, 
                    g.sel(quantile=0.95), c='black')
            ax.axhline(m_x, c='red')
            ax.axhline(m_y, c='green')

            # percentage error
            ax.axhline(m_x - (m_x*0.20), c='pink', ls=':')
            ax.axhline(m_x + (m_x*0.20), c='pink', ls=':')
            ax.axhline(m_x - (m_x*0.40), c='navy', ls=':')
            ax.axhline(m_x + (m_x*0.40), c='navy', ls=':')
            ax.axhline(m_x - (m_x*0.60), c='grey', ls=':')
            ax.axhline(m_x + (m_x*0.60), c='grey', ls=':')
            
            txt = ['20 %', '20 %', '40 %', '40 %', '60 %', '60 %']
            print (ax.lines)
            for i, line in enumerate(ax.lines[-6:]):
                y = line.get_ydata()[-1]
                print (line)
                ax.annotate(txt[i], xy=(1,y), xytext=(6,0), 
                            color=line.get_color(), 
                            xycoords=ax.get_yaxis_transform(),
                            textcoords='offset points',
                            size=8, va='center')

        render(self.axs[0], model_delta_x_mean, model_delta_y_mean,
               glider_delta_mean)
        render(self.axs[1], model_delta_x_std, model_delta_y_std,
               glider_delta_std)

        # labels 
        self.axs[1].set_xlabel('Ensemble size')
        self.axs[0].set_ylabel(r'$\Delta b_{x,y}$' + '\n' +
                               t0.lstrip('2013')[1:] + ' :: ' +
                               t1.lstrip('2013')[1:]+
                               '\n [mean]')
        self.axs[1].set_ylabel(r'$\Delta b_{x,y}$' + '\n' +
                               t0.lstrip('2013')[1:] + ' :: ' +
                               t1.lstrip('2013')[1:]+
                               '\n [standard deviation]')

        self.axs[0].set_ylim(-1e-8,4.5e-8)
        self.axs[1].set_ylim(-1e-8,5.0e-8)
        self.axs[0].set_xticks([])

        plt.savefig(self.case + '_bg_change_err_estimate' + self.append +
                   '_'+ t0 + '_' + t1 + '.png', dpi=600)

class bootstrap_plotting(object):
    def __init__(self, append=''):
        self.data_path = config.data_path()
        if append == '':
            self.append = ''
        else:
            self.append='_' + append

        self.hist_range = (0,2e-8)

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
        ds = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_' +
                           str(n).zfill(2) + '_hist' + self.append 
                          + '_' + by_time + '.nc')
        try:
            ds = ds.rename({'time_counter_bins':'time_counter'})
        except:
            print ('')
        print (ds)

        ds_all = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_' +
                          str(n).zfill(2) + '_hist' + self.append + '.nc')
        if style=='bar':
            for i, (_, week) in enumerate(ds.groupby('time_counter')):
                self.axs.flatten()[i].barh(week.bin_left, 
                                 week.hist_u_dec - week.hist_l_dec, 
                                 height=week.bin_right - week.bin_left,
                                 color=c,
                                 alpha=0.2,
                                 left=week.hist_l_dec, 
                                 align='edge',
                                 label='gliders: ' + str(n))
                self.axs.flatten()[i].text(0.3, 0.9, 
                                  week.time_counter.dt.strftime('%m-%d').values,
                                  transform=self.axs.flatten()[i].transAxes,
                                  fontsize=6)
                #self.axs.flatten()[i].text(0.3, 0.8, 
                #                  str(week.sample_size.values),
                #                  transform=self.axs.flatten()[i].transAxes,
                #                  fontsize=6)

            self.axs[1,-1].barh(ds_all.bin_left, 
                             ds_all.hist_u_dec - ds_all.hist_l_dec, 
                             height=ds_all.bin_right - ds_all.bin_left,
                             color=c,
                             alpha=0.2,
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
        try:
            ds = ds.rename({'time_counter_bins':'time_counter'})
        except:
            print ('')
        ds = ds.isel(time_counter=slice(None,-1))
        ds_all = xr.open_dataset(self.path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                        self.append + '.nc')
       
        ds['hist'] = (ds.hist_x + ds.hist_y) / 2
        ds_all['hist'] = (ds_all.hist_x + ds_all.hist_y) / 2

        if style=='bar':
            for i, (_, week) in enumerate(ds.groupby('time_counter')):
                print (week.time_counter)
                self.axs.flatten()[i].vlines(week.hist,
                        week.bin_left, week.bin_right,
                       transform=self.axs.flatten()[i].transData,
                       colors='black', lw=0.8, label='model bgx')
            self.axs[1,-1].vlines(ds_all.hist,
                       ds_all.bin_left, ds_all.bin_right,
                       transform=self.axs[1,-1].transData,
                       colors='black', lw=0.8, label='model bgx')
        if style=='plot':
            self.ax.plot(ds.bin_centers, ds.hist, c='black', lw=0.8,
                         label='model bg')

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
       
        # time splitting
        dbdx = dbdx.swap_dims({'distance':'time'})
        print (dbdx)
        hist_ds = dbdx.resample(time='1W', skipna=True).map(
                                get_hist)
        for i, (_,week) in enumerate(hist_ds.groupby('time')):
            if i == 15: continue
            ax = self.axs.flatten()[i]
            ax.vlines(week.hist, week.bin_left, week.bin_right,
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
        self.add_giddy(self.axs[0,0])

        sample_sizes = [1, 4, 20]
        colours = ['g', 'b', 'r', 'y', 'c']

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
        
        plt.savefig(case + '_bg_sampling_skill' + self.append + '_'
                    + by_time + '.png',dpi=600)

    def plot_rmse_over_ensemble_sizes_and_week(self, case):
        ''' plot the root mean squared error of the 1 s.d. (? not decile)
            from the **real** mean over week and ensemble size
            contourf
        '''
        # load weekly model hists
        m = xr.open_dataset(self.data_path + case + 
                     '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist' + 
                     self.append + '_weekly.nc').isel(
                     time_counter=slice(None,-1))
        
        # pre merge function
        def pre_proc(ds):
            ds = ds.expand_dims('ensemble_size')
            return ds
         
        # load weekly glider hists
        prep = case + '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_'
        ensemble_list = [self.data_path + prep + str(i).zfill(2) + '_hist' + 
                         self.append + '_weekly.nc'
                         for i in range(1,31)]
        ensembles = xr.open_mfdataset(ensemble_list, 
                                   combine='nested', concat_dim='ensemble_size',
                                     preprocess=pre_proc).load()
        ensembles = ensembles.assign_coords(ensemble_size=np.arange(1,31))

        # mean of the vectors
        m_bg_abs = (m.hist_x + m.hist_y) / 2

        # rmse
        def rmsep(pred, true):
            norm = (pred - true)/true 
            return np.sqrt(((norm)**2).mean(dim='bin_centers')) * 100

        # calculate rmse
        rmse_l = rmsep(ensembles.hist_l_dec, m_bg_abs)
        rmse_u = rmsep(ensembles.hist_u_dec, m_bg_abs)
        rmse_mean = rmsep(ensembles.hist_mean, m_bg_abs)

        # initialise plot
        fig, axs = plt.subplots(2, figsize=(6.5,4))
        plt.subplots_adjust(left=0.08, right=0.87, hspace=0.1, bottom=0.15,
                            top=0.98)

        # render
        cmap = plt.cm.inferno
        lev = np.linspace(0,300,11)
        p0 = axs[0].contourf(rmse_u.time_counter, rmse_u.ensemble_size, rmse_u,
                             levels=lev, cmap=cmap)
        lev = np.linspace(0,100,11)
        p1 = axs[1].contourf(rmse_l.time_counter, rmse_l.ensemble_size, rmse_l,
                             levels=lev, cmap=cmap)

        # colour bar upper
        pos = axs[0].get_position()
        cbar_ax = fig.add_axes([0.88, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p0, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.1, 0.5, 'RMSE of\nbuoyancy gradients (%)', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left', multialignment='center')

        # colour bar lower
        pos = axs[1].get_position()
        cbar_ax = fig.add_axes([0.88, pos.y0, 0.02, pos.y1 - pos.y0])
        cbar = fig.colorbar(p1, cax=cbar_ax, orientation='vertical')
        cbar.ax.text(4.1, 0.5, 'RMSE of\nbuoyancy gradients (%)', fontsize=8,
                     rotation=90, transform=cbar.ax.transAxes,
                     va='center', ha='left', multialignment='center')

        # text labels
        axs[0].text(0.99, 0.98, 'upper decile', c='w', va='top', ha='right',
                    transform=axs[0].transAxes)
        axs[1].text(0.99, 0.98, 'lower decile', c='w', va='top', ha='right',
                    transform=axs[1].transAxes)

        # axes labels
        for ax in axs:
            ax.set_ylabel('ensemble size')
            ax.set_xticks(rmse_mean.time_counter)
        axs[1].set_xlabel('date (month-day)')

        # set xlabels
        week_labels = rmse_mean.time_counter.dt.strftime('%m-%d').values
        axs[0].set_xticklabels([])
        axs[1].set_xticklabels(week_labels)

        # rotate labels
        for label in axs[1].get_xticklabels(which='major'):
            label.set(rotation=35, horizontalalignment='right')

        plt.savefig(case + '_bg_RMSE_weekly' 
                    + self.append + '.png', dpi=600)

def plot_hist(by_time=None):
    cases = ['EXP10', 'EXP08', 'EXP13']
    cases = ['EXP10']
    if by_time:
        boot = bootstrap_plotting()
        boot.plot_histogram_buoyancy_gradients_and_samples_over_time(
                                                              'EXP10', by_time)
        #boot.plot_rmse_over_ensemble_sizes_and_week('EXP10')
    
    else:
        for case in cases:
            print ('case: ', case)
            m = bootstrap_glider_samples(case, var='b_x_ml', load_samples=False,
                                         subset='')
            m.plot_histogram_buoyancy_gradients_and_samples()
            #m.plot_rmse_over_ensemble_sizes()


def prep_hist(by_time=None):
    cases = ['EXP10', 'EXP08', 'EXP13']
    cases = ['EXP10']
    for case in cases:
        m = bootstrap_glider_samples(case, var='b_x_ml', load_samples=True,
                                     subset='')
        if by_time:
             m.append =  m.append + '_' + by_time
        m.get_full_model_hist(save=True, by_time=by_time)
        for n in range(1,31):
            print (n)
            m.get_glider_sampled_hist(n=n, save=True, by_time=by_time)

def prep_timeseries(subset=''):
    cases = ['EXP10']
    #cases = ['EXP10', 'EXP08', 'EXP13']
    for case in cases:
        m = bootstrap_glider_samples(case, var='b_x_ml', load_samples=True,
                                     subset=subset)
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
##plot_quantify_delta_bg()
##plot_quantify_delta_bg(subset='north')
##plot_quantify_delta_bg(subset='south')


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

prep_hist(by_time='2W_rolling')
prep_hist(by_time='3W_rolling')
#prep_hist(by_time='2W_rolling')
#prep_hist(by_time='1W_rolling')
#prep_hist(by_time='1W_rolling')
#prep_hist(by_time='3W_rolling')
#prep_hist(by_time='2W_rolling')
#prep_timeseries()
#plot_timeseries()
#plot_quantify_delta_bg()
print ('done 1')
#m = bootstrap_glider_samples('EXP08')
#m.histogram_buoyancy_gradients_and_samples()
#print ('done 2')
#m = bootstrap_glider_samples('EXP10')
#m.histogram_buoyancy_gradients_and_samples()
#m.plot_error_bars()

#def plot_histogram():
#    m = glider_nemo('EXP03')
#    m.load_nemo_bg()
#    #m.load_glider_nemo()
#    #m.sub_sample_nemo()
#    m.histogram_buoyancy_gradient()
#
#plot_histogram()
