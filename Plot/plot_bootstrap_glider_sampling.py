import xarray as xr
import config
import iniNEMO.Process.model_object as mo
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import dask
import matplotlib
#import itertools
from get_transects import get_transects

matplotlib.rcParams.update({'font.size': 8})

class bootstrap_glider_samples(object):
    '''
    for ploting bootstrap samples of buoyancy gradients
    '''

    def __init__(self, case, offset=False, var='b_x_ml', load_samples=True):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'

        self.hist_range = (0,2e-8)
        #self.hist_range = (0,5e-8)
        def expand_sample_dim(ds):
            ds = ds.expand_dims('sample')
            ds['lon_offset'] = ds.attrs['lon_offset']
            ds['lat_offset'] = ds.attrs['lat_offset']
            ds = ds.set_coords(['lon_offset','lat_offset','time_counter'])
            da = ds[var]
            #if var == 'b_x_ml':
            #    da = np.abs(da)
            #da = da.sel(ctd_depth=10, method='nearest').dropna(dim='distance')
            #da = self.get_transects(da)
            return da
        if load_samples:
            sample_size = 100
        else:
            sample_size = 1
        prep = 'GliderRandomSampling/glider_uniform_interp_1000_' 
        #prep = 'GliderRandomSampling/glider_uniform_interp_1000_transects_' 
        sample_list = [self.data_path + prep + 
                       str(i).zfill(2) + '.nc' for i in range(sample_size)]
        self.samples = xr.open_mfdataset(sample_list, 
                                     combine='nested', concat_dim='sample',
                                     preprocess=expand_sample_dim).load()

        # depth average
        #self.samples = self.samples.mean('ctd_depth', skipna=True)
        #self.samples = self.samples.sel(ctd_depth=10, method='nearest')
        self.samples = self.samples.sel(ctd_depth=10, method='nearest')

        for i in range(self.samples.sample.size):
            print ('sample: ', i)
            var10 = self.samples.isel(sample=i).dropna(dim='distance')
            var10 = get_transects(var10)

        # set time to float for averaging
        float_time = self.samples.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        self.samples['time_counter'] = clean_float_time
 
        # absolute value of buoyancy gradients
        #self.samples = np.abs(self.samples)
        #self.samples['b_x_ml'] = np.abs(self.samples.b_x_ml)

        #for i in range(self.samples.sample.size):
        #    self.samples[i] = self.get_transects(self.samples.isel(sample=i))

        #self.model_rho = xr.open_dataset(self.data_path + 'rho.nc')
        #self.model_mld = xr.open_dataset(self.data_path +
        #                   'SOCHIC_PATCH_3h_20120101_20121231_grid_T.nc').mld


#    def get_transects(self, data, concat_dim='distance', method='cycle',
#                      shrink=None):
#        if method == '2nd grad':
#            a = np.abs(np.diff(data.lat, 
#            append=data.lon.max(), prepend=data.lon.min(), n=2))# < 0.001))[0]
#            idx = np.where(a>0.006)[0]
#        crit = [0,1,2,3]
#        if method == 'cycle':
#            #data = data.isel(distance=slice(0,400))
#            data['orig_lon'] = data.lon - data.lon_offset
#            data['orig_lat'] = data.lat - data.lat_offset
#            idx=[]
#            crit_iter = itertools.cycle(crit)
#            start = True
#            a = next(crit_iter)
#            for i in range(data[concat_dim].size)[::shrink]:
#                da = data.isel({concat_dim:i})
#                if (a == 0) and (start == True):
#                    test = ((da.orig_lat < -60.04) and (da.orig_lon > 0.176))
#                elif a == 0:
#                    test = (da.orig_lon > 0.176)
#                elif a == 1:
#                    test = (da.orig_lat > -59.93)
#                elif a == 2:
#                    test = (da.orig_lon < -0.173)
#                elif a == 3:
#                    test = (da.orig_lat > -59.93)
#                if test: 
#                    start = False
#                    idx.append(i)
#                    a = next(crit_iter)
#        da = np.split(data, idx)
#        transect = np.arange(len(da))
#        pop_list=[]
#        for i, arr in enumerate(da):
#            if len(da[i]) < 1:
#                pop_list.append(i) 
#            else:
#                da[i] = da[i].assign_coords({'transect':i})
#        for i in pop_list:
#            da.pop(i)
#        da = xr.concat(da, dim=concat_dim)
#        # remove initial and mid path excursions
#        da = da.where(da.transect>1, drop=True)
#        da = da.where(da.transect != da.lat.idxmin().transect, drop=True)
#        return da

    def get_model_buoyancy_gradients(self):
        ''' restrict the model time to glider time '''
    
        bg = xr.open_dataset(config.data_path() + self.case +
                             '/SOCHIC_PATCH_3h_20121209_20130331_bg.nc')
        
         
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

    def get_hist_stats(self, hist_set, bins):    
        ''' get mean, lower and upper deciles of group of histograms '''
        bin_centers = bins[:-1] + bins[1:] / 2
        hist_array = xr.DataArray(hist_set, dims=('sets', 'bin_centers'), 
                                  coords={'bin_centers': bin_centers})
        hist_mean = hist_array.mean('sets')
        hist_l_quant, hist_u_quant = hist_array.quantile([0.1,0.9],'sets')
        return hist_mean, hist_l_quant, hist_u_quant

    def get_glider_sampled_hist(self, n=1, save=False):
        '''
        add sample set of means and std to histogram
        n = sample size
        '''
 
        set_size = self.samples.sizes['sample']

        # get random group
        random = np.random.randint(set_size, size=(set_size,n))

        hists = []
        for sample in random:
            sample_set = self.samples.isel(sample=sample)#.b_x_ml
            set_stacked = sample_set.stack(z=('distance','sample'))
            hist, bins = np.histogram(set_stacked.dropna('z', how='all'),
                                range=self.hist_range, density=True, bins=20)
            #                    range=(1e-9,5e-8), density=True)
            hists.append(hist)
        hist_mean, hist_l_quant, hist_u_quant = self.get_hist_stats(hists, bins)
        if save:
            bin_centers = bins[:-1] + bins[1:] / 2
            hist_ds = xr.Dataset({'hist_mean':(['bin_centers'], hist_mean),
                                  'hist_l_dec':(['bin_centers'], hist_l_quant),
                                  'hist_u_dec':(['bin_centers'], hist_u_quant)},
                                      coords={
                               'bin_centers': (['bin_centers'], bin_centers),
                               'bin_left'   : (['bin_centers'], bins[:-1]),
                               'bin_right'  : (['bin_centers'], bins[1:])})
            hist_ds.to_netcdf(self.data_path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_glider_' +
                          str(n) + '_hist.nc')
        return hist_mean, hist_l_quant, hist_u_quant 

    def get_glider_sample_lims(self, n=1):
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

    def get_full_model_hist(self, save=False):
        # load buoyancy gradients       
        self.get_model_buoyancy_gradients()

        self.bg = self.bg.sel(deptht=10, method='nearest')
        self.bg = np.abs(self.bg)

        stacked_bgx = self.bg.bx.stack(z=('time_counter','x','y'))
        stacked_bgy = self.bg.by.stack(z=('time_counter','x','y'))

        hist_x, bins = np.histogram(stacked_bgx.dropna('z', how='all'),
                                    range=self.hist_range, density=True, bins=20)
        hist_y, bins = np.histogram(stacked_bgy.dropna('z', how='all'),
                                    range=self.hist_range, density=True, bins=20)
        if save:
            bin_centers = bins[:-1] + bins[1:] / 2
            hist_ds = xr.Dataset({'hist_x':(['bin_centers'], hist_x),
                                     'hist_y':(['bin_centers'], hist_y)},
                                      coords={
                               'bin_centers': (['bin_centers'], bin_centers),
                               'bin_left'   : (['bin_centers'], bins[:-1]),
                               'bin_right'  : (['bin_centers'], bins[1:])})
            hist_ds.to_netcdf(self.data_path + 
                          '/SOCHIC_PATCH_3h_20121209_20130331_bg_model_hist.nc')
        return hist_x, hist_y, bins

    def get_sampled_model_hist(self):
        ''' return mean and std of sampled model hists within a sample set '''

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
                           str(n) + '_hist.nc')
        print (ds)
        if style=='bar':
            self.ax.bar(ds.bin_left, 
                    ds.hist_u_dec - ds.hist_l_dec, 
                    width=ds.bin_right - ds.bin_left,
                    color=c,
                    alpha=0.2,
                    bottom=ds.hist_l_dec, 
                    label='gliders: ' + str(n))
        if style=='plot':
            self.ax.fill_between(ds.bin_centers, ds.hist_l_dec,
                                                 ds.hist_u_dec,
                                 color=c, edgecolor=None, alpha=0.2)
            self.ax.plot(ds.bin_centers, ds.hist_mean, c=c,
                         label='gliders: ' + str(n))

    def add_model_lines(self):
        hist_x, hist_y, bins = self.get_model_hist()
        bin_centers = bins[:-1] + bins[1:] / 2
        self.ax.plot(bin_centers, hist_x, c='black', lw=2, zorder=1,
                     label='Model')

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
        plt.hist(stacked_bgx, bins=50, density=True, alpha=0.3,
                 label='model bgx', fill=False, edgecolor='red',
                 histtype='step')
        plt.hist(stacked_bgy, bins=50, density=True, alpha=0.3,
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
            self.render_glider_sample_set(n=n, c=colours[i])
        print ('model')
        plt.show()
        #self.add_model_bar()

        self.ax.set_xlabel('Buoyancy Gradient')
        self.ax.set_ylabel('PDF')

        plt.legend()
        self.ax.set_xlim(0, 2e-8)
        self.ax.set_ylim(0, 3.6e8)
        plt.savefig(self.case + '_bg_sampling_skill.png', dpi=600)

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



m = bootstrap_glider_samples('EXP10', var='b_x_ml', load_samples=False)
m.plot_histogram_buoyancy_gradients_and_samples()
#m = bootstrap_glider_samples('EXP10', var='b_x_ml', load_samples=True)
#m.get_full_model_hist(save=True)
#for n in range(1,21):
#    print (n)
#    m.get_glider_sampled_hist(n=n, save=True)
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
