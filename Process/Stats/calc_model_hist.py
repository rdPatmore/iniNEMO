import config
import xarray as xr
import numpy as np

class model_hist(object):
    """
    Calculate PDF of variable
    """

    def __init__(self, case, depth='10'):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'
        self.glider_fn = 'glider_uniform_interp_1000_parallel_transects.nc'

        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
        self.d0 = '20121209'
        self.d1 = '20130331'
        self.date_str = '_{0}_{1}_'.format(self.d0,self.d1)

        self.depth = depth

    def quadrant_partition(self, da, quadrant):
        ''' 
        partition domain into one of four quadrants:
        upper_left, upper_right, lower_left, lower_right

        NB: copy from Physics/calc_glider_relevent_diags.py needs unifying
        '''

        # define mid-latitude and -longitude
        da_lon_mean = da.nav_lon.mean().values
        da_lat_mean = da.nav_lat.mean().values

        # get quadrant bounds
        if quadrant == 'upper_left':
            bounds = (da.nav_lon < da_lon_mean) & (da.nav_lat > da_lat_mean)
        elif quadrant == 'upper_right':
            bounds = (da.nav_lon > da_lon_mean) & (da.nav_lat > da_lat_mean)
        elif quadrant == 'lower_left':
            bounds = (da.nav_lon < da_lon_mean) & (da.nav_lat < da_lat_mean)
        elif quadrant == 'lower_right':
            bounds = (da.nav_lon > da_lon_mean) & (da.nav_lat < da_lat_mean)
        else:
            print ("no quadrent selected: do nothing")
            return da

        # cut to quadrant
        da_quad = da.where(bounds.compute(), drop=True)

        return da_quad

    def get_Ro_hist(self):
        """
        Create files for histogram plotting. Takes output from model_object.
    
        """
    
        # load buoyancy gradients       
        self.get_model_rossby_number()
    
        # get hist
        self.get_ro_z_hist(self.Ro, save=True)

    def get_model_rossby_number(self, ml=False):
        ''' get model Rossby number '''
    
        Ro = xr.open_dataset(config.data_path_old() + self.case +
                             '/rossby_number.nc', chunks=-1)
       
        Ro = np.abs(Ro.isel(depth=0)).Ro.load()
        print ('loaded')
        
        # load samples
        self.samples = xr.open_dataset(self.data_path + 'GliderRandomSampling/'
                                     + self.glider_fn, chunks={'sample':1})

        # unify times
        self.samples['time_counter'] = self.samples.time_counter.isel(
                                       sample=0).drop('sample')
     
        # set glider time bounds
        float_time = self.samples.time_counter.astype('float64')
        clean_float_time = float_time.where(float_time > 0, np.nan)
        start = clean_float_time.min().astype('datetime64[ns]')
        end   = clean_float_time.max().astype('datetime64[ns]')

        # cut to glider time span
        self.Ro = Ro.sel(time_counter=slice(start,end))

    def get_N2(self, partition=None, quad=None):
        ''' get N2 '''

        path = self.data_path + 'ProcessedVars' + self.file_id

        N2_str = 'bn2_' + self.depth

        if partition:
            self.var = xr.open_dataset(path + N2_str + '_ice_oce_miz.nc',
                       chunks='auto')[N2_str + '_' + partition]
            self.var_name = N2_str + '_' + partition
        else:
            self.var = xr.open_dataarray(path + N2_str + '.nc',
                       chunks={'time_counter':1})
            self.var_name = 'bn2_' + self.depth
        
        if quad:
            self.var = self.quadrant_partition(self.var, quad)
            self.var_name = self.var_name + '_' + quad

    def get_M2(self, partition=None, quad=None):
        ''' get M2 '''

        path = self.data_path + 'ProcessedVars' + self.file_id

        M4_str = 'bg_mod2_' + self.depth

        if partition:
            M4 = xr.open_dataset(path + M4_str + '_ice_oce_miz.nc',
                       chunks='auto')[M4_str + '_' + partition]
            self.var_name = M4_str + '_' + partition
        else:
            M4 = xr.open_dataarray(path + M4_str + '.nc',
                       chunks='auto')
            self.var_name = 'bg_mod2_' + self.depth

        # get M2 from M4
        self.var = M4 ** 0.5

        if quad:
            self.var = self.quadrant_partition(self.var, quad)
            self.var_name = self.var_name + '_' + quad

    def get_M2_over_N2(self, partition=None, quad=None):
        ''' get M2/N2 '''

        path = self.data_path + 'ProcessedVars' + self.file_id

        N2_str = 'bn2_' + self.depth
        M4_str = 'bg_mod2_' + self.depth

        if partition:
            M4 = xr.open_dataset(path + M4_str + '_ice_oce_miz.nc',
                       chunks='auto')[M4_str + '_' + partition]
            N2 = xr.open_dataset(path + N2_str + '_ice_oce_miz.nc',
                       chunks='auto')[N2_str + '_' + partition]
            self.var_name = 'M2_over_N2_{}_'.format(self.depth) + partition
        else:
            M4 = xr.open_dataarray(path + M4_str + '.nc',
                       chunks='auto')
            N2 = xr.open_dataarray(path + N2_str + '.nc',
                       chunks='auto')
            N2 = N2.isel(x=slice(2,-2),y=slice(2,-2))
            self.var_name = 'M2_over_N2_' + self.depth

        # get M2 from M4
        M2 = M4 ** 0.5

        M2['time_counter'] = N2.time_counter
        self.var = M2/N2

        if quad:
            self.var = self.quadrant_partition(self.var, quad)
            self.var_name = self.var_name + '_' + quad

    def get_M2_and_N2(self, partition=None, quad=None):
        ''' get M2 and N2 as separate variable assignments '''

        path = self.data_path + 'ProcessedVars' + self.file_id

        N2_str = 'bn2_' + self.depth
        M4_str = 'bg_mod2_' + self.depth

        if partition:
            M4 = xr.open_dataset(path + M4_str + '_mid_ice_oce_miz.nc',
                       chunks='auto')[M4_str + '_' + partition]
            N2 = xr.open_dataset(path + N2_str + '_ice_oce_miz.nc',
                       chunks='auto')[N2_str + '_' + partition]
        else:
            M4 = xr.open_dataarray(path + M4_str + '.nc',
                       chunks='auto')
            N2 = xr.open_dataarray(path + N2_str + '.nc',
                       chunks='auto')
            N2 = N2.isel(x=slice(2,-2),y=slice(2,-2))

        # get M2 from M4
        M2 = M4 ** 0.5

        M2['time_counter'] = N2.time_counter

        self.var0 = M2
        self.var1 = N2

        if quad:
            self.var0 = self.quadrant_partition(self.var0, quad)
            self.var1 = self.quadrant_partition(self.var1, quad)
            self.var0.name = self.var0.name + '_' + quad
            self.var1.name = self.var1.name + '_' + quad

    def cut_time_window(self, dates):
        ''' reduce time period to bounds '''

        if type(dates) is list:
            self.var = self.var.sel(time_counter=slice(dates[0],dates[1]))
            self.d0 = dates[0]
            self.d1 = dates[1]
            self.date_str = '_{0}_{1}_'.format(self.d0,self.d1)
        else:
            self.var = self.var.sel(time_counter=dates, method='nearest')
            self.date_str = '_{0}_'.format(dates[:8])

    def cut_time_window_2var(self, dates):
        ''' reduce time period to bounds '''

        if type(dates) is list:
            self.var0 = self.var0.sel(time_counter=slice(dates[0],dates[1]))
            self.var1 = self.var1.sel(time_counter=slice(dates[0],dates[1]))
            self.d0 = dates[0]
            self.d1 = dates[1]
            self.date_str = '_{0}_{1}_'.format(self.d0,self.d1)
        else:
            self.var0 = self.var0.sel(time_counter=dates, method='nearest')
            self.var1 = self.var1.sel(time_counter=dates, method='nearest')
            self.date_str = '_{0}_'.format(dates[:8])

    def get_var_z_hist(self, lims, bins=20, save=True, density=True):
        ''' calculate histogram and assign to xarray dataset '''

        # stack dimensions
        stacked = self.var.stack(z=self.var.dims)

        # histogram
        hist_var, bins = np.histogram(
                            stacked.dropna('z', how='all'),
                            range=lims, density=density, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # assign to dataset
        hist_ds = xr.Dataset(
                   {'hist_' + self.var_name:(['bin_centers'], hist_var)},
                   coords={'bin_centers': (['bin_centers'], bin_centers),
                           'bin_left'   : (['bin_centers'], bins[:-1]),
                           'bin_right'  : (['bin_centers'], bins[1:])})
        if save:
            hist_ds.to_netcdf(self.data_path + 
            '/BGHists/SOCHIC_PATCH_3h' + self.date_str
              + self.var_name + '_model_hist.nc')
        return hist_ds

    def get_2d_var_z_hist(self, bins=20, save=True, density=True):
        ''' calculate 2d histogram and assign to xarray dataset '''

        # stack dimensions
        v0_stacked = self.var0.stack(z=self.var0.dims)
        v1_stacked = self.var1.stack(z=self.var0.dims)

        # histogram
        hist_var, x_bins, y_bins = np.histogram2d(
                            v0_stacked.dropna('z', how='all'),
                            v1_stacked.dropna('z', how='all'),
                            density=density, bins=bins)
        x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2

        # assign to dataset
        hist_ds = xr.Dataset(
                   {'hist_' + self.var0.name + '_' + self.var1.name:(
                                           ['x_bin_centers','y_bin_centers'],
                                           hist_var)},
                   coords={'x_bin_centers': (['x_bin_centers'], x_bin_centers),
                           'x_bin_left'   : (['x_bin_centers'], x_bins[:-1]),
                           'x_bin_right'  : (['x_bin_centers'], x_bins[1:]),
                           'y_bin_centers': (['y_bin_centers'], y_bin_centers),
                           'y_bin_left'   : (['y_bin_centers'], y_bins[:-1]),
                           'y_bin_right'  : (['y_bin_centers'], y_bins[1:])})
        if save:
            hist_ds.to_netcdf(self.data_path + 
            '/BGHists/SOCHIC_PATCH_3h{0}{1}_{1}'.format(
                self.date_str, self.var0.name, self.var1.name) +
               '_model_hist.nc')
        return hist_ds

if __name__ == "__main__":

    def M2_over_N2_hist():
        hist = model_hist('EXP10')
        hist.get_M2_over_N2()
        hist.cut_time_window(['20121209','20130111'])
        hist.var = hist.var.compute()
        bins = np.logspace(-16,0,50)
        hist.get_var_z_hist(lims=[0, 1e0], bins=bins, density=True)

    def M2_over_N2_hist_partition(dates, depth='ml_mid', quad=None):
        hist = model_hist('EXP10')
        hist_list = []
        for partition in ['ice','oce','miz']:
            hist.get_M2_over_N2(partition=partition, quad=quad)
            hist.cut_time_window(dates)
            hist.var = hist.var.compute()
            bins = np.logspace(-8,0,50)
            hist_list.append(hist.get_var_z_hist(lims=[0, 1e0], bins=bins,
                             density=False, save=False))
        hist_partition = xr.merge(hist_list)
        if not quad: quad = ''
        hist_partition.to_netcdf(hist.data_path +
        '/BGHists/SOCHIC_PATCH_3h' + hist.date_str +  quad + '_'
              + 'M2_over_N2_{}_ice_oce_miz_model_hist.nc'.format(depth))

    def M2_hist():
        hist = model_hist('EXP10')
        hist.get_M2()
        hist.cut_time_window(['20121209','20130111'])
        bins = np.logspace(-27,-11,50)
        hist.get_var_z_hist(lims=[0, 1e0], bins=bins, density=True)

    def M2_hist_partition(dates, depth='ml_mid', quad=None):
        hist = model_hist('EXP10')
        hist_list = []
        for partition in ['ice','oce','miz']:
            hist.get_M2(partition=partition, quad=quad)
            hist.cut_time_window(dates)
            bins = np.logspace(-16,-2,50)
            hist_list.append(hist.get_var_z_hist(lims=[0, 1e0], bins=bins,
                             density=False, save=False))
        hist_partition = xr.merge(hist_list)
        if not quad: quad = ''
        hist_partition.to_netcdf(hist.data_path +
        '/BGHists/SOCHIC_PATCH_3h' + hist.date_str + quad + '_'
              + 'bg_mod2_{}_ice_oce_miz_model_hist.nc'.format(depth))

    def N2_hist():
        hist = model_hist('EXP10')
        hist.get_N2()
        hist.cut_time_window(['20121209','20130111'])
        bins = np.logspace(-16,-2,50)
        hist.get_var_z_hist(lims=[0, 1e0], bins=bins, density=True)

    def N2_hist_partition(dates, depth='ml_mid', quad=None):
        hist = model_hist('EXP10')
        hist_list = []
        for partition in ['ice','oce','miz']:
            hist.get_N2(partition=partition, quad=quad)
            #hist.cut_time_window('20121223 12:00:00')
            hist.cut_time_window(dates)
            bins = np.logspace(-16,-2,50)
            hist_list.append(hist.get_var_z_hist(lims=[0, 1e0], bins=bins,
                             density=False, save=False))
        hist_partition = xr.merge(hist_list)
        if not quad: quad = ''
        hist_partition.to_netcdf(hist.data_path +
        '/BGHists/SOCHIC_PATCH_3h' + hist.date_str + quad + '_'
              + 'bn2_{}_ice_oce_miz_model_hist.nc'.format(depth))

    def M2_N2_2d_hist():
        hist = model_hist('EXP10')
        hist.get_M2_and_N2()
        hist.cut_time_window_2var(['20121209','20130111'])
        hist.var0 = hist.var0.compute()
        hist.var1 = hist.var1.compute()
        M2_bins = np.logspace(-16,-2,100)
        N2_bins = np.logspace(-16,-2,100)
        hist.get_2d_var_z_hist(bins=[M2_bins,N2_bins], save=True, density=False)

    def M2_N2_2d_hist_partition(dates, depth='ml_mid',  quad=None):
        hist = model_hist('EXP10')
        hist_list = []
        for partition in ['ice','oce','miz']:
            hist.get_M2_and_N2(partition=partition, quad=quad)
            hist.cut_time_window_2var(dates)
            hist.var0 = hist.var0.compute()
            hist.var1 = hist.var1.compute()
            M2_bins = np.logspace(-16,-2,100)
            N2_bins = np.logspace(-16,-2,100)
            hist_list.append(hist.get_2d_var_z_hist(bins=[M2_bins,N2_bins],
                             save=False, density=False))
        hist_partition = xr.merge(hist_list)
        if not quad: quad = ''
        hist_partition.to_netcdf(hist.data_path + 
        '/BGHists/SOCHIC_PATCH_3h' + hist.date_str + quad + '_'
         + 'bg_mod2_bn2_{}_ice_oce_miz_model_hist.nc'.format(depth))

    #M2_over_N2_hist()
    #N2_hist()
    #M2_hist()
    #M2_N2_2d_hist()
    print ('start')
    #dates = '20121223 12:00:00'
    dates = ['20121209','20130111']
    depth='10'
    #quad = 'lower_right'
    quad = None
    M2_hist_partition(dates, depth)
    print ('0')
    N2_hist_partition(dates, depth)
    print ('1')
    M2_over_N2_hist_partition(dates, depth)
    print ('2')
    M2_N2_2d_hist_partition(dates, depth)
    print ('end')
