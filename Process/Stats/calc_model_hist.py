import config
import xarray as xr
import numpy as np

class model_hist(object):
    """
    Calculate PDF of variable
    """

    def __init__(self, case):
        self.root = config.root()
        self.case = case
        self.data_path = config.data_path() + self.case + '/'
        self.glider_fn = 'glider_uniform_interp_1000_parallel_transects.nc'

        self.file_id = '/SOCHIC_PATCH_3h_20121209_20130331_'
        self.d0 = '20121209'
        self.d1 = '20130331'

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

    def get_N2(self):
        ''' get N2 '''

        path = self.data_path + 'ProcessedVars' + self.file_id
        self.var = xr.open_dataarray(path + 'bn2_ml_mid.nc',
                   chunks={'time_counter':1})
        self.var_name = 'bn2_ml_mid'

    def get_M2(self):
        ''' get M2 '''

        path = self.data_path + 'ProcessedVars' + self.file_id
        self.var = xr.open_dataarray(path + 'bg_mod2_ml_mid.nc',
                   chunks='auto')
        self.var_name = 'bg_mod2_ml_mid'

    def get_M2_over_N2(self):
        ''' get M2/N2 '''

        path = self.data_path + 'ProcessedVars' + self.file_id
        M2 = xr.open_dataarray(path + 'bg_mod2_ml_mid.nc',
                   chunks='auto')
        N2 = xr.open_dataarray(path + 'bn2_ml_mid.nc',
                   chunks='auto')
        N2 = N2.isel(x=slice(2,-2),y=slice(2,-2))
        M2['time_counter'] = N2.time_counter
        self.var = M2/N2
        self.var_name = 'M2_over_N2_ml_mid'

    def get_M2_and_N2(self):
        ''' get M2 and N2 as separate variable assignments '''
        path = self.data_path + 'ProcessedVars' + self.file_id
        M2 = xr.open_dataarray(path + 'bg_mod2_ml_mid.nc',
                   chunks='auto')
        N2 = xr.open_dataarray(path + 'bn2_ml_mid.nc',
                   chunks='auto')
        N2 = N2.isel(x=slice(2,-2),y=slice(2,-2))
        M2['time_counter'] = N2.time_counter

        self.var0 = M2
        self.var1 = N2

    def cut_time(self, dates):
        ''' reduce time period to bounds '''
        self.var = self.var.sel(time_counter=slice(dates[0],dates[1]))
        self.d0 = dates[0]
        self.d1 = dates[1]

    def cut_time_2var(self, dates):
        ''' reduce time period to bounds '''
        self.var0 = self.var0.sel(time_counter=slice(dates[0],dates[1]))
        self.var1 = self.var1.sel(time_counter=slice(dates[0],dates[1]))
        self.d0 = dates[0]
        self.d1 = dates[1]

    def get_var_z_hist(self, lims, bins=20, save=True, density=True):
        ''' calculate histogram and assign to xarray dataset '''

        # stack dimensions
        stacked = self.var.stack(z=('time_counter','x','y'))

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
            '/BGHists/SOCHIC_PATCH_3h_{0}_{1}_'.format(self.d0, self.d1)
              + self.var_name + '_model_hist.nc')
        return hist_ds

    def get_2d_var_z_hist(self, bins=20, save=True, density=True):
        ''' calculate 2d histogram and assign to xarray dataset '''

        # stack dimensions
        v0_stacked = self.var0.stack(z=('time_counter','x','y'))
        v1_stacked = self.var1.stack(z=('time_counter','x','y'))

        # histogram
        hist_var, x_bins, y_bins = np.histogram2d(
                            v0_stacked.dropna('z', how='all'),
                            v1_stacked.dropna('z', how='all'),
                            density=density, bins=bins)
        x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
        y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2

        # assign to dataset
        hist_ds = xr.Dataset(
                   {'hist_' + self.var1.name + '_' + self.var1.name:(
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
            '/BGHists/SOCHIC_PATCH_3h_{0}_{1}_{2}_{3}'.format(
                self.d0, self.d1, self.var0.name, self.var1.name) +
               '_model_hist.nc')
        return hist_ds

if __name__ == "__main__":

    def M2_over_N2_hist():
        hist = model_hist('EXP10')
        hist.get_M2_over_N2()
        hist.cut_time(['20121209','20130111'])
        hist.var = hist.var.compute()
        bins = np.logspace(-16,0,50)
        hist.get_var_z_hist(lims=[0, 1e0], bins=bins, density=True)

    def M2_hist():
        hist = model_hist('EXP10')
        hist.get_M2()
        hist.cut_time(['20121209','20130111'])
        bins = np.logspace(-27,-11,50)
        hist.get_var_z_hist(lims=[0, 1e0], bins=bins, density=True)

    def N2_hist():
        hist = model_hist('EXP10')
        hist.get_N2()
        hist.cut_time(['20121209','20130111'])
        bins = np.logspace(-16,-2,50)
        hist.get_var_z_hist(lims=[0, 1e0], bins=bins, density=True)

    def M2_N2_2d_hist():
        hist = model_hist('EXP10')
        hist.get_M2_and_N2()
        hist.cut_time_2var(['20121209','20130111'])
        hist.var0 = hist.var0.compute()
        hist.var1 = hist.var1.compute()
        M2_bins = np.logspace(-27,-11,100)
        N2_bins = np.logspace(-16,-2,100)
        hist.get_2d_var_z_hist(bins=[M2_bins,N2_bins], save=True, density=True)

    #M2_over_N2_hist()
    #N2_hist()
    #M2_hist()
    M2_N2_2d_hist()
