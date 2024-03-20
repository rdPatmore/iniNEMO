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

    def get_hist(self):
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

    def get_ro_z_hist(self, ro, bins=20, save=True):
        ''' calculate histogram and assign to xarray dataset '''

        # stack dimensions
        stacked = ro.stack(z=('time_counter','x','y'))

        # histogram
        hist_ro, bins = np.histogram(
                            stacked.dropna('z', how='all'),
                            range=[0,2], density=False, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # assign to dataset
        hist_ds = xr.Dataset({'hist_ro':(['bin_centers'], hist_ro)},
                   coords={'bin_centers': (['bin_centers'], bin_centers),
                           'bin_left'   : (['bin_centers'], bins[:-1]),
                           'bin_right'  : (['bin_centers'], bins[1:])})
        if save:
            hist_ds.to_netcdf(self.data_path + 
                  '/BGHists/' + self.file_id + '_Ro_model_hist.nc')
        return hist_ds

hist = model_hist('EXP10')
hist.get_hist()
