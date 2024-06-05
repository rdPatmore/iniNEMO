import config 
import xarray as xr
import iniNEMO.Process.Common.spatial_integrals_and_masking as sim

class glider_relevant_metrics(object):

    def __init__(self, case, file_id):
        self.case = case
        self.file_id = file_id
        self.raw_preamble = config.data_path() + case + '/RawOutput/' +\
                            self.file_id
        self.data_path = config.data_path() + case + '/'

    def mld_time_series_ice_partition(self):
        ''' get mld time series partitioned according to sea ice cover ''' 

        # get mld
        mld = xr.open_dataset(self.raw_preamble + 'grid_T.nc',
                            chunks={'time_counter':1}).mldr10_3

        # partition into zones
        im = sim.integrals_and_masks(self.case, self.file_id, mld, 'mld')
        im.horizontal_mean_ice_oce_zones()

    def save_ml_bg(self):
        ''' save mixed layer buoyancy gradients '''

        # get bg norm
        fn = self.data_path + 'ProcessedVars/' + self.file_id + 'bg_mod2.nc'
        bg = xr.open_dataset(fn, chunks={'time_counter':1}).bg_mod2

        # restrict to mixed layer and save
        im = sim.integrals_and_masks(self.case, self.file_id, bg, 'bg_mod2')
        im.mask_by_ml(save=True, cut=[slice(2,-2), slice(2,-2)])

    def bg_norm_time_series_ice_partition(self):
        ''' get bg norm time series partitioned according to sea ice cover ''' 

        # get bg norm
        fn = self.data_path + 'ProcessedVars/' + self.file_id + 'bg_mod2.nc'
        bg = xr.open_dataset(fn, chunks={'time_counter':1}).bg_mod2

        # partition into zones
        im = sim.integrals_and_masks(self.case, self.file_id, bg, 'bg_mod2')
        im.domain_mean_ice_oce_zones()

    def N2_mld_time_series_ice_partition(self):
        '''
        get N mixed layer depth time series partitioned according
        to sea ice cover 
        ''' 

        # get N at mld 
        fn = self.data_path + 'ProcessedVars/' + self.file_id + 'N2_mld.nc'
        N2_mld = xr.open_dataset(fn, chunks={'time_counter':1}).bn2

        # partition into zones
        im = sim.integrals_and_masks(self.case, self.file_id, N2_mld, 'N2_mld')
        im.horizontal_mean_ice_oce_zones()

    def save_ml_T_and_S(self):
        ''' save mixed layer temperature and salinity '''

        # get temperature and salinity
        fn = self.raw_preamble + 'grid_T.nc'
        ds = xr.open_dataset(fn, chunks={'time_counter':1})
        salt = ds.vosaline
        temp = ds.votemper

        # restrict temperature to mixed layer and save
        im = sim.integrals_and_masks(self.case, self.file_id, temp, 'votemper')
        im.mask_by_ml(save=True)

        # restrict salinity to mixed layer and save
        im = sim.integrals_and_masks(self.case, self.file_id, salt, 'vosaline')
        im.mask_by_ml(save=True)

    def temperature_time_series_ice_partition(self):
        '''
        get temperature time series partitioned according
        to sea ice cover 
        ''' 
    
        # get data
        fn = self.data_path + 'ProcessedVars/' + self.file_id + 'votemper_ml.nc'
        temp = xr.open_dataarray(fn, chunks={'time_counter':1})

        # partition temperature into zones
        im = sim.integrals_and_masks(self.case, self.file_id, temp, 'votemper')
        im.domain_mean_ice_oce_zones()

    def salinity_time_series_ice_partition(self):
        '''
        get and salinity time series partitioned according
        to sea ice cover 
        ''' 
    
        # get data
        fn = self.data_path + 'ProcessedVars/' + self.file_id + 'vosaline_ml.nc'
        salt = xr.open_dataarray(fn, chunks={'time_counter':1})

        # partition salinity into zones
        im = sim.integrals_and_masks(self.case, self.file_id, salt, 'vosaline')
        im.domain_mean_ice_oce_zones()


if __name__ == '__main__':
    case = 'EXP10'
    file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
    grm = glider_relevant_metrics(case, file_id)
    #grm.temperature_time_series_ice_partition()
    #grm.salinity_time_series_ice_partition()
    #grm.bg_norm_time_series_ice_partition()
    grm.N2_mld_time_series_ice_partition()
    #grm.save_ml_T_and_S()
    #grm.mld_time_series_ice_partition()
