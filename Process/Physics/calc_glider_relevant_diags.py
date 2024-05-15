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

    def T_and_S_time_series_ice_partition(self):
        '''
        get temperature and salinity time series partitioned according
        to sea ice cover 
        ''' 
    
        # get data
        fn = self.raw_preamble + 'grid_T.nc'
        ds = xr.open_dataset(fn, chunks={'time_counter':1})

        # get temperature 
        temp = ds.votemper
        salt = ds.vosaline

        # partition temperature into zones
        im = sim.integrals_and_masks(self.case, self.file_id, temp,
                                    'temperature')
        im.domain_mean_ice_oce_zones()

        # partition salinity into zones
        im = sim.integrals_and_masks(self.case, self.file_id, salt, 'salinity')
        im.domain_mean_ice_oce_zones()


case = 'EXP10'
file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
grm = glider_relevant_metrics(case, file_id)
grm.T_and_S_time_series_ice_partition()
grm.bg_norm_time_series_ice_partition()
grm.N2_mld_time_series_ice_partition()
#grm.mld_time_series_ice_partition()
