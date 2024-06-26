import config 
import xarray as xr
import iniNEMO.Process.Common.spatial_integrals_and_masking as sim

class glider_relevant_metrics(object):

    def __init__(self, case, file_id):
        self.case = case
        self.file_id = file_id
        self.data_path = config.data_path() + case + '/'
        self.raw_preamble = self.data_path + '/RawOutput/' + self.file_id
        self.proc_preamble = self.data_path + 'ProcessedVars/' + self.file_id

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
        fn = self.proc_preamble + 'bg_mod2.nc'
        bg = xr.open_dataset(fn, chunks={'time_counter':1}).bg_mod2

        # restrict to mixed layer and save
        im = sim.integrals_and_masks(self.case, self.file_id, bg, 'bg_mod2')
        im.mask_by_ml(save=True, cut=[slice(2,-2), slice(2,-2)])

    def bg_norm_time_series_ice_partition(self, ml_mid=False):
        ''' get bg norm time series partitioned according to sea ice cover ''' 

        # get bg norm
        if ml_mid:
            append = '_ml_mid.nc'
        else:
            append = '.nc'

        # get bg norm
        fn = self.proc_preamble +'bg_mod2' + append
        bg = xr.open_dataset(fn, chunks={'time_counter':1}).bg_mod2

        # intialise partitioning
        im = sim.integrals_and_masks(self.case, self.file_id, bg, 'bg_mod2')

        # partition into zones
        if mld_mid:
            im.horizontal_mean_ice_oce_zones()
        else:
            im.domain_mean_ice_oce_zones()

    def save_ml_mid_bg_mod2(self):
        ''' save mixed layer mid point of buoyancy gradient norm'''

        # get bg norm
        fn = self.proc_preamble + 'bg_mod2.nc'
        bg = xr.open_dataset(fn, chunks={'time_counter':1}).bg_mod2

        # save bg norm at ml mid point
        im = sim.integrals_and_masks(self.case, self.file_id, bg, 'bg_mod2')
        im.extract_by_depth_at_mld_mid_pt(save=True)

    def N2_mld_time_series_ice_partition(self):
        '''
        get N mixed layer depth time series partitioned according
        to sea ice cover 
        ''' 

        # get N at mld 
        fn = self.proc_preamble + 'N2_mld.nc'
        N2_mld = xr.open_dataset(fn, chunks={'time_counter':1}).bn2

        # partition into zones
        im = sim.integrals_and_masks(self.case, self.file_id, N2_mld, 'N2_mld')
        im.horizontal_mean_ice_oce_zones()

    def wind_speed_time_series_ice_partition(self):
        '''
        get wind speed time series partitioned according
        to sea ice cover 
        ''' 

        # get wind speed
        wind = xr.open_dataset(self.raw_preamble + 'grid_T.nc',
                               chunks={'time_counter':1}).windsp

        # partition into zones
        im = sim.integrals_and_masks(self.case, self.file_id, wind, 'windsp')
        im.horizontal_mean_ice_oce_zones()

    def taum_time_series_ice_partition(self):
        '''
        get wind stess time series partitioned according
        to sea ice cover 
        ''' 

        # get wind speed
        taum = xr.open_dataset(self.raw_preamble + 'grid_T.nc',
                               chunks={'time_counter':1}).taum

        # partition into zones
        im = sim.integrals_and_masks(self.case, self.file_id, taum, 'taum')
        im.horizontal_mean_ice_oce_zones()

    def fresh_water_flux_time_series_ice_partition(self):
        '''
        get surface fresh water flux time series partitioned according
        to sea ice cover 
        ''' 

        # get fresh water flux 
        wfo = xr.open_dataset(self.raw_preamble + 'grid_T.nc',
                               chunks={'time_counter':1}).wfo

        # partition into zones
        im = sim.integrals_and_masks(self.case, self.file_id, wfo, 'wfo')
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
        fn = self.proc_preamble + 'votemper_ml.nc'
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
    grm.save_ml_mid_bg_norm()
    #grm.bg_norm_time_series_ice_partition(mld_mid=True)
    #grm.temperature_time_series_ice_partition()
    #grm.salinity_time_series_ice_partition()
    #grm.N2_mld_time_series_ice_partition()
    #grm.taum_time_series_ice_partition()
    #grm.fresh_water_flux_time_series_ice_partition()
    #grm.save_ml_T_and_S()
    #grm.mld_time_series_ice_partition()
