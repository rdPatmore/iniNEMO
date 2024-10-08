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

    def raw_var_time_series_ice_partition(self, var_str, fn='grid_T',
                                          quadrant=None):
        '''
        get raw var time series partitioned according to sea ice cover

        parameters
        ---------
        var: variable name in source file
        var_str: name used in output variables and file name
        fn: input file appendage
        quadrant: enact quadran subsetting
        ''' 

        # get var
        ds = xr.open_dataset(self.raw_preamble + fn + '.nc',
                            chunks={'time_counter':1})[var_str]

        im = sim.integrals_and_masks(self.case, self.file_id, ds, var_str)
        # get cfg and icemsk and cut rims
        im.get_domain_vars_and_cut_rims()

        # partition by lat-lon quadrant
        if quadrant:
            im.var  = self.quadrant_partition(im.var, quadrant)
            im.icemsk = self.quadrant_partition(im.icemsk, quadrant)
            im.cfg = self.quadrant_partition(im.cfg, quadrant)
            im.var_str = var_str + '_' + quadrant # update save name

        # partition into zones, mean horizontally and save
        im.horizontal_mean_ice_oce_zones(save=True)

    def save_ml_bg(self):
        ''' save mixed layer buoyancy gradients '''

        # get bg norm
        fn = self.proc_preamble + 'bg_mod2.nc'
        bg = xr.open_dataset(fn, chunks={'time_counter':1}).bg_mod2

        # restrict to mixed layer and save
        im = sim.integrals_and_masks(self.case, self.file_id, bg, 'bg_mod2')
        im.mask_by_ml(save=True, cut=[slice(2,-2), slice(2,-2)])

    def quadrant_partition(self, da, quadrant):
        ''' 
        partition domain into one of four quadrants:
        upper_left, upper_right, lower_left, lower_right
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
        bg = xr.open_dataset(fn, chunks={'time_counter':100}).bg_mod2

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
        ds = xr.open_dataset(fn, chunks={'time_counter':100})
        salt = ds.vosaline
        temp = ds.votemper

        # restrict temperature to mixed layer and save
        im = sim.integrals_and_masks(self.case, self.file_id, temp, 'votemper')
        im.mask_by_ml(save=True)

        # restrict salinity to mixed layer and save
        im = sim.integrals_and_masks(self.case, self.file_id, salt, 'vosaline')
        im.mask_by_ml(save=True)

    def save_ml_mid_raw_var_tpt(self, var='votemper'):
        ''' save mixed layer mid point of t-point variable '''

        # get bg norm
        fn = self.raw_preamble + 'grid_T.nc'
        da = xr.open_dataset(fn, chunks={'time_counter':100})[var]

        # save bg norm at ml mid point
        im = sim.integrals_and_masks(self.case, self.file_id, da, var)
        im.extract_by_depth_at_mld_mid_pt(save=True)

    def save_ml_mid_raw_var_wpt(self, var='bn2'):
        ''' save mixed layer mid point of w-point variable '''

        # get bg norm
        fn = self.raw_preamble + 'grid_W.nc'
        da = xr.open_dataset(fn, chunks={'time_counter':100})[var]

        # unify time
        fn = self.raw_preamble + 'grid_T.nc'
        ds_T = xr.open_dataset(fn, chunks={'time_counter':100})
        da['time_counter'] = ds_T.time_counter

        # rename depth
        da = da.rename({'depthw':'deptht'})

        # save bg norm at ml mid point
        im = sim.integrals_and_masks(self.case, self.file_id, da, var)
        im.extract_by_depth_at_mld_mid_pt(save=True)

    def var_time_series_ice_partition(self, var_str, ml_mid=True, quadrant=None,
                                      depth_integral=False):
        '''
        get 3D variable time series partitioned according to sea ice cover 
        ''' 

        # set integral parameters
        # TODO: these names do not propogate to the save name, which is expected
        #       by the plotting routines
        if ml_mid:
            var_str = var_str + '_ml_mid'
            depth_integral = False
        else:
            var_str = '_ml'

        # get data
        fn = self.proc_preamble + var_str + '.nc'
        da = xr.open_dataarray(fn, chunks={'time_counter':100})

        # initialise partitioning object
        im = sim.integrals_and_masks(self.case, self.file_id, da, var_str)

        # get cfg and icemsk and cut rims
        im.get_domain_vars_and_cut_rims()

        # partition by lat-lon quadrant
        if quadrant:
            im.var  = self.quadrant_partition(im.var, quadrant)
            im.icemsk = self.quadrant_partition(im.icemsk, quadrant)
            im.cfg = self.quadrant_partition(im.cfg, quadrant)
            im.var_str = var_str + '_' + quadrant # update save name

        # partition temperature into zones
        if depth_integral:
            im.domain_mean_ice_oce_zones()
        else:
            im.horizontal_mean_ice_oce_zones(save=True)

    def ice_miz_open_partition_area(self, threshold=0.2, quadrant=None):
        '''
        get area of open ocean, marginal ice zone and sea ice covered
        regions
        '''

        cfg = xr.open_dataset(self.data_path + 'Grid/domain_cfg.nc',
                              chunks=-1).squeeze()

        # load ice concentration
        icemsk = xr.open_dataset(
                     self.data_path + 'RawOutput/' + self.file_id + 'icemod.nc',
                     chunks={'time_counter':1}).siconc

        # partition by lat-lon quadrant
        if quadrant:
            cfg = self.quadrant_partition(cfg, quadrant)
            icemsk = self.quadrant_partition(icemsk, quadrant)

        # get masks
        miz_msk = ((icemsk > threshold) & (icemsk < (1 - threshold))).load()
        ice_msk = (icemsk > (1 - threshold)).load()
        oce_msk = (icemsk < threshold).load()

        # find area 
        area = cfg.e2t * cfg.e1t

        # mask by ice concentration
        area_integ_miz = area.where(miz_msk).sum(['x','y'])
        area_integ_ice = area.where(ice_msk).sum(['x','y'])
        area_integ_oce = area.where(oce_msk).sum(['x','y'])

        # set variable names
        area_integ_miz.name = 'area_miz'
        area_integ_ice.name = 'area_ice'
        area_integ_oce.name = 'area_oce'

        # merge variables
        area_integ = xr.merge([area_integ_miz.load(),
                               area_integ_ice.load(),
                               area_integ_oce.load()])
 
        # save
        if quadrant:
            fn = self.data_path + 'TimeSeries/area_{}_ice_oce_miz.nc'.format(
                                                                      quadrant)
        else:
            fn = self.data_path + 'TimeSeries/area_ice_oce_miz.nc'
        area_integ.to_netcdf(fn)

    def var_ice_partition(self, da, var_str):
        ''' partition variable by ice concentration '''


        # initialise partitioning object
        im = sim.integrals_and_masks(self.case, self.file_id, da, var_str)

        # get cfg and icemsk and cut rims
        im.get_domain_vars_and_cut_rims()

        # save partitioned variable
        im.get_ice_oce_masked_var()

    def ice_parition_N_M_ml_mid(self):
        ''' partition N2 and M4 by ice concentration '''
        
        # get N2
        fn = self.proc_preamble +  + 'bn2_ml_mid.nc'
        N2 = xr.open_dataarray(fn, chunks={'time_counter':100})

        # get M4
        fn = self.proc_preamble +  + 'bg_mod2_ml_mid.nc'
        M4 = xr.open_dataarray(fn, chunks={'time_counter':100})

        # save partitions
        self.var_ice_partition(N2, 'bn2_ml_mid')
        self.var_ice_partition(M4, 'bg_mod2_ml_mid')

    def ice_parition_N_M_at_depth(self, depth=0):
        ''' partition N2 and M4 by ice concentration '''
 
        # get N2
        fn = self.raw_preamble + 'grid_W.nc'
        N2 = xr.open_dataset(fn, chunks={'time_counter':100}).bn2

        # get M2
        fn = self.proc_preamble + 'bg_mod2.nc'
        M4 = xr.open_dataarray(fn, chunks={'time_counter':100})

        # cut depth
        N2 = N2.sel(depthw=depth, method='nearest')
        M4 = M4.sel(deptht=depth, method='nearest')

        # save partitions
        self.var_ice_partition(N2, 'bn2_' + str(depth))
        self.var_ice_partition(M4, 'bg_mod2_' + str(depth))


if __name__ == '__main__':
    import time
    import dask
    dask.config.set(scheduler='single-threaded')

    def ice_partition_N_M():
        case = 'EXP10'
        file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
        grm = glider_relevant_metrics(case, file_id)
        grm.ice_parition_N_M_at_depth(depth=10)
        grm.ice_parition_N_M_at_depth(depth=300)

    def get_mld_quadrant_time_series():
        case = 'EXP10'
        file_id = 'SOCHIC_PATCH_3h_20121209_20130331_'
        var_list = ['mldr10_3']
        for var in var_list:
            print ('var:', var)
            for quad in ['upper_right','upper_left','lower_right','lower_left']:
                print ('quad:', quad)
                grm.raw_var_time_series_ice_partition(var,
                                                      fn='grid_T',
                                                      quadrant=quad)

    # start timer
    start = time.time()

    ice_partition_N_M()

    # end timer
    start = time.time()
    end = time.time()
    print('time elapsed (minutes): ', (end - start)/60)

    #grm.ice_miz_open_partition_area()

    #var_list = ['bn2']
    #for var in var_list:
    #    print ('var:', var)
    #    for quad in ['upper_right','upper_left','lower_right','lower_left']:
    #        print ('quad:', quad)
    #        grm.var_time_series_ice_partition(var_str=var, ml_mid=True,
    #                              quadrant=quad)
    #        #grm.ice_miz_open_partition_area(quadrant=quad)
    #grm.var_time_series_ice_partition(var='votemper', ml_mid=False)
    #grm.var_time_series_ice_partition(var='bn2', ml_mid=True)
    #grm.save_ml_mid_raw_var()
    #grm.save_ml_mid_raw_var_wpt()
    #grm.save_ml_mid_raw_var(var='vosaline')
    #grm.bg_norm_time_series_ice_partition(mld_mid=True)
    #grm.N2_mld_time_series_ice_partition()
    #grm.taum_time_series_ice_partition()
    #grm.fresh_water_flux_time_series_ice_partition()
    #grm.save_ml_T_and_S()
    #grm.mld_time_series_ice_partition()
