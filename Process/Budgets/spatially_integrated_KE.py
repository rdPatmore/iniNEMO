import xarray as xr
import config
import matplotlib.pyplot as plt
import numpy as np
import dask
from dask.diagnostics import ProgressBar

class KE_integrals(object):
    '''
    Calculate KE budgets integrated in the dimensions of choice

    This is designed for TKE to start with
    '''

    def __init__(self, case):
        self.file_id = '/SOCHIC_PATCH_15mi_20121209_20121211_'
        self.proc_preamble = config.data_path() + case + '/ProcessedVars' \
                           +  self.file_id
        self.raw_preamble = config.data_path() + case + '/RawOutput' \
                           +  self.file_id
        self.path = config.data_path() + case + '/'

    def mask_by_ml_mean(self, var):
        ''' mask variable by mixed layer depth mean '''

        # get time-mean grid_T
        kwargs = {'chunks':-1, 'decode_cf':False} 
        ds_mean = xr.open_dataset(self.preamble + 'grid_T.nc',
                        **kwargs).mean('time_counter')

        # convert cell thickness to depths
        deps = ds_mean.e3t.cumsum('deptht')

        return var.where(deps < ds_mean.mldr10_3, drop=False)

    def get_ml_masked_tke(self):
        ''' mask tke according to mixed layer depth '''


        # get time-mean grid_T
        kwargs = {'chunks':-1, 'decode_cf':False} 
        ds = xr.open_dataset(self.raw_preamble + 'grid_T.nc',
                        **kwargs)
        #ds = ds.isel(time_counter=slice(None,None,2))
        ds_mean = ds.mean('time_counter')

        # alias mld and e3t
        mld_mean = ds_mean.mldr10_3
        self.e3t_mean = ds_mean.e3t

        # get tke
        kwargs = {'chunks':{'time_counter':100}} 
        tke = xr.open_dataset(self.proc_preamble + 'TKE_budget_full.nc',
                              **kwargs)

        # convert cell thickness to depths
        deps = ds_mean.e3t.cumsum('deptht')

        # mask below time-mean mixed layer
        self.tke_mld = tke.where(deps < mld_mean, drop=False)

        # restore unmasked 2d variables
        for var in list(tke.keys()):
            if var[-2:] == '2d':
                print (var)
                self.tke_mld[var] = tke[var]

    def vertically_integrated_ml_KE(self, KE_type='TKE'):
        ''' vertically integrated KE budget '''

        # get data
        self.get_ml_masked_tke()

        # calculate vertical integral
        tke_integ = (self.tke_mld * self.e3t_mean).sum('deptht')

        with ProgressBar():
            tke_integ.to_netcdf(self.proc_preamble +
                               'TKE_budget_z_integ.nc')

    def domain_integrated_ml_KE(self, KE_type='TKE'):
        ''' domain integrated KE budget '''

        # get data and domain_cfg
        self.get_ml_masked_tke()
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        # calculate domain integral
        tke_integ = (self.tke_mld * self.e3t_mean * cfg.e2t * cfg.e1t).sum()

        with ProgressBar():
            tke_integ.to_netcdf(self.proc_preamble + 'TKE_budget_domain_integ.nc')

    def domain_mean_ml_KE_ice_oce_zones(self, threshold=0.2):
        '''
        split TKE budget into three variables
            - ice area
            - ocean area
            - MIZ area
        then integrate over domain and weight by volume
        '''

        # get data and domain_cfg
        self.get_ml_masked_tke()
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        # load ice concentration
        icemsk = xr.open_dataset(self.raw_preamble + 'icemod.nc',
                            chunks={'time_counter':10}, decode_cf=True).siconc

        # mask e3t by mean mixed layer depth
        e3t_mean_ml = self.mask_by_ml_mean(self.e3t_mean)

        # cut edges
        cut=slice(10,-10)
        self.tke_mld = self.tke_mld.isel(x=cut,y=cut)
        cfg = cfg.isel(x=cut,y=cut)
        icemsk = icemsk.isel(x=cut,y=cut)
        e3t_mean_ml = e3t_mean_ml.isel(x=cut,y=cut)
        self.e3t_mean = self.e3t_mean.isel(x=cut,y=cut)

        # mean ice conc over time series
        icemsk_mean = icemsk.mean('time_counter')

        # get masks
        miz_msk = (icemsk_mean > threshold) & (icemsk_mean < (1 - threshold))
        ice_msk = icemsk_mean > (1 - threshold)
        oce_msk = icemsk_mean < threshold

        # mask by ice concentration
        tke_mld_miz = self.tke_mld.where(miz_msk)
        tke_mld_ice = self.tke_mld.where(ice_msk)
        tke_mld_oce = self.tke_mld.where(oce_msk)

        # find volume of each partition
        area = cfg.e2t * cfg.e1t
        t_vol = area * e3t_mean_ml
        t_vol_miz = t_vol.where(miz_msk).sum()
        t_vol_ice = t_vol.where(ice_msk).sum()
        t_vol_oce = t_vol.where(oce_msk).sum()
        print ('miz area', t_vol_miz.values)
        print ('ice area', t_vol_ice.values)
        print ('oce area', t_vol_oce.values)

        # calculate volume weighted mean
        t_volume = self.e3t_mean * area
        tke_integ_miz = (tke_mld_miz * t_volume).sum() / t_vol_miz
        tke_integ_ice = (tke_mld_ice * t_volume).sum() / t_vol_ice
        tke_integ_oce = (tke_mld_oce * t_volume).sum() / t_vol_oce

        # save
        with ProgressBar():
            fn = self.proc_preamble + 'TKE_budget_domain_integ_{}.nc'
            tke_integ_miz.to_netcdf(fn.format('miz'))
            tke_integ_ice.to_netcdf(fn.format('ice'))
            tke_integ_oce.to_netcdf(fn.format('oce'))

    def horizontal_mean_ml_KE_ice_oce_zones(self, threshold=0.2):
        '''
        split TKE budget into three variables
            - ice area
            - ocean area
            - MIZ
        then integrate in the horizontal and weight by area
        '''

        # get data and domain_cfg
        self.get_ml_masked_tke()
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        # load ice concentration
        icemsk = xr.open_dataset(self.raw_preamble + 'icemod.nc',
                            chunks={'time_counter':10}, decode_cf=True).siconc

        # cut edges
        cut=slice(10,-10)
        self.tke_mld = self.tke_mld.isel(x=cut,y=cut)
        cfg = cfg.isel(x=cut,y=cut)
        icemsk = icemsk.isel(x=cut,y=cut)

        # mean ice conc over time series
        icemsk_mean = icemsk.mean('time_counter')

        # get masks
        miz_msk = (icemsk_mean > threshold) & (icemsk_mean < (1 - threshold))
        ice_msk = icemsk_mean > (1 - threshold)
        oce_msk = icemsk_mean < threshold

        # mask by ice concentration
        tke_mld_miz = self.tke_mld.where(miz_msk)
        tke_mld_ice = self.tke_mld.where(ice_msk)
        tke_mld_oce = self.tke_mld.where(oce_msk)

        # find area of each partition
        area = cfg.e2t * cfg.e1t
        area_miz = area.where(miz_msk).sum()
        area_ice = area.where(ice_msk).sum()
        area_oce = area.where(oce_msk).sum()
        print ('miz area', area_miz.values)
        print ('ice area', area_ice.values)
        print ('oce area', area_oce.values)

        # calculate lateral weighted mean
        tke_integ_miz = (tke_mld_miz * area).sum(dim=['x','y']) / area_miz
        tke_integ_ice = (tke_mld_ice * area).sum(dim=['x','y']) / area_ice
        tke_integ_oce = (tke_mld_oce * area).sum(dim=['x','y']) / area_oce

        # save
        with ProgressBar():
            fn = self.proc_preamble + 'TKE_budget_horizontal_integ_{}.nc'
            tke_integ_miz.to_netcdf(fn.format('miz'))
            tke_integ_ice.to_netcdf(fn.format('ice'))
            tke_integ_oce.to_netcdf(fn.format('oce'))

ke = KE_integrals('TRD00')
#ke.domain_mean_ml_KE_ice_oce_zones()
#ke.horizontal_mean_ml_KE_ice_oce_zones()
ke.vertically_integrated_ml_KE()
