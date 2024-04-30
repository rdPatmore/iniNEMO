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
        self.preamble = config.data_path() + case +  self.file_id
        self.path = config.data_path() + case + '/'

    def get_ml_masked_tke(self):
        ''' mask tke according to mixed layer depth '''


        # get time-mean grid_T
        kwargs = {'chunks':-1, 'decode_cf':False} 
        ds_mean = xr.open_dataset(self.preamble + 'grid_T.nc',
                        **kwargs).mean('time_counter')

        # alias mld and e3t
        mld_mean = ds_mean.mldr10_3
        self.e3t_mean = ds_mean.e3t

        # get tke
        kwargs = {'chunks':{'time_counter':100}} 
        tke = xr.open_dataset(self.preamble + 'TKE_budget_full.nc', **kwargs)

        # mask below time-mean mixed layer
        self.tke_mld = tke.where(tke.deptht < mld_mean, drop=False)

        # restore unmasked 2d variables
        for var in list(tke.keys()):
            if var[-2:] == '2d':
                print (var)
                self.tke_mld[var] = tke[var]

    def vertically_integrated_ml_KE(self, KE_type='TKE'):
        ''' vertically integrated KE budget '''

        # get data
        self.get_ml_masked_tke()

        #tke = self.tke_mld.trd_tau.isel(x=20)
        ##tke_e3t = (tke * self.e3t_mean)
        #p = plt.pcolor(self.e3t_mean)
        #plt.colorbar(p)
        #plt.savefig('test.png')
        #print (sdkf)
        # calculate vertical integral
        tke_integ = (self.tke_mld * self.e3t_mean).sum('deptht')

        with ProgressBar():
            tke_integ.to_netcdf(self.preamble + 'TKE_budget_z_integ.nc')

    def domain_integrated_ml_KE(self, KE_type='TKE'):
        ''' domain integrated KE budget '''

        # get data and domain_cfg
        self.get_ml_masked_tke()
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        # calculate domain integral
        tke_integ = (self.tke_mld * self.e3t_mean * cfg.e2t * cfg.e1t).sum()

        with ProgressBar():
            tke_integ.to_netcdf(self.preamble + 'TKE_budget_domain_integ.nc')

    def domain_mean_ml_KE_ice_oce_zones(self, threshold=0.2):
        '''
        split TKE budget into two variables
            - ice area
            - ocean area
        then integrate over domain and weight by area
        '''

        # get data and domain_cfg
        self.get_ml_masked_tke()
        cfg = xr.open_dataset(self.path + 'domain_cfg.nc', chunks=-1)

        # load ice concentration
        icemsk = xr.open_dataset(self.preamble + 'icemod.nc',
                            chunks={'time_counter':10}, decode_cf=True).siconc

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

        # calculate domain weighted mean
        t_volume = self.e3t_mean * area
        tke_integ_miz = (tke_mld_miz * t_volume).sum() / area_miz
        tke_integ_ice = (tke_mld_ice * t_volume).sum() / area_ice
        tke_integ_oce = (tke_mld_oce * t_volume).sum() / area_oce

        # save
        with ProgressBar():
            fn = self.preamble + 'TKE_budget_domain_integ_{}.nc'
            tke_integ_miz.to_netcdf(fn.format('miz'))
            tke_integ_ice.to_netcdf(fn.format('ice'))
            tke_integ_oce.to_netcdf(fn.format('oce'))

ke = KE_integrals('TRD00')
ke.domain_mean_ml_KE_ice_oce_zones()
#ke.vertically_integrated_ml_KE()
