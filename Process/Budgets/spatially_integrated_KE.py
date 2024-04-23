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
        tke_integ = (self.tke_mld * self.e3t_mean * cfg.glamt * cfg.gphit).sum()

        with ProgressBar():
            tke_integ.to_netcdf(self.preamble + 'TKE_budget_domain_integ.nc')

ke = KE_integrals('TRD00')
#ke.domain_integrated_ml_KE()
ke.vertically_integrated_ml_KE()
