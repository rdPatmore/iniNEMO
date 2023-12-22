import xarray as xr
import config
import matplotlib.pyplot as plt
import numpy as np
import dask
from dask.diagnostics import ProgressBar

class KE_integrals(object):

    def __init__(self, case):
        self.file_id = '/SOCHIC_PATCH_15mi_20121209_20121211_'
        self.preamble = config.data_path() + case +  self.file_id
        self.path = config.data_path() + case + '/'

    def vertically_integrated_ml_KE(KE_type='TKE'):
        ''' vertically integrated KE budget '''

        kwargs = {'chunks':{'time_counter':100} ,'decode_cf':False} 
        self.mld = xr.open_dataset(self.preamble + 'grid_T.nc', **kwargs
                                   ).mldr10_3

        print (self.mld)

    def domain_integrated_ml_KE(KE_type='TKE'):
